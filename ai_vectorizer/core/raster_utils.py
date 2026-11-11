# -*- coding: utf-8 -*-
"""
Raster helpers shared by preview and tracing cache code.
"""

import math

import numpy as np


MAX_RASTER_BLOCK_PIXELS = 25_000_000
MAX_RASTER_BLOCK_BYTES = 64 * 1024 * 1024
MAX_RASTER_BYTES_PER_VALUE = 8


BYTE_DEPTH_TO_DTYPE = {
    1: np.uint8,
    2: np.uint16,
    4: np.float32,
    8: np.float64,
}

QGIS_DATA_TYPE_TO_DTYPE = {
    "Byte": np.uint8,
    "Int8": np.int8,
    "UInt16": np.uint16,
    "Int16": np.int16,
    "UInt32": np.uint32,
    "Int32": np.int32,
    "UInt64": np.uint64,
    "Int64": np.int64,
    "Float32": np.float32,
    "Float64": np.float64,
}

PACKED_COLOR_DATA_TYPES = {"ARGB32", "ARGB32_Premultiplied"}

UNSUPPORTED_QGIS_DATA_TYPES = {
    "UnknownDataType",
    "ARGB32",
    "ARGB32_Premultiplied",
    "CInt16",
    "CInt32",
    "CFloat32",
    "CFloat64",
}


def _data_type_name(data_type):
    if data_type is None:
        return None

    name = getattr(data_type, "name", None)
    if isinstance(name, str) and name:
        return name

    text = str(data_type)
    if "." in text:
        return text.rsplit(".", 1)[-1]
    return text


def _resolve_numpy_dtype(data_type, bytes_per_value):
    data_type_name = _data_type_name(data_type)
    if data_type_name in UNSUPPORTED_QGIS_DATA_TYPES:
        return None

    dtype = QGIS_DATA_TYPE_TO_DTYPE.get(data_type_name)
    if dtype is not None:
        if np.dtype(dtype).itemsize != bytes_per_value:
            return None
        return dtype

    return BYTE_DEPTH_TO_DTYPE.get(bytes_per_value)


def _positive_integral_value(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
        return None
    return int(numeric)


def _raster_nodata_mask(block, array, source_nodata_value=None):
    """Combine provider sentinels and the block bitmap without importing QGIS."""
    mask = np.zeros(array.shape, dtype=bool)
    values = [source_nodata_value]
    block_nodata = None
    block_nodata_present = False
    if hasattr(block, "hasNoDataValue") and hasattr(block, "noDataValue"):
        try:
            if block.hasNoDataValue():
                block_nodata = block.noDataValue()
                values.append(block_nodata)
                block_nodata_present = True
        except Exception:  # nosec B110
            pass
    for value in values:
        if value is None:
            continue
        try:
            mask |= np.isnan(array) if np.isnan(value) else array == value
        except (TypeError, ValueError, OverflowError):
            continue
    if (block_nodata_present and type(block).__name__ == "QgsRasterBlock"
            and type(block).__module__ == "qgis._core"):
        # QgsRasterBlock::isNoData() uses the sentinel instead of its bitmap
        # when hasNoDataValue() is true (qgsrasterblock.h). Vectorize the same
        # qgsDoubleNear epsilon without crossing the SIP boundary per pixel.
        # Other duck-typed blocks may combine both mechanisms and still take
        # the bitmap path below; concrete QGIS bitmap-only blocks do too.
        try:
            with np.errstate(invalid="ignore", over="ignore"):
                difference = array.astype(np.float64) - float(block_nodata)
                mask |= np.isnan(array) | (np.abs(difference) <= 4 * np.finfo(np.float64).eps)
            return mask
        except (TypeError, ValueError, OverflowError):
            pass
    try:
        has_nodata = bool(block.hasNoData()) if hasattr(block, "hasNoData") else False
    except Exception:
        has_nodata = False
    if not has_nodata or not hasattr(block, "isNoData"):
        return mask
    pixel_count = int(array.size)
    try:
        flat = np.fromiter((bool(block.isNoData(i)) for i in range(pixel_count)),
                           dtype=bool, count=pixel_count)
    except TypeError:
        height, width = array.shape
        try:
            flat = np.fromiter((bool(block.isNoData(y, x))
                                for y in range(height) for x in range(width)),
                               dtype=bool, count=pixel_count)
        except Exception:
            return None
    except Exception:
        return None
    return mask | flat.reshape(array.shape)


def _raster_block_array(block, width, height, data_type=None, *, packed=False):
    """Decode only a bounded, exactly sized block; no raster interpretation."""
    width = _positive_integral_value(width)
    height = _positive_integral_value(height)
    if width is None or height is None or width * height > MAX_RASTER_BLOCK_PIXELS:
        return None
    count = width * height
    try:
        if block is None or not block.isValid():
            return None
        payload = block.data()
        size = len(payload)
    except Exception:
        return None
    if (size <= 0 or size > MAX_RASTER_BLOCK_BYTES
            or size > count * MAX_RASTER_BYTES_PER_VALUE or size % count):
        return None
    if data_type is None and hasattr(block, "dataType"):
        try:
            data_type = block.dataType()
        except Exception:
            data_type = None
    name = _data_type_name(data_type)
    if packed:
        if name not in PACKED_COLOR_DATA_TYPES or size != count * 4:
            return None
        dtype = np.uint32  # QRgb is the native-endian integer 0xAARRGGBB.
    else:
        dtype = _resolve_numpy_dtype(data_type, size // count)
        if dtype is None:
            return None
    try:
        raw = bytes(payload)
    except Exception:
        return None
    if len(raw) != size:
        return None
    return np.frombuffer(raw, dtype=dtype, count=count).reshape(height, width)


def _display_array(array, nodata_mask):
    """Normalize valid values only; missing pixels always represent white paper."""
    valid = np.isfinite(array) & ~nodata_mask
    if array.dtype == np.uint8:
        result = array.copy()
        result[~valid] = 255
        return result
    if not valid.any():
        return np.full(array.shape, 255, dtype=np.uint8)
    values = array.astype(np.float64, copy=False)
    low, high = float(values[valid].min()), float(values[valid].max())
    if high <= low:
        # Preserve the legacy zero-contrast mapping for the observed values.
        # A black float line against NaN/NoData still needs white missing
        # pixels; whitening the entire constant-valid block would erase it.
        return np.where(valid, 0, 255).astype(np.uint8)
    values = np.where(valid, values, low)
    midpoint, half_range = low / 2.0 + high / 2.0, high / 2.0 - low / 2.0
    if math.isfinite(half_range) and half_range > 0:
        normalized = ((values - midpoint) / half_range + 1.0) * 127.5
    else:
        span = high - low
        if not math.isfinite(span) or span <= 0:
            return np.full(array.shape, 255, dtype=np.uint8)
        normalized = ((values - low) / span) * 255.0
    result = np.clip(normalized, 0, 255).astype(np.uint8)
    result[~valid] = 255
    return result


def _native_array(array, nodata_mask):
    result = array.copy()
    if np.issubdtype(result.dtype, np.integer):
        result[nodata_mask] = np.iinfo(result.dtype).max
    else:
        # Unknown float ranges cannot provide fixed white-paper DNs. Keep
        # missing values explicit; the reader never attests these as stable Ink.
        result[~np.isfinite(result) | nodata_mask] = np.nan
    return np.ascontiguousarray(result)


def raster_block_to_uint8(block, width, height, data_type=None, *, source_nodata_value=None):
    """Return a bounded display band, with NoData/NaN composited as white paper."""
    array = _raster_block_array(block, width, height, data_type)
    if array is None:
        return None
    mask = _raster_nodata_mask(block, array, source_nodata_value)
    return None if mask is None else _display_array(array, mask)


def raster_block_to_native_array(block, width, height, data_type=None, *, source_nodata_value=None):
    """Preserve native DNs; use fixed bright integer NoData and explicit float NaN.

    Integer DNs retain pan-stable semantics. Floating arrays remain diagnostic
    data only; they cannot attest a stable Ink or native Byte learning boundary.
    Packed rendered colors use the separate display-only RGB reader.
    """
    array = _raster_block_array(block, width, height, data_type)
    if array is None:
        return None
    mask = _raster_nodata_mask(block, array, source_nodata_value)
    return None if mask is None else _native_array(array, mask)


def raster_block_to_rgb_uint8(block, width, height, data_type=None):
    """Composite QGIS ARGB/ARGB-premultiplied rendered pixels over white paper.

    This is display RGB, never proof of three native Byte source bands.
    Opaque channels are copied exactly; alpha-zero hidden colors cannot create
    dark ink. Premultiplied values already contain alpha and are not multiplied
    a second time.
    """
    if data_type is None:
        try:
            data_type = block.dataType()
        except Exception:
            return None
    packed = _raster_block_array(block, width, height, data_type, packed=True)
    if packed is None:
        return None
    mask = _raster_nodata_mask(block, packed)
    if mask is None:
        return None
    alpha = (packed >> 24).astype(np.uint16)
    rgb = np.empty((*packed.shape, 3), dtype=np.uint8)
    premultiplied = _data_type_name(data_type) == "ARGB32_Premultiplied"
    for channel, shift in enumerate((16, 8, 0)):
        value = ((packed >> shift) & 255).astype(np.uint16)
        if premultiplied:
            composite = np.minimum(value + (255 - alpha), 255)
        else:
            composite = (value * alpha + 255 * (255 - alpha) + 127) // 255
        rgb[..., channel] = composite.astype(np.uint8)
    rgb[mask] = 255
    return rgb


def stable_integer_band_to_uint8(values):
    """Map integer source DNs to uint8 using only the dtype's fixed range."""
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("stable Ink display conversion requires an integer array")
    limits = np.iinfo(array.dtype)
    if limits.max <= limits.min:
        return np.full(array.shape, 255, dtype=np.uint8)
    normalized = ((array.astype(np.float64, copy=False) - float(limits.min))
                  * (255.0 / (float(limits.max) - float(limits.min))))
    return np.ascontiguousarray(np.clip(normalized, 0.0, 255.0).astype(np.uint8))


def _provider_data_type(provider, band):
    try:
        return provider.dataType(band)
    except Exception:
        return None


def _provider_source_nodata(provider, band):
    try:
        # Respect an explicit user decision to disable the source sentinel.
        if hasattr(provider, "useSourceNoDataValue") and not provider.useSourceNoDataValue(band):
            return None
        if provider.sourceHasNoDataValue(band):
            return provider.sourceNoDataValue(band)
    except Exception:  # nosec B110
        pass
    return None


def _provider_is_alpha(provider, band):
    try:
        value = provider.colorInterpretation(band)
        name = _data_type_name(value)
        # QGIS QgsRaster.AlphaBand is 6, including older integer enum bindings.
        return name == "AlphaBand" or (name == "6" and int(value) == 6)
    except Exception:
        return False


def _provider_block(provider, band, extent, width, height, data_type):
    name = _data_type_name(data_type)
    if name in UNSUPPORTED_QGIS_DATA_TYPES and name not in PACKED_COLOR_DATA_TYPES:
        return None
    dtype = QGIS_DATA_TYPE_TO_DTYPE.get(name)
    size = 4 if name in PACKED_COLOR_DATA_TYPES else np.dtype(dtype).itemsize if dtype is not None else 8
    if width * height * size > MAX_RASTER_BLOCK_BYTES:
        return None
    try:
        return provider.block(band, extent, width, height)
    except Exception:
        return None


def read_raster_bands_with_native(provider, extent, width, height, max_bands=3):
    """Read bounded display/native bands once; honor explicit transparency.

    Unmodified integer sources can attest stable native Ink DNs. Packed ARGB
    and explicit alpha-band composites return display RGB only and stable=False;
    they must not silently enter native Byte model or source-grid evidence paths.
    An undeclared fourth band is never assumed to be alpha.
    """
    width, height = _positive_integral_value(width), _positive_integral_value(height)
    requested = _positive_integral_value(max_bands)
    if width is None or height is None or requested is None or width * height > MAX_RASTER_BLOCK_PIXELS:
        return [], [], False
    try:
        available = _positive_integral_value(provider.bandCount())
    except Exception:
        return [], [], False
    if available is None:
        return [], [], False
    all_bands = range(1, min(available, 256) + 1)
    alpha_bands = [band for band in all_bands if _provider_is_alpha(provider, band)]
    if len(alpha_bands) > 1:
        return [], [], False
    selected = [band for band in all_bands if band not in alpha_bands][:min(requested, 256)]
    display, native = [], []
    missing = np.zeros((height, width), dtype=bool)
    for band in selected:
        data_type = _provider_data_type(provider, band)
        block = _provider_block(provider, band, extent, width, height, data_type)
        if data_type is None and block is not None:
            try:
                data_type = block.dataType()
            except Exception:  # nosec B110
                pass
        if _data_type_name(data_type) in PACKED_COLOR_DATA_TYPES:
            rgb = raster_block_to_rgb_uint8(block, width, height, data_type)
            if rgb is None or len(selected) != 1 or alpha_bands:
                return [], [], False
            return [rgb[..., i].copy() for i in range(min(3, requested))], [], False
        raw = _raster_block_array(block, width, height, data_type)
        if raw is None:
            continue
        nodata = _raster_nodata_mask(block, raw, _provider_source_nodata(provider, band))
        if nodata is None:
            continue
        missing |= nodata | ~np.isfinite(raw)
        display.append(_display_array(raw, nodata))
        native.append(_native_array(raw, nodata))
    if not display:
        return [], [], False
    # A missing RGB pixel must not become a false colored stroke through the
    # other valid channels. Opaque pixels without NoData remain byte-exact.
    for shown, source in zip(display, native):
        shown[missing] = 255
        source[missing] = np.iinfo(source.dtype).max if np.issubdtype(source.dtype, np.integer) else np.nan
    if alpha_bands:
        alpha_band = alpha_bands[0]
        dtype = _provider_data_type(provider, alpha_band)
        block = _provider_block(provider, alpha_band, extent, width, height, dtype)
        raw = _raster_block_array(block, width, height, dtype)
        if raw is None or not np.issubdtype(raw.dtype, np.unsignedinteger):
            return [], [], False
        invalid = _raster_nodata_mask(block, raw, _provider_source_nodata(provider, alpha_band))
        if invalid is None:
            return [], [], False
        alpha = np.rint(raw.astype(np.float64) * (255.0 / float(np.iinfo(raw.dtype).max))).astype(np.uint16)
        alpha[invalid] = 0
        display = [((shown.astype(np.uint16) * alpha + 255 * (255 - alpha) + 127) // 255).astype(np.uint8)
                   for shown in display]
        return display, [], False
    stable = all(np.issubdtype(band.dtype, np.integer) for band in native)
    return display, native, stable


def read_raster_bands(provider, extent, width, height, max_bands=3):
    """Read up to max_bands display channels with white-paper transparency."""
    return read_raster_bands_with_native(provider, extent, width, height, max_bands)[0]


def raster_has_native_grid(raster_layer):
    """Duck-typed source-grid check shared by tracing and edge preview reads."""
    try:
        provider_type = str(raster_layer.providerType()).lower()
    except (AttributeError, RuntimeError, TypeError):
        provider_type = ""
    if provider_type == "wms":
        return False
    try:
        dimensions = (float(raster_layer.width()), float(raster_layer.height()))
        return all(math.isfinite(value) and value > 0 and value.is_integer() for value in dimensions)
    except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError):
        return False


def compute_resampled_dimensions(
    source_extent_width,
    source_extent_height,
    source_width,
    source_height,
    read_extent_width,
    read_extent_height,
    max_dimension,
    min_dimension=1,
):
    """Return a safe raster read size derived from source and requested extents."""
    max_dimension = _positive_integral_value(max_dimension) or 1
    min_dimension = min(
        max_dimension,
        _positive_integral_value(min_dimension) or 1,
    )

    try:
        source_extent_width = float(source_extent_width)
        source_extent_height = float(source_extent_height)
        source_width = float(source_width)
        source_height = float(source_height)
        read_extent_width = float(read_extent_width)
        read_extent_height = float(read_extent_height)
    except (TypeError, ValueError, OverflowError):
        return min_dimension, min_dimension
    dimensions = (
        source_extent_width,
        source_extent_height,
        source_width,
        source_height,
        read_extent_width,
        read_extent_height,
    )
    if not all(np.isfinite(value) and value > 0 for value in dimensions):
        return min_dimension, min_dimension

    res_x = source_extent_width / source_width
    res_y = source_extent_height / source_height
    if not np.isfinite(res_x) or not np.isfinite(res_y) or res_x <= 0 or res_y <= 0:
        return min_dimension, min_dimension

    out_w = read_extent_width / res_x
    out_h = read_extent_height / res_y
    if not np.isfinite(out_w) or not np.isfinite(out_h) or out_w <= 0 or out_h <= 0:
        return min_dimension, min_dimension

    scale = min(1.0, max_dimension / max(out_w, out_h))
    scaled_w = max(min_dimension, int(round(out_w * scale)))
    scaled_h = max(min_dimension, int(round(out_h * scale)))
    return min(max_dimension, scaled_w), min(max_dimension, scaled_h)
