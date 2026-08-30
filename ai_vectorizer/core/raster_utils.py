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


def _raster_nodata_mask(block, array):
    """Return the block's NoData pixels without importing QGIS."""
    has_nodata_value = False
    if hasattr(block, "hasNoDataValue"):
        try:
            has_nodata_value = bool(block.hasNoDataValue())
        except Exception:
            has_nodata_value = False

    if has_nodata_value and hasattr(block, "noDataValue"):
        try:
            nodata_value = block.noDataValue()
            if np.isnan(nodata_value):
                return np.isnan(array)
            return array == nodata_value
        except (TypeError, ValueError, OverflowError):
            pass
        # Unsupported provider NoData APIs fall back safely.
        except Exception:  # nosec B110
            pass

    has_nodata = False
    if hasattr(block, "hasNoData"):
        try:
            has_nodata = bool(block.hasNoData())
        except Exception:
            has_nodata = False
    if not has_nodata or not hasattr(block, "isNoData"):
        return np.zeros(array.shape, dtype=bool)

    pixel_count = int(array.size)
    try:
        flat_mask = np.fromiter(
            (bool(block.isNoData(index)) for index in range(pixel_count)),
            dtype=bool,
            count=pixel_count,
        )
    except TypeError:
        height, width = array.shape
        try:
            flat_mask = np.fromiter(
                (
                    bool(block.isNoData(row, column))
                    for row in range(height)
                    for column in range(width)
                ),
                dtype=bool,
                count=pixel_count,
            )
        except Exception:
            return None
    except Exception:
        return None
    return flat_mask.reshape(array.shape)


def raster_block_to_uint8(block, width, height, data_type=None):
    """Convert a QgsRasterBlock payload to a normalized uint8 array."""
    width = _positive_integral_value(width)
    height = _positive_integral_value(height)
    if width is None or height is None:
        return None
    pixel_count = width * height
    if pixel_count > MAX_RASTER_BLOCK_PIXELS:
        return None

    try:
        valid = block is not None and bool(block.isValid())
    except Exception:
        return None
    if not valid:
        return None

    try:
        payload = block.data()
        payload_size = len(payload)
    except Exception:
        return None
    if (
        payload_size <= 0
        or payload_size > MAX_RASTER_BLOCK_BYTES
        or payload_size > pixel_count * MAX_RASTER_BYTES_PER_VALUE
        or payload_size % pixel_count != 0
    ):
        return None

    bytes_per_value = payload_size // pixel_count
    if data_type is None and hasattr(block, "dataType"):
        try:
            data_type = block.dataType()
        except Exception:
            data_type = None

    dtype = _resolve_numpy_dtype(data_type, bytes_per_value)
    if dtype is None:
        return None

    try:
        raw = bytes(payload)
    except Exception:
        return None
    if len(raw) != payload_size:
        return None

    array = np.frombuffer(raw, dtype=dtype, count=pixel_count)
    if array.size != pixel_count:
        return None

    array = array.reshape((height, width))
    nodata_mask = _raster_nodata_mask(block, array)
    if nodata_mask is None:
        return None
    if array.dtype == np.uint8:
        result = array.copy()
        if nodata_mask.any():
            valid_values = result[~nodata_mask]
            if valid_values.size == 0:
                return np.zeros((height, width), dtype=np.uint8)
            result[nodata_mask] = np.uint8(np.median(valid_values))
        return result

    # Float64 avoids discarding otherwise valid high-dynamic-range provider
    # values before normalization.
    array = array.astype(np.float64, copy=False)

    valid_mask = np.isfinite(array) & ~nodata_mask
    if not valid_mask.any():
        return np.zeros((height, width), dtype=np.uint8)

    if not valid_mask.all():
        fill_value = float(np.median(array[valid_mask]))
        array = np.where(valid_mask, array, fill_value)

    min_value = float(array[valid_mask].min())
    max_value = float(array[valid_mask].max())
    if max_value <= min_value:
        return np.zeros((height, width), dtype=np.uint8)

    midpoint = min_value / 2.0 + max_value / 2.0
    half_range = max_value / 2.0 - min_value / 2.0
    if math.isfinite(half_range) and half_range > 0:
        normalized = ((array - midpoint) / half_range + 1.0) * 127.5
    else:
        value_range = max_value - min_value
        if not math.isfinite(value_range) or value_range <= 0:
            return np.zeros((height, width), dtype=np.uint8)
        normalized = ((array - min_value) / value_range) * 255.0
    normalized = np.clip(normalized, 0, 255)
    return normalized.astype(np.uint8)


def raster_block_to_native_array(block, width, height, data_type=None):
    """Return one bounded raster block without view-dependent stretching.

    This is an opt-in Ink v2 boundary.  The long-standing
    :func:`raster_block_to_uint8` contract deliberately keeps its per-block
    display stretch for legacy callers.  Integer source values retain their
    dtype here so source-grid morphology sees identical DNs after pan/zoom.
    Floating rasters are returned for diagnostics, but callers must not claim
    pan-stable v2 evidence without an external, fixed source range.
    """

    width = _positive_integral_value(width)
    height = _positive_integral_value(height)
    if width is None or height is None:
        return None
    pixel_count = width * height
    if pixel_count > MAX_RASTER_BLOCK_PIXELS:
        return None

    try:
        valid = block is not None and bool(block.isValid())
    except Exception:
        return None
    if not valid:
        return None

    try:
        payload = block.data()
        payload_size = len(payload)
    except Exception:
        return None
    if (
        payload_size <= 0
        or payload_size > MAX_RASTER_BLOCK_BYTES
        or payload_size > pixel_count * MAX_RASTER_BYTES_PER_VALUE
        or payload_size % pixel_count != 0
    ):
        return None

    bytes_per_value = payload_size // pixel_count
    if data_type is None and hasattr(block, "dataType"):
        try:
            data_type = block.dataType()
        except Exception:
            data_type = None
    dtype = _resolve_numpy_dtype(data_type, bytes_per_value)
    if dtype is None:
        return None

    try:
        raw = bytes(payload)
    except Exception:
        return None
    if len(raw) != payload_size:
        return None
    array = np.frombuffer(raw, dtype=dtype, count=pixel_count)
    if array.size != pixel_count:
        return None
    array = array.reshape((height, width))
    nodata_mask = _raster_nodata_mask(block, array)
    if nodata_mask is None:
        return None

    if np.issubdtype(array.dtype, np.integer):
        result = array.copy()
        if nodata_mask.any():
            # A block median would reintroduce pan dependence through NoData.
            # The dtype maximum is a fixed, conservative paper-bright value:
            # it cannot invent a dark Ink stroke at a missing source pixel.
            result[nodata_mask] = np.iinfo(result.dtype).max
        return np.ascontiguousarray(result)

    result = array.astype(array.dtype, copy=True)
    valid_mask = np.isfinite(result) & ~nodata_mask
    if not valid_mask.any():
        return np.zeros((height, width), dtype=result.dtype)
    if not valid_mask.all():
        result[~valid_mask] = np.asarray(
            np.median(result[valid_mask]),
            dtype=result.dtype,
        )
    return np.ascontiguousarray(result)


def stable_integer_band_to_uint8(values):
    """Map integer source DNs to uint8 using only the dtype's fixed range."""

    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("stable Ink display conversion requires an integer array")
    limits = np.iinfo(array.dtype)
    if limits.max <= limits.min:
        return np.zeros(array.shape, dtype=np.uint8)
    normalized = (
        (array.astype(np.float64, copy=False) - float(limits.min))
        * (255.0 / (float(limits.max) - float(limits.min)))
    )
    return np.ascontiguousarray(
        np.clip(normalized, 0.0, 255.0).astype(np.uint8)
    )


def read_raster_bands_with_native(provider, extent, width, height, max_bands=3):
    """Read legacy-display and native arrays from each provider block once.

    Returns ``(display_bands, native_bands, native_integer_stable)``.  Display
    bands preserve :func:`read_raster_bands` semantics for the frozen Ink v1
    fallback.  ``native_integer_stable`` is true only when every selected band
    retained an integer source dtype suitable for pan-stable Ink v2 evidence.
    """

    display_bands = []
    native_bands = []
    width = _positive_integral_value(width)
    height = _positive_integral_value(height)
    requested_bands = _positive_integral_value(max_bands)
    if width is None or height is None or requested_bands is None:
        return display_bands, native_bands, False
    if width * height > MAX_RASTER_BLOCK_PIXELS:
        return display_bands, native_bands, False
    try:
        available_bands = _positive_integral_value(provider.bandCount())
    except Exception:
        return display_bands, native_bands, False
    if available_bands is None:
        return display_bands, native_bands, False

    stable = True
    band_limit = min(requested_bands, available_bands, 256)
    for band_number in range(1, band_limit + 1):
        data_type = None
        if hasattr(provider, "dataType"):
            try:
                data_type = provider.dataType(band_number)
            except Exception:
                data_type = None
        data_type_name = _data_type_name(data_type)
        if data_type_name in UNSUPPORTED_QGIS_DATA_TYPES:
            continue
        known_dtype = QGIS_DATA_TYPE_TO_DTYPE.get(data_type_name)
        bytes_per_value = (
            np.dtype(known_dtype).itemsize
            if known_dtype is not None
            else MAX_RASTER_BYTES_PER_VALUE
        )
        if width * height * bytes_per_value > MAX_RASTER_BLOCK_BYTES:
            continue
        # A provider-specific band failure is isolated; remaining bands can
        # still supply a stable display/native pair.
        try:
            block = provider.block(band_number, extent, width, height)
        except Exception:  # nosec B112
            continue
        display = raster_block_to_uint8(
            block,
            width,
            height,
            data_type=data_type,
        )
        if display is None:
            continue
        native = raster_block_to_native_array(
            block,
            width,
            height,
            data_type=data_type,
        )
        display_bands.append(display)
        if native is None:
            stable = False
            continue
        native_bands.append(native)
        stable = stable and np.issubdtype(native.dtype, np.integer)

    if len(native_bands) != len(display_bands):
        stable = False
    return display_bands, native_bands, bool(display_bands) and stable


def read_raster_bands(provider, extent, width, height, max_bands=3):
    """Read up to max_bands raster bands as uint8 arrays."""
    bands = []
    width = _positive_integral_value(width)
    height = _positive_integral_value(height)
    requested_bands = _positive_integral_value(max_bands)
    if width is None or height is None or requested_bands is None:
        return bands
    if width * height > MAX_RASTER_BLOCK_PIXELS:
        return bands
    try:
        available_bands = _positive_integral_value(provider.bandCount())
    except Exception:
        return bands
    if available_bands is None:
        return bands
    band_limit = min(requested_bands, available_bands, 256)
    for band_number in range(1, band_limit + 1):
        data_type = None
        if hasattr(provider, "dataType"):
            try:
                data_type = provider.dataType(band_number)
            except Exception:
                data_type = None
        data_type_name = _data_type_name(data_type)
        if data_type_name in UNSUPPORTED_QGIS_DATA_TYPES:
            continue
        known_dtype = QGIS_DATA_TYPE_TO_DTYPE.get(data_type_name)
        bytes_per_value = (
            np.dtype(known_dtype).itemsize
            if known_dtype is not None
            else MAX_RASTER_BYTES_PER_VALUE
        )
        if width * height * bytes_per_value > MAX_RASTER_BLOCK_BYTES:
            continue
        try:
            block = provider.block(band_number, extent, width, height)
        # A failed optional band is intentionally skipped.
        except Exception:  # nosec B112
            continue
        band = raster_block_to_uint8(block, width, height, data_type=data_type)
        if band is not None:
            bands.append(band)
    return bands


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
