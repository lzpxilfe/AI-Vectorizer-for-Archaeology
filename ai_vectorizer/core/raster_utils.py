# -*- coding: utf-8 -*-
"""
Raster helpers shared by preview and tracing cache code.
"""

import numpy as np

from .dependencies import get_cv2


BYTE_DEPTH_TO_DTYPE = {
    1: np.uint8,
    2: np.uint16,
    4: np.float32,
    8: np.float64,
}

QGIS_DATA_TYPE_TO_DTYPE = {
    "Byte": np.uint8,
    "UInt16": np.uint16,
    "Int16": np.int16,
    "UInt32": np.uint32,
    "Int32": np.int32,
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


def raster_block_to_uint8(block, width, height, data_type=None):
    """Convert a QgsRasterBlock payload to a normalized uint8 array."""
    if block is None or not block.isValid():
        return None

    raw = bytes(block.data())
    pixel_count = int(width) * int(height)
    if pixel_count <= 0 or not raw or len(raw) % pixel_count != 0:
        return None

    bytes_per_value = len(raw) // pixel_count
    if data_type is None and hasattr(block, "dataType"):
        try:
            data_type = block.dataType()
        except Exception:
            data_type = None

    dtype = _resolve_numpy_dtype(data_type, bytes_per_value)
    if dtype is None:
        return None

    array = np.frombuffer(raw, dtype=dtype, count=pixel_count)
    if array.size != pixel_count:
        return None

    array = array.reshape((height, width))
    if array.dtype == np.uint8:
        return array.copy()

    array = array.astype(np.float32, copy=False)

    finite_mask = np.isfinite(array)
    if not finite_mask.any():
        return np.zeros((height, width), dtype=np.uint8)

    if not finite_mask.all():
        fill_value = float(np.nanmin(array[finite_mask]))
        array = np.where(finite_mask, array, fill_value)

    min_value = float(array.min())
    max_value = float(array.max())
    if max_value <= min_value:
        return np.zeros((height, width), dtype=np.uint8)

    cv2 = get_cv2()
    if cv2 is not None:
        normalized = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX)
    else:
        scale = 255.0 / (max_value - min_value)
        normalized = np.clip((array - min_value) * scale, 0, 255)
    return normalized.astype(np.uint8)


def read_raster_bands(provider, extent, width, height, max_bands=3):
    """Read up to max_bands raster bands as uint8 arrays."""
    bands = []
    band_limit = min(int(max_bands), int(provider.bandCount()))
    for band_number in range(1, band_limit + 1):
        block = provider.block(band_number, extent, width, height)
        data_type = None
        if hasattr(provider, "dataType"):
            try:
                data_type = provider.dataType(band_number)
            except Exception:
                data_type = None
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
    source_width = max(1, int(source_width))
    source_height = max(1, int(source_height))
    max_dimension = max(1, int(max_dimension))
    min_dimension = max(1, int(min_dimension))

    res_x = float(source_extent_width) / source_width
    res_y = float(source_extent_height) / source_height
    if res_x <= 0 or res_y <= 0:
        return min_dimension, min_dimension

    out_w = max(min_dimension, int(round(float(read_extent_width) / res_x)))
    out_h = max(min_dimension, int(round(float(read_extent_height) / res_y)))
    return min(max_dimension, out_w), min(max_dimension, out_h)
