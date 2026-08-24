import numpy as np

from ai_vectorizer.core.raster_utils import (
    MAX_RASTER_BLOCK_BYTES,
    compute_resampled_dimensions,
    raster_block_to_uint8,
    read_raster_bands,
)


class _DataType:
    def __init__(self, name):
        self.name = name


class _RasterBlock:
    def __init__(self, values, data_type, *, nodata_value=None, nodata_indices=()):
        self._values = np.asarray(values)
        self._data_type = _DataType(data_type)
        self._nodata_value = nodata_value
        self._nodata_indices = frozenset(nodata_indices)

    def isValid(self):
        return True

    def data(self):
        return self._values.tobytes()

    def dataType(self):
        return self._data_type

    def hasNoDataValue(self):
        return self._nodata_value is not None

    def noDataValue(self):
        return self._nodata_value

    def hasNoData(self):
        return bool(self._nodata_indices)

    def isNoData(self, index):
        return index in self._nodata_indices


class _OversizedPayload:
    def __len__(self):
        return 9

    def __bytes__(self):
        raise AssertionError("oversized payload must be rejected before copying")


class _OversizedRasterBlock:
    def isValid(self):
        return True

    def data(self):
        return _OversizedPayload()


class _PayloadLargerThanMemoryBudget:
    def __len__(self):
        return 4096 * 4096 * 8

    def __bytes__(self):
        raise AssertionError("over-budget payload must be rejected before copying")


def test_signed_int8_raster_values_are_not_reinterpreted_as_unsigned():
    block = _RasterBlock(np.array([-128, 0, 127], dtype=np.int8), "Int8")

    converted = raster_block_to_uint8(block, 3, 1)

    np.testing.assert_array_equal(converted, np.array([[0, 128, 255]], dtype=np.uint8))


def test_signed_int64_raster_values_use_their_declared_numeric_type():
    block = _RasterBlock(np.array([-2, 0, 2], dtype=np.int64), "Int64")

    converted = raster_block_to_uint8(block, 3, 1)

    np.testing.assert_array_equal(converted, np.array([[0, 127, 255]], dtype=np.uint8))


def test_finite_nodata_sentinel_is_excluded_from_normalization():
    block = _RasterBlock(
        np.array([-9999.0, 100.0, 150.0, 200.0], dtype=np.float32),
        "Float32",
        nodata_value=-9999.0,
    )

    converted = raster_block_to_uint8(block, 4, 1)

    np.testing.assert_array_equal(converted, np.array([[127, 0, 127, 255]], dtype=np.uint8))


def test_internal_nodata_bitmap_is_respected_for_byte_rasters():
    block = _RasterBlock(
        np.array([0, 10, 20], dtype=np.uint8),
        "Byte",
        nodata_indices=(0,),
    )

    converted = raster_block_to_uint8(block, 3, 1)

    np.testing.assert_array_equal(converted, np.array([[15, 10, 20]], dtype=np.uint8))


def test_invalid_dimensions_are_rejected_without_reading_payload():
    class UnreadableBlock:
        def isValid(self):
            return True

        def data(self):
            raise AssertionError("invalid dimensions must be rejected first")

    assert raster_block_to_uint8(UnreadableBlock(), np.nan, 1) is None
    assert raster_block_to_uint8(UnreadableBlock(), -1, 1) is None
    assert raster_block_to_uint8(UnreadableBlock(), 1.5, 1) is None


def test_oversized_block_payload_is_rejected_before_copying():
    assert raster_block_to_uint8(_OversizedRasterBlock(), 1, 1) is None


def test_byte_budget_is_enforced_before_copying_large_float_payload():
    class Block:
        def isValid(self):
            return True

        def data(self):
            return _PayloadLargerThanMemoryBudget()

    assert 4096 * 4096 * 8 > MAX_RASTER_BLOCK_BYTES
    assert raster_block_to_uint8(Block(), 4096, 4096, _DataType("Float64")) is None


def test_oversized_band_request_is_rejected_before_provider_allocation():
    class Provider:
        def bandCount(self):
            return 1

        def block(self, *args):
            raise AssertionError("oversized request must not reach the provider")

    assert read_raster_bands(
        Provider(),
        extent=None,
        width=5_001,
        height=5_000,
    ) == []


def test_float64_byte_budget_is_checked_before_provider_allocation():
    class Provider:
        def bandCount(self):
            return 1

        def dataType(self, _band_number):
            return _DataType("Float64")

        def block(self, *args):
            raise AssertionError("over-budget request must not reach the provider")

    assert read_raster_bands(
        Provider(),
        extent=None,
        width=4096,
        height=4096,
    ) == []


def test_extreme_float64_range_normalizes_without_overflow():
    block = _RasterBlock(
        np.array([-1e308, 0.0, 1e308], dtype=np.float64),
        "Float64",
    )

    converted = raster_block_to_uint8(block, 3, 1)

    np.testing.assert_array_equal(converted, np.array([[0, 127, 255]], dtype=np.uint8))


def test_resampling_uses_one_scale_factor_to_preserve_aspect_ratio():
    assert compute_resampled_dimensions(
        10_000,
        1_000,
        10_000,
        1_000,
        10_000,
        1_000,
        1_000,
    ) == (1_000, 100)
    assert compute_resampled_dimensions(
        2_000,
        500,
        2_000,
        500,
        2_000,
        500,
        1_000,
    ) == (1_000, 250)


def test_resampling_falls_back_safely_for_malformed_metadata():
    assert compute_resampled_dimensions(
        np.nan,
        100,
        100,
        100,
        100,
        100,
        800,
        min_dimension=2,
    ) == (2, 2)
    assert compute_resampled_dimensions(
        100,
        100,
        np.inf,
        100,
        100,
        100,
        800,
        min_dimension=2,
    ) == (2, 2)
