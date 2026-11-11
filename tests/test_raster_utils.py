import numpy as np
import pytest

from ai_vectorizer.core.raster_utils import (
    MAX_RASTER_BLOCK_BYTES,
    compute_resampled_dimensions,
    raster_block_to_native_array,
    raster_block_to_rgb_uint8,
    raster_block_to_uint8,
    read_raster_bands,
    read_raster_bands_with_native,
    stable_integer_band_to_uint8,
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

    np.testing.assert_array_equal(converted, np.array([[255, 0, 127, 255]], dtype=np.uint8))


def test_internal_nodata_bitmap_is_respected_for_byte_rasters():
    block = _RasterBlock(
        np.array([0, 10, 20], dtype=np.uint8),
        "Byte",
        nodata_indices=(0,),
    )

    converted = raster_block_to_uint8(block, 3, 1)

    np.testing.assert_array_equal(converted, np.array([[255, 10, 20]], dtype=np.uint8))


def test_native_integer_nodata_uses_fixed_bright_value_not_block_median():
    block = _RasterBlock(
        np.array([10, 1000, 2000], dtype=np.uint16),
        "UInt16",
        nodata_indices=(0,),
    )

    native = raster_block_to_native_array(block, 3, 1)

    np.testing.assert_array_equal(
        native,
        np.array([[65535, 1000, 2000]], dtype=np.uint16),
    )


def test_stable_integer_display_does_not_depend_on_block_extrema():
    first = stable_integer_band_to_uint8(
        np.array([[0, 1000]], dtype=np.uint16)
    )
    second = stable_integer_band_to_uint8(
        np.array([[1000, 60000]], dtype=np.uint16)
    )

    assert int(first[0, 1]) == int(second[0, 0])


def test_combined_reader_uses_one_provider_block_and_attests_integer_dtypes():
    class Provider:
        def __init__(self, block):
            self.block_value = block
            self.calls = 0

        def bandCount(self):
            return 1

        def dataType(self, _band_number):
            return self.block_value.dataType()

        def block(self, *_args):
            self.calls += 1
            return self.block_value

    integer_provider = Provider(
        _RasterBlock(np.array([100, 200], dtype=np.uint16), "UInt16")
    )
    display, native, stable = read_raster_bands_with_native(
        integer_provider,
        extent=None,
        width=2,
        height=1,
    )
    assert integer_provider.calls == 1
    assert stable is True
    np.testing.assert_array_equal(display[0], np.array([[0, 255]], dtype=np.uint8))
    np.testing.assert_array_equal(
        native[0], np.array([[100, 200]], dtype=np.uint16)
    )

    float_provider = Provider(
        _RasterBlock(np.array([0.0, 1.0], dtype=np.float32), "Float32")
    )
    _display, _native, stable = read_raster_bands_with_native(
        float_provider,
        extent=None,
        width=2,
        height=1,
    )
    assert stable is False


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


@pytest.mark.parametrize("values", [
    [np.nan, np.nan], [np.inf, -np.inf], [-9999.0, -9999.0],
])
def test_empty_float_background_is_white_paper(values):
    block = _RasterBlock(np.array(values, dtype=np.float32), "Float32", nodata_value=-9999.0)
    np.testing.assert_array_equal(raster_block_to_uint8(block, 2, 1), [[255, 255]])
    assert np.isnan(raster_block_to_native_array(block, 2, 1)).all()


def test_nan_and_infinite_pixels_do_not_become_midgray_strokes():
    block = _RasterBlock(np.array([0.0, 1.0, np.nan, np.inf, -np.inf], dtype=np.float32), "Float32")
    np.testing.assert_array_equal(raster_block_to_uint8(block, 5, 1), [[0, 255, 255, 255, 255]])


def test_nodata_value_and_internal_bitmap_are_both_honored():
    block = _RasterBlock(np.array([0, 50, 100], dtype=np.uint8), "Byte", nodata_value=0, nodata_indices=(1,))
    np.testing.assert_array_equal(raster_block_to_uint8(block, 3, 1), [[255, 255, 100]])


def test_all_byte_nodata_pixels_are_white():
    block = _RasterBlock(np.zeros(3, dtype=np.uint8), "Byte", nodata_indices=(0, 1, 2))
    np.testing.assert_array_equal(raster_block_to_uint8(block, 3, 1), [[255, 255, 255]])


class _Provider:
    def __init__(self, blocks, *, alpha=None, nodata=None, use_nodata=True):
        self.blocks = blocks
        self.alpha = alpha
        self.nodata = nodata
        self.use_nodata = use_nodata
        self.calls = []

    def bandCount(self):
        return len(self.blocks)

    def dataType(self, band):
        return self.blocks[band - 1].dataType()

    def block(self, band, *_args):
        self.calls.append(band)
        return self.blocks[band - 1]

    def colorInterpretation(self, band):
        return _DataType("AlphaBand" if band == self.alpha else "Undefined")

    def sourceHasNoDataValue(self, _band):
        return self.nodata is not None

    def sourceNoDataValue(self, _band):
        return self.nodata

    def useSourceNoDataValue(self, _band):
        return self.use_nodata


def test_provider_source_nodata_without_block_sentinel_and_user_override():
    block = _RasterBlock(np.array([0, 10, 20], dtype=np.uint8), "Byte")
    for use, expected in ((True, [255, 10, 20]), (False, [0, 10, 20])):
        provider = _Provider([block], nodata=0, use_nodata=use)
        display, native, stable = read_raster_bands_with_native(provider, None, 3, 1)
        np.testing.assert_array_equal(display[0], [expected])
        np.testing.assert_array_equal(native[0], [expected])
        assert stable and provider.calls == [1]


def test_opaque_rgb_byte_values_remain_exact_native_source_samples():
    bands = [np.array([13, 81, 254], dtype=np.uint8), np.array([29, 8, 255], dtype=np.uint8),
             np.array([201, 17, 0], dtype=np.uint8)]
    provider = _Provider([_RasterBlock(b, "Byte") for b in bands])
    display, native, stable = read_raster_bands_with_native(provider, None, 3, 1)
    assert stable and provider.calls == [1, 2, 3]
    for expected, shown, source in zip(bands, display, native):
        np.testing.assert_array_equal(shown, expected[None])
        np.testing.assert_array_equal(source, expected[None])


def test_missing_rgb_channel_whitens_the_whole_missing_pixel():
    provider = _Provider([
        _RasterBlock(np.array([0, 15], dtype=np.uint8), "Byte", nodata_indices=(0,)),
        _RasterBlock(np.array([20, 16], dtype=np.uint8), "Byte"),
        _RasterBlock(np.array([30, 17], dtype=np.uint8), "Byte"),
    ])
    display, native, stable = read_raster_bands_with_native(provider, None, 2, 1)
    assert stable
    np.testing.assert_array_equal(np.stack(display, axis=-1), [[[255, 255, 255], [15, 16, 17]]])
    np.testing.assert_array_equal(np.stack(native, axis=-1), np.stack(display, axis=-1))


@pytest.mark.parametrize("kind, values", [
    ("ARGB32", [0x00000000, 0x00008020, 0x800A14C8, 0xFF0D1BCC]),
    ("ARGB32_Premultiplied", [0x00000000, 0x00008020, 0x80050A64, 0xFF0D1BCC]),
])
def test_packed_alpha_is_composited_once_and_opaque_color_is_exact(kind, values):
    block = _RasterBlock(np.array(values, dtype=np.uint32), kind)
    expected = np.array([[[255, 255, 255], [255, 255, 255], [132, 137, 227], [13, 27, 204]]], dtype=np.uint8)
    np.testing.assert_array_equal(raster_block_to_rgb_uint8(block, 4, 1), expected)
    provider = _Provider([block])
    display, native, stable = read_raster_bands_with_native(provider, None, 4, 1)
    np.testing.assert_array_equal(np.stack(display, axis=-1), expected)
    assert native == [] and stable is False and provider.calls == [1]
    # A packed rendered sample may never be advertised as a native integer band.
    assert raster_block_to_native_array(block, 4, 1) is None


def test_oversized_packed_color_request_stops_before_provider_allocation():
    provider = _Provider([_RasterBlock(np.zeros(1, dtype=np.uint32), "ARGB32")])
    assert read_raster_bands(provider, None, 5000, 4000) == []
    assert provider.calls == []


def test_packed_color_invalid_payload_is_not_reinterpreted_as_numeric():
    block = _RasterBlock(np.zeros(3, dtype=np.uint8), "ARGB32")
    assert raster_block_to_rgb_uint8(block, 3, 1) is None
    assert read_raster_bands(_Provider([block]), None, 3, 1) == []


def test_explicit_byte_alpha_band_uses_white_paper_without_native_attestation():
    provider = _Provider([
        _RasterBlock(np.array([0, 10, 13], dtype=np.uint8), "Byte"),
        _RasterBlock(np.array([0, 20, 27], dtype=np.uint8), "Byte"),
        _RasterBlock(np.array([0, 200, 204], dtype=np.uint8), "Byte"),
        _RasterBlock(np.array([0, 128, 255], dtype=np.uint8), "Byte"),
    ], alpha=4)
    display, native, stable = read_raster_bands_with_native(provider, None, 3, 1)
    np.testing.assert_array_equal(np.stack(display, axis=-1), [[[255, 255, 255], [132, 137, 227], [13, 27, 204]]])
    assert provider.calls == [1, 2, 3, 4] and native == [] and stable is False


def test_undeclared_fourth_band_is_not_guessed_to_be_alpha():
    provider = _Provider([_RasterBlock(np.array([10, 20], dtype=np.uint8), "Byte") for _ in range(4)])
    display, native, stable = read_raster_bands_with_native(provider, None, 2, 1)
    assert provider.calls == [1, 2, 3] and len(display) == len(native) == 3 and stable
    assert all(b.tolist() == [[10, 20]] for b in display)


def test_explicit_gray_alpha_is_not_mistaken_for_a_color_channel():
    provider = _Provider([
        _RasterBlock(np.array([0, 25], dtype=np.uint8), "Byte"),
        _RasterBlock(np.array([0, 255], dtype=np.uint8), "Byte"),
    ], alpha=2)
    display, native, stable = read_raster_bands_with_native(provider, None, 2, 1)
    assert len(display) == 1 and not native and not stable
    np.testing.assert_array_equal(display[0], [[255, 25]])


def test_nodata_alpha_is_transparent_and_unsupported_alpha_declines_the_read():
    gray = _RasterBlock(np.array([0, 25], dtype=np.uint8), "Byte")
    alpha = _RasterBlock(np.array([255, 255], dtype=np.uint8), "Byte", nodata_indices=(0,))
    np.testing.assert_array_equal(read_raster_bands(_Provider([gray, alpha], alpha=2), None, 2, 1)[0], [[255, 25]])
    invalid_alpha = _RasterBlock(np.array([0., 1.], dtype=np.float32), "Float32")
    assert read_raster_bands(_Provider([gray, invalid_alpha], alpha=2), None, 2, 1) == []


def test_uint16_alpha_uses_fixed_range_instead_of_block_contrast():
    provider = _Provider([
        _RasterBlock(np.array([0, 0, 0], dtype=np.uint8), "Byte"),
        _RasterBlock(np.array([0, 32768, 65535], dtype=np.uint16), "UInt16"),
    ], alpha=2)
    np.testing.assert_array_equal(read_raster_bands(provider, None, 3, 1)[0], [[255, 127, 0]])


def test_block_declared_packed_type_is_used_when_provider_type_is_unavailable():
    class Provider(_Provider):
        def dataType(self, band):
            raise RuntimeError("provider has no declared type")

    block = _RasterBlock(np.array([0, 0xFF0D1BCC], dtype=np.uint32), "ARGB32")
    display, native, stable = read_raster_bands_with_native(Provider([block]), None, 2, 1)
    np.testing.assert_array_equal(np.stack(display, axis=-1), [[[255, 255, 255], [13, 27, 204]]])
    assert native == [] and stable is False


@pytest.mark.parametrize("missing,nodata", [(np.nan, None), (-9999.0, -9999.0)])
def test_single_valued_float_ink_survives_missing_background(missing, nodata):
    values = np.full((5, 7), missing, dtype=np.float32)
    values[2, 1:6] = 0.0
    block = _RasterBlock(values, "Float32", nodata_value=nodata)
    converted = raster_block_to_uint8(block, 7, 5)
    expected = np.full((5, 7), 255, dtype=np.uint8)
    expected[2, 1:6] = 0
    np.testing.assert_array_equal(converted, expected)


def test_constant_valid_float_preserves_legacy_zero_contrast_mapping():
    block = _RasterBlock(np.full((2, 3), 100.0, dtype=np.float32), "Float32")
    np.testing.assert_array_equal(raster_block_to_uint8(block, 3, 2), np.zeros((2, 3), dtype=np.uint8))
