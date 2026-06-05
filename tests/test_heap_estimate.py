"""Tests for benchmark-derived heap size estimates."""

import pytest
import torch as th

from sb3_extra_buffers.gpu_buffers.size_estimation import (
    estimate_save_ratio,
    parse_compression_method,
)
from sb3_extra_buffers.gpu_buffers.utils import estimate_max_slot_bytes

REFERENCE_FLAT_LEN = 84 * 84 * 4


def test_parse_compression_method():
    assert parse_compression_method("rle") == ("rle", None)
    assert parse_compression_method("zstd3") == ("zstd", 3)
    assert parse_compression_method("zstd-5") == ("zstd", -5)
    assert parse_compression_method("lz4-frame/5") == ("lz4-frame", 5)


@pytest.mark.parametrize(
    "method,expected_ratio",
    [
        ("none", 1.001),
        ("rle", 0.753),
        ("zstd3", 0.055),
        ("zstd-3", 0.069),
        ("zstd-5", 0.077),
        ("gzip1", 0.096),
        ("igzip3", 0.076),
    ],
)
def test_benchmark_rows(method, expected_ratio):
    ratio = estimate_save_ratio(method)
    assert ratio == pytest.approx(expected_ratio, rel=0.1)


def test_zstd_bare_uses_default_level():
    assert estimate_save_ratio("zstd") == pytest.approx(estimate_save_ratio("zstd-3"), rel=1e-6)


def test_unmeasured_valid_levels_in_family_envelope():
    for method in ("gzip2", "igzip2", "zstd7", "lz4-frame/3"):
        family, level = parse_compression_method(method)
        ratio = estimate_save_ratio(method)
        del family, level
        assert 0.044 <= ratio <= 0.36


def test_outside_hull_zstd_negative():
    ratio = estimate_save_ratio("zstd-7")
    assert ratio >= 0.077


@pytest.mark.parametrize("method", ["gzip10", "igzip9", "zstd0"])
def test_invalid_level_raises(method):
    with pytest.raises(ValueError):
        estimate_save_ratio(method)


def test_explicit_compresslevel_overrides_method_suffix():
    from_zstd3 = estimate_save_ratio("zstd3")
    explicit = estimate_save_ratio("zstd", compresslevel=3)
    assert explicit == pytest.approx(from_zstd3, rel=1e-6)
    assert estimate_save_ratio("zstd3", compresslevel=5) == pytest.approx(
        estimate_save_ratio("zstd5"),
        rel=1e-6,
    )


def test_estimate_max_slot_bytes_explicit_compresslevel():
    with_level = estimate_max_slot_bytes(
        REFERENCE_FLAT_LEN,
        th.uint8,
        th.uint16,
        "zstd",
        compresslevel=3,
    )
    from_name = estimate_max_slot_bytes(REFERENCE_FLAT_LEN, th.uint8, th.uint16, "zstd3")
    assert with_level == from_name


def test_overalloc_factor_scales():
    base = estimate_max_slot_bytes(REFERENCE_FLAT_LEN, th.uint8, th.uint16, "none", overalloc_factor=1.5)
    double = estimate_max_slot_bytes(REFERENCE_FLAT_LEN, th.uint8, th.uint16, "none", overalloc_factor=3.0)
    assert double == pytest.approx(int(base * 2), rel=0.01)


def test_estimates_use_legacy_floor_for_rle():
    flat_len = REFERENCE_FLAT_LEN
    elem = th.uint8
    runs = th.uint16
    legacy_rle = flat_len + flat_len * th.tensor([], dtype=runs).element_size()
    assert estimate_max_slot_bytes(flat_len, elem, runs, "rle") == legacy_rle


def test_zstd_estimate_at_most_legacy_cap():
    flat_len = REFERENCE_FLAT_LEN
    elem = th.uint8
    runs = th.uint16
    legacy_zstd = flat_len // 4 + flat_len // 255 + 64
    assert estimate_max_slot_bytes(flat_len, elem, runs, "zstd3") <= legacy_zstd
