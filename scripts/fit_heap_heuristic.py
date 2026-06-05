#!/usr/bin/env python3
"""Fit per-codec save-memory ratio models from MsPacman benchmark (README Save Mem %).

Dev-only: requires scipy. Run::

    uv pip install scipy
    uv run scripts/fit_heap_heuristic.py

Copy the printed block into ``sb3_extra_buffers/gpu_buffers/size_estimation.py``.
Extrapolation policy: evaluate poly, clamp to benchmark min/max ratio per family;
if level is outside measured level range, use max(fitted, family_max_ratio).
"""

# (label, family, level, save_mem_pct) — README benchmark table, Save Mem % only
BENCHMARK_ROWS = [
    ("baseline", "baseline", None, 100.0),
    ("none", "none", None, 100.1),
    ("zstd-100", "zstd", -100, 36.0),
    ("zstd-50", "zstd", -50, 28.4),
    ("zstd-20", "zstd", -20, 16.8),
    ("zstd-5", "zstd", -5, 7.7),
    ("zstd-3", "zstd", -3, 6.9),
    ("zstd-1", "zstd", -1, 6.1),
    ("zstd1", "zstd", 1, 5.7),
    ("zstd3", "zstd", 3, 5.5),
    ("zstd5", "zstd", 5, 5.2),
    ("zstd10", "zstd", 10, 4.9),
    ("zstd15", "zstd", 15, 4.5),
    ("zstd22", "zstd", 22, 4.4),
    ("rle", "rle", None, 75.3),
    ("rle-jit", "rle", None, 75.3),
    ("rle-old", "rle", None, 75.3),
    ("igzip0", "igzip", 0, 12.0),
    ("igzip1", "igzip", 1, 10.6),
    ("igzip3", "igzip", 3, 7.6),
    ("gzip1", "gzip", 1, 9.6),
    ("gzip3", "gzip", 3, 7.6),
    ("lz4-frame/1", "lz4-frame", 1, 10.9),
    ("lz4-frame/5", "lz4-frame", 5, 7.0),
    ("lz4-frame/9", "lz4-frame", 9, 6.8),
    ("lz4-frame/12", "lz4-frame", 12, 6.7),
    ("lz4-block/1", "lz4-block", 1, 7.7),
    ("lz4-block/5", "lz4-block", 5, 7.0),
    ("lz4-block/9", "lz4-block", 9, 6.7),
    ("lz4-block/16", "lz4-block", 16, 6.6),
]

LEVELLED_FAMILIES = ("zstd", "gzip", "igzip", "lz4-frame", "lz4-block")

# Sample unmeasured levels for extrapolation demo in script output
SAMPLE_UNMEASURED = [
    ("gzip", 2),
    ("gzip", 7),
    ("igzip", 2),
    ("zstd", 7),
    ("zstd", -7),
    ("lz4-frame", 3),
    ("lz4-block", 3),
]


def _group_by_family():
    groups = {}
    for _label, family, level, pct in BENCHMARK_ROWS:
        groups.setdefault(family, []).append((level, pct / 100.0))
    return groups


def _fit_polynomial(levels, ratios, degree):
    import numpy as np

    coeffs = np.polyfit(levels, ratios, degree)
    pred = np.polyval(coeffs, levels)
    max_err = float(np.max(np.abs(pred - ratios)))
    return coeffs, max_err


def _pick_degree(levels, ratios):
    import numpy as np

    n = len(levels)
    best_degree = 1
    best_coeffs = np.polyfit(levels, ratios, 1)
    best_err = float(np.max(np.abs(np.polyval(best_coeffs, levels) - ratios)))
    for degree in range(2, min(3, n - 1) + 1):
        if degree >= n:
            break
        coeffs, err = _fit_polynomial(levels, ratios, degree)
        if err <= best_err * 1.05:
            best_degree = degree
            best_coeffs = coeffs
            best_err = err
    return best_degree, best_coeffs, best_err


def _format_coeffs(coeffs):

    c = [float(x) for x in coeffs]
    if len(c) == 1:
        return f"({c[0]:.12g},)"
    return "(" + ", ".join(f"{x:.12g}" for x in c) + ")"


def main():
    import numpy as np

    try:
        import scipy  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install scipy: uv pip install scipy") from exc

    groups = _group_by_family()
    family_models = {}

    print("=== Validation (benchmark rows) ===")
    print(f"{'label':<16} {'family':<12} {'level':>6} {'actual%':>8} {'fitted%':>8} {'err':>8}")
    for label, family, level, pct in BENCHMARK_ROWS:
        ratio = pct / 100.0
        if level is None:
            family_models[family] = {"kind": "constant", "ratio": ratio}
            fitted = ratio
        else:
            if family not in family_models or family_models[family].get("kind") != "poly":
                levels = [lv for lv, r in groups[family] if lv is not None]
                ratios = [r for lv, r in groups[family] if lv is not None]
                deg, coeffs, max_err = _pick_degree(levels, ratios)
                lv_arr = [lv for lv, r in groups[family] if lv is not None]
                family_models[family] = {
                    "kind": "poly",
                    "coeffs": coeffs,
                    "min_level": min(lv_arr),
                    "max_level": max(lv_arr),
                    "min_ratio": min(ratios),
                    "max_ratio": max(ratios),
                    "max_err": max_err,
                    "degree": deg,
                }
            model = family_models[family]
            fitted = float(np.polyval(model["coeffs"], level))
            fitted = max(model["min_ratio"], min(model["max_ratio"], fitted))
            if level < model["min_level"] or level > model["max_level"]:
                fitted = max(fitted, model["max_ratio"])
        err = abs(fitted - ratio) * 100
        print(f"{label:<16} {family:<12} {str(level):>6} {pct:8.2f} {fitted * 100:8.2f} {err:8.3f}")

    print("\n=== Sample unmeasured levels (after clamp / outside-hull rule) ===")
    for family, level in SAMPLE_UNMEASURED:
        model = family_models[family]
        fitted = float(np.polyval(model["coeffs"], level))
        fitted = max(model["min_ratio"], min(model["max_ratio"], fitted))
        if level < model["min_level"] or level > model["max_level"]:
            fitted = max(fitted, model["max_ratio"])
        print(f"  {family} level {level:>4} -> ratio {fitted:.4f} ({fitted * 100:.2f}%)")

    print("\n=== Copy into sb3_extra_buffers/gpu_buffers/size_estimation.py ===\n")
    print("# MsPacman Save Mem benchmark — generated by scripts/fit_heap_heuristic.py")
    print("# Extrapolation: poly eval, clamp to family benchmark min/max ratio;")
    print("# outside measured level range -> max(fitted, family_max_ratio).")
    print(f"_SAVE_RATIO_BASELINE = {groups['baseline'][0][1]:.6g}")
    print(f"_SAVE_RATIO_NONE = {groups['none'][0][1]:.6g}")
    print(f"_SAVE_RATIO_RLE = {groups['rle'][0][1]:.6g}")
    print("_DEFAULT_ZSTD_LEVEL = -3  # README recommends zstd-3 when level omitted")
    print()

    for family in LEVELLED_FAMILIES:
        model = family_models[family]
        coeffs = model["coeffs"]
        # high_to_low = list(reversed([float(c) for c in coeffs]))
        print(f"_{family.upper().replace('-', '_')}_POLY = {_format_coeffs(coeffs)}")
        print(
            f"_{family.upper().replace('-', '_')}_BOUNDS = "
            f"({model['min_level']}, {model['max_level']}, "
            f"{model['min_ratio']:.6g}, {model['max_ratio']:.6g})"
        )
    print()
    print("# family_models max fit error (pct points):")
    for family in LEVELLED_FAMILIES:
        max_err = family_models[family]["max_err"] * 100
        print(f"#   {family}: degree {family_models[family]['degree']}, max_err {max_err:.3f}%")


if __name__ == "__main__":
    main()
