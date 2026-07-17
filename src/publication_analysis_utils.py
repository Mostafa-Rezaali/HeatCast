#!/usr/bin/env python3
"""Shared utilities for MeshFlowNet publication analyses."""

from __future__ import annotations

import glob
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


STAT_KEYS: Tuple[str, ...] = (
    "pred_sum",
    "truth_sum",
    "pred_sq_sum",
    "truth_sq_sum",
    "pred_truth_sum",
    "persist_sum",
    "persist_sq_sum",
    "persist_truth_sum",
    "count",
)

EXTENT = (-130.0, -60.0, 25.0, 50.0)
LAT_RANGE = (25.0, 50.0)
LON_RANGE = (-130.0, -60.0)


NOAA_REGION_BOXES: Mapping[str, Tuple[float, float, float, float]] = {
    "Northeast": (37.0, 48.0, -80.0, -67.0),
    "Southeast": (25.0, 37.0, -90.0, -75.0),
    "Midwest": (37.0, 48.0, -104.0, -80.0),
    "Great Plains": (25.0, 48.0, -104.0, -95.0),
    "Southwest": (25.0, 37.0, -120.0, -104.0),
    "Northwest": (42.0, 50.0, -125.0, -110.0),
    "West": (37.0, 42.0, -125.0, -104.0),
}


def expand_inputs(patterns: Sequence[str]) -> List[str]:
    paths: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif os.path.exists(pattern):
            paths.append(pattern)
    return sorted(dict.fromkeys(paths))


def default_fold_stat_paths(root: str = ".") -> List[str]:
    return sorted(
        glob.glob(os.path.join(root, "hindcast_stats", "hindcast_tac_stats_cvfold*_test.npz"))
    )


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def stitch_stats(paths: Sequence[str]) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[int], int]:
    if not paths:
        raise FileNotFoundError("No hindcast stat files were provided.")

    aggregate: Dict[str, np.ndarray] | None = None
    mask = None
    years = set()
    total_samples = 0

    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if aggregate is None:
                aggregate = {key: np.asarray(data[key], dtype=np.float64).copy() for key in STAT_KEYS}
                mask = np.asarray(data["mask"], dtype=np.uint8).copy()
            else:
                for key in STAT_KEYS:
                    aggregate[key] += np.asarray(data[key], dtype=np.float64)

            if "years" in data:
                years.update(int(y) for y in np.atleast_1d(data["years"]).astype(int).tolist())
            if "n_samples" in data:
                total_samples += int(np.asarray(data["n_samples"]).item())

    assert aggregate is not None and mask is not None
    return aggregate, mask, sorted(years), total_samples


def corr_map_from_sums(
    x_sum: np.ndarray,
    y_sum: np.ndarray,
    x_sq_sum: np.ndarray,
    y_sq_sum: np.ndarray,
    xy_sum: np.ndarray,
    count: np.ndarray,
    mask: np.ndarray,
    min_count: int = 2,
) -> np.ndarray:
    valid = (mask > 0.5) & (count >= min_count)
    safe_count = np.maximum(count, 1.0)
    cov = xy_sum - (x_sum * y_sum) / safe_count
    x_var = x_sq_sum - (x_sum * x_sum) / safe_count
    y_var = y_sq_sum - (y_sum * y_sum) / safe_count
    denom = np.sqrt(np.maximum(x_var, 0.0) * np.maximum(y_var, 0.0))
    valid &= denom > 1e-12
    corr = np.full(count.shape, np.nan, dtype=np.float32)
    corr[valid] = (cov[valid] / denom[valid]).astype(np.float32)
    return corr


def model_persistence_corr_maps(
    stats: Mapping[str, np.ndarray], mask: np.ndarray, min_count: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    model = corr_map_from_sums(
        stats["pred_sum"],
        stats["truth_sum"],
        stats["pred_sq_sum"],
        stats["truth_sq_sum"],
        stats["pred_truth_sum"],
        stats["count"],
        mask,
        min_count=min_count,
    )
    persistence = corr_map_from_sums(
        stats["persist_sum"],
        stats["truth_sum"],
        stats["persist_sq_sum"],
        stats["truth_sq_sum"],
        stats["persist_truth_sum"],
        stats["count"],
        mask,
        min_count=min_count,
    )
    return model, persistence


def land_mean(field: np.ndarray, mask: np.ndarray) -> float:
    valid = (mask > 0.5) & np.isfinite(field)
    return float(np.nanmean(field[valid])) if np.any(valid) else float("nan")


def mse_from_sums(
    pred_sq_sum: np.ndarray,
    truth_sq_sum: np.ndarray,
    pred_truth_sum: np.ndarray,
    count: np.ndarray,
    mask: np.ndarray,
) -> float:
    valid = (mask > 0.5) & (count > 0)
    sqerr_sum = pred_sq_sum + truth_sq_sum - 2.0 * pred_truth_sum
    denom = np.maximum(count, 1.0)
    per_pixel = np.full(count.shape, np.nan, dtype=np.float64)
    per_pixel[valid] = sqerr_sum[valid] / denom[valid]
    return land_mean(per_pixel, mask)


def climatology_mse_from_stats(stats: Mapping[str, np.ndarray], mask: np.ndarray) -> float:
    zeros = np.zeros_like(stats["truth_sq_sum"])
    return mse_from_sums(
        zeros,
        stats["truth_sq_sum"],
        zeros,
        stats["count"],
        mask,
    )


def conus_lat_lon(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    lat_1d = np.linspace(LAT_RANGE[1], LAT_RANGE[0], h, dtype=np.float32)
    lon_1d = np.linspace(LON_RANGE[0], LON_RANGE[1], w, dtype=np.float32)
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
    return lat_1d, lon_1d, lat2d, lon2d


def region_masks(shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
    _, _, lat2d, lon2d = conus_lat_lon(shape)
    masks = {}
    for name, (lat0, lat1, lon0, lon1) in NOAA_REGION_BOXES.items():
        masks[name] = (lat2d >= lat0) & (lat2d <= lat1) & (lon2d >= lon0) & (lon2d <= lon1)
    return masks


def fisher_z(r: np.ndarray) -> np.ndarray:
    r = np.clip(np.asarray(r, dtype=np.float64), -0.999999, 0.999999)
    return 0.5 * np.log((1.0 + r) / (1.0 - r))


def normal_p_two_sided(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    erfc_vec = np.vectorize(math.erfc)
    return erfc_vec(np.abs(z) / math.sqrt(2.0))


def fdr_bh(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p_values, dtype=np.float64)
    valid = np.isfinite(p)
    corrected = np.full(p.shape, np.nan, dtype=np.float64)
    reject = np.zeros(p.shape, dtype=bool)
    if not np.any(valid):
        return reject, corrected

    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = float(len(ranked))
    thresholds = alpha * (np.arange(1, len(ranked) + 1) / m)
    passed = ranked <= thresholds
    if np.any(passed):
        cutoff = np.max(np.where(passed)[0])
        reject_valid = np.zeros_like(pv, dtype=bool)
        reject_valid[order[: cutoff + 1]] = True
    else:
        reject_valid = np.zeros_like(pv, dtype=bool)

    q = np.empty_like(ranked)
    running = 1.0
    for i in range(len(ranked) - 1, -1, -1):
        running = min(running, ranked[i] * m / float(i + 1))
        q[i] = running
    corrected_valid = np.empty_like(pv)
    corrected_valid[order] = np.clip(q, 0.0, 1.0)

    reject[valid] = reject_valid
    corrected[valid] = corrected_valid
    return reject, corrected


def write_latex_table(path: str | os.PathLike[str], header: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = "l" + "r" * (len(header) - 1)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"\\begin{{tabular}}{{{spec}}}\\hline\n")
        f.write(" & ".join(header) + " \\\\ \\hline\n")
        for row in rows:
            f.write(" & ".join(str(x) for x in row) + " \\\\\n")
        f.write("\\hline\\end{tabular}\n")

