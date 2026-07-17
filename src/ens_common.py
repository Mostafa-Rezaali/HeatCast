#!/usr/bin/env python3
"""Lightweight numerical helpers for the ECMWF ENS benchmark."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


ENS_BENCHMARK_BANNER = """ECMWF ENS benchmark on the HeatCast scoreboard
================================================
Coverage: score only the per-fold intersection of ENS reforecast years and HeatCast test years.
Initialization: score the downloaded MJJAS S2S hdate initializations common to ENS and HeatCast.
Variable: ENS daily maximum 2 m temperature versus PRISM T2max.
Members: ENS probabilities use the 11-member reforecast fraction, calibrated on fold validation years.
Cycle discipline: bias correction and calibration are fit separately for each model cycle before merging.
Discipline: identical PRISM events, grid, thresholds, init dates, test years, and fold-safe calibration."""


def fit_quantile_mapping(
    source: np.ndarray,
    target: np.ndarray,
    quantile_levels: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit monotonic per-pixel quantile pairs along axis zero."""
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError(
            f"Quantile mapping expects source/target shaped (samples,pixels), got "
            f"{source.shape} and {target.shape}."
        )
    levels = np.asarray(quantile_levels, dtype=np.float64)
    if levels.ndim != 1 or levels.size < 2 or np.any(np.diff(levels) <= 0):
        raise ValueError("quantile_levels must be a strictly increasing 1D sequence.")
    with np.errstate(all="ignore"):
        source_q = np.nanquantile(source, levels, axis=0).astype(np.float32)
        target_q = np.nanquantile(target, levels, axis=0).astype(np.float32)
    source_q = np.maximum.accumulate(source_q, axis=0)
    target_q = np.maximum.accumulate(target_q, axis=0)
    return source_q, target_q


def apply_quantile_mapping(
    values: np.ndarray,
    source_quantiles: np.ndarray,
    target_quantiles: np.ndarray,
    block_size: int = 4096,
) -> np.ndarray:
    """Apply per-pixel piecewise-linear quantile mapping with clipped tails."""
    values = np.asarray(values, dtype=np.float32)
    source_q = np.asarray(source_quantiles, dtype=np.float32)
    target_q = np.asarray(target_quantiles, dtype=np.float32)
    if source_q.shape != target_q.shape or source_q.ndim != 2:
        raise ValueError("source_quantiles and target_quantiles must share shape (quantiles,pixels).")
    if values.shape[-1] != source_q.shape[1]:
        raise ValueError(
            f"Value pixel dimension {values.shape[-1]} does not match mapping {source_q.shape[1]}."
        )
    original_shape = values.shape
    flat = values.reshape(-1, values.shape[-1])
    out = np.full(flat.shape, np.nan, dtype=np.float32)
    for start in range(0, flat.shape[1], int(block_size)):
        stop = min(start + int(block_size), flat.shape[1])
        value_block = flat[:, start:stop]
        source_block = source_q[:, start:stop]
        target_block = target_q[:, start:stop]
        finite = np.isfinite(value_block)
        source_finite = np.all(np.isfinite(source_block), axis=0)
        target_finite = np.all(np.isfinite(target_block), axis=0)
        usable_pixels = source_finite & target_finite
        if not np.any(usable_pixels):
            continue
        v = value_block[:, usable_pixels]
        sq = source_block[:, usable_pixels]
        tq = target_block[:, usable_pixels]
        index = np.sum(v[:, None, :] >= sq[None, :, :], axis=1) - 1
        below = v <= sq[0][None, :]
        above = v >= sq[-1][None, :]
        index = np.clip(index, 0, sq.shape[0] - 2)
        pixel_first_index = index.T
        sq_t = sq.T
        tq_t = tq.T
        x0 = np.take_along_axis(sq_t, pixel_first_index, axis=1).T
        x1 = np.take_along_axis(sq_t, pixel_first_index + 1, axis=1).T
        y0 = np.take_along_axis(tq_t, pixel_first_index, axis=1).T
        y1 = np.take_along_axis(tq_t, pixel_first_index + 1, axis=1).T
        fraction = np.divide(
            v - x0,
            x1 - x0,
            out=np.zeros_like(v, dtype=np.float32),
            where=np.abs(x1 - x0) > 1e-8,
        )
        mapped = y0 + fraction * (y1 - y0)
        mapped[below] = np.broadcast_to(tq[0], mapped.shape)[below]
        mapped[above] = np.broadcast_to(tq[-1], mapped.shape)[above]
        mapped[~finite[:, usable_pixels]] = np.nan
        destination = out[:, start:stop]
        destination[:, usable_pixels] = mapped
        out[:, start:stop] = destination
    return out.reshape(original_shape)


def member_fraction_probability(member_window_means: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    members = np.asarray(member_window_means, dtype=np.float32)
    threshold = np.asarray(threshold, dtype=np.float32)
    if members.ndim != 2 or threshold.ndim != 1 or members.shape[1] != threshold.size:
        raise ValueError("Expected member_window_means=(members,pixels), threshold=(pixels,).")
    valid = np.isfinite(members) & np.isfinite(threshold[None, :])
    count = valid.sum(axis=0)
    exceed = ((members > threshold[None, :]) & valid).sum(axis=0)
    probability = np.divide(
        exceed,
        count,
        out=np.full(threshold.shape, np.nan, dtype=np.float32),
        where=count > 0,
    )
    return probability.astype(np.float32)


def common_init_indices(
    first: Mapping[int, object],
    second: Mapping[int, object],
) -> Tuple[int, ...]:
    """Return sorted common init-time indices."""
    return tuple(sorted(set(int(v) for v in first) & set(int(v) for v in second)))


def intersection_years(
    first_years: Iterable[int],
    second_years: Iterable[int],
) -> Tuple[int, ...]:
    return tuple(sorted(set(int(v) for v in first_years) & set(int(v) for v in second_years)))


def bilinear_regrid_regular(
    values: np.ndarray,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    block_size: int = 65536,
) -> np.ndarray:
    """Bilinearly regrid arrays with trailing (lat,lon) dimensions."""
    values = np.asarray(values, dtype=np.float32)
    lat = np.asarray(source_lat, dtype=np.float64).reshape(-1)
    lon = np.asarray(source_lon, dtype=np.float64).reshape(-1)
    if values.shape[-2:] != (lat.size, lon.size):
        raise ValueError(
            f"Values trailing shape {values.shape[-2:]} does not match lat/lon {(lat.size, lon.size)}."
        )
    lon = ((lon + 180.0) % 360.0) - 180.0
    lat_order = np.argsort(lat)
    lon_order = np.argsort(lon)
    lat = lat[lat_order]
    lon = lon[lon_order]
    values = np.take(np.take(values, lat_order, axis=-2), lon_order, axis=-1)
    target_lat_flat = np.asarray(target_lat, dtype=np.float64).reshape(-1)
    target_lon_flat = ((np.asarray(target_lon, dtype=np.float64).reshape(-1) + 180.0) % 360.0) - 180.0
    leading = values.shape[:-2]
    source_flat = values.reshape(-1, lat.size * lon.size)
    result = np.empty((source_flat.shape[0], target_lat_flat.size), dtype=np.float32)
    for start in range(0, target_lat_flat.size, int(block_size)):
        stop = min(start + int(block_size), target_lat_flat.size)
        tlat = target_lat_flat[start:stop]
        tlon = target_lon_flat[start:stop]
        yi1 = np.clip(np.searchsorted(lat, tlat, side="right"), 1, lat.size - 1)
        xi1 = np.clip(np.searchsorted(lon, tlon, side="right"), 1, lon.size - 1)
        yi0 = yi1 - 1
        xi0 = xi1 - 1
        wy = np.divide(tlat - lat[yi0], lat[yi1] - lat[yi0], out=np.zeros_like(tlat), where=lat[yi1] != lat[yi0])
        wx = np.divide(tlon - lon[xi0], lon[xi1] - lon[xi0], out=np.zeros_like(tlon), where=lon[xi1] != lon[xi0])
        i00 = yi0 * lon.size + xi0
        i01 = yi0 * lon.size + xi1
        i10 = yi1 * lon.size + xi0
        i11 = yi1 * lon.size + xi1
        result[:, start:stop] = (
            source_flat[:, i00] * ((1.0 - wy) * (1.0 - wx))[None, :]
            + source_flat[:, i01] * ((1.0 - wy) * wx)[None, :]
            + source_flat[:, i10] * (wy * (1.0 - wx))[None, :]
            + source_flat[:, i11] * (wy * wx)[None, :]
        )
    return result.reshape(*leading, *np.asarray(target_lat).shape)
