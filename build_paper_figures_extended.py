#!/usr/bin/env python3
"""Build extended probabilistic HeatCast/ENS paper figures and tables.

This script complements build_paper_figures_tables.py.  It never rewrites
figures 1-4 or tables 1-6.  Summary values are read from existing CSVs whenever
available.  Chunk-level products are computed only from saved incremental arrays
using the same fold loaders, calibrators, and accumulators used by the ENS
stacking evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import exceedance_eval as ee
from build_paper_figures_tables import (
    ENS_MODEL,
    HEATCAST_MODEL,
    REFERENCE_MODEL,
    STACK_MODEL,
    WINDOW_LABEL,
    ensure_matplotlib,
    f,
    fmt,
    model_label,
    read_csv,
    savefig,
    write_csv,
    write_markdown_table,
)
from ens_compare import chunk_map, fit_heatcast_c, load_chunk, resolve_ens_run_groups
from ens_heatcast_stack_opportunity import (
    MODEL_NAMES,
    SUBSETS,
    build_reservoir_for_fold,
    fit_opportunity_boundaries,
    fit_stacker_for_excluded_fold,
    paired_chunk,
)
from stitch_exceedance_folds import load_fold_inputs


EXTENDED_FIGURE_DIR = "paper_figures_extended"
SPATIAL_CSV = "figure_5_spatial_skill.csv"
RELIABILITY_CSV = "figure_6_reliability_decomposition.csv"
CASE_CSV = "figure_7_case_study_fields.csv"
PER_LEAD_CSV = "figure_8_per_lead_profile.csv"
DISCARD_CSV = "figure_9_discard_curve.csv"


def git_output(args: Sequence[str], root: Path) -> str:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def source_entry(path: Path, columns: Sequence[str], note: str) -> Dict[str, object]:
    return {"path": str(path), "columns": list(columns), "note": note}


def load_land_mask() -> np.ndarray:
    import cfm_mesh_train as cfm

    return np.asarray(cfm.load_conus_mask(cfm.Config).cpu().numpy() > 0.5, dtype=bool)


def load_lat_lon(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    from publication_analysis_utils import conus_lat_lon

    _, _, lat2d, lon2d = conus_lat_lon(shape)
    return np.asarray(lat2d, dtype=np.float32), np.asarray(lon2d, dtype=np.float32)


def land_to_grid(values: np.ndarray, land_mask: np.ndarray, fill: float = np.nan) -> np.ndarray:
    out = np.full(land_mask.shape, fill, dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size != int(np.sum(land_mask)):
        raise RuntimeError(f"Land vector length {arr.size} does not match land mask count {int(np.sum(land_mask))}.")
    out[land_mask] = arr
    return out


def finite_prob_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(np.asarray(arrays[0]).shape, dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(np.asarray(arr))
    return mask


def auc_per_cell(prob: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Vectorized per-cell ROC-AUC using average ranks for binary labels.

    prob and truth are arrays with shape (sample, cell).  NaNs are allowed in
    prob and mask that sample/cell pair.
    """
    p = np.asarray(prob, dtype=np.float32)
    y = np.asarray(truth, dtype=np.float32)
    if p.shape != y.shape:
        raise ValueError(f"prob/truth shape mismatch: {p.shape} vs {y.shape}")
    valid = np.isfinite(p) & np.isfinite(y)
    pos = valid & (y > 0.5)
    neg = valid & (y <= 0.5)
    n_pos = np.sum(pos, axis=0).astype(np.float64)
    n_neg = np.sum(neg, axis=0).astype(np.float64)
    filled = np.where(valid, p, np.inf)
    order = np.argsort(filled, axis=0, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    row_numbers = np.arange(1, p.shape[0] + 1, dtype=np.float32)[:, None]
    np.put_along_axis(ranks, order, row_numbers, axis=0)
    sum_pos_ranks = np.sum(np.where(pos, ranks, 0.0), axis=0, dtype=np.float64)
    denom = n_pos * n_neg
    auc = np.full(p.shape[1], np.nan, dtype=np.float32)
    ok = denom > 0
    auc[ok] = ((sum_pos_ranks[ok] - n_pos[ok] * (n_pos[ok] + 1.0) / 2.0) / denom[ok]).astype(np.float32)
    return auc


def murphy_decomposition(prob: np.ndarray, truth: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    p = np.asarray(prob, dtype=np.float64).reshape(-1)
    y = np.asarray(truth, dtype=np.float64).reshape(-1)
    valid = np.isfinite(p) & np.isfinite(y)
    p = np.clip(p[valid], 0.0, 1.0)
    y = y[valid]
    if p.size == 0:
        return {"brier": float("nan"), "reliability": float("nan"), "resolution": float("nan"), "uncertainty": float("nan"), "count": 0}
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.clip(np.digitize(p, bins, right=False) - 1, 0, int(n_bins) - 1)
    n = float(p.size)
    ybar = float(np.mean(y))
    reliability = 0.0
    resolution = 0.0
    for bin_idx in range(int(n_bins)):
        m = idx == bin_idx
        if not np.any(m):
            continue
        weight = float(np.sum(m)) / n
        pbar = float(np.mean(p[m]))
        obar = float(np.mean(y[m]))
        reliability += weight * (pbar - obar) ** 2
        resolution += weight * (obar - ybar) ** 2
    uncertainty = ybar * (1.0 - ybar)
    brier = float(np.mean((p - y) ** 2))
    return {
        "brier": brier,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "murphy_reconstructed_brier": float(reliability - resolution + uncertainty),
        "count": int(p.size),
    }


def bootstrap_ci_flags(year_values: np.ndarray, reps: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(year_values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2:
        nan = np.full(values.shape[-1], np.nan, dtype=np.float32)
        return nan, nan, np.zeros(values.shape[-1], dtype=bool)
    rng = np.random.default_rng(int(seed))
    n_years = values.shape[0]
    boot = np.empty((int(reps), values.shape[1]), dtype=np.float32)
    for i in range(int(reps)):
        idx = rng.integers(0, n_years, size=n_years)
        boot[i] = np.nanmean(values[idx], axis=0)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    flags = (lo > 0.0) | (hi < 0.0)
    return lo.astype(np.float32), hi.astype(np.float32), flags.astype(bool)


@dataclass
class ReliabilityBins:
    counts: np.ndarray
    prob_sum: np.ndarray
    truth_sum: np.ndarray
    brier_sum: np.ndarray

    @classmethod
    def create(cls, n_bins: int) -> "ReliabilityBins":
        return cls(
            counts=np.zeros(int(n_bins), dtype=np.float64),
            prob_sum=np.zeros(int(n_bins), dtype=np.float64),
            truth_sum=np.zeros(int(n_bins), dtype=np.float64),
            brier_sum=np.zeros(int(n_bins), dtype=np.float64),
        )

    def update(self, prob: np.ndarray, truth: np.ndarray) -> None:
        p = np.asarray(prob, dtype=np.float64).reshape(-1)
        y = np.asarray(truth, dtype=np.float64).reshape(-1)
        valid = np.isfinite(p) & np.isfinite(y)
        p = np.clip(p[valid], 0.0, 1.0)
        y = y[valid]
        if p.size == 0:
            return
        idx = np.clip((p * self.counts.size).astype(int), 0, self.counts.size - 1)
        self.counts += np.bincount(idx, minlength=self.counts.size)
        self.prob_sum += np.bincount(idx, weights=p, minlength=self.counts.size)
        self.truth_sum += np.bincount(idx, weights=y, minlength=self.counts.size)
        self.brier_sum += np.bincount(idx, weights=(p - y) ** 2, minlength=self.counts.size)

    def rows(self, model: str) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for idx in range(self.counts.size):
            count = int(self.counts[idx])
            rows.append({
                "model": model,
                "bin": idx,
                "count": count,
                "mean_forecast_probability": float(self.prob_sum[idx] / count) if count else float("nan"),
                "observed_frequency": float(self.truth_sum[idx] / count) if count else float("nan"),
            })
        return rows

    def decomposition(self) -> Dict[str, float]:
        total = float(np.sum(self.counts))
        if total <= 0:
            return {"brier": float("nan"), "reliability": float("nan"), "resolution": float("nan"), "uncertainty": float("nan"), "count": 0}
        ybar = float(np.sum(self.truth_sum) / total)
        reliability = 0.0
        resolution = 0.0
        for idx, count in enumerate(self.counts):
            if count <= 0:
                continue
            weight = float(count) / total
            pbar = float(self.prob_sum[idx] / count)
            obar = float(self.truth_sum[idx] / count)
            reliability += weight * (pbar - obar) ** 2
            resolution += weight * (obar - ybar) ** 2
        uncertainty = ybar * (1.0 - ybar)
        return {
            "brier": float(np.sum(self.brier_sum) / total),
            "reliability": float(reliability),
            "resolution": float(resolution),
            "uncertainty": float(uncertainty),
            "murphy_reconstructed_brier": float(reliability - resolution + uncertainty),
            "count": int(total),
        }


@dataclass
class ExtendedProducts:
    land_mask: Optional[np.ndarray] = None
    lat_land: Optional[np.ndarray] = None
    lon_land: Optional[np.ndarray] = None
    brier_sum: Dict[str, np.ndarray] = field(default_factory=dict)
    valid_count: Optional[np.ndarray] = None
    year_brier_sum: Dict[str, Dict[int, np.ndarray]] = field(default_factory=dict)
    year_valid_count: Dict[int, np.ndarray] = field(default_factory=dict)
    reliability: Dict[str, ReliabilityBins] = field(default_factory=dict)
    sample_prob_stack: List[np.ndarray] = field(default_factory=list)
    sample_truth: List[np.ndarray] = field(default_factory=list)
    sample_case_rows: List[Dict[str, object]] = field(default_factory=list)
    case_payloads: List[Dict[str, object]] = field(default_factory=list)
    discard_year_acc: Dict[str, Dict[int, ee.EvaluationAccumulator]] = field(default_factory=dict)
    discard_acc: Dict[str, ee.EvaluationAccumulator] = field(default_factory=dict)
    per_year_acc: Dict[Tuple[int, int], ee.EvaluationAccumulator] = field(default_factory=dict)
    rows_seen: int = 0
    samples_seen: int = 0


def make_fold_inputs(args: argparse.Namespace, window_leads: Sequence[int]) -> Dict[int, Dict[str, object]]:
    heatcast_runs = tuple(v.strip() for v in args.heatcast_runs.split(",") if v.strip())
    ens_runs = tuple(v.strip() for v in args.ens_runs.split(",") if v.strip())
    heatcast_root = Path(args.heatcast_root)
    ens_root = Path(args.ens_root)
    ens_groups = resolve_ens_run_groups(ens_runs, heatcast_runs, ens_root, window_leads)
    fold_inputs: Dict[int, Dict[str, object]] = {}
    for heat_name in heatcast_runs:
        heat_manifest, heat_calibration, heat_chunks = load_fold_inputs(heatcast_root, heat_name, window_leads)
        fold = int(heat_manifest["source_fold"])
        ens_sources = []
        for ens_name in ens_groups[fold]:
            ens_manifest, _, ens_chunks = load_fold_inputs(ens_root, ens_name, window_leads)
            if int(ens_manifest["source_fold"]) != fold:
                raise RuntimeError(f"Fold mismatch: HeatCast={fold}, ENS={ens_manifest['source_fold']}.")
            ens_sources.append((ens_name, ens_manifest, chunk_map(ens_chunks)))
        heat_map = chunk_map(heat_chunks)
        common = tuple(sorted(set(heat_map) & set().union(*(set(source[2]) for source in ens_sources))))
        if not common:
            raise RuntimeError(f"Fold {fold}: empty common init intersection.")
        fit_args = SimpleNamespace(
            calibration_steps=int(args.calibration_steps),
            calibration_lr=float(args.calibration_lr),
            calibration_l2=float(args.calibration_l2),
        )
        heat_c = fit_heatcast_c(heat_calibration, fit_args)
        fold_inputs[fold] = {
            "heat_name": heat_name,
            "manifest": heat_manifest,
            "heat_calibration": heat_calibration,
            "heat_map": heat_map,
            "ens_sources": ens_sources,
            "common": common,
            "heat_c": heat_c,
            "boundaries": fit_opportunity_boundaries(heat_calibration, heat_c),
        }
    return fold_inputs


def fit_stackers(fold_inputs: Mapping[int, Mapping[str, object]], args: argparse.Namespace) -> Dict[int, object]:
    reservoir_x: Dict[int, np.ndarray] = {}
    reservoir_y: Dict[int, np.ndarray] = {}
    for fold in sorted(fold_inputs):
        fold_id, x, y = build_reservoir_for_fold(
            fold,
            fold_inputs[fold],
            int(args.max_stack_samples_per_fold),
            int(args.seed),
            int(args.progress_every),
        )
        reservoir_x[fold_id] = x
        reservoir_y[fold_id] = y
    fit_args = SimpleNamespace(
        calibration_steps=int(args.calibration_steps),
        calibration_lr=float(args.calibration_lr),
        calibration_l2=float(args.calibration_l2),
    )
    return {
        fold: fit_stacker_for_excluded_fold(fold, reservoir_x, reservoir_y, fit_args)
        for fold in sorted(fold_inputs)
    }


def initialize_products(land_mask: np.ndarray, n_bins: int) -> ExtendedProducts:
    lat2d, lon2d = load_lat_lon(land_mask.shape)
    land_count = int(np.sum(land_mask))
    products = ExtendedProducts(
        land_mask=land_mask,
        lat_land=lat2d[land_mask],
        lon_land=lon2d[land_mask],
        valid_count=np.zeros(land_count, dtype=np.float64),
    )
    for model in (REFERENCE_MODEL, ENS_MODEL, HEATCAST_MODEL, STACK_MODEL):
        products.brier_sum[model] = np.zeros(land_count, dtype=np.float64)
        products.year_brier_sum[model] = {}
        products.reliability[model] = ReliabilityBins.create(n_bins)
    for retained in ("100", "50", "25", "10", "5", "1"):
        key = f"top_{retained}pct"
        products.discard_acc[key] = ee.EvaluationAccumulator(MODEL_NAMES, {})
        products.discard_year_acc[key] = {}
    return products


def update_extended_products(
    products: ExtendedProducts,
    fold: int,
    data: Mapping[str, object],
    stack_prob: np.ndarray,
    confidence_thresholds: Mapping[str, float],
) -> None:
    truth = np.asarray(data["truth"], dtype=np.float32)
    base = np.asarray(data["base"], dtype=np.float32)
    ens = np.asarray(data["ens_calibrated"], dtype=np.float32)
    heat = np.asarray(data["heatcast_C"], dtype=np.float32)
    stack = np.asarray(stack_prob, dtype=np.float32)
    year = int(data["year"])
    month = int(data["month"])
    forecasts = {
        REFERENCE_MODEL: base,
        ENS_MODEL: ens,
        HEATCAST_MODEL: heat,
        STACK_MODEL: stack,
    }
    valid = finite_prob_mask(truth, base, ens, heat, stack)
    products.rows_seen += int(np.sum(valid))
    products.samples_seen += 1
    products.valid_count[valid] += 1.0
    yvalid = truth[valid]
    for model, prob in forecasts.items():
        products.brier_sum[model][valid] += (np.clip(prob[valid], 0.0, 1.0) - yvalid) ** 2
        products.year_brier_sum[model].setdefault(year, np.zeros_like(products.brier_sum[model]))
        products.year_brier_sum[model][year][valid] += (np.clip(prob[valid], 0.0, 1.0) - yvalid) ** 2
        products.reliability[model].update(prob[valid], yvalid)
    products.year_valid_count.setdefault(year, np.zeros_like(products.valid_count))
    products.year_valid_count[year][valid] += 1.0
    products.sample_prob_stack.append(stack.astype(np.float32))
    products.sample_truth.append(truth.astype(np.float32))

    year_acc = products.per_year_acc.setdefault((fold, year), ee.EvaluationAccumulator(MODEL_NAMES, {}))
    for name, prob in forecasts.items():
        year_acc.update(name, prob, truth, valid, month)

    confidence = np.abs(heat - base)
    for retained, threshold in confidence_thresholds.items():
        key = f"top_{retained}pct"
        mask = valid if retained == "100" else (valid & (confidence >= threshold))
        acc = products.discard_acc[key]
        year_acc = products.discard_year_acc[key].setdefault(year, ee.EvaluationAccumulator(MODEL_NAMES, {}))
        for name, prob in forecasts.items():
            acc.update(name, prob, truth, mask, month)
            year_acc.update(name, prob, truth, mask, month)

    event_fraction = float(np.nanmean(truth[valid])) if np.any(valid) else float("nan")
    products.sample_case_rows.append({
        "fold": fold,
        "year": year,
        "month": month,
        "init_time_index": int(data["init_time_index"]),
        "target_center_time_index": int(data["target_center_time_index"]),
        "event_fraction": event_fraction,
        "stack_mean_probability": float(np.nanmean(stack[valid])) if np.any(valid) else float("nan"),
        "ens_mean_probability": float(np.nanmean(ens[valid])) if np.any(valid) else float("nan"),
        "payload_index": len(products.case_payloads),
    })
    if len(products.case_payloads) < 64:
        products.case_payloads.append({
            "truth": truth.astype(np.float32),
            "stack": stack.astype(np.float32),
            "ens": ens.astype(np.float32),
            "heatcast": heat.astype(np.float32),
            "meta": dict(products.sample_case_rows[-1]),
        })


def confidence_thresholds_from_calibration(fold_info: Mapping[str, object]) -> Dict[str, float]:
    heat_c = fold_info["heat_c"]
    calibration = fold_info["heat_calibration"]
    features = np.column_stack([calibration["init_margin"], calibration["forecast_margin"]]).astype(np.float32)
    heat_prob = heat_c.predict_features(features)
    confidence = np.abs(heat_prob - np.asarray(calibration["base_rate"], dtype=np.float32))
    valid = confidence[np.isfinite(confidence)]
    output = {"100": float("-inf")}
    for retained in (50, 25, 10, 5, 1):
        output[str(retained)] = float(np.nanquantile(valid, 1.0 - retained / 100.0))
    return output


def stream_chunk_products(args: argparse.Namespace, manifest_sources: Dict[str, object]) -> Optional[ExtendedProducts]:
    window_leads = ee.parse_int_list(args.window_leads)
    try:
        land_mask = load_land_mask()
        fold_inputs = make_fold_inputs(args, window_leads)
        stackers = fit_stackers(fold_inputs, args)
    except Exception as exc:
        manifest_sources["chunk_streaming_status"] = f"unavailable: {exc}"
        return None
    products = initialize_products(land_mask, int(args.reliability_bins))
    for fold in sorted(fold_inputs):
        thresholds = confidence_thresholds_from_calibration(fold_inputs[fold])
        common = tuple(fold_inputs[fold]["common"])
        for index, init_t in enumerate(common):
            data = paired_chunk(
                fold,
                int(init_t),
                fold_inputs[fold]["heat_map"][int(init_t)],
                fold_inputs[fold]["ens_sources"],
                fold_inputs[fold]["heat_c"],
            )
            stack_prob = stackers[fold].predict_features(np.asarray(data["features"], dtype=np.float32))
            update_extended_products(products, fold, data, stack_prob, thresholds)
            if (index + 1) % max(1, int(args.progress_every)) == 0:
                print(f"  extended stream fold {fold}: {index + 1}/{len(common)} paired inits")
    manifest_sources["chunk_streaming_status"] = "complete"
    manifest_sources["chunk_streaming_rows"] = products.rows_seen
    return products


def write_spatial_skill(products: ExtendedProducts, out_dir: Path, reps: int, seed: int, sources: Dict[str, object]) -> None:
    if products is None or products.land_mask is None:
        write_csv(out_dir / SPATIAL_CSV, [{"status": "not_available", "reason": sources.get("chunk_streaming_status", "")}])
        return
    valid_count = np.maximum(products.valid_count, 1.0)
    base_brier = products.brier_sum[REFERENCE_MODEL] / valid_count
    stack_brier = products.brier_sum[STACK_MODEL] / valid_count
    ens_brier = products.brier_sum[ENS_MODEL] / valid_count
    stack_bss = 1.0 - stack_brier / np.maximum(base_brier, 1e-12)
    ens_bss = 1.0 - ens_brier / np.maximum(base_brier, 1e-12)
    delta_bss = stack_bss - ens_bss

    years = sorted(products.year_valid_count)
    year_stack_bss = []
    year_delta_bss = []
    for year in years:
        yc = np.maximum(products.year_valid_count[year], 1.0)
        y_base = products.year_brier_sum[REFERENCE_MODEL][year] / yc
        y_stack = 1.0 - (products.year_brier_sum[STACK_MODEL][year] / yc) / np.maximum(y_base, 1e-12)
        y_ens = 1.0 - (products.year_brier_sum[ENS_MODEL][year] / yc) / np.maximum(y_base, 1e-12)
        year_stack_bss.append(y_stack.astype(np.float32))
        year_delta_bss.append((y_stack - y_ens).astype(np.float32))
    _, _, stack_flag = bootstrap_ci_flags(np.stack(year_stack_bss), reps, seed)
    _, _, delta_flag = bootstrap_ci_flags(np.stack(year_delta_bss), reps, seed + 17)

    with tempfile.TemporaryDirectory() as tmp:
        stack_path = Path(tmp) / "stack_prob.dat"
        truth_path = Path(tmp) / "truth.dat"
        n_samples = len(products.sample_prob_stack)
        land_count = int(np.sum(products.land_mask))
        if n_samples >= 2:
            stack_mm = np.memmap(stack_path, dtype="float32", mode="w+", shape=(n_samples, land_count))
            truth_mm = np.memmap(truth_path, dtype="float32", mode="w+", shape=(n_samples, land_count))
            for i, (prob, truth) in enumerate(zip(products.sample_prob_stack, products.sample_truth)):
                stack_mm[i] = prob
                truth_mm[i] = truth
            stack_auc = auc_per_cell(np.asarray(stack_mm), np.asarray(truth_mm))
        else:
            stack_auc = np.full(land_count, np.nan, dtype=np.float32)

    rows = []
    for idx in range(stack_bss.size):
        rows.append({
            "cell_index_land_order": idx,
            "lat": float(products.lat_land[idx]),
            "lon": float(products.lon_land[idx]),
            "stack_bss": float(stack_bss[idx]),
            "ens_bss": float(ens_bss[idx]),
            "delta_bss_stack_minus_ens": float(delta_bss[idx]),
            "stack_roc_auc": float(stack_auc[idx]),
            "stack_bss_ci_excludes_zero": bool(stack_flag[idx]),
            "delta_bss_ci_excludes_zero": bool(delta_flag[idx]),
            "valid_count": int(products.valid_count[idx]),
        })
    write_csv(out_dir / SPATIAL_CSV, rows)

    plt = ensure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5), constrained_layout=True)
    panels = [
        ("Stack BSS", stack_bss, "RdBu_r", -0.10, 0.10, stack_flag),
        ("Stack - ENS BSS", delta_bss, "RdBu_r", -0.06, 0.06, delta_flag),
        ("Stack ROC-AUC", stack_auc, "viridis", 0.5, 0.8, None),
    ]
    for ax, (title, values, cmap, vmin, vmax, flag) in zip(axes, panels):
        grid = land_to_grid(values, products.land_mask)
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_axis_off()
        if flag is not None and np.any(flag):
            flag_grid = land_to_grid(flag.astype(np.float32), products.land_mask, fill=0.0)
            ax.contour(flag_grid, levels=[0.5], colors="black", linewidths=0.3)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle("Spatial probabilistic W34 skill")
    savefig(fig, out_dir / "figure_5_spatial_skill")
    plt.close(fig)
    sources["figure_5_spatial_skill"] = source_entry(out_dir / SPATIAL_CSV, ["stack_bss", "delta_bss_stack_minus_ens", "stack_roc_auc", "ci_excludes_zero"], "computed from saved fold incremental chunks")


def write_reliability_decomposition(products: ExtendedProducts, stack_dir: Path, evidence_dir: Path, out_dir: Path, sources: Dict[str, object]) -> None:
    if products is None:
        write_csv(out_dir / RELIABILITY_CSV, [{"status": "not_available", "reason": sources.get("chunk_streaming_status", "")}])
        return
    rows: List[Dict[str, object]] = []
    for model in (ENS_MODEL, HEATCAST_MODEL, STACK_MODEL):
        dec = products.reliability[model].decomposition()
        rows.append({"model": model, **dec})
    for model, rel in products.reliability.items():
        for row in rel.rows(model):
            rows.append({"row_type": "reliability_bin", **row})
    write_csv(out_dir / RELIABILITY_CSV, rows)

    plt = ensure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    colors = {ENS_MODEL: "#4C78A8", HEATCAST_MODEL: "#F58518", STACK_MODEL: "#54A24B"}
    for model in (ENS_MODEL, HEATCAST_MODEL, STACK_MODEL):
        rel_rows = products.reliability[model].rows(model)
        x = np.array([r["mean_forecast_probability"] for r in rel_rows], dtype=float)
        y = np.array([r["observed_frequency"] for r in rel_rows], dtype=float)
        n = np.array([r["count"] for r in rel_rows], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & (n > 0)
        axes[0].plot(x[valid], y[valid], marker="o", color=colors[model], label=model_label(model))
        if np.sum(n) > 0:
            axes[0].bar(x[valid], n[valid] / np.sum(n), width=0.025, color=colors[model], alpha=0.12)
    axes[0].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Forecast probability")
    axes[0].set_ylabel("Observed frequency")
    axes[0].set_title("Reliability and sharpness")
    axes[0].legend(frameon=False)

    dec_rows = [row for row in rows if row.get("model") in (ENS_MODEL, HEATCAST_MODEL, STACK_MODEL) and row.get("row_type", "") == ""]
    models = [row["model"] for row in dec_rows]
    x = np.arange(len(models))
    width = 0.23
    axes[1].bar(x - width, [float(row["reliability"]) for row in dec_rows], width=width, label="Reliability")
    axes[1].bar(x, [float(row["resolution"]) for row in dec_rows], width=width, label="Resolution")
    axes[1].bar(x + width, [float(row["uncertainty"]) for row in dec_rows], width=width, label="Uncertainty")
    axes[1].set_xticks(x, [model_label(model) for model in models], rotation=20, ha="right")
    axes[1].set_ylabel("Brier decomposition term")
    axes[1].set_title("Murphy decomposition")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, out_dir / "figure_6_reliability_decomposition")
    plt.close(fig)
    sources["figure_6_reliability_decomposition"] = source_entry(out_dir / RELIABILITY_CSV, ["mean_forecast_probability", "observed_frequency", "reliability", "resolution", "uncertainty"], "computed from saved fold incremental chunks with Murphy decomposition")


def write_case_studies(products: ExtendedProducts, out_dir: Path, sources: Dict[str, object], n_cases: int = 3) -> None:
    if products is None or products.land_mask is None:
        write_csv(out_dir / CASE_CSV, [{"status": "not_available", "reason": sources.get("chunk_streaming_status", "")}])
        return
    top_rows = sorted(products.sample_case_rows, key=lambda row: f(row["event_fraction"]), reverse=True)[:int(n_cases)]
    if not top_rows:
        write_csv(out_dir / CASE_CSV, [{"status": "not_available", "reason": "no case payloads"}])
        return
    write_csv(out_dir / CASE_CSV, top_rows)
    plt = ensure_matplotlib()
    fig, axes = plt.subplots(len(top_rows), 3, figsize=(10.5, 3.0 * len(top_rows)), squeeze=False, constrained_layout=True)
    for row_idx, row in enumerate(top_rows):
        payload = products.case_payloads[int(row["payload_index"])]
        maps = [
            ("Observed exceedance", payload["truth"], "Greys", 0.0, 1.0),
            ("HeatCast+ENS probability", payload["stack"], "magma", 0.0, 0.5),
            ("ENS probability", payload["ens"], "magma", 0.0, 0.5),
        ]
        for col_idx, (title, values, cmap, vmin, vmax) in enumerate(maps):
            ax = axes[row_idx][col_idx]
            im = ax.imshow(land_to_grid(values, products.land_mask), cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_axis_off()
            ax.set_title(title if row_idx == 0 else "")
            if col_idx == 0:
                ax.text(
                    0.0,
                    1.02,
                    f"fold {row['fold']}, year {row['year']}, month {row['month']}, event frac={float(row['event_fraction']):.3f}",
                    transform=ax.transAxes,
                    fontsize=8,
                )
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    fig.suptitle("W34 probabilistic case studies")
    savefig(fig, out_dir / "figure_7_case_studies")
    plt.close(fig)
    sources["figure_7_case_studies"] = source_entry(out_dir / CASE_CSV, ["event_fraction", "stack_mean_probability", "ens_mean_probability"], "case fields computed from saved fold incremental chunks")


def write_per_lead_profile(args: argparse.Namespace, out_dir: Path, sources: Dict[str, object]) -> None:
    candidates = [
        Path(args.per_lead_csv) if args.per_lead_csv else None,
        Path(args.stack_dir) / "per_lead_profile.csv",
        Path(args.stack_dir) / "figure_8_per_lead_profile.csv",
    ]
    source = next((path for path in candidates if path is not None and path.exists()), None)
    if source is None:
        rows = [{
            "status": "not_available",
            "reason": "No per-lead stitched CSV was found. Saved window chunks do not contain individual lead predictions, so this panel requires an upstream per-lead export.",
        }]
        write_csv(out_dir / PER_LEAD_CSV, rows)
        return
    rows = read_csv(source)
    write_csv(out_dir / PER_LEAD_CSV, rows)
    plt = ensure_matplotlib()
    by_model: Dict[str, List[Mapping[str, str]]] = {}
    for row in rows:
        by_model.setdefault(row.get("model", ""), []).append(row)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), constrained_layout=True)
    for model, model_rows in by_model.items():
        model_rows = sorted(model_rows, key=lambda r: f(r.get("lead", "nan")))
        lead = [f(r.get("lead")) for r in model_rows]
        bss = [f(r.get("bss")) for r in model_rows]
        tac = [f(r.get("tac")) for r in model_rows]
        axes[0].plot(lead, bss, marker="o", label=model_label(model))
        axes[1].plot(lead, tac, marker="o", label=model_label(model))
    axes[0].set_ylabel("BSS")
    axes[1].set_ylabel("TAC")
    for ax in axes:
        ax.set_xlabel("Lead day")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle("Per-lead skill across W34 tube")
    savefig(fig, out_dir / "figure_8_per_lead_profile")
    plt.close(fig)
    sources["figure_8_per_lead_profile"] = source_entry(source, ["lead", "model", "bss", "tac"], "read from upstream per-lead stitched CSV")


def write_discard_curve(products: ExtendedProducts, out_dir: Path, sources: Dict[str, object], reps: int, seed: int) -> None:
    if products is None:
        write_csv(out_dir / DISCARD_CSV, [{"status": "not_available", "reason": sources.get("chunk_streaming_status", "")}])
        return
    rows: List[Dict[str, object]] = []
    for key in sorted(products.discard_acc, key=lambda value: int(value.split("_")[1].replace("pct", "")), reverse=True):
        retained = int(key.split("_")[1].replace("pct", ""))
        summary = {row["model"]: row for row in products.discard_acc[key].summary_rows(REFERENCE_MODEL)}
        stack_bss = float(summary[STACK_MODEL]["bss_vs_monthly_climo"])
        rows.append({
            "retained_fraction": retained / 100.0,
            "retained_percent": retained,
            "model": STACK_MODEL,
            "bss": stack_bss,
            "roc_auc": float(summary[STACK_MODEL]["roc_auc"]),
            "ece": float(summary[STACK_MODEL]["ece"]),
            "valid_count": int(summary[STACK_MODEL]["valid_count"]),
        })
    write_csv(out_dir / DISCARD_CSV, rows)
    plt = ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = [float(row["retained_percent"]) for row in rows]
    y = [float(row["bss"]) for row in rows]
    ax.plot(x, y, marker="o", color="#54A24B")
    ax.invert_xaxis()
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Retained highest-confidence cells (%)")
    ax.set_ylabel("Stack BSS")
    ax.set_title("Forecasts-of-opportunity discard curve")
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, out_dir / "figure_9_opportunity_discard_curve")
    plt.close(fig)
    sources["figure_9_opportunity_discard_curve"] = source_entry(out_dir / DISCARD_CSV, ["retained_fraction", "bss", "roc_auc", "ece"], "computed from stack probabilities and validation-year confidence thresholds")


def write_table_7(stack_dir: Path, fig_dir: Path, table_dir: Path, sources: Dict[str, object]) -> None:
    score_rows = [row for row in read_csv(stack_dir / "heatcast_ens_stack_head_to_head.csv") if row.get("section") == "score"]
    by_model = {row["model"]: row for row in score_rows}
    dec_rows = read_csv(fig_dir / RELIABILITY_CSV, required=False)
    dec_by_model = {row.get("model"): row for row in dec_rows if row.get("model") in MODEL_NAMES and not row.get("row_type")}
    output = []
    for model in (REFERENCE_MODEL, ENS_MODEL, HEATCAST_MODEL, STACK_MODEL):
        row = by_model.get(model, {})
        dec = dec_by_model.get(model, {})
        output.append({
            "model": model_label(model),
            "bss": fmt(row.get("bss_vs_monthly_climo")),
            "roc_auc": fmt(row.get("roc_auc"), signed=False),
            "reliability_slope": fmt(row.get("reliability_slope"), signed=False),
            "ece": fmt(row.get("ece"), signed=False),
            "resolution": fmt(dec.get("resolution"), signed=False),
            "sharpness_proxy_uncertainty": fmt(dec.get("uncertainty"), signed=False),
            "brier": fmt(row.get("brier"), signed=False),
        })
    write_csv(table_dir / "table_7_stack_ablation_probability.csv", output)
    write_markdown_table(table_dir / "table_7_stack_ablation_probability.md", output)
    sources["table_7_stack_ablation_probability"] = [
        source_entry(stack_dir / "heatcast_ens_stack_head_to_head.csv", ["bss_vs_monthly_climo", "roc_auc", "reliability_slope", "ece", "brier"], "read from existing head-to-head score rows"),
        source_entry(fig_dir / RELIABILITY_CSV, ["resolution", "uncertainty"], "Murphy decomposition emitted by figure 6"),
    ]


def year_to_fold_from_head_to_head(stack_dir: Path) -> Dict[int, int]:
    rows = read_csv(stack_dir / "heatcast_ens_stack_head_to_head.csv")
    mapping: Dict[int, int] = {}
    for row in rows:
        if row.get("section") != "coverage":
            continue
        fold = int(f(row.get("fold")))
        for year in str(row.get("intersection_years", "")).split():
            mapping[int(year)] = fold
    return mapping


def write_table_8(stack_dir: Path, table_dir: Path, sources: Dict[str, object]) -> None:
    per_year_candidates = [
        stack_dir / "ens_heatcast_per_year.csv",
        stack_dir / "robustness_by_year.csv",
    ]
    source = next((path for path in per_year_candidates if path.exists()), None)
    if source is None:
        write_csv(table_dir / "table_8_per_year_head_to_head.csv", [{"status": "not_available", "reason": "No per-year CSV found."}])
        return
    rows = read_csv(source)
    fold_map = year_to_fold_from_head_to_head(stack_dir)
    by_year_model: Dict[Tuple[int, str], Mapping[str, str]] = {}
    for row in rows:
        year = int(f(row.get("year", row.get("fold", "nan"))))
        model = str(row.get("model", ""))
        by_year_model[(year, model)] = row
    output = []
    for year in sorted({year for year, _ in by_year_model}):
        ens = by_year_model.get((year, ENS_MODEL))
        stack = by_year_model.get((year, STACK_MODEL))
        if not ens or not stack:
            continue
        output.append({
            "year": year,
            "fold": fold_map.get(year, ""),
            "stack_bss": fmt(stack.get("bss_vs_monthly_climo")),
            "ens_bss": fmt(ens.get("bss_vs_monthly_climo")),
            "delta_bss_stack_minus_ens": fmt(f(stack.get("bss_vs_monthly_climo")) - f(ens.get("bss_vs_monthly_climo"))),
            "stack_auc": fmt(stack.get("roc_auc"), signed=False),
            "ens_auc": fmt(ens.get("roc_auc"), signed=False),
        })
    fold2_removed = [row for row in output if str(row.get("fold")) != "2"]
    if fold2_removed:
        output.append({
            "year": "fold2_removed",
            "fold": "",
            "delta_bss_stack_minus_ens": fmt(np.nanmean([f(row["delta_bss_stack_minus_ens"]) for row in fold2_removed])),
            "stack_bss": "",
            "ens_bss": "",
            "stack_auc": "",
            "ens_auc": "",
        })
    write_csv(table_dir / "table_8_per_year_head_to_head.csv", output)
    write_markdown_table(table_dir / "table_8_per_year_head_to_head.md", output)
    sources["table_8_per_year_head_to_head"] = source_entry(source, ["year", "model", "bss_vs_monthly_climo", "roc_auc"], "read from existing per-year/robustness CSV and coverage fold map")


def write_table_9(root: Path, stack_dir: Path, table_dir: Path, sources: Dict[str, object]) -> None:
    count = "4.6M"
    logs = sorted(root.glob("*.log"))
    output = [
        {
            "system": "HeatCast",
            "parameter_count": count,
            "training_gpu_hours": "",
            "inference_cost_per_forecast": "",
            "cost_source": "Parameter count from model logs; GPU-hour/inference fields require retained Slurm accounting or run manifest.",
            "notes": "Leave blank rather than fabricate if accounting logs are unavailable.",
        },
        {
            "system": "ECMWF S2S ENS",
            "parameter_count": "",
            "training_gpu_hours": "",
            "inference_cost_per_forecast": "",
            "cost_source": "",
            "notes": "Order-of-magnitude operational cost requires a cited external source; intentionally blank until sourced.",
        },
    ]
    write_csv(table_dir / "table_9_computational_cost_comparison.csv", output)
    write_markdown_table(table_dir / "table_9_computational_cost_comparison.md", output)
    sources["table_9_computational_cost_comparison"] = {
        "heatcast_parameter_count": "training log text, approximate recurring parameter count",
        "logs_found": [str(path) for path in logs[:20]],
        "ens_cost": "blank until a citable source is supplied",
    }


def update_manifest(out_dir: Path, sources: Mapping[str, object], root: Path) -> None:
    manifest_path = out_dir / "reproducibility_manifest.json"
    existing: Dict[str, object] = {}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing.update({
        "extended_created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output(["rev-parse", "HEAD"], root),
        "extended_outputs": sources,
    })
    manifest_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def copy_inputs_to_repro(root: Path, out_dir: Path, args: argparse.Namespace) -> None:
    repro = out_dir / "reproducibility"
    repro.mkdir(parents=True, exist_ok=True)
    for source in (
        Path(args.stack_dir) / "heatcast_ens_stack_head_to_head.csv",
        Path(args.stack_dir) / "opportunity_pair_summary.csv",
        Path(args.stack_dir) / "opportunity_pair_bootstrap.csv",
        Path(args.stack_dir) / "driver_pair_summary.csv",
        Path(args.stack_dir) / "driver_pair_parent_bootstrap.csv",
        Path(args.evidence_dir) / "operational_block.csv",
        Path(args.evidence_dir) / "mechanism_block.csv",
        Path(args.opportunity_dir) / "driver_opportunity_summary.csv",
        Path(args.opportunity_dir) / "driver_interaction_paired_bootstrap.csv",
    ):
        if source.exists():
            shutil.copy2(source, repro / source.name)
    for script in ("submit_paper_figures_extended.slurm", "submit_paper_figures_tables.slurm"):
        path = root / script
        if path.exists():
            shutil.copy2(path, repro / script)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack_dir", default=f"ens_heatcast_stack_opportunity/{WINDOW_LABEL}")
    parser.add_argument("--evidence_dir", default=f"paper_evidence_blocks/{WINDOW_LABEL}")
    parser.add_argument("--opportunity_dir", default=f"exceedance_eval_incremental/opportunity_{WINDOW_LABEL}")
    parser.add_argument("--output_dir", default=f"{EXTENDED_FIGURE_DIR}/{WINDOW_LABEL}")
    parser.add_argument("--heatcast_root", default="exceedance_eval_incremental")
    parser.add_argument("--ens_root", default="ens_exceedance_incremental")
    parser.add_argument(
        "--heatcast_runs",
        default="cvfold0_w34_dist_v1,cvfold1_w34_dist_v1,cvfold2_w34_dist_v1,cvfold3_w34_dist_v1,cvfold4_w34_dist_v1",
    )
    parser.add_argument(
        "--ens_runs",
        default="cvfold{F}_ens_w34,cvfold{F}_ens_w34_rt2024",
    )
    parser.add_argument("--window_leads", default="15,16,17,18,19,20,21,22,23,24,25,26,27,28")
    parser.add_argument("--per_lead_csv", default="")
    parser.add_argument("--calibration_steps", type=int, default=200)
    parser.add_argument("--calibration_lr", type=float, default=0.1)
    parser.add_argument("--calibration_l2", type=float, default=1e-4)
    parser.add_argument("--max_stack_samples_per_fold", type=int, default=300000)
    parser.add_argument("--reliability_bins", type=int, default=10)
    parser.add_argument("--spatial_bootstrap_reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--skip_chunk_products", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    for path in (fig_dir, table_dir):
        path.mkdir(parents=True, exist_ok=True)
    sources: Dict[str, object] = {
        "headline": source_entry(Path(args.stack_dir) / "heatcast_ens_stack_head_to_head.csv", ["section", "model", "bss_vs_monthly_climo", "roc_auc"], "existing stack headline CSV"),
        "operational": source_entry(Path(args.evidence_dir) / "operational_block.csv", ["model", "bss", "roc_auc", "reliability_slope", "ece"], "existing paper evidence operational block"),
    }

    products = None if args.skip_chunk_products else stream_chunk_products(args, sources)
    write_spatial_skill(products, fig_dir, int(args.spatial_bootstrap_reps), int(args.seed), sources)
    write_reliability_decomposition(products, Path(args.stack_dir), Path(args.evidence_dir), fig_dir, sources)
    write_case_studies(products, fig_dir, sources)
    write_per_lead_profile(args, fig_dir, sources)
    write_discard_curve(products, fig_dir, sources, int(args.spatial_bootstrap_reps), int(args.seed))
    write_table_7(Path(args.stack_dir), fig_dir, table_dir, sources)
    write_table_8(Path(args.stack_dir), table_dir, sources)
    write_table_9(root, Path(args.stack_dir), table_dir, sources)
    copy_inputs_to_repro(root, out_dir, args)
    update_manifest(out_dir, sources, root)

    print("Extended paper figures/tables complete")
    print(f"  output_dir={out_dir}")
    print(f"  figures={fig_dir}")
    print(f"  tables={table_dir}")
    print(f"  manifest={out_dir / 'reproducibility_manifest.json'}")


if __name__ == "__main__":
    main()
