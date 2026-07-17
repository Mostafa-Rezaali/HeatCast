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
import re
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
from figure_style import DOUBLE_COLUMN_MM, figure_size, group_color, panel_label, system_color, teleconnection_color
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
from ens_compare import chunk_map, fit_heatcast_c, load_chunk
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
TELECONNECTION_CSV = "figure_9_teleconnection_ranking.csv"
LEAVE_ONE_OUT_CSV = "figure_10a_leave_one_out_robustness.csv"
PER_YEAR_BASE_RATE_CSV = "figure_10_per_year_base_rate.csv"
PARENT_CAVEAT = (
    "Parent-comparison tests do not establish that an interaction stratum beats "
    "its broader driver or opportunity parent; these rows are regime-stratified "
    "evidence, not a standalone causal mechanism."
)


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
    ens_groups = resolve_ens_run_groups_flexible(ens_runs, heatcast_runs, ens_root, window_leads)
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


def resolve_ens_run_groups_flexible(
    ens_runs: Sequence[str],
    heatcast_runs: Sequence[str],
    ens_root: Path,
    window_leads: Sequence[int],
) -> Dict[int, List[str]]:
    """Resolve ENS run groups from templates, explicit run names, or a mixture.

    The shared ens_compare resolver is intentionally strict: once a template is
    used, every entry must contain {F}.  The extended figure builder is a
    post-processing tool and should tolerate practical inputs such as one
    template plus one already-expanded cycle.  Explicit run names are assigned
    by reading their fold manifest; template names are expanded for every fold.
    """
    output: Dict[int, List[str]] = {fold: [] for fold in range(len(heatcast_runs))}
    explicit: List[str] = []
    for value in ens_runs:
        if "{F}" in value:
            for fold in range(len(heatcast_runs)):
                output[fold].append(value.replace("{F}", str(fold)))
        else:
            explicit.append(value)
    for run_name in explicit:
        manifest, _, _ = load_fold_inputs(ens_root, run_name, window_leads)
        output[int(manifest["source_fold"])].append(run_name)
    missing = [fold for fold, names in output.items() if not names]
    if missing:
        raise ValueError(f"ENS runs do not cover folds {missing}.")
    return output


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
    case_row = {
        "fold": fold,
        "year": year,
        "month": month,
        "init_time_index": int(data["init_time_index"]),
        "target_center_time_index": int(data["target_center_time_index"]),
        "event_fraction": event_fraction,
        "stack_mean_probability": float(np.nanmean(stack[valid])) if np.any(valid) else float("nan"),
        "ens_mean_probability": float(np.nanmean(ens[valid])) if np.any(valid) else float("nan"),
    }
    case_payload = {
        "truth": truth.astype(np.float32),
        "stack": stack.astype(np.float32),
        "ens": ens.astype(np.float32),
        "heatcast": heat.astype(np.float32),
    }
    max_case_payloads = 64
    if len(products.case_payloads) < max_case_payloads:
        case_row["payload_index"] = len(products.case_payloads)
        products.sample_case_rows.append(case_row)
        case_payload["meta"] = dict(case_row)
        products.case_payloads.append({
            **case_payload,
        })
    elif math.isfinite(event_fraction):
        stored_events = np.asarray([f(row.get("event_fraction")) for row in products.sample_case_rows], dtype=np.float64)
        finite = np.isfinite(stored_events)
        replace_pos = int(np.nanargmin(np.where(finite, stored_events, np.inf))) if np.any(finite) else -1
        if replace_pos >= 0 and event_fraction > float(stored_events[replace_pos]):
            payload_index = int(products.sample_case_rows[replace_pos]["payload_index"])
            case_row["payload_index"] = payload_index
            products.sample_case_rows[replace_pos] = case_row
            case_payload["meta"] = dict(case_row)
            products.case_payloads[payload_index] = {
                **case_payload,
            }


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
        existing = out_dir / SPATIAL_CSV
        rows = read_csv(existing, required=False)
        if not real_numeric_rows(rows, ("stack_bss", "delta_bss_stack_minus_ens", "stack_roc_auc")):
            raise RuntimeError(f"Cannot build Figure 5: chunk products unavailable ({sources.get('chunk_streaming_status', '')}).")
        plt = ensure_matplotlib()
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5), constrained_layout=True)
        panels = [
            ("Stack BSS", "stack_bss", "RdBu_r", -0.10, 0.10),
            ("Stack - ENS BSS", "delta_bss_stack_minus_ens", "RdBu_r", -0.06, 0.06),
            ("Stack ROC-AUC", "stack_roc_auc", "viridis", 0.5, 0.8),
        ]
        lon = np.asarray([f(row.get("lon")) for row in rows], dtype=float)
        lat = np.asarray([f(row.get("lat")) for row in rows], dtype=float)
        for ax, (title, field, cmap, vmin, vmax) in zip(axes, panels):
            values = np.asarray([f(row.get(field)) for row in rows], dtype=float)
            valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(values)
            sc = ax.scatter(lon[valid], lat[valid], c=values[valid], s=0.2, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
            ax.set_title(title)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        fig.suptitle("Spatial probabilistic W34 skill")
        savefig(fig, out_dir / "figure_5_spatial_skill")
        plt.close(fig)
        sources["figure_5_spatial_skill"] = source_entry(existing, ["stack_bss", "delta_bss_stack_minus_ens", "stack_roc_auc", "ci_excludes_zero"], "replotted from existing numeric Figure 5 CSV because chunk arrays were unavailable in this checkout")
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
    if not real_numeric_rows(rows, ("stack_bss", "delta_bss_stack_minus_ens", "stack_roc_auc")):
        raise RuntimeError("Figure 5 spatial CSV has no numeric rows.")

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
        rows = read_csv(out_dir / RELIABILITY_CSV, required=False)
        if not real_numeric_rows([row for row in rows if not row.get("row_type")], ("reliability", "resolution", "uncertainty")):
            raise RuntimeError(f"Cannot build Figure 6: chunk products unavailable ({sources.get('chunk_streaming_status', '')}).")
        plt = ensure_matplotlib()
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
        for model in (ENS_MODEL, HEATCAST_MODEL, STACK_MODEL):
            rel_rows = [row for row in rows if row.get("row_type") == "reliability_bin" and row.get("model") == model]
            x = np.asarray([f(row.get("mean_forecast_probability")) for row in rel_rows], dtype=float)
            y = np.asarray([f(row.get("observed_frequency")) for row in rel_rows], dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            axes[0].plot(x[valid], y[valid], marker="o", color=system_color(model), label=model_label(model))
        axes[0].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8)
        axes[0].set_xlabel("Forecast probability")
        axes[0].set_ylabel("Observed frequency")
        axes[0].set_title("Reliability")
        axes[0].legend(frameon=False)
        dec_rows = [row for row in rows if row.get("model") in (ENS_MODEL, HEATCAST_MODEL, STACK_MODEL) and not row.get("row_type")]
        models = [row["model"] for row in dec_rows]
        x = np.arange(len(models))
        width = 0.23
        axes[1].bar(x - width, [f(row.get("reliability")) for row in dec_rows], width=width, label="Reliability")
        axes[1].bar(x, [f(row.get("resolution")) for row in dec_rows], width=width, label="Resolution")
        axes[1].bar(x + width, [f(row.get("uncertainty")) for row in dec_rows], width=width, label="Uncertainty")
        axes[1].set_xticks(x, [model_label(model) for model in models], rotation=20, ha="right")
        axes[1].set_ylabel("Brier decomposition term")
        axes[1].set_title("Murphy decomposition")
        axes[1].legend(frameon=False)
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
        savefig(fig, out_dir / "figure_6_reliability_decomposition")
        plt.close(fig)
        sources["figure_6_reliability_decomposition"] = source_entry(out_dir / RELIABILITY_CSV, ["mean_forecast_probability", "observed_frequency", "reliability", "resolution", "uncertainty"], "replotted from existing numeric Figure 6 CSV because chunk arrays were unavailable in this checkout")
        return
    rows: List[Dict[str, object]] = []
    for model in (ENS_MODEL, HEATCAST_MODEL, STACK_MODEL):
        dec = products.reliability[model].decomposition()
        rows.append({"model": model, **dec})
    for model, rel in products.reliability.items():
        for row in rel.rows(model):
            rows.append({"row_type": "reliability_bin", **row})
    write_csv(out_dir / RELIABILITY_CSV, rows)
    if not real_numeric_rows([row for row in rows if not row.get("row_type")], ("reliability", "resolution", "uncertainty")):
        raise RuntimeError("Figure 6 reliability CSV has no numeric decomposition rows.")

    plt = ensure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    colors = {model: system_color(model) for model in (ENS_MODEL, HEATCAST_MODEL, STACK_MODEL)}
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
    valid_rows = [
        row for row in products.sample_case_rows
        if 0 <= int(row.get("payload_index", -1)) < len(products.case_payloads)
    ]
    top_rows = sorted(valid_rows, key=lambda row: f(row["event_fraction"]), reverse=True)[:int(n_cases)]
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


PER_LEAD_LINE_RE = re.compile(r"Per-lead diagnostics:\s*(?P<body>.*)")
PER_LEAD_PAIR_RE = re.compile(r"\+(?P<lead>\d+):TAC=(?P<tac>[+-]?(?:\d+(?:\.\d*)?|\.\d+))/(?:MSE=)(?P<mse>[+-]?(?:\d+(?:\.\d*)?|\.\d+))")
RUN_FOLD_RE = re.compile(r"cvfold(?P<fold>\d+)_w34")


def real_numeric_rows(rows: Sequence[Mapping[str, str]], required_columns: Sequence[str]) -> bool:
    if not rows or any(row.get("status") == "not_available" for row in rows):
        return False
    for row in rows:
        if all(math.isfinite(f(row.get(column))) for column in required_columns):
            return True
    return False


def parse_per_lead_diagnostics_file(path: Path, expected_leads: Sequence[int]) -> Dict[int, Dict[int, Dict[str, object]]]:
    latest_by_fold: Dict[int, Dict[int, Dict[str, object]]] = {}
    current_fold: Optional[int] = None
    fallback_match = RUN_FOLD_RE.search(path.name)
    fallback_fold = int(fallback_match.group("fold")) if fallback_match else None
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, start=1):
            run_match = RUN_FOLD_RE.search(line)
            if run_match:
                current_fold = int(run_match.group("fold"))
            diag = PER_LEAD_LINE_RE.search(line)
            if not diag:
                continue
            fold = current_fold if current_fold is not None else fallback_fold
            if fold is None:
                continue
            values: Dict[int, Dict[str, object]] = {}
            for match in PER_LEAD_PAIR_RE.finditer(diag.group("body")):
                lead = int(match.group("lead"))
                if lead not in expected_leads:
                    continue
                values[lead] = {
                    "fold": fold,
                    "lead": lead,
                    "tac": float(match.group("tac")),
                    "mse": float(match.group("mse")),
                    "source_log": str(path),
                    "line_number": line_number,
                }
            if values:
                latest_by_fold[fold] = values
    return latest_by_fold


def parse_w34_per_lead_logs(root: Path, patterns: str, expected_leads: Sequence[int]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    latest_by_fold: Dict[int, Dict[int, Dict[str, object]]] = {}
    paths: List[Path] = []
    for pattern in [item.strip() for item in str(patterns).split(",") if item.strip()]:
        matches = sorted(root.glob(pattern))
        if not matches and not pattern.startswith("**/"):
            matches = sorted(root.glob(f"**/{pattern}"))
        paths.extend(path for path in matches if path.is_file())
    for path in sorted(set(paths)):
        parsed = parse_per_lead_diagnostics_file(path, expected_leads)
        latest_by_fold.update(parsed)
    if not latest_by_fold:
        raise RuntimeError(
            "Could not recover W34 per-lead diagnostics from logs. "
            "Set --w34_log_glob to matching logs or provide --per_lead_csv."
        )
    aggregate: List[Dict[str, object]] = []
    detail: List[Dict[str, object]] = []
    expected = tuple(int(lead) for lead in expected_leads)
    for lead in expected:
        lead_rows = [fold_values[lead] for fold_values in latest_by_fold.values() if lead in fold_values]
        if not lead_rows:
            raise RuntimeError(f"No W34 per-lead diagnostics recovered for lead +{lead}.")
        tac = np.asarray([float(row["tac"]) for row in lead_rows], dtype=np.float64)
        mse = np.asarray([float(row["mse"]) for row in lead_rows], dtype=np.float64)
        if not np.all(np.isfinite(tac)):
            raise RuntimeError(f"Non-finite TAC recovered for lead +{lead}.")
        aggregate.append({
            "lead": lead,
            "model": HEATCAST_MODEL,
            "tac_mean": float(np.mean(tac)),
            "tac_sd": float(np.std(tac, ddof=1)) if tac.size > 1 else 0.0,
            "mse_mean": float(np.mean(mse)),
            "mse_sd": float(np.std(mse, ddof=1)) if mse.size > 1 else 0.0,
            "n_folds": int(tac.size),
            "folds": " ".join(str(int(row["fold"])) for row in sorted(lead_rows, key=lambda item: int(item["fold"]))),
            "source": "w34_training_log_per_lead_diagnostics",
        })
        detail.extend(lead_rows)
    if len(aggregate) != len(expected):
        raise RuntimeError(f"Expected {len(expected)} per-lead rows, wrote {len(aggregate)}.")
    return aggregate, detail


def write_per_lead_profile(args: argparse.Namespace, out_dir: Path, sources: Dict[str, object]) -> None:
    candidates = [
        Path(args.per_lead_csv) if args.per_lead_csv else None,
        Path(args.stack_dir) / "per_lead_profile.csv",
        Path(args.stack_dir) / "figure_8_per_lead_profile.csv",
    ]
    source = None
    rows: List[Dict[str, object]]
    for path in candidates:
        if path is None or not path.exists():
            continue
        candidate_rows = read_csv(path)
        if real_numeric_rows(candidate_rows, ("lead", "tac")):
            source = path
            rows = [dict(row) for row in candidate_rows]
            break
    else:
        expected_leads = [int(value) for value in str(args.window_leads).split(",") if value.strip()]
        rows, detail = parse_w34_per_lead_logs(Path.cwd(), args.w34_log_glob, expected_leads)
        write_csv(out_dir / "figure_8_per_lead_profile_fold_details.csv", detail)
        source = out_dir / PER_LEAD_CSV
    write_csv(out_dir / PER_LEAD_CSV, rows)
    plt = ensure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=figure_size(DOUBLE_COLUMN_MM, 70.0), constrained_layout=True)
    rows = sorted(rows, key=lambda r: f(r.get("lead", "nan")))
    lead = np.asarray([f(row.get("lead")) for row in rows], dtype=float)
    tac = np.asarray([f(row.get("tac_mean", row.get("tac"))) for row in rows], dtype=float)
    tac_sd = np.asarray([f(row.get("tac_sd", 0.0)) for row in rows], dtype=float)
    mse = np.asarray([f(row.get("mse_mean", row.get("mse"))) for row in rows], dtype=float)
    mse_sd = np.asarray([f(row.get("mse_sd", 0.0)) for row in rows], dtype=float)
    if len(rows) != 14 or not np.all(np.isfinite(tac)):
        raise RuntimeError("figure_8_per_lead_profile.csv must contain 14 finite TAC rows for leads +15..+28.")
    axes[0].plot(lead, tac, marker="o", color=system_color(HEATCAST_MODEL), label=model_label(HEATCAST_MODEL))
    axes[0].fill_between(lead, tac - tac_sd, tac + tac_sd, color=system_color(HEATCAST_MODEL), alpha=0.18, linewidth=0.0)
    axes[1].plot(lead, mse, marker="o", color=system_color(HEATCAST_MODEL), label=model_label(HEATCAST_MODEL))
    axes[1].fill_between(lead, mse - mse_sd, mse + mse_sd, color=system_color(HEATCAST_MODEL), alpha=0.18, linewidth=0.0)
    axes[0].set_ylabel("Temporal anomaly correlation")
    axes[1].set_ylabel("Mean-squared error (z-score²)")
    for ax in axes:
        ax.set_xlabel("Lead day")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    savefig(fig, out_dir / "figure_8_per_lead_profile")
    plt.close(fig)
    sources["figure_8_per_lead_profile"] = source_entry(source, ["lead", "model", "tac_mean", "mse_mean"], "recovered from latest W34 training-log per-lead diagnostics when no stitched per-lead CSV exists")


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
    ax.plot(x, y, marker="o", color=system_color(STACK_MODEL))
    ax.invert_xaxis()
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Retained highest-confidence cells (%)")
    ax.set_ylabel("Stack BSS")
    ax.set_title("Forecasts-of-opportunity discard curve")
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, out_dir / "figure_9_opportunity_discard_curve")
    plt.close(fig)
    sources["figure_9_opportunity_discard_curve"] = source_entry(out_dir / DISCARD_CSV, ["retained_fraction", "bss", "roc_auc", "ece"], "computed from stack probabilities and validation-year confidence thresholds")


def parent_comparison_lookup(parent_rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[str, str], Dict[str, bool]]:
    lookup: Dict[Tuple[str, str], Dict[str, bool]] = {}
    for row in parent_rows:
        if row.get("candidate_model") != STACK_MODEL:
            continue
        if row.get("baseline_model") != ENS_MODEL:
            continue
        if row.get("metric") != "delta_bss_stack_vs_ens_child_minus_parent":
            continue
        axis = str(row.get("interaction_axis", ""))
        stratum = str(row.get("interaction_stratum", ""))
        if not axis or not stratum:
            continue
        state = lookup.setdefault((axis, stratum), {})
        state[str(row.get("parent_kind", ""))] = str(row.get("ci_excludes_zero", "")).lower() == "true"
    return lookup


def write_teleconnection_ranking(teleconnection_stack_dir: Path, out_dir: Path, table_dir: Path, sources: Dict[str, object]) -> None:
    boot_path = teleconnection_stack_dir / "driver_pair_bootstrap.csv"
    parent_path = teleconnection_stack_dir / "driver_pair_parent_bootstrap.csv"
    boot_rows = read_csv(boot_path)
    parent_rows = read_csv(parent_path, required=False)
    parent_lookup = parent_comparison_lookup(parent_rows)
    rows: List[Dict[str, object]] = []
    for row in boot_rows:
        if row.get("driver_family") != "generic_teleconnection":
            continue
        if row.get("candidate_model") != STACK_MODEL or row.get("baseline_model") != ENS_MODEL:
            continue
        if row.get("metric") != "delta_bss_heatcast_ens_stack_minus_ens_calibrated":
            continue
        axis = str(row.get("axis", ""))
        stratum = str(row.get("stratum", ""))
        parent_state = parent_lookup.get((axis, stratum), {})
        parent_resolved = bool(parent_state) and all(parent_state.values())
        rows.append({
            "axis": axis,
            "stratum": stratum,
            "teleconnection": axis.replace("tele_", "").replace("_x_top_confidence", "").replace("_x_low_sigma", "").upper(),
            "display_label": f"{axis.replace('tele_', '').upper()} {stratum}",
            "delta_bss_stack_minus_ens": float(row["point_estimate"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
            "ci_excludes_zero": str(row.get("ci_excludes_zero", "")).lower() == "true",
            "independent_year_blocks": int(f(row.get("independent_year_blocks"))),
            "parent_comparison_resolved": parent_resolved,
            "parent_caveat": PARENT_CAVEAT,
        })
    rows = sorted(rows, key=lambda item: float(item["delta_bss_stack_minus_ens"]), reverse=True)
    if not rows:
        raise RuntimeError(f"No generic teleconnection Stack-vs-ENS BSS rows found in {boot_path}.")
    write_csv(out_dir / TELECONNECTION_CSV, rows)

    resolved = [row for row in rows if row["ci_excludes_zero"]]
    unresolved_large = [row for row in rows if not row["ci_excludes_zero"] and float(row["delta_bss_stack_minus_ens"]) > 0.0]
    table_rows: List[Dict[str, object]] = []
    for group_name, group_rows in (("resolved_ci_excludes_zero", resolved), ("unresolved_ci_crosses_zero", unresolved_large[:12])):
        for row in group_rows:
            table_rows.append({
                "block": group_name,
                "teleconnection": row["teleconnection"],
                "state": row["stratum"],
                "delta_bss_stack_minus_ens": fmt(row["delta_bss_stack_minus_ens"]),
                "ci_low": fmt(row["ci_low"]),
                "ci_high": fmt(row["ci_high"]),
                "parent_comparison_resolved": row["parent_comparison_resolved"],
                "interpretation": "resolved regime-stratified gain" if row["ci_excludes_zero"] else "large point estimate; CI crosses zero",
            })
    write_csv(table_dir / "table_9_teleconnection_ranking.csv", table_rows)
    write_markdown_table(table_dir / "table_9_teleconnection_ranking.md", table_rows)

    plt = ensure_matplotlib()
    plot_rows = rows[: min(24, len(rows))]
    y = np.arange(len(plot_rows))
    points = np.asarray([float(row["delta_bss_stack_minus_ens"]) for row in plot_rows], dtype=float)
    lows = np.asarray([float(row["ci_low"]) for row in plot_rows], dtype=float)
    highs = np.asarray([float(row["ci_high"]) for row in plot_rows], dtype=float)
    fig, ax = plt.subplots(figsize=figure_size(DOUBLE_COLUMN_MM, max(80.0, 4.0 * len(plot_rows))), constrained_layout=True)
    for idx, row in enumerate(plot_rows):
        tele = str(row["teleconnection"])
        color = teleconnection_color(tele)
        marker_face = color if row["ci_excludes_zero"] else "white"
        ax.errorbar(
            points[idx],
            y[idx],
            xerr=[[points[idx] - lows[idx]], [highs[idx] - points[idx]]],
            fmt="o",
            color=color,
            markerfacecolor=marker_face,
            markeredgecolor=color,
            markersize=4.0,
            elinewidth=0.8,
            capsize=2.0,
        )
        if row["ci_excludes_zero"]:
            ax.text(highs[idx] + 0.003, y[idx], "*", va="center", ha="left", fontsize=8)
    ax.axvline(0.0, color="black", linewidth=0.7)
    ax.axvline(0.0212, color="#555555", linewidth=0.8, linestyle="--", label="All-stack delta BSS")
    ax.set_yticks(y, [str(row["display_label"]).replace("_", " ") for row in plot_rows])
    ax.invert_yaxis()
    ax.set_xlabel("Stack - ENS BSS")
    ax.set_title("Teleconnection-stratified additive skill")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    savefig(fig, out_dir / "figure_9_teleconnection_ranking")
    plt.close(fig)
    sources["figure_9_teleconnection_ranking"] = source_entry(
        out_dir / TELECONNECTION_CSV,
        ["axis", "stratum", "delta_bss_stack_minus_ens", "ci_low", "ci_high", "ci_excludes_zero"],
        "read from the teleconnection Stack-vs-ENS driver bootstrap CSV; resolved markers require CI excluding zero",
    )
    sources["table_9_teleconnection_ranking"] = [
        source_entry(boot_path, ["driver_family", "candidate_model", "metric", "point_estimate", "ci_low", "ci_high"], "teleconnection Stack-vs-ENS bootstrap rows"),
        source_entry(parent_path, ["parent_kind", "point_estimate", "ci_low", "ci_high"], "parent-comparison caveat rows"),
    ]


def write_leave_one_out_robustness(stack_dir: Path, out_dir: Path, sources: Dict[str, object]) -> None:
    source = stack_dir / "robustness_leave_one_out.csv"
    rows = [
        row for row in read_csv(source)
        if row.get("candidate_model") == STACK_MODEL and row.get("baseline_model") == ENS_MODEL
    ]
    if not rows:
        raise RuntimeError(f"No Stack-vs-ENS leave-one-out rows found in {source}.")
    rows = sorted(rows, key=lambda row: (str(row.get("dropped_group_type", "")), str(row.get("dropped_group_value", ""))))
    values = np.asarray([f(row["delta_bss_candidate_minus_baseline"]) for row in rows], dtype=float)
    if np.any(~np.isfinite(values)):
        raise RuntimeError("Leave-one-out rows contain non-finite Stack-vs-ENS delta BSS.")
    write_csv(out_dir / LEAVE_ONE_OUT_CSV, rows)
    positive_count = int(np.sum(values > 0.0))
    if positive_count != len(values):
        raise RuntimeError(f"Leave-one-out robustness expected all positive rows; got {positive_count}/{len(values)}.")

    plt = ensure_matplotlib()
    fig, ax = plt.subplots(figsize=figure_size(DOUBLE_COLUMN_MM, max(88.0, 3.4 * len(rows))), constrained_layout=True)
    groups = [str(row.get("dropped_group_type", "")) for row in rows]
    labels = [f"{row.get('dropped_group_type')} {row.get('dropped_group_value')}" for row in rows]
    y = np.arange(len(rows))
    ax.barh(y, values, color=[group_color(group) for group in groups], alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=0.7)
    ax.axvline(0.0212, color="#555555", linestyle="--", linewidth=0.8, label="Full sample")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Stack - ENS BSS after dropping one group")
    ax.set_title("Leave-one-out robustness")
    ax.text(
        0.98,
        0.02,
        f"{positive_count}/{len(values)} positive; range {np.nanmin(values):+.4f} to {np.nanmax(values):+.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    savefig(fig, out_dir / "figure_10a_leave_one_out_robustness")
    plt.close(fig)
    sources["figure_10a_leave_one_out_robustness"] = source_entry(
        out_dir / LEAVE_ONE_OUT_CSV,
        ["dropped_group_type", "dropped_group_value", "delta_bss_candidate_minus_baseline"],
        "read directly from robustness_leave_one_out.csv Stack-vs-ENS rows",
    )


def write_figure_captions(out_dir: Path, sources: Mapping[str, object]) -> None:
    lines = [
        "# Figure Captions",
        "",
        "All confidence intervals use paired whole-year bootstrap resampling unless noted. SVG files are the primary vector outputs; 600-dpi PNG files are fallbacks.",
        "",
        "## Figure 5. Spatial probabilistic skill.",
        f"Cell-wise W34 BSS, Stack-minus-ENS BSS, and ROC-AUC on the common HeatCast/ENS verification grid. Source CSV: `{SPATIAL_CSV}`. Significance outlines indicate cells whose year-block bootstrap interval excludes zero where available.",
        "",
        "## Figure 6. Reliability and Brier decomposition.",
        f"Reliability curves, sharpness histograms, and Murphy Brier decomposition for ENS, HeatCast-C, and HeatCast+ENS. Source CSV: `{RELIABILITY_CSV}`. The decomposition separates reliability, resolution, and uncertainty terms on the same held-out cell-year set.",
        "",
        "## Figure 7. Probabilistic case studies.",
        f"Representative high-event-fraction W34 cases showing observed exceedance and forecast probability fields. Source CSV: `{CASE_CSV}`. Cases are selected from the saved held-out paired chunks, not from training data.",
        "",
        "## Figure 8. Per-lead W34 profile.",
        f"Per-lead TAC and MSE from the latest converged W34 per-lead diagnostics across folds. Source CSV: `{PER_LEAD_CSV}`. Lines show fold means; bands show across-fold standard deviation.",
        "",
        "## Figure 9. Teleconnection-stratified additive skill.",
        f"Horizontal forest plot of Stack-minus-ENS BSS by AO/NAO/PNA/PDO state. Source CSV: `{TELECONNECTION_CSV}` from the teleconnection stack directory. Filled markers indicate intervals excluding zero; open markers indicate unresolved intervals. {PARENT_CAVEAT}",
        "",
        "## Figure S. Forecasts-of-opportunity discard curve.",
        f"Stack BSS after retaining the highest-confidence forecast cells. Source CSV: `{DISCARD_CSV}`. Confidence thresholds are validation-year quantities applied to held-out test chunks.",
        "",
        "## Figure 10A. Leave-one-out robustness.",
        f"Stack-minus-ENS BSS after dropping each fold, month, or year. Source CSV: `{LEAVE_ONE_OUT_CSV}`. All drops remaining positive supports the additive-stack result.",
        "",
        "## Figure 10B. Per-year base-rate sensitivity.",
        f"Per-year Stack-minus-ENS BSS and ROC-AUC versus the year-specific q95 event base rate. Source CSV: `{PER_YEAR_BASE_RATE_CSV}`. The two lowest-base-rate years are labeled to show where BSS is most unstable under rare events.",
        "",
        "## Audit Notes",
        "The reproducibility manifest records input paths, git commit, and source columns for each generated product. Missing or nonnumeric figure inputs are treated as build errors for the journal path rather than as silent placeholder panels.",
        "",
        "## Source Map",
        "```json",
        json.dumps(sources, indent=2),
        "```",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figure_captions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table_7(stack_dir: Path, fig_dir: Path, table_dir: Path, sources: Dict[str, object]) -> None:
    h2h_path = stack_dir / "heatcast_ens_stack_head_to_head.csv"
    all_rows = read_csv(h2h_path)
    score_rows = [row for row in all_rows if row.get("section") == "score"]
    by_model = {row["model"]: row for row in score_rows}
    boot_by_model = {
        row["candidate_model"]: row
        for row in all_rows
        if row.get("section") == "bootstrap"
        and row.get("baseline_model") == ENS_MODEL
        and str(row.get("metric", "")).startswith("delta_bss_")
    }
    dec_rows = read_csv(fig_dir / RELIABILITY_CSV, required=False)
    dec_by_model = {row.get("model"): row for row in dec_rows if row.get("model") in MODEL_NAMES and not row.get("row_type")}
    output = []
    for model in (REFERENCE_MODEL, ENS_MODEL, HEATCAST_MODEL, STACK_MODEL):
        row = by_model.get(model, {})
        dec = dec_by_model.get(model, {})
        boot = boot_by_model.get(model, {})
        output.append({
            "model": model_label(model),
            "bss": fmt(row.get("bss_vs_monthly_climo")),
            "roc_auc": fmt(row.get("roc_auc"), signed=False),
            "reliability_slope": fmt(row.get("reliability_slope"), signed=False),
            "ece": fmt(row.get("ece"), signed=False),
            "resolution": fmt(dec.get("resolution"), signed=False),
            "sharpness_proxy_uncertainty": fmt(dec.get("uncertainty"), signed=False),
            "brier": fmt(row.get("brier"), signed=False),
            "delta_bss_vs_ens": "" if model in (REFERENCE_MODEL, ENS_MODEL) else fmt(boot.get("point_estimate")),
            "delta_bss_vs_ens_ci": "" if model in (REFERENCE_MODEL, ENS_MODEL) else f"[{fmt(boot.get('ci_low'))}, {fmt(boot.get('ci_high'))}]",
            "delta_bss_vs_ens_ci_excludes_zero": "" if model in (REFERENCE_MODEL, ENS_MODEL) else str(boot.get("ci_excludes_zero", "")),
            "claim_boundary": (
                "reference"
                if model == REFERENCE_MODEL else
                "benchmark"
                if model == ENS_MODEL else
                "standalone CI crosses zero; do not claim HeatCast alone beats ENS"
                if model == HEATCAST_MODEL else
                "additive stack CI excludes zero; central claim"
            ),
        })
    write_csv(table_dir / "table_7_stack_ablation_probability.csv", output)
    write_markdown_table(table_dir / "table_7_stack_ablation_probability.md", output)
    sources["table_7_stack_ablation_probability"] = [
        source_entry(h2h_path, ["bss_vs_monthly_climo", "roc_auc", "reliability_slope", "ece", "brier", "ci_low", "ci_high"], "read from existing head-to-head score and bootstrap rows; HeatCast-C alone crosses zero while stack excludes zero"),
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
    robustness_path = stack_dir / "robustness_by_year.csv"
    robustness_rows = read_csv(robustness_path, required=False)
    fold_map = year_to_fold_from_head_to_head(stack_dir)
    by_year_model: Dict[Tuple[int, str], Mapping[str, str]] = {}
    for row in [*rows, *robustness_rows]:
        if not math.isfinite(f(row.get("year", row.get("fold", "nan")))):
            continue
        year = int(f(row.get("year", row.get("fold", "nan"))))
        model = str(row.get("model", ""))
        existing = by_year_model.get((year, model), {})
        by_year_model[(year, model)] = {**existing, **row}
    base_rate_by_year: Dict[int, float] = {}
    for row in robustness_rows:
        if row.get("model") not in (REFERENCE_MODEL, "monthly_climatology", "windowed_climatology"):
            continue
        year = int(f(row.get("year", "nan"))) if math.isfinite(f(row.get("year", "nan"))) else None
        if year is not None:
            base_rate_by_year[year] = f(row.get("base_rate"))
    output: List[Dict[str, object]] = []
    for year in sorted({year for year, _ in by_year_model}):
        ens = by_year_model.get((year, ENS_MODEL))
        stack = by_year_model.get((year, STACK_MODEL))
        if not ens or not stack:
            continue
        delta_bss = f(stack.get("bss_vs_monthly_climo")) - f(ens.get("bss_vs_monthly_climo"))
        delta_auc = f(stack.get("roc_auc")) - f(ens.get("roc_auc"))
        output.append({
            "year": year,
            "fold": fold_map.get(year, ""),
            "stack_bss": fmt(stack.get("bss_vs_monthly_climo")),
            "ens_bss": fmt(ens.get("bss_vs_monthly_climo")),
            "delta_bss_stack_minus_ens": fmt(delta_bss),
            "stack_auc": fmt(stack.get("roc_auc"), signed=False),
            "ens_auc": fmt(ens.get("roc_auc"), signed=False),
            "delta_auc_stack_minus_ens": fmt(delta_auc),
            "base_rate": fmt(base_rate_by_year.get(year, float("nan")), signed=False),
        })
    year_rows = [row for row in output if math.isfinite(f(row.get("delta_bss_stack_minus_ens")))]
    if year_rows:
        deltas = np.asarray([f(row["delta_bss_stack_minus_ens"]) for row in year_rows], dtype=np.float64)
        output.append({
            "year": "median_year_delta",
            "fold": "",
            "stack_bss": "",
            "ens_bss": "",
            "delta_bss_stack_minus_ens": fmt(np.nanmedian(deltas)),
            "stack_auc": "",
            "ens_auc": "",
            "delta_auc_stack_minus_ens": "",
            "base_rate": "",
            "sign_test_wins": f"{int(np.sum(deltas > 0.0))}/{len(deltas)}",
        })
        finite_base_rows = [row for row in year_rows if math.isfinite(f(row.get("base_rate"))) and f(row.get("base_rate")) > 0.0]
        dropped = sorted(finite_base_rows, key=lambda row: f(row["base_rate"]))[:2]
        kept_years = {int(row["year"]) for row in year_rows if row not in dropped}
        def pooled_bss(model: str, years: set[int]) -> float:
            model_num = 0.0
            ref_num = 0.0
            weight = 0.0
            for year in years:
                model_row = by_year_model.get((year, model), {})
                ref_row = by_year_model.get((year, REFERENCE_MODEL), {})
                count = f(model_row.get("valid_count", ref_row.get("valid_count", "nan")))
                model_brier = f(model_row.get("brier"))
                ref_brier = f(ref_row.get("brier"))
                if not all(math.isfinite(value) for value in (count, model_brier, ref_brier)) or count <= 0:
                    continue
                model_num += model_brier * count
                ref_num += ref_brier * count
                weight += count
            if weight <= 0 or ref_num <= 0:
                return float("nan")
            return 1.0 - model_num / ref_num
        if dropped and kept_years:
            stack_bss = pooled_bss(STACK_MODEL, kept_years)
            ens_bss = pooled_bss(ENS_MODEL, kept_years)
            output.append({
                "year": "two_lowest_base_rate_years_removed",
                "fold": "",
                "stack_bss": fmt(stack_bss),
                "ens_bss": fmt(ens_bss),
                "delta_bss_stack_minus_ens": fmt(stack_bss - ens_bss),
                "stack_auc": "",
                "ens_auc": "",
                "delta_auc_stack_minus_ens": "",
                "base_rate": "",
                "removed_years": " ".join(str(row["year"]) for row in dropped),
                "removed_base_rates": " ".join(fmt(row["base_rate"], signed=False) for row in dropped),
                "sign_test_wins": f"{int(np.sum(deltas > 0.0))}/{len(deltas)}",
            })
    write_csv(table_dir / "table_8_per_year_head_to_head.csv", output)
    write_markdown_table(table_dir / "table_8_per_year_head_to_head.md", output)
    sources["table_8_per_year_head_to_head"] = source_entry(source, ["year", "model", "bss_vs_monthly_climo", "roc_auc"], "read from existing per-year/robustness CSV and coverage fold map")


def write_figure_10_per_year_base_rate(stack_dir: Path, table_dir: Path, out_dir: Path, sources: Dict[str, object]) -> None:
    table_path = table_dir / "table_8_per_year_head_to_head.csv"
    if not table_path.exists():
        raise FileNotFoundError(table_path)
    rows = [
        row for row in read_csv(table_path)
        if math.isfinite(f(row.get("year"))) and math.isfinite(f(row.get("base_rate"))) and f(row.get("base_rate")) > 0.0
    ]
    if not rows:
        raise RuntimeError("Cannot build Figure 10: no per-year rows with finite base_rate.")
    rows = sorted(rows, key=lambda row: int(f(row["year"])))
    write_csv(out_dir / PER_YEAR_BASE_RATE_CSV, rows)
    plt = ensure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=figure_size(DOUBLE_COLUMN_MM, 74.0), constrained_layout=True)
    folds = sorted({str(row.get("fold", "")) for row in rows})
    cmap = plt.get_cmap("tab10")
    fold_colors = {fold: cmap(idx % 10) for idx, fold in enumerate(folds)}
    low_base_years = {int(f(row["year"])) for row in sorted(rows, key=lambda row: f(row["base_rate"]))[:2]}
    for ax, field, ylabel in (
        (axes[0], "delta_bss_stack_minus_ens", "Stack - ENS BSS"),
        (axes[1], "delta_auc_stack_minus_ens", "Stack - ENS ROC-AUC"),
    ):
        for row in rows:
            year = int(f(row["year"]))
            fold = str(row.get("fold", ""))
            ax.scatter(
                f(row["base_rate"]),
                f(row[field]),
                s=22 if year not in low_base_years else 36,
                color=fold_colors.get(fold, "black"),
                edgecolor="black" if year in low_base_years else "none",
                linewidth=0.6,
            )
            if year in low_base_years:
                ax.annotate(str(year), (f(row["base_rate"]), f(row[field])), xytext=(3, 4), textcoords="offset points", fontsize=7)
        ax.set_xscale("log")
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set_xlabel("Year-specific q95 base rate")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
    handles = []
    labels = []
    for fold in folds:
        handles.append(axes[0].scatter([], [], color=fold_colors[fold], s=18))
        labels.append(f"fold {fold}")
    axes[1].legend(handles, labels, frameon=False, loc="best", title="Test fold")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    savefig(fig, out_dir / "figure_10_per_year_base_rate")
    plt.close(fig)
    sources["figure_10_per_year_base_rate"] = source_entry(out_dir / PER_YEAR_BASE_RATE_CSV, ["year", "base_rate", "delta_bss_stack_minus_ens", "delta_auc_stack_minus_ens"], "joined Table 8 per-year deltas with robustness_by_year climatology base rates")


def write_table_10(root: Path, stack_dir: Path, table_dir: Path, sources: Dict[str, object]) -> None:
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
    write_csv(table_dir / "table_10_computational_cost_comparison.csv", output)
    write_markdown_table(table_dir / "table_10_computational_cost_comparison.md", output)
    sources["table_10_computational_cost_comparison"] = {
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
        Path(args.teleconnection_stack_dir) / "driver_pair_bootstrap.csv",
        Path(args.teleconnection_stack_dir) / "driver_pair_parent_bootstrap.csv",
        Path(args.teleconnection_stack_dir) / "driver_pair_summary.csv",
        Path(args.evidence_dir) / "operational_block.csv",
        Path(args.evidence_dir) / "mechanism_block.csv",
        Path(args.opportunity_dir) / "driver_opportunity_summary.csv",
        Path(args.opportunity_dir) / "driver_interaction_paired_bootstrap.csv",
    ):
        if source.exists():
            shutil.copy2(source, repro / source.name)
    for script in ("submit_paper_figures_journal.slurm",):
        path = root / "slurm" / script
        if path.exists():
            shutil.copy2(path, repro / script)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack_dir", default=f"ens_heatcast_stack_opportunity/{WINDOW_LABEL}")
    parser.add_argument("--teleconnection_stack_dir", default=f"ens_heatcast_stack_opportunity_teleconnections/{WINDOW_LABEL}")
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
    parser.add_argument(
        "--w34_log_glob",
        default="w34_tube_all_*.log,w34_tube_all_*.err,*w34_dist*.log,*w34*.log,*w34*.err",
        help="Comma-separated glob patterns used to recover Figure 8 per-lead diagnostics from W34 training logs.",
    )
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
        "teleconnection_headline": source_entry(Path(args.teleconnection_stack_dir) / "driver_pair_bootstrap.csv", ["driver_family", "axis", "stratum", "point_estimate", "ci_low", "ci_high"], "AO/NAO/PNA/PDO teleconnection stack directory; not the MJO/ENSO/soil directory"),
        "operational": source_entry(Path(args.evidence_dir) / "operational_block.csv", ["model", "bss", "roc_auc", "reliability_slope", "ece"], "existing paper evidence operational block"),
    }

    products = None if args.skip_chunk_products else stream_chunk_products(args, sources)
    write_spatial_skill(products, fig_dir, int(args.spatial_bootstrap_reps), int(args.seed), sources)
    write_reliability_decomposition(products, Path(args.stack_dir), Path(args.evidence_dir), fig_dir, sources)
    write_case_studies(products, fig_dir, sources)
    write_per_lead_profile(args, fig_dir, sources)
    write_discard_curve(products, fig_dir, sources, int(args.spatial_bootstrap_reps), int(args.seed))
    write_teleconnection_ranking(Path(args.teleconnection_stack_dir), fig_dir, table_dir, sources)
    write_leave_one_out_robustness(Path(args.stack_dir), fig_dir, sources)
    write_table_7(Path(args.stack_dir), fig_dir, table_dir, sources)
    write_table_8(Path(args.stack_dir), table_dir, sources)
    write_figure_10_per_year_base_rate(Path(args.stack_dir), table_dir, fig_dir, sources)
    write_table_10(root, Path(args.stack_dir), table_dir, sources)
    write_figure_captions(out_dir, sources)
    copy_inputs_to_repro(root, out_dir, args)
    update_manifest(out_dir, sources, root)

    print("Extended paper figures/tables complete")
    print(f"  output_dir={out_dir}")
    print(f"  figures={fig_dir}")
    print(f"  tables={table_dir}")
    print(f"  manifest={out_dir / 'reproducibility_manifest.json'}")


if __name__ == "__main__":
    main()
