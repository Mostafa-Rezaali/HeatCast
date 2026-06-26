#!/usr/bin/env python3
"""Paired HeatCast/ENS stacking and opportunity comparisons.

This script reads saved incremental test chunks only.  It aligns HeatCast and
ENS by init_time_index, merges duplicate ENS cycles the same way as
ens_compare.py, fits a cross-fitted logistic stacker that excludes the scored
fold, and reports paired HeatCast-vs-ENS and stack-vs-ENS comparisons.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import exceedance_eval as ee
from ens_common import ENS_BENCHMARK_BANNER
from ens_compare import (
    add_metric,
    chunk_map,
    fit_heatcast_c,
    load_chunk,
    merge_cycle_probabilities,
    resolve_ens_run_groups,
    scalar,
    weighted_fold_auc,
)
from stitch_exceedance_folds import load_fold_inputs


REFERENCE = "windowed_climatology"
ENS_MODEL = "ens_calibrated"
HEATCAST_MODEL = "heatcast_C"
STACK_MODEL = "heatcast_ens_stack"
MODEL_NAMES = (REFERENCE, "ens_raw_fraction", ENS_MODEL, HEATCAST_MODEL, STACK_MODEL)
STACK_FEATURE_NAMES = (
    "ens_calibrated_logit",
    "ens_raw_logit",
    "heatcast_C_logit",
    "heatcast_init_margin",
    "heatcast_forecast_margin",
    "heatcast_sigma",
)
SUBSETS = ("all", "heatcast_top10_confidence", "heatcast_low_sigma_tercile", "heatcast_top10_and_low_sigma")
BASE_DRIVER_AXES = ("mjo_phase", "enso_state", "soil_moisture_tercile")
DRIVER_PARENT_SELECTIONS = {
    "top_confidence": "heatcast_top10_confidence",
    "low_sigma": "heatcast_low_sigma_tercile",
}


def logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float32), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p)).astype(np.float32)


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(np.asarray(arrays[0]).shape, dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def stack_features(
    ens_raw: np.ndarray,
    ens_calibrated: np.ndarray,
    heat_prob: np.ndarray,
    heat_chunk: Mapping[str, np.ndarray],
) -> np.ndarray:
    return np.column_stack([
        logit(ens_calibrated),
        logit(ens_raw),
        logit(heat_prob),
        np.asarray(heat_chunk["init_margin"], dtype=np.float32),
        np.asarray(heat_chunk["forecast_margin"], dtype=np.float32),
        np.asarray(heat_chunk["model_sigma"], dtype=np.float32),
    ]).astype(np.float32)


def score_rows_from_folds(
    fold_accumulators: Mapping[int, ee.EvaluationAccumulator],
    model_names: Sequence[str] = MODEL_NAMES,
) -> List[Dict[str, object]]:
    pooled = ee.EvaluationAccumulator(model_names, {})
    for source in fold_accumulators.values():
        for model in model_names:
            add_metric(pooled.metrics[model], source.metrics[model])
    rows = pooled.summary_rows(REFERENCE)
    for row in rows:
        row["weighted_per_fold_roc_auc"] = weighted_fold_auc(fold_accumulators, str(row["model"]))
        row["roc_auc"] = row["weighted_per_fold_roc_auc"]
    return rows


def aggregate_selected_years(
    by_fold_year: Mapping[Tuple[int, int], ee.EvaluationAccumulator],
    selected_years: Sequence[int],
    model_names: Sequence[str] = MODEL_NAMES,
) -> Dict[int, ee.EvaluationAccumulator]:
    selected_counts = defaultdict(int)
    for year in selected_years:
        selected_counts[int(year)] += 1
    output: Dict[int, ee.EvaluationAccumulator] = {}
    for (fold, year), source in by_fold_year.items():
        weight = selected_counts.get(int(year), 0)
        if weight <= 0:
            continue
        target = output.setdefault(int(fold), ee.EvaluationAccumulator(model_names, {}))
        for model in model_names:
            add_metric(target.metrics[model], source.metrics[model], weight)
    return output


def bootstrap_delta_rows(
    by_fold_year: Mapping[Tuple[int, int], ee.EvaluationAccumulator],
    years: Sequence[int],
    candidates: Sequence[str],
    baseline: str,
    reps: int,
    seed: int,
    label: str,
) -> List[Dict[str, object]]:
    rng = np.random.default_rng(int(seed))
    year_values = np.array(sorted(set(int(value) for value in years)), dtype=np.int16)
    if year_values.size < 2:
        raise RuntimeError("Year-block bootstrap requires at least two years.")

    point_rows = {
        str(row["model"]): row
        for row in score_rows_from_folds(aggregate_selected_years(by_fold_year, year_values))
    }
    output: List[Dict[str, object]] = []
    boot_values = {candidate: {"bss": [], "auc": []} for candidate in candidates}
    for _ in range(int(reps)):
        selected = rng.choice(year_values, size=year_values.size, replace=True)
        rows = {
            str(row["model"]): row
            for row in score_rows_from_folds(aggregate_selected_years(by_fold_year, selected))
        }
        for candidate in candidates:
            boot_values[candidate]["bss"].append(
                float(rows[candidate]["bss_vs_monthly_climo"]) - float(rows[baseline]["bss_vs_monthly_climo"])
            )
            boot_values[candidate]["auc"].append(
                float(rows[candidate]["weighted_per_fold_roc_auc"]) - float(rows[baseline]["weighted_per_fold_roc_auc"])
            )

    for candidate in candidates:
        point_bss = float(point_rows[candidate]["bss_vs_monthly_climo"]) - float(point_rows[baseline]["bss_vs_monthly_climo"])
        point_auc = float(point_rows[candidate]["weighted_per_fold_roc_auc"]) - float(point_rows[baseline]["weighted_per_fold_roc_auc"])
        for metric, point in (("delta_bss", point_bss), ("delta_auc", point_auc)):
            array = np.asarray(boot_values[candidate][metric.split("_")[1]], dtype=np.float64)
            lo, hi = np.nanpercentile(array, [2.5, 97.5])
            output.append({
                "comparison_set": label,
                "candidate_model": candidate,
                "baseline_model": baseline,
                "metric": f"{metric}_{candidate}_minus_{baseline}",
                "point_estimate": point,
                "ci_low": float(lo),
                "ci_high": float(hi),
                "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
                "bootstrap_reps": int(reps),
                "independent_year_blocks": int(year_values.size),
            })
    return output


def driver_key(axis: str, stratum: str) -> str:
    return f"{axis}::{stratum}"


def split_driver_key(key: str) -> Tuple[str, str]:
    if "::" not in str(key):
        raise ValueError(f"Invalid driver stratum key: {key!r}")
    axis, stratum = str(key).split("::", 1)
    return axis, stratum


def base_driver_axis(axis: str) -> str:
    return str(axis).split("_x_", 1)[0]


def driver_family(axis: str) -> str:
    base = base_driver_axis(axis)
    if base in BASE_DRIVER_AXES:
        return "base_driver"
    if base.startswith("tele_"):
        return "generic_teleconnection"
    if base.startswith("alldata_"):
        return "alldata"
    return "other_driver"


def driver_interaction_parent_pairs(keys: Iterable[str]) -> List[Tuple[str, str, str, str]]:
    """Return driver interaction child/parent pairs for paired Stack-vs-ENS tests."""
    key_set = set(str(key) for key in keys)
    pairs: List[Tuple[str, str, str, str]] = []
    for key in sorted(key_set):
        axis, stratum = split_driver_key(key)
        if "__" not in stratum or "_x_" not in axis:
            continue
        driver_axis = base_driver_axis(axis)
        driver_stratum = stratum.split("__", 1)[0]
        driver_parent = driver_key(driver_axis, driver_stratum)
        if axis.endswith("_x_top_confidence"):
            selection_parent = DRIVER_PARENT_SELECTIONS["top_confidence"]
            selection_kind = "selection_parent_top_confidence"
        elif axis.endswith("_x_low_sigma"):
            selection_parent = DRIVER_PARENT_SELECTIONS["low_sigma"]
            selection_kind = "selection_parent_low_sigma"
        else:
            continue
        if selection_parent:
            pairs.append((key, selection_kind, "subset", selection_parent))
        if driver_parent in key_set:
            pairs.append((key, "driver_parent", "driver", driver_parent))
    return pairs


def _delta_from_rows(
    rows: Mapping[str, Mapping[str, object]],
    candidate: str,
    baseline: str,
    metric: str,
) -> float:
    if metric == "bss":
        field = "bss_vs_monthly_climo"
    elif metric == "auc":
        field = "weighted_per_fold_roc_auc"
    else:
        raise ValueError(metric)
    return float(rows[candidate][field]) - float(rows[baseline][field])


def bootstrap_parent_delta_rows(
    child_by_year: Mapping[Tuple[int, int], ee.EvaluationAccumulator],
    parent_by_year: Mapping[Tuple[int, int], ee.EvaluationAccumulator],
    child_label: str,
    parent_label: str,
    parent_kind: str,
    reps: int,
    seed: int,
    candidate: str = STACK_MODEL,
    baseline: str = ENS_MODEL,
) -> List[Dict[str, object]]:
    """Bootstrap whether a child stratum improves Stack-vs-ENS delta over its parent."""
    years = np.array(sorted({int(year) for _, year in child_by_year}), dtype=np.int16)
    if years.size < 2:
        return []
    rng = np.random.default_rng(int(seed))
    child_point = {
        str(row["model"]): row
        for row in score_rows_from_folds(aggregate_selected_years(child_by_year, years))
    }
    parent_point = {
        str(row["model"]): row
        for row in score_rows_from_folds(aggregate_selected_years(parent_by_year, years))
    }
    samples = {"bss": [], "auc": []}
    for _ in range(int(reps)):
        selected = rng.choice(years, size=years.size, replace=True)
        child_rows = {
            str(row["model"]): row
            for row in score_rows_from_folds(aggregate_selected_years(child_by_year, selected))
        }
        parent_rows = {
            str(row["model"]): row
            for row in score_rows_from_folds(aggregate_selected_years(parent_by_year, selected))
        }
        for metric in samples:
            samples[metric].append(
                _delta_from_rows(child_rows, candidate, baseline, metric)
                - _delta_from_rows(parent_rows, candidate, baseline, metric)
            )

    output: List[Dict[str, object]] = []
    child_axis, child_stratum = split_driver_key(child_label)
    if "::" in parent_label:
        parent_axis, parent_stratum = split_driver_key(parent_label)
    else:
        parent_axis, parent_stratum = "opportunity_selection", parent_label
    for metric in ("bss", "auc"):
        point = (
            _delta_from_rows(child_point, candidate, baseline, metric)
            - _delta_from_rows(parent_point, candidate, baseline, metric)
        )
        values = np.asarray(samples[metric], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size:
            lo, hi = np.nanpercentile(finite, [2.5, 97.5])
            lower_tail = (np.sum(finite <= 0.0) + 1.0) / (finite.size + 1.0)
            upper_tail = (np.sum(finite >= 0.0) + 1.0) / (finite.size + 1.0)
            p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
        else:
            lo = hi = p_value = float("nan")
        output.append({
            "interaction_axis": child_axis,
            "interaction_stratum": child_stratum,
            "interaction_family": driver_family(child_axis),
            "parent_kind": parent_kind,
            "parent_axis": parent_axis,
            "parent_stratum": parent_stratum,
            "parent_family": driver_family(parent_axis) if "::" in parent_label else "opportunity_selection",
            "candidate_model": candidate,
            "baseline_model": baseline,
            "metric": f"delta_{metric}_stack_vs_ens_child_minus_parent",
            "point_estimate": float(point),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
            "p_value": float(p_value),
            "bootstrap_reps": int(reps),
            "independent_year_blocks": int(years.size),
        })
    return output


def summarize_fold_accumulators(
    accumulators: Mapping[int, ee.EvaluationAccumulator],
    label_name: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for label, acc in sorted(accumulators.items()):
        for row in acc.summary_rows(REFERENCE):
            rows.append({label_name: label, **row})
    return rows


def summarize_named_accumulators(
    accumulators: Mapping[str, ee.EvaluationAccumulator],
    label_name: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for label, acc in sorted(accumulators.items()):
        for row in acc.summary_rows(REFERENCE):
            rows.append({label_name: label, **row})
    return rows


def summarize_driver_accumulators(
    accumulators: Mapping[str, ee.EvaluationAccumulator],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for key, acc in sorted(accumulators.items()):
        axis, stratum = split_driver_key(key)
        for row in acc.summary_rows(REFERENCE):
            rows.append({"axis": axis, "stratum": stratum, "driver_family": driver_family(axis), **row})
    return rows


def aggregate_eval_accumulators(
    sources: Iterable[ee.EvaluationAccumulator],
    model_names: Sequence[str] = MODEL_NAMES,
) -> ee.EvaluationAccumulator:
    target = ee.EvaluationAccumulator(model_names, {})
    for source in sources:
        for model in model_names:
            add_metric(target.metrics[model], source.metrics[model])
    return target


def delta_summary_row(
    label: str,
    acc: ee.EvaluationAccumulator,
    candidate: str,
    baseline: str = ENS_MODEL,
) -> Dict[str, object]:
    rows = {str(row["model"]): row for row in acc.summary_rows(REFERENCE)}
    return {
        "label": label,
        "candidate_model": candidate,
        "baseline_model": baseline,
        "candidate_bss": rows[candidate]["bss_vs_monthly_climo"],
        "baseline_bss": rows[baseline]["bss_vs_monthly_climo"],
        "delta_bss_candidate_minus_baseline": (
            float(rows[candidate]["bss_vs_monthly_climo"])
            - float(rows[baseline]["bss_vs_monthly_climo"])
        ),
        "candidate_roc_auc": rows[candidate]["roc_auc"],
        "baseline_roc_auc": rows[baseline]["roc_auc"],
        "delta_auc_candidate_minus_baseline": (
            float(rows[candidate]["roc_auc"]) - float(rows[baseline]["roc_auc"])
        ),
        "candidate_ece": rows[candidate]["ece"],
        "baseline_ece": rows[baseline]["ece"],
        "valid_count": rows[candidate]["valid_count"],
    }


def leave_one_out_rows(
    group_accumulators: Mapping[object, ee.EvaluationAccumulator],
    group_name: str,
    candidates: Sequence[str] = (HEATCAST_MODEL, STACK_MODEL),
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    labels = list(group_accumulators)
    for dropped in sorted(labels):
        kept = [acc for label, acc in group_accumulators.items() if label != dropped]
        if not kept:
            continue
        pooled = aggregate_eval_accumulators(kept)
        for candidate in candidates:
            row = delta_summary_row(f"drop_{group_name}_{dropped}", pooled, candidate)
            row["dropped_group_type"] = group_name
            row["dropped_group_value"] = dropped
            rows.append(row)
    return rows


def load_region_masks_for_land_order(cell_count: int) -> Dict[str, np.ndarray]:
    import cfm_mesh_train as cfm
    from publication_analysis_utils import region_masks

    land_mask = np.asarray(cfm.load_conus_mask(cfm.Config).cpu().numpy() > 0.5, dtype=bool)
    land_count = int(np.sum(land_mask))
    if land_count != int(cell_count):
        raise RuntimeError(
            f"Region mask land_count={land_count} does not match chunk cell_count={cell_count}; "
            "saved chunks may not be full land-order arrays."
        )
    return {
        name: np.asarray(mask, dtype=bool).ravel()[land_mask.ravel()]
        for name, mask in region_masks(land_mask.shape).items()
    }


def paired_chunk(
    fold: int,
    init_t: int,
    heat_path: Path,
    ens_sources: Sequence[Tuple[str, Mapping[str, object], Mapping[int, Path]]],
    heat_c,
) -> Dict[str, np.ndarray | int]:
    heat = load_chunk(heat_path)
    matching_sources = [source for source in ens_sources if init_t in source[2]]
    if not matching_sources:
        raise RuntimeError(f"Fold {fold}, init={init_t}: no matching ENS source.")
    ens_chunks = []
    for ens_name, ens_manifest, ens_map in matching_sources:
        ens = load_chunk(ens_map[init_t])
        for key in ("truth", "base_rate"):
            if heat[key].shape != ens[key].shape or not np.allclose(heat[key], ens[key], equal_nan=True):
                raise RuntimeError(f"Fold {fold}, init={init_t}: HeatCast/{ens_name} {key} differs.")
        for key in ("year", "month", "target_center_time_index"):
            if scalar(heat, key) != scalar(ens, key):
                raise RuntimeError(f"Fold {fold}, init={init_t}: HeatCast/{ens_name} {key} differs.")
        ens_chunks.append(ens)

    ens_raw, ens_calibrated = merge_cycle_probabilities(ens_chunks)
    heat_prob = heat_c.predict_features(np.column_stack([
        np.asarray(heat["init_margin"], dtype=np.float32),
        np.asarray(heat["forecast_margin"], dtype=np.float32),
    ]).astype(np.float32))
    truth = np.asarray(heat["truth"], dtype=np.float32)
    base = np.asarray(heat["base_rate"], dtype=np.float32)
    sigma = np.asarray(heat["model_sigma"], dtype=np.float32)
    features = stack_features(ens_raw, ens_calibrated, heat_prob, heat)
    out = {
        "truth": truth,
        "base": base,
        "ens_raw": ens_raw,
        "ens_calibrated": ens_calibrated,
        "heatcast_C": heat_prob,
        "features": features,
        "sigma": sigma,
        "year": scalar(heat, "year"),
        "month": scalar(heat, "month"),
        "source_fold": scalar(heat, "source_fold") if "source_fold" in heat else fold,
        "init_time_index": scalar(heat, "init_time_index"),
        "target_center_time_index": scalar(heat, "target_center_time_index"),
    }
    for key in ("mu_z", "truth_z"):
        if key in heat:
            out[key] = np.asarray(heat[key], dtype=np.float32)
    return out


def fit_opportunity_boundaries(calibration: Mapping[str, np.ndarray], heat_c) -> Dict[str, float]:
    features = np.column_stack([
        calibration["init_margin"],
        calibration["forecast_margin"],
    ]).astype(np.float32)
    heat_prob = heat_c.predict_features(features)
    base = np.asarray(calibration["base_rate"], dtype=np.float32)
    sigma = np.asarray(calibration["model_sigma"], dtype=np.float32)
    confidence = np.abs(heat_prob - base)
    valid_conf = confidence[np.isfinite(confidence)]
    valid_sigma = sigma[np.isfinite(sigma)]
    if valid_conf.size < 100 or valid_sigma.size < 100:
        raise RuntimeError("Not enough calibration cells to fit opportunity boundaries.")
    return {
        "top10_confidence_threshold": float(np.nanquantile(valid_conf, 0.90)),
        "low_sigma_threshold": float(np.nanquantile(valid_sigma, 1.0 / 3.0)),
    }


def sample_for_stack(
    data: Mapping[str, np.ndarray | int],
    max_rows: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(data["truth"], dtype=np.float32)
    features = np.asarray(data["features"], dtype=np.float32)
    ens_cal = np.asarray(data["ens_calibrated"], dtype=np.float32)
    heat_prob = np.asarray(data["heatcast_C"], dtype=np.float32)
    mask = finite_mask(truth, ens_cal, heat_prob, *[features[:, i] for i in range(features.shape[1])])
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.empty((0, features.shape[1]), dtype=np.float32), np.empty(0, dtype=np.float32)
    take = min(int(max_rows), int(idx.size))
    selected = rng.choice(idx, size=take, replace=False) if idx.size > take else idx
    return features[selected].astype(np.float32), truth[selected].astype(np.float32)


def downsample_rows(
    x_parts: Sequence[np.ndarray],
    y_parts: Sequence[np.ndarray],
    max_rows: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if not x_parts:
        return np.empty((0, len(STACK_FEATURE_NAMES)), dtype=np.float32), np.empty(0, dtype=np.float32)
    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    if x.shape[0] > int(max_rows):
        idx = rng.choice(np.arange(x.shape[0]), size=int(max_rows), replace=False)
        x = x[idx]
        y = y[idx]
    return x.astype(np.float32), y.astype(np.float32)


def fit_stacker_for_excluded_fold(
    fold: int,
    reservoir_x: Mapping[int, np.ndarray],
    reservoir_y: Mapping[int, np.ndarray],
    args: argparse.Namespace,
):
    x_parts = [reservoir_x[other] for other in sorted(reservoir_x) if int(other) != int(fold)]
    y_parts = [reservoir_y[other] for other in sorted(reservoir_y) if int(other) != int(fold)]
    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    if x.shape[0] < 1000:
        raise RuntimeError(f"Fold {fold}: not enough cross-fit rows to fit HeatCast+ENS stacker.")
    return ee.fit_model_output_logistic_calibrator(
        x,
        y,
        STACK_FEATURE_NAMES,
        calibration_split=f"crossfit_excluding_fold{fold}",
        steps=args.calibration_steps,
        lr=args.calibration_lr,
        l2=args.calibration_l2,
    )


def build_reservoir_for_fold(
    fold: int,
    info: Mapping[str, object],
    max_stack_samples_per_fold: int,
    seed: int,
    progress_every: int,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Build a bounded paired reservoir for one fold."""
    rng = np.random.default_rng(int(seed) + 1009 * int(fold))
    common = info["common"]
    per_init = max(1, int(np.ceil(float(max_stack_samples_per_fold) / max(len(common), 1))))
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    for index, init_t in enumerate(common):
        data = paired_chunk(
            fold,
            int(init_t),
            info["heat_map"][int(init_t)],
            info["ens_sources"],
            info["heat_c"],
        )
        x_part, y_part = sample_for_stack(data, per_init, rng)
        if x_part.size:
            x_parts.append(x_part)
            y_parts.append(y_part)
        if (index + 1) % max(1, int(progress_every)) == 0:
            print(f"  fold {fold}: sampled stack rows from {index + 1}/{len(common)} paired inits")
    x, y = downsample_rows(x_parts, y_parts, int(max_stack_samples_per_fold), rng)
    print(f"  fold {fold}: stack reservoir rows={y.size}")
    return int(fold), x, y


def score_fold_chunks(
    fold: int,
    info: Mapping[str, object],
    stacker,
    progress_every: int,
    region_masks: Mapping[str, np.ndarray] | None = None,
    driver_lookup: Optional[object] = None,
) -> Tuple[
    int,
    ee.EvaluationAccumulator,
    Dict[Tuple[int, int], ee.EvaluationAccumulator],
    Dict[Tuple[int, int], ee.EvaluationAccumulator],
    Dict[Tuple[int, int, int], ee.EvaluationAccumulator],
    Dict[str, ee.EvaluationAccumulator],
    Dict[str, Dict[Tuple[int, int], ee.EvaluationAccumulator]],
    Dict[str, ee.EvaluationAccumulator],
    Dict[str, Dict[Tuple[int, int], ee.EvaluationAccumulator]],
    Dict[str, ee.EvaluationAccumulator],
    Dict[str, Dict[Tuple[int, int], ee.EvaluationAccumulator]],
    Dict[str, object],
    set[int],
]:
    """Score all paired chunks for one fold, returning independent accumulators."""
    fold_acc = ee.EvaluationAccumulator(MODEL_NAMES, {})
    by_fold_year: Dict[Tuple[int, int], ee.EvaluationAccumulator] = {}
    by_fold_month: Dict[Tuple[int, int], ee.EvaluationAccumulator] = {}
    by_fold_month_year: Dict[Tuple[int, int, int], ee.EvaluationAccumulator] = {}
    subset_acc = {name: ee.EvaluationAccumulator(MODEL_NAMES, {}) for name in SUBSETS}
    subset_year_acc: Dict[str, Dict[Tuple[int, int], ee.EvaluationAccumulator]] = {
        name: {} for name in SUBSETS
    }
    region_acc = {
        name: ee.EvaluationAccumulator(MODEL_NAMES, {})
        for name in (region_masks or {})
    }
    region_year_acc: Dict[str, Dict[Tuple[int, int], ee.EvaluationAccumulator]] = {
        name: {} for name in (region_masks or {})
    }
    driver_acc: Dict[str, ee.EvaluationAccumulator] = {}
    driver_year_acc: Dict[str, Dict[Tuple[int, int], ee.EvaluationAccumulator]] = {}
    fold_years = set()
    boundaries = info["boundaries"]
    duplicate_cycle_inits = 0
    soil_undefined = 0
    soil_total = 0
    for index, init_t in enumerate(info["common"]):
        heat_path = info["heat_map"][int(init_t)]
        data = paired_chunk(
            fold,
            int(init_t),
            heat_path,
            info["ens_sources"],
            info["heat_c"],
        )
        year = int(data["year"])
        month = int(data["month"])
        if year not in info["manifest"]["test_years"]:
            raise RuntimeError(f"Fold {fold}, init={init_t}: chunk year {year} is not in fold test years.")
        matching_sources = [source for source in info["ens_sources"] if int(init_t) in source[2]]
        duplicate_cycle_inits += int(len(matching_sources) > 1)
        truth = np.asarray(data["truth"], dtype=np.float32)
        base = np.asarray(data["base"], dtype=np.float32)
        ens_raw = np.asarray(data["ens_raw"], dtype=np.float32)
        ens_cal = np.asarray(data["ens_calibrated"], dtype=np.float32)
        heat_prob = np.asarray(data["heatcast_C"], dtype=np.float32)
        features = np.asarray(data["features"], dtype=np.float32)
        sigma = np.asarray(data["sigma"], dtype=np.float32)
        stack_prob = stacker.predict_features(features)
        mask = finite_mask(truth, base, ens_raw, ens_cal, heat_prob, stack_prob)
        forecasts = {
            REFERENCE: base,
            "ens_raw_fraction": ens_raw,
            ENS_MODEL: ens_cal,
            HEATCAST_MODEL: heat_prob,
            STACK_MODEL: stack_prob,
        }
        year_acc = by_fold_year.setdefault((fold, year), ee.EvaluationAccumulator(MODEL_NAMES, {}))
        month_acc = by_fold_month.setdefault((fold, month), ee.EvaluationAccumulator(MODEL_NAMES, {}))
        month_year_acc = by_fold_month_year.setdefault((fold, month, year), ee.EvaluationAccumulator(MODEL_NAMES, {}))
        for name, probability in forecasts.items():
            fold_acc.update(name, probability, truth, mask, month)
            year_acc.update(name, probability, truth, mask, month)
            month_acc.update(name, probability, truth, mask, month)
            month_year_acc.update(name, probability, truth, mask, month)

        confidence = np.abs(heat_prob - base)
        subset_masks = {
            "all": mask,
            "heatcast_top10_confidence": mask & (confidence >= boundaries["top10_confidence_threshold"]),
            "heatcast_low_sigma_tercile": mask & (sigma <= boundaries["low_sigma_threshold"]),
            "heatcast_top10_and_low_sigma": (
                mask
                & (confidence >= boundaries["top10_confidence_threshold"])
                & (sigma <= boundaries["low_sigma_threshold"])
            ),
        }
        for subset_name, subset_mask in subset_masks.items():
            update_subset_accumulators(
                subset_acc,
                subset_year_acc,
                subset_name,
                fold,
                year,
                month,
                forecasts,
                truth,
                subset_mask,
            )
        for region_name, region_mask in (region_masks or {}).items():
            region_selection = mask & region_mask
            update_subset_accumulators(
                region_acc,
                region_year_acc,
                region_name,
                fold,
                year,
                month,
                forecasts,
                truth,
                region_selection,
            )
        if driver_lookup is not None:
            driver_selections = driver_lookup.sample_strata(fold, heat_path, data)[1]
            top_selection = confidence >= boundaries["top10_confidence_threshold"]
            low_sigma = sigma <= boundaries["low_sigma_threshold"]
            for driver_axis, strata in driver_selections.items():
                for stratum, driver_selection in strata.items():
                    driver_selection = np.asarray(driver_selection, dtype=bool)
                    base_key = driver_key(driver_axis, stratum)
                    top_key = driver_key(
                        f"{driver_axis}_x_top_confidence",
                        f"{stratum}__top_10pct_ge_p90",
                    )
                    low_key = driver_key(
                        f"{driver_axis}_x_low_sigma",
                        f"{stratum}__bottom_sigma_tercile",
                    )
                    for key, selection in (
                        (base_key, driver_selection),
                        (top_key, driver_selection & top_selection),
                        (low_key, driver_selection & low_sigma),
                    ):
                        selected_mask = mask & selection
                        if not np.any(selected_mask):
                            continue
                        if key not in driver_acc:
                            driver_acc[key] = ee.EvaluationAccumulator(MODEL_NAMES, {})
                            driver_year_acc[key] = {}
                        update_subset_accumulators(
                            driver_acc,
                            driver_year_acc,
                            key,
                            fold,
                            year,
                            month,
                            forecasts,
                            truth,
                            selected_mask,
                        )
                if driver_axis == "soil_moisture_tercile":
                    undefined = np.asarray(strata.get("undefined", np.zeros_like(mask)), dtype=bool)
                    soil_undefined += int(np.sum(undefined))
                    soil_total += int(undefined.size)
        fold_years.add(year)
        if (index + 1) % max(1, int(progress_every)) == 0:
            print(f"  fold {fold}: scored {index + 1}/{len(info['common'])} paired inits")
    coverage_row = {
        "fold": fold,
        "heatcast_run": info["heat_name"],
        "ens_run": " + ".join(source[0] for source in info["ens_sources"]),
        "common_init_count": len(info["common"]),
        "duplicate_cycle_init_count": duplicate_cycle_inits,
        "intersection_years": " ".join(str(value) for value in sorted(fold_years)),
        "intersection_year_count": len(fold_years),
    }
    print(
        f"Fold {fold}: scored common_inits={len(info['common'])}, "
        f"duplicate-cycle inits={duplicate_cycle_inits}, years={sorted(fold_years)}"
    )
    if driver_lookup is not None:
        undefined_fraction = soil_undefined / max(soil_total, 1)
        if undefined_fraction >= 0.05:
            raise RuntimeError(f"Fold {fold}: undefined soil percentile fraction={undefined_fraction:.4f} >= 0.05.")
        print(f"Fold {fold}: paired slow-driver strata scored; soil undefined={undefined_fraction:.4%}.")
    return (
        fold,
        fold_acc,
        by_fold_year,
        by_fold_month,
        by_fold_month_year,
        subset_acc,
        subset_year_acc,
        region_acc,
        region_year_acc,
        driver_acc,
        driver_year_acc,
        coverage_row,
        fold_years,
    )


def update_subset_accumulators(
    subset_acc: Mapping[str, ee.EvaluationAccumulator],
    subset_year_acc: Mapping[str, Dict[Tuple[int, int], ee.EvaluationAccumulator]],
    subset_name: str,
    fold: int,
    year: int,
    month: int,
    forecasts: Mapping[str, np.ndarray],
    truth: np.ndarray,
    mask: np.ndarray,
) -> None:
    if not np.any(mask):
        return
    year_acc = subset_year_acc[subset_name].setdefault((int(fold), int(year)), ee.EvaluationAccumulator(MODEL_NAMES, {}))
    for name, probability in forecasts.items():
        subset_acc[subset_name].update(name, probability, truth, mask, month)
        year_acc.update(name, probability, truth, mask, month)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heatcast_runs", required=True, help="Comma-separated five HeatCast run names.")
    parser.add_argument(
        "--ens_runs",
        required=True,
        help="Comma-separated ENS runs, or cycle templates containing {F}, grouped and merged per fold.",
    )
    parser.add_argument("--window_leads", default="15,16,17,18,19,20,21,22,23,24,25,26,27,28")
    parser.add_argument("--heatcast_root", default="exceedance_eval_incremental")
    parser.add_argument("--ens_root", default="ens_exceedance_incremental")
    parser.add_argument("--output_dir", default="ens_heatcast_stack_opportunity")
    parser.add_argument("--bootstrap_reps", type=int, default=5000)
    parser.add_argument(
        "--stratum_bootstrap_reps",
        type=int,
        default=0,
        help="Bootstrap reps for region/driver/opportunity strata; 0 uses --bootstrap_reps.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration_steps", type=int, default=200)
    parser.add_argument("--calibration_lr", type=float, default=0.1)
    parser.add_argument("--calibration_l2", type=float, default=1e-4)
    parser.add_argument("--max_stack_samples_per_fold", type=int, default=500000)
    parser.add_argument("--fold_workers", type=int, default=1, help="Number of folds to stream concurrently.")
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--disable_region_robustness", action="store_true")
    parser.add_argument(
        "--driver_table_dir",
        default="",
        help="Optional fold-safe slow-driver table directory from build_driver_tables.py.",
    )
    parser.add_argument("--emit_per_year", action="store_true")
    args = parser.parse_args()

    print(ENS_BENCHMARK_BANNER)
    heatcast_runs = tuple(value.strip() for value in args.heatcast_runs.split(",") if value.strip())
    ens_runs = tuple(value.strip() for value in args.ens_runs.split(",") if value.strip())
    if len(heatcast_runs) < 2:
        raise RuntimeError("Cross-fitted stacker requires at least two HeatCast folds.")
    window_leads = ee.parse_int_list(args.window_leads)
    heatcast_root = Path(args.heatcast_root)
    ens_root = Path(args.ens_root)
    ens_groups = resolve_ens_run_groups(ens_runs, heatcast_runs, ens_root, window_leads)

    fold_inputs: Dict[int, Dict[str, object]] = {}
    all_years = set()
    total_common_inits = 0
    fold_workers = max(1, min(int(args.fold_workers), len(heatcast_runs)))
    stratum_reps = int(args.stratum_bootstrap_reps) if int(args.stratum_bootstrap_reps) > 0 else int(args.bootstrap_reps)

    print("Loading fold metadata and fitting per-fold HeatCast-C calibrators.")
    for heat_name in heatcast_runs:
        heat_manifest, heat_calibration, heat_chunks = load_fold_inputs(heatcast_root, heat_name, window_leads)
        fold = int(heat_manifest["source_fold"])
        if fold in fold_inputs:
            raise RuntimeError(f"Duplicate HeatCast source_fold={fold}.")
        ens_sources = []
        for ens_name in ens_groups[fold]:
            ens_manifest, _, ens_chunks = load_fold_inputs(ens_root, ens_name, window_leads)
            if int(ens_manifest["source_fold"]) != fold:
                raise RuntimeError(f"Fold mismatch: HeatCast={fold}, ENS={ens_manifest['source_fold']}.")
            if set(ens_manifest["train_years"]) & set(ens_manifest["test_years"]):
                raise RuntimeError(f"Fold {fold}, ENS {ens_name}: train/test overlap.")
            ens_sources.append((ens_name, ens_manifest, chunk_map(ens_chunks)))
        heat_map = chunk_map(heat_chunks)
        ens_union = set().union(*(set(source[2]) for source in ens_sources))
        common = tuple(sorted(set(heat_map) & ens_union))
        if not common:
            raise RuntimeError(f"Fold {fold}: empty common-init HeatCast/ENS intersection.")
        heat_c = fit_heatcast_c(heat_calibration, args)
        boundaries = fit_opportunity_boundaries(heat_calibration, heat_c)
        fold_inputs[fold] = {
            "heat_name": heat_name,
            "manifest": heat_manifest,
            "heat_map": heat_map,
            "ens_sources": ens_sources,
            "common": common,
            "heat_c": heat_c,
            "boundaries": boundaries,
        }
        all_years.update(int(year) for year in heat_manifest["test_years"])
        total_common_inits += len(common)
        print(
            f"Fold {fold}: common_inits={len(common)}, test_years={sorted(heat_manifest['test_years'])}, "
            f"top10_conf>={boundaries['top10_confidence_threshold']:.4f}, "
            f"low_sigma<={boundaries['low_sigma_threshold']:.4f}"
        )

    print(f"Building bounded paired reservoirs for cross-fitted HeatCast+ENS stacker with fold_workers={fold_workers}.")
    reservoir_x: Dict[int, np.ndarray] = {}
    reservoir_y: Dict[int, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=fold_workers) as pool:
        futures = [
            pool.submit(
                build_reservoir_for_fold,
                fold,
                fold_inputs[fold],
                int(args.max_stack_samples_per_fold),
                int(args.seed),
                int(args.progress_every),
            )
            for fold in sorted(fold_inputs)
        ]
        for future in as_completed(futures):
            fold, x, y = future.result()
            reservoir_x[fold] = x
            reservoir_y[fold] = y

    stackers = {
        fold: fit_stacker_for_excluded_fold(fold, reservoir_x, reservoir_y, args)
        for fold in sorted(fold_inputs)
    }
    for fold, stacker in sorted(stackers.items()):
        print(
            f"Fold {fold}: fitted stacker excluding scored fold, "
            f"n={stacker.n_samples}, event_rate={stacker.event_rate:.4f}"
        )

    fold_acc: Dict[int, ee.EvaluationAccumulator] = {}
    by_fold_year: Dict[Tuple[int, int], ee.EvaluationAccumulator] = {}
    by_fold_month: Dict[Tuple[int, int], ee.EvaluationAccumulator] = {}
    by_fold_month_year: Dict[Tuple[int, int, int], ee.EvaluationAccumulator] = {}
    subset_acc = {name: ee.EvaluationAccumulator(MODEL_NAMES, {}) for name in SUBSETS}
    subset_year_acc: Dict[str, Dict[Tuple[int, int], ee.EvaluationAccumulator]] = {
        name: {} for name in SUBSETS
    }
    region_acc: Dict[str, ee.EvaluationAccumulator] = {}
    region_year_acc: Dict[str, Dict[Tuple[int, int], ee.EvaluationAccumulator]] = {}
    driver_acc: Dict[str, ee.EvaluationAccumulator] = {}
    driver_year_acc: Dict[str, Dict[Tuple[int, int], ee.EvaluationAccumulator]] = {}
    coverage_rows: List[Dict[str, object]] = []
    scored_years = set()
    first_fold = sorted(fold_inputs)[0]
    first_init = int(fold_inputs[first_fold]["common"][0])
    first_data = paired_chunk(
        first_fold,
        first_init,
        fold_inputs[first_fold]["heat_map"][first_init],
        fold_inputs[first_fold]["ens_sources"],
        fold_inputs[first_fold]["heat_c"],
    )
    land_count = len(first_data["truth"])
    driver_lookup = None
    if str(args.driver_table_dir).strip():
        from forecasts_of_opportunity import load_driver_lookup

        driver_lookup = load_driver_lookup(
            Path(args.driver_table_dir),
            [fold_inputs[fold]["manifest"] for fold in sorted(fold_inputs)],
            int(land_count),
        )
        print(f"Paired slow-driver Stack-vs-ENS tests enabled from {args.driver_table_dir}.")
    region_masks = None
    if not args.disable_region_robustness:
        try:
            region_masks = load_region_masks_for_land_order(land_count)
            region_acc = {
                name: ee.EvaluationAccumulator(MODEL_NAMES, {})
                for name in region_masks
            }
            region_year_acc = {name: {} for name in region_masks}
            print(f"Region robustness enabled for {len(region_masks)} regions.")
        except Exception as exc:
            region_masks = None
            print(f"WARNING: region robustness disabled: {exc}")

    print(f"Scoring paired test chunks with cross-fitted stacker using fold_workers={fold_workers}.")
    with ThreadPoolExecutor(max_workers=fold_workers) as pool:
        futures = [
            pool.submit(
                score_fold_chunks,
                fold,
                fold_inputs[fold],
                stackers[fold],
                int(args.progress_every),
                region_masks,
                driver_lookup,
            )
            for fold in sorted(fold_inputs)
        ]
        for future in as_completed(futures):
            (
                fold,
                fold_acc_one,
                by_fold_year_one,
                by_fold_month_one,
                by_fold_month_year_one,
                subset_acc_one,
                subset_year_acc_one,
                region_acc_one,
                region_year_acc_one,
                driver_acc_one,
                driver_year_acc_one,
                coverage_row,
                fold_years,
            ) = future.result()
            fold_acc[fold] = fold_acc_one
            by_fold_year.update(by_fold_year_one)
            by_fold_month.update(by_fold_month_one)
            by_fold_month_year.update(by_fold_month_year_one)
            coverage_rows.append(coverage_row)
            scored_years.update(fold_years)
            for subset_name in SUBSETS:
                for model in MODEL_NAMES:
                    add_metric(subset_acc[subset_name].metrics[model], subset_acc_one[subset_name].metrics[model])
                subset_year_acc[subset_name].update(subset_year_acc_one[subset_name])
            for region_name in region_acc_one:
                if region_name not in region_acc:
                    region_acc[region_name] = ee.EvaluationAccumulator(MODEL_NAMES, {})
                    region_year_acc[region_name] = {}
                for model in MODEL_NAMES:
                    add_metric(region_acc[region_name].metrics[model], region_acc_one[region_name].metrics[model])
                region_year_acc[region_name].update(region_year_acc_one[region_name])
            for key, acc_one in driver_acc_one.items():
                if key not in driver_acc:
                    driver_acc[key] = ee.EvaluationAccumulator(MODEL_NAMES, {})
                    driver_year_acc[key] = {}
                for model in MODEL_NAMES:
                    add_metric(driver_acc[key].metrics[model], acc_one.metrics[model])
                driver_year_acc[key].update(driver_year_acc_one[key])

    rows = score_rows_from_folds(fold_acc)
    by_name = {str(row["model"]): row for row in rows}
    print(f"Running headline year-block bootstrap: reps={int(args.bootstrap_reps)}")
    bootstrap_rows = bootstrap_delta_rows(
        by_fold_year,
        sorted(scored_years),
        (HEATCAST_MODEL, STACK_MODEL),
        ENS_MODEL,
        int(args.bootstrap_reps),
        int(args.seed),
        "all",
    )
    subset_rows: List[Dict[str, object]] = []
    subset_bootstrap_rows: List[Dict[str, object]] = []
    print(f"Running opportunity-subset bootstraps: subsets={len(SUBSETS)}, reps={stratum_reps}")
    for subset_name in SUBSETS:
        for row in subset_acc[subset_name].summary_rows(REFERENCE):
            subset_rows.append({"subset": subset_name, **row})
        subset_years = sorted({year for _, year in subset_year_acc[subset_name]})
        if len(subset_years) >= 2:
            subset_bootstrap_rows.extend(
                bootstrap_delta_rows(
                    subset_year_acc[subset_name],
                    subset_years,
                    (HEATCAST_MODEL, STACK_MODEL),
                    ENS_MODEL,
                    stratum_reps,
                    int(args.seed) + 1000 + SUBSETS.index(subset_name),
                    subset_name,
                )
            )
    month_acc = {
        month: aggregate_eval_accumulators(
            acc for (fold, month_key), acc in by_fold_month.items() if month_key == month
        )
        for month in sorted({month for _, month in by_fold_month})
    }
    year_acc = {
        year: aggregate_eval_accumulators(
            acc for (fold, year_key), acc in by_fold_year.items() if year_key == year
        )
        for year in sorted({year for _, year in by_fold_year})
    }
    dominance_rows: List[Dict[str, object]] = []
    dominance_rows.extend(leave_one_out_rows(fold_acc, "fold"))
    dominance_rows.extend(leave_one_out_rows(month_acc, "month"))
    dominance_rows.extend(leave_one_out_rows(year_acc, "year"))

    region_rows: List[Dict[str, object]] = summarize_named_accumulators(region_acc, "region") if region_acc else []
    region_bootstrap_rows: List[Dict[str, object]] = []
    if region_year_acc:
        print(f"Running region bootstraps: regions={len(region_year_acc)}, reps={stratum_reps}")
    for region_name, region_by_year in region_year_acc.items():
        region_years = sorted({year for _, year in region_by_year})
        if len(region_years) >= 2:
            print(f"  region bootstrap {region_name}: years={len(region_years)}")
            region_bootstrap_rows.extend(
                bootstrap_delta_rows(
                    region_by_year,
                    region_years,
                    (HEATCAST_MODEL, STACK_MODEL),
                    ENS_MODEL,
                    stratum_reps,
                    int(args.seed) + 2000 + len(region_bootstrap_rows),
                    f"region_{region_name}",
                )
            )

    driver_rows: List[Dict[str, object]] = summarize_driver_accumulators(driver_acc) if driver_acc else []
    driver_bootstrap_rows: List[Dict[str, object]] = []
    driver_parent_rows: List[Dict[str, object]] = []
    driver_items = sorted(driver_year_acc.items())
    if driver_items:
        family_counts: Dict[str, int] = defaultdict(int)
        for key, _ in driver_items:
            axis, _stratum = split_driver_key(key)
            family_counts[driver_family(axis)] += 1
        family_text = ", ".join(f"{name}={count}" for name, count in sorted(family_counts.items()))
        print(
            f"Running paired driver Stack-vs-ENS bootstraps: strata={len(driver_items)}, "
            f"families=({family_text}), reps={stratum_reps}"
        )
    for driver_index, (key, key_by_year) in enumerate(driver_items, start=1):
        key_years = sorted({year for _, year in key_by_year})
        if len(key_years) >= 2:
            axis, stratum = split_driver_key(key)
            print(f"  driver bootstrap {driver_index}/{len(driver_items)} {axis}:{stratum}, years={len(key_years)}")
            rows_one = bootstrap_delta_rows(
                key_by_year,
                key_years,
                (HEATCAST_MODEL, STACK_MODEL),
                ENS_MODEL,
                stratum_reps,
                int(args.seed) + 3000 + len(driver_bootstrap_rows),
                f"{axis}:{stratum}",
            )
            for row in rows_one:
                row["axis"] = axis
                row["stratum"] = stratum
                row["driver_family"] = driver_family(axis)
            driver_bootstrap_rows.extend(rows_one)
    parent_pairs = driver_interaction_parent_pairs(driver_year_acc)
    if parent_pairs:
        print(f"Running paired driver parent-comparison bootstraps: pairs={len(parent_pairs)}, reps={stratum_reps}")
    for pair_index, (child_key, parent_kind, parent_source, parent_key) in enumerate(parent_pairs, start=1):
        parent_by_year = (
            subset_year_acc[parent_key]
            if parent_source == "subset"
            else driver_year_acc.get(parent_key, {})
        )
        if parent_by_year:
            child_axis, child_stratum = split_driver_key(child_key)
            print(
                f"  driver parent bootstrap {pair_index}/{len(parent_pairs)} "
                f"{child_axis}:{child_stratum} vs {parent_kind}"
            )
            driver_parent_rows.extend(
                bootstrap_parent_delta_rows(
                    driver_year_acc[child_key],
                    parent_by_year,
                    child_key,
                    parent_key,
                    parent_kind,
                    stratum_reps,
                    int(args.seed) + 4000 + len(driver_parent_rows),
                )
            )

    out_dir = Path(args.output_dir) / f"window_{ee.lead_list_label(window_leads)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_rows: List[Dict[str, object]] = []
    year_text = " ".join(str(value) for value in sorted(scored_years))
    for row in rows:
        combined_rows.append({
            "section": "score",
            "intersection_years": year_text,
            "intersection_year_count": len(scored_years),
            "common_init_count": total_common_inits,
            **row,
        })
    combined_rows.extend({"section": "coverage", **row} for row in coverage_rows)
    combined_rows.extend({"section": "bootstrap", **row} for row in bootstrap_rows)
    ee.write_csv(out_dir / "heatcast_ens_stack_head_to_head.csv", combined_rows)
    ee.write_csv(out_dir / "opportunity_pair_summary.csv", subset_rows)
    ee.write_csv(out_dir / "opportunity_pair_bootstrap.csv", subset_bootstrap_rows)
    ee.write_csv(out_dir / "robustness_by_fold.csv", summarize_fold_accumulators(fold_acc, "fold"))
    ee.write_csv(out_dir / "robustness_by_month.csv", summarize_fold_accumulators(month_acc, "month"))
    ee.write_csv(out_dir / "robustness_by_year.csv", summarize_fold_accumulators(year_acc, "year"))
    ee.write_csv(out_dir / "robustness_leave_one_out.csv", dominance_rows)
    if region_rows:
        ee.write_csv(out_dir / "robustness_by_region.csv", region_rows)
        ee.write_csv(out_dir / "robustness_region_bootstrap.csv", region_bootstrap_rows)
    if driver_rows:
        ee.write_csv(out_dir / "driver_pair_summary.csv", driver_rows)
        ee.write_csv(out_dir / "driver_pair_bootstrap.csv", driver_bootstrap_rows)
        ee.write_csv(out_dir / "driver_pair_parent_bootstrap.csv", driver_parent_rows)
    ee.write_csv(
        out_dir / "stacker_coefficients.csv",
        [
            {"fold": fold, **row}
            for fold, stacker in sorted(stackers.items())
            for row in stacker.coefficient_rows()
        ],
    )
    global_acc = ee.EvaluationAccumulator(MODEL_NAMES, {})
    for source in fold_acc.values():
        for model in MODEL_NAMES:
            add_metric(global_acc.metrics[model], source.metrics[model])
    ee.plot_reliability(
        out_dir / "reliability_overlay.png",
        {name: global_acc.metrics[name].rel.table() for name in MODEL_NAMES},
    )

    print("\nPaired HeatCast/ENS stack summary")
    print("=================================")
    for row in rows:
        print(
            f"{row['model']:<24} N={int(row['valid_count'])} "
            f"Brier={row['brier']:.5f} BSS={row['bss_vs_monthly_climo']:+.4f} "
            f"weighted-fold-AUC={row['weighted_per_fold_roc_auc']:.3f} "
            f"slope={row['reliability_slope']:.3f} ECE={row['ece']:.4f}"
        )
    print(f"Year-block bootstrap: {len(scored_years)} independent intersection-year blocks")
    for row in bootstrap_rows:
        print(
            f"  {row['metric']}: estimate={row['point_estimate']:+.4f} "
            f"CI=[{row['ci_low']:+.4f},{row['ci_high']:+.4f}], "
            f"excludes_zero={row['ci_excludes_zero']}"
        )
    print("\nOpportunity paired comparisons")
    print("==============================")
    for row in subset_bootstrap_rows:
        if row["metric"].startswith("delta_bss"):
            print(
                f"  {row['comparison_set']}: {row['candidate_model']} vs ENS "
                f"delta_BSS={row['point_estimate']:+.4f} "
                f"CI=[{row['ci_low']:+.4f},{row['ci_high']:+.4f}], "
                f"excludes_zero={row['ci_excludes_zero']}"
            )
    print("\nRobustness checks")
    print("=================")
    for group_type in ("fold", "month", "year"):
        stack_rows = [
            row for row in dominance_rows
            if row["dropped_group_type"] == group_type and row["candidate_model"] == STACK_MODEL
        ]
        if stack_rows:
            deltas = np.asarray([row["delta_bss_candidate_minus_baseline"] for row in stack_rows], dtype=np.float64)
            print(
                f"  leave-one-{group_type}: Stack-vs-ENS delta_BSS "
                f"min={np.nanmin(deltas):+.4f}, max={np.nanmax(deltas):+.4f}, "
                f"n={len(stack_rows)}"
            )
    if region_acc:
        print(f"  region robustness: wrote {len(region_acc)} regions with year-block bootstrap.")
    else:
        print("  region robustness: unavailable from current chunk metadata/mask state.")
    if driver_acc:
        print("\nPaired driver-stratified Stack-vs-ENS tests")
        print("===========================================")
        family_counts: Dict[str, int] = defaultdict(int)
        for key in driver_acc:
            axis, _stratum = split_driver_key(key)
            family_counts[driver_family(axis)] += 1
        print(
            "  driver families: "
            + ", ".join(f"{name}={count}" for name, count in sorted(family_counts.items()))
        )
        print(
            f"  wrote {len(driver_acc)} driver/intersection strata, "
            f"{len(driver_bootstrap_rows)} Stack-vs-ENS bootstrap rows, "
            f"{len(driver_parent_rows)} parent-comparison rows."
        )
        stack_bss_rows = [
            row for row in driver_bootstrap_rows
            if row.get("candidate_model") == STACK_MODEL and str(row.get("metric", "")).startswith("delta_bss")
        ]
        stack_bss_rows.sort(key=lambda row: float(row["point_estimate"]), reverse=True)
        for row in stack_bss_rows[:8]:
            print(
                f"  top paired driver {row['driver_family']} {row['axis']}:{row['stratum']} "
                f"Stack-vs-ENS delta_BSS={row['point_estimate']:+.4f} "
                f"CI=[{row['ci_low']:+.4f},{row['ci_high']:+.4f}], "
                f"excludes_zero={row['ci_excludes_zero']}"
            )
        required = {
            ("mjo_phase_x_top_confidence", "phase_8__top_10pct_ge_p90", "selection_parent_top_confidence"),
            ("mjo_phase_x_top_confidence", "phase_8__top_10pct_ge_p90", "driver_parent"),
            ("mjo_phase_x_low_sigma", "phase_8__bottom_sigma_tercile", "selection_parent_low_sigma"),
            ("mjo_phase_x_low_sigma", "phase_8__bottom_sigma_tercile", "driver_parent"),
        }
        found = 0
        for row in driver_parent_rows:
            if row["metric"] != "delta_bss_stack_vs_ens_child_minus_parent":
                continue
            key = (str(row["interaction_axis"]), str(row["interaction_stratum"]), str(row["parent_kind"]))
            if key in required:
                found += 1
                print(
                    f"  {row['interaction_axis']}:{row['interaction_stratum']} vs "
                    f"{row['parent_kind']} {row['parent_axis']}:{row['parent_stratum']} "
                    f"delta(Stack-ENS BSS)={row['point_estimate']:+.4f} "
                    f"CI=[{row['ci_low']:+.4f},{row['ci_high']:+.4f}], "
                    f"excludes_zero={row['ci_excludes_zero']}"
                )
        print(f"  required phase-8 parent comparisons found={found}/4")
    else:
        print("  paired driver-stratified Stack-vs-ENS tests: skipped (no --driver_table_dir).")
    print("Cross-fit assert: PASS (each scored fold excluded from its own HeatCast+ENS stacker fit).")
    print("Paired alignment assert: PASS (HeatCast and ENS matched by init_time_index and identical truth/base fields).")
    print(
        f"HEADLINE: HeatCast-C BSS={by_name[HEATCAST_MODEL]['bss_vs_monthly_climo']:+.4f}, "
        f"ENS BSS={by_name[ENS_MODEL]['bss_vs_monthly_climo']:+.4f}, "
        f"Stack BSS={by_name[STACK_MODEL]['bss_vs_monthly_climo']:+.4f}"
    )
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
