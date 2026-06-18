#!/usr/bin/env python3
"""Paired HeatCast/ENS stacking and opportunity comparisons.

This script reads saved incremental test chunks only.  It aligns HeatCast and
ENS by init_time_index, merges duplicate ENS cycles the same way as
ens_compare.py, fits a cross-fitted logistic stacker that excludes the scored
fold, and reports paired HeatCast-vs-ENS and stack-vs-ENS comparisons.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

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
    return {
        "truth": truth,
        "base": base,
        "ens_raw": ens_raw,
        "ens_calibrated": ens_calibrated,
        "heatcast_C": heat_prob,
        "features": features,
        "sigma": sigma,
        "year": scalar(heat, "year"),
        "month": scalar(heat, "month"),
        "target_center_time_index": scalar(heat, "target_center_time_index"),
    }


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration_steps", type=int, default=200)
    parser.add_argument("--calibration_lr", type=float, default=0.1)
    parser.add_argument("--calibration_l2", type=float, default=1e-4)
    parser.add_argument("--max_stack_samples_per_fold", type=int, default=500000)
    parser.add_argument("--progress_every", type=int, default=100)
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
    rng = np.random.default_rng(int(args.seed))

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

    print("Building bounded paired reservoirs for cross-fitted HeatCast+ENS stacker.")
    reservoir_x: Dict[int, np.ndarray] = {}
    reservoir_y: Dict[int, np.ndarray] = {}
    for fold in sorted(fold_inputs):
        info = fold_inputs[fold]
        common = info["common"]
        per_init = max(1, int(np.ceil(float(args.max_stack_samples_per_fold) / max(len(common), 1))))
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
            if (index + 1) % max(1, int(args.progress_every)) == 0:
                print(f"  fold {fold}: sampled stack rows from {index + 1}/{len(common)} paired inits")
        reservoir_x[fold], reservoir_y[fold] = downsample_rows(
            x_parts, y_parts, int(args.max_stack_samples_per_fold), rng,
        )
        print(f"  fold {fold}: stack reservoir rows={reservoir_y[fold].size}")

    stackers = {
        fold: fit_stacker_for_excluded_fold(fold, reservoir_x, reservoir_y, args)
        for fold in sorted(fold_inputs)
    }
    for fold, stacker in sorted(stackers.items()):
        print(
            f"Fold {fold}: fitted stacker excluding scored fold, "
            f"n={stacker.n_samples}, event_rate={stacker.event_rate:.4f}"
        )

    global_acc = ee.EvaluationAccumulator(MODEL_NAMES, {})
    fold_acc: Dict[int, ee.EvaluationAccumulator] = {}
    by_fold_year: Dict[Tuple[int, int], ee.EvaluationAccumulator] = {}
    subset_acc = {name: ee.EvaluationAccumulator(MODEL_NAMES, {}) for name in SUBSETS}
    subset_year_acc: Dict[str, Dict[Tuple[int, int], ee.EvaluationAccumulator]] = {
        name: {} for name in SUBSETS
    }
    coverage_rows: List[Dict[str, object]] = []
    scored_years = set()

    print("Scoring paired test chunks with cross-fitted stacker.")
    for fold in sorted(fold_inputs):
        info = fold_inputs[fold]
        fold_acc[fold] = ee.EvaluationAccumulator(MODEL_NAMES, {})
        fold_years = set()
        stacker = stackers[fold]
        boundaries = info["boundaries"]
        duplicate_cycle_inits = 0
        for index, init_t in enumerate(info["common"]):
            data = paired_chunk(
                fold,
                int(init_t),
                info["heat_map"][int(init_t)],
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
            for name, probability in forecasts.items():
                global_acc.update(name, probability, truth, mask, month)
                fold_acc[fold].update(name, probability, truth, mask, month)
                year_acc.update(name, probability, truth, mask, month)

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
            fold_years.add(year)
            scored_years.add(year)
            if (index + 1) % max(1, int(args.progress_every)) == 0:
                print(f"  fold {fold}: scored {index + 1}/{len(info['common'])} paired inits")
        coverage_rows.append({
            "fold": fold,
            "heatcast_run": info["heat_name"],
            "ens_run": " + ".join(source[0] for source in info["ens_sources"]),
            "common_init_count": len(info["common"]),
            "duplicate_cycle_init_count": duplicate_cycle_inits,
            "intersection_years": " ".join(str(value) for value in sorted(fold_years)),
            "intersection_year_count": len(fold_years),
        })
        print(
            f"Fold {fold}: scored common_inits={len(info['common'])}, "
            f"duplicate-cycle inits={duplicate_cycle_inits}, years={sorted(fold_years)}"
        )

    rows = score_rows_from_folds(fold_acc)
    by_name = {str(row["model"]): row for row in rows}
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
                    int(args.bootstrap_reps),
                    int(args.seed) + 1000 + SUBSETS.index(subset_name),
                    subset_name,
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
    ee.write_csv(
        out_dir / "stacker_coefficients.csv",
        [
            {"fold": fold, **row}
            for fold, stacker in sorted(stackers.items())
            for row in stacker.coefficient_rows()
        ],
    )
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
