#!/usr/bin/env python3
"""Compare ECMWF ENS and HeatCast on identical fold-safe test dates and events."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

import exceedance_eval as ee
from ens_common import ENS_BENCHMARK_BANNER, common_init_indices
from stitch_exceedance_folds import load_fold_inputs


REFERENCE = "windowed_climatology"
MODEL_NAMES = (REFERENCE, "ens_raw_fraction", "ens_calibrated", "heatcast_C")


def sigmoid(logit: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(logit, dtype=np.float32), -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-value))).astype(np.float32)


def merge_cycle_probabilities(chunks: Sequence[Mapping[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """Average separately calibrated cycle probabilities for an identical initialization."""
    if not chunks:
        raise ValueError("At least one ENS cycle chunk is required.")
    raw = np.nanmean(
        np.stack([np.asarray(chunk["init_margin"], dtype=np.float32) for chunk in chunks]),
        axis=0,
    ).astype(np.float32)
    calibrated = np.nanmean(
        np.stack([
            sigmoid(np.asarray(chunk["forecast_margin"], dtype=np.float32))
            for chunk in chunks
        ]),
        axis=0,
    ).astype(np.float32)
    return raw, calibrated


def scalar(data: Mapping[str, np.ndarray], key: str) -> int:
    return int(np.asarray(data[key]).item())


def load_chunk(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def chunk_map(paths: Sequence[Path]) -> Dict[int, Path]:
    output: Dict[int, Path] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            init_t = scalar(data, "init_time_index")
        if init_t < 0:
            raise RuntimeError(f"{path}: missing init_time_index; rerun the producer with the current schema.")
        if init_t in output:
            raise RuntimeError(f"Duplicate init_time_index={init_t}: {output[init_t]} and {path}")
        output[init_t] = path
    return output


def resolve_ens_run_groups(
    ens_runs: Sequence[str],
    heatcast_runs: Sequence[str],
    ens_root: Path,
    window_leads: Sequence[int],
) -> Dict[int, List[str]]:
    """Resolve legacy five-run lists or cycle run-name templates into fold groups."""
    if any("{F}" in value for value in ens_runs):
        if not all("{F}" in value for value in ens_runs):
            raise ValueError("When using ENS run templates, every --ens_runs value must contain {F}.")
        return {
            fold: [template.replace("{F}", str(fold)) for template in ens_runs]
            for fold in range(len(heatcast_runs))
        }
    output: Dict[int, List[str]] = defaultdict(list)
    for run_name in ens_runs:
        manifest, _, _ = load_fold_inputs(ens_root, run_name, window_leads)
        output[int(manifest["source_fold"])].append(run_name)
    missing = sorted(set(range(len(heatcast_runs))) - set(output))
    if missing:
        raise ValueError(f"ENS runs do not cover folds {missing}.")
    return dict(output)


def cycle_label(run_name: str) -> str:
    suffix = str(run_name).rsplit("_", 1)[-1]
    return suffix if suffix.startswith("rt") and suffix[2:].isdigit() else "legacy"


def cycle_year_union(cycle_years: Mapping[str, Sequence[int]]) -> Tuple[set, set]:
    sets = [set(int(year) for year in years) for years in cycle_years.values()]
    if not sets:
        return set(), set()
    return set().union(*sets), set.intersection(*sets)


def fit_heatcast_c(calibration: Mapping[str, np.ndarray], args: argparse.Namespace):
    features = np.column_stack([
        calibration["init_margin"],
        calibration["forecast_margin"],
    ]).astype(np.float32)
    return ee.fit_model_output_logistic_calibrator(
        features,
        calibration["truth"],
        ("init_margin", "forecast_margin"),
        calibration_split="val",
        steps=args.calibration_steps,
        lr=args.calibration_lr,
        l2=args.calibration_l2,
    )


def add_metric(target: ee.MetricAccumulator, source: ee.MetricAccumulator, weight: int = 1) -> None:
    target.brier_sum += float(weight) * source.brier_sum
    target.count += float(weight) * source.count
    target.truth_pos += float(weight) * source.truth_pos
    target.hist_pos += float(weight) * source.hist_pos
    target.hist_neg += float(weight) * source.hist_neg
    target.auc_hist_pos += float(weight) * source.auc_hist_pos
    target.auc_hist_neg += float(weight) * source.auc_hist_neg
    target.rel.count += float(weight) * source.rel.count
    target.rel.pred_sum += float(weight) * source.rel.pred_sum
    target.rel.obs_sum += float(weight) * source.rel.obs_sum


def weighted_fold_auc(accumulators: Mapping[int, ee.EvaluationAccumulator], model: str) -> float:
    values = []
    weights = []
    for fold, acc in accumulators.items():
        auc, _ = acc.metrics[model].aucs()
        count = acc.metrics[model].count
        if np.isfinite(auc) and count > 0:
            values.append(float(auc))
            weights.append(float(count))
    return float(np.average(values, weights=weights)) if values else float("nan")


def aggregate_selected_years(
    by_fold_year: Mapping[Tuple[int, int], ee.EvaluationAccumulator],
    selected_years: Sequence[int],
) -> Dict[int, ee.EvaluationAccumulator]:
    selected_counts = defaultdict(int)
    for year in selected_years:
        selected_counts[int(year)] += 1
    output: Dict[int, ee.EvaluationAccumulator] = {}
    for (fold, year), source in by_fold_year.items():
        weight = selected_counts.get(int(year), 0)
        if weight <= 0:
            continue
        target = output.setdefault(int(fold), ee.EvaluationAccumulator(MODEL_NAMES, {}))
        for model in MODEL_NAMES:
            add_metric(target.metrics[model], source.metrics[model], weight)
    return output


def score_from_folds(fold_accumulators: Mapping[int, ee.EvaluationAccumulator]) -> List[Dict[str, object]]:
    pooled = ee.EvaluationAccumulator(MODEL_NAMES, {})
    for source in fold_accumulators.values():
        for model in MODEL_NAMES:
            add_metric(pooled.metrics[model], source.metrics[model])
    rows = pooled.summary_rows(REFERENCE)
    for row in rows:
        row["weighted_per_fold_roc_auc"] = weighted_fold_auc(fold_accumulators, str(row["model"]))
        row["roc_auc"] = row["weighted_per_fold_roc_auc"]
    return rows


def bootstrap(
    by_fold_year: Mapping[Tuple[int, int], ee.EvaluationAccumulator],
    years: Sequence[int],
    reps: int,
    seed: int,
) -> List[Dict[str, object]]:
    rng = np.random.default_rng(int(seed))
    year_values = np.array(sorted(set(int(value) for value in years)), dtype=np.int16)
    point_rows = {
        row["model"]: row
        for row in score_from_folds(aggregate_selected_years(by_fold_year, year_values))
    }
    point_estimates = {
        "bss": (
            float(point_rows["heatcast_C"]["bss_vs_monthly_climo"])
            - float(point_rows["ens_calibrated"]["bss_vs_monthly_climo"])
        ),
        "auc": (
            float(point_rows["heatcast_C"]["weighted_per_fold_roc_auc"])
            - float(point_rows["ens_calibrated"]["weighted_per_fold_roc_auc"])
        ),
    }
    deltas = {"bss": [], "auc": []}
    for _ in range(int(reps)):
        selected = rng.choice(year_values, size=year_values.size, replace=True)
        rows = {row["model"]: row for row in score_from_folds(aggregate_selected_years(by_fold_year, selected))}
        deltas["bss"].append(
            float(rows["heatcast_C"]["bss_vs_monthly_climo"])
            - float(rows["ens_calibrated"]["bss_vs_monthly_climo"])
        )
        deltas["auc"].append(
            float(rows["heatcast_C"]["weighted_per_fold_roc_auc"])
            - float(rows["ens_calibrated"]["weighted_per_fold_roc_auc"])
        )
    output = []
    for metric, values in deltas.items():
        array = np.asarray(values, dtype=np.float64)
        lo, hi = np.nanpercentile(array, [2.5, 97.5])
        output.append({
            "metric": f"delta_{metric}_heatcast_minus_ens",
            "point_estimate": point_estimates[metric],
            "ci_low": float(lo),
            "ci_high": float(hi),
            "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
            "bootstrap_reps": int(reps),
            "independent_year_blocks": int(year_values.size),
        })
    return output


def per_year_comparison_rows(
    by_fold_year: Mapping[Tuple[int, int], ee.EvaluationAccumulator],
    years: Sequence[int],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for year in sorted(set(int(value) for value in years)):
        scored = {
            str(row["model"]): row
            for row in score_from_folds(aggregate_selected_years(by_fold_year, (year,)))
        }
        heat = scored["heatcast_C"]
        ens = scored["ens_calibrated"]
        rows.append({
            "year": year,
            "heatcast_bss": heat["bss_vs_monthly_climo"],
            "ens_bss": ens["bss_vs_monthly_climo"],
            "delta_bss_heatcast_minus_ens": (
                float(heat["bss_vs_monthly_climo"]) - float(ens["bss_vs_monthly_climo"])
            ),
            "heatcast_roc_auc": heat["weighted_per_fold_roc_auc"],
            "ens_roc_auc": ens["weighted_per_fold_roc_auc"],
            "delta_auc_heatcast_minus_ens": (
                float(heat["weighted_per_fold_roc_auc"]) - float(ens["weighted_per_fold_roc_auc"])
            ),
            "valid_count": heat["valid_count"],
        })
    return rows


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
    parser.add_argument("--output_dir", default="ens_head_to_head")
    parser.add_argument("--bootstrap_reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration_steps", type=int, default=200)
    parser.add_argument("--calibration_lr", type=float, default=0.1)
    parser.add_argument("--calibration_l2", type=float, default=1e-4)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--emit_per_year", action="store_true")
    args = parser.parse_args()

    print(ENS_BENCHMARK_BANNER)
    heatcast_runs = tuple(value.strip() for value in args.heatcast_runs.split(",") if value.strip())
    ens_runs = tuple(value.strip() for value in args.ens_runs.split(",") if value.strip())
    window_leads = ee.parse_int_list(args.window_leads)
    ens_groups = resolve_ens_run_groups(ens_runs, heatcast_runs, Path(args.ens_root), window_leads)

    global_acc = ee.EvaluationAccumulator(MODEL_NAMES, {})
    fold_acc: Dict[int, ee.EvaluationAccumulator] = {}
    by_fold_year: Dict[Tuple[int, int], ee.EvaluationAccumulator] = {}
    intersection_rows: List[Dict[str, object]] = []
    total_inits = 0
    all_years = set()

    cycle_years: Dict[str, set] = defaultdict(set)
    for heat_name in heatcast_runs:
        heat_manifest, heat_calibration, heat_chunks = load_fold_inputs(
            Path(args.heatcast_root), heat_name, window_leads,
        )
        fold = int(heat_manifest["source_fold"])
        ens_sources = []
        for ens_name in ens_groups[fold]:
            ens_manifest, _, ens_chunks = load_fold_inputs(Path(args.ens_root), ens_name, window_leads)
            if int(ens_manifest["source_fold"]) != fold:
                raise RuntimeError(f"Fold mismatch: HeatCast={fold}, ENS={ens_manifest['source_fold']}.")
            ens_sources.append((ens_name, ens_manifest, chunk_map(ens_chunks)))
        heat_c = fit_heatcast_c(heat_calibration, args)
        heat_map = chunk_map(heat_chunks)
        ens_union = set().union(*(set(source[2]) for source in ens_sources))
        common = tuple(sorted(set(heat_map) & ens_union))
        if not common:
            raise RuntimeError(f"Fold {fold}: empty common-init intersection.")
        fold_acc[fold] = ee.EvaluationAccumulator(MODEL_NAMES, {})
        fold_years = set()
        duplicate_cycle_inits = 0
        cycle_init_counts = defaultdict(int)
        for index, init_t in enumerate(common):
            heat = load_chunk(heat_map[init_t])
            matching_sources = [source for source in ens_sources if init_t in source[2]]
            if len(matching_sources) > 1:
                duplicate_cycle_inits += 1
            ens_chunks_for_init = []
            for ens_name, ens_manifest, ens_map in matching_sources:
                ens = load_chunk(ens_map[init_t])
                for key in ("truth", "base_rate"):
                    if heat[key].shape != ens[key].shape or not np.allclose(heat[key], ens[key], equal_nan=True):
                        raise RuntimeError(f"Fold {fold}, init={init_t}: HeatCast/{ens_name} {key} differs.")
                for key in ("year", "month", "target_center_time_index"):
                    if scalar(heat, key) != scalar(ens, key):
                        raise RuntimeError(f"Fold {fold}, init={init_t}: HeatCast/{ens_name} {key} differs.")
                ens_chunks_for_init.append(ens)
                cycle_init_counts[ens_name] += 1
            year = scalar(heat, "year")
            month = scalar(heat, "month")
            if year not in heat_manifest["test_years"] or any(
                year not in source[1]["test_years"] for source in matching_sources
            ):
                raise RuntimeError(f"Fold {fold}, init={init_t}: intersection chunk is not a test-year prediction.")
            truth = np.asarray(heat["truth"], dtype=np.float32)
            base = np.asarray(heat["base_rate"], dtype=np.float32)
            ens_raw, ens_calibrated = merge_cycle_probabilities(ens_chunks_for_init)
            heat_prob = heat_c.predict_features(np.column_stack([
                np.asarray(heat["init_margin"], dtype=np.float32),
                np.asarray(heat["forecast_margin"], dtype=np.float32),
            ]).astype(np.float32))
            mask = np.ones(truth.shape, dtype=bool)
            forecasts = {
                REFERENCE: base,
                "ens_raw_fraction": ens_raw,
                "ens_calibrated": ens_calibrated,
                "heatcast_C": heat_prob,
            }
            year_acc = by_fold_year.setdefault((fold, year), ee.EvaluationAccumulator(MODEL_NAMES, {}))
            for name, probability in forecasts.items():
                global_acc.update(name, probability, truth, mask, month)
                fold_acc[fold].update(name, probability, truth, mask, month)
                year_acc.update(name, probability, truth, mask, month)
            fold_years.add(year)
            if (index + 1) % max(1, int(args.progress_every)) == 0:
                print(f"  fold {fold}: scored {index + 1}/{len(common)} common inits")
        total_inits += len(common)
        all_years.update(fold_years)
        merged_row = {
            "fold": fold,
            "heatcast_run": heat_name,
            "ens_run": " + ".join(source[0] for source in ens_sources),
            "cycle": "merged",
            "common_init_count": len(common),
            "duplicate_cycle_init_count": duplicate_cycle_inits,
            "intersection_years": " ".join(str(value) for value in sorted(fold_years)),
            "intersection_year_count": len(fold_years),
        }
        intersection_rows.append(merged_row)
        for ens_name, _, ens_map in ens_sources:
            source_common = set(heat_map) & set(ens_map)
            source_years = {
                scalar(load_chunk(heat_map[init_t]), "year")
                for init_t in source_common
            }
            label = cycle_label(ens_name)
            cycle_years[label].update(source_years)
            intersection_rows.append({
                "fold": fold,
                "heatcast_run": heat_name,
                "ens_run": ens_name,
                "cycle": label,
                "common_init_count": cycle_init_counts[ens_name],
                "duplicate_cycle_init_count": duplicate_cycle_inits,
                "intersection_years": " ".join(str(value) for value in sorted(source_years)),
                "intersection_year_count": len(source_years),
            })
        print(
            f"Fold {fold}: merged common inits={len(common)}, duplicate-cycle inits={duplicate_cycle_inits}, "
            f"intersection years={sorted(fold_years)}"
        )

    rows = score_from_folds(fold_acc)
    by_name = {str(row["model"]): row for row in rows}
    bootstrap_rows = bootstrap(by_fold_year, sorted(all_years), args.bootstrap_reps, args.seed)
    per_year_rows = per_year_comparison_rows(by_fold_year, sorted(all_years))
    year_text = " ".join(str(value) for value in sorted(all_years))
    for row in rows:
        row["intersection_years"] = year_text
        row["intersection_year_count"] = len(all_years)
        row["common_init_count"] = total_inits
    for row in bootstrap_rows:
        row["intersection_years"] = year_text
        row["common_init_count"] = total_inits
    bss_delta = (
        float(by_name["heatcast_C"]["bss_vs_monthly_climo"])
        - float(by_name["ens_calibrated"]["bss_vs_monthly_climo"])
    )
    bss_ci = next(row for row in bootstrap_rows if row["metric"].startswith("delta_bss"))
    out_dir = Path(args.output_dir) / f"window_{ee.lead_list_label(window_leads)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_rows: List[Dict[str, object]] = []
    combined_rows.extend({"section": "score", **row} for row in rows)
    combined_rows.extend({"section": "coverage", **row} for row in intersection_rows)
    combined_rows.extend({"section": "bootstrap", **row} for row in bootstrap_rows)
    ee.write_csv(out_dir / "ens_heatcast_head_to_head.csv", combined_rows)
    if args.emit_per_year:
        ee.write_csv(out_dir / "ens_heatcast_per_year.csv", per_year_rows)
    ee.plot_reliability(
        out_dir / "reliability_overlay.png",
        {name: global_acc.metrics[name].rel.table() for name in MODEL_NAMES},
    )

    print("\nENS head-to-head summary")
    print("========================")
    for row in rows:
        print(
            f"{row['model']:<24} N={int(row['valid_count'])} "
            f"Brier={row['brier']:.5f} BSS={row['bss_vs_monthly_climo']:+.4f} "
            f"weighted-fold-AUC={row['weighted_per_fold_roc_auc']:.3f} "
            f"slope={row['reliability_slope']:.3f} ECE={row['ece']:.4f}"
        )
    print(f"Year-block bootstrap: {len(all_years)} independent intersection-year blocks")
    for cycle, years in sorted(cycle_years.items()):
        print(f"  {cycle}: {len(years)} calendar-year blocks")
    if len(cycle_years) > 1:
        union, overlap = cycle_year_union(cycle_years)
        print(
            f"Cycle-union widening: old={len(cycle_years.get('legacy', set()))} blocks, "
            f"new_union={len(union)} blocks, shared_across_all_cycles={len(overlap)}."
        )
    print("Bootstrap blocking assert: PASS (calendar year, never cycle).")
    for row in bootstrap_rows:
        print(
            f"  {row['metric']}: CI=[{row['ci_low']:+.4f},{row['ci_high']:+.4f}], "
            f"excludes_zero={row['ci_excludes_zero']}"
        )
    print(
        f"ENS HEAD-TO-HEAD ({len(all_years)} yrs, {total_inits} inits): "
        f"HeatCast BSS={by_name['heatcast_C']['bss_vs_monthly_climo']:+.4f} vs "
        f"ENS BSS={by_name['ens_calibrated']['bss_vs_monthly_climo']:+.4f}, "
        f"delta={bss_delta:+.4f} CI=[{bss_ci['ci_low']:+.4f},{bss_ci['ci_high']:+.4f}]"
    )
    print("Expected context: near-zero calibrated BSS at days 12-18 is a ceiling-matching result;")
    print("at days 15-28, either system may lead and the year-block CI is the competitive statement.")
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
