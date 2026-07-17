#!/usr/bin/env python3
"""Stitch saved fold-safe exceedance arrays and score A-F across all held-out years."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

import exceedance_eval as ee


REFERENCE_NAME = "pooled_climatology"
BASELINE_NAME = "incremental_A_init_margin"
CANDIDATE_NAMES = (
    "incremental_C_init_plus_forecast",
    "incremental_D_init_forecast_sigma",
    ee.CLIMATOLOGY_ANCHORED_MODEL_NAME,
    ee.NESTED_SHRINKAGE_MODEL_NAME,
)
REQUIRED_MODEL_NAMES = (
    "incremental_A_init_margin",
    "incremental_B_forecast_margin",
    "incremental_C_init_plus_forecast",
    "incremental_D_init_forecast_sigma",
    ee.CLIMATOLOGY_ANCHORED_MODEL_NAME,
    ee.NESTED_SHRINKAGE_MODEL_NAME,
)


def _scalar(data: Mapping[str, np.ndarray], key: str) -> Any:
    return np.asarray(data[key]).item()


def _years(data: Mapping[str, np.ndarray], key: str) -> set[int]:
    return set(int(v) for v in np.atleast_1d(data[key]).astype(int).tolist())


def run_dir(input_root: Path, run_name: str, window_leads: Sequence[int]) -> Path:
    return input_root / run_name / "test" / f"window_{ee.lead_list_label(window_leads)}"


def load_fold_inputs(
    input_root: Path,
    run_name: str,
    window_leads: Sequence[int],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], List[Path]]:
    root = run_dir(input_root, run_name, window_leads)
    manifest_path = root / "incremental_arrays" / "manifest.npz"
    calibration_path = root / "incremental_arrays" / "calibration_pairs.npz"
    chunk_dir = root / "incremental_arrays" / "test_chunks"
    if not manifest_path.exists() or not calibration_path.exists() or not chunk_dir.is_dir():
        raise FileNotFoundError(
            f"Missing stitchable arrays for {run_name}: expected {manifest_path}, "
            f"{calibration_path}, and {chunk_dir}. Rerun src/exceedance_eval.py for this fold with "
            "--incremental_skill_diagnostic --save_incremental_arrays."
        )
    with np.load(manifest_path, allow_pickle=False) as data:
        manifest = {
            "root": root,
            "run_name": str(_scalar(data, "run_name")),
            "source_fold": int(_scalar(data, "source_fold")),
            "target_mode": str(_scalar(data, "target_mode")),
            "window_leads": tuple(int(v) for v in np.atleast_1d(data["window_leads"]).tolist()),
            "train_years": _years(data, "train_years"),
            "calibration_years": _years(data, "calibration_years"),
            "test_years": _years(data, "test_years"),
            "calibration_split": str(_scalar(data, "calibration_split")),
            "eval_split": str(_scalar(data, "eval_split")),
            "sample_count": int(_scalar(data, "sample_count")),
            "valid_cell_count": int(_scalar(data, "valid_cell_count")),
        }
    if manifest["run_name"] != run_name:
        raise RuntimeError(
            f"{run_name}: manifest run_name={manifest['run_name']!r} does not match requested run."
        )
    if manifest["target_mode"] != "window":
        raise RuntimeError(
            f"{run_name}: target_mode={manifest['target_mode']!r}; pooled stitcher requires window mode."
        )
    if manifest["window_leads"] != tuple(int(v) for v in window_leads):
        raise RuntimeError(
            f"{run_name}: saved window leads {manifest['window_leads']} do not match requested "
            f"{tuple(int(v) for v in window_leads)}."
        )
    if manifest["calibration_split"] != "val" or manifest["eval_split"] != "test":
        raise RuntimeError(
            f"{run_name}: expected calibration_split=val and eval_split=test, got "
            f"{manifest['calibration_split']}/{manifest['eval_split']}."
        )
    with np.load(calibration_path, allow_pickle=False) as data:
        calibration = {
            "init_margin": np.asarray(data["init_margin"], dtype=np.float32),
            "forecast_margin": np.asarray(data["forecast_margin"], dtype=np.float32),
            "model_sigma": np.asarray(data["model_sigma"], dtype=np.float32),
            "truth": np.asarray(data["truth"], dtype=np.float32),
            "base_rate": np.asarray(data["base_rate"], dtype=np.float32),
            "year": np.asarray(data["year"], dtype=np.int16),
            "source_fold": np.full(
                np.asarray(data["truth"]).size,
                int(_scalar(data, "source_fold")),
                dtype=np.int16,
            ),
        }
    calibration_sizes = {key: int(value.size) for key, value in calibration.items()}
    if len(set(calibration_sizes.values())) != 1:
        raise RuntimeError(f"{run_name}: inconsistent calibration array lengths: {calibration_sizes}")
    if not set(int(v) for v in np.unique(calibration["year"])).issubset(manifest["calibration_years"]):
        raise RuntimeError(f"{run_name}: calibration pairs contain years outside the manifest calibration years.")
    if not np.all(np.isfinite(calibration["model_sigma"])):
        raise RuntimeError(f"{run_name}: calibration model_sigma contains non-finite values; Model D requires sigma.")
    chunks = sorted(chunk_dir.glob("sample_*.npz"))
    if len(chunks) != manifest["sample_count"]:
        raise RuntimeError(
            f"{run_name}: manifest sample_count={manifest['sample_count']} but found {len(chunks)} chunks."
        )
    return manifest, calibration, chunks


def audit_folds(manifests: Sequence[Mapping[str, Any]], expected_folds: int) -> List[Dict[str, object]]:
    if len(manifests) != int(expected_folds):
        raise RuntimeError(f"Expected {expected_folds} folds, got {len(manifests)}.")
    folds = [int(m["source_fold"]) for m in manifests]
    if len(set(folds)) != len(folds):
        raise RuntimeError(f"Duplicate source folds: {folds}")
    test_owner: Dict[int, int] = {}
    rows: List[Dict[str, object]] = []
    for manifest in manifests:
        fold = int(manifest["source_fold"])
        train = set(manifest["train_years"])
        calibration = set(manifest["calibration_years"])
        test = set(manifest["test_years"])
        if train & test:
            raise RuntimeError(f"Fold {fold}: train/test year overlap: {sorted(train & test)}")
        if calibration & test:
            raise RuntimeError(f"Fold {fold}: own calibration/test year overlap: {sorted(calibration & test)}")
        for year in test:
            if year in test_owner:
                raise RuntimeError(
                    f"Test year {year} contributed by folds {test_owner[year]} and {fold}; "
                    "pooled test predictions must be unique."
                )
            test_owner[year] = fold
        rows.append({
            "source_fold": fold,
            "run_name": manifest["run_name"],
            "train_years": " ".join(str(v) for v in sorted(train)),
            "calibration_years": " ".join(str(v) for v in sorted(calibration)),
            "test_years": " ".join(str(v) for v in sorted(test)),
            "own_train_test_overlap": len(train & test),
            "own_calibration_test_overlap": len(calibration & test),
            "saved_samples": manifest["sample_count"],
            "saved_valid_cells": manifest["valid_cell_count"],
        })
    return rows


def concatenate_calibration(calibrations: Sequence[Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = ("init_margin", "forecast_margin", "model_sigma", "truth", "base_rate", "year", "source_fold")
    return {key: np.concatenate([np.asarray(item[key]) for item in calibrations], axis=0) for key in keys}


def fit_models_from_calibration(
    calibration: Mapping[str, np.ndarray],
    calibration_split: str,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, object]]]:
    features = np.column_stack([
        calibration["init_margin"],
        calibration["forecast_margin"],
        calibration["model_sigma"],
    ]).astype(np.float32)
    models, nested_selection_rows = ee.fit_incremental_skill_models(
        features,
        calibration["truth"],
        calibration["base_rate"],
        calibration["year"],
        calibration_split=calibration_split,
        alpha_grid=ee.parse_float_list(args.incremental_alpha_grid),
        l2_grid=ee.parse_float_list(args.incremental_l2_grid),
        steps=int(args.calibration_steps),
        lr=float(args.calibration_lr),
        l2=float(args.calibration_l2),
    )
    missing_models = [name for name in REQUIRED_MODEL_NAMES if name not in models]
    if missing_models:
        raise RuntimeError(f"A-F fit is incomplete for {calibration_split}; missing models: {missing_models}")
    return models, nested_selection_rows


def fit_cross_fitted_models(
    manifests: Sequence[Mapping[str, Any]],
    calibrations: Sequence[Mapping[str, np.ndarray]],
    args: argparse.Namespace,
) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, object]], List[Dict[str, object]]]:
    """Fit each outer fold's calibrators on its own held-out validation years only."""
    models_by_fold: Dict[int, Dict[str, Any]] = {}
    selection_rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    for manifest, calibration in zip(manifests, calibrations):
        fold = int(manifest["source_fold"])
        source_folds = set(int(v) for v in np.unique(calibration["source_fold"]))
        fit_years = set(int(v) for v in np.unique(calibration["year"]))
        expected_fit_years = set(manifest["calibration_years"])
        test_years = set(manifest["test_years"])
        train_years = set(manifest["train_years"])
        if source_folds != {fold}:
            raise RuntimeError(
                f"Fold {fold}: cross-fitted calibration rows came from source folds {sorted(source_folds)}."
            )
        if not fit_years.issubset(expected_fit_years):
            raise RuntimeError(
                f"Fold {fold}: calibration years outside its validation split: "
                f"{sorted(fit_years - expected_fit_years)}"
            )
        if fit_years & test_years:
            raise RuntimeError(
                f"Fold {fold}: cross-fitted calibration/test overlap: {sorted(fit_years & test_years)}"
            )
        if train_years & test_years:
            raise RuntimeError(f"Fold {fold}: source model train/test overlap: {sorted(train_years & test_years)}")
        models, nested_rows = fit_models_from_calibration(
            calibration,
            calibration_split=f"fold{fold}_own_validation",
            args=args,
        )
        models_by_fold[fold] = models
        selection_rows.extend({"source_fold": fold, **row} for row in nested_rows)
        audit_rows.append({
            "source_fold": fold,
            "fit_source": "same_fold_validation_only",
            "fit_years": " ".join(str(v) for v in sorted(fit_years)),
            "test_years": " ".join(str(v) for v in sorted(test_years)),
            "source_model_train_years": " ".join(str(v) for v in sorted(train_years)),
            "fit_test_overlap": len(fit_years & test_years),
            "source_model_train_test_overlap": len(train_years & test_years),
            "fit_rows": int(calibration["truth"].size),
        })
        print(
            f"Cross-fitted fold {fold}: fit A-F on {len(fit_years)} own validation years "
            f"({int(calibration['truth'].size)} rows), score only {len(test_years)} test years."
        )
    return models_by_fold, selection_rows, audit_rows


def predict_models(
    models: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    fields = {
        "init_margin": np.asarray(arrays["init_margin"], dtype=np.float32),
        "forecast_margin": np.asarray(arrays["forecast_margin"], dtype=np.float32),
        "predicted_sigma": np.asarray(arrays["model_sigma"], dtype=np.float32),
    }
    base_rate = np.asarray(arrays["base_rate"], dtype=np.float32)
    out: Dict[str, np.ndarray] = {}
    for name, model in models.items():
        feature_matrix = np.column_stack([fields[feature] for feature in model.feature_names]).astype(np.float32)
        valid = np.all(np.isfinite(feature_matrix), axis=1) & np.isfinite(base_rate)
        prob = np.full(base_rate.shape, np.nan, dtype=np.float32)
        if isinstance(model, (ee.ClimatologyAnchoredLogisticCalibrator, ee.NestedYearShrinkageCalibrator)):
            prob[valid] = model.predict_features(feature_matrix[valid], base_rate[valid])
        else:
            prob[valid] = model.predict_features(feature_matrix[valid])
        out[name] = prob
    return out


def bootstrap_candidates(
    by_year: Mapping[int, ee.EvaluationAccumulator],
    candidate_names: Iterable[str],
    reps: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    per_year: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    for index, candidate in enumerate(candidate_names):
        candidate_per_year, candidate_summary = ee.year_block_bootstrap_incremental_comparison(
            by_year,
            REFERENCE_NAME,
            BASELINE_NAME,
            candidate,
            reps=reps,
            seed=seed + 1009 * index,
        )
        for row in candidate_per_year:
            per_year.append({"candidate_model": candidate, **row})
        for row in candidate_summary:
            lo = float(row["ci_2.5"])
            hi = float(row["ci_97.5"])
            summaries.append({
                **row,
                "ci_excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0.0 or hi < 0.0)),
            })
    return per_year, summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="Comma-separated fold run names.")
    parser.add_argument("--window_leads", default="12,13,14,15,16,17,18")
    parser.add_argument("--calibrator", choices=["platt"], default="platt")
    parser.add_argument(
        "--fit_mode",
        choices=["pooled_once", "cross_fitted"],
        default="pooled_once",
        help=(
            "pooled_once fits A-F once on all saved validation rows; cross_fitted fits each "
            "fold's A-F calibrators on that fold's own validation rows and scores only its test rows."
        ),
    )
    parser.add_argument("--input_root", default="exceedance_eval_incremental")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--expected_folds", type=int, default=5)
    parser.add_argument("--expected_year_blocks", type=int, default=43)
    parser.add_argument("--bootstrap_reps", type=int, default=5000)
    parser.add_argument("--calibration_steps", type=int, default=200)
    parser.add_argument("--calibration_lr", type=float, default=0.1)
    parser.add_argument("--calibration_l2", type=float, default=1e-4)
    parser.add_argument("--incremental_alpha_grid", default="0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--incremental_l2_grid", default="0,0.0001,0.001,0.01,0.1,1.0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    runs = tuple(v.strip() for v in args.runs.split(",") if v.strip())
    window_leads = ee.parse_int_list(args.window_leads)
    input_root = Path(args.input_root)
    default_output_name = (
        f"stitched_window_{ee.lead_list_label(window_leads)}"
        if args.fit_mode == "pooled_once"
        else f"stitched_cross_fitted_window_{ee.lead_list_label(window_leads)}"
    )
    output_dir = ee.ensure_dir(Path(args.output_dir) if args.output_dir else input_root / default_output_name)

    loaded = [load_fold_inputs(input_root, run_name, window_leads) for run_name in runs]
    manifests = [item[0] for item in loaded]
    calibrations = [item[1] for item in loaded]
    chunks_by_fold = [(item[0], item[2]) for item in loaded]
    audit_rows = audit_folds(manifests, args.expected_folds)
    test_years = set().union(*(set(m["test_years"]) for m in manifests))
    calibration_years = set().union(*(set(m["calibration_years"]) for m in manifests))
    if len(test_years) != int(args.expected_year_blocks):
        raise RuntimeError(
            f"Expected {args.expected_year_blocks} independent pooled test-year blocks, got {len(test_years)}: "
            f"{sorted(test_years)}"
        )
    cross_fold_overlap = sorted(test_years & calibration_years)
    print(f"Held-out monotonic setting: {args.calibrator} forced.")
    print("Fold-safety audit: PASS (each fold's contributed test years are absent from its own train/calibration years)")
    print(f"Pooled test-year blocks: {len(test_years)} ({min(test_years)}-{max(test_years)})")
    print(
        "Cross-fold calendar-year overlap audit: "
        f"{len(cross_fold_overlap)} pooled test years also occur in another fold's calibration set."
    )
    if cross_fold_overlap and args.fit_mode == "pooled_once":
        print(
            "WARNING: pooled-once calibration is not globally year-disjoint because the five validation-year "
            "unions rotate across the same record. Interpret this requested pooled-once result as a cross-fold "
            "calibration analysis, not a strictly untouched 43-year final test."
        )

    fit_audit_rows: List[Dict[str, object]] = []
    if args.fit_mode == "cross_fitted":
        models_by_fold, nested_selection_rows, fit_audit_rows = fit_cross_fitted_models(
            manifests,
            calibrations,
            args,
        )
        print(
            "Cross-fitted leakage audit: PASS (each fold's calibrators use only that same fold's "
            "validation years; its test years are untouched until scoring)."
        )
    else:
        pooled_cal = concatenate_calibration(calibrations)
        own_overlap_count = 0
        for manifest in manifests:
            fold = int(manifest["source_fold"])
            rows = pooled_cal["source_fold"] == fold
            own_overlap_count += int(np.sum(np.isin(pooled_cal["year"][rows], list(manifest["test_years"]))))
        if own_overlap_count:
            raise RuntimeError(f"Found {own_overlap_count} own-fold test-year calibration rows.")
        pooled_models, nested_selection_rows = fit_models_from_calibration(
            pooled_cal,
            calibration_split="pooled_fold_calibration",
            args=args,
        )
        models_by_fold = {int(manifest["source_fold"]): pooled_models for manifest in manifests}
    model_names = list(next(iter(models_by_fold.values())))
    inconsistent_model_sets = {
        fold: sorted(set(fold_models) ^ set(model_names))
        for fold, fold_models in models_by_fold.items()
        if set(fold_models) != set(model_names)
    }
    if inconsistent_model_sets:
        raise RuntimeError(f"A-F model sets differ across folds: {inconsistent_model_sets}")
    pooled_acc = ee.EvaluationAccumulator([REFERENCE_NAME, *model_names], {})
    by_year: Dict[int, ee.EvaluationAccumulator] = defaultdict(
        lambda: ee.EvaluationAccumulator([REFERENCE_NAME, *model_names], {})
    )
    total_chunks = 0
    total_cells = 0
    fold_streamed_cells: Counter[int] = Counter()
    fold_streamed_samples: Counter[int] = Counter()
    for manifest, chunks in chunks_by_fold:
        fold = int(manifest["source_fold"])
        models = models_by_fold[fold]
        for chunk_path in chunks:
            with np.load(chunk_path, allow_pickle=False) as data:
                source_fold = int(_scalar(data, "source_fold"))
                year = int(_scalar(data, "year"))
                month = int(_scalar(data, "month"))
                if source_fold != fold:
                    raise RuntimeError(f"{chunk_path}: source_fold={source_fold}, expected {fold}.")
                if year not in manifest["test_years"]:
                    raise RuntimeError(f"{chunk_path}: year {year} is not a test year for fold {fold}.")
                arrays = {
                    "init_margin": np.asarray(data["init_margin"], dtype=np.float32),
                    "forecast_margin": np.asarray(data["forecast_margin"], dtype=np.float32),
                    "model_sigma": np.asarray(data["model_sigma"], dtype=np.float32),
                    "truth": np.asarray(data["truth"], dtype=np.float32),
                    "base_rate": np.asarray(data["base_rate"], dtype=np.float32),
                }
            array_sizes = {name: int(value.size) for name, value in arrays.items()}
            if len(set(array_sizes.values())) != 1:
                raise RuntimeError(f"{chunk_path}: inconsistent saved array lengths: {array_sizes}")
            if not np.all(np.isfinite(arrays["model_sigma"])):
                raise RuntimeError(f"{chunk_path}: model_sigma contains non-finite values; Model D cannot be pooled.")
            n = arrays["truth"].size
            mask = np.ones(n, dtype=bool)
            predictions = predict_models(models, arrays)
            pooled_acc.update(REFERENCE_NAME, arrays["base_rate"], arrays["truth"], mask, month)
            by_year[year].update(REFERENCE_NAME, arrays["base_rate"], arrays["truth"], mask, month)
            for name, prob in predictions.items():
                pooled_acc.update(name, prob, arrays["truth"], mask, month)
                by_year[year].update(name, prob, arrays["truth"], mask, month)
            total_chunks += 1
            total_cells += n
            fold_streamed_cells[fold] += n
            fold_streamed_samples[fold] += 1
            if total_chunks % 250 == 0:
                print(f"  streamed {total_chunks} test chunks, {total_cells} valid cells")

    for manifest in manifests:
        fold = int(manifest["source_fold"])
        if fold_streamed_samples[fold] != int(manifest["sample_count"]):
            raise RuntimeError(
                f"Fold {fold}: streamed {fold_streamed_samples[fold]} samples, "
                f"manifest reports {manifest['sample_count']}."
            )
        if fold_streamed_cells[fold] != int(manifest["valid_cell_count"]):
            raise RuntimeError(
                f"Fold {fold}: streamed {fold_streamed_cells[fold]} valid cells, "
                f"manifest reports {manifest['valid_cell_count']}."
            )
    if set(by_year) != test_years:
        raise RuntimeError(
            f"Saved test chunks cover years {sorted(by_year)}, expected pooled years {sorted(test_years)}."
        )
    summary_rows = pooled_acc.summary_rows(REFERENCE_NAME)
    summary_by_model = {row["model"]: row for row in summary_rows}
    reference_count = int(summary_by_model[REFERENCE_NAME]["valid_count"])
    count_mismatches = {
        name: int(summary_by_model[name]["valid_count"])
        for name in REQUIRED_MODEL_NAMES
        if int(summary_by_model[name]["valid_count"]) != reference_count
    }
    if count_mismatches:
        raise RuntimeError(
            "Pooled A-F valid-cell counts differ from pooled climatology: "
            f"reference={reference_count}, mismatches={count_mismatches}"
        )
    per_year_rows, bootstrap_rows = bootstrap_candidates(
        by_year,
        [name for name in CANDIDATE_NAMES if name in model_names],
        reps=int(args.bootstrap_reps),
        seed=int(args.seed) + 7001,
    )
    ee.write_csv(output_dir / "pooled_incremental_results.csv", summary_rows)
    ee.write_csv(output_dir / "pooled_bootstrap_vs_A.csv", bootstrap_rows)
    ee.write_csv(output_dir / "pooled_per_year_metrics_vs_A.csv", per_year_rows)
    ee.write_csv(output_dir / "pooled_fold_audit.csv", audit_rows)
    if fit_audit_rows:
        ee.write_csv(output_dir / "cross_fitted_calibration_audit.csv", fit_audit_rows)
    ee.write_csv(
        output_dir / "pooled_cross_fold_overlap_audit.csv",
        [{
            "pooled_test_year_blocks": len(test_years),
            "pooled_calibration_year_blocks": len(calibration_years),
            "cross_fold_calendar_year_overlap_count": len(cross_fold_overlap),
            "cross_fold_calendar_year_overlap": " ".join(str(v) for v in cross_fold_overlap),
            "interpretation": (
                (
                    "Cross-fitted scoring is leakage-clean because each fold uses only its own validation rows."
                    if args.fit_mode == "cross_fitted"
                    else "Pooled-once cross-fold calibration analysis; not a globally untouched final test."
                )
                if cross_fold_overlap else "Calibration and test year unions are globally disjoint."
            ),
        }],
    )
    ee.write_csv(output_dir / "pooled_nested_year_selection.csv", nested_selection_rows)
    coefficient_models = (
        [(f"fold{fold}", fold_models) for fold, fold_models in sorted(models_by_fold.items())]
        if args.fit_mode == "cross_fitted"
        else [("pooled_once", next(iter(models_by_fold.values())))]
    )
    ee.write_csv(
        output_dir / "pooled_incremental_coefficients.csv",
        [
            {"fit_scope": fit_scope, "diagnostic_model": name, **row}
            for fit_scope, fold_models in coefficient_models
            for name, model in fold_models.items()
            for row in model.coefficient_rows()
        ],
    )

    print(f"\nStitched A-through-F incremental table ({args.fit_mode})")
    print("=" * 56)
    for row in summary_rows:
        if row["model"] == REFERENCE_NAME:
            continue
        print(
            f"{row['model']:40s} N={int(row['valid_count'])} "
            f"BSS={row['bss_vs_monthly_climo']:+.4f} ROC-AUC={row['roc_auc']:.4f} "
            f"slope={row['reliability_slope']:.3f} ECE={row['ece']:.4f}"
        )
    print("\nWhole-year bootstrap versus Model A")
    print("===================================")
    print(f"Independent year blocks: {len(by_year)}")
    for row in bootstrap_rows:
        excludes = "YES" if row["ci_excludes_zero"] else "NO"
        print(
            f"{row['candidate_model']:40s} {row['metric']}: "
            f"estimate={float(row['estimate']):+.4f}, "
            f"95% CI=[{float(row['ci_2.5']):+.4f}, {float(row['ci_97.5']):+.4f}], "
            f"excludes_zero={excludes}"
        )
    c_auc = next(
        row for row in bootstrap_rows
        if row["candidate_model"] == "incremental_C_init_plus_forecast"
        and row["metric"] == "delta_roc_auc_candidate_minus_baseline"
    )
    print(
        f"\nHeadline ({args.fit_mode}): Model C stitched delta-ROC-AUC CI versus A "
        f"{'EXCLUDES' if c_auc['ci_excludes_zero'] else 'DOES NOT EXCLUDE'} zero."
    )
    print(f"Saved stitched outputs to: {output_dir}")


if __name__ == "__main__":
    main()
