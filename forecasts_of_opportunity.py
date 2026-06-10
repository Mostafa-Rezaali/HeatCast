#!/usr/bin/env python3
"""Stream fold-safe saved exceedance arrays and verify forecasts of opportunity."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


HIST_BINS = 512
EXPECTED_YEARS = set(range(1981, 2024))
TOP_CONFIDENCE_PERCENTILE = 90
_AUC_FROM_HIST = None


def parse_int_list(text: str) -> Tuple[int, ...]:
    return tuple(int(value.strip()) for value in str(text).split(",") if value.strip())


def parse_str_list(text: str) -> Tuple[str, ...]:
    return tuple(value.strip() for value in str(text).split(",") if value.strip())


def percentile_edges(values: np.ndarray, percentiles: Sequence[int]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise RuntimeError("Cannot fit opportunity boundaries from an empty finite calibration array.")
    return np.percentile(values, np.asarray(percentiles, dtype=np.float64)).astype(np.float64)


def fit_boundaries(
    calibration: Mapping[str, np.ndarray],
    model_c: Any,
    confidence_percentiles: Sequence[int],
) -> Dict[str, np.ndarray]:
    """Fit all selection boundaries from calibration arrays only."""
    features = np.column_stack([
        np.asarray(calibration["init_margin"], dtype=np.float32),
        np.asarray(calibration["forecast_margin"], dtype=np.float32),
    ])
    base_rate = np.asarray(calibration["base_rate"], dtype=np.float32)
    probabilities = np.asarray(model_c.predict_features(features), dtype=np.float32)
    confidence = np.abs(probabilities - base_rate)
    return {
        "confidence_percentiles": np.asarray(confidence_percentiles, dtype=np.int16),
        "confidence_edges": percentile_edges(confidence, confidence_percentiles),
        "sigma_edges": percentile_edges(np.asarray(calibration["model_sigma"]), range(10, 100, 10)),
        "forecast_margin_edges": percentile_edges(
            np.abs(np.asarray(calibration["forecast_margin"])), range(10, 100, 10)
        ),
    }


def assign_deciles(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    out = np.full(values.shape, -1, dtype=np.int8)
    valid = np.isfinite(values)
    out[valid] = np.digitize(values[valid], np.asarray(edges), right=False).astype(np.int8)
    return out


def roc_auc_from_hist(pos_hist: np.ndarray, neg_hist: np.ndarray) -> float:
    if _AUC_FROM_HIST is not None:
        return float(_AUC_FROM_HIST(pos_hist, neg_hist))
    pos_total = float(np.sum(pos_hist))
    neg_total = float(np.sum(neg_hist))
    if pos_total <= 0.0 or neg_total <= 0.0:
        return float("nan")
    tp = np.cumsum(np.asarray(pos_hist, dtype=np.float64)[::-1])
    fp = np.cumsum(np.asarray(neg_hist, dtype=np.float64)[::-1])
    tpr = np.r_[0.0, tp / pos_total, 1.0]
    fpr = np.r_[0.0, fp / neg_total, 1.0]
    return float(np.trapz(tpr, fpr))


@dataclass
class OpportunityStats:
    count: float = 0.0
    event_count: float = 0.0
    pred_sum: float = 0.0
    brier_sum: float = 0.0
    climo_brier_sum: float = 0.0
    hist_count: np.ndarray = field(default_factory=lambda: np.zeros(HIST_BINS, dtype=np.float64))
    hist_pred_sum: np.ndarray = field(default_factory=lambda: np.zeros(HIST_BINS, dtype=np.float64))
    hist_pos: np.ndarray = field(default_factory=lambda: np.zeros(HIST_BINS, dtype=np.float64))

    def update(
        self,
        probability: np.ndarray,
        truth: np.ndarray,
        base_rate: np.ndarray,
        selection: Optional[np.ndarray] = None,
    ) -> None:
        p = np.asarray(probability).reshape(-1)
        y = np.asarray(truth).reshape(-1)
        base = np.asarray(base_rate).reshape(-1)
        if selection is not None:
            selected = np.asarray(selection, dtype=bool).reshape(-1)
            p = p[selected]
            y = y[selected]
            base = base[selected]
        valid = np.isfinite(p) & np.isfinite(y) & np.isfinite(base)
        if not np.any(valid):
            return
        p = np.clip(p[valid].astype(np.float64), 0.0, 1.0)
        y = y[valid].astype(np.float64)
        base = np.clip(base[valid].astype(np.float64), 0.0, 1.0)
        index = np.minimum((p * HIST_BINS).astype(np.int64), HIST_BINS - 1)
        self.count += float(p.size)
        self.event_count += float(np.sum(y))
        self.pred_sum += float(np.sum(p))
        self.brier_sum += float(np.sum((p - y) ** 2))
        self.climo_brier_sum += float(np.sum((base - y) ** 2))
        self.hist_count += np.bincount(index, minlength=HIST_BINS)
        self.hist_pred_sum += np.bincount(index, weights=p, minlength=HIST_BINS)
        self.hist_pos += np.bincount(index, weights=y, minlength=HIST_BINS)

    def add(self, other: "OpportunityStats", weight: float = 1.0) -> None:
        self.count += weight * other.count
        self.event_count += weight * other.event_count
        self.pred_sum += weight * other.pred_sum
        self.brier_sum += weight * other.brier_sum
        self.climo_brier_sum += weight * other.climo_brier_sum
        self.hist_count += weight * other.hist_count
        self.hist_pred_sum += weight * other.hist_pred_sum
        self.hist_pos += weight * other.hist_pos

    def metrics(self) -> Dict[str, float]:
        if self.count <= 0:
            return {
                key: float("nan")
                for key in (
                    "n", "event_rate", "mean_probability", "brier", "brier_climo",
                    "bss_unconditional", "brier_conditional", "bss_conditional",
                    "roc_auc", "reliability_slope", "ece",
                )
            }
        event_rate = self.event_count / self.count
        brier = self.brier_sum / self.count
        brier_climo = self.climo_brier_sum / self.count
        brier_conditional = event_rate * (1.0 - event_rate)
        occupied = self.hist_count > 0
        mean_pred = np.divide(
            self.hist_pred_sum, self.hist_count,
            out=np.full(HIST_BINS, np.nan), where=occupied,
        )
        obs_freq = np.divide(
            self.hist_pos, self.hist_count,
            out=np.full(HIST_BINS, np.nan), where=occupied,
        )
        if np.sum(occupied) >= 2:
            x0 = np.average(mean_pred[occupied], weights=self.hist_count[occupied])
            y0 = np.average(obs_freq[occupied], weights=self.hist_count[occupied])
            denom = np.sum(self.hist_count[occupied] * (mean_pred[occupied] - x0) ** 2)
            slope = (
                float(np.sum(
                    self.hist_count[occupied]
                    * (mean_pred[occupied] - x0)
                    * (obs_freq[occupied] - y0)
                ) / denom)
                if denom > 0 else float("nan")
            )
        else:
            slope = float("nan")
        ece = float(np.sum(
            self.hist_count[occupied] * np.abs(mean_pred[occupied] - obs_freq[occupied])
        ) / self.count)
        return {
            "n": self.count,
            "event_rate": event_rate,
            "mean_probability": self.pred_sum / self.count,
            "brier": brier,
            "brier_climo": brier_climo,
            "bss_unconditional": 1.0 - brier / brier_climo if brier_climo > 0 else float("nan"),
            "brier_conditional": brier_conditional,
            "bss_conditional": 1.0 - brier / brier_conditional if brier_conditional > 0 else float("nan"),
            "roc_auc": roc_auc_from_hist(self.hist_pos, self.hist_count - self.hist_pos),
            "reliability_slope": slope,
            "ece": ece,
        }

    def histogram_rows(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.hist_count, self.hist_pred_sum, self.hist_pos


def aggregate_year_stats(
    by_year: Mapping[Tuple[str, str, int], OpportunityStats],
    axis: str,
    stratum: str,
    year_weights: Mapping[int, int],
) -> OpportunityStats:
    result = OpportunityStats()
    for year, weight in year_weights.items():
        stats = by_year.get((axis, stratum, int(year)))
        if stats is not None and weight:
            result.add(stats, float(weight))
    return result


def year_block_bootstrap(
    by_year: Mapping[Tuple[str, str, int], OpportunityStats],
    confidence_strata: Sequence[str],
    top_stratum: str,
    years: Sequence[int],
    reps: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    years_array = np.asarray(sorted(int(year) for year in years), dtype=np.int16)
    if years_array.size < 2:
        raise RuntimeError("Year-block bootstrap requires at least two independent years.")
    rng = np.random.default_rng(int(seed))
    values: Dict[str, Dict[str, List[float]]] = {
        stratum: {"bss": [], "auc": []} for stratum in confidence_strata
    }
    values[top_stratum]["delta_bss_vs_pooled"] = []
    for _ in range(int(reps)):
        sampled = rng.choice(years_array, size=years_array.size, replace=True)
        unique, counts = np.unique(sampled, return_counts=True)
        weights = {int(year): int(count) for year, count in zip(unique, counts)}
        replicate_metrics: Dict[str, Dict[str, float]] = {}
        for stratum in confidence_strata:
            metrics = aggregate_year_stats(by_year, "confidence", stratum, weights).metrics()
            replicate_metrics[stratum] = metrics
            values[stratum]["bss"].append(metrics["bss_unconditional"])
            values[stratum]["auc"].append(metrics["roc_auc"])
        values[top_stratum]["delta_bss_vs_pooled"].append(
            replicate_metrics[top_stratum]["bss_unconditional"]
            - replicate_metrics["all"]["bss_unconditional"]
        )
    output: Dict[str, Dict[str, float]] = {}
    for stratum, metrics in values.items():
        output[stratum] = {}
        for name, samples in metrics.items():
            finite = np.asarray(samples, dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            output[stratum][f"{name}_ci_low"] = (
                float(np.percentile(finite, 2.5)) if finite.size else float("nan")
            )
            output[stratum][f"{name}_ci_high"] = (
                float(np.percentile(finite, 97.5)) if finite.size else float("nan")
            )
    return output


def confidence_selections(
    confidence: np.ndarray,
    percentiles: Sequence[int],
    edges: Sequence[float],
) -> List[Tuple[str, np.ndarray]]:
    selections = [("all", np.ones(np.asarray(confidence).shape, dtype=bool))]
    for percentile, edge in zip(percentiles, edges):
        retained = 100 - int(percentile)
        selections.append((
            f"top_{retained}pct_ge_p{int(percentile)}",
            np.asarray(confidence) >= float(edge),
        ))
    return selections


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def histogram_text(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.12g}" for value in np.asarray(values).reshape(-1))


def retained_percent_from_confidence_stratum(stratum: str) -> int:
    match = re.fullmatch(r"top_(\d+)pct_ge_p(\d+)", str(stratum))
    if match is None:
        raise ValueError(f"Invalid confidence stratum label: {stratum!r}")
    retained_percent = int(match.group(1))
    threshold_percentile = int(match.group(2))
    if retained_percent + threshold_percentile != 100:
        raise ValueError(
            f"Inconsistent confidence stratum label {stratum!r}: "
            f"retained {retained_percent}% plus threshold percentile {threshold_percentile} != 100."
        )
    return retained_percent


def update_stratum(
    global_stats: MutableMapping[Tuple[str, str], OpportunityStats],
    by_year: MutableMapping[Tuple[str, str, int], OpportunityStats],
    axis: str,
    stratum: str,
    year: int,
    probability: np.ndarray,
    truth: np.ndarray,
    base_rate: np.ndarray,
    selection: np.ndarray,
) -> None:
    delta = OpportunityStats()
    delta.update(probability, truth, base_rate, selection)
    global_stats[(axis, stratum)].add(delta)
    by_year[(axis, stratum, int(year))].add(delta)


def stream_chunks(
    fold_inputs: Sequence[Tuple[Mapping[str, Any], Any, Mapping[str, np.ndarray], Sequence[Path]]],
    land_count: int,
    region_land_masks: Mapping[str, np.ndarray],
    confidence_percentiles: Sequence[int],
    progress_every: int = 250,
) -> Tuple[Dict[Tuple[str, str], OpportunityStats], Dict[Tuple[str, str, int], OpportunityStats], set[int]]:
    global_stats: Dict[Tuple[str, str], OpportunityStats] = defaultdict(OpportunityStats)
    by_year: Dict[Tuple[str, str, int], OpportunityStats] = defaultdict(OpportunityStats)
    observed_years: set[int] = set()
    processed = 0
    for manifest, model_c, boundaries, chunks in fold_inputs:
        fold = int(manifest["source_fold"])
        test_years = set(int(year) for year in manifest["test_years"])
        for chunk_path in chunks:
            with np.load(chunk_path, allow_pickle=False) as data:
                arrays = {
                    key: np.asarray(data[key])
                    for key in ("init_margin", "forecast_margin", "model_sigma", "truth", "base_rate")
                }
                lengths = {key: int(value.size) for key, value in arrays.items()}
                if any(length != int(land_count) for length in lengths.values()):
                    raise RuntimeError(
                        f"{chunk_path}: array lengths {lengths} do not equal land-mask count {land_count}."
                    )
                year = int(np.asarray(data["year"]).item())
                month = int(np.asarray(data["month"]).item())
                source_fold = int(np.asarray(data["source_fold"]).item())
            if source_fold != fold or year not in test_years:
                raise RuntimeError(
                    f"{chunk_path}: fold/year mismatch, got fold={source_fold}, year={year}; "
                    f"expected fold={fold}, year in {sorted(test_years)}."
                )
            observed_years.add(year)
            features = np.column_stack([arrays["init_margin"], arrays["forecast_margin"]]).astype(np.float32)
            probability = np.asarray(model_c.predict_features(features), dtype=np.float32)
            truth = np.asarray(arrays["truth"], dtype=np.float32)
            base_rate = np.asarray(arrays["base_rate"], dtype=np.float32)
            confidence = np.abs(probability - base_rate)

            selections = confidence_selections(
                confidence,
                confidence_percentiles,
                boundaries["confidence_edges"],
            )
            for stratum, selection in selections:
                update_stratum(
                    global_stats, by_year, "confidence", stratum, year,
                    probability, truth, base_rate, selection,
                )

            sigma_deciles = assign_deciles(arrays["model_sigma"], boundaries["sigma_edges"])
            for decile in range(10):
                update_stratum(
                    global_stats, by_year, "model_sigma_decile", f"decile_{decile + 1}", year,
                    probability, truth, base_rate, sigma_deciles == decile,
                )
            margin_deciles = assign_deciles(
                np.abs(arrays["forecast_margin"]), boundaries["forecast_margin_edges"]
            )
            for decile in range(10):
                update_stratum(
                    global_stats, by_year, "abs_forecast_margin_decile", f"decile_{decile + 1}", year,
                    probability, truth, base_rate, margin_deciles == decile,
                )
            update_stratum(
                global_stats, by_year, "month", str(month), year,
                probability, truth, base_rate, np.ones(land_count, dtype=bool),
            )
            top_decile_name = f"top_10pct_ge_p{TOP_CONFIDENCE_PERCENTILE}"
            top_decile = dict(selections)[top_decile_name]
            for region_name, region_selection in region_land_masks.items():
                update_stratum(
                    global_stats, by_year, "region", region_name, year,
                    probability, truth, base_rate, region_selection,
                )
                update_stratum(
                    global_stats, by_year, "region_x_top_confidence",
                    f"{region_name}__{top_decile_name}", year,
                    probability, truth, base_rate, region_selection & top_decile,
                )
            processed += 1
            if processed % int(progress_every) == 0:
                print(f"  streamed {processed} test chunks")
    return global_stats, by_year, observed_years


def build_summary_rows(
    global_stats: Mapping[Tuple[str, str], OpportunityStats],
    bootstrap: Mapping[str, Mapping[str, float]],
) -> List[Dict[str, Any]]:
    pooled_bss = global_stats[("confidence", "all")].metrics()["bss_unconditional"]
    rows: List[Dict[str, Any]] = []
    for axis, stratum in sorted(global_stats):
        metrics = global_stats[(axis, stratum)].metrics()
        row: Dict[str, Any] = {"axis": axis, "stratum": stratum, **metrics}
        if axis == "confidence":
            row.update(bootstrap.get(stratum, {}))
            row["delta_bss_vs_pooled"] = metrics["bss_unconditional"] - pooled_bss
        rows.append(row)
    return rows


def build_by_year_rows(
    by_year: Mapping[Tuple[str, str, int], OpportunityStats],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for axis, stratum, year in sorted(by_year):
        stats = by_year[(axis, stratum, year)]
        hist_count, hist_pred_sum, hist_pos = stats.histogram_rows()
        rows.append({
            "axis": axis,
            "stratum": stratum,
            "year": year,
            "count": stats.count,
            "event_count": stats.event_count,
            "pred_sum": stats.pred_sum,
            "brier_sum": stats.brier_sum,
            "climo_brier_sum": stats.climo_brier_sum,
            "hist_count": histogram_text(hist_count),
            "hist_pred_sum": histogram_text(hist_pred_sum),
            "hist_pos": histogram_text(hist_pos),
        })
    return rows


def reliability_plot(stats: OpportunityStats, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    occupied = stats.hist_count > 0
    mean_pred = np.divide(
        stats.hist_pred_sum, stats.hist_count,
        out=np.full(HIST_BINS, np.nan), where=occupied,
    )
    obs_freq = np.divide(
        stats.hist_pos, stats.hist_count,
        out=np.full(HIST_BINS, np.nan), where=occupied,
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    ax.plot(mean_pred[occupied], obs_freq[occupied], color="#176b87", linewidth=1.5)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency", title=title, xlim=(0, 1), ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def discard_curve_plot(summary_rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [row for row in summary_rows if row["axis"] == "confidence" and row["stratum"] != "all"]
    rows.sort(key=lambda row: retained_percent_from_confidence_stratum(str(row["stratum"])), reverse=True)
    retained = np.array(
        [retained_percent_from_confidence_stratum(str(row["stratum"])) for row in rows],
        dtype=float,
    )
    bss = np.array([float(row["bss_unconditional"]) for row in rows])
    low = np.array([float(row.get("bss_ci_low", np.nan)) for row in rows])
    high = np.array([float(row.get("bss_ci_high", np.nan)) for row in rows])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(retained, bss, yerr=np.vstack([bss - low, high - bss]), marker="o", color="#176b87", capsize=3)
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    ax.set(xlabel="Retained highest-confidence cells (%)", ylabel="BSS vs stored climatology", title="Forecasts-of-opportunity discard curve")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    global _AUC_FROM_HIST

    import cfm_mesh_train as cfm
    import exceedance_eval as ee
    import stitch_exceedance_folds as stitch
    from publication_analysis_utils import region_masks

    input_root = Path(args.input_root)
    run_names = parse_str_list(args.run_names)
    window_leads = parse_int_list(args.window_leads)
    confidence_percentiles = parse_int_list(args.confidence_percentiles)
    if TOP_CONFIDENCE_PERCENTILE not in confidence_percentiles:
        raise ValueError(
            f"--confidence_percentiles must include {TOP_CONFIDENCE_PERCENTILE} for the headline top-decile test."
        )
    if (
        any(percentile <= 0 or percentile >= 100 for percentile in confidence_percentiles)
        or len(set(confidence_percentiles)) != len(confidence_percentiles)
        or tuple(sorted(confidence_percentiles)) != tuple(confidence_percentiles)
    ):
        raise ValueError(
            "--confidence_percentiles must contain unique ascending integers strictly between 0 and 100."
        )
    _AUC_FROM_HIST = ee._roc_auc_from_hist
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else input_root / f"opportunity_window_{ee.lead_list_label(window_leads)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests: List[Mapping[str, Any]] = []
    calibrations: List[Mapping[str, np.ndarray]] = []
    chunk_lists: List[Sequence[Path]] = []
    for run_name in run_names:
        manifest, calibration, chunks = stitch.load_fold_inputs(input_root, run_name, window_leads)
        manifests.append(manifest)
        calibrations.append(calibration)
        chunk_lists.append(chunks)
    audit_rows = stitch.audit_folds(manifests, len(run_names))
    test_union = set().union(*(set(manifest["test_years"]) for manifest in manifests))
    if test_union != EXPECTED_YEARS:
        raise RuntimeError(
            f"Expected pooled test years 1981..2023; missing={sorted(EXPECTED_YEARS - test_union)}, "
            f"extra={sorted(test_union - EXPECTED_YEARS)}."
        )
    for row in audit_rows:
        print(
            f"Fold {row['source_fold']}: train/test overlap={row['own_train_test_overlap']}, "
            f"calibration/test overlap={row['own_calibration_test_overlap']}, "
            f"test years={row['test_years']}"
        )
    print("Fold leakage audit: PASS")

    models: List[Any] = []
    boundaries: List[Mapping[str, np.ndarray]] = []
    for manifest, calibration in zip(manifests, calibrations):
        fold = int(manifest["source_fold"])
        model_c = ee.fit_model_output_logistic_calibrator(
            np.column_stack([calibration["init_margin"], calibration["forecast_margin"]]),
            calibration["truth"],
            ("init_margin", "forecast_margin"),
            calibration_split=f"fold{fold}_own_validation",
            steps=200,
            lr=0.1,
            l2=1e-4,
        )
        models.append(model_c)
        boundaries.append(fit_boundaries(calibration, model_c, confidence_percentiles))
        print(f"Fold {fold}: Model C and all stratum boundaries fit on calibration years only.")
    print("Boundary-fit-before-test-streaming assert: PASS")

    land_mask = np.asarray(cfm.load_conus_mask(cfm.Config).cpu().numpy() > 0.5, dtype=bool)
    land_count = int(np.sum(land_mask))
    region_land_masks = {
        name: np.asarray(mask, dtype=bool).ravel()[land_mask.ravel()]
        for name, mask in region_masks(land_mask.shape).items()
    }
    fold_inputs = list(zip(manifests, models, boundaries, chunk_lists))
    global_stats, by_year, observed_years = stream_chunks(
        fold_inputs,
        land_count,
        region_land_masks,
        confidence_percentiles,
        progress_every=250,
    )
    if observed_years != EXPECTED_YEARS:
        raise RuntimeError(f"Streamed year set is not exactly 1981..2023: {sorted(observed_years)}")
    print(f"Chunk length and pooled-year asserts: PASS ({land_count} land cells, {len(observed_years)} years)")

    confidence_strata = [name for name, _ in confidence_selections(
        np.array([0.0]), confidence_percentiles, np.zeros(len(confidence_percentiles))
    )]
    top_stratum = f"top_10pct_ge_p{TOP_CONFIDENCE_PERCENTILE}"
    bootstrap = year_block_bootstrap(
        by_year,
        confidence_strata,
        top_stratum,
        sorted(observed_years),
        reps=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    reproducibility_check = year_block_bootstrap(
        by_year,
        confidence_strata,
        top_stratum,
        sorted(observed_years),
        reps=min(int(args.n_bootstrap), 20),
        seed=int(args.seed),
    )
    reproducibility_repeat = year_block_bootstrap(
        by_year,
        confidence_strata,
        top_stratum,
        sorted(observed_years),
        reps=min(int(args.n_bootstrap), 20),
        seed=int(args.seed),
    )
    if reproducibility_check != reproducibility_repeat:
        raise RuntimeError("Bootstrap reproducibility assert failed.")
    print(f"Year-block bootstrap reproducibility: PASS ({len(observed_years)} independent year blocks)")

    summary_rows = build_summary_rows(global_stats, bootstrap)
    by_year_rows = build_by_year_rows(by_year)
    write_csv(out_dir / "opportunity_summary.csv", summary_rows)
    write_csv(out_dir / "opportunity_by_year.csv", by_year_rows)
    reliability_plot(global_stats[("confidence", "all")], out_dir / "reliability_pooled.png", "Pooled Model C reliability")
    reliability_plot(global_stats[("confidence", top_stratum)], out_dir / "reliability_top_band.png", "Top-decile Model C reliability")
    discard_curve_plot(summary_rows, out_dir / "bss_vs_confidence_percentile.png")

    top = global_stats[("confidence", top_stratum)].metrics()
    pooled = global_stats[("confidence", "all")].metrics()
    top_ci = bootstrap[top_stratum]
    excludes_zero = top_ci["bss_ci_low"] > 0.0 or top_ci["bss_ci_high"] < 0.0
    excludes_pooled = (
        top_ci["bss_ci_low"] > pooled["bss_unconditional"]
        or top_ci["bss_ci_high"] < pooled["bss_unconditional"]
    )
    print("\nForecasts-of-opportunity verification")
    print("====================================")
    print(f"Output: {out_dir}")
    print(f"Independent year blocks: {len(observed_years)}")
    print(
        f"Top-decile BSS CI excludes zero={excludes_zero}, "
        f"excludes pooled point estimate={excludes_pooled}"
    )
    print(
        "OPPORTUNITY HEADLINE: top 10 percent confidence "
        f"BSS={top['bss_unconditional']:+.4f} "
        f"CI=[{top_ci['bss_ci_low']:+.4f},{top_ci['bss_ci_high']:+.4f}] "
        f"vs pooled BSS={pooled['bss_unconditional']:+.4f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_root",
        default="/blue/nessie/mostafarezaali/Teleconnection/exceedance_eval_incremental",
    )
    parser.add_argument(
        "--run_names",
        default="cvfold0_dist_v2_normfix,cvfold1_dist_v2_normfix,cvfold2_dist_v2_normfix,cvfold3_dist_v2_normfix,cvfold4_dist_v2_normfix",
    )
    parser.add_argument("--window_leads", default="12,13,14,15,16,17,18")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--confidence_percentiles", default="50,80,90,95,99")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
