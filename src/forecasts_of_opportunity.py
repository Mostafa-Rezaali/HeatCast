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
BASE_DRIVER_AXES = ("mjo_phase", "enso_state", "soil_moisture_tercile")
DRIVER_AXES = BASE_DRIVER_AXES
_AUC_FROM_HIST = None


def _trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return float(trapezoid(y, x))


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
        "sigma_bottom_tercile_edge": np.array(
            percentile_edges(np.asarray(calibration["model_sigma"]), [100.0 / 3.0])[0],
            dtype=np.float64,
        ),
    }


def mjo_stratum(phase: int, amplitude: float) -> str:
    if not np.isfinite(amplitude) or int(phase) not in range(1, 9):
        raise RuntimeError(f"Invalid MJO phase/amplitude at init: phase={phase}, amplitude={amplitude}.")
    return "inactive" if float(amplitude) < 1.0 else f"phase_{int(phase)}"


def enso_stratum(nino34: float) -> str:
    if not np.isfinite(nino34):
        raise RuntimeError("Missing Nino3.4 value at init.")
    if float(nino34) >= 0.5:
        return "el_nino"
    if float(nino34) <= -0.5:
        return "la_nina"
    return "neutral"


def teleconnection_stratum(value: float, threshold: float) -> str:
    if not np.isfinite(value):
        raise RuntimeError("Missing generic teleconnection-index value at init.")
    if float(value) >= float(threshold):
        return "positive"
    if float(value) <= -float(threshold):
        return "negative"
    return "neutral"


def soil_tercile_selections(percentile: np.ndarray) -> Dict[str, np.ndarray]:
    values = np.asarray(percentile)
    finite = np.isfinite(values)
    return {
        "dry": finite & (values <= 33.0),
        "mid": finite & (values > 33.0) & (values < 67.0),
        "wet": finite & (values >= 67.0),
        "undefined": ~finite,
    }


def load_init_sidecar(root: Path) -> Dict[int, int]:
    path = Path(root) / "incremental_arrays" / "init_dates.npz"
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {
            int(sample): int(init_t)
            for sample, init_t in zip(data["sample_index"], data["init_time_index"])
        }


def resolve_chunk_init_time(
    chunk_path: Path,
    chunk_data: Mapping[str, np.ndarray],
    sidecar: Mapping[int, int],
) -> int:
    init_t = (
        int(np.asarray(chunk_data["init_time_index"]).item())
        if "init_time_index" in chunk_data
        else -1
    )
    if init_t >= 0:
        return init_t
    sample_index = int(Path(chunk_path).stem.split("_")[-1])
    if sample_index in sidecar:
        return int(sidecar[sample_index])
    raise RuntimeError(
        f"{chunk_path}: missing a usable init_time_index. Run recover_chunk_init_dates.py "
        "for legacy chunks before slow-driver stratification."
    )


@dataclass
class DriverLookup:
    mjo_phase: np.ndarray
    mjo_amplitude: np.ndarray
    nino34: np.ndarray
    teleconnection_names: Tuple[str, ...]
    teleconnection_values: np.ndarray
    teleconnection_threshold: float
    alldata_names: Tuple[str, ...]
    alldata_values: np.ndarray
    alldata_threshold: float
    sidecars: Mapping[int, Mapping[int, int]]
    soil_rows: Mapping[int, Mapping[int, int]]
    soil_memmaps: Mapping[int, np.memmap]

    def sample_strata(
        self,
        fold: int,
        chunk_path: Path,
        chunk_data: Mapping[str, np.ndarray],
    ) -> Tuple[int, Dict[str, Dict[str, np.ndarray]]]:
        init_t = resolve_chunk_init_time(chunk_path, chunk_data, self.sidecars.get(fold, {}))
        if init_t < 0 or init_t >= len(self.mjo_phase):
            raise RuntimeError(f"{chunk_path}: init_time_index={init_t} outside driver table.")
        soil_row = self.soil_rows.get(fold, {}).get(init_t)
        if soil_row is None:
            raise RuntimeError(f"{chunk_path}: no fold-{fold} soil-percentile row for init {init_t}.")
        soil = np.asarray(self.soil_memmaps[fold][soil_row], dtype=np.float32)
        full = np.ones(soil.shape, dtype=bool)
        strata = {
            "mjo_phase": {mjo_stratum(self.mjo_phase[init_t], self.mjo_amplitude[init_t]): full},
            "enso_state": {enso_stratum(self.nino34[init_t]): full},
            "soil_moisture_tercile": soil_tercile_selections(soil),
        }
        for index, name in enumerate(self.teleconnection_names):
            strata[f"tele_{name}"] = {
                teleconnection_stratum(self.teleconnection_values[index, init_t], self.teleconnection_threshold): full
            }
        for index, name in enumerate(self.alldata_names):
            strata[f"alldata_{name}"] = {
                teleconnection_stratum(self.alldata_values[index, init_t], self.alldata_threshold): full
            }
        return init_t, strata


def load_driver_lookup(
    driver_table_dir: Path,
    manifests: Sequence[Mapping[str, Any]],
    land_count: int,
) -> DriverLookup:
    global_path = Path(driver_table_dir) / "mjo_enso_by_init.npz"
    if not global_path.exists():
        raise FileNotFoundError(f"Missing slow-driver table: {global_path}")
    with np.load(global_path, allow_pickle=False) as data:
        phase = np.asarray(data["mjo_phase"], dtype=np.int8)
        amplitude = np.asarray(data["mjo_amplitude"], dtype=np.float32)
        nino34 = np.asarray(data["nino34"], dtype=np.float32)
        tele_names = tuple(str(value) for value in np.asarray(data["teleconnection_names"], dtype=str).tolist()) if "teleconnection_names" in data else ()
        tele_values = np.asarray(data["teleconnection_values"], dtype=np.float32) if "teleconnection_values" in data else np.empty((0, phase.size), dtype=np.float32)
        tele_threshold = float(np.asarray(data["teleconnection_threshold"]).item()) if "teleconnection_threshold" in data else 0.5
        alldata_names = tuple(str(value) for value in np.asarray(data["alldata_names"], dtype=str).tolist()) if "alldata_names" in data else ()
        alldata_values = np.asarray(data["alldata_values"], dtype=np.float32) if "alldata_values" in data else np.empty((0, phase.size), dtype=np.float32)
        alldata_threshold = float(np.asarray(data["alldata_threshold"]).item()) if "alldata_threshold" in data else 0.5
    if not (phase.size == amplitude.size == nino34.size):
        raise RuntimeError("Global slow-driver arrays have inconsistent lengths.")
    if tele_values.shape != (len(tele_names), phase.size):
        raise RuntimeError(f"Generic teleconnection arrays have inconsistent shape: {tele_values.shape}, names={len(tele_names)}, time={phase.size}.")
    if alldata_values.shape != (len(alldata_names), phase.size):
        raise RuntimeError(f"AllData driver arrays have inconsistent shape: {alldata_values.shape}, names={len(alldata_names)}, time={phase.size}.")
    sidecars: Dict[int, Mapping[int, int]] = {}
    soil_rows: Dict[int, Mapping[int, int]] = {}
    soil_memmaps: Dict[int, np.memmap] = {}
    for manifest in manifests:
        fold = int(manifest["source_fold"])
        sidecars[fold] = load_init_sidecar(Path(manifest["root"]))
        index_path = Path(driver_table_dir) / f"fold{fold}_smpct_index.npz"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing fold-safe soil table index: {index_path}")
        with np.load(index_path, allow_pickle=False) as data:
            init_indices = np.asarray(data["init_time_index"], dtype=np.int32)
            shape = tuple(int(value) for value in np.asarray(data["shape"]).tolist())
            data_file = str(np.asarray(data["data_file"]).item())
            train_years = set(int(value) for value in np.asarray(data["train_years"]).tolist())
            undefined_fraction = float(np.asarray(data["undefined_fraction"]).item())
        if train_years != set(int(value) for value in manifest["train_years"]):
            raise RuntimeError(f"Fold {fold}: soil table train years do not match manifest.")
        if len(shape) != 2 or shape[1] != int(land_count):
            raise RuntimeError(f"Fold {fold}: soil memmap shape={shape}, expected second dim={land_count}.")
        if undefined_fraction >= 0.05:
            raise RuntimeError(f"Fold {fold}: cached undefined soil fraction={undefined_fraction:.4f} >= 0.05.")
        soil_rows[fold] = {int(init_t): row for row, init_t in enumerate(init_indices)}
        soil_memmaps[fold] = np.memmap(
            Path(driver_table_dir) / data_file,
            mode="r",
            dtype=np.float16,
            shape=shape,
        )
    return DriverLookup(
        phase,
        amplitude,
        nino34,
        tele_names,
        tele_values,
        tele_threshold,
        alldata_names,
        alldata_values,
        alldata_threshold,
        sidecars,
        soil_rows,
        soil_memmaps,
    )


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
    return _trapezoid_integral(tpr, fpr)


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


def year_block_bootstrap_axes(
    by_year: Mapping[Tuple[str, str, int], OpportunityStats],
    axes: Sequence[str],
    years: Sequence[int],
    reps: int,
    seed: int,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Bootstrap requested strata over whole years, preserving both BSS references."""
    years_array = np.asarray(sorted(int(year) for year in years), dtype=np.int16)
    if years_array.size < 2:
        raise RuntimeError("Year-block bootstrap requires at least two independent years.")
    requested = {
        (axis, stratum)
        for axis, stratum, _ in by_year
        if axis in set(str(value) for value in axes)
    }
    rng = np.random.default_rng(int(seed))
    samples: Dict[Tuple[str, str], Dict[str, List[float]]] = {
        key: {"bss_unconditional": [], "bss_conditional": [], "roc_auc": []}
        for key in requested
    }
    for _ in range(int(reps)):
        sampled = rng.choice(years_array, size=years_array.size, replace=True)
        unique, counts = np.unique(sampled, return_counts=True)
        weights = {int(year): int(count) for year, count in zip(unique, counts)}
        for axis, stratum in requested:
            metrics = aggregate_year_stats(by_year, axis, stratum, weights).metrics()
            for name in samples[(axis, stratum)]:
                samples[(axis, stratum)][name].append(metrics[name])
    output: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, metrics in samples.items():
        output[key] = {}
        for name, values in metrics.items():
            finite = np.asarray(values, dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            output[key][f"{name}_ci_low"] = (
                float(np.percentile(finite, 2.5)) if finite.size else float("nan")
            )
            output[key][f"{name}_ci_high"] = (
                float(np.percentile(finite, 97.5)) if finite.size else float("nan")
            )
    return output


def holm_adjust_pvalues(pvalues: Mapping[Any, float]) -> Dict[Any, float]:
    """Holm step-down family-wise-error adjustment."""
    finite = sorted(
        ((key, float(value)) for key, value in pvalues.items() if np.isfinite(value)),
        key=lambda item: item[1],
    )
    adjusted: Dict[Any, float] = {key: float("nan") for key in pvalues}
    running = 0.0
    total = len(finite)
    for rank, (key, value) in enumerate(finite):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[key] = running
    return adjusted


def interaction_parent_pairs(
    by_year: Mapping[Tuple[str, str, int], OpportunityStats],
    top_stratum: str,
) -> List[Tuple[Tuple[str, str], str, Tuple[str, str]]]:
    """Return interaction/parent pairs needed for physical opportunity claims."""
    interactions = sorted({
        (axis, stratum)
        for axis, stratum, _ in by_year
        if axis.endswith("_x_top_confidence") or axis.endswith("_x_low_sigma")
    })
    pairs: List[Tuple[Tuple[str, str], str, Tuple[str, str]]] = []
    for axis, stratum in interactions:
        if "__" not in stratum:
            continue
        driver_axis = axis.split("_x_", 1)[0]
        driver_stratum = stratum.split("__", 1)[0]
        selection_parent = (
            ("confidence", top_stratum)
            if axis.endswith("_x_top_confidence")
            else ("low_sigma", "bottom_sigma_tercile")
        )
        pairs.append(((axis, stratum), "selection_parent", selection_parent))
        pairs.append(((axis, stratum), "driver_parent", (driver_axis, driver_stratum)))
    return pairs


def paired_year_block_bootstrap_interactions(
    global_stats: Mapping[Tuple[str, str], OpportunityStats],
    by_year: Mapping[Tuple[str, str, int], OpportunityStats],
    years: Sequence[int],
    top_stratum: str,
    reps: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Compare each interaction with its selection and driver parents using paired year blocks."""
    years_array = np.asarray(sorted(int(year) for year in years), dtype=np.int16)
    if years_array.size < 2:
        raise RuntimeError("Paired interaction bootstrap requires at least two independent years.")
    pairs = interaction_parent_pairs(by_year, top_stratum)
    metrics = ("bss_unconditional", "bss_conditional")
    samples: Dict[Tuple[Tuple[str, str], str, Tuple[str, str], str], List[float]] = {
        (interaction, parent_kind, parent, metric): []
        for interaction, parent_kind, parent in pairs
        for metric in metrics
    }
    rng = np.random.default_rng(int(seed))
    for _ in range(int(reps)):
        sampled = rng.choice(years_array, size=years_array.size, replace=True)
        unique, counts = np.unique(sampled, return_counts=True)
        weights = {int(year): int(count) for year, count in zip(unique, counts)}
        required = {
            key
            for interaction, _, parent in pairs
            for key in (interaction, parent)
        }
        replicate = {
            key: aggregate_year_stats(by_year, key[0], key[1], weights).metrics()
            for key in required
        }
        for interaction, parent_kind, parent in pairs:
            for metric in metrics:
                samples[(interaction, parent_kind, parent, metric)].append(
                    replicate[interaction][metric] - replicate[parent][metric]
                )

    rows: List[Dict[str, Any]] = []
    for interaction, parent_kind, parent in pairs:
        interaction_metrics = global_stats[interaction].metrics()
        parent_metrics = global_stats[parent].metrics()
        for metric in metrics:
            values = np.asarray(samples[(interaction, parent_kind, parent, metric)], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size:
                low, high = np.percentile(finite, [2.5, 97.5])
                lower_tail = (np.sum(finite <= 0.0) + 1.0) / (finite.size + 1.0)
                upper_tail = (np.sum(finite >= 0.0) + 1.0) / (finite.size + 1.0)
                p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
            else:
                low = high = p_value = float("nan")
            rows.append({
                "interaction_axis": interaction[0],
                "interaction_stratum": interaction[1],
                "parent_kind": parent_kind,
                "parent_axis": parent[0],
                "parent_stratum": parent[1],
                "metric": metric,
                "interaction_value": interaction_metrics[metric],
                "parent_value": parent_metrics[metric],
                "delta": interaction_metrics[metric] - parent_metrics[metric],
                "delta_ci_low": float(low),
                "delta_ci_high": float(high),
                "p_value": float(p_value),
                "p_holm_mjo": float("nan"),
                "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
                "independent_year_blocks": int(years_array.size),
                "bootstrap_reps": int(reps),
            })

    families: Dict[Tuple[str, str, str], Dict[int, float]] = defaultdict(dict)
    for index, row in enumerate(rows):
        if row["interaction_axis"].startswith("mjo_phase_x_"):
            phase_match = re.fullmatch(r"phase_([1-8])__.+", str(row["interaction_stratum"]))
            if phase_match is not None:
                family = (str(row["interaction_axis"]), str(row["parent_kind"]), str(row["metric"]))
                families[family][index] = float(row["p_value"])
    for family in families.values():
        adjusted = holm_adjust_pvalues(family)
        for index, value in adjusted.items():
            rows[index]["p_holm_mjo"] = value
    return rows


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
    driver_lookup: Optional[DriverLookup] = None,
    progress_every: int = 250,
) -> Tuple[Dict[Tuple[str, str], OpportunityStats], Dict[Tuple[str, str, int], OpportunityStats], set[int]]:
    global_stats: Dict[Tuple[str, str], OpportunityStats] = defaultdict(OpportunityStats)
    by_year: Dict[Tuple[str, str, int], OpportunityStats] = defaultdict(OpportunityStats)
    observed_years: set[int] = set()
    processed = 0
    soil_undefined = 0
    soil_total = 0
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
                driver_selections = (
                    driver_lookup.sample_strata(fold, chunk_path, data)[1]
                    if driver_lookup is not None
                    else None
                )
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
            if driver_selections is not None:
                low_sigma = (
                    np.asarray(arrays["model_sigma"])
                    <= float(np.asarray(boundaries["sigma_bottom_tercile_edge"]).item())
                )
                update_stratum(
                    global_stats, by_year, "low_sigma", "bottom_sigma_tercile", year,
                    probability, truth, base_rate, low_sigma,
                )
                for driver_axis, strata in driver_selections.items():
                    for stratum, driver_selection in strata.items():
                        update_stratum(
                            global_stats, by_year, driver_axis, stratum, year,
                            probability, truth, base_rate, driver_selection,
                        )
                        update_stratum(
                            global_stats, by_year, f"{driver_axis}_x_top_confidence",
                            f"{stratum}__{top_decile_name}", year,
                            probability, truth, base_rate, driver_selection & top_decile,
                        )
                        update_stratum(
                            global_stats, by_year, f"{driver_axis}_x_low_sigma",
                            f"{stratum}__bottom_sigma_tercile", year,
                            probability, truth, base_rate, driver_selection & low_sigma,
                        )
                    if driver_axis == "soil_moisture_tercile":
                        soil_undefined += int(np.sum(strata["undefined"]))
                        soil_total += int(land_count)
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
    if driver_lookup is not None:
        undefined_fraction = soil_undefined / max(soil_total, 1)
        if undefined_fraction >= 0.05:
            raise RuntimeError(
                f"Streamed undefined soil-percentile fraction={undefined_fraction:.4f} >= 0.05."
            )
        print(f"Slow-driver hot-loop audit: PASS (soil undefined={undefined_fraction:.4%}; no NetCDF reads)")
    return global_stats, by_year, observed_years


def build_summary_rows(
    global_stats: Mapping[Tuple[str, str], OpportunityStats],
    bootstrap: Mapping[str, Mapping[str, float]],
    axis_bootstrap: Optional[Mapping[Tuple[str, str], Mapping[str, float]]] = None,
) -> List[Dict[str, Any]]:
    pooled_bss = global_stats[("confidence", "all")].metrics()["bss_unconditional"]
    rows: List[Dict[str, Any]] = []
    for axis, stratum in sorted(global_stats):
        metrics = global_stats[(axis, stratum)].metrics()
        row: Dict[str, Any] = {"axis": axis, "stratum": stratum, **metrics}
        if axis == "confidence":
            row.update(bootstrap.get(stratum, {}))
            row["delta_bss_vs_pooled"] = metrics["bss_unconditional"] - pooled_bss
        if axis_bootstrap is not None:
            row.update(axis_bootstrap.get((axis, stratum), {}))
        if "_x_" in axis and "__" in stratum:
            parent_axis = axis.split("_x_", 1)[0]
            parent_stratum = stratum.split("__", 1)[0]
            parent = global_stats.get((parent_axis, parent_stratum))
            if parent is not None and parent.count > 0:
                row["retained_fraction_within_driver"] = metrics["n"] / parent.count
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


def driver_conditional_bss_plot(
    summary_rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    mjo_order = [f"phase_{phase}" for phase in range(1, 9)]
    soil_order = ["dry", "mid", "wet"]
    panels = [
        ("mjo_phase", mjo_order, "Active MJO phase"),
        ("soil_moisture_tercile", soil_order, "Antecedent soil moisture"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, (axis, order, title) in zip(axes, panels):
        indexed = {
            str(row["stratum"]): row
            for row in summary_rows
            if row["axis"] == axis
        }
        rows = [indexed[name] for name in order if name in indexed]
        labels = [str(row["stratum"]).replace("phase_", "P").replace("_", " ") for row in rows]
        values = np.asarray([float(row["bss_conditional"]) for row in rows])
        low = np.asarray([float(row.get("bss_conditional_ci_low", np.nan)) for row in rows])
        high = np.asarray([float(row.get("bss_conditional_ci_high", np.nan)) for row in rows])
        yerr = np.vstack([np.maximum(values - low, 0.0), np.maximum(high - values, 0.0)])
        ax.bar(np.arange(len(rows)), values, color="#176b87")
        ax.errorbar(np.arange(len(rows)), values, yerr=yerr, fmt="none", color="black", capsize=3)
        ax.axhline(0.0, color="0.5", linestyle="--", linewidth=1)
        ax.set_xticks(np.arange(len(rows)), labels)
        ax.set(title=title, ylabel="Conditional BSS")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def print_driver_verdicts(
    global_stats: Mapping[Tuple[str, str], OpportunityStats],
    axis_bootstrap: Mapping[Tuple[str, str], Mapping[str, float]],
) -> None:
    pooled = global_stats[("confidence", "all")].metrics()["bss_unconditional"]
    driver_axes = sorted({
        axis
        for axis, _ in global_stats
        if axis in BASE_DRIVER_AXES or axis.startswith("tele_")
    })
    for axis in driver_axes:
        candidates = [
            (stratum, stats.metrics())
            for (candidate_axis, stratum), stats in global_stats.items()
            if candidate_axis == axis and stratum != "undefined"
        ]
        if not candidates:
            continue
        stratum, metrics = max(candidates, key=lambda item: item[1]["bss_unconditional"])
        ci = axis_bootstrap.get((axis, stratum), {})
        low = float(ci.get("bss_unconditional_ci_low", np.nan))
        high = float(ci.get("bss_unconditional_ci_high", np.nan))
        print(
            f"DRIVER {axis}: best stratum {stratum} "
            f"BSS={metrics['bss_unconditional']:+.4f} CI=[{low:+.4f},{high:+.4f}] "
            f"vs pooled {pooled:+.4f}"
        )


def print_paired_interaction_headlines(rows: Sequence[Mapping[str, Any]]) -> None:
    required = {
        ("mjo_phase_x_top_confidence", "phase_8__top_10pct_ge_p90", "selection_parent"),
        ("mjo_phase_x_top_confidence", "phase_8__top_10pct_ge_p90", "driver_parent"),
        ("mjo_phase_x_low_sigma", "phase_8__bottom_sigma_tercile", "selection_parent"),
        ("mjo_phase_x_low_sigma", "phase_8__bottom_sigma_tercile", "driver_parent"),
    }
    print("Paired phase-8 interaction tests (whole-year bootstrap):")
    found = 0
    for row in rows:
        key = (
            str(row["interaction_axis"]),
            str(row["interaction_stratum"]),
            str(row["parent_kind"]),
        )
        if key not in required or row["metric"] != "bss_unconditional":
            continue
        found += 1
        holm = float(row["p_holm_mjo"])
        holm_text = f"{holm:.4g}" if np.isfinite(holm) else "nan"
        print(
            f"  {row['interaction_stratum']} vs {row['parent_axis']}:{row['parent_stratum']}: "
            f"delta_BSS={float(row['delta']):+.4f} "
            f"CI=[{float(row['delta_ci_low']):+.4f},{float(row['delta_ci_high']):+.4f}] "
            f"p={float(row['p_value']):.4g}, Holm-MJO={holm_text}"
        )
    if found != len(required):
        print(f"  WARNING: found {found}/{len(required)} required phase-8 comparisons.")


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
    bootstrap_axes = parse_str_list(args.bootstrap_axes)
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
    driver_lookup = (
        load_driver_lookup(Path(args.driver_table_dir), manifests, land_count)
        if args.driver_table_dir
        else None
    )
    if bootstrap_axes and driver_lookup is None:
        raise ValueError("--bootstrap_axes requires --driver_table_dir.")
    global_stats, by_year, observed_years = stream_chunks(
        fold_inputs,
        land_count,
        region_land_masks,
        confidence_percentiles,
        driver_lookup=driver_lookup,
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

    if driver_lookup is not None:
        dynamic_driver_axes = tuple(
            BASE_DRIVER_AXES
            + tuple(f"tele_{name}" for name in driver_lookup.teleconnection_names)
            + tuple(f"alldata_{name}" for name in driver_lookup.alldata_names)
        )
        if not bootstrap_axes:
            bootstrap_axes = dynamic_driver_axes
        elif "all_drivers" in bootstrap_axes:
            bootstrap_axes = tuple(
                axis
                for axis in bootstrap_axes
                if axis != "all_drivers"
            ) + dynamic_driver_axes
    expanded_bootstrap_axes = tuple(dict.fromkeys(
        candidate
        for axis in bootstrap_axes
        for candidate in (axis, f"{axis}_x_top_confidence", f"{axis}_x_low_sigma")
    ))
    axis_bootstrap = (
        year_block_bootstrap_axes(
            by_year,
            expanded_bootstrap_axes,
            sorted(observed_years),
            reps=int(args.n_bootstrap),
            seed=int(args.seed),
        )
        if bootstrap_axes
        else {}
    )
    paired_interaction_rows = (
        paired_year_block_bootstrap_interactions(
            global_stats,
            by_year,
            sorted(observed_years),
            top_stratum,
            reps=int(args.n_bootstrap),
            seed=int(args.seed),
        )
        if driver_lookup is not None
        else []
    )
    summary_rows = build_summary_rows(global_stats, bootstrap, axis_bootstrap)
    by_year_rows = build_by_year_rows(by_year)
    write_csv(out_dir / "opportunity_summary.csv", summary_rows)
    write_csv(out_dir / "opportunity_by_year.csv", by_year_rows)
    reliability_plot(global_stats[("confidence", "all")], out_dir / "reliability_pooled.png", "Pooled Model C reliability")
    reliability_plot(global_stats[("confidence", top_stratum)], out_dir / "reliability_top_band.png", "Top-decile Model C reliability")
    discard_curve_plot(summary_rows, out_dir / "bss_vs_confidence_percentile.png")
    if driver_lookup is not None:
        driver_axis_prefixes = tuple(
            BASE_DRIVER_AXES
            + tuple(f"tele_{name}" for name in driver_lookup.teleconnection_names)
            + tuple(f"alldata_{name}" for name in driver_lookup.alldata_names)
        )
        driver_rows = [
            row for row in summary_rows
            if any(str(row["axis"]).startswith(axis) for axis in driver_axis_prefixes)
        ]
        write_csv(out_dir / "driver_opportunity_summary.csv", driver_rows)
        write_csv(out_dir / "driver_interaction_paired_bootstrap.csv", paired_interaction_rows)
        driver_conditional_bss_plot(driver_rows, out_dir / "driver_conditional_bss.png")
        print(
            "Slow-driver leakage audit: PASS (MJO/ENSO/generic teleconnections/AllData drivers are external init-date indices; "
            "soil percentiles and interaction boundaries use fold train/calibration years only; "
            "no test outcomes define strata)."
        )

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
    if driver_lookup is not None:
        print_driver_verdicts(global_stats, axis_bootstrap)
        print_paired_interaction_headlines(paired_interaction_rows)


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
    parser.add_argument(
        "--driver_table_dir",
        default=None,
        help="Cached output directory from build_driver_tables.py. Omit to preserve the original analysis.",
    )
    parser.add_argument(
        "--bootstrap_axes",
        default="all_drivers",
        help="Comma-separated slow-driver axes to year-block bootstrap. Use all_drivers to include MJO, ENSO, soil, and every tele_* index.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
