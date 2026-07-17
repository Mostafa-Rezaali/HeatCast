#!/usr/bin/env python3
"""Fold-safe heat-exceedance evaluation for existing HeatCast checkpoints.

Evaluation only: no architecture, training-loop, or loss changes. The default
daily mode derives center-lead exceedance probabilities from the existing point
forecast mean using month-specific train-year q95 thresholds and climatological
anomaly spread. Window mode evaluates exceedance of a configurable multi-lead
mean forecast with its own train-year windowed q95 threshold and forecast-error
spread.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import cfm_mesh_train as cfm
from publication_analysis_utils import NOAA_REGION_BOXES, conus_lat_lon, ensure_dir, region_masks


MJJAS_MONTHS = (5, 6, 7, 8, 9)
BASE_DATE = datetime(1981, 5, 1)


def _trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return float(trapezoid(y, x))


def time_datetimes(time_values: Sequence[float]) -> List[datetime]:
    return [BASE_DATE + timedelta(days=float(v)) for v in time_values]


def parse_int_list(text: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(text).split(",") if x.strip())


def parse_float_list(text: str) -> Tuple[float, ...]:
    return tuple(float(x.strip()) for x in str(text).split(",") if x.strip())


def lead_list_label(leads: Sequence[int]) -> str:
    return "-".join(str(int(x)) for x in leads)


def window_center_lead(window_leads: Sequence[int]) -> int:
    if not window_leads:
        raise ValueError("Window lead list cannot be empty.")
    med = float(np.median(np.asarray(window_leads, dtype=np.float64)))
    return int(math.floor(med + 0.5))


def window_lead_indices(predicted_leads: Sequence[int], window_leads: Sequence[int]) -> Tuple[int, ...]:
    pred = tuple(int(x) for x in predicted_leads)
    missing = [int(x) for x in window_leads if int(x) not in pred]
    if missing:
        raise RuntimeError(
            "Requested --window_leads are not all present in the model prediction tube. "
            f"Requested={tuple(int(x) for x in window_leads)}, predicted={pred}, missing={tuple(missing)}. "
            "A wider-tube retrain is required for this window."
        )
    return tuple(pred.index(int(lead)) for lead in window_leads)


def configure_from_args(args: argparse.Namespace) -> None:
    cfm.Config.DETERMINISTIC = True
    cfm.Config.RUN_NAME = args.run_name
    cfm.Config.CV_STRIDE = int(args.cv_stride)
    fold = int(args.cv_fold) % int(args.cv_stride)
    cfm.Config.CV_FOLD = fold
    cfm.Config.CV_TEST_OFFSETS = (fold,)
    cfm.Config.CV_VAL_OFFSETS = ((fold + 1) % int(args.cv_stride),)

    if args.multi_lead_tube or "tube" in args.run_name.lower() or getattr(args, "target_mode", "daily") == "window":
        cfm.Config.MULTI_LEAD_TUBE = True
        cfm.Config.PREDICTION_LEADS = parse_int_list(args.prediction_leads)
    else:
        cfm.Config.MULTI_LEAD_TUBE = False

    cfm.Config.GRADIENT_LOSS_WEIGHT = 0.0
    cfm.Config.TUBE_DECODE_CHUNK_SIZE = max(0, int(args.tube_decode_chunk_size))
    if getattr(args, "use_model_sigma", False):
        cfm.Config.DISTRIBUTIONAL_HEAD = True
        cfm.Config.IMAGE_CHANNELS = 2
        cfm.Config.SIGMA_FLOOR = 0.1
    cfm.apply_run_name(args.run_name)
    cfm.apply_extended_global_fields()
    cfm.set_random_seed(int(args.seed))


def checkpoint_path_for(run_name: str, checkpoint: str) -> str:
    if checkpoint not in {"best_monitor", "best_tac", "best_r2", "auto"}:
        return checkpoint

    candidates = []
    if checkpoint in {"best_monitor", "auto"}:
        candidates.append(cfm.Config.MONITOR_MODEL_SAVE_PATH)
    if checkpoint in {"best_tac", "auto"}:
        candidates.append(cfm.Config.TAC_MODEL_SAVE_PATH)
    if checkpoint in {"best_r2", "auto"}:
        candidates.append(cfm.Config.MODEL_SAVE_PATH)

    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not resolve checkpoint={checkpoint!r} for run_name={run_name}. Tried: {candidates}"
    )


def configure_structure_from_checkpoint(checkpoint_path: str, prediction_leads_arg: str) -> None:
    """Apply checkpoint architecture metadata before building datasets and cache paths."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("ema_state_dict", checkpoint.get("model_state_dict"))
    if state_dict is None:
        raise KeyError(f"Checkpoint {checkpoint_path} has no model_state_dict or ema_state_dict")
    keys = tuple(k.removeprefix("module.") for k in state_dict)
    checkpoint_tube = bool(checkpoint.get("multi_lead_tube", False)) or any(
        key.startswith(("lead_embedding.", "lead_time_proj.", "tube_temporal_"))
        for key in keys
    )
    if checkpoint_tube:
        cfm.Config.MULTI_LEAD_TUBE = True
        saved_leads = checkpoint.get("prediction_leads")
        cfm.Config.PREDICTION_LEADS = (
            tuple(int(x) for x in saved_leads)
            if saved_leads is not None
            else parse_int_list(prediction_leads_arg)
        )
        clean_state = {k.removeprefix("module."): v for k, v in state_dict.items()}
        lead_weight = clean_state.get("lead_embedding.weight")
        if lead_weight is not None and len(cfm.prediction_leads(cfm.Config)) != int(lead_weight.shape[0]):
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} contains {int(lead_weight.shape[0])} tube leads, "
                f"but configured prediction leads are {cfm.prediction_leads(cfm.Config)}. "
                "Pass the checkpoint's exact --prediction_leads."
            )
        print(f"Checkpoint structure: tube leads {cfm.prediction_leads(cfm.Config)}")
    elif cfm.Config.MULTI_LEAD_TUBE:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is single-lead, but tube evaluation was requested."
        )

    if bool(checkpoint.get("distributional_head", False)) or int(
        checkpoint.get("image_channels", cfm.Config.IMAGE_CHANNELS)
    ) == 2:
        cfm.Config.DISTRIBUTIONAL_HEAD = True
        cfm.Config.IMAGE_CHANNELS = 2
        cfm.Config.SIGMA_FLOOR = float(checkpoint.get("sigma_floor", cfm.Config.SIGMA_FLOOR))


def split_indices_for_config(shared_data: Mapping[str, np.ndarray]):
    time_values = np.asarray(shared_data["time_values"])
    runs = cfm.detect_continuous_runs(time_values)
    all_valid = cfm.build_valid_indices(
        runs,
        lead_time=cfm.max_prediction_lead(cfm.Config),
        min_history=cfm.required_input_history(cfm.Config),
    )
    return cfm.build_crossval_split(all_valid, time_values)


def load_norm_stats() -> Dict[str, torch.Tensor]:
    stats_path = cfm.get_norm_stats_path(cfm.Config)
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Missing norm stats for this fold/config: {stats_path}. "
            "Run training setup first; Stage 1 evaluation does not create training stats."
        )
    return cfm.load_norm_stats_npz(stats_path)


def train_year_mask_from_years(time_values: Sequence[float], train_years: Iterable[int]) -> np.ndarray:
    train_years = set(int(y) for y in train_years)
    dts = time_datetimes(time_values)
    return np.array([dt.year in train_years for dt in dts], dtype=bool)


def month_doy_year_arrays(time_values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dts = time_datetimes(time_values)
    months = np.array([dt.month for dt in dts], dtype=np.int16)
    years = np.array([dt.year for dt in dts], dtype=np.int16)
    doys = cfm.compute_doy_array(np.asarray(time_values)).astype(np.int16)
    return months, years, doys


def _source_mtime(paths: Sequence[str]) -> float:
    existing = [os.path.getmtime(p) for p in paths if p and os.path.exists(p)]
    return max(existing) if existing else 0.0


def exceedance_cache_path() -> str:
    suffix = (
        "_tube" + "-".join(str(x) for x in cfm.prediction_leads(cfm.Config))
        if cfm.Config.MULTI_LEAD_TUBE else ""
    )
    fold = int(getattr(cfm.Config, "CV_FOLD", cfm.Config.CV_TEST_OFFSETS[0]))
    return os.path.join(
        cfm.Config.OUTPUT_DIR,
        "data_cache",
        f"month_q95_sigma_direct15{suffix}_{cfm.cv_split_tag(cfm.Config)}_fold{fold}.npz",
    )


def window_exceedance_cache_path(window_leads: Sequence[int]) -> str:
    suffix = (
        "_tube" + lead_list_label(cfm.prediction_leads(cfm.Config))
        if cfm.Config.MULTI_LEAD_TUBE else ""
    )
    fold = int(getattr(cfm.Config, "CV_FOLD", cfm.Config.CV_TEST_OFFSETS[0]))
    return os.path.join(
        cfm.Config.OUTPUT_DIR,
        "data_cache",
        (
            f"window_q95_sigma_direct15{suffix}_win{lead_list_label(window_leads)}_"
            f"{cfm.cv_split_tag(cfm.Config)}_fold{fold}.npz"
        ),
    )


def _normalize_field(field: np.ndarray, norm_stats: Mapping[str, torch.Tensor]) -> np.ndarray:
    hi_mean = float(norm_stats["hi_mean"])
    hi_std = float(norm_stats["hi_std"])
    out = np.asarray(field, dtype=np.float32).copy()
    valid = np.isfinite(out) & (out != 0.0)
    out[valid] = (out[valid] - hi_mean) / (hi_std + 1e-8)
    out[~valid] = np.nan
    return out


def window_mean_z_from_shared(
    heat: np.ndarray,
    init_t: int,
    window_leads: Sequence[int],
    norm_stats: Mapping[str, torch.Tensor],
) -> np.ndarray:
    fields = [
        _normalize_field(np.asarray(heat[:, :, int(init_t) + int(lead)]), norm_stats)
        for lead in window_leads
    ]
    with np.errstate(all="ignore"):
        return np.nanmean(np.stack(fields, axis=0), axis=0).astype(np.float32)


def build_month_q95(shared_data, train_year_mask, norm_stats) -> np.ndarray:
    """Month-specific pixelwise train-year q95 in normalized z units."""
    heat = shared_data["heat_index"]
    time_values = np.asarray(shared_data["time_values"])
    months, _, _ = month_doy_year_arrays(time_values)
    h, w = cfm.Config.IMAGE_SIZE
    q95_z = np.full((12, h, w), np.nan, dtype=np.float32)
    land_mask = np.isfinite(np.asarray(heat[:, :, 0])) & (np.asarray(heat[:, :, 0]) != 0.0)

    for month in MJJAS_MONTHS:
        indices = np.where(train_year_mask & (months == month))[0]
        if indices.size == 0:
            raise RuntimeError(f"No train-year days found for month={month}")
        fields = np.asarray(heat[:, :, indices], dtype=np.float32)
        valid = np.isfinite(fields) & (fields != 0.0)
        fields = (fields - float(norm_stats["hi_mean"])) / (float(norm_stats["hi_std"]) + 1e-8)
        fields[~valid] = np.nan
        with np.errstate(all="ignore"):
            q95_z[month] = np.nanpercentile(fields, 95.0, axis=2).astype(np.float32)
        q95_z[month][~land_mask] = np.nan
        del fields, valid
    return q95_z


def build_month_sigma_clim(shared_data, climo_by_doy, train_year_mask, norm_stats) -> np.ndarray:
    """Month-specific std of train-year daily anomalies about train-year daily climatology."""
    heat = shared_data["heat_index"]
    time_values = np.asarray(shared_data["time_values"])
    months, _, doys = month_doy_year_arrays(time_values)
    h, w = cfm.Config.IMAGE_SIZE
    sigma = np.full((12, h, w), np.nan, dtype=np.float32)
    land_mask = np.isfinite(np.asarray(heat[:, :, 0])) & (np.asarray(heat[:, :, 0]) != 0.0)

    for month in MJJAS_MONTHS:
        indices = np.where(train_year_mask & (months == month))[0]
        anomalies = []
        for ti in indices:
            field_z = _normalize_field(np.asarray(heat[:, :, ti]), norm_stats)
            anomalies.append(field_z - np.asarray(climo_by_doy[int(doys[ti])], dtype=np.float32))
        stack = np.stack(anomalies, axis=2)
        with np.errstate(all="ignore"):
            sigma_m = np.nanstd(stack, axis=2).astype(np.float32)
        sigma_m = np.where(np.isfinite(sigma_m), sigma_m, np.nan)
        sigma_m[land_mask] = np.maximum(sigma_m[land_mask], 0.1)
        sigma_m[~land_mask] = np.nan
        sigma[month] = sigma_m
        del stack, anomalies
    return sigma


def build_month_base_rate(shared_data, train_year_mask, norm_stats, q95_z) -> np.ndarray:
    heat = shared_data["heat_index"]
    time_values = np.asarray(shared_data["time_values"])
    months, _, _ = month_doy_year_arrays(time_values)
    h, w = cfm.Config.IMAGE_SIZE
    base_rate = np.full((12, h, w), np.nan, dtype=np.float32)
    land_mask = np.isfinite(np.asarray(heat[:, :, 0])) & (np.asarray(heat[:, :, 0]) != 0.0)

    for month in MJJAS_MONTHS:
        indices = np.where(train_year_mask & (months == month))[0]
        count = np.zeros((h, w), dtype=np.float32)
        exceed = np.zeros((h, w), dtype=np.float32)
        threshold = q95_z[month]
        for ti in indices:
            field_z = _normalize_field(np.asarray(heat[:, :, ti]), norm_stats)
            valid = land_mask & np.isfinite(field_z) & np.isfinite(threshold)
            count[valid] += 1.0
            exceed[valid] += (field_z[valid] > threshold[valid]).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = exceed / np.maximum(count, 1.0)
        rate[~land_mask] = np.nan
        base_rate[month] = rate.astype(np.float32)
    return base_rate


def load_or_build_exceedance_stats(shared_data, train_years, norm_stats, climo_by_doy):
    path = exceedance_cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    time_values = np.asarray(shared_data["time_values"])
    train_mask = train_year_mask_from_years(time_values, train_years)
    norm_path = cfm.get_norm_stats_path(cfm.Config)
    climo_path = cfm.get_climatology_cache_path(cfm.Config)
    src_mtime = _source_mtime([norm_path, climo_path, cfm.Config.TRAINING_DATA_PATH])

    def _cache_ok() -> bool:
        if not os.path.exists(path):
            return False
        if os.path.getmtime(path) < src_mtime:
            return False
        try:
            with np.load(path, allow_pickle=True) as data:
                cached_years = set(np.atleast_1d(data["train_years"]).astype(int).tolist())
                cached_fold = int(data["cv_fold"]) if "cv_fold" in data.files else None
                return (
                    cached_years == set(int(y) for y in train_years)
                    and cached_fold == int(getattr(cfm.Config, "CV_FOLD", cfm.Config.CV_TEST_OFFSETS[0]))
                    and str(data["cv_split"].item()) == cfm.cv_split_tag(cfm.Config)
                    and abs(float(data["hi_mean"]) - float(norm_stats["hi_mean"])) < 1e-6
                    and abs(float(data["hi_std"]) - float(norm_stats["hi_std"])) < 1e-6
                )
        except Exception:
            return False

    if _cache_ok():
        print(f"Loading fold-safe exceedance stats from {path}")
        with np.load(path, allow_pickle=False) as data:
            return (
                np.asarray(data["q95_z"], dtype=np.float32),
                np.asarray(data["sigma_clim"], dtype=np.float32),
                np.asarray(data["base_rate"], dtype=np.float32),
            )

    print("Building fold-safe month-specific q95/base-rate/sigma stats from train years only...")
    q95_z = build_month_q95(shared_data, train_mask, norm_stats)
    sigma_clim = build_month_sigma_clim(shared_data, climo_by_doy, train_mask, norm_stats)
    base_rate = build_month_base_rate(shared_data, train_mask, norm_stats, q95_z)

    for month in MJJAS_MONTHS:
        rate = np.nanmean(base_rate[month])
        print(f"  Train build check month={month}: mean exceedance rate={rate:.4f}")
        if not (0.025 <= rate <= 0.075):
            raise RuntimeError(f"Train exceedance rate for month {month} is not near 5%: {rate:.4f}")

    np.savez_compressed(
        path,
        q95_z=q95_z.astype(np.float32),
        sigma_clim=sigma_clim.astype(np.float32),
        base_rate=base_rate.astype(np.float32),
        train_years=np.array(sorted(int(y) for y in train_years), dtype=np.int16),
        cv_fold=np.array(int(getattr(cfm.Config, "CV_FOLD", cfm.Config.CV_TEST_OFFSETS[0])), dtype=np.int16),
        cv_split=np.array(cfm.cv_split_tag(cfm.Config), dtype=object),
        hi_mean=np.array(float(norm_stats["hi_mean"]), dtype=np.float32),
        hi_std=np.array(float(norm_stats["hi_std"]), dtype=np.float32),
        months=np.array(MJJAS_MONTHS, dtype=np.int8),
        norm_stats_mtime=np.array(os.path.getmtime(norm_path) if os.path.exists(norm_path) else 0.0),
        climo_mtime=np.array(os.path.getmtime(climo_path) if os.path.exists(climo_path) else 0.0),
    )
    print(f"  Saved exceedance stats to {path}")
    return q95_z, sigma_clim, base_rate


def load_or_build_window_exceedance_stats(
    model,
    train_dataset,
    shared_data,
    train_years,
    norm_stats,
    climo_by_doy,
    months: np.ndarray,
    years: np.ndarray,
    window_leads: Sequence[int],
    lead_indices: Sequence[int],
    mask_np: np.ndarray,
    device,
    checkpoint_path: str,
    progress_every: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train-year windowed q95/base-rate and forecast-error spread.

    The threshold is built on the exact windowed-mean truth quantity being
    evaluated. The spread is built from train-year forecast errors
    truth_week - mu_week, so it is checkpoint-dependent.
    """
    path = window_exceedance_cache_path(window_leads)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    norm_path = cfm.get_norm_stats_path(cfm.Config)
    climo_path = cfm.get_climatology_cache_path(cfm.Config)
    src_mtime = _source_mtime([norm_path, climo_path, cfm.Config.TRAINING_DATA_PATH, checkpoint_path])
    center_lead = window_center_lead(window_leads)
    train_year_set = set(int(y) for y in train_years)
    fold = int(getattr(cfm.Config, "CV_FOLD", cfm.Config.CV_TEST_OFFSETS[0]))

    def _cache_ok() -> bool:
        if not os.path.exists(path):
            return False
        if os.path.getmtime(path) < src_mtime:
            return False
        try:
            with np.load(path, allow_pickle=True) as data:
                cached_years = set(np.atleast_1d(data["train_years"]).astype(int).tolist())
                cached_leads = tuple(np.atleast_1d(data["window_leads"]).astype(int).tolist())
                cached_pred_leads = tuple(np.atleast_1d(data["prediction_leads"]).astype(int).tolist())
                return (
                    cached_years == train_year_set
                    and cached_leads == tuple(int(x) for x in window_leads)
                    and cached_pred_leads == cfm.prediction_leads(cfm.Config)
                    and int(data["cv_fold"]) == fold
                    and str(data["cv_split"].item()) == cfm.cv_split_tag(cfm.Config)
                    and abs(float(data["hi_mean"]) - float(norm_stats["hi_mean"])) < 1e-6
                    and abs(float(data["hi_std"]) - float(norm_stats["hi_std"])) < 1e-6
                )
        except Exception:
            return False

    if _cache_ok():
        print(f"Loading fold-safe windowed exceedance stats from {path}")
        with np.load(path, allow_pickle=False) as data:
            return (
                np.asarray(data["q95_z"], dtype=np.float32),
                np.asarray(data["sigma_error"], dtype=np.float32),
                np.asarray(data["base_rate"], dtype=np.float32),
            )

    print(
        "Building train-year-only windowed q95/base-rate/sigma stats "
        f"for leads {tuple(int(x) for x in window_leads)}..."
    )
    h, w = cfm.Config.IMAGE_SIZE
    truth_by_month: Dict[int, List[np.ndarray]] = defaultdict(list)
    err_by_month: Dict[int, List[np.ndarray]] = defaultdict(list)
    loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            mu_week, truth_week, t = predict_target_field(
                model,
                batch,
                device,
                target_mode="window",
                lead_indices=tuple(lead_indices),
                center_idx=cfm.center_lead_index(cfm.Config),
                image_size=cfm.Config.IMAGE_SIZE,
            )
            center_t = int(t) + int(center_lead)
            month = int(months[center_t])
            if month not in MJJAS_MONTHS:
                continue
            if int(years[center_t]) not in train_year_set:
                raise RuntimeError("Windowed stats builder leaked outside train years.")
            truth_week = truth_week.astype(np.float32)
            mu_week = mu_week.astype(np.float32)
            truth_week[~mask_np] = np.nan
            mu_week[~mask_np] = np.nan
            truth_by_month[month].append(truth_week)
            err_by_month[month].append(truth_week - mu_week)
            if (batch_idx + 1) % max(1, int(progress_every)) == 0:
                print(f"  window stats train inference processed {batch_idx + 1}/{len(train_dataset)}")

    q95_z = np.full((12, h, w), np.nan, dtype=np.float32)
    sigma_error = np.full((12, h, w), np.nan, dtype=np.float32)
    base_rate = np.full((12, h, w), np.nan, dtype=np.float32)
    for month in MJJAS_MONTHS:
        if not truth_by_month[month]:
            print(f"  Windowed train build check month={month}: no center-month samples")
            continue
        truth_stack = np.stack(truth_by_month[month], axis=2).astype(np.float32)
        with np.errstate(all="ignore"):
            q = np.nanpercentile(truth_stack, 95.0, axis=2).astype(np.float32)
        q[~mask_np] = np.nan
        valid = np.isfinite(truth_stack) & np.isfinite(q[:, :, None])
        exceed = truth_stack > q[:, :, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = exceed.sum(axis=2).astype(np.float32) / np.maximum(valid.sum(axis=2), 1)
        rate[~mask_np] = np.nan

        err_stack = np.stack(err_by_month[month], axis=2).astype(np.float32)
        with np.errstate(all="ignore"):
            sig = np.nanstd(err_stack, axis=2).astype(np.float32)
        sig = np.where(np.isfinite(sig), sig, np.nan)
        sig[mask_np] = np.maximum(sig[mask_np], 0.1)
        sig[~mask_np] = np.nan

        q95_z[month] = q
        base_rate[month] = rate.astype(np.float32)
        sigma_error[month] = sig.astype(np.float32)
        mean_rate = float(np.nanmean(rate))
        print(f"  Windowed train build check month={month}: mean exceedance rate={mean_rate:.4f}")
        if not (0.025 <= mean_rate <= 0.075):
            raise RuntimeError(
                f"Windowed train exceedance rate for month {month} is not near 5%: {mean_rate:.4f}"
            )
        del truth_stack, err_stack

    np.savez_compressed(
        path,
        q95_z=q95_z.astype(np.float32),
        sigma_error=sigma_error.astype(np.float32),
        base_rate=base_rate.astype(np.float32),
        train_years=np.array(sorted(train_year_set), dtype=np.int16),
        cv_fold=np.array(fold, dtype=np.int16),
        cv_split=np.array(cfm.cv_split_tag(cfm.Config), dtype=object),
        prediction_leads=np.array(cfm.prediction_leads(cfm.Config), dtype=np.int16),
        window_leads=np.array(tuple(int(x) for x in window_leads), dtype=np.int16),
        center_lead=np.array(center_lead, dtype=np.int16),
        hi_mean=np.array(float(norm_stats["hi_mean"]), dtype=np.float32),
        hi_std=np.array(float(norm_stats["hi_std"]), dtype=np.float32),
        months=np.array(MJJAS_MONTHS, dtype=np.int8),
        norm_stats_mtime=np.array(os.path.getmtime(norm_path) if os.path.exists(norm_path) else 0.0),
        climo_mtime=np.array(os.path.getmtime(climo_path) if os.path.exists(climo_path) else 0.0),
        checkpoint_mtime=np.array(os.path.getmtime(checkpoint_path) if os.path.exists(checkpoint_path) else 0.0),
    )
    print(f"  Saved windowed exceedance stats to {path}")
    return q95_z, sigma_error, base_rate


def normal_cdf(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import ndtr

        return ndtr(x).astype(np.float32)
    except Exception:
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(x, dtype=np.float32))
            return (0.5 * (1.0 + torch.erf(t / math.sqrt(2.0)))).numpy().astype(np.float32)


@dataclass
class ReliabilityStats:
    bins: int = 10

    def __post_init__(self):
        self.count = np.zeros(self.bins, dtype=np.float64)
        self.pred_sum = np.zeros(self.bins, dtype=np.float64)
        self.obs_sum = np.zeros(self.bins, dtype=np.float64)

    def update(self, prob: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> None:
        valid = mask & np.isfinite(prob) & np.isfinite(truth)
        if not np.any(valid):
            return
        p = np.clip(prob[valid].astype(np.float64), 0.0, 1.0)
        y = truth[valid].astype(np.float64)
        idx = np.minimum((p * self.bins).astype(np.int64), self.bins - 1)
        self.count += np.bincount(idx, minlength=self.bins)
        self.pred_sum += np.bincount(idx, weights=p, minlength=self.bins)
        self.obs_sum += np.bincount(idx, weights=y, minlength=self.bins)

    def table(self) -> List[Dict[str, float]]:
        rows = []
        for i in range(self.bins):
            n = self.count[i]
            rows.append({
                "bin": i,
                "lo": i / self.bins,
                "hi": (i + 1) / self.bins,
                "count": n,
                "mean_pred": self.pred_sum[i] / n if n > 0 else float("nan"),
                "obs_freq": self.obs_sum[i] / n if n > 0 else float("nan"),
            })
        return rows

    def slope_ece(self) -> Tuple[float, float]:
        rows = self.table()
        x = np.array([r["mean_pred"] for r in rows], dtype=np.float64)
        y = np.array([r["obs_freq"] for r in rows], dtype=np.float64)
        w = np.array([r["count"] for r in rows], dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y) & (w > 0)
        if valid.sum() < 2:
            slope = float("nan")
        else:
            x0 = np.average(x[valid], weights=w[valid])
            y0 = np.average(y[valid], weights=w[valid])
            denom = np.sum(w[valid] * (x[valid] - x0) ** 2)
            slope = float(np.sum(w[valid] * (x[valid] - x0) * (y[valid] - y0)) / denom) if denom > 0 else float("nan")
        total = np.sum(w[valid])
        ece = float(np.sum(w[valid] * np.abs(x[valid] - y[valid])) / total) if total > 0 else float("nan")
        return slope, ece


def reliability_from_arrays(prob: np.ndarray, truth: np.ndarray) -> Tuple[float, float, float]:
    p = np.asarray(prob, dtype=np.float32).reshape(-1)
    y = np.asarray(truth, dtype=np.float32).reshape(-1)
    valid = np.isfinite(p) & np.isfinite(y)
    if valid.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    rel = ReliabilityStats()
    rel.update(p[valid], y[valid], np.ones(int(valid.sum()), dtype=bool))
    slope, ece = rel.slope_ece()
    brier = float(np.mean((np.clip(p[valid], 0.0, 1.0) - y[valid]) ** 2))
    return slope, ece, brier


def _fit_positive_platt_params(
    x: np.ndarray,
    y: np.ndarray,
    steps: int = 600,
    lr: float = 0.05,
    l2: float = 1e-4,
) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 100 or np.unique(y).size < 2:
        raise RuntimeError("Not enough valid positive/negative samples for Platt calibration.")
    base = np.clip(float(np.mean(y)), 1e-5, 1.0 - 1e-5)
    b = math.log(base / (1.0 - base))
    theta = 0.0  # a = exp(theta), enforcing monotonic increasing calibration.
    for _ in range(int(steps)):
        a = math.exp(float(np.clip(theta, -10.0, 10.0)))
        logits = np.clip(a * x + b, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        err = p - y
        grad_b = float(np.mean(err))
        grad_a = float(np.mean(err * x)) + float(l2) * a
        b -= float(lr) * grad_b
        theta -= float(lr) * grad_a * a
    a = math.exp(float(np.clip(theta, -10.0, 10.0)))
    return float(a), float(b)


def _fit_isotonic_arrays(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 100 or np.unique(y).size < 2:
        raise RuntimeError("Not enough valid positive/negative samples for isotonic calibration.")
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    try:
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(x_sorted, y_sorted)
        return (
            np.asarray(iso.X_thresholds_, dtype=np.float32),
            np.asarray(iso.y_thresholds_, dtype=np.float32),
        )
    except Exception:
        # Minimal PAVA fallback: sufficient for monotonic calibration if sklearn is unavailable.
        blocks: List[List[float]] = []
        for val in y_sorted:
            blocks.append([float(val), 1.0])
            while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
                v1, n1 = blocks.pop()
                v0, n0 = blocks.pop()
                blocks.append([(v0 * n0 + v1 * n1) / (n0 + n1), n0 + n1])
        y_fit = np.empty_like(y_sorted, dtype=np.float64)
        pos = 0
        for value, count in blocks:
            n = int(count)
            y_fit[pos:pos + n] = value
            pos += n
        unique_x, inverse = np.unique(x_sorted, return_inverse=True)
        y_thr = np.zeros_like(unique_x, dtype=np.float64)
        counts = np.bincount(inverse)
        sums = np.bincount(inverse, weights=y_fit)
        valid_counts = counts > 0
        y_thr[valid_counts] = sums[valid_counts] / counts[valid_counts]
        return unique_x.astype(np.float32), np.clip(y_thr, 0.0, 1.0).astype(np.float32)


@dataclass
class MarginMonotonicCalibrator:
    method: str
    pooled: Any
    by_month: Dict[int, Any]
    calibration_split: str
    n_samples: int
    event_rate: float
    calibration_slope: float
    calibration_ece: float
    calibration_brier: float

    def _predict_model(self, model: Any, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        if self.method == "platt":
            a, b = model
            logits = np.clip(float(a) * x_arr + float(b), -30.0, 30.0)
            return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
        x_thr, y_thr = model
        return np.interp(
            x_arr,
            np.asarray(x_thr, dtype=np.float32),
            np.asarray(y_thr, dtype=np.float32),
            left=float(y_thr[0]),
            right=float(y_thr[-1]),
        ).astype(np.float32)

    def predict_scores(self, x: np.ndarray, months: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32).reshape(-1)
        month_arr = np.asarray(months, dtype=np.int16).reshape(-1)
        out = np.empty_like(x_arr, dtype=np.float32)
        for month in np.unique(month_arr):
            mask = month_arr == month
            model = self.by_month.get(int(month), self.pooled)
            out[mask] = self._predict_model(model, x_arr[mask])
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def predict_grid(
        self,
        mu_z: np.ndarray,
        q95: np.ndarray,
        sigma: np.ndarray,
        month: int,
        mask: np.ndarray,
    ) -> np.ndarray:
        valid = mask & np.isfinite(mu_z) & np.isfinite(q95) & np.isfinite(sigma)
        out = np.full(mask.shape, np.nan, dtype=np.float32)
        if not np.any(valid):
            return out
        score = (mu_z[valid] - q95[valid]) / np.maximum(sigma[valid], 0.1)
        pred = self.predict_scores(score, np.full(score.shape, int(month), dtype=np.int16))
        if pred.shape != score.shape or not np.all(np.isfinite(pred)):
            raise RuntimeError(
                f"{self.method} calibrator produced non-finite or misaligned probabilities "
                f"for month={month}: pred_shape={pred.shape}, score_shape={score.shape}"
            )
        out[valid] = pred
        return out

    def coefficient_rows(self) -> List[Dict[str, object]]:
        rows = [{
            "method": self.method,
            "scope": "pooled",
            "month": "",
            "model": repr(self.pooled),
            "calibration_split": self.calibration_split,
            "n_samples": self.n_samples,
            "event_rate": self.event_rate,
            "calibration_slope": self.calibration_slope,
            "calibration_ece": self.calibration_ece,
            "calibration_brier": self.calibration_brier,
        }]
        for month, model in sorted(self.by_month.items()):
            rows.append({
                "method": self.method,
                "scope": "month",
                "month": month,
                "model": repr(model),
                "calibration_split": self.calibration_split,
                "n_samples": self.n_samples,
                "event_rate": self.event_rate,
                "calibration_slope": self.calibration_slope,
                "calibration_ece": self.calibration_ece,
                "calibration_brier": self.calibration_brier,
            })
        return rows


def fit_margin_monotonic_calibrator(
    x: np.ndarray,
    y: np.ndarray,
    months: np.ndarray,
    method: str,
    calibration_split: str,
    min_month_samples: int,
    min_month_positives: int,
) -> MarginMonotonicCalibrator:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    months = np.asarray(months, dtype=np.int16).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(months)
    x = x[valid]
    y = y[valid]
    months = months[valid]
    if x.size < 100 or np.unique(y).size < 2:
        raise RuntimeError(f"Not enough calibration samples for {method}.")

    if method == "platt":
        pooled = _fit_positive_platt_params(x, y)
    elif method == "isotonic":
        pooled = _fit_isotonic_arrays(x, y)
    else:
        raise ValueError(f"Unsupported calibrator method: {method}")

    by_month: Dict[int, Any] = {}
    for month in MJJAS_MONTHS:
        m = months == int(month)
        n_pos = int(np.sum(y[m] > 0.5))
        n_neg = int(np.sum(y[m] <= 0.5))
        if int(np.sum(m)) < int(min_month_samples) or n_pos < int(min_month_positives) or n_neg < int(min_month_positives):
            continue
        try:
            by_month[int(month)] = (
                _fit_positive_platt_params(x[m], y[m])
                if method == "platt"
                else _fit_isotonic_arrays(x[m], y[m])
            )
        except RuntimeError:
            continue

    cal = MarginMonotonicCalibrator(
        method=method,
        pooled=pooled,
        by_month=by_month,
        calibration_split=calibration_split,
        n_samples=int(x.size),
        event_rate=float(np.mean(y)),
        calibration_slope=float("nan"),
        calibration_ece=float("nan"),
        calibration_brier=float("nan"),
    )
    p = cal.predict_scores(x, months)
    slope, ece, brier = reliability_from_arrays(p, y)
    cal.calibration_slope = slope
    cal.calibration_ece = ece
    cal.calibration_brier = brier
    return cal


def choose_monotonic_calibrator(
    candidates: Mapping[str, MarginMonotonicCalibrator],
    requested: str,
) -> MarginMonotonicCalibrator:
    available = {k: v for k, v in candidates.items() if v is not None}
    if requested != "auto":
        if requested not in available:
            raise RuntimeError(f"Requested calibrator {requested!r} could not be fit.")
        return available[requested]
    if not available:
        raise RuntimeError("No monotonic calibrator candidates could be fit.")
    return min(
        available.values(),
        key=lambda c: (
            float(c.calibration_ece) if np.isfinite(c.calibration_ece) else float("inf"),
            abs(float(c.calibration_slope) - 1.0) if np.isfinite(c.calibration_slope) else float("inf"),
        ),
    )


def score_calibrator_on_pairs(
    name: str,
    cal: MarginMonotonicCalibrator,
    x: np.ndarray,
    y: np.ndarray,
    months: np.ndarray,
) -> Dict[str, object]:
    prob = cal.predict_scores(x, months)
    slope, ece, brier = reliability_from_arrays(prob, y)
    return {
        "candidate": name,
        "calibration_split": cal.calibration_split,
        "n_samples": int(np.asarray(y).size),
        "event_rate": float(np.mean(y)) if np.asarray(y).size else float("nan"),
        "inner_selection_slope": slope,
        "inner_selection_ece": ece,
        "inner_selection_brier": brier,
    }


def fit_inner_selected_monotonic_calibrator(
    x: np.ndarray,
    y: np.ndarray,
    months: np.ndarray,
    pair_years: np.ndarray,
    calibration_split: str,
    requested_calibrator: str,
    min_month_samples: int,
    min_month_positives: int,
) -> Tuple[MarginMonotonicCalibrator, Dict[str, MarginMonotonicCalibrator], List[Dict[str, object]], str]:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    months = np.asarray(months, dtype=np.int16).reshape(-1)
    pair_years = np.asarray(pair_years, dtype=np.int16).reshape(-1)

    full_candidates: Dict[str, MarginMonotonicCalibrator] = {}
    for method in ("platt", "isotonic"):
        try:
            full_candidates[method] = fit_margin_monotonic_calibrator(
                x,
                y,
                months,
                method=method,
                calibration_split=calibration_split,
                # Use pooled monotonic maps for the selected forecast so global ROC-AUC
                # remains invariant under calibration.
                min_month_samples=10**12,
                min_month_positives=10**12,
            )
        except Exception as exc:
            print(f"  full {method} calibrator failed to fit: {exc}")

    if requested_calibrator != "auto":
        if requested_calibrator not in full_candidates:
            raise RuntimeError(f"Requested calibrator {requested_calibrator!r} could not be fit.")
        return (
            full_candidates[requested_calibrator],
            full_candidates,
            [],
            f"requested {requested_calibrator}",
        )

    unique_years = np.array(sorted(set(int(v) for v in pair_years)), dtype=np.int16)
    if unique_years.size < 3:
        if "platt" not in full_candidates:
            raise RuntimeError("Too few calibration years for inner selection and Platt could not be fit.")
        return full_candidates["platt"], full_candidates, [], "defaulted to Platt: fewer than 3 calibration years"

    n_select = max(1, int(math.ceil(0.25 * float(unique_years.size))))
    select_years = set(int(v) for v in unique_years[-n_select:])
    fit_mask = np.array([int(v) not in select_years for v in pair_years], dtype=bool)
    select_mask = ~fit_mask
    if (
        int(np.sum(fit_mask)) < 100
        or int(np.sum(select_mask)) < 100
        or np.unique(y[fit_mask]).size < 2
        or np.unique(y[select_mask]).size < 2
    ):
        if "platt" not in full_candidates:
            raise RuntimeError("Inner calibration split too thin and Platt could not be fit.")
        return full_candidates["platt"], full_candidates, [], "defaulted to Platt: inner split too thin"

    inner_rows: List[Dict[str, object]] = []
    for method in ("platt", "isotonic"):
        try:
            inner_cal = fit_margin_monotonic_calibrator(
                x[fit_mask],
                y[fit_mask],
                months[fit_mask],
                method=method,
                calibration_split=f"{calibration_split}_inner_fit",
                min_month_samples=10**12,
                min_month_positives=10**12,
            )
            row = score_calibrator_on_pairs(
                method,
                inner_cal,
                x[select_mask],
                y[select_mask],
                months[select_mask],
            )
            row["inner_fit_years"] = " ".join(str(int(v)) for v in unique_years if int(v) not in select_years)
            row["inner_selection_years"] = " ".join(str(v) for v in sorted(select_years))
            inner_rows.append(row)
        except Exception as exc:
            print(f"  inner {method} calibrator failed to fit/score: {exc}")

    row_by_name = {str(row["candidate"]): row for row in inner_rows}
    if "platt" not in full_candidates:
        raise RuntimeError("Platt calibrator could not be fit; refusing to auto-select isotonic.")
    selected = "platt"
    reason = "Platt default: isotonic did not beat Platt inner-selection ECE by at least 0.005"
    if "isotonic" in full_candidates and "platt" in row_by_name and "isotonic" in row_by_name:
        platt_ece = float(row_by_name["platt"]["inner_selection_ece"])
        iso_ece = float(row_by_name["isotonic"]["inner_selection_ece"])
        if np.isfinite(platt_ece) and np.isfinite(iso_ece) and iso_ece <= platt_ece - 0.005:
            selected = "isotonic"
            reason = (
                "Isotonic selected: inner-selection ECE improved by "
                f"{platt_ece - iso_ece:.4f} >= 0.005"
            )
    return full_candidates[selected], full_candidates, inner_rows, reason


@dataclass
class MetricAccumulator:
    name: str
    hist_bins: int = 10001

    def __post_init__(self):
        self.brier_sum = 0.0
        self.count = 0.0
        self.truth_pos = 0.0
        self.rel = ReliabilityStats()
        self.hist_pos = np.zeros(self.hist_bins, dtype=np.float64)
        self.hist_neg = np.zeros(self.hist_bins, dtype=np.float64)
        self.auc_hist_pos = np.zeros(self.hist_bins, dtype=np.float64)
        self.auc_hist_neg = np.zeros(self.hist_bins, dtype=np.float64)

    def update(self, prob: np.ndarray, truth: np.ndarray, mask: np.ndarray, auc_score: Optional[np.ndarray] = None) -> None:
        valid = mask & np.isfinite(prob) & np.isfinite(truth)
        if auc_score is not None:
            valid = valid & np.isfinite(auc_score)
        if not np.any(valid):
            return
        p = np.clip(prob[valid].astype(np.float64), 0.0, 1.0)
        y = truth[valid].astype(np.float64)
        self.brier_sum += float(np.sum((p - y) ** 2))
        self.count += float(p.size)
        self.truth_pos += float(np.sum(y))
        self.rel.update(prob, truth, mask)
        idx = np.minimum((p * (self.hist_bins - 1)).round().astype(np.int64), self.hist_bins - 1)
        self.hist_pos += np.bincount(idx, weights=y, minlength=self.hist_bins)
        self.hist_neg += np.bincount(idx, weights=1.0 - y, minlength=self.hist_bins)
        auc_values = p if auc_score is None else np.clip(auc_score[valid].astype(np.float64), 0.0, 1.0)
        auc_idx = np.minimum((auc_values * (self.hist_bins - 1)).round().astype(np.int64), self.hist_bins - 1)
        self.auc_hist_pos += np.bincount(auc_idx, weights=y, minlength=self.hist_bins)
        self.auc_hist_neg += np.bincount(auc_idx, weights=1.0 - y, minlength=self.hist_bins)

    def brier(self) -> float:
        return self.brier_sum / self.count if self.count > 0 else float("nan")

    def aucs(self) -> Tuple[float, float]:
        pos_total = float(self.auc_hist_pos.sum())
        neg_total = float(self.auc_hist_neg.sum())
        if pos_total <= 0 or neg_total <= 0:
            return float("nan"), float("nan")
        tp = np.cumsum(self.auc_hist_pos[::-1])
        fp = np.cumsum(self.auc_hist_neg[::-1])
        tpr = np.r_[0.0, tp / pos_total, 1.0]
        fpr = np.r_[0.0, fp / neg_total, 1.0]
        roc_auc = _trapezoid_integral(tpr, fpr)
        precision = tp / np.maximum(tp + fp, 1.0)
        recall = tp / pos_total
        pr_auc = _trapezoid_integral(np.r_[precision[0], precision], np.r_[0.0, recall])
        return roc_auc, pr_auc

    def threshold_scores(self, thresholds: Sequence[float]) -> Dict[str, float]:
        out = {}
        pos_total = float(self.hist_pos.sum())
        for th in thresholds:
            idx = int(np.clip(round(th * (self.hist_bins - 1)), 0, self.hist_bins - 1))
            hits = float(self.hist_pos[idx:].sum())
            false_alarms = float(self.hist_neg[idx:].sum())
            misses = max(pos_total - hits, 0.0)
            out[f"hit_rate_{th:g}"] = hits / (hits + misses) if (hits + misses) > 0 else float("nan")
            out[f"false_alarm_ratio_{th:g}"] = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else float("nan")
        return out


class EvaluationAccumulator:
    def __init__(self, model_names: Sequence[str], region_masks_map: Mapping[str, np.ndarray]):
        self.metrics = {name: MetricAccumulator(name) for name in model_names}
        self.monthly = defaultdict(lambda: {name: MetricAccumulator(name) for name in model_names})
        self.region_abs = {
            name: {region: 0.0 for region in region_masks_map}
            for name in model_names
        }
        self.region_count = {
            name: {region: 0 for region in region_masks_map}
            for name in model_names
        }
        self.region_masks = region_masks_map

    def update(
        self,
        name: str,
        prob: np.ndarray,
        truth: np.ndarray,
        mask: np.ndarray,
        month: int,
        auc_score: Optional[np.ndarray] = None,
    ) -> None:
        self.metrics[name].update(prob, truth, mask, auc_score=auc_score)
        self.monthly[int(month)][name].update(prob, truth, mask, auc_score=auc_score)
        for region, rmask in self.region_masks.items():
            valid = mask & rmask & np.isfinite(prob) & np.isfinite(truth)
            if not np.any(valid):
                continue
            pred_count = float(np.sum(prob[valid]))
            obs_count = float(np.sum(truth[valid]))
            self.region_abs[name][region] += abs(pred_count - obs_count)
            self.region_count[name][region] += 1

    def summary_rows(self, reference_name: str) -> List[Dict[str, float]]:
        ref = self.metrics[reference_name].brier()
        rows = []
        for name, acc in self.metrics.items():
            roc_auc, pr_auc = acc.aucs()
            slope, ece = acc.rel.slope_ece()
            brier = acc.brier()
            row = {
                "model": name,
                "valid_count": acc.count,
                "brier": brier,
                "bss_vs_monthly_climo": 1.0 - brier / ref if np.isfinite(ref) and ref > 0 else float("nan"),
                "base_rate": acc.truth_pos / acc.count if acc.count > 0 else float("nan"),
                "reliability_slope": slope,
                "ece": ece,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
            row.update(acc.threshold_scores((0.1, 0.2, 0.3, 0.5)))
            rows.append(row)
        return rows

    def monthly_rows(self, reference_name: str) -> List[Dict[str, float]]:
        rows = []
        for month in sorted(self.monthly):
            ref = self.monthly[month][reference_name].brier()
            for name, acc in self.monthly[month].items():
                roc_auc, pr_auc = acc.aucs()
                slope, ece = acc.rel.slope_ece()
                brier = acc.brier()
                rows.append({
                    "month": month,
                    "model": name,
                    "brier": brier,
                    "bss_vs_monthly_climo": 1.0 - brier / ref if np.isfinite(ref) and ref > 0 else float("nan"),
                    "reliability_slope": slope,
                    "ece": ece,
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                })
        return rows

    def region_rows(self) -> List[Dict[str, float]]:
        rows = []
        for name, by_region in self.region_abs.items():
            for region, value in by_region.items():
                denom = max(self.region_count[name].get(region, 0), 1)
                rows.append({
                    "model": name,
                    "region": region,
                    "expected_count_mae": value / denom,
                    "n_samples": self.region_count[name].get(region, 0),
                })
        return rows


def _roc_auc_from_hist(pos_hist: np.ndarray, neg_hist: np.ndarray) -> float:
    pos_total = float(np.sum(pos_hist))
    neg_total = float(np.sum(neg_hist))
    if pos_total <= 0.0 or neg_total <= 0.0:
        return float("nan")
    tp = np.cumsum(np.asarray(pos_hist, dtype=np.float64)[::-1])
    fp = np.cumsum(np.asarray(neg_hist, dtype=np.float64)[::-1])
    tpr = np.r_[0.0, tp / pos_total, 1.0]
    fpr = np.r_[0.0, fp / neg_total, 1.0]
    return _trapezoid_integral(tpr, fpr)


def year_block_bootstrap_incremental_comparison(
    by_year: Mapping[int, EvaluationAccumulator],
    reference_name: str,
    baseline_name: str,
    candidate_name: str,
    reps: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Bootstrap whole test years for candidate-minus-baseline BSS and ROC-AUC."""
    year_values = np.array(sorted(int(year) for year in by_year), dtype=np.int16)
    if year_values.size < 2:
        raise RuntimeError("Year-block bootstrap requires at least two evaluation years.")
    model_names = (reference_name, baseline_name, candidate_name)
    per_year_rows: List[Dict[str, object]] = []
    for year in year_values:
        rows = {
            row["model"]: row
            for row in by_year[int(year)].summary_rows(reference_name)
        }
        per_year_rows.append({
            "year": int(year),
            "baseline_bss": rows[baseline_name]["bss_vs_monthly_climo"],
            "candidate_bss": rows[candidate_name]["bss_vs_monthly_climo"],
            "delta_bss_candidate_minus_baseline": (
                rows[candidate_name]["bss_vs_monthly_climo"]
                - rows[baseline_name]["bss_vs_monthly_climo"]
            ),
            "baseline_roc_auc": rows[baseline_name]["roc_auc"],
            "candidate_roc_auc": rows[candidate_name]["roc_auc"],
            "delta_roc_auc_candidate_minus_baseline": (
                rows[candidate_name]["roc_auc"] - rows[baseline_name]["roc_auc"]
            ),
            "valid_count": rows[candidate_name]["valid_count"],
        })

    brier_sum = {
        name: np.array(
            [by_year[int(year)].metrics[name].brier_sum for year in year_values],
            dtype=np.float64,
        )
        for name in model_names
    }
    counts = {
        name: np.array(
            [by_year[int(year)].metrics[name].count for year in year_values],
            dtype=np.float64,
        )
        for name in model_names
    }
    auc_pos = {
        name: np.stack(
            [by_year[int(year)].metrics[name].auc_hist_pos for year in year_values],
            axis=0,
        )
        for name in (baseline_name, candidate_name)
    }
    auc_neg = {
        name: np.stack(
            [by_year[int(year)].metrics[name].auc_hist_neg for year in year_values],
            axis=0,
        )
        for name in (baseline_name, candidate_name)
    }

    def compare(weights: np.ndarray) -> Tuple[float, float]:
        brier = {
            name: float(weights @ brier_sum[name]) / max(float(weights @ counts[name]), 1.0)
            for name in model_names
        }
        ref = brier[reference_name]
        delta_bss = (
            (1.0 - brier[candidate_name] / ref)
            - (1.0 - brier[baseline_name] / ref)
        )
        auc = {
            name: _roc_auc_from_hist(
                np.tensordot(weights, auc_pos[name], axes=(0, 0)),
                np.tensordot(weights, auc_neg[name], axes=(0, 0)),
            )
            for name in (baseline_name, candidate_name)
        }
        return float(delta_bss), float(auc[candidate_name] - auc[baseline_name])

    point_weights = np.ones(year_values.size, dtype=np.float64)
    point_delta_bss, point_delta_auc = compare(point_weights)
    rng = np.random.default_rng(seed)
    n_reps = max(1, int(reps))
    bootstrap = np.empty((n_reps, 2), dtype=np.float64)
    for rep in range(n_reps):
        selected = rng.integers(0, year_values.size, size=year_values.size)
        weights = np.bincount(selected, minlength=year_values.size).astype(np.float64)
        bootstrap[rep] = compare(weights)

    summary_rows = []
    for metric_name, estimate, values in (
        ("delta_bss_candidate_minus_baseline", point_delta_bss, bootstrap[:, 0]),
        ("delta_roc_auc_candidate_minus_baseline", point_delta_auc, bootstrap[:, 1]),
    ):
        finite = values[np.isfinite(values)]
        summary_rows.append({
            "metric": metric_name,
            "estimate": estimate,
            "ci_2.5": float(np.quantile(finite, 0.025)) if finite.size else float("nan"),
            "ci_97.5": float(np.quantile(finite, 0.975)) if finite.size else float("nan"),
            "probability_gt_zero": float(np.mean(finite > 0.0)) if finite.size else float("nan"),
            "bootstrap_reps": int(finite.size),
            "block": "test_year",
            "baseline_model": baseline_name,
            "candidate_model": candidate_name,
        })
    return per_year_rows, summary_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_reliability(path: Path, rel_tables: Mapping[str, List[Mapping[str, float]]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, rows in rel_tables.items():
        x = np.array([r["mean_pred"] for r in rows], dtype=np.float64)
        y = np.array([r["obs_freq"] for r in rows], dtype=np.float64)
        n = np.array([r["count"] for r in rows], dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y) & (n > 0)
        if np.any(valid):
            ax.plot(x[valid], y[valid], marker="o", label=name)
    ax.set_xlabel("Predicted exceedance probability")
    ax.set_ylabel("Observed exceedance frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def find_run_start(runs: Sequence[Tuple[int, int]], t: int) -> int:
    for start, end in runs:
        if start <= t <= end:
            return start
    raise RuntimeError(f"Could not locate continuous season for t={t}")


def trailing_exceedance_probability(
    heat: np.ndarray,
    t: int,
    target_t: int,
    window: int,
    run_start: int,
    q95_z: np.ndarray,
    months: np.ndarray,
    norm_stats: Mapping[str, torch.Tensor],
    land_mask: np.ndarray,
) -> np.ndarray:
    lo = max(run_start, t - int(window) + 1)
    hist = np.arange(lo, t + 1, dtype=np.int64)
    if hist.size == 0:
        return np.zeros_like(land_mask, dtype=np.float32)
    if np.max(hist) >= target_t:
        raise RuntimeError("Persistence baseline attempted to use target/future information.")

    count = np.zeros(land_mask.shape, dtype=np.float32)
    exceed = np.zeros(land_mask.shape, dtype=np.float32)
    for ti in hist:
        month = int(months[ti])
        field_z = _normalize_field(np.asarray(heat[:, :, ti]), norm_stats)
        threshold = q95_z[month]
        valid = land_mask & np.isfinite(field_z) & np.isfinite(threshold)
        count[valid] += 1.0
        exceed[valid] += (field_z[valid] > threshold[valid]).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = exceed / np.maximum(count, 1.0)
    out[~land_mask] = np.nan
    return out.astype(np.float32)


def trailing_windowed_exceedance_probability(
    heat: np.ndarray,
    t: int,
    history_windows: int,
    run_start: int,
    q95_z: np.ndarray,
    months: np.ndarray,
    norm_stats: Mapping[str, torch.Tensor],
    land_mask: np.ndarray,
    window_leads: Sequence[int],
    center_lead: int,
) -> np.ndarray:
    latest_hist_init = int(t) - max(int(x) for x in window_leads)
    if latest_hist_init < run_start:
        out = np.zeros_like(land_mask, dtype=np.float32)
        out[~land_mask] = np.nan
        return out
    lo = max(run_start, latest_hist_init - int(history_windows) + 1)
    hist_inits = np.arange(lo, latest_hist_init + 1, dtype=np.int64)
    if hist_inits.size == 0:
        out = np.zeros_like(land_mask, dtype=np.float32)
        out[~land_mask] = np.nan
        return out
    if max(int(h) + max(int(x) for x in window_leads) for h in hist_inits) > int(t):
        raise RuntimeError("Windowed persistence attempted to use post-init information.")

    count = np.zeros(land_mask.shape, dtype=np.float32)
    exceed = np.zeros(land_mask.shape, dtype=np.float32)
    for hist_t in hist_inits:
        center_t = int(hist_t) + int(center_lead)
        month = int(months[center_t])
        if month not in MJJAS_MONTHS:
            continue
        field_z = window_mean_z_from_shared(heat, int(hist_t), window_leads, norm_stats)
        threshold = q95_z[month]
        valid = land_mask & np.isfinite(field_z) & np.isfinite(threshold)
        count[valid] += 1.0
        exceed[valid] += (field_z[valid] > threshold[valid]).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = exceed / np.maximum(count, 1.0)
    out[~land_mask] = np.nan
    return out.astype(np.float32)


@dataclass
class PooledLogistic:
    mean: float
    std: float
    coef: float
    intercept: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (x.astype(np.float32) - self.mean) / (self.std + 1e-8)
        logits = np.clip(self.intercept + self.coef * z, -30.0, 30.0)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


@dataclass
class ModelOutputLogisticCalibrator:
    feature_names: Tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray
    intercept: float
    calibration_split: str
    n_samples: int
    event_rate: float

    def predict_features(self, x: np.ndarray) -> np.ndarray:
        z = (x.astype(np.float32) - self.mean) / (self.std + 1e-8)
        logits = np.clip(self.intercept + z @ self.coef, -30.0, 30.0)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def predict_grid(
        self,
        mu_z: np.ndarray,
        q95: np.ndarray,
        sigma: np.ndarray,
        month: int,
        region_features: Mapping[str, np.ndarray],
        mask: np.ndarray,
    ) -> np.ndarray:
        valid = mask & np.isfinite(mu_z) & np.isfinite(q95) & np.isfinite(sigma)
        out = np.full(mask.shape, np.nan, dtype=np.float32)
        flat_idx = np.flatnonzero(valid.ravel())
        if flat_idx.size == 0:
            return out
        x = model_output_calibration_features(mu_z, q95, sigma, month, region_features, flat_idx)
        out.ravel()[flat_idx] = self.predict_features(x)
        return out

    def coefficient_rows(self) -> List[Dict[str, object]]:
        rows = [{
            "feature": "intercept",
            "coef": self.intercept,
            "feature_mean": "",
            "feature_std": "",
            "calibration_split": self.calibration_split,
            "n_samples": self.n_samples,
            "event_rate": self.event_rate,
        }]
        for name, coef, mean, std in zip(self.feature_names, self.coef, self.mean, self.std):
            rows.append({
                "feature": name,
                "coef": float(coef),
                "feature_mean": float(mean),
                "feature_std": float(std),
                "calibration_split": self.calibration_split,
                "n_samples": self.n_samples,
                "event_rate": self.event_rate,
            })
        return rows


@dataclass
class ClimatologyAnchoredLogisticCalibrator:
    feature_names: Tuple[str, ...]
    coef: np.ndarray
    calibration_split: str
    n_samples: int
    event_rate: float
    mean_base_rate: float

    def predict_features(self, x: np.ndarray, base_rate: np.ndarray) -> np.ndarray:
        base = np.clip(np.asarray(base_rate, dtype=np.float32), 1e-5, 1.0 - 1e-5)
        fixed_offset = np.log(base / (1.0 - base))
        logits = np.clip(fixed_offset + x.astype(np.float32) @ self.coef, -30.0, 30.0)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def coefficient_rows(self) -> List[Dict[str, object]]:
        rows = [{
            "feature": "fixed_logit_train_climatology",
            "coef": 1.0,
            "feature_mean": self.mean_base_rate,
            "feature_std": "fixed_offset_no_intercept",
            "calibration_split": self.calibration_split,
            "n_samples": self.n_samples,
            "event_rate": self.event_rate,
        }]
        for name, coef in zip(self.feature_names, self.coef):
            rows.append({
                "feature": name,
                "coef": float(coef),
                "feature_mean": "",
                "feature_std": "",
                "calibration_split": self.calibration_split,
                "n_samples": self.n_samples,
                "event_rate": self.event_rate,
            })
        return rows


@dataclass
class NestedYearShrinkageCalibrator:
    feature_names: Tuple[str, ...]
    coef: np.ndarray
    alpha: float
    selected_l2: float
    calibration_split: str
    n_samples: int
    event_rate: float
    mean_base_rate: float

    def predict_features(self, x: np.ndarray, base_rate: np.ndarray) -> np.ndarray:
        base = np.clip(np.asarray(base_rate, dtype=np.float32), 1e-5, 1.0 - 1e-5)
        fixed_offset = np.log(base / (1.0 - base))
        logits = np.clip(
            fixed_offset + float(self.alpha) * (x.astype(np.float32) @ self.coef),
            -30.0,
            30.0,
        )
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def coefficient_rows(self) -> List[Dict[str, object]]:
        rows = [{
            "feature": "fixed_logit_train_climatology",
            "coef": 1.0,
            "feature_mean": self.mean_base_rate,
            "feature_std": "fixed_offset_no_intercept",
            "calibration_split": self.calibration_split,
            "n_samples": self.n_samples,
            "event_rate": self.event_rate,
            "alpha": self.alpha,
            "selected_l2": self.selected_l2,
        }]
        for name, coef in zip(self.feature_names, self.coef):
            rows.append({
                "feature": name,
                "coef": float(coef),
                "feature_mean": "",
                "feature_std": "",
                "calibration_split": self.calibration_split,
                "n_samples": self.n_samples,
                "event_rate": self.event_rate,
                "alpha": self.alpha,
                "selected_l2": self.selected_l2,
            })
        return rows


def calibration_feature_names(region_names: Sequence[str]) -> Tuple[str, ...]:
    return (
        "stage1_logit",
        "margin_z",
        "mu_z",
        "sigma_clim",
        *[f"month_{m}" for m in MJJAS_MONTHS],
        *[f"region_{name}" for name in region_names],
    )


def model_output_calibration_features(
    mu_z: np.ndarray,
    q95: np.ndarray,
    sigma: np.ndarray,
    month: int,
    region_features: Mapping[str, np.ndarray],
    flat_idx: np.ndarray,
) -> np.ndarray:
    mu = mu_z.ravel()[flat_idx].astype(np.float32)
    q = q95.ravel()[flat_idx].astype(np.float32)
    sig = np.maximum(sigma.ravel()[flat_idx].astype(np.float32), 0.1)
    margin = mu - q
    stage1_logit = np.clip(margin / sig, -10.0, 10.0)
    month_flags = np.zeros((flat_idx.size, len(MJJAS_MONTHS)), dtype=np.float32)
    if int(month) in MJJAS_MONTHS:
        month_flags[:, MJJAS_MONTHS.index(int(month))] = 1.0
    region_flags = np.column_stack([
        rmask.ravel()[flat_idx].astype(np.float32)
        for rmask in region_features.values()
    ]) if region_features else np.empty((flat_idx.size, 0), dtype=np.float32)
    return np.column_stack([
        stage1_logit,
        margin,
        mu,
        sig,
        month_flags,
        region_flags,
    ]).astype(np.float32)


def fit_model_output_logistic_calibrator(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str],
    calibration_split: str,
    steps: int,
    lr: float,
    l2: float,
) -> ModelOutputLogisticCalibrator:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    valid = np.all(np.isfinite(x), axis=1) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.shape[0] < 100 or np.unique(y).size < 2:
        raise RuntimeError("Not enough valid positive/negative samples to fit calibrated logistic model.")

    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-8
    z = (x - mean) / std
    coef = np.zeros(z.shape[1], dtype=np.float64)
    base = np.clip(y.mean(), 1e-5, 1.0 - 1e-5)
    intercept = float(math.log(base / (1.0 - base)))

    for _ in range(int(steps)):
        logits = np.clip(intercept + z @ coef, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        err = p - y
        intercept -= float(lr) * float(err.mean())
        coef -= float(lr) * ((z.T @ err) / z.shape[0] + float(l2) * coef)

    return ModelOutputLogisticCalibrator(
        feature_names=tuple(feature_names),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        coef=coef.astype(np.float32),
        intercept=float(intercept),
        calibration_split=calibration_split,
        n_samples=int(y.size),
        event_rate=float(y.mean()),
    )


def fit_climatology_anchored_logistic_calibrator(
    features: np.ndarray,
    labels: np.ndarray,
    base_rates: np.ndarray,
    feature_names: Sequence[str],
    calibration_split: str,
    steps: int,
    lr: float,
    l2: float,
) -> ClimatologyAnchoredLogisticCalibrator:
    """Fit slopes only; train-climatology logit remains the fixed offset."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    base = np.asarray(base_rates, dtype=np.float64)
    valid = np.all(np.isfinite(x), axis=1) & np.isfinite(y) & np.isfinite(base)
    x = x[valid]
    y = y[valid]
    base = np.clip(base[valid], 1e-5, 1.0 - 1e-5)
    if x.shape[0] < 100 or np.unique(y).size < 2:
        raise RuntimeError("Not enough valid positive/negative samples to fit climatology-anchored logistic model.")

    fixed_offset = np.log(base / (1.0 - base))
    coef = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(int(steps)):
        logits = np.clip(fixed_offset + x @ coef, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        gradient = (x.T @ (p - y)) / x.shape[0] + float(l2) * coef
        weight = np.maximum(p * (1.0 - p), 1e-8)
        hessian = (x.T @ (x * weight[:, None])) / x.shape[0]
        hessian += (float(l2) + 1e-8) * np.eye(x.shape[1], dtype=np.float64)
        try:
            newton_step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            newton_step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        current_objective = float(
            np.mean(np.logaddexp(0.0, logits) - y * logits)
            + 0.5 * float(l2) * np.dot(coef, coef)
        )
        step_scale = min(1.0, max(float(lr), 1e-3) * 10.0)
        accepted = False
        while step_scale >= 1e-4:
            candidate = coef - step_scale * newton_step
            candidate_logits = np.clip(fixed_offset + x @ candidate, -30.0, 30.0)
            candidate_objective = float(
                np.mean(np.logaddexp(0.0, candidate_logits) - y * candidate_logits)
                + 0.5 * float(l2) * np.dot(candidate, candidate)
            )
            if candidate_objective <= current_objective:
                coef = candidate
                accepted = True
                break
            step_scale *= 0.5
        if not accepted or np.linalg.norm(step_scale * newton_step) < 1e-7:
            break

    return ClimatologyAnchoredLogisticCalibrator(
        feature_names=tuple(feature_names),
        coef=coef.astype(np.float32),
        calibration_split=calibration_split,
        n_samples=int(y.size),
        event_rate=float(y.mean()),
        mean_base_rate=float(base.mean()),
    )


def fit_nested_year_shrinkage_calibrator(
    features: np.ndarray,
    labels: np.ndarray,
    base_rates: np.ndarray,
    pair_years: np.ndarray,
    feature_names: Sequence[str],
    calibration_split: str,
    alpha_grid: Sequence[float],
    l2_grid: Sequence[float],
    steps: int,
    lr: float,
) -> Tuple[NestedYearShrinkageCalibrator, List[Dict[str, object]]]:
    """Select shrinkage and regularization by leave-one-calibration-year-out Brier."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32).reshape(-1)
    base = np.asarray(base_rates, dtype=np.float32).reshape(-1)
    pair_years = np.asarray(pair_years, dtype=np.int16).reshape(-1)
    valid = (
        np.all(np.isfinite(x), axis=1)
        & np.isfinite(y)
        & np.isfinite(base)
        & np.isfinite(pair_years)
    )
    x = x[valid]
    y = y[valid]
    base = np.clip(base[valid], 1e-5, 1.0 - 1e-5)
    pair_years = pair_years[valid]
    unique_years = np.array(sorted(set(int(v) for v in pair_years)), dtype=np.int16)
    if unique_years.size < 3:
        raise RuntimeError("Nested year-wise shrinkage requires at least three calibration years.")
    alphas = tuple(sorted(set(float(v) for v in alpha_grid)))
    l2_values = tuple(sorted(set(float(v) for v in l2_grid)))
    if not alphas or any(v < 0.0 or v > 1.0 for v in alphas):
        raise ValueError("Nested shrinkage alpha grid must contain values in [0, 1].")
    if 0.0 not in alphas:
        raise ValueError("Nested shrinkage alpha grid must include 0 so climatology is an available fallback.")
    if not l2_values or any(v < 0.0 for v in l2_values):
        raise ValueError("Nested shrinkage L2 grid must contain non-negative values.")

    fixed_offset = np.log(base / (1.0 - base))
    selection_rows: List[Dict[str, object]] = []
    for l2 in l2_values:
        oof_signal = np.full(y.shape, np.nan, dtype=np.float32)
        for hold_year in unique_years:
            hold = pair_years == int(hold_year)
            fit = ~hold
            if int(np.sum(fit)) < 100 or np.unique(y[fit]).size < 2:
                raise RuntimeError(f"Nested shrinkage inner fit is too thin after holding out year {int(hold_year)}.")
            inner = fit_climatology_anchored_logistic_calibrator(
                x[fit],
                y[fit],
                base[fit],
                feature_names,
                calibration_split=f"{calibration_split}_leave_{int(hold_year)}_out",
                steps=steps,
                lr=lr,
                l2=l2,
            )
            oof_signal[hold] = x[hold] @ inner.coef
        if not np.all(np.isfinite(oof_signal)):
            raise RuntimeError("Nested shrinkage failed to produce complete out-of-year calibration signals.")
        for alpha in alphas:
            logits = np.clip(fixed_offset + float(alpha) * oof_signal, -30.0, 30.0)
            prob = 1.0 / (1.0 + np.exp(-logits))
            slope, ece, brier = reliability_from_arrays(prob, y)
            selection_rows.append({
                "alpha": float(alpha),
                "l2": float(l2),
                "leave_one_year_out_brier": brier,
                "leave_one_year_out_ece": ece,
                "leave_one_year_out_slope": slope,
                "calibration_years": " ".join(str(int(v)) for v in unique_years),
                "n_samples": int(y.size),
            })

    selected = min(
        selection_rows,
        key=lambda row: (
            float(row["leave_one_year_out_brier"]),
            float(row["alpha"]),
            -float(row["l2"]),
        ),
    )
    selected_alpha = float(selected["alpha"])
    selected_l2 = float(selected["l2"])
    full = fit_climatology_anchored_logistic_calibrator(
        x,
        y,
        base,
        feature_names,
        calibration_split=calibration_split,
        steps=steps,
        lr=lr,
        l2=selected_l2,
    )
    model = NestedYearShrinkageCalibrator(
        feature_names=tuple(feature_names),
        coef=full.coef.copy(),
        alpha=selected_alpha,
        selected_l2=selected_l2,
        calibration_split=calibration_split,
        n_samples=int(y.size),
        event_rate=float(y.mean()),
        mean_base_rate=float(base.mean()),
    )
    for row in selection_rows:
        row["selected"] = (
            float(row["alpha"]) == selected_alpha and float(row["l2"]) == selected_l2
        )
    return model, selection_rows


def resolve_calibration_split(requested: str, eval_split: str) -> str:
    if requested != "auto":
        return requested
    return "val" if eval_split != "val" else "test"


def predict_target_field(
    model,
    batch,
    device,
    target_mode: str,
    lead_indices: Sequence[int],
    center_idx: int,
    image_size: Tuple[int, int],
    return_sigma: bool = False,
):
    y, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, t_idx, batch_mask = batch
    h, w = image_size
    pred = cfm.predict_direct(
        model,
        x_t.to(device),
        x_tm1.to(device),
        x_tm2.to(device),
        spatial_c.to(device),
        vec_c.to(device),
        global_fields.to(device),
        batch_mask.to(device),
        device,
    )
    raw_model = model.module if hasattr(model, "module") else model
    sigma_pred = getattr(raw_model, "last_sigma", None)
    sigma_z = None
    if target_mode == "window":
        if not cfm.Config.MULTI_LEAD_TUBE:
            raise RuntimeError("--target_mode window requires a multi-lead tube checkpoint.")
        if not lead_indices:
            raise RuntimeError("--target_mode window requires at least one lead index.")
        idx = torch.as_tensor(tuple(int(i) for i in lead_indices), device=pred.device, dtype=torch.long)
        mu_z = pred[0].index_select(0, idx)[:, :h, :w].mean(dim=0).detach().cpu().numpy().astype(np.float32)
        truth_z = y[0].index_select(0, idx.cpu())[:, :h, :w].mean(dim=0).numpy().astype(np.float32)
        if sigma_pred is not None:
            sig = sigma_pred[0].index_select(0, idx)[:, :h, :w].float()
            sigma_z = (torch.sqrt(sig.square().sum(dim=0).clamp_min(1e-8)) / max(len(lead_indices), 1)).detach().cpu().numpy().astype(np.float32)
    else:
        if cfm.Config.MULTI_LEAD_TUBE:
            mu_z = pred[0, center_idx, :h, :w].detach().cpu().numpy().astype(np.float32)
            truth_z = y[0, center_idx, :h, :w].numpy().astype(np.float32)
            if sigma_pred is not None:
                sigma_z = sigma_pred[0, center_idx, :h, :w].detach().cpu().numpy().astype(np.float32)
        else:
            mu_z = pred[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
            truth_z = y[0, 0, :h, :w].numpy().astype(np.float32)
            if sigma_pred is not None:
                sigma_z = sigma_pred[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
    if return_sigma:
        if sigma_z is None:
            raise RuntimeError("--use_model_sigma requested, but checkpoint/model did not emit last_sigma.")
        return mu_z, truth_z, int(t_idx.item()), sigma_z
    return mu_z, truth_z, int(t_idx.item())


INCREMENTAL_SKILL_SPECS: Dict[str, Tuple[str, ...]] = {
    "incremental_A_init_margin": ("init_margin",),
    "incremental_B_forecast_margin": ("forecast_margin",),
    "incremental_C_init_plus_forecast": ("init_margin", "forecast_margin"),
    "incremental_D_init_forecast_sigma": ("init_margin", "forecast_margin", "predicted_sigma"),
}
CLIMATOLOGY_ANCHORED_MODEL_NAME = "incremental_E_climo_offset_init_forecast"
NESTED_SHRINKAGE_MODEL_NAME = "incremental_F_nested_year_shrinkage"


def collect_incremental_skill_pairs(
    model,
    dataset,
    split_name: str,
    split_years: Iterable[int],
    q95_z: np.ndarray,
    base_rate: np.ndarray,
    months: np.ndarray,
    years: np.ndarray,
    mask_np: np.ndarray,
    device,
    target_mode: str,
    lead_indices: Sequence[int],
    center_idx: int,
    target_center_lead: int,
    max_cases: int,
    max_samples: int,
    seed: int,
    progress_every: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect held-out init/forecast/sigma features without touching eval years."""
    rng = np.random.default_rng(seed)
    split_year_set = set(int(y) for y in split_years)
    n_cases = min(len(dataset), max(1, int(max_cases)))
    subset_idx = np.unique(np.linspace(0, len(dataset) - 1, n_cases, dtype=np.int64))
    loader = DataLoader(Subset(dataset, subset_idx.tolist()), batch_size=1, shuffle=False, num_workers=0)
    per_case = max(1, int(math.ceil(max_samples / max(len(subset_idx), 1))))
    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    pair_years: List[np.ndarray] = []
    pair_base_rates: List[np.ndarray] = []
    include_sigma = bool(cfm.Config.DISTRIBUTIONAL_HEAD)
    print(
        "Collecting incremental-skill pairs: "
        f"split={split_name}, cases={len(subset_idx)}, max_pixels={int(max_samples)}, "
        f"target_mode={target_mode}, predicted_sigma={include_sigma}"
    )
    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if include_sigma:
                mu_z, truth_z, t, sigma_model = predict_target_field(
                    model,
                    batch,
                    device,
                    target_mode=target_mode,
                    lead_indices=lead_indices,
                    center_idx=center_idx,
                    image_size=cfm.Config.IMAGE_SIZE,
                    return_sigma=True,
                )
            else:
                mu_z, truth_z, t = predict_target_field(
                    model,
                    batch,
                    device,
                    target_mode=target_mode,
                    lead_indices=lead_indices,
                    center_idx=center_idx,
                    image_size=cfm.Config.IMAGE_SIZE,
                )
                sigma_model = np.full_like(mu_z, np.nan, dtype=np.float32)
            target_t = int(t) + int(target_center_lead)
            month = int(months[target_t])
            target_year = int(years[target_t])
            if month not in MJJAS_MONTHS:
                continue
            if target_year not in split_year_set:
                raise RuntimeError(
                    f"Incremental-skill pair year leakage for split={split_name}: target_year={target_year}"
                )
            q = q95_z[month]
            climo_rate = base_rate[month]
            init_z = batch[1][0, 0, : q.shape[0], : q.shape[1]].numpy().astype(np.float32)
            valid = (
                mask_np
                & np.isfinite(init_z)
                & np.isfinite(mu_z)
                & np.isfinite(truth_z)
                & np.isfinite(q)
                & np.isfinite(climo_rate)
            )
            candidates = np.flatnonzero(valid.ravel())
            if candidates.size == 0:
                continue
            chosen = rng.choice(candidates, size=min(per_case, candidates.size), replace=False)
            features.append(np.column_stack([
                init_z.ravel()[chosen] - q.ravel()[chosen],
                mu_z.ravel()[chosen] - q.ravel()[chosen],
                sigma_model.ravel()[chosen],
            ]).astype(np.float32))
            labels.append((truth_z.ravel()[chosen] > q.ravel()[chosen]).astype(np.float32))
            pair_years.append(np.full(chosen.size, target_year, dtype=np.int16))
            pair_base_rates.append(climo_rate.ravel()[chosen].astype(np.float32))
            if (batch_idx + 1) % max(1, int(progress_every)) == 0:
                print(f"  incremental-skill pairs processed {batch_idx + 1}/{len(subset_idx)}")
    if not features:
        raise RuntimeError("No incremental-skill calibration pairs were collected.")
    x = np.concatenate(features, axis=0)[:max_samples]
    y = np.concatenate(labels, axis=0)[:max_samples]
    py = np.concatenate(pair_years, axis=0)[:max_samples]
    br = np.concatenate(pair_base_rates, axis=0)[:max_samples]
    if not set(int(v) for v in np.unique(py)).issubset(split_year_set):
        raise RuntimeError("Incremental-skill calibration pairs contain years outside the calibration split.")
    return x, y, py, br


def fit_incremental_skill_models(
    features: np.ndarray,
    labels: np.ndarray,
    base_rates: np.ndarray,
    pair_years: np.ndarray,
    calibration_split: str,
    alpha_grid: Sequence[float],
    l2_grid: Sequence[float],
    steps: int,
    lr: float,
    l2: float,
) -> Tuple[Dict[str, Any], List[Dict[str, object]]]:
    column_index = {"init_margin": 0, "forecast_margin": 1, "predicted_sigma": 2}
    models: Dict[str, Any] = {}
    for model_name, feature_names in INCREMENTAL_SKILL_SPECS.items():
        if "predicted_sigma" in feature_names and not np.any(np.isfinite(features[:, 2])):
            print(f"Skipping {model_name}: checkpoint does not provide predicted sigma.")
            continue
        idx = [column_index[name] for name in feature_names]
        models[model_name] = fit_model_output_logistic_calibrator(
            features[:, idx],
            labels,
            feature_names,
            calibration_split=calibration_split,
            steps=steps,
            lr=lr,
            l2=l2,
        )
        print(
            f"Fitted {model_name} on {calibration_split}: "
            f"features={feature_names}, n={models[model_name].n_samples}, "
            f"event_rate={models[model_name].event_rate:.4f}"
        )
    anchored_features = ("init_margin", "forecast_margin")
    anchored_idx = [column_index[name] for name in anchored_features]
    models[CLIMATOLOGY_ANCHORED_MODEL_NAME] = fit_climatology_anchored_logistic_calibrator(
        features[:, anchored_idx],
        labels,
        base_rates,
        anchored_features,
        calibration_split=calibration_split,
        steps=steps,
        lr=lr,
        l2=l2,
    )
    anchored = models[CLIMATOLOGY_ANCHORED_MODEL_NAME]
    finite_base_rates = np.asarray(base_rates, dtype=np.float32)
    finite_base_rates = finite_base_rates[np.isfinite(finite_base_rates)][:1024]
    zero_margin_prob = anchored.predict_features(
        np.zeros((finite_base_rates.size, len(anchored_features)), dtype=np.float32),
        finite_base_rates,
    )
    anchor_error = float(np.max(np.abs(zero_margin_prob - finite_base_rates)))
    if anchor_error > 1e-6:
        raise RuntimeError(
            "Climatology-anchored model invariant failed: zero-margin probability does not equal "
            f"the train-only base rate (max_abs_error={anchor_error:.3g})."
        )
    print(
        f"Fitted {CLIMATOLOGY_ANCHORED_MODEL_NAME} on {calibration_split}: "
        f"fixed_offset=train_month_pixel_climatology, learned_intercept=False, "
        f"features={anchored_features}, n={anchored.n_samples}, "
        f"event_rate={anchored.event_rate:.4f}, mean_base_rate={anchored.mean_base_rate:.4f}, "
        f"zero_margin_anchor_error={anchor_error:.3g}"
    )
    nested, nested_selection_rows = fit_nested_year_shrinkage_calibrator(
        features[:, anchored_idx],
        labels,
        base_rates,
        pair_years,
        anchored_features,
        calibration_split=calibration_split,
        alpha_grid=alpha_grid,
        l2_grid=l2_grid,
        steps=steps,
        lr=lr,
    )
    models[NESTED_SHRINKAGE_MODEL_NAME] = nested
    nested_zero_margin_prob = nested.predict_features(
        np.zeros((finite_base_rates.size, len(anchored_features)), dtype=np.float32),
        finite_base_rates,
    )
    nested_anchor_error = float(np.max(np.abs(nested_zero_margin_prob - finite_base_rates)))
    if nested_anchor_error > 1e-6:
        raise RuntimeError(
            "Nested shrinkage invariant failed: zero-margin probability does not equal "
            f"the train-only base rate (max_abs_error={nested_anchor_error:.3g})."
        )
    print(
        f"Fitted {NESTED_SHRINKAGE_MODEL_NAME} on {calibration_split}: "
        f"alpha={nested.alpha:.3f}, l2={nested.selected_l2:g}, "
        f"selection=leave-one-calibration-year-out Brier, learned_intercept=False, "
        f"n={nested.n_samples}, event_rate={nested.event_rate:.4f}, "
        f"zero_margin_anchor_error={nested_anchor_error:.3g}"
    )
    return models, nested_selection_rows


def predict_incremental_skill_grid(
    calibrator: Any,
    feature_fields: Mapping[str, np.ndarray],
    mask: np.ndarray,
    base_rate: Optional[np.ndarray] = None,
) -> np.ndarray:
    fields = [np.asarray(feature_fields[name], dtype=np.float32) for name in calibrator.feature_names]
    valid = mask.copy()
    for field in fields:
        valid &= np.isfinite(field)
    if isinstance(calibrator, (ClimatologyAnchoredLogisticCalibrator, NestedYearShrinkageCalibrator)):
        if base_rate is None:
            raise RuntimeError("Climatology-anchored incremental model requires a train-only base-rate field.")
        valid &= np.isfinite(base_rate)
    out = np.full(mask.shape, np.nan, dtype=np.float32)
    if np.any(valid):
        x = np.column_stack([field[valid] for field in fields]).astype(np.float32)
        if isinstance(calibrator, (ClimatologyAnchoredLogisticCalibrator, NestedYearShrinkageCalibrator)):
            out[valid] = calibrator.predict_features(x, base_rate[valid])
        else:
            out[valid] = calibrator.predict_features(x)
    return out


def save_incremental_calibration_arrays(
    out_dir: Path,
    features: np.ndarray,
    labels: np.ndarray,
    pair_years: np.ndarray,
    base_rates: np.ndarray,
    train_years: Iterable[int],
    calibration_years: Iterable[int],
    test_years: Iterable[int],
    calibration_split: str,
) -> Path:
    array_dir = ensure_dir(out_dir / "incremental_arrays")
    fold = int(getattr(cfm.Config, "CV_FOLD", cfm.Config.CV_TEST_OFFSETS[0]))
    path = array_dir / "calibration_pairs.npz"
    np.savez_compressed(
        path,
        init_margin=np.asarray(features[:, 0], dtype=np.float32),
        forecast_margin=np.asarray(features[:, 1], dtype=np.float32),
        model_sigma=np.asarray(features[:, 2], dtype=np.float32),
        truth=np.asarray(labels, dtype=np.uint8),
        base_rate=np.asarray(base_rates, dtype=np.float32),
        year=np.asarray(pair_years, dtype=np.int16),
        source_fold=np.array(fold, dtype=np.int16),
        train_years=np.array(sorted(int(v) for v in train_years), dtype=np.int16),
        calibration_years=np.array(sorted(int(v) for v in calibration_years), dtype=np.int16),
        test_years=np.array(sorted(int(v) for v in test_years), dtype=np.int16),
        calibration_split=np.array(str(calibration_split)),
    )
    return path


def prepare_incremental_test_chunk_dir(out_dir: Path) -> Path:
    chunk_dir = ensure_dir(out_dir / "incremental_arrays" / "test_chunks")
    for path in chunk_dir.glob("sample_*.npz"):
        path.unlink()
    return chunk_dir


def save_incremental_test_chunk(
    chunk_dir: Path,
    sample_index: int,
    source_fold: int,
    target_year: int,
    target_month: int,
    init_margin: np.ndarray,
    forecast_margin: np.ndarray,
    model_sigma: Optional[np.ndarray],
    truth: np.ndarray,
    base_rate: np.ndarray,
    mask: np.ndarray,
    mu_z: Optional[np.ndarray] = None,
    truth_z: Optional[np.ndarray] = None,
    init_time_index: Optional[int] = None,
    target_center_time_index: Optional[int] = None,
) -> int:
    valid = (
        mask
        & np.isfinite(init_margin)
        & np.isfinite(forecast_margin)
        & np.isfinite(truth)
        & np.isfinite(base_rate)
    )
    if not np.any(valid):
        return 0
    sigma = (
        np.asarray(model_sigma, dtype=np.float32)[valid]
        if model_sigma is not None
        else np.full(int(np.sum(valid)), np.nan, dtype=np.float32)
    )
    mu_z_saved = (
        np.asarray(mu_z, dtype=np.float32)[valid]
        if mu_z is not None
        else np.full(int(np.sum(valid)), np.nan, dtype=np.float32)
    )
    truth_z_saved = (
        np.asarray(truth_z, dtype=np.float32)[valid]
        if truth_z is not None
        else np.full(int(np.sum(valid)), np.nan, dtype=np.float32)
    )
    path = chunk_dir / f"sample_{int(sample_index):05d}.npz"
    np.savez_compressed(
        path,
        init_margin=np.asarray(init_margin[valid], dtype=np.float32),
        forecast_margin=np.asarray(forecast_margin[valid], dtype=np.float32),
        model_sigma=sigma,
        truth=np.asarray(truth[valid] > 0.5, dtype=np.uint8),
        mu_z=mu_z_saved,
        truth_z=truth_z_saved,
        base_rate=np.asarray(base_rate[valid], dtype=np.float32),
        year=np.array(int(target_year), dtype=np.int16),
        month=np.array(int(target_month), dtype=np.int8),
        source_fold=np.array(int(source_fold), dtype=np.int16),
        init_time_index=np.array(-1 if init_time_index is None else int(init_time_index), dtype=np.int32),
        target_center_time_index=np.array(
            -1 if target_center_time_index is None else int(target_center_time_index),
            dtype=np.int32,
        ),
    )
    return int(np.sum(valid))


def save_incremental_array_manifest(
    out_dir: Path,
    run_name: str,
    target_mode: str,
    window_leads: Sequence[int],
    train_years: Iterable[int],
    calibration_years: Iterable[int],
    test_years: Iterable[int],
    calibration_split: str,
    eval_split: str,
    sample_count: int,
    valid_cell_count: int,
) -> Path:
    array_dir = ensure_dir(out_dir / "incremental_arrays")
    fold = int(getattr(cfm.Config, "CV_FOLD", cfm.Config.CV_TEST_OFFSETS[0]))
    path = array_dir / "manifest.npz"
    np.savez_compressed(
        path,
        run_name=np.array(str(run_name)),
        source_fold=np.array(fold, dtype=np.int16),
        target_mode=np.array(str(target_mode)),
        window_leads=np.array(tuple(int(v) for v in window_leads), dtype=np.int16),
        train_years=np.array(sorted(int(v) for v in train_years), dtype=np.int16),
        calibration_years=np.array(sorted(int(v) for v in calibration_years), dtype=np.int16),
        test_years=np.array(sorted(int(v) for v in test_years), dtype=np.int16),
        calibration_split=np.array(str(calibration_split)),
        eval_split=np.array(str(eval_split)),
        sample_count=np.array(int(sample_count), dtype=np.int32),
        valid_cell_count=np.array(int(valid_cell_count), dtype=np.int64),
        schema_version=np.array(2, dtype=np.int16),
        schema_note=np.array("test_chunks include binary truth plus continuous mu_z/truth_z fields when exported by exceedance_eval.py schema>=2"),
    )
    return path


def monotonic_calibrator_cache_path(
    target_mode: str,
    window_leads: Sequence[int],
    calibration_split: str,
    requested_calibrator: str,
    checkpoint_path: str,
    use_model_sigma: bool = False,
) -> str:
    suffix = (
        f"win{lead_list_label(window_leads)}"
        if target_mode == "window"
        else f"daily{int(cfm.Config.LEAD_TIME)}"
    )
    pred = "tube" + lead_list_label(cfm.prediction_leads(cfm.Config)) if cfm.Config.MULTI_LEAD_TUBE else "single"
    fold = int(getattr(cfm.Config, "CV_FOLD", cfm.Config.CV_TEST_OFFSETS[0]))
    ckpt_tag = hashlib.sha1(os.path.abspath(checkpoint_path).encode("utf-8")).hexdigest()[:10]
    sigma_tag = "modelsigma" if use_model_sigma else "cachedsigma"
    return os.path.join(
        cfm.Config.OUTPUT_DIR,
        "data_cache",
        (
            f"monotonic_calibrator_v2_{target_mode}_{suffix}_{sigma_tag}_{pred}_"
            f"{cfm.cv_split_tag(cfm.Config)}_fold{fold}_cal{calibration_split}_"
            f"{requested_calibrator}_ckpt{ckpt_tag}.pkl"
        ),
    )


def collect_margin_label_pairs(
    model,
    dataset,
    split_name: str,
    split_years: Iterable[int],
    q95_z: np.ndarray,
    sigma: np.ndarray,
    months: np.ndarray,
    years: np.ndarray,
    mask_np: np.ndarray,
    device,
    target_mode: str,
    lead_indices: Sequence[int],
    center_idx: int,
    target_center_lead: int,
    max_cases: int,
    max_samples: int,
    seed: int,
    progress_every: int,
    use_model_sigma: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    split_year_set = set(int(y) for y in split_years)
    n_cases = min(len(dataset), max(1, int(max_cases)))
    subset_idx = np.unique(np.linspace(0, len(dataset) - 1, n_cases, dtype=np.int64))
    loader = DataLoader(Subset(dataset, subset_idx.tolist()), batch_size=1, shuffle=False, num_workers=0)
    per_case = max(1, int(math.ceil(max_samples / max(len(subset_idx), 1))))
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ms: List[np.ndarray] = []
    target_year_values: List[np.ndarray] = []
    print(
        "Collecting calibration pairs: "
        f"split={split_name}, cases={len(subset_idx)}, max_pixels={int(max_samples)}, "
        f"target_mode={target_mode}"
    )
    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if use_model_sigma:
                mu_z, truth_z, t, sigma_model = predict_target_field(
                    model,
                    batch,
                    device,
                    target_mode=target_mode,
                    lead_indices=lead_indices,
                    center_idx=center_idx,
                    image_size=cfm.Config.IMAGE_SIZE,
                    return_sigma=True,
                )
            else:
                mu_z, truth_z, t = predict_target_field(
                    model,
                    batch,
                    device,
                    target_mode=target_mode,
                    lead_indices=lead_indices,
                    center_idx=center_idx,
                    image_size=cfm.Config.IMAGE_SIZE,
                )
                sigma_model = None
            target_t = int(t) + int(target_center_lead)
            month = int(months[target_t])
            target_year = int(years[target_t])
            if month not in MJJAS_MONTHS:
                continue
            if target_year not in split_year_set:
                raise RuntimeError(
                    f"Calibration/eval pair year leakage for split={split_name}: target_year={target_year}"
                )
            q = q95_z[month]
            sig = sigma_model if use_model_sigma else sigma[month]
            valid = mask_np & np.isfinite(mu_z) & np.isfinite(truth_z) & np.isfinite(q) & np.isfinite(sig)
            candidates = np.flatnonzero(valid.ravel())
            if candidates.size == 0:
                continue
            chosen = rng.choice(candidates, size=min(per_case, candidates.size), replace=False)
            score = ((mu_z.ravel()[chosen] - q.ravel()[chosen]) / np.maximum(sig.ravel()[chosen], 0.1)).astype(np.float32)
            label = (truth_z.ravel()[chosen] > q.ravel()[chosen]).astype(np.float32)
            xs.append(score)
            ys.append(label)
            ms.append(np.full(score.shape, month, dtype=np.int16))
            target_year_values.append(np.full(score.shape, target_year, dtype=np.int16))
            if sum(len(a) for a in xs) >= max_samples:
                break
            if (batch_idx + 1) % max(1, int(progress_every)) == 0:
                print(f"  calibration pairs processed {batch_idx + 1}/{len(subset_idx)}")
    if not xs:
        raise RuntimeError(f"No calibration pairs collected for split={split_name}.")
    return (
        np.concatenate(xs)[:max_samples].astype(np.float32),
        np.concatenate(ys)[:max_samples].astype(np.float32),
        np.concatenate(ms)[:max_samples].astype(np.int16),
        np.concatenate(target_year_values)[:max_samples].astype(np.int16),
    )


def load_or_fit_monotonic_calibrator(
    model,
    calibration_dataset,
    calibration_split: str,
    calibration_years: Iterable[int],
    eval_years: Iterable[int],
    q95_z: np.ndarray,
    sigma: np.ndarray,
    months: np.ndarray,
    years: np.ndarray,
    mask_np: np.ndarray,
    device,
    target_mode: str,
    window_leads: Sequence[int],
    lead_indices: Sequence[int],
    center_idx: int,
    target_center_lead: int,
    checkpoint_path: str,
    requested_calibrator: str,
    max_cases: int,
    max_samples: int,
    min_month_samples: int,
    min_month_positives: int,
    seed: int,
    progress_every: int,
    use_model_sigma: bool = False,
) -> Tuple[MarginMonotonicCalibrator, Dict[str, MarginMonotonicCalibrator], List[Dict[str, object]], str]:
    path = monotonic_calibrator_cache_path(
        target_mode, window_leads, calibration_split, requested_calibrator, checkpoint_path,
        use_model_sigma=use_model_sigma,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    calibration_year_set = set(int(y) for y in calibration_years)
    eval_year_set = set(int(y) for y in eval_years)
    if calibration_year_set & eval_year_set:
        raise RuntimeError("Calibration/eval year overlap before fitting calibrator.")
    src_mtime = _source_mtime([checkpoint_path, cfm.get_norm_stats_path(cfm.Config), cfm.Config.TRAINING_DATA_PATH])
    if os.path.exists(path) and os.path.getmtime(path) >= src_mtime:
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            cached_years = set(int(y) for y in payload.get("calibration_years", []))
            cached_eval_years = set(int(y) for y in payload.get("eval_years", []))
            if (
                cached_years == calibration_year_set
                and cached_eval_years == eval_year_set
                and tuple(payload.get("window_leads", ())) == tuple(int(x) for x in window_leads)
                and payload.get("target_mode") == target_mode
                and payload.get("requested_calibrator") == requested_calibrator
                and payload.get("selector_version") == 2
                and bool(payload.get("use_model_sigma", False)) == bool(use_model_sigma)
            ):
                print(f"Loading held-out monotonic calibrator from {path}")
                return (
                    payload["chosen"],
                    payload["candidates"],
                    payload.get("inner_selection_rows", []),
                    payload.get("selection_reason", "cached selection"),
                )
        except Exception:
            pass

    x, y, month_arr, pair_years = collect_margin_label_pairs(
        model,
        calibration_dataset,
        calibration_split,
        calibration_years,
        q95_z,
        sigma,
        months,
        years,
        mask_np,
        device,
        target_mode,
        lead_indices,
        center_idx,
        target_center_lead,
        max_cases,
        max_samples,
        seed,
        progress_every,
        use_model_sigma=use_model_sigma,
    )
    if set(int(y0) for y0 in np.unique(pair_years)) & eval_year_set:
        raise RuntimeError("Eval year appeared in calibration pairs.")

    chosen, candidates, inner_selection_rows, selection_reason = fit_inner_selected_monotonic_calibrator(
        x,
        y,
        month_arr,
        pair_years,
        calibration_split,
        requested_calibrator,
        min_month_samples=int(min_month_samples),
        min_month_positives=int(min_month_positives),
    )
    print("Held-out monotonic calibration candidates")
    print("----------------------------------------")
    for name, cal in candidates.items():
        print(
            f"{name:8s} full-cal ECE={cal.calibration_ece:.4f}, "
            f"slope={cal.calibration_slope:.3f}, Brier={cal.calibration_brier:.5f}, "
            f"month_models={len(cal.by_month)}"
        )
    if inner_selection_rows:
        print("Inner-selection calibration scores")
        for row in inner_selection_rows:
            print(
                f"{row['candidate']:8s} inner ECE={float(row['inner_selection_ece']):.4f}, "
                f"slope={float(row['inner_selection_slope']):.3f}, "
                f"Brier={float(row['inner_selection_brier']):.5f}"
            )
    print(f"Chosen calibrator: {chosen.method} ({selection_reason})")

    with open(path, "wb") as f:
        pickle.dump({
            "chosen": chosen,
            "candidates": candidates,
            "calibration_years": sorted(calibration_year_set),
            "eval_years": sorted(eval_year_set),
            "window_leads": tuple(int(x) for x in window_leads),
            "target_mode": target_mode,
            "requested_calibrator": requested_calibrator,
            "checkpoint_path": os.path.abspath(checkpoint_path),
            "selector_version": 2,
            "use_model_sigma": bool(use_model_sigma),
            "inner_selection_rows": inner_selection_rows,
            "selection_reason": selection_reason,
        }, f)
    print(f"  Saved held-out monotonic calibrator to {path}")
    return chosen, candidates, inner_selection_rows, selection_reason


def train_model_output_calibrator(
    model,
    calibration_dataset,
    calibration_split: str,
    q95_z: np.ndarray,
    sigma_clim: np.ndarray,
    months: np.ndarray,
    region_features: Mapping[str, np.ndarray],
    mask_np: np.ndarray,
    device,
    center_idx: int,
    target_mode: str,
    lead_indices: Sequence[int],
    target_center_lead: int,
    max_cases: int,
    max_samples: int,
    seed: int,
    steps: int,
    lr: float,
    l2: float,
) -> ModelOutputLogisticCalibrator:
    rng = np.random.default_rng(seed)
    n_cases = min(len(calibration_dataset), max(1, int(max_cases)))
    subset_idx = np.unique(np.linspace(0, len(calibration_dataset) - 1, n_cases, dtype=np.int64))
    loader = DataLoader(Subset(calibration_dataset, subset_idx.tolist()), batch_size=1, shuffle=False, num_workers=0)
    per_case = max(1, int(math.ceil(max_samples / max(len(subset_idx), 1))))
    feature_names = calibration_feature_names(tuple(region_features.keys()))
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    print(
        "Training calibrated model-output logistic layer: "
        f"split={calibration_split}, cases={len(subset_idx)}, max_pixels={int(max_samples)}"
    )
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            mu_z, truth_z, t = predict_target_field(
                model,
                batch,
                device,
                target_mode=target_mode,
                lead_indices=lead_indices,
                center_idx=center_idx,
                image_size=cfm.Config.IMAGE_SIZE,
            )
            target_t = t + int(target_center_lead)
            month = int(months[target_t])
            if month not in MJJAS_MONTHS:
                continue
            q = q95_z[month]
            sigma = sigma_clim[month]
            valid = mask_np & np.isfinite(mu_z) & np.isfinite(truth_z) & np.isfinite(q) & np.isfinite(sigma)
            candidates = np.flatnonzero(valid.ravel())
            if candidates.size == 0:
                continue
            chosen = rng.choice(candidates, size=min(per_case, candidates.size), replace=False)
            xs.append(model_output_calibration_features(mu_z, q, sigma, month, region_features, chosen))
            truth = (truth_z.ravel()[chosen] > q.ravel()[chosen]).astype(np.float32)
            ys.append(truth)
            if sum(x.shape[0] for x in xs) >= max_samples:
                break
            if (batch_idx + 1) % 50 == 0:
                print(f"  calibration cases processed {batch_idx + 1}/{len(subset_idx)}")

    if not xs:
        raise RuntimeError("No samples collected for calibrated model-output logistic layer.")
    features = np.concatenate(xs, axis=0)[:max_samples]
    labels = np.concatenate(ys, axis=0)[:max_samples]
    calibrator = fit_model_output_logistic_calibrator(
        features,
        labels,
        feature_names,
        calibration_split=calibration_split,
        steps=int(steps),
        lr=float(lr),
        l2=float(l2),
    )
    print(
        "Calibrated model-output logistic fitted: "
        f"n={calibrator.n_samples}, event_rate={calibrator.event_rate:.4f}, "
        f"intercept={calibrator.intercept:.4f}"
    )
    return calibrator


def train_pooled_logistic_baseline(
    shared_data,
    train_indices: Sequence[int],
    q95_z: np.ndarray,
    norm_stats: Mapping[str, torch.Tensor],
    months: np.ndarray,
    target_mode: str,
    window_leads: Sequence[int],
    target_center_lead: int,
    max_samples: int,
    seed: int,
) -> PooledLogistic:
    rng = np.random.default_rng(seed)
    heat = shared_data["heat_index"]
    land_mask = np.isfinite(np.asarray(heat[:, :, 0])) & (np.asarray(heat[:, :, 0]) != 0.0)
    land_flat = np.flatnonzero(land_mask.ravel())
    per_time = max(1, int(math.ceil(max_samples / max(len(train_indices), 1))))

    xs = []
    ys = []
    for t in train_indices:
        target_t = int(t) + int(target_center_lead)
        month = int(months[target_t])
        if month not in MJJAS_MONTHS:
            continue
        chosen = rng.choice(land_flat, size=min(per_time, land_flat.size), replace=False)
        x_t_z = _normalize_field(np.asarray(heat[:, :, int(t)]), norm_stats).ravel()
        if target_mode == "window":
            truth_field = window_mean_z_from_shared(heat, int(t), window_leads, norm_stats)
            y_z = truth_field.ravel()
        else:
            y_z = _normalize_field(np.asarray(heat[:, :, target_t]), norm_stats).ravel()
        q = q95_z[month].ravel()
        valid = np.isfinite(x_t_z[chosen]) & np.isfinite(y_z[chosen]) & np.isfinite(q[chosen])
        if np.any(valid):
            idx = chosen[valid]
            xs.append((x_t_z[idx] - q[idx]).astype(np.float32))
            ys.append((y_z[idx] > q[idx]).astype(np.float32))
        if sum(len(a) for a in xs) >= max_samples:
            break

    if not xs:
        raise RuntimeError("Could not sample training data for pooled logistic baseline.")
    x = np.concatenate(xs)[:max_samples].astype(np.float64)
    y = np.concatenate(ys)[:max_samples].astype(np.float64)
    mean = float(np.mean(x))
    std = float(np.std(x) + 1e-8)
    z = (x - mean) / std

    coef = 0.0
    intercept = math.log((y.mean() + 1e-4) / (1.0 - y.mean() + 1e-4))
    lr = 0.2
    for _ in range(300):
        logits = np.clip(intercept + coef * z, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        err = p - y
        intercept -= lr * float(err.mean())
        coef -= lr * float(np.mean(err * z))
    print(
        "Pooled logistic baseline trained on train years only: "
        f"n={len(y)}, event_rate={y.mean():.4f}, coef={coef:.4f}, intercept={intercept:.4f}"
    )
    return PooledLogistic(mean=mean, std=std, coef=float(coef), intercept=float(intercept))


def predict_center_field(model, batch, device, center_idx: int, image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, int]:
    y, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, t_idx, batch_mask = batch
    h, w = image_size
    pred = cfm.predict_direct(
        model,
        x_t.to(device),
        x_tm1.to(device),
        x_tm2.to(device),
        spatial_c.to(device),
        vec_c.to(device),
        global_fields.to(device),
        batch_mask.to(device),
        device,
    )
    if cfm.Config.MULTI_LEAD_TUBE:
        mu_z = pred[0, center_idx, :h, :w].detach().cpu().numpy().astype(np.float32)
        truth_z = y[0, center_idx, :h, :w].numpy().astype(np.float32)
    else:
        mu_z = pred[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
        truth_z = y[0, 0, :h, :w].numpy().astype(np.float32)
    return mu_z, truth_z, int(t_idx.item())


def select_dataset(split: str, config, shared_data, norm_stats, climo, train_indices, val_indices, test_indices):
    if split == "train":
        return cfm.ClimateDataset(config, mode="train", train_indices=train_indices, normalization_stats=norm_stats, shared_data=shared_data, target_climatology=climo)
    if split == "val":
        return cfm.ClimateDataset(config, mode="val", val_indices=val_indices, normalization_stats=norm_stats, shared_data=shared_data, target_climatology=climo)
    if split == "test":
        return cfm.ClimateDataset(config, mode="test", test_indices=test_indices, normalization_stats=norm_stats, shared_data=shared_data, target_climatology=climo)
    raise ValueError(f"Unsupported split: {split}")


def evaluate(args: argparse.Namespace) -> None:
    configure_from_args(args)
    target_mode = str(args.target_mode).lower()
    eval_split = args.eval_split if args.eval_split is not None else (args.split if args.split is not None else "test")
    if eval_split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported eval split: {eval_split}")
    checkpoint_request = args.checkpoint or ("best_tac" if target_mode == "window" else "best_monitor")
    ckpt_path = checkpoint_path_for(args.run_name, checkpoint_request)
    configure_structure_from_checkpoint(ckpt_path, args.prediction_leads)
    predicted_leads = cfm.prediction_leads(cfm.Config)
    window_leads = parse_int_list(args.window_leads) if args.window_leads else predicted_leads
    lead_indices: Tuple[int, ...] = ()
    if target_mode == "window":
        if not cfm.Config.MULTI_LEAD_TUBE:
            raise RuntimeError("--target_mode window requires --multi_lead_tube or a tube run_name.")
        lead_indices = window_lead_indices(predicted_leads, window_leads)
        target_center_lead = window_center_lead(window_leads)
        out_dir = ensure_dir(Path(args.output_dir) / args.run_name / eval_split / f"window_{lead_list_label(window_leads)}")
        print(
            "Windowed exceedance mode: "
            f"window_leads={window_leads}, predicted_leads={predicted_leads}, "
            f"center_month_lead={target_center_lead}"
        )
        if args.use_model_sigma:
            print("Exceedance probability sigma source: model-predicted distributional sigma")
    else:
        target_center_lead = int(cfm.Config.LEAD_TIME)
        out_dir = ensure_dir(Path(args.output_dir) / args.run_name / eval_split)
        if args.use_model_sigma:
            print("Exceedance probability sigma source: model-predicted distributional sigma")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(int(args.local_rank))

    shared_data = cfm.prepare_shared_data(cfm.Config, rank=0, world_size=1, ddp=False)
    time_values = np.asarray(shared_data["time_values"])
    months, years, doys = month_doy_year_arrays(time_values)
    runs = cfm.detect_continuous_runs(time_values)
    train_indices, val_indices, test_indices, train_years, val_years, test_years = split_indices_for_config(shared_data)
    print(f"CV split: {cfm.cv_split_tag(cfm.Config)}")
    print(f"Train years ({len(train_years)}): {sorted(train_years)}")
    print(f"Val years: {sorted(val_years)}")
    print(f"Test years: {sorted(test_years)}")

    split_years = {"train": train_years, "val": val_years, "test": test_years}[eval_split]
    if set(split_years) & set(train_years) and eval_split != "train":
        raise RuntimeError("Leakage check failed: evaluation split overlaps training years.")

    norm_stats = load_norm_stats()
    climo = cfm.load_or_build_train_climatology(shared_data, train_indices, norm_stats, cfm.Config, ddp=False)
    daily_q95_z, daily_sigma_clim, daily_base_rate = load_or_build_exceedance_stats(
        shared_data, train_years, norm_stats, climo
    )
    for month in MJJAS_MONTHS:
        rate = float(np.nanmean(daily_base_rate[month]))
        print(f"Daily build check month={month}: train-year mean exceedance rate={rate:.4f}")
        if np.isfinite(rate) and not (0.025 <= rate <= 0.075):
            raise RuntimeError(f"Train exceedance rate for month {month} is not near 5%: {rate:.4f}")

    dataset = select_dataset(
        eval_split, cfm.Config, shared_data, norm_stats, climo,
        train_indices, val_indices, test_indices,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    conus_mask = cfm.load_conus_mask(cfm.Config)
    mask_np = conus_mask.numpy() > 0.5
    mesh = cfm.build_mesh_once(cfm.Config, conus_mask, device, ddp=False)
    print(f"Loading checkpoint: {ckpt_path}")
    model = cfm._load_meshflownet_checkpoint(ckpt_path, mesh, device)

    if target_mode == "window":
        train_dataset_for_stats = select_dataset(
            "train", cfm.Config, shared_data, norm_stats, climo,
            train_indices, val_indices, test_indices,
        )
        q95_z, sigma_clim, base_rate = load_or_build_window_exceedance_stats(
            model,
            train_dataset_for_stats,
            shared_data,
            train_years,
            norm_stats,
            climo,
            months,
            years,
            window_leads,
            lead_indices,
            mask_np,
            device,
            ckpt_path,
            progress_every=int(args.progress_every),
        )
        for month in MJJAS_MONTHS:
            rate = float(np.nanmean(base_rate[month]))
            if np.isfinite(rate):
                print(f"Windowed build check month={month}: train-year mean exceedance rate={rate:.4f}")
                if not (0.025 <= rate <= 0.075):
                    raise RuntimeError(
                        f"Windowed train exceedance rate for month {month} is not near 5%: {rate:.4f}"
                    )
            else:
                print(f"Windowed build check month={month}: no center-month samples")
    else:
        q95_z, sigma_clim, base_rate = daily_q95_z, daily_sigma_clim, daily_base_rate

    regions = {
        name: (rmask & mask_np)
        for name, rmask in region_masks(cfm.Config.IMAGE_SIZE).items()
    }
    center_idx = cfm.center_lead_index(cfm.Config) if cfm.Config.MULTI_LEAD_TUBE else 0

    split_year_map = {"train": train_years, "val": val_years, "test": test_years}
    calibration_split = resolve_calibration_split(args.calibration_split, eval_split)
    calibration_years = split_year_map[calibration_split]
    train_year_set = set(int(y) for y in train_years)
    calibration_year_set = set(int(y) for y in calibration_years)
    eval_year_set = set(int(y) for y in split_years)
    if train_year_set & calibration_year_set:
        raise RuntimeError(
            f"Disjointness assert failed: calibration split '{calibration_split}' overlaps model-training years."
        )
    if train_year_set & eval_year_set:
        raise RuntimeError(
            f"Disjointness assert failed: eval split '{eval_split}' overlaps model-training years."
        )
    if calibration_year_set & eval_year_set:
        raise RuntimeError(
            f"Disjointness assert failed: calibration split '{calibration_split}' overlaps eval split '{eval_split}'."
        )
    print(
        "Disjoint split roles: "
        f"model_train={len(train_year_set)} years, "
        f"calibration={calibration_split} ({len(calibration_year_set)} years), "
        f"eval={eval_split} ({len(eval_year_set)} years)"
    )

    calibration_dataset = select_dataset(
        calibration_split, cfm.Config, shared_data, norm_stats, climo,
        train_indices, val_indices, test_indices,
    )
    monotonic_calibrator, monotonic_candidates, inner_selection_rows, selection_reason = load_or_fit_monotonic_calibrator(
        model,
        calibration_dataset,
        calibration_split,
        calibration_years,
        split_years,
        q95_z,
        sigma_clim,
        months,
        years,
        mask_np,
        device,
        target_mode,
        window_leads,
        lead_indices,
        center_idx,
        target_center_lead,
        ckpt_path,
        requested_calibrator=str(args.calibrator),
        max_cases=int(args.max_calibration_cases),
        max_samples=int(args.max_calibration_samples),
        seed=int(args.seed) + 137,
        min_month_samples=int(args.min_calibration_samples_per_month),
        min_month_positives=int(args.min_calibration_positives_per_month),
        progress_every=int(args.progress_every),
        use_model_sigma=bool(args.use_model_sigma),
    )
    fixed_sigma_calibrator = None
    fixed_sigma_candidates = {}
    fixed_inner_selection_rows: List[Dict[str, object]] = []
    fixed_selection_reason = ""
    if args.use_model_sigma:
        fixed_sigma_calibrator, fixed_sigma_candidates, fixed_inner_selection_rows, fixed_selection_reason = load_or_fit_monotonic_calibrator(
            model,
            calibration_dataset,
            calibration_split,
            calibration_years,
            split_years,
            q95_z,
            sigma_clim,
            months,
            years,
            mask_np,
            device,
            target_mode,
            window_leads,
            lead_indices,
            center_idx,
            target_center_lead,
            ckpt_path,
            requested_calibrator=str(args.calibrator),
            max_cases=int(args.max_calibration_cases),
            max_samples=int(args.max_calibration_samples),
            seed=int(args.seed) + 907,
            min_month_samples=int(args.min_calibration_samples_per_month),
            min_month_positives=int(args.min_calibration_positives_per_month),
            progress_every=int(args.progress_every),
            use_model_sigma=False,
        )

    incremental_models: Dict[str, Any] = {}
    nested_selection_rows: List[Dict[str, object]] = []
    incremental_array_chunk_dir: Optional[Path] = None
    incremental_saved_cells = 0
    incremental_saved_samples = 0
    if args.incremental_skill_diagnostic:
        incremental_x, incremental_y, incremental_years, incremental_base_rates = collect_incremental_skill_pairs(
            model,
            calibration_dataset,
            calibration_split,
            calibration_years,
            q95_z,
            base_rate,
            months,
            years,
            mask_np,
            device,
            target_mode,
            lead_indices,
            center_idx,
            target_center_lead,
            max_cases=int(args.max_calibration_cases),
            max_samples=int(args.max_calibration_samples),
            seed=int(args.seed) + 1709,
            progress_every=int(args.progress_every),
        )
        if set(int(v) for v in np.unique(incremental_years)) & eval_year_set:
            raise RuntimeError("Incremental-skill fitting leaked evaluation years.")
        if args.save_incremental_arrays:
            calibration_array_path = save_incremental_calibration_arrays(
                out_dir,
                incremental_x,
                incremental_y,
                incremental_years,
                incremental_base_rates,
                train_years,
                calibration_years,
                test_years,
                calibration_split,
            )
            incremental_array_chunk_dir = prepare_incremental_test_chunk_dir(out_dir)
            print(f"Saved incremental calibration arrays to: {calibration_array_path}")
        incremental_models, nested_selection_rows = fit_incremental_skill_models(
            incremental_x,
            incremental_y,
            incremental_base_rates,
            incremental_years,
            calibration_split=calibration_split,
            alpha_grid=parse_float_list(args.incremental_alpha_grid),
            l2_grid=parse_float_list(args.incremental_l2_grid),
            steps=int(args.calibration_steps),
            lr=float(args.calibration_lr),
            l2=float(args.calibration_l2),
        )

    train_dataset_for_old_logistic = select_dataset(
        "train", cfm.Config, shared_data, norm_stats, climo,
        train_indices, val_indices, test_indices,
    )
    train_x, train_y, train_months, _ = collect_margin_label_pairs(
        model,
        train_dataset_for_old_logistic,
        "train",
        train_years,
        q95_z,
        sigma_clim,
        months,
        years,
        mask_np,
        device,
        target_mode,
        lead_indices,
        center_idx,
        target_center_lead,
        max_cases=min(int(args.max_calibration_cases), 512),
        max_samples=int(args.max_calibration_samples),
        seed=int(args.seed) + 503,
        progress_every=int(args.progress_every),
        use_model_sigma=bool(args.use_model_sigma),
    )
    old_train_calibrator = fit_margin_monotonic_calibrator(
        train_x,
        train_y,
        train_months,
        method="platt",
        calibration_split="train",
        min_month_samples=10**12,
        min_month_positives=10**12,
    )
    print(
        "Old train-fitted single-logistic diagnostic: "
        f"ECE={old_train_calibrator.calibration_ece:.4f}, "
        f"slope={old_train_calibrator.calibration_slope:.3f}, "
        f"Brier={old_train_calibrator.calibration_brier:.5f}, "
        f"n={old_train_calibrator.n_samples}"
    )

    logistic = train_pooled_logistic_baseline(
        shared_data, train_indices, q95_z, norm_stats, months,
        target_mode=target_mode,
        window_leads=window_leads,
        target_center_lead=target_center_lead,
        max_samples=int(args.max_logistic_samples), seed=int(args.seed)
    )

    persistence_windows = parse_int_list(args.persistence_windows)
    if target_mode == "window":
        reference_name = "windowed_climatology"
        thresholded_name = "thresholded_window_mean"
        pooled_name = "pooled_logistic_window_init_margin"
        analytic_name = "windowed_mu_sigma_error"
    else:
        reference_name = "monthly_climatology"
        thresholded_name = "thresholded_point_model"
        pooled_name = "pooled_logistic_init_margin"
        analytic_name = "stage1_mu_sigma_clim"
    old_calibrated_name = "old_train_logistic_margin"
    calibrated_name = "heldout_monotonic_calibrator"
    fixed_sigma_analytic_name = f"{analytic_name}_cached_sigma"
    fixed_sigma_calibrated_name = "heldout_monotonic_cached_sigma"
    model_names = [
        reference_name,
        "persistence_init",
        *[f"persistence_trailing{w}" for w in persistence_windows],
        thresholded_name,
        pooled_name,
        analytic_name,
        old_calibrated_name,
        calibrated_name,
    ]
    if args.use_model_sigma:
        model_names.extend([fixed_sigma_analytic_name, fixed_sigma_calibrated_name])
    model_names.extend(incremental_models)
    acc = EvaluationAccumulator(model_names, regions)
    year_incremental_acc: Optional[Dict[int, EvaluationAccumulator]] = (
        defaultdict(
            lambda: EvaluationAccumulator(
                [reference_name, "incremental_A_init_margin", NESTED_SHRINKAGE_MODEL_NAME],
                {},
            )
        )
        if NESTED_SHRINKAGE_MODEL_NAME in incremental_models
        else None
    )
    daily_compare_acc = (
        EvaluationAccumulator(["monthly_climatology", "stage1_mu_sigma_clim"], {})
        if target_mode == "window" else None
    )

    h, w = cfm.Config.IMAGE_SIZE
    heat = shared_data["heat_index"]
    print(f"Evaluating split={eval_split}, samples={len(dataset)}")
    leakage_ok = True
    causal_ok = True
    analytic_valid_total = 0
    calibrated_valid_total = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            y, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, t_idx, batch_mask = batch
            t = int(t_idx.item())
            target_t = t + int(target_center_lead)
            target_month = int(months[target_t])
            if target_month not in MJJAS_MONTHS:
                continue
            if int(years[target_t]) in train_years and eval_split != "train":
                leakage_ok = False

            x_t_dev = x_t.to(device)
            x_tm1_dev = x_tm1.to(device)
            x_tm2_dev = x_tm2.to(device)
            spatial_c_dev = spatial_c.to(device)
            vec_c_dev = vec_c.to(device)
            global_fields_dev = global_fields.to(device)
            mask_dev = batch_mask.to(device)

            pred = cfm.predict_direct(
                model, x_t_dev, x_tm1_dev, x_tm2_dev,
                spatial_c_dev, vec_c_dev, global_fields_dev, mask_dev, device,
            )
            raw_model = model.module if hasattr(model, "module") else model
            sigma_pred = getattr(raw_model, "last_sigma", None)
            if target_mode == "window":
                idx = torch.as_tensor(lead_indices, device=pred.device, dtype=torch.long)
                cpu_idx = torch.as_tensor(lead_indices, dtype=torch.long)
                mu_z = pred[0].index_select(0, idx)[:, :h, :w].mean(dim=0).detach().cpu().numpy().astype(np.float32)
                truth_z = y[0].index_select(0, cpu_idx)[:, :h, :w].mean(dim=0).numpy().astype(np.float32)
                center_mu_z = pred[0, center_idx, :h, :w].detach().cpu().numpy().astype(np.float32)
                center_truth_z = y[0, center_idx, :h, :w].numpy().astype(np.float32)
                if args.use_model_sigma or args.incremental_skill_diagnostic:
                    if sigma_pred is None:
                        if args.use_model_sigma:
                            raise RuntimeError("--use_model_sigma requested, but model.last_sigma is missing.")
                        sigma_model_z = None
                    else:
                        sig = sigma_pred[0].index_select(0, idx)[:, :h, :w].float()
                        sigma_model_z = (torch.sqrt(sig.square().sum(dim=0).clamp_min(1e-8)) / max(len(lead_indices), 1)).detach().cpu().numpy().astype(np.float32)
                else:
                    sigma_model_z = None
            elif cfm.Config.MULTI_LEAD_TUBE:
                mu_z = pred[0, center_idx, :h, :w].detach().cpu().numpy().astype(np.float32)
                truth_z = y[0, center_idx, :h, :w].numpy().astype(np.float32)
                center_mu_z = mu_z
                center_truth_z = truth_z
                if args.use_model_sigma or args.incremental_skill_diagnostic:
                    if sigma_pred is None:
                        if args.use_model_sigma:
                            raise RuntimeError("--use_model_sigma requested, but model.last_sigma is missing.")
                        sigma_model_z = None
                    else:
                        sigma_model_z = sigma_pred[0, center_idx, :h, :w].detach().cpu().numpy().astype(np.float32)
                else:
                    sigma_model_z = None
            else:
                mu_z = pred[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
                truth_z = y[0, 0, :h, :w].numpy().astype(np.float32)
                center_mu_z = mu_z
                center_truth_z = truth_z
                if args.use_model_sigma or args.incremental_skill_diagnostic:
                    if sigma_pred is None:
                        if args.use_model_sigma:
                            raise RuntimeError("--use_model_sigma requested, but model.last_sigma is missing.")
                        sigma_model_z = None
                    else:
                        sigma_model_z = sigma_pred[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
                else:
                    sigma_model_z = None
            if batch_idx == 0:
                mu_land = mu_z[mask_np & np.isfinite(mu_z)]
                if mu_land.size:
                    p = np.nanpercentile(mu_land, [1, 50, 99])
                    print(
                        "First-sample mu_z land range check: "
                        f"min={np.nanmin(mu_land):+.3f}, p01={p[0]:+.3f}, "
                        f"p50={p[1]:+.3f}, p99={p[2]:+.3f}, max={np.nanmax(mu_land):+.3f}"
                    )

            q = q95_z[target_month]
            sigma = sigma_model_z if args.use_model_sigma else sigma_clim[target_month]
            fixed_sigma = sigma_clim[target_month]
            truth = (truth_z > q).astype(np.float32)
            truth[~mask_np] = np.nan
            stage1 = normal_cdf((mu_z - q) / np.maximum(sigma, 0.1))
            stage1[~mask_np] = np.nan
            fixed_stage1 = None
            fixed_calibrated = None
            if args.use_model_sigma:
                fixed_stage1 = normal_cdf((mu_z - q) / np.maximum(fixed_sigma, 0.1))
                fixed_stage1[~mask_np] = np.nan
                fixed_calibrated = fixed_sigma_calibrator.predict_grid(mu_z, q, fixed_sigma, target_month, mask_np)
            old_train_calibrated = old_train_calibrator.predict_grid(mu_z, q, sigma, target_month, mask_np)
            calibrated = monotonic_calibrator.predict_grid(mu_z, q, sigma, target_month, mask_np)
            stage1_valid = mask_np & np.isfinite(stage1) & np.isfinite(truth)
            calibrated_valid = mask_np & np.isfinite(calibrated) & np.isfinite(truth)
            if not np.array_equal(stage1_valid, calibrated_valid):
                missing = int(np.sum(stage1_valid & ~calibrated_valid))
                extra = int(np.sum(calibrated_valid & ~stage1_valid))
                raise RuntimeError(
                    "Held-out calibrated forecast valid-cell mask differs from analytic forecast "
                    f"at sample={batch_idx}, target_t={target_t}, month={target_month}: "
                    f"analytic_count={int(np.sum(stage1_valid))}, calibrated_count={int(np.sum(calibrated_valid))}, "
                    f"missing={missing}, extra={extra}"
                )
            analytic_valid_total += int(np.sum(stage1_valid))
            calibrated_valid_total += int(np.sum(calibrated_valid))
            monthly_climo = base_rate[target_month]
            thresholded = (mu_z > q).astype(np.float32)
            thresholded[~mask_np] = np.nan

            init_margin = x_t[0, 0].numpy().astype(np.float32) - q
            logistic_prob = logistic.predict(init_margin)
            logistic_prob[~mask_np] = np.nan
            incremental_probs: Dict[str, np.ndarray] = {}
            if incremental_models:
                incremental_feature_fields = {
                    "init_margin": init_margin,
                    "forecast_margin": mu_z - q,
                    "predicted_sigma": (
                        sigma_model_z
                        if sigma_model_z is not None
                        else np.full_like(mu_z, np.nan, dtype=np.float32)
                    ),
                }
                incremental_probs = {
                    name: predict_incremental_skill_grid(
                        calibrator,
                        incremental_feature_fields,
                        mask_np,
                        base_rate=base_rate[target_month],
                    )
                    for name, calibrator in incremental_models.items()
                }
                if incremental_array_chunk_dir is not None:
                    saved_cells = save_incremental_test_chunk(
                        incremental_array_chunk_dir,
                        batch_idx,
                        int(getattr(cfm.Config, "CV_FOLD", cfm.Config.CV_TEST_OFFSETS[0])),
                        int(years[target_t]),
                        target_month,
                        init_margin,
                        mu_z - q,
                        sigma_model_z,
                        truth,
                        monthly_climo,
                        mask_np,
                        mu_z=mu_z,
                        truth_z=truth_z,
                        init_time_index=t,
                        target_center_time_index=target_t,
                    )
                    incremental_saved_cells += saved_cells
                    incremental_saved_samples += int(saved_cells > 0)

            run_start = find_run_start(runs, t)
            if t >= target_t:
                causal_ok = False

            forecasts = {
                reference_name: monthly_climo,
                thresholded_name: thresholded,
                pooled_name: logistic_prob,
                analytic_name: stage1,
                old_calibrated_name: old_train_calibrated,
                calibrated_name: calibrated,
            }
            auc_scores = {
                analytic_name: stage1,
                old_calibrated_name: stage1,
                calibrated_name: stage1,
            }
            if args.use_model_sigma:
                forecasts[fixed_sigma_analytic_name] = fixed_stage1
                forecasts[fixed_sigma_calibrated_name] = fixed_calibrated
                auc_scores[fixed_sigma_analytic_name] = fixed_stage1
                auc_scores[fixed_sigma_calibrated_name] = fixed_stage1
            forecasts.update(incremental_probs)
            if target_mode == "window":
                forecasts["persistence_init"] = trailing_windowed_exceedance_probability(
                    heat, t, 1, run_start, q95_z, months, norm_stats, mask_np,
                    window_leads, target_center_lead,
                )
                for window in persistence_windows:
                    forecasts[f"persistence_trailing{window}"] = trailing_windowed_exceedance_probability(
                        heat, t, window, run_start, q95_z, months, norm_stats, mask_np,
                        window_leads, target_center_lead,
                    )
            else:
                init_prob = (x_t[0, 0].numpy().astype(np.float32) > q).astype(np.float32)
                init_prob[~mask_np] = np.nan
                forecasts["persistence_init"] = init_prob
                for window in persistence_windows:
                    forecasts[f"persistence_trailing{window}"] = trailing_exceedance_probability(
                        heat, t, target_t, window, run_start, q95_z, months, norm_stats, mask_np
                    )

            for name, prob in forecasts.items():
                acc.update(name, prob, truth, mask_np, target_month, auc_score=auc_scores.get(name))
            if year_incremental_acc is not None:
                target_year = int(years[target_t])
                for name in (reference_name, "incremental_A_init_margin", NESTED_SHRINKAGE_MODEL_NAME):
                    year_incremental_acc[target_year].update(
                        name,
                        forecasts[name],
                        truth,
                        mask_np,
                        target_month,
                    )

            if daily_compare_acc is not None:
                daily_target_t = t + int(cfm.Config.LEAD_TIME)
                daily_month = int(months[daily_target_t])
                if daily_month in MJJAS_MONTHS:
                    qd = daily_q95_z[daily_month]
                    sigd = daily_sigma_clim[daily_month]
                    daily_truth = (center_truth_z > qd).astype(np.float32)
                    daily_truth[~mask_np] = np.nan
                    daily_stage1 = normal_cdf((center_mu_z - qd) / np.maximum(sigd, 0.1))
                    daily_stage1[~mask_np] = np.nan
                    daily_compare_acc.update(
                        "monthly_climatology", daily_base_rate[daily_month], daily_truth, mask_np, daily_month
                    )
                    daily_compare_acc.update(
                        "stage1_mu_sigma_clim", daily_stage1, daily_truth, mask_np, daily_month
                    )

            if (batch_idx + 1) % max(1, int(args.progress_every)) == 0:
                print(f"  processed {batch_idx + 1}/{len(dataset)} samples")

    if not leakage_ok:
        raise RuntimeError("Leakage assert failed: evaluation target year was in train years.")
    if not causal_ok:
        raise RuntimeError("Causality assert failed for persistence baseline.")
    if analytic_valid_total != calibrated_valid_total:
        raise RuntimeError(
            "Held-out calibrated valid-cell total differs from analytic forecast: "
            f"analytic={analytic_valid_total}, calibrated={calibrated_valid_total}"
        )

    summary_rows = acc.summary_rows(reference_name)
    monthly_rows = acc.monthly_rows(reference_name)
    region_rows = acc.region_rows()
    rel_tables = {name: metric.rel.table() for name, metric in acc.metrics.items()}
    bootstrap_per_year_rows: List[Dict[str, object]] = []
    bootstrap_summary_rows: List[Dict[str, object]] = []
    if year_incremental_acc is not None:
        bootstrap_per_year_rows, bootstrap_summary_rows = year_block_bootstrap_incremental_comparison(
            year_incremental_acc,
            reference_name,
            "incremental_A_init_margin",
            NESTED_SHRINKAGE_MODEL_NAME,
            reps=int(args.year_bootstrap_reps),
            seed=int(args.seed) + 2909,
        )

    write_csv(out_dir / "exceedance_results.csv", summary_rows)
    write_csv(out_dir / "monthly_exceedance_results.csv", monthly_rows)
    write_csv(out_dir / "regional_exceedance_count_mae.csv", region_rows)
    write_csv(out_dir / "old_train_logistic_margin_coefficients.csv", old_train_calibrator.coefficient_rows())
    write_csv(out_dir / "heldout_monotonic_calibrator_coefficients.csv", monotonic_calibrator.coefficient_rows())
    write_csv(
        out_dir / "heldout_monotonic_calibrator_candidates.csv",
        [
            {
                "candidate": name,
                "selected": name == monotonic_calibrator.method,
                "selection_reason": selection_reason if name == monotonic_calibrator.method else "",
                "calibration_split": cal.calibration_split,
                "n_samples": cal.n_samples,
                "event_rate": cal.event_rate,
                "calibration_slope": cal.calibration_slope,
                "calibration_ece": cal.calibration_ece,
                "calibration_brier": cal.calibration_brier,
                "month_models": len(cal.by_month),
            }
            for name, cal in sorted(monotonic_candidates.items())
        ],
    )
    write_csv(out_dir / "heldout_monotonic_inner_selection.csv", inner_selection_rows)
    if incremental_models:
        write_csv(
            out_dir / "incremental_skill_coefficients.csv",
            [
                {"diagnostic_model": model_name, **row}
                for model_name, calibrator in incremental_models.items()
                for row in calibrator.coefficient_rows()
            ],
        )
        write_csv(out_dir / "incremental_nested_year_selection.csv", nested_selection_rows)
        write_csv(out_dir / "incremental_nested_test_year_metrics.csv", bootstrap_per_year_rows)
        write_csv(out_dir / "incremental_nested_year_bootstrap.csv", bootstrap_summary_rows)
        if incremental_array_chunk_dir is not None:
            manifest_path = save_incremental_array_manifest(
                out_dir,
                args.run_name,
                target_mode,
                window_leads,
                train_years,
                calibration_years,
                test_years,
                calibration_split,
                eval_split,
                incremental_saved_samples,
                incremental_saved_cells,
            )
            print(
                "Saved stitchable incremental arrays: "
                f"samples={incremental_saved_samples}, valid_cells={incremental_saved_cells}, "
                f"manifest={manifest_path}"
            )
    if args.use_model_sigma and fixed_sigma_calibrator is not None:
        write_csv(out_dir / "heldout_monotonic_cached_sigma_coefficients.csv", fixed_sigma_calibrator.coefficient_rows())
        write_csv(
            out_dir / "heldout_monotonic_cached_sigma_candidates.csv",
            [
                {
                    "candidate": name,
                    "selected": name == fixed_sigma_calibrator.method,
                    "selection_reason": fixed_selection_reason if name == fixed_sigma_calibrator.method else "",
                    "calibration_split": cal.calibration_split,
                    "n_samples": cal.n_samples,
                    "event_rate": cal.event_rate,
                    "calibration_slope": cal.calibration_slope,
                    "calibration_ece": cal.calibration_ece,
                    "calibration_brier": cal.calibration_brier,
                    "month_models": len(cal.by_month),
                }
                for name, cal in sorted(fixed_sigma_candidates.items())
            ],
        )
        write_csv(out_dir / "heldout_monotonic_cached_sigma_inner_selection.csv", fixed_inner_selection_rows)
    for name, rows in rel_tables.items():
        write_csv(out_dir / f"reliability_{name}.csv", rows)
    plot_reliability(out_dir / "reliability_diagram.png", rel_tables)

    stage1 = next(row for row in summary_rows if row["model"] == analytic_name)
    old_train_row = next(row for row in summary_rows if row["model"] == old_calibrated_name)
    calibrated_row = next(row for row in summary_rows if row["model"] == calibrated_name)
    fixed_stage1_row = next((row for row in summary_rows if row["model"] == fixed_sigma_analytic_name), None)
    fixed_calibrated_row = next((row for row in summary_rows if row["model"] == fixed_sigma_calibrated_name), None)
    thresholded = next(row for row in summary_rows if row["model"] == thresholded_name)
    climo = next(row for row in summary_rows if row["model"] == reference_name)
    stage1_rel = acc.metrics[analytic_name].rel.table()
    calibrated_rel = acc.metrics[calibrated_name].rel.table()
    bin_02 = next((r for r in stage1_rel if r["lo"] <= 0.2 < r["hi"]), None)
    calibrated_bin_02 = next((r for r in calibrated_rel if r["lo"] <= 0.2 < r["hi"]), None)
    daily_stage1_bss = float("nan")
    if daily_compare_acc is not None:
        daily_rows = daily_compare_acc.summary_rows("monthly_climatology")
        daily_stage1_bss = next(
            row["bss_vs_monthly_climo"]
            for row in daily_rows
            if row["model"] == "stage1_mu_sigma_clim"
        )

    print("\nExceedance evaluation summary")
    print("=============================")
    for row in summary_rows:
        print(
            f"{row['model']:28s} N={int(row['valid_count'])} Brier={row['brier']:.5f} "
            f"BSS={row['bss_vs_monthly_climo']:+.4f} "
            f"slope={row['reliability_slope']:.3f} ECE={row['ece']:.4f} "
            f"ROC-AUC={row['roc_auc']:.3f} PR-AUC={row['pr_auc']:.3f}"
        )
    print("\nAcceptance checks")
    print("=================")
    print(
        "Disjointness asserts: PASS "
        f"(train={len(train_year_set)} years, calibration={calibration_split}, eval={eval_split})"
    )
    print(
        "Valid-cell mask assert: PASS "
        f"(analytic={analytic_valid_total}, heldout_calibrated={calibrated_valid_total})"
    )
    print("Leakage asserts: PASS (train-only thresholds/stats; calibration/eval years held out)")
    print("Causal persistence asserts: PASS (history windows end at init day)")
    print(f"{reference_name} Brier reference: {climo['brier']:.5f}")
    print("Held-out monotonic calibration candidates:")
    for name, cal in sorted(monotonic_candidates.items()):
        selected = "selected" if name == monotonic_calibrator.method else "candidate"
        print(
            f"  {name}: ECE={cal.calibration_ece:.4f}, "
            f"slope={cal.calibration_slope:.3f}, Brier={cal.calibration_brier:.5f} ({selected})"
        )
    if inner_selection_rows:
        print("Inner-selection calibration scores:")
        for row in inner_selection_rows:
            print(
                f"  {row['candidate']}: ECE={float(row['inner_selection_ece']):.4f}, "
                f"slope={float(row['inner_selection_slope']):.3f}, "
                f"Brier={float(row['inner_selection_brier']):.5f}"
            )
    print(f"Calibrator selection: {monotonic_calibrator.method} ({selection_reason})")
    if incremental_models:
        incremental_rows = {
            row["model"]: row
            for row in summary_rows
            if row["model"] in incremental_models
        }
        anchored_row = incremental_rows[CLIMATOLOGY_ANCHORED_MODEL_NAME]
        nested_row = incremental_rows[NESTED_SHRINKAGE_MODEL_NAME]
        reference_valid_count = next(
            row["valid_count"] for row in summary_rows if row["model"] == reference_name
        )
        if anchored_row["valid_count"] != reference_valid_count or nested_row["valid_count"] != reference_valid_count:
            raise RuntimeError(
                "Climatology-anchored incremental model valid-cell count differs from reference: "
                f"anchored={anchored_row['valid_count']}, nested={nested_row['valid_count']}, "
                f"reference={reference_valid_count}"
            )
        init_bss = incremental_rows["incremental_A_init_margin"]["bss_vs_monthly_climo"]
        print("Incremental-skill diagnostic (fit on calibration split, scored on eval split):")
        for name in (*INCREMENTAL_SKILL_SPECS, CLIMATOLOGY_ANCHORED_MODEL_NAME, NESTED_SHRINKAGE_MODEL_NAME):
            if name not in incremental_rows:
                continue
            row = incremental_rows[name]
            print(
                f"  {name}: BSS={row['bss_vs_monthly_climo']:+.4f}, "
                f"delta_vs_A={row['bss_vs_monthly_climo'] - init_bss:+.4f}, "
                f"ROC-AUC={row['roc_auc']:.3f}, ECE={row['ece']:.4f}"
            )
        forecast_models = [
            name for name in (
                "incremental_B_forecast_margin",
                "incremental_C_init_plus_forecast",
                "incremental_D_init_forecast_sigma",
                CLIMATOLOGY_ANCHORED_MODEL_NAME,
                NESTED_SHRINKAGE_MODEL_NAME,
            )
            if name in incremental_rows
        ]
        best_forecast_name = max(
            forecast_models,
            key=lambda name: incremental_rows[name]["bss_vs_monthly_climo"],
        )
        best_forecast_delta = (
            incremental_rows[best_forecast_name]["bss_vs_monthly_climo"] - init_bss
        )
        verdict = "PASS" if best_forecast_delta > 0.0 else "FAIL"
        print(
            f"Incremental-skill verdict: {verdict} "
            f"(best={best_forecast_name}, delta_BSS_vs_init={best_forecast_delta:+.4f})"
        )
        print(
            "Incremental-skill leakage assert: PASS "
            f"(fit={calibration_split}, eval={eval_split}, disjoint years)"
        )
        print(
            "Climatology-anchor asserts: PASS "
            f"(fixed train-only pixel/month offset, no learned intercept, "
            f"valid_cells={int(anchored_row['valid_count'])})"
        )
        print(
            "Climatology-anchor comparison: "
            f"delta_BSS_vs_A={anchored_row['bss_vs_monthly_climo'] - init_bss:+.4f}, "
            f"delta_BSS_vs_C="
            f"{anchored_row['bss_vs_monthly_climo'] - incremental_rows['incremental_C_init_plus_forecast']['bss_vs_monthly_climo']:+.4f}"
        )
        nested = incremental_models[NESTED_SHRINKAGE_MODEL_NAME]
        print(
            "Nested year-wise shrinkage selection: "
            f"alpha={nested.alpha:.3f}, l2={nested.selected_l2:g}, "
            "criterion=leave-one-validation-year-out Brier"
        )
        for row in bootstrap_summary_rows:
            print(
                f"Year-block bootstrap {row['metric']}: estimate={float(row['estimate']):+.4f}, "
                f"95% CI=[{float(row['ci_2.5']):+.4f}, {float(row['ci_97.5']):+.4f}], "
                f"P(>0)={float(row['probability_gt_zero']):.3f}, "
                f"reps={int(row['bootstrap_reps'])}"
            )
        print(
            "Nested-shrinkage leakage assert: PASS "
            "(alpha and L2 selected only by leave-one-calibration-year-out predictions; "
            "test years used once for reporting)"
        )
    if target_mode == "window":
        print(
            f"Windowed BSS vs windowed climatology: {stage1['bss_vs_monthly_climo']:+.4f} "
            f"(daily Stage 1 BSS same checkpoint: {daily_stage1_bss:+.4f})"
        )
    else:
        print(f"Stage 1 BSS vs monthly climatology: {stage1['bss_vs_monthly_climo']:+.4f}")
    print(
        f"Old train-logistic BSS vs {reference_name}: "
        f"{old_train_row['bss_vs_monthly_climo']:+.4f}"
    )
    print(
        f"Held-out monotonic BSS vs {reference_name}: "
        f"{calibrated_row['bss_vs_monthly_climo']:+.4f}"
    )
    if args.use_model_sigma and fixed_stage1_row is not None and fixed_calibrated_row is not None:
        print(
            "Cached-sigma analytic comparison: "
            f"BSS={fixed_stage1_row['bss_vs_monthly_climo']:+.4f}, "
            f"slope={fixed_stage1_row['reliability_slope']:.3f}, ECE={fixed_stage1_row['ece']:.4f}"
        )
        print(
            "Cached-sigma held-out monotonic comparison: "
            f"BSS={fixed_calibrated_row['bss_vs_monthly_climo']:+.4f}, "
            f"slope={fixed_calibrated_row['reliability_slope']:.3f}, ECE={fixed_calibrated_row['ece']:.4f}"
        )
    print(
        "Thresholded point-model baseline reported: "
        f"Brier={thresholded['brier']:.5f}, BSS={thresholded['bss_vs_monthly_climo']:+.4f}"
    )
    if bin_02 is not None:
        print(
            f"{analytic_name} reliability 0.2-bin sanity: "
            f"count={int(bin_02['count'])}, mean_pred={bin_02['mean_pred']:.3f}, "
            f"obs_freq={bin_02['obs_freq']:.3f}"
        )
    if calibrated_bin_02 is not None:
        print(
            "Held-out calibrated reliability 0.2-bin sanity: "
            f"count={int(calibrated_bin_02['count'])}, "
            f"mean_pred={calibrated_bin_02['mean_pred']:.3f}, "
            f"obs_freq={calibrated_bin_02['obs_freq']:.3f}"
        )
        print(
            "Held-out calibrated reliability summary: "
            f"slope={calibrated_row['reliability_slope']:.3f}, ECE={calibrated_row['ece']:.4f}"
        )
    auc_diff = abs(float(calibrated_row["roc_auc"]) - float(stage1["roc_auc"]))
    print(
        "Monotonic AUC check: "
        f"analytic={stage1['roc_auc']:.4f}, heldout={calibrated_row['roc_auc']:.4f}, "
        f"abs_diff={auc_diff:.4g}"
    )
    if np.isfinite(auc_diff) and auc_diff > 0.005:
        raise RuntimeError(
            "Monotonic AUC assert failed: analytic and held-out calibrated ROC-AUC differ by "
            f"{auc_diff:.4f} > 0.005."
        )
    print(
        "PR-AUC is secondary only; selection/headline metric is BSS with reliability diagnostics."
    )
    print(f"Saved exceedance outputs to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--cv_fold", type=int, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default=None, help="Backward-compatible alias for --eval_split.")
    parser.add_argument("--eval_split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", default="exceedance_eval")
    parser.add_argument("--cv_stride", type=int, default=5)
    parser.add_argument("--multi_lead_tube", action="store_true")
    parser.add_argument("--prediction_leads", default="12,13,14,15,16,17,18")
    parser.add_argument("--tube_decode_chunk_size", type=int, default=0, help="Decode this many tube leads at a time to bound GPU memory; 0 decodes all leads together.")
    parser.add_argument("--target_mode", choices=["daily", "window"], default="daily")
    parser.add_argument("--window_leads", default=None, help="Comma-separated lead offsets for --target_mode window; defaults to predicted tube leads.")
    parser.add_argument("--use_model_sigma", action="store_true", help="Use distributional model sigma in Phi((mu-q95)/sigma) instead of cached sigma.")
    parser.add_argument("--incremental_skill_diagnostic", action="store_true", help="Fit A-F incremental diagnostics, including climatology-anchored and nested year-wise shrinkage models, on calibration years and score on eval years.")
    parser.add_argument("--save_incremental_arrays", action="store_true", help="Persist minimal calibration pairs and full test-cell incremental arrays for stitch_exceedance_folds.py.")
    parser.add_argument("--incremental_alpha_grid", default="0,0.1,0.25,0.5,0.75,1.0", help="Shrinkage alpha candidates for nested year-wise Model F; must include 0.")
    parser.add_argument("--incremental_l2_grid", default="0,0.0001,0.001,0.01,0.1,1.0", help="L2 candidates selected by leave-one-calibration-year-out Brier for Model F.")
    parser.add_argument("--year_bootstrap_reps", type=int, default=1000, help="Whole-test-year bootstrap replicates for Model F versus Model A.")
    parser.add_argument("--persistence_windows", default="7,14,30")
    parser.add_argument("--max_logistic_samples", type=int, default=1000000)
    parser.add_argument("--calibration_split", choices=["auto", "train", "val", "test"], default="val")
    parser.add_argument("--calibrator", choices=["platt", "isotonic", "auto"], default="auto")
    parser.add_argument("--allow_eval_split_calibration", action="store_true")
    parser.add_argument("--max_calibration_cases", type=int, default=1000000)
    parser.add_argument("--max_calibration_samples", type=int, default=250000)
    parser.add_argument("--min_calibration_samples_per_month", type=int, default=20000)
    parser.add_argument("--min_calibration_positives_per_month", type=int, default=100)
    parser.add_argument("--calibration_steps", type=int, default=200)
    parser.add_argument("--calibration_lr", type=float, default=0.1)
    parser.add_argument("--calibration_l2", type=float, default=1e-4)
    parser.add_argument("--progress_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
