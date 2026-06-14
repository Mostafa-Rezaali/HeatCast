#!/usr/bin/env python3
"""Bias-correct and score ECMWF ENS reforecasts on the HeatCast exceedance scoreboard."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

import cfm_mesh_train as cfm
import exceedance_eval as ee
from ens_common import (
    ENS_BENCHMARK_BANNER,
    apply_quantile_mapping,
    fit_quantile_mapping,
    member_fraction_probability,
)
from ens_ingest import normalize_rt_tag, validate_ingested_output


def configure_fold(fold: int, window_leads: Sequence[int], cv_stride: int) -> None:
    fold = int(fold) % int(cv_stride)
    cfm.Config.CV_STRIDE = int(cv_stride)
    cfm.Config.CV_FOLD = fold
    cfm.Config.CV_TEST_OFFSETS = (fold,)
    cfm.Config.CV_VAL_OFFSETS = ((fold + 1) % int(cv_stride),)
    cfm.Config.MULTI_LEAD_TUBE = True
    cfm.Config.PREDICTION_LEADS = tuple(int(value) for value in window_leads)
    cfm.apply_extended_global_fields()


def load_ens_scoring_shared_data(config=cfm.Config) -> Dict[str, np.ndarray]:
    """Load only the read-only HeatCast arrays required by ENS scoring."""
    cache_dir = Path(config.OUTPUT_DIR) / "data_cache"
    paths = {
        "heat_index": cache_dir / "heat_index.npy",
        "time_values": cache_dir / "time_values.npy",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing HeatCast scoring cache files: "
            f"{missing}. Run a HeatCast training/evaluation setup once to build them."
        )
    try:
        shared = {
            name: np.load(path, mmap_mode="r")
            for name, path in paths.items()
        }
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Could not open read-only HeatCast scoring caches under {cache_dir}: {exc}. "
            "Repair the disk cache before ENS scoring."
        ) from exc
    heat = shared["heat_index"]
    time_values = shared["time_values"]
    if heat.ndim != 3 or time_values.ndim != 1 or heat.shape[-1] != time_values.size:
        raise RuntimeError(
            "Invalid HeatCast scoring cache shapes: "
            f"heat_index={heat.shape}, time_values={time_values.shape}."
        )
    print(
        "Loaded lightweight ENS scoring cache: "
        f"heat_index={heat.shape}, time_values={time_values.shape}; "
        "global/local predictor caches intentionally skipped."
    )
    return shared


def load_ingested_files(root: Path, window_leads: Sequence[int]) -> Dict[int, Path]:
    output: Dict[int, Path] = {}
    missing_leads: List[str] = []
    invalid_files: List[str] = []
    required = set(int(value) for value in window_leads)
    for path in sorted(root.glob("init_*.npz")):
        valid, reason = validate_ingested_output(path, window_leads)
        if not valid:
            invalid_files.append(f"{path.name}: {reason}")
            continue
        with np.load(path, allow_pickle=False) as data:
            init_t = int(np.asarray(data["init_time_index"]).item())
            available = set(np.atleast_1d(data["leads"]).astype(int).tolist())
        if not required.issubset(available):
            missing_leads.append(f"{path.name}: missing {sorted(required - available)}")
            continue
        if init_t in output:
            raise RuntimeError(f"Duplicate ingested ENS init_time_index={init_t}: {output[init_t]} and {path}")
        output[init_t] = path
    if invalid_files:
        preview = "\n".join(f"  {line}" for line in invalid_files[:100])
        raise RuntimeError(
            f"Found {len(invalid_files)} invalid ingested ENS outputs under {root}:\n{preview}\n"
            "Rerun submit_ens_ingest.slurm; its resume validator will remove and rebuild them."
        )
    if missing_leads:
        print("Skipped ingested files that do not contain the requested window:")
        for line in missing_leads[:25]:
            print(f"  {line}")
    if not output:
        raise FileNotFoundError(f"No usable init_*.npz files found under {root}")
    return output


def load_window_stats(window_leads: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, Path]:
    path = Path(ee.window_exceedance_cache_path(window_leads))
    if not path.exists():
        raise FileNotFoundError(
            f"Missing fold-safe PRISM window threshold cache: {path}. "
            "Run the corresponding HeatCast window evaluator first. ENS must not define its own events."
        )
    with np.load(path, allow_pickle=False) as data:
        cached_leads = tuple(np.atleast_1d(data["window_leads"]).astype(int).tolist())
        cached_fold = int(np.asarray(data["cv_fold"]).item())
        if cached_leads != tuple(int(value) for value in window_leads):
            raise RuntimeError(f"Window cache leads {cached_leads} do not match requested {tuple(window_leads)}.")
        if cached_fold != int(cfm.Config.CV_FOLD):
            raise RuntimeError(f"Window cache fold={cached_fold} does not match fold={cfm.Config.CV_FOLD}.")
        return (
            np.asarray(data["q95_z"], dtype=np.float32),
            np.asarray(data["base_rate"], dtype=np.float32),
            path,
        )


def load_ens_members(path: Path, window_leads: Sequence[int]) -> Dict[int, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        values = np.asarray(data["t2max"], dtype=np.float32)
        leads = tuple(np.atleast_1d(data["leads"]).astype(int).tolist())
    if values.ndim != 4:
        raise RuntimeError(f"{path}: expected t2max=(member,lead,H,W), got {values.shape}.")
    return {int(lead): values[:, leads.index(int(lead))] for lead in window_leads}


def quantile_cache_dir(cache_root: Path, window_leads: Sequence[int], rt_tag: str = "") -> Path:
    return (
        cache_root
        / (normalize_rt_tag(rt_tag) or "legacy")
        / f"fold{int(cfm.Config.CV_FOLD)}"
        / f"window_{ee.lead_list_label(window_leads)}"
    )


def mapping_path(cache_dir: Path, month: int, lead: int) -> Path:
    return cache_dir / f"month{int(month):02d}_lead{int(lead):02d}.npz"


def required_target_months_by_lead(
    files_by_init: Mapping[int, Path],
    months: np.ndarray,
    window_leads: Sequence[int],
) -> Dict[int, Tuple[int, ...]]:
    required: Dict[int, Tuple[int, ...]] = {}
    for lead in window_leads:
        target_months = {
            int(months[int(init_t) + int(lead)])
            for init_t in files_by_init
            if int(init_t) + int(lead) < len(months)
            and int(months[int(init_t) + int(lead)]) in ee.MJJAS_MONTHS
        }
        if not target_months:
            raise RuntimeError(f"No valid ENS target months available for lead={lead}.")
        required[int(lead)] = tuple(sorted(target_months))
    return required


def build_or_load_quantile_mapping(
    files_by_init: Mapping[int, Path],
    train_years: Iterable[int],
    shared_data: Mapping[str, np.ndarray],
    norm_stats: Mapping[str, object],
    land_mask: np.ndarray,
    window_leads: Sequence[int],
    cache_root: Path,
    quantile_count: int,
    rt_tag: str = "",
) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    rt_tag = normalize_rt_tag(rt_tag)
    cache_dir = quantile_cache_dir(cache_root, window_leads, rt_tag)
    cache_dir.mkdir(parents=True, exist_ok=True)
    months, years, _ = ee.month_doy_year_arrays(shared_data["time_values"])
    heat = shared_data["heat_index"]
    train_year_set = set(int(value) for value in train_years)
    quantile_levels = np.linspace(0.0, 1.0, int(quantile_count), dtype=np.float64)
    land_count = int(np.sum(land_mask))
    mappings: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    sanity_errors: List[float] = []
    required_months = required_target_months_by_lead(files_by_init, months, window_leads)
    print(
        "Required ENS target months by lead: "
        + ", ".join(f"+{lead}:{required_months[int(lead)]}" for lead in window_leads)
    )

    for lead in window_leads:
        grouped: Dict[int, List[Tuple[int, Path]]] = defaultdict(list)
        for init_t, path in files_by_init.items():
            target_t = int(init_t) + int(lead)
            if target_t >= len(years) or int(years[target_t]) not in train_year_set:
                continue
            month = int(months[target_t])
            if month in ee.MJJAS_MONTHS:
                grouped[month].append((int(init_t), path))

        for month in required_months[int(lead)]:
            path = mapping_path(cache_dir, month, int(lead))
            pairs = grouped.get(month, [])
            mapping_init_indices = np.array(
                sorted(int(init_t) for init_t, _ in pairs),
                dtype=np.int32,
            )
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    if (
                        int(np.asarray(data["cv_fold"]).item()) == int(cfm.Config.CV_FOLD)
                        and tuple(np.atleast_1d(data["window_leads"]).astype(int).tolist())
                        == tuple(int(value) for value in window_leads)
                        and set(np.atleast_1d(data["train_years"]).astype(int).tolist()) == train_year_set
                        and (
                            str(np.asarray(data["rt_tag"]).item()) if "rt_tag" in data.files else ""
                        ) == rt_tag
                        and "mapping_init_indices" in data.files
                        and np.array_equal(
                            np.atleast_1d(data["mapping_init_indices"]).astype(np.int32),
                            mapping_init_indices,
                        )
                        and np.asarray(data["source_quantiles"]).shape == (int(quantile_count), land_count)
                        and np.asarray(data["target_quantiles"]).shape == (int(quantile_count), land_count)
                    ):
                        mappings[(month, int(lead))] = (
                            np.asarray(data["source_quantiles"], dtype=np.float32),
                            np.asarray(data["target_quantiles"], dtype=np.float32),
                        )
                        sanity_errors.append(float(np.asarray(data["train_mean_mae"]).item()))
                        continue
            if not pairs:
                raise RuntimeError(
                    f"No train-year ENS inits available for required target month={month}, lead={lead}. "
                    f"Required target months for this lead are {required_months[int(lead)]}."
                )
            source_rows: List[np.ndarray] = []
            target_rows: List[np.ndarray] = []
            for init_t, init_path in pairs:
                source = load_ens_members(init_path, (int(lead),))[int(lead)][:, land_mask]
                target = ee._normalize_field(
                    np.asarray(heat[:, :, int(init_t) + int(lead)]),
                    norm_stats,
                )[land_mask]
                source_rows.append(source.astype(np.float32))
                target_rows.append(target[None, :].astype(np.float32))
            source_stack = np.concatenate(source_rows, axis=0)
            target_stack = np.concatenate(target_rows, axis=0)
            source_q, target_q = fit_quantile_mapping(source_stack, target_stack, quantile_levels)
            mapped = apply_quantile_mapping(source_stack, source_q, target_q)
            with np.errstate(all="ignore"):
                mean_error = np.abs(np.nanmean(mapped, axis=0) - np.nanmean(target_stack, axis=0))
                train_mean_mae = float(np.nanmean(mean_error))
            print(
                f"  qmap fold={cfm.Config.CV_FOLD} month={month} lead={lead}: "
                f"inits={len(pairs)}, train monthly-mean MAE={train_mean_mae:.4f} z"
            )
            if not np.isfinite(train_mean_mae) or train_mean_mae > 0.1:
                raise RuntimeError(
                    f"Quantile mapping train reproduction failed for month={month}, lead={lead}: "
                    f"mean absolute per-pixel monthly-mean error={train_mean_mae:.4f} z > 0.1."
                )
            np.savez_compressed(
                path,
                source_quantiles=source_q.astype(np.float32),
                target_quantiles=target_q.astype(np.float32),
                quantile_levels=quantile_levels.astype(np.float32),
                train_mean_mae=np.array(train_mean_mae, dtype=np.float32),
                cv_fold=np.array(int(cfm.Config.CV_FOLD), dtype=np.int16),
                window_leads=np.array(tuple(int(value) for value in window_leads), dtype=np.int16),
                train_years=np.array(sorted(train_year_set), dtype=np.int16),
                month=np.array(month, dtype=np.int8),
                lead=np.array(int(lead), dtype=np.int16),
                land_count=np.array(land_count, dtype=np.int32),
                rt_tag=np.array(rt_tag),
                mapping_init_indices=mapping_init_indices,
            )
            mappings[(month, int(lead))] = (source_q, target_q)
            sanity_errors.append(train_mean_mae)
            del source_stack, target_stack, mapped, source_rows, target_rows

    print(f"Quantile mapping build check: mean train monthly-mean MAE={np.mean(sanity_errors):.4f} z")
    return mappings


def score_init(
    init_t: int,
    path: Path,
    shared_data: Mapping[str, np.ndarray],
    norm_stats: Mapping[str, object],
    land_mask: np.ndarray,
    months: np.ndarray,
    years: np.ndarray,
    window_leads: Sequence[int],
    mappings: Mapping[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
    q95_z: np.ndarray,
    base_rate: np.ndarray,
) -> Dict[str, object]:
    heat = shared_data["heat_index"]
    native = load_ens_members(path, window_leads)
    mapped_members: List[np.ndarray] = []
    truth_fields: List[np.ndarray] = []
    for lead in window_leads:
        target_t = int(init_t) + int(lead)
        month = int(months[target_t])
        source_q, target_q = mappings[(month, int(lead))]
        mapped_members.append(
            apply_quantile_mapping(native[int(lead)][:, land_mask], source_q, target_q)
        )
        truth_fields.append(
            ee._normalize_field(np.asarray(heat[:, :, target_t]), norm_stats)[land_mask]
        )
    with np.errstate(all="ignore"):
        member_window = np.nanmean(np.stack(mapped_members, axis=1), axis=1).astype(np.float32)
        truth_window_land = np.nanmean(np.stack(truth_fields, axis=0), axis=0).astype(np.float32)
        spread_land = np.nanstd(member_window, axis=0).astype(np.float32)
    center_t = int(init_t) + ee.window_center_lead(window_leads)
    center_month = int(months[center_t])
    threshold_land = q95_z[center_month][land_mask]
    base_land = base_rate[center_month][land_mask]
    raw_land = member_fraction_probability(member_window, threshold_land)
    truth_land = (truth_window_land > threshold_land).astype(np.float32)
    return {
        "init_t": int(init_t),
        "center_t": center_t,
        "month": center_month,
        "year": int(years[center_t]),
        "raw_land": raw_land,
        "truth_land": truth_land,
        "base_land": base_land.astype(np.float32),
        "sigma_land": spread_land,
    }


def reservoir_append(
    rng: np.random.Generator,
    records: List[Dict[str, np.ndarray]],
    record: Mapping[str, object],
    max_samples: int,
) -> None:
    valid = (
        np.isfinite(record["raw_land"])
        & np.isfinite(record["truth_land"])
        & np.isfinite(record["base_land"])
        & np.isfinite(record["sigma_land"])
    )
    indices = np.flatnonzero(valid)
    if indices.size > int(max_samples):
        indices = rng.choice(indices, size=int(max_samples), replace=False)
    records.append({
        "raw": np.asarray(record["raw_land"])[indices].astype(np.float32),
        "truth": np.asarray(record["truth_land"])[indices].astype(np.float32),
        "base": np.asarray(record["base_land"])[indices].astype(np.float32),
        "sigma": np.asarray(record["sigma_land"])[indices].astype(np.float32),
        "year": np.full(indices.size, int(record["year"]), dtype=np.int16),
    })


def concatenate_records(records: Sequence[Mapping[str, np.ndarray]], max_samples: int, seed: int) -> Dict[str, np.ndarray]:
    combined = {
        key: np.concatenate([np.asarray(record[key]) for record in records], axis=0)
        for key in ("raw", "truth", "base", "sigma", "year")
    }
    if combined["truth"].size > int(max_samples):
        rng = np.random.default_rng(int(seed))
        indices = rng.choice(combined["truth"].size, size=int(max_samples), replace=False)
        combined = {key: value[indices] for key, value in combined.items()}
    return combined


def probability_logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float32), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv_fold", type=int, required=True)
    parser.add_argument("--run_name", default=None, help="Default: cvfold{fold}_ens_w{window label}.")
    parser.add_argument("--window_leads", default="15,16,17,18,19,20,21,22,23,24,25,26,27,28")
    parser.add_argument("--input_dir", default="/blue/nessie/mostafarezaali/Teleconnection/ens_reforecast/regridded")
    parser.add_argument("--output_root", default="ens_exceedance_incremental")
    parser.add_argument("--qmap_cache_root", default="/blue/nessie/mostafarezaali/Teleconnection/ens_reforecast/quantile_mapping")
    parser.add_argument("--cv_stride", type=int, default=5)
    parser.add_argument("--quantile_count", type=int, default=51)
    parser.add_argument(
        "--rt_tag",
        default="",
        help="Cycle tag such as rt2024. Keys input, quantile mapping, and default run name.",
    )
    parser.add_argument(
        "--weekdays",
        default=None,
        help="Deprecated. Downloaded S2S hdate initializations are authoritative and are not weekday-filtered.",
    )
    parser.add_argument("--max_calibration_samples", type=int, default=1000000)
    parser.add_argument("--calibration_steps", type=int, default=200)
    parser.add_argument("--calibration_lr", type=float, default=0.1)
    parser.add_argument("--calibration_l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress_every", type=int, default=25)
    args = parser.parse_args()

    window_leads = ee.parse_int_list(args.window_leads)
    rt_tag = normalize_rt_tag(args.rt_tag)
    configure_fold(args.cv_fold, window_leads, args.cv_stride)
    run_suffix = f"_{rt_tag}" if rt_tag else ""
    run_name = args.run_name or (
        f"cvfold{int(args.cv_fold)}_ens_w{ee.lead_list_label(window_leads)}{run_suffix}"
    )
    print(ENS_BENCHMARK_BANNER)
    print(f"Fold={args.cv_fold}; run={run_name}; cycle={rt_tag or 'legacy'}; window={window_leads}")

    shared_data = load_ens_scoring_shared_data(cfm.Config)
    norm_stats = ee.load_norm_stats()
    train_indices, val_indices, test_indices, train_years, val_years, test_years = ee.split_indices_for_config(shared_data)
    if set(train_years) & set(val_years) or set(train_years) & set(test_years) or set(val_years) & set(test_years):
        raise RuntimeError("Fold train/calibration/test years are not disjoint.")
    months, years, _ = ee.month_doy_year_arrays(shared_data["time_values"])
    land_mask = cfm.load_conus_mask(cfm.Config).cpu().numpy() > 0.5
    q95_z, base_rate, stats_path = load_window_stats(window_leads)
    print(f"Using existing PRISM event definition: {stats_path}")
    input_dir = Path(args.input_dir)
    if rt_tag and str(input_dir) == "/blue/nessie/mostafarezaali/Teleconnection/ens_reforecast/regridded":
        input_dir = input_dir.with_name(f"regridded_{rt_tag}")
    files_by_init = load_ingested_files(input_dir, window_leads)
    valid_init_indices = set(int(value) for value in train_indices + val_indices + test_indices)
    files_by_init = {
        init_t: path for init_t, path in files_by_init.items()
        if int(init_t) in valid_init_indices
    }
    if not files_by_init:
        raise RuntimeError(
            "No ENS inits remain after applying HeatCast continuous-season and lead-window guards."
        )
    if args.weekdays:
        print(
            "Ignoring deprecated --weekdays: downloaded S2S hdate initializations are authoritative. "
            "Historical hdates need not share the real-time model-date weekday."
        )
    available_years = {int(years[init_t]) for init_t in files_by_init}
    retained = {
        "train": sorted(available_years & set(int(v) for v in train_years)),
        "val": sorted(available_years & set(int(v) for v in val_years)),
        "test": sorted(available_years & set(int(v) for v in test_years)),
    }
    print(f"Available ENS intersection years: {retained}")
    if not retained["val"] or not retained["test"]:
        raise RuntimeError("ENS coverage has an empty calibration-year or test-year intersection for this fold.")

    mappings = build_or_load_quantile_mapping(
        files_by_init,
        train_years,
        shared_data,
        norm_stats,
        land_mask,
        window_leads,
        Path(args.qmap_cache_root),
        args.quantile_count,
        rt_tag,
    )

    calibration_records: List[Dict[str, np.ndarray]] = []
    calibration_inits = [
        (init_t, path) for init_t, path in files_by_init.items()
        if int(years[init_t]) in set(int(v) for v in val_years)
    ]
    rng = np.random.default_rng(int(args.seed))
    per_init_cap = max(1000, int(math.ceil(args.max_calibration_samples / max(len(calibration_inits), 1))))
    for index, (init_t, path) in enumerate(sorted(calibration_inits)):
        record = score_init(
            init_t, path, shared_data, norm_stats, land_mask, months, years,
            window_leads, mappings, q95_z, base_rate,
        )
        reservoir_append(rng, calibration_records, record, per_init_cap)
        if (index + 1) % max(1, int(args.progress_every)) == 0:
            print(f"  calibration scored {index + 1}/{len(calibration_inits)} ENS inits")
    calibration = concatenate_records(calibration_records, args.max_calibration_samples, args.seed)
    calibrator = ee.fit_model_output_logistic_calibrator(
        calibration["raw"][:, None],
        calibration["truth"],
        ("ens_member_fraction",),
        calibration_split="val",
        steps=args.calibration_steps,
        lr=args.calibration_lr,
        l2=args.calibration_l2,
    )
    calibrated_probability = calibrator.predict_features(calibration["raw"][:, None])
    calibration_features = np.column_stack([
        calibration["raw"],
        probability_logit(calibrated_probability),
        calibration["sigma"],
    ]).astype(np.float32)

    out_dir = Path(args.output_root) / run_name / "test" / f"window_{ee.lead_list_label(window_leads)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ee.save_incremental_calibration_arrays(
        out_dir,
        calibration_features,
        calibration["truth"],
        calibration["year"],
        calibration["base"],
        train_years,
        val_years,
        test_years,
        "val",
    )
    chunk_dir = ee.prepare_incremental_test_chunk_dir(out_dir)
    test_inits = [
        (init_t, path) for init_t, path in files_by_init.items()
        if int(years[init_t]) in set(int(v) for v in test_years)
    ]
    sample_count = 0
    valid_cell_count = 0
    for index, (init_t, path) in enumerate(sorted(test_inits)):
        record = score_init(
            init_t, path, shared_data, norm_stats, land_mask, months, years,
            window_leads, mappings, q95_z, base_rate,
        )
        calibrated = calibrator.predict_features(np.asarray(record["raw_land"])[:, None])
        raw_grid = np.full(land_mask.shape, np.nan, dtype=np.float32)
        calibrated_logit_grid = np.full(land_mask.shape, np.nan, dtype=np.float32)
        truth_grid = np.full(land_mask.shape, np.nan, dtype=np.float32)
        base_grid = np.full(land_mask.shape, np.nan, dtype=np.float32)
        sigma_grid = np.full(land_mask.shape, np.nan, dtype=np.float32)
        raw_grid[land_mask] = record["raw_land"]
        calibrated_logit_grid[land_mask] = probability_logit(calibrated)
        truth_grid[land_mask] = record["truth_land"]
        base_grid[land_mask] = record["base_land"]
        sigma_grid[land_mask] = record["sigma_land"]
        written = ee.save_incremental_test_chunk(
            chunk_dir,
            sample_count,
            int(args.cv_fold),
            int(record["year"]),
            int(record["month"]),
            raw_grid,
            calibrated_logit_grid,
            sigma_grid,
            truth_grid,
            base_grid,
            land_mask,
            init_time_index=int(init_t),
            target_center_time_index=int(record["center_t"]),
        )
        if written > 0:
            sample_count += 1
            valid_cell_count += int(written)
        if (index + 1) % max(1, int(args.progress_every)) == 0:
            print(f"  test chunks saved {index + 1}/{len(test_inits)} ENS inits")

    ee.save_incremental_array_manifest(
        out_dir,
        run_name,
        "window",
        window_leads,
        train_years,
        val_years,
        test_years,
        "val",
        "test",
        sample_count,
        valid_cell_count,
    )
    metadata = {
        "run_name": run_name,
        "rt_tag": rt_tag,
        "fold": int(args.cv_fold),
        "window_leads": list(window_leads),
        "available_years": sorted(available_years),
        "intersection_years": retained,
        "calibrator": {
            "feature_names": list(calibrator.feature_names),
            "mean": calibrator.mean.tolist(),
            "std": calibrator.std.tolist(),
            "coef": calibrator.coef.tolist(),
            "intercept": calibrator.intercept,
            "n_samples": calibrator.n_samples,
            "event_rate": calibrator.event_rate,
        },
        "test_init_count": sample_count,
        "valid_cell_count": valid_cell_count,
    }
    (out_dir / "ens_score_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("ENS fold scoring complete")
    print(f"  output={out_dir}")
    print(f"  test intersection years={retained['test']}")
    print(f"  test inits={sample_count}, valid cells={valid_cell_count}")


if __name__ == "__main__":
    main()
