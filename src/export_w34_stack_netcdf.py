#!/usr/bin/env python3
"""Export W34 HeatCast+ENS stack probabilities and truth to MATLAB-readable NetCDF.

This script reads the saved fold-safe incremental arrays, reconstructs the
cross-fitted HeatCast+ENS stacker used in the paper analysis, aligns HeatCast
and ENS by init_time_index, sorts samples chronologically, and writes one
NetCDF file with dimensions:

  (y, x, time)

The exported product is the paired W34 HeatCast/ENS held-out test intersection
because the stack requires ENS chunks. It does not rerun the neural network and
does not fabricate train/validation stack probabilities when saved chunks are
not available. All map variables are written as 3-D matrices with time as the
third dimension so MATLAB reads them as [y, x, time].
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

import exceedance_eval as ee
from ens_compare import chunk_map, resolve_ens_run_groups
from ens_heatcast_stack_opportunity import (
    STACK_FEATURE_NAMES,
    build_reservoir_for_fold,
    fit_opportunity_boundaries,
    fit_stacker_for_excluded_fold,
    fit_heatcast_c,
    paired_chunk,
)
from stitch_exceedance_folds import load_fold_inputs


BASE_DATE = datetime(1981, 5, 1)
WINDOW_LEADS_DEFAULT = "15,16,17,18,19,20,21,22,23,24,25,26,27,28"
HEATCAST_RUNS_DEFAULT = (
    "cvfold0_w34_dist_v1,cvfold1_w34_dist_v1,cvfold2_w34_dist_v1,"
    "cvfold3_w34_dist_v1,cvfold4_w34_dist_v1"
)
ENS_RUNS_DEFAULT = "cvfold{F}_ens_w34,cvfold{F}_ens_w34_rt2024"
PRISM_SHAPE = (621, 1405)
PRISM_CELL_DEG = 1.0 / 24.0
PRISM_LON_LEFT_CENTER = -125.0
PRISM_LAT_BOTTOM_CENTER = 24.0833333333333


def parse_csv_list(text: str) -> Tuple[str, ...]:
    return tuple(value.strip() for value in str(text).split(",") if value.strip())


def yyyymmdd_from_time_value(value: float) -> int:
    dt = BASE_DATE + timedelta(days=float(value))
    return dt.year * 10000 + dt.month * 100 + dt.day


def load_time_values(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing time axis file: {path}. Expected data_cache/time_values.npy "
            "from the HeatCast preprocessing cache."
        )
    values = np.asarray(np.load(path), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise RuntimeError(f"Invalid time axis in {path}: shape={values.shape}.")
    return values


def prism_pixel_center_lat_lon(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return PRISM 4 km pixel-center coordinates for the standard CONUS grid."""
    h, w = shape
    if (h, w) != PRISM_SHAPE:
        raise RuntimeError(
            f"PRISM pixel-center fallback is defined only for shape={PRISM_SHAPE}, got {(h, w)}. "
            "Add real coordinate variables to the source NetCDF for this grid."
        )
    lon_1d = PRISM_LON_LEFT_CENTER + np.arange(w, dtype=np.float32) * np.float32(PRISM_CELL_DEG)
    lat_1d_south_to_north = PRISM_LAT_BOTTOM_CENTER + np.arange(h, dtype=np.float32) * np.float32(PRISM_CELL_DEG)
    lat_1d = lat_1d_south_to_north[::-1].astype(np.float32)
    lon2d, lat2d = np.meshgrid(lon_1d.astype(np.float32), lat_1d)
    return lat_1d, lon_1d.astype(np.float32), lat2d.astype(np.float32), lon2d.astype(np.float32)


def _coord_from_source(path: Path, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Read lat/lon coordinates from the target source file when they are present."""
    if not path.exists():
        return None
    try:
        from netCDF4 import Dataset
    except Exception:
        return None
    lat_names = ("lat", "latitude", "y")
    lon_names = ("lon", "longitude", "x")
    try:
        with Dataset(path, "r") as ds:
            lat = next((np.asarray(ds.variables[name][:], dtype=np.float32) for name in lat_names if name in ds.variables), None)
            lon = next((np.asarray(ds.variables[name][:], dtype=np.float32) for name in lon_names if name in ds.variables), None)
    except Exception:
        return None
    if lat is None or lon is None:
        return None
    if lat.ndim == 1 and lon.ndim == 1:
        if lat.size == shape[0] and lon.size == shape[1]:
            lon2d, lat2d = np.meshgrid(lon, lat)
            return lat.astype(np.float32), lon.astype(np.float32), lat2d.astype(np.float32), lon2d.astype(np.float32)
        if lat.size == shape[1] and lon.size == shape[0]:
            lon2d, lat2d = np.meshgrid(lat, lon)
            return lon.astype(np.float32), lat.astype(np.float32), lat2d.astype(np.float32), lon2d.astype(np.float32)
    if lat.ndim == 2 and lon.ndim == 2:
        if lat.shape == shape and lon.shape == shape:
            return lat[:, 0].astype(np.float32), lon[0, :].astype(np.float32), lat.astype(np.float32), lon.astype(np.float32)
        if lat.T.shape == shape and lon.T.shape == shape:
            lat2d = lat.T.astype(np.float32)
            lon2d = lon.T.astype(np.float32)
            return lat2d[:, 0].astype(np.float32), lon2d[0, :].astype(np.float32), lat2d, lon2d
    return None


def target_lat_lon(config, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    coords = _coord_from_source(Path(config.TRAINING_DATA_PATH), shape)
    if coords is not None:
        return (*coords, f"source:{config.TRAINING_DATA_PATH}")
    lat_1d, lon_1d, lat2d, lon2d = prism_pixel_center_lat_lon(shape)
    return lat_1d, lon_1d, lat2d, lon2d, "prism_4km_pixel_center_fallback"


def load_land_metadata() -> Dict[str, np.ndarray]:
    import cfm_mesh_train as cfm

    mask = np.asarray(cfm.load_conus_mask(cfm.Config).cpu().numpy() > 0.5, dtype=bool)
    lat_1d, lon_1d, lat2d, lon2d, coord_source = target_lat_lon(cfm.Config, mask.shape)
    row2d, col2d = np.indices(mask.shape)
    return {
        "mask": mask,
        "coord_source": coord_source,
        "lat_1d": np.asarray(lat_1d, dtype=np.float32),
        "lon_1d": np.asarray(lon_1d, dtype=np.float32),
        "lat_land": np.asarray(lat2d[mask], dtype=np.float32),
        "lon_land": np.asarray(lon2d[mask], dtype=np.float32),
        "row_land": np.asarray(row2d[mask], dtype=np.int32),
        "col_land": np.asarray(col2d[mask], dtype=np.int32),
    }


def load_fold_norm_scales(
    fold_inputs: Mapping[int, Mapping[str, object]],
    window_leads: Sequence[int],
) -> Dict[int, Tuple[float, float]]:
    """Return fold-specific target normalization mean/std in degrees C."""
    import cfm_mesh_train as cfm

    original = {
        "CV_VAL_OFFSETS": tuple(cfm.Config.CV_VAL_OFFSETS),
        "CV_TEST_OFFSETS": tuple(cfm.Config.CV_TEST_OFFSETS),
        "MULTI_LEAD_TUBE": bool(getattr(cfm.Config, "MULTI_LEAD_TUBE", False)),
        "PREDICTION_LEADS": tuple(getattr(cfm.Config, "PREDICTION_LEADS", ())),
        "LEAD_TIME": int(getattr(cfm.Config, "LEAD_TIME", 15)),
    }
    leads = tuple(int(lead) for lead in window_leads)
    if not leads:
        raise ValueError("window_leads must not be empty.")
    out: Dict[int, Tuple[float, float]] = {}
    try:
        for fold in sorted(fold_inputs):
            cfm.Config.CV_TEST_OFFSETS = (int(fold),)
            cfm.Config.CV_VAL_OFFSETS = ((int(fold) + 1) % int(cfm.Config.CV_STRIDE),)
            cfm.Config.MULTI_LEAD_TUBE = True
            cfm.Config.PREDICTION_LEADS = leads
            cfm.Config.LEAD_TIME = original["LEAD_TIME"] if original["LEAD_TIME"] in leads else leads[0]
            stats_path = cfm.get_norm_stats_path(cfm.Config)
            if not Path(stats_path).exists():
                raise FileNotFoundError(f"Missing fold normalization stats for degree-C export: {stats_path}")
            with np.load(stats_path, allow_pickle=False) as stats:
                if "hi_mean" not in stats or "hi_std" not in stats:
                    raise RuntimeError(f"Normalization stats lack hi_mean/hi_std: {stats_path}")
                hi_mean = float(stats["hi_mean"])
                hi_std = float(stats["hi_std"])
            if not np.isfinite(hi_mean) or not np.isfinite(hi_std) or hi_std <= 0:
                raise RuntimeError(
                    f"Invalid normalization scale in {stats_path}: hi_mean={hi_mean}, hi_std={hi_std}"
                )
            out[int(fold)] = (hi_mean, hi_std)
    finally:
        for key, value in original.items():
            setattr(cfm.Config, key, value)
    return out


def finite_or_fill(values: np.ndarray, fill_value: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return np.where(np.isfinite(arr), arr, np.float32(fill_value)).astype(np.float32)


def land_vector_to_grid(values: np.ndarray, land_mask: np.ndarray, fill_value: float) -> np.ndarray:
    grid = np.full(land_mask.shape, np.float32(fill_value), dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size != int(np.sum(land_mask)):
        raise RuntimeError(f"Land vector length {arr.size} does not match land mask count {int(np.sum(land_mask))}.")
    grid[land_mask] = finite_or_fill(arr, fill_value)
    return grid


def assert_continuous_heatcast_chunks(heat_name: str, chunks: Sequence[Path]) -> None:
    for path in chunks:
        with np.load(path, allow_pickle=False) as data:
            if "mu_z" not in data.files or "truth_z" not in data.files:
                init_idx = int(data["init_time_index"]) if "init_time_index" in data.files else -1
                target_idx = int(data["target_center_time_index"]) if "target_center_time_index" in data.files else -1
                raise RuntimeError(
                    f"{heat_name}: stale incremental chunk lacks mu_z/truth_z: {path} "
                    f"(init_time_index={init_idx}, target_center_time_index={target_idx}). "
                    "Rerun slurm/submit_w34_eval_stitch.slurm after pulling the schema-v2 code."
                )


def build_fold_inputs(args: argparse.Namespace, window_leads: Sequence[int]) -> Dict[int, Dict[str, object]]:
    heatcast_runs = parse_csv_list(args.heatcast_runs)
    ens_runs = parse_csv_list(args.ens_runs)
    ens_groups = resolve_ens_run_groups(ens_runs, heatcast_runs, Path(args.ens_root), window_leads)
    fold_inputs: Dict[int, Dict[str, object]] = {}
    for heat_name in heatcast_runs:
        manifest, calibration, chunks = load_fold_inputs(Path(args.heatcast_root), heat_name, window_leads)
        if bool(args.require_continuous_fields):
            assert_continuous_heatcast_chunks(heat_name, chunks)
        fold = int(manifest["source_fold"])
        if fold in fold_inputs:
            raise RuntimeError(f"Duplicate HeatCast source_fold={fold}.")
        ens_sources = []
        for ens_name in ens_groups[fold]:
            ens_manifest, _, ens_chunks = load_fold_inputs(Path(args.ens_root), ens_name, window_leads)
            if int(ens_manifest["source_fold"]) != fold:
                raise RuntimeError(f"Fold mismatch: HeatCast={fold}, ENS={ens_manifest['source_fold']}.")
            ens_sources.append((ens_name, ens_manifest, chunk_map(ens_chunks)))
        heat_map = chunk_map(chunks)
        common = tuple(sorted(set(heat_map) & set().union(*(set(source[2]) for source in ens_sources))))
        if not common:
            raise RuntimeError(f"Fold {fold}: no common HeatCast/ENS init_time_index values.")
        fit_args = SimpleNamespace(
            calibration_steps=int(args.calibration_steps),
            calibration_lr=float(args.calibration_lr),
            calibration_l2=float(args.calibration_l2),
        )
        heat_c = fit_heatcast_c(calibration, fit_args)
        fold_inputs[fold] = {
            "heat_name": heat_name,
            "manifest": manifest,
            "calibration": calibration,
            "heat_map": heat_map,
            "ens_sources": ens_sources,
            "common": common,
            "heat_c": heat_c,
            "boundaries": fit_opportunity_boundaries(calibration, heat_c),
        }
        print(
            f"Fold {fold}: common paired samples={len(common)}, "
            f"test_years={sorted(manifest['test_years'])}"
        )
    return fold_inputs


def fit_crossfit_stackers(args: argparse.Namespace, fold_inputs: Mapping[int, Mapping[str, object]]):
    reservoir_x: Dict[int, np.ndarray] = {}
    reservoir_y: Dict[int, np.ndarray] = {}
    fold_workers = max(1, min(int(args.fold_workers), len(fold_inputs)))
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
            reservoir_x[int(fold)] = x
            reservoir_y[int(fold)] = y
    fit_args = SimpleNamespace(
        calibration_steps=int(args.calibration_steps),
        calibration_lr=float(args.calibration_lr),
        calibration_l2=float(args.calibration_l2),
    )
    stackers = {
        fold: fit_stacker_for_excluded_fold(fold, reservoir_x, reservoir_y, fit_args)
        for fold in sorted(fold_inputs)
    }
    for fold, stacker in sorted(stackers.items()):
        print(f"Fold {fold}: stacker fit excluding scored fold, n={stacker.n_samples}")
    return stackers


def collect_records(
    fold_inputs: Mapping[int, Mapping[str, object]],
    stackers: Mapping[int, object],
    time_values: np.ndarray,
    sort_by: str,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for fold in sorted(fold_inputs):
        info = fold_inputs[fold]
        for init_t in info["common"]:
            init_idx = int(init_t)
            data = paired_chunk(
                fold,
                init_idx,
                info["heat_map"][init_idx],
                info["ens_sources"],
                info["heat_c"],
            )
            target_idx = int(data["target_center_time_index"])
            if init_idx >= time_values.size or target_idx >= time_values.size:
                raise RuntimeError(
                    f"Time index outside time axis: init={init_idx}, target={target_idx}, "
                    f"time_values={time_values.size}"
                )
            records.append({
                "fold": int(fold),
                "init_time_index": init_idx,
                "target_center_time_index": target_idx,
                "sort_key": (target_idx, init_idx) if sort_by == "target" else (init_idx, target_idx),
                "data": data,
                "stacker": stackers[fold],
            })
    records.sort(key=lambda row: row["sort_key"])
    return records


def create_var(ds, name: str, dtype, dimensions: Tuple[str, ...], **kwargs):
    fill_value = kwargs.pop("fill_value", None)
    zlib = kwargs.pop("zlib", True)
    complevel = kwargs.pop("complevel", 4)
    chunksizes = kwargs.pop("chunksizes", None)
    if dtype in ("S1", str):
        zlib = False
    var = ds.createVariable(
        name,
        dtype,
        dimensions,
        zlib=zlib,
        complevel=complevel if zlib else 0,
        chunksizes=chunksizes,
        fill_value=fill_value,
    )
    for key, value in kwargs.items():
        setattr(var, key, value)
    return var


def write_netcdf(
    path: Path,
    records: Sequence[Mapping[str, object]],
    fold_inputs: Mapping[int, Mapping[str, object]],
    time_values: np.ndarray,
    land_meta: Mapping[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    try:
        from netCDF4 import Dataset
    except Exception as exc:
        raise RuntimeError("netCDF4 is required for compressed MATLAB-readable export.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise RuntimeError("No records to export.")
    land_count = int(np.asarray(records[0]["data"]["truth"]).size)
    if land_count != int(np.asarray(land_meta["lat_land"]).size):
        raise RuntimeError(
            f"Chunk land count {land_count} does not match mask land count {np.asarray(land_meta['lat_land']).size}."
        )
    sample_count = len(records)
    fill = np.float32(args.fill_value)
    chunk_samples = max(1, min(int(args.chunk_samples), sample_count))
    variables = set(parse_csv_list(args.variables))
    fold_norm_scales = load_fold_norm_scales(fold_inputs, ee.parse_int_list(args.window_leads))

    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", sample_count)
        ds.createDimension("land_cell", land_count)
        ds.createDimension("y", int(np.asarray(land_meta["mask"]).shape[0]))
        ds.createDimension("x", int(np.asarray(land_meta["mask"]).shape[1]))
        ds.createDimension("window_lead", len(ee.parse_int_list(args.window_leads)))

        ds.title = "HeatCast W34 HeatCast+ENS stack export"
        ds.product = "cross_fitted_heatcast_ens_stack_probabilities"
        ds.scope = "paired_heatcast_ens_heldout_test_intersection"
        ds.note = (
            "The HeatCast+ENS stack is available only where saved HeatCast and ENS chunks "
            "share init_time_index. split_code is 2 for held-out test samples in this export. "
            "All forecast/truth variables use dimensions (y, x, time) for MATLAB plotting. "
            "model_output_3d and ground_truth_3d are continuous W34 mean z-score fields "
            "when source chunks contain mu_z/truth_z; otherwise they are NaN and the "
            "probability/binary exceedance variables remain available."
        )
        ds.window_leads = args.window_leads
        ds.sort_by = args.sort_by
        ds.heatcast_runs = args.heatcast_runs
        ds.ens_runs = args.ens_runs
        ds.coordinate_source = str(land_meta.get("coord_source", "unknown"))
        ds.stack_feature_names = ",".join(STACK_FEATURE_NAMES)
        ds.fold_manifests_json = json.dumps({
            int(fold): {
                "run_name": info["manifest"]["run_name"],
                "train_years": sorted(int(v) for v in info["manifest"]["train_years"]),
                "calibration_years": sorted(int(v) for v in info["manifest"]["calibration_years"]),
                "test_years": sorted(int(v) for v in info["manifest"]["test_years"]),
            }
            for fold, info in sorted(fold_inputs.items())
        })
        ds.created_utc = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        create_var(ds, "window_lead", "i4", ("window_lead",), long_name="forecast lead offsets in days")[:] = np.asarray(ee.parse_int_list(args.window_leads), dtype=np.int32)
        lon2d, lat2d = np.meshgrid(np.asarray(land_meta["lon_1d"], dtype=np.float32), np.asarray(land_meta["lat_1d"], dtype=np.float32))
        create_var(ds, "lat", "f4", ("y", "x"), units="degrees_north")[:] = lat2d
        create_var(ds, "lon", "f4", ("y", "x"), units="degrees_east")[:] = lon2d
        create_var(ds, "lat_land", "f4", ("land_cell",), units="degrees_north")[:] = land_meta["lat_land"]
        create_var(ds, "lon_land", "f4", ("land_cell",), units="degrees_east")[:] = land_meta["lon_land"]
        create_var(ds, "row_land", "i4", ("land_cell",), long_name="row index in full CONUS grid")[:] = land_meta["row_land"]
        create_var(ds, "col_land", "i4", ("land_cell",), long_name="column index in full CONUS grid")[:] = land_meta["col_land"]
        create_var(ds, "lat_1d", "f4", ("y",), units="degrees_north")[:] = land_meta["lat_1d"]
        create_var(ds, "lon_1d", "f4", ("x",), units="degrees_east")[:] = land_meta["lon_1d"]
        create_var(ds, "land_mask", "i1", ("y", "x"), zlib=True, long_name="CONUS land mask")[:] = np.asarray(land_meta["mask"], dtype=np.int8)

        time_var = create_var(
            ds,
            "time",
            "f8",
            ("time",),
            zlib=False,
            units="days since 1981-05-01 00:00:00",
            calendar="standard",
            long_name="target window center date",
        )
        init_time_var = create_var(
            ds,
            "init_time",
            "f8",
            ("time",),
            zlib=False,
            units="days since 1981-05-01 00:00:00",
            calendar="standard",
            long_name="forecast initialization date",
        )
        target_index_var = create_var(ds, "target_center_time_index", "i4", ("time",), zlib=False)
        init_index_var = create_var(ds, "init_time_index", "i4", ("time",), zlib=False)
        target_date_var = create_var(ds, "target_date_yyyymmdd", "i4", ("time",), zlib=False)
        init_date_var = create_var(ds, "init_date_yyyymmdd", "i4", ("time",), zlib=False)
        fold_var = create_var(ds, "source_fold", "i2", ("time",), zlib=False)
        split_var = create_var(ds, "split_code", "i1", ("time",), zlib=False)
        split_var.flag_values = "0,1,2"
        split_var.flag_meanings = "train validation test"
        year_var = create_var(ds, "year", "i2", ("time",), zlib=False)
        month_var = create_var(ds, "month", "i1", ("time",), zlib=False)
        z_mean_var = create_var(
            ds,
            "z_score_mean_c",
            "f4",
            ("time",),
            zlib=False,
            units="degree_C",
            long_name="Fold-specific target normalization mean used for z-score fields",
        )
        z_std_var = create_var(
            ds,
            "z_score_std_c",
            "f4",
            ("time",),
            zlib=False,
            units="degree_C",
            long_name="Fold-specific target normalization standard deviation used to convert z-score errors to degrees C",
        )

        map_chunks = (
            int(np.asarray(land_meta["mask"]).shape[0]),
            min(256, int(np.asarray(land_meta["mask"]).shape[1])),
            chunk_samples,
        )
        out_vars = {}
        variable_specs = {
            "model_output": ("model_output_3d", "Continuous HeatCast W34 mean prediction in z-score units", "z-score"),
            "ground_truth": ("ground_truth_3d", "Continuous observed W34 mean truth in z-score units", "z-score"),
            "truth": ("truth_exceedance", "Observed W34 window exceedance label", "1"),
            "stack": ("prob_heatcast_ens_stack", "Cross-fitted HeatCast+ENS stack exceedance probability", "1"),
            "heatcast": ("prob_heatcast_C", "HeatCast-C calibrated exceedance probability", "1"),
            "ens": ("prob_ens_calibrated", "Calibrated ENS exceedance probability", "1"),
            "ens_raw": ("prob_ens_raw_fraction", "Raw ENS member exceedance fraction", "1"),
            "base": ("prob_climatology", "Fold-safe monthly climatological base-rate probability", "1"),
            "sigma": ("heatcast_model_sigma", "HeatCast distributional model sigma on W34 window", "z-score"),
            "init_margin": ("heatcast_init_margin", "Initial-state exceedance margin feature", "sigma units"),
            "forecast_margin": ("heatcast_forecast_margin", "HeatCast forecast exceedance margin feature", "sigma units"),
        }
        for key, (var_name, long_name, units) in variable_specs.items():
            if key not in variables:
                continue
            out_vars[key] = create_var(
                ds,
                var_name,
                "f4",
                ("y", "x", "time"),
                fill_value=fill,
                zlib=True,
                complevel=int(args.compression_level),
                chunksizes=map_chunks,
                long_name=long_name,
                units=units,
            )

        for idx, record in enumerate(records):
            data = record["data"]
            if bool(args.require_continuous_fields) and ("mu_z" not in data or "truth_z" not in data):
                raise RuntimeError(
                    f"Saved incremental chunk lacks continuous mu_z/truth_z fields "
                    f"(fold={record['fold']}, init_time_index={record['init_time_index']}, "
                    f"target_center_time_index={record['target_center_time_index']}). "
                    "Rerun src/exceedance_eval.py with --incremental_skill_diagnostic --save_incremental_arrays "
                    "after pulling the schema>=2 code."
                )
            stack_prob = record["stacker"].predict_features(np.asarray(data["features"], dtype=np.float32))
            init_idx = int(record["init_time_index"])
            target_idx = int(record["target_center_time_index"])
            init_days = float(time_values[init_idx])
            target_days = float(time_values[target_idx])
            time_var[idx] = target_days
            init_time_var[idx] = init_days
            target_index_var[idx] = target_idx
            init_index_var[idx] = init_idx
            target_date_var[idx] = yyyymmdd_from_time_value(target_days)
            init_date_var[idx] = yyyymmdd_from_time_value(init_days)
            fold_var[idx] = int(record["fold"])
            split_var[idx] = 2
            year_var[idx] = int(data["year"])
            month_var[idx] = int(data["month"])
            z_mean_var[idx], z_std_var[idx] = fold_norm_scales[int(record["fold"])]

            arrays = {
                "model_output": np.asarray(
                    data["mu_z"] if "mu_z" in data else np.full_like(data["truth"], np.nan, dtype=np.float32),
                    dtype=np.float32,
                ),
                "ground_truth": np.asarray(
                    data["truth_z"] if "truth_z" in data else np.full_like(data["truth"], np.nan, dtype=np.float32),
                    dtype=np.float32,
                ),
                "truth": np.asarray(data["truth"], dtype=np.float32),
                "stack": np.asarray(stack_prob, dtype=np.float32),
                "heatcast": np.asarray(data["heatcast_C"], dtype=np.float32),
                "ens": np.asarray(data["ens_calibrated"], dtype=np.float32),
                "ens_raw": np.asarray(data["ens_raw"], dtype=np.float32),
                "base": np.asarray(data["base"], dtype=np.float32),
                "sigma": np.asarray(data["sigma"], dtype=np.float32),
                "init_margin": np.asarray(data["features"][:, 3], dtype=np.float32),
                "forecast_margin": np.asarray(data["features"][:, 4], dtype=np.float32),
            }
            for key, var in out_vars.items():
                var[:, :, idx] = land_vector_to_grid(arrays[key], np.asarray(land_meta["mask"], dtype=bool), fill)
            if (idx + 1) % max(1, int(args.progress_every)) == 0:
                print(f"  wrote {idx + 1}/{sample_count} samples")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="matlab_exports/w34_heatcast_ens_stack.nc")
    parser.add_argument("--heatcast_root", default="exceedance_eval_incremental")
    parser.add_argument("--ens_root", default="ens_exceedance_incremental")
    parser.add_argument("--heatcast_runs", default=HEATCAST_RUNS_DEFAULT)
    parser.add_argument("--ens_runs", default=ENS_RUNS_DEFAULT)
    parser.add_argument("--window_leads", default=WINDOW_LEADS_DEFAULT)
    parser.add_argument("--time_values_path", default="data_cache/time_values.npy")
    parser.add_argument("--sort_by", choices=("target", "init"), default="target")
    parser.add_argument(
        "--variables",
        default="model_output,ground_truth,truth,stack,heatcast,ens,ens_raw,base,sigma,init_margin,forecast_margin",
        help="Comma-separated export fields. Available: model_output,ground_truth,truth,stack,heatcast,ens,ens_raw,base,sigma,init_margin,forecast_margin.",
    )
    parser.add_argument("--max_stack_samples_per_fold", type=int, default=300000)
    parser.add_argument("--fold_workers", type=int, default=5)
    parser.add_argument("--calibration_steps", type=int, default=200)
    parser.add_argument("--calibration_lr", type=float, default=0.1)
    parser.add_argument("--calibration_l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress_every", type=int, default=50)
    parser.add_argument("--chunk_samples", type=int, default=1)
    parser.add_argument("--compression_level", type=int, default=4)
    parser.add_argument("--fill_value", type=float, default=float("nan"))
    parser.add_argument(
        "--require_continuous_fields",
        action="store_true",
        help="Fail if saved chunks do not contain continuous mu_z/truth_z fields.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    window_leads = ee.parse_int_list(args.window_leads)
    time_values = load_time_values(Path(args.time_values_path))
    land_meta = load_land_metadata()
    fold_inputs = build_fold_inputs(args, window_leads)
    stackers = fit_crossfit_stackers(args, fold_inputs)
    records = collect_records(fold_inputs, stackers, time_values, args.sort_by)
    years = sorted({int(record["data"]["year"]) for record in records})
    print(
        f"Exporting {len(records)} paired W34 samples, years={years[0]}-{years[-1]} "
        f"({len(years)} years), land_cells={len(land_meta['lat_land'])}"
    )
    write_netcdf(Path(args.output), records, fold_inputs, time_values, land_meta, args)
    print(f"NetCDF export complete: {args.output}")
    print(
        "MATLAB variables are y x time cubes: prob_heatcast_ens_stack, "
        "model_output_3d, ground_truth_3d, truth_exceedance, prob_heatcast_C, prob_ens_calibrated. "
        "Use lat/lon as y x grids and time as the third dimension coordinate."
    )


if __name__ == "__main__":
    main()
