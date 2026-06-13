#!/usr/bin/env python3
"""Ingest native ECMWF ENS reforecasts and regrid daily T2max to the HeatCast grid."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from ens_common import ENS_BENCHMARK_BANNER, bilinear_regrid_regular
from publication_analysis_utils import conus_lat_lon


BASE_DATE = datetime(1981, 5, 1)


def parse_int_list(text: str) -> Tuple[int, ...]:
    return tuple(int(value.strip()) for value in str(text).split(",") if value.strip())


def load_training_dates(training_path: Path) -> Tuple[np.ndarray, Dict[str, int]]:
    try:
        from netCDF4 import Dataset, num2date
    except ImportError as exc:
        raise RuntimeError("ens_ingest.py requires netCDF4 to resolve the HeatCast time axis.") from exc
    with Dataset(training_path, "r") as dataset:
        time_var = dataset.variables["time"]
        values = np.asarray(time_var[:], dtype=np.float64)
        units = getattr(time_var, "units", "")
        calendar = getattr(time_var, "calendar", "standard")
        if units:
            converted = num2date(values, units=units, calendar=calendar)
            dates = np.array([
                np.datetime64(datetime(int(value.year), int(value.month), int(value.day)), "D")
                for value in converted
            ])
        else:
            dates = np.array([np.datetime64(BASE_DATE + timedelta(days=float(value)), "D") for value in values])
    lookup = {str(value).replace("-", ""): index for index, value in enumerate(dates.astype(str))}
    return dates, lookup


def requested_init_dates(
    dates: np.ndarray,
    weekdays: Sequence[int],
    start_year: int | None,
    end_year: int | None,
) -> List[str]:
    output = []
    for value in dates.astype("datetime64[D]").astype(str):
        dt = datetime.strptime(value, "%Y-%m-%d")
        if dt.month not in (5, 6, 7, 8, 9):
            continue
        if start_year is not None and dt.year < int(start_year):
            continue
        if end_year is not None and dt.year > int(end_year):
            continue
        if dt.weekday() in set(int(day) for day in weekdays):
            output.append(dt.strftime("%Y%m%d"))
    return output


def load_init_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing ENS initialization list: {path}")
    labels = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        label = line.strip()
        if not label:
            continue
        try:
            value = datetime.strptime(label, "%Y%m%d")
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: expected YYYYMMDD, got {label!r}.") from exc
        if value.month not in (5, 6, 7, 8, 9):
            raise ValueError(f"{path}:{line_number}: init {label} is outside MJJAS.")
        if label not in seen:
            labels.append(label)
            seen.add(label)
    if not labels:
        raise RuntimeError(f"No initialization dates found in {path}.")
    return labels


def find_raw_file(raw_dir: Path, init_label: str) -> Path | None:
    candidates = []
    for suffix in (".nc", ".nc4", ".grib", ".grib2", ".grb", ".grb2"):
        candidates.extend(sorted(raw_dir.glob(f"*{init_label}*{suffix}")))
    unique = list(dict.fromkeys(candidates))
    if len(unique) > 1:
        raise RuntimeError(f"Multiple raw ENS files match init {init_label}: {unique}")
    return unique[0] if unique else None


def _coordinate_name(data_array, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in data_array.coords or name in data_array.dims:
            return name
    raise RuntimeError(f"Could not find any of coordinates/dimensions {tuple(candidates)} in {data_array.dims}.")


def _optional_coordinate_name(data_array, candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if name in data_array.coords or name in data_array.dims:
            return name
    return None


def _optional_dimension_name(data_array, candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if name in data_array.dims:
            return name
    return None


def _lead_days(values: np.ndarray, units: str = "") -> np.ndarray:
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.timedelta64):
        hours = values.astype("timedelta64[s]").astype(np.float64) / 3600.0
        return np.ceil(hours / 24.0).astype(np.int16)
    numeric = values.astype(np.float64)
    unit_text = str(units).lower()
    if "hour" in unit_text or np.nanmax(numeric) > 100:
        return np.ceil(numeric / 24.0).astype(np.int16)
    return np.ceil(numeric).astype(np.int16)


def _native_group_components(
    dataset,
    variable: str,
    default_member: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, str, np.ndarray, np.ndarray, np.ndarray]:
    if variable not in dataset:
        raise RuntimeError(f"Missing variable {variable!r}; available={list(dataset.data_vars)}")
    data = dataset[variable].squeeze(drop=True)
    member_candidates = ("number", "member", "realization", "ensemble")
    member_dim = _optional_dimension_name(data, member_candidates)
    lead_dim = _coordinate_name(data, ("step", "lead", "leadtime", "forecast_period", "time"))
    lat_dim = _coordinate_name(data, ("latitude", "lat"))
    lon_dim = _coordinate_name(data, ("longitude", "lon"))
    if member_dim is None:
        scalar_member = _optional_coordinate_name(data, member_candidates)
        if scalar_member is not None:
            values = np.asarray(data[scalar_member].values).reshape(-1)
            if values.size != 1:
                raise RuntimeError(
                    f"Member coordinate {scalar_member!r} is not a dimension and has {values.size} values."
                )
            member_dim = scalar_member
            data = data.expand_dims(member_dim)
        elif default_member is None:
            raise RuntimeError(f"Could not find an ensemble-member dimension in {data.dims}.")
        else:
            member_dim = "number"
            data = data.expand_dims({member_dim: [int(default_member)]})
    data = data.transpose(member_dim, lead_dim, lat_dim, lon_dim)
    return (
        np.asarray(data.values, dtype=np.float32),
        np.asarray(data[lead_dim].values),
        str(data[lead_dim].attrs.get("units", "")),
        np.asarray(data[lat_dim].values, dtype=np.float32),
        np.asarray(data[lon_dim].values, dtype=np.float32),
        np.asarray(data[member_dim].values).reshape(-1),
    )


def _validate_matching_group(
    path: Path,
    reference: Tuple[np.ndarray, np.ndarray, str, np.ndarray, np.ndarray, np.ndarray],
    candidate: Tuple[np.ndarray, np.ndarray, str, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    ref_raw, ref_lead, ref_units, ref_lat, ref_lon, _ = reference
    raw, lead, units, lat, lon, _ = candidate
    if raw.shape[1:] != ref_raw.shape[1:]:
        raise RuntimeError(f"{path}: control/perturbed GRIB shapes differ: {ref_raw.shape} vs {raw.shape}.")
    if not np.array_equal(_lead_days(lead, units), _lead_days(ref_lead, ref_units)):
        raise RuntimeError(f"{path}: control/perturbed GRIB lead coordinates differ.")
    if not np.allclose(lat, ref_lat) or not np.allclose(lon, ref_lon):
        raise RuntimeError(f"{path}: control/perturbed GRIB latitude/longitude coordinates differ.")


def load_native_daily_max(
    path: Path,
    variable: str,
    max_lead: int,
    expected_members: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("ens_ingest.py requires xarray; GRIB files additionally require cfgrib.") from exc
    engine = "cfgrib" if path.suffix.lower() in {".grib", ".grib2", ".grb", ".grb2"} else None
    datasets = []
    try:
        if engine == "cfgrib":
            groups = []
            for data_type, default_member in (("cf", 0), ("pf", None)):
                dataset = xr.open_dataset(
                    path,
                    engine=engine,
                    backend_kwargs={
                        "filter_by_keys": {"dataType": data_type},
                        "indexpath": "",
                    },
                )
                datasets.append(dataset)
                try:
                    groups.append(_native_group_components(dataset, variable, default_member))
                except RuntimeError as exc:
                    raise RuntimeError(f"{path} dataType={data_type}: {exc}") from exc
            _validate_matching_group(path, groups[0], groups[1])
            raw = np.concatenate([group[0] for group in groups], axis=0)
            member_values = np.concatenate([group[5] for group in groups])
            lead_values, lead_units, source_lat, source_lon = groups[0][1:5]
        else:
            dataset = xr.open_dataset(path, engine=engine)
            datasets.append(dataset)
            raw, lead_values, lead_units, source_lat, source_lon, member_values = _native_group_components(
                dataset,
                variable,
            )
        if member_values.size != int(expected_members):
            raise RuntimeError(
                f"{path}: expected {expected_members} members, found {member_values.size}."
            )
        if np.unique(member_values).size != member_values.size:
            raise RuntimeError(f"{path}: duplicate ensemble member labels: {member_values.tolist()}.")
        lead_days = _lead_days(lead_values, lead_units)
        daily = np.full((raw.shape[0], int(max_lead), raw.shape[2], raw.shape[3]), np.nan, dtype=np.float32)
        for lead in range(1, int(max_lead) + 1):
            indices = np.where(lead_days == lead)[0]
            if indices.size == 0:
                raise RuntimeError(f"{path}: no {variable} values found for lead day {lead}.")
            with np.errstate(all="ignore"):
                daily[:, lead - 1] = np.nanmax(raw[:, indices], axis=1)
        return (
            daily,
            source_lat,
            source_lon,
            member_values,
        )
    finally:
        for dataset in datasets:
            dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", default="/blue/nessie/mostafarezaali/Teleconnection/ens_reforecast/raw")
    parser.add_argument("--output_dir", default="/blue/nessie/mostafarezaali/Teleconnection/ens_reforecast/regridded")
    parser.add_argument("--training_data_path", default="/blue/nessie/mostafarezaali/Teleconnection/VDM_Training_Data_Extended_v2.nc")
    parser.add_argument("--variable", default="mx2t6")
    parser.add_argument("--weekdays", default="0,3", help="Python weekdays; default Monday,Thursday.")
    parser.add_argument("--start_year", type=int, default=None)
    parser.add_argument("--end_year", type=int, default=None)
    parser.add_argument("--max_lead", type=int, default=28)
    parser.add_argument("--expected_members", type=int, default=11)
    parser.add_argument(
        "--init_list",
        default=None,
        help="Optional YYYYMMDD list from download_ecmwf_s2s.py; uses only available S2S inits.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print(ENS_BENCHMARK_BANNER)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dates, date_lookup = load_training_dates(Path(args.training_data_path))
    import cfm_mesh_train as cfm

    cfm.Config.TRAINING_DATA_PATH = str(args.training_data_path)
    land_mask = cfm.load_conus_mask(cfm.Config).cpu().numpy() > 0.5
    if args.init_list:
        init_labels = load_init_list(Path(args.init_list))
        outside_axis = [label for label in init_labels if label not in date_lookup]
        if outside_axis:
            print(
                f"Skipping {len(outside_axis)} downloaded S2S inits outside the HeatCast time axis."
            )
            init_labels = [label for label in init_labels if label in date_lookup]
        if not init_labels:
            raise RuntimeError("No init-list dates overlap the HeatCast MJJAS time axis.")
    else:
        init_labels = requested_init_dates(
            dates,
            parse_int_list(args.weekdays),
            args.start_year,
            args.end_year,
        )
    raw_files = {label: find_raw_file(raw_dir, label) for label in init_labels}
    missing = [label for label, path in raw_files.items() if path is None]
    if missing:
        preview = "\n".join(f"  init_{label}: no GRIB/NetCDF file under {raw_dir}" for label in missing[:100])
        raise FileNotFoundError(
            f"Missing {len(missing)} requested ENS initialization files. Download them before ingestion:\n{preview}"
        )

    _, _, target_lat, target_lon = conus_lat_lon((621, 1405))
    completed = 0
    for label in init_labels:
        output_path = output_dir / f"init_{label}.npz"
        if output_path.exists() and not args.overwrite:
            continue
        native, source_lat, source_lon, members = load_native_daily_max(
            raw_files[label],
            args.variable,
            args.max_lead,
            args.expected_members,
        )
        regridded = bilinear_regrid_regular(native, source_lat, source_lon, target_lat, target_lon)
        init_time_index = date_lookup.get(label)
        if init_time_index is None:
            print(f"Skipping init {label}: date is outside the HeatCast MJJAS time axis.")
            continue
        regridded[:, :, ~land_mask] = np.nan
        np.savez_compressed(
            output_path,
            t2max=regridded.astype(np.float32),
            leads=np.arange(1, int(args.max_lead) + 1, dtype=np.int16),
            members=np.asarray(members),
            init_date=np.array(label),
            init_time_index=np.array(init_time_index, dtype=np.int32),
            variable=np.array(str(args.variable)),
        )
        completed += 1
        if completed % 25 == 0:
            print(f"  regridded {completed}/{len(init_labels)} initialization files")
    print(f"ENS ingestion complete: wrote {completed} files to {output_dir}")


if __name__ == "__main__":
    main()
