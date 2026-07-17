#!/usr/bin/env python3
"""Build cached MJO, ENSO, and fold-safe antecedent soil-moisture driver tables."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple
import xml.etree.ElementTree as ET
import zipfile

import numpy as np


BASE_DATE = datetime(1981, 5, 1)
MJJAS_MONTHS = (5, 6, 7, 8, 9)


def sanitize_index_name(name: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name).strip())
    clean = "_".join(part for part in clean.split("_") if part)
    if not clean:
        raise ValueError(f"Invalid teleconnection index name: {name!r}")
    return clean


def parse_rmm_file(path: Path) -> Dict[int, Tuple[int, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing BOM RMM file: {path}")
    output: Dict[int, Tuple[int, float]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            year, month, day = (int(parts[0]), int(parts[1]), int(parts[2]))
            # BOM's documented columns are:
            # year month day RMM1 RMM2 phase amplitude [optional source/method].
            # Recent files append a method label after amplitude, so phase and
            # amplitude must be read by position rather than from the line end.
            phase = int(float(parts[5]))
            amplitude = float(parts[6])
            key = int(f"{year:04d}{month:02d}{day:02d}")
        except ValueError:
            continue
        output[key] = (phase, amplitude)
    if not output:
        raise RuntimeError(f"No RMM observations parsed from {path}")
    return output


def parse_nino34_file(path: Path) -> Dict[Tuple[int, int], float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing NOAA PSL Nino3.4 file: {path}")
    output: Dict[Tuple[int, int], float] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.replace(",", " ").split()
        if not parts:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        if len(parts) >= 13:
            try:
                for month, value in enumerate(parts[1:13], start=1):
                    output[(year, month)] = float(value)
                continue
            except ValueError:
                pass
        if len(parts) >= 3:
            try:
                month = int(parts[1])
                # NOAA PSL long-form tables commonly store the anomaly in the
                # final column after total and climatology-adjusted values.
                output[(year, month)] = float(parts[-1])
            except ValueError:
                continue
    if not output:
        raise RuntimeError(f"No monthly Nino3.4 observations parsed from {path}")
    return output


def parse_monthly_index_file(path: Path) -> Dict[Tuple[int, int], float]:
    """Parse common monthly teleconnection-index text tables.

    Supported formats:
    - wide NOAA/PSL style: year jan feb ... dec
    - long style: year month value [... optional columns], using the final value
    Lines beginning with # are ignored.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing monthly teleconnection file: {path}")
    output: Dict[Tuple[int, int], float] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.replace(",", " ").split()
        try:
            year = int(float(parts[0]))
        except Exception:
            continue
        if len(parts) >= 13:
            try:
                for month, value in enumerate(parts[1:13], start=1):
                    output[(year, month)] = float(value)
                continue
            except ValueError:
                pass
        if len(parts) >= 3:
            try:
                month = int(float(parts[1]))
                if 1 <= month <= 12:
                    output[(year, month)] = float(parts[-1])
            except ValueError:
                continue
    if not output:
        raise RuntimeError(f"No monthly observations parsed from {path}")
    return output


def parse_named_index_paths(text: str) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for item in str(text).split(","):
        spec = item.strip()
        if not spec:
            continue
        if "=" not in spec:
            raise ValueError(
                "Teleconnection index paths must be NAME=/path/to/file entries "
                f"separated by commas; got {spec!r}."
            )
        name, path = spec.split("=", 1)
        clean = sanitize_index_name(name)
        if clean in paths:
            raise ValueError(f"Duplicate teleconnection index name after sanitizing: {clean}")
        paths[clean] = Path(path).expanduser()
    return paths


def parse_column_list(text: str) -> Tuple[str, ...]:
    return tuple(value.strip() for value in str(text).split(",") if value.strip())


def _xlsx_col_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in str(cell_ref) if ch.isalpha()).upper()
    if not letters:
        raise ValueError(f"Invalid XLSX cell reference: {cell_ref!r}")
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch) - ord("A") + 1)
    return value - 1


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> Sequence[str]:
    try:
        xml = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return ()
    root = ET.fromstring(xml)
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings = []
    for item in root.findall("x:si", ns):
        parts = [node.text or "" for node in item.findall(".//x:t", ns)]
        strings.append("".join(parts))
    return strings


def _xlsx_cell_value(cell: ET.Element, shared_strings: Sequence[str]) -> object:
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        parts = [node.text or "" for node in cell.findall(".//x:t", ns)]
        return "".join(parts)
    value_node = cell.find("x:v", ns)
    if value_node is None or value_node.text is None:
        return ""
    raw = value_node.text
    if cell_type == "s":
        index = int(float(raw))
        return shared_strings[index] if 0 <= index < len(shared_strings) else ""
    if cell_type == "b":
        return bool(int(float(raw)))
    return raw


def read_first_xlsx_sheet(path: Path) -> Tuple[Sequence[str], Sequence[Sequence[object]]]:
    """Read the first worksheet from an XLSX file using only the Python stdlib."""
    if not path.exists():
        raise FileNotFoundError(f"Missing AllData workbook: {path}")
    with zipfile.ZipFile(path) as archive:
        shared = _xlsx_shared_strings(archive)
        sheet_names = sorted(
            name for name in archive.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )
        if not sheet_names:
            raise RuntimeError(f"No worksheets found in {path}")
        ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        root = ET.fromstring(archive.read(sheet_names[0]))
        rows = []
        max_col = 0
        for row_node in root.findall(".//x:sheetData/x:row", ns):
            values: Dict[int, object] = {}
            for cell in row_node.findall("x:c", ns):
                col = _xlsx_col_index(cell.attrib.get("r", "A1"))
                values[col] = _xlsx_cell_value(cell, shared)
                max_col = max(max_col, col)
            rows.append([values.get(col, "") for col in range(max_col + 1)])
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    headers = [str(value).strip() for value in rows[0]]
    return headers, rows[1:]


def _parse_alldata_date(value: object) -> np.datetime64:
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[D]")
    text = str(value).strip()
    if not text:
        raise ValueError("empty date")
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%Y%m%d"):
        try:
            return np.datetime64(datetime.strptime(text.split()[0], fmt).date(), "D")
        except ValueError:
            pass
    serial = float(text)
    # Excel's effective date origin for serial values, including the 1900 leap-year bug.
    return np.datetime64((datetime(1899, 12, 30) + timedelta(days=serial)).date(), "D")


def _parse_float(value: object) -> float:
    text = str(value).strip()
    if not text:
        return float("nan")
    return float(text)


def build_alldata_drivers(
    time_values: Sequence[float],
    workbook_path: Path | None,
    requested_columns: Sequence[str],
) -> Tuple[Tuple[str, ...], np.ndarray]:
    if workbook_path is None or str(workbook_path).strip() == "":
        return (), np.empty((0, len(time_values)), dtype=np.float32)
    headers, rows = read_first_xlsx_sheet(Path(workbook_path))
    normalized_headers = [sanitize_index_name(header) for header in headers]
    try:
        date_col = normalized_headers.index("date")
    except ValueError as exc:
        raise RuntimeError(f"{workbook_path}: could not find a Date column.") from exc

    expected_dates = np.array(
        [np.datetime64((BASE_DATE + timedelta(days=float(value))).date(), "D") for value in time_values],
        dtype="datetime64[D]",
    )
    by_date: Dict[np.datetime64, Sequence[object]] = {}
    for row in rows:
        if date_col >= len(row):
            continue
        try:
            date_value = _parse_alldata_date(row[date_col])
        except Exception:
            continue
        if date_value in by_date:
            raise RuntimeError(f"{workbook_path}: duplicate Date row for {date_value}.")
        by_date[date_value] = row
    missing = [str(date) for date in expected_dates if date not in by_date]
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"{workbook_path}: Date column does not cover the HeatCast MJJAS axis; "
            f"missing {len(missing)} dates, first={preview}."
        )
    if len(by_date) != len(expected_dates):
        extra = len(by_date) - len(expected_dates)
        raise RuntimeError(
            f"{workbook_path}: Date row count differs from HeatCast axis "
            f"({len(by_date)} vs {len(expected_dates)}, extra={extra})."
        )

    requested_clean = tuple(sanitize_index_name(value) for value in requested_columns)
    selected: Dict[str, int] = {}
    if requested_clean:
        lookup = {name: idx for idx, name in enumerate(normalized_headers)}
        for name in requested_clean:
            if name not in lookup:
                raise RuntimeError(f"{workbook_path}: requested AllData column {name!r} not found.")
            if lookup[name] == date_col:
                continue
            selected[name] = lookup[name]
    else:
        for col, name in enumerate(normalized_headers):
            if col == date_col or not name:
                continue
            values = []
            for date_value in expected_dates:
                row = by_date[date_value]
                cell = row[col] if col < len(row) else ""
                values.append(_parse_float(cell))
            if np.all(np.isfinite(values)):
                selected[name] = col

    if not selected:
        raise RuntimeError(
            f"{workbook_path}: no usable numeric AllData columns found. "
            "Set ALLDATA_COLUMNS to a comma-separated list if needed."
        )
    names = tuple(selected)
    matrix = np.full((len(names), len(expected_dates)), np.nan, dtype=np.float32)
    for row_index, (name, col) in enumerate(selected.items()):
        for date_index, date_value in enumerate(expected_dates):
            row = by_date[date_value]
            cell = row[col] if col < len(row) else ""
            matrix[row_index, date_index] = _parse_float(cell)
        if np.any(~np.isfinite(matrix[row_index])):
            raise RuntimeError(f"{workbook_path}: AllData column {name} has missing/non-numeric HeatCast dates.")
    print(
        f"AllData workbook matched HeatCast axis exactly: {workbook_path} "
        f"({len(expected_dates)} MJJAS dates, columns={','.join(names)})"
    )
    return names, matrix


def soil_percentile_against_train(
    train_values: np.ndarray,
    target_values: np.ndarray,
) -> np.ndarray:
    train = np.asarray(train_values, dtype=np.float32)
    target = np.asarray(target_values, dtype=np.float32)
    if train.ndim != 2 or target.ndim != 2 or train.shape[0] != target.shape[0]:
        raise ValueError("Expected train=(pixels,days), target=(pixels,inits).")
    result = np.full(target.shape, np.nan, dtype=np.float32)
    for pixel in range(train.shape[0]):
        reference = np.sort(train[pixel, np.isfinite(train[pixel])])
        finite_target = np.isfinite(target[pixel])
        if reference.size and np.any(finite_target):
            result[pixel, finite_target] = (
                100.0
                * np.searchsorted(reference, target[pixel, finite_target], side="right")
                / reference.size
            )
    return result


def load_chunk_init_index(chunk_path: Path, sidecar: Mapping[int, int] | None = None) -> int:
    with np.load(chunk_path, allow_pickle=False) as data:
        init_t = int(np.asarray(data["init_time_index"]).item()) if "init_time_index" in data else -1
    if init_t >= 0:
        return init_t
    sample_index = int(chunk_path.stem.split("_")[-1])
    if sidecar is not None and sample_index in sidecar:
        return int(sidecar[sample_index])
    raise RuntimeError(
        f"{chunk_path}: no native init_time_index and no recovery sidecar entry. "
        "Run recover_chunk_init_dates.py first."
    )


def load_sidecar(root: Path) -> Dict[int, int]:
    path = root / "incremental_arrays" / "init_dates.npz"
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {
            int(sample): int(init_t)
            for sample, init_t in zip(data["sample_index"], data["init_time_index"])
        }


def build_global_driver_table(
    time_values: Sequence[float],
    rmm: Mapping[int, Tuple[int, float]],
    nino: Mapping[Tuple[int, int], float],
    teleconnections: Mapping[str, Mapping[Tuple[int, int], float]],
    teleconnection_threshold: float,
    alldata_names: Sequence[str],
    alldata_values: np.ndarray,
    alldata_threshold: float,
    output_path: Path,
) -> None:
    dates = [BASE_DATE + timedelta(days=float(value)) for value in time_values]
    phase = np.full(len(dates), -1, dtype=np.int8)
    amplitude = np.full(len(dates), np.nan, dtype=np.float32)
    nino34 = np.full(len(dates), np.nan, dtype=np.float32)
    tele_names = tuple(sorted(teleconnections))
    tele_values = np.full((len(tele_names), len(dates)), np.nan, dtype=np.float32)
    alldata_names = tuple(str(value) for value in alldata_names)
    alldata_values = np.asarray(alldata_values, dtype=np.float32)
    if alldata_values.shape != (len(alldata_names), len(dates)):
        raise RuntimeError(
            f"AllData driver shape mismatch: values={alldata_values.shape}, "
            f"names={len(alldata_names)}, dates={len(dates)}."
        )
    for index, date in enumerate(dates):
        key = int(date.strftime("%Y%m%d"))
        if key in rmm:
            phase[index], amplitude[index] = rmm[key]
        if (date.year, date.month) in nino:
            nino34[index] = nino[(date.year, date.month)]
        for tele_idx, name in enumerate(tele_names):
            values = teleconnections[name]
            if (date.year, date.month) in values:
                tele_values[tele_idx, index] = values[(date.year, date.month)]
    mjjas = np.array([date.month in MJJAS_MONTHS for date in dates])
    if np.any((phase[mjjas] < 1) | (phase[mjjas] > 8) | ~np.isfinite(amplitude[mjjas])):
        raise RuntimeError("RMM table does not cover every MJJAS date on the HeatCast time axis.")
    if np.any(~np.isfinite(nino34[mjjas])):
        raise RuntimeError("Nino3.4 table does not cover every MJJAS month on the HeatCast time axis.")
    for tele_idx, name in enumerate(tele_names):
        if np.any(~np.isfinite(tele_values[tele_idx, mjjas])):
            raise RuntimeError(f"Teleconnection index {name} does not cover every MJJAS month on the HeatCast time axis.")
    for alldata_idx, name in enumerate(alldata_names):
        if np.any(~np.isfinite(alldata_values[alldata_idx, mjjas])):
            raise RuntimeError(f"AllData driver {name} does not cover every MJJAS date on the HeatCast time axis.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        init_time_index=np.arange(len(dates), dtype=np.int32),
        init_date=np.array([int(date.strftime("%Y%m%d")) for date in dates], dtype=np.int32),
        mjo_phase=phase,
        mjo_amplitude=amplitude,
        nino34=nino34,
        teleconnection_names=np.array(tele_names, dtype="U64"),
        teleconnection_values=tele_values,
        teleconnection_threshold=np.array(float(teleconnection_threshold), dtype=np.float32),
        alldata_names=np.array(alldata_names, dtype="U64"),
        alldata_values=alldata_values,
        alldata_threshold=np.array(float(alldata_threshold), dtype=np.float32),
    )
    print(
        f"Saved global driver table: {output_path} "
        f"(teleconnections={','.join(tele_names) or 'none'}, "
        f"alldata={','.join(alldata_names) or 'none'})"
    )


def build_fold_soil_table(
    fold: int,
    manifest: Mapping[str, object],
    chunks: Sequence[Path],
    shared_data: Mapping[str, np.ndarray],
    land_mask: np.ndarray,
    cache_dir: Path,
    block_pixels: int,
) -> None:
    time_values = np.asarray(shared_data["time_values"])
    dates = [BASE_DATE + timedelta(days=float(value)) for value in time_values]
    months = np.array([date.month for date in dates], dtype=np.int8)
    years = np.array([date.year for date in dates], dtype=np.int16)
    train_years = set(int(value) for value in manifest["train_years"])
    sidecar = load_sidecar(Path(manifest["root"]))
    init_indices = np.array(
        [load_chunk_init_index(path, sidecar) for path in chunks],
        dtype=np.int32,
    )
    if len(np.unique(init_indices)) != len(init_indices):
        raise RuntimeError(f"Fold {fold}: duplicate init indices in soil-table request.")
    land_flat = np.flatnonzero(land_mask.ravel())
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / f"fold{fold}_smpct_float16.dat"
    index_path = cache_dir / f"fold{fold}_smpct_index.npz"
    if data_path.exists() and index_path.exists():
        try:
            with np.load(index_path, allow_pickle=False) as data:
                cached_init = np.asarray(data["init_time_index"], dtype=np.int32)
                cached_shape = tuple(int(value) for value in np.asarray(data["shape"]).tolist())
                cached_train = set(int(value) for value in np.asarray(data["train_years"]).tolist())
                cached_undefined = float(np.asarray(data["undefined_fraction"]).item())
            expected_shape = (len(init_indices), len(land_flat))
            expected_bytes = int(np.prod(expected_shape)) * np.dtype(np.float16).itemsize
            if (
                np.array_equal(cached_init, init_indices)
                and cached_shape == expected_shape
                and cached_train == train_years
                and cached_undefined < 0.05
                and data_path.stat().st_size == expected_bytes
            ):
                print(f"Reusing fold-safe soil-percentile cache for fold {fold}: {data_path}")
                return
        except Exception:
            pass
    soil = np.asarray(shared_data["soil_moisture"])
    if soil.ndim != 3 or soil.shape[-1] != len(time_values):
        raise RuntimeError(
            f"Expected soil_moisture=(H,W,T) with T={len(time_values)}, got {soil.shape}."
        )
    soil_flat = soil.reshape(-1, len(time_values))
    output = np.memmap(
        data_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(init_indices), len(land_flat)),
    )
    for month in MJJAS_MONTHS:
        train_t = np.where(np.isin(years, list(train_years)) & (months == month))[0]
        target_rows = np.where(months[init_indices] == month)[0]
        if train_t.size == 0 or target_rows.size == 0:
            continue
        target_t = init_indices[target_rows]
        for start in range(0, len(land_flat), int(block_pixels)):
            stop = min(start + int(block_pixels), len(land_flat))
            pixels = land_flat[start:stop]
            train_values = soil_flat[pixels][:, train_t]
            target_values = soil_flat[pixels][:, target_t]
            output[target_rows, start:stop] = soil_percentile_against_train(
                train_values,
                target_values,
            ).T.astype(np.float16)
    output.flush()
    undefined_fraction = float(np.mean(~np.isfinite(output)))
    if undefined_fraction >= 0.05:
        raise RuntimeError(
            f"Fold {fold}: undefined soil-percentile fraction={undefined_fraction:.4f} >= 0.05."
        )
    np.savez_compressed(
        index_path,
        init_time_index=init_indices,
        shape=np.array(output.shape, dtype=np.int64),
        dtype=np.array("float16"),
        data_file=np.array(data_path.name),
        source_fold=np.array(fold, dtype=np.int16),
        train_years=np.array(sorted(train_years), dtype=np.int16),
        undefined_fraction=np.array(undefined_fraction, dtype=np.float32),
        land_count=np.array(len(land_flat), dtype=np.int32),
    )
    print(
        f"Saved fold {fold} soil-percentile memmap: {data_path} "
        f"(undefined={undefined_fraction:.4%})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmm_path", required=True)
    parser.add_argument("--nino34_path", required=True)
    parser.add_argument(
        "--teleconnection_index_paths",
        default="",
        help="Optional comma-separated NAME=/path monthly index files. "
        "Common names: pna,nao,ao,pdo,amo,epo,wpo,npo.",
    )
    parser.add_argument(
        "--teleconnection_threshold",
        type=float,
        default=0.5,
        help="Fixed absolute threshold for generic monthly index positive/neutral/negative strata.",
    )
    parser.add_argument(
        "--alldata_path",
        default="",
        help="Optional AllData.xlsx workbook with a Date column exactly matching the HeatCast MJJAS axis.",
    )
    parser.add_argument(
        "--alldata_columns",
        default="",
        help="Optional comma-separated AllData columns to stratify. Default: all finite numeric non-Date columns.",
    )
    parser.add_argument(
        "--alldata_threshold",
        type=float,
        default=0.5,
        help="Fixed absolute threshold for AllData positive/neutral/negative strata.",
    )
    parser.add_argument("--input_root", default="exceedance_eval_incremental")
    parser.add_argument(
        "--run_names",
        default="cvfold0_dist_v2_normfix,cvfold1_dist_v2_normfix,cvfold2_dist_v2_normfix,cvfold3_dist_v2_normfix,cvfold4_dist_v2_normfix",
    )
    parser.add_argument("--window_leads", default="12,13,14,15,16,17,18")
    parser.add_argument("--output_dir", default="data_cache/slow_driver_tables")
    parser.add_argument("--block_pixels", type=int, default=2048)
    args = parser.parse_args()

    import cfm_mesh_train as cfm
    import exceedance_eval as ee
    import stitch_exceedance_folds as stitch

    run_names = tuple(value.strip() for value in args.run_names.split(",") if value.strip())
    window_leads = ee.parse_int_list(args.window_leads)
    output_dir = Path(args.output_dir)
    cfm.apply_extended_global_fields()
    shared_data = cfm.prepare_shared_data(cfm.Config, rank=0, world_size=1, ddp=False)
    land_mask = np.asarray(cfm.load_conus_mask(cfm.Config).cpu().numpy() > 0.5, dtype=bool)
    alldata_names, alldata_values = build_alldata_drivers(
        shared_data["time_values"],
        Path(args.alldata_path).expanduser() if str(args.alldata_path).strip() else None,
        parse_column_list(args.alldata_columns),
    )
    build_global_driver_table(
        shared_data["time_values"],
        parse_rmm_file(Path(args.rmm_path)),
        parse_nino34_file(Path(args.nino34_path)),
        {
            name: parse_monthly_index_file(path)
            for name, path in parse_named_index_paths(args.teleconnection_index_paths).items()
        },
        float(args.teleconnection_threshold),
        alldata_names,
        alldata_values,
        float(args.alldata_threshold),
        output_dir / "mjo_enso_by_init.npz",
    )
    for run_name in run_names:
        manifest, _, chunks = stitch.load_fold_inputs(Path(args.input_root), run_name, window_leads)
        build_fold_soil_table(
            int(manifest["source_fold"]),
            manifest,
            chunks,
            shared_data,
            land_mask,
            output_dir,
            args.block_pixels,
        )
    print("Slow-driver tables complete. Soil percentiles use fold train years only.")


if __name__ == "__main__":
    main()
