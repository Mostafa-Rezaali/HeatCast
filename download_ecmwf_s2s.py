#!/usr/bin/env python3
"""Download ECMWF ENS S2S T2max reforecasts for the HeatCast benchmark.

MARS requests queue server-side, so this downloader may run on a HiPerGator
login node. The subsequent ingestion and scoring stages remain Slurm jobs.

Prerequisites:
  pip install "cdsapi>=0.7.7" --user
  Register at https://ecds.ecmwf.int, accept the S2S reforecasts license at
  https://ecds.ecmwf.int/datasets/s2s-reforecasts, and configure ~/.cdsapirc.
  Load ecCodes so grib_copy is available.

Outputs:
  raw_dir/ens_init_{HDATE}_rt{RTYEAR}.grib
  raw_dir/init_list_rt{RTYEAR}.txt
  raw_dir/init_list.txt  (combined unique hdate list)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Sequence


def mjjas_mon_thu(year: int) -> Iterable[date]:
    day = date(int(year), 5, 1)
    end = date(int(year), 9, 30)
    while day <= end:
        if day.weekday() in (0, 3):
            yield day
        day += timedelta(days=1)


def retrieve(
    client,
    model_date: date,
    hdates: Sequence[date],
    kind: str,
    target: Path,
    area: str,
    max_step_hours: int,
) -> None:
    request = {
        "class": "s2",
        "expver": "prod",
        "model": "glob",
        "origin": "ecmf",
        "stream": "enfh",
        "time": "00:00:00",
        "levtype": "sfc",
        "param": "121",
        "step": [str(step) for step in range(6, int(max_step_hours) + 1, 6)],
        "date": model_date.strftime("%Y-%m-%d"),
        "hdate": [value.strftime("%Y-%m-%d") for value in hdates],
        "grid": "1.5/1.5",
        "area": str(area),
        "type": str(kind),
    }
    if kind == "pf":
        request["number"] = [str(number) for number in range(1, 11)]
    client.retrieve("s2s-reforecasts", request, str(target))


def split_by_hdate(grib_path: Path, out_dir: Path) -> None:
    pattern = str(out_dir / f"part_[hdate]_{grib_path.stem}.grib")
    subprocess.run(["grib_copy", str(grib_path), pattern], check=True)


def hindcast_dates(model_date: date, years: int) -> list[date]:
    output = []
    for back in range(int(years), 0, -1):
        try:
            output.append(model_date.replace(year=model_date.year - back))
        except ValueError:
            continue
    return output


def parse_year_list(value: str) -> tuple[int, ...]:
    years = tuple(dict.fromkeys(int(item.strip()) for item in str(value).split(",") if item.strip()))
    if not years:
        raise ValueError("At least one real-time year is required.")
    return years


def assemble_by_hdate(downloads: Path, parts: Path, raw_dir: Path, rt_year: int) -> list[str]:
    print(f"Splitting rt{rt_year} model files by hdate...")
    for grib_path in sorted(downloads.glob("model_*.grib")):
        split_by_hdate(grib_path, parts)

    labels = sorted({path.name.split("_")[1] for path in parts.glob("part_*.grib")})
    print(f"Assembling {len(labels)} per-init files...")
    for label in labels:
        pieces = sorted(parts.glob(f"part_{label}_*.grib"))
        if len(pieces) != 2:
            raise RuntimeError(
                f"Hindcast init {label}: expected one control and one perturbed part, found {pieces}."
            )
        final = raw_dir / f"ens_init_{label}_rt{int(rt_year)}.grib"
        with final.open("wb") as output:
            for piece in pieces:
                output.write(piece.read_bytes())
    (raw_dir / f"init_list_rt{int(rt_year)}.txt").write_text(
        "\n".join(labels) + "\n",
        encoding="utf-8",
    )
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw_dir",
        default="/blue/nessie/mostafarezaali/Teleconnection/ens_reforecast/raw",
    )
    parser.add_argument(
        "--rt_year",
        type=int,
        default=None,
        help="Single-value alias for --rt_years.",
    )
    parser.add_argument(
        "--rt_years",
        default="2022",
        help="Comma-separated real-time years whose Mon/Thu model dates anchor the reforecasts.",
    )
    parser.add_argument("--hindcast_years", type=int, default=20)
    parser.add_argument("--max_lead_days", type=int, default=28)
    parser.add_argument("--area", default="50/-125/24/-66", help="CONUS box N/W/S/E.")
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Only split and assemble already-downloaded model files.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rt_years = (int(args.rt_year),) if args.rt_year is not None else parse_year_list(args.rt_years)
    max_step_hours = int(args.max_lead_days) * 24
    client = None
    if not args.skip_download:
        try:
            import cdsapi
        except ImportError as exc:
            raise RuntimeError(
                "Missing cdsapi>=0.7.7. Install it and configure ~/.cdsapirc for ECDS."
            ) from exc
        client = cdsapi.Client()

    combined_labels = set()
    for rt_year in rt_years:
        downloads = raw_dir / "downloads" / f"rt{rt_year}"
        parts = raw_dir / "parts" / f"rt{rt_year}"
        for directory in (downloads, parts):
            directory.mkdir(parents=True, exist_ok=True)
        model_dates = list(mjjas_mon_thu(rt_year))
        print(
            f"{len(model_dates)} model dates (MJJAS Mon/Thu {rt_year}), "
            f"{args.hindcast_years} hindcast years each, steps 6..{max_step_hours}h."
        )
        for model_date in model_dates:
            if not args.skip_download:
                hdates = hindcast_dates(model_date, args.hindcast_years)
                for kind in ("cf", "pf"):
                    target = downloads / f"model_{model_date.strftime('%Y%m%d')}_{kind}.grib"
                    if target.exists() and target.stat().st_size > 0:
                        print(f"  exists, skipping: rt{rt_year}/{target.name}")
                        continue
                    print(f"  retrieving rt{rt_year}/{target.name} ({len(hdates)} hdates)...")
                    retrieve(client, model_date, hdates, kind, target, args.area, max_step_hours)
        labels = assemble_by_hdate(downloads, parts, raw_dir, rt_year)
        combined_labels.update(labels)
        print(f"Completed rt{rt_year}: {len(labels)} tagged init files.")

    combined = sorted(combined_labels)
    (raw_dir / "init_list.txt").write_text("\n".join(combined) + "\n", encoding="utf-8")
    print(f"Done: {len(combined)} unique hdates across cycles; combined labels in init_list.txt")
    print("Next: submit submit_ens_ingest.slurm, then submit_ens_score_compare.slurm.")


if __name__ == "__main__":
    sys.exit(main())
