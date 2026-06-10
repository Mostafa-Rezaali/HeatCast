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
  raw_dir/ens_init_{HDATE}.grib
  raw_dir/init_list.txt
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
        "step": "/".join(str(step) for step in range(6, int(max_step_hours) + 1, 6)),
        "date": model_date.strftime("%Y-%m-%d"),
        "hdate": "/".join(value.strftime("%Y-%m-%d") for value in hdates),
        "grid": "1.5/1.5",
        "area": str(area),
        "type": str(kind),
    }
    if kind == "pf":
        request["number"] = "1/2/3/4/5/6/7/8/9/10"
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


def assemble_by_hdate(downloads: Path, parts: Path, raw_dir: Path) -> list[str]:
    print("Splitting model files by hdate...")
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
        final = raw_dir / f"ens_init_{label}.grib"
        with final.open("wb") as output:
            for piece in pieces:
                output.write(piece.read_bytes())
    (raw_dir / "init_list.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
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
        default=2022,
        help="Real-time year whose Mon/Thu model dates anchor the reforecasts.",
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
    downloads = raw_dir / "downloads"
    parts = raw_dir / "parts"
    for directory in (raw_dir, downloads, parts):
        directory.mkdir(parents=True, exist_ok=True)

    model_dates = list(mjjas_mon_thu(args.rt_year))
    max_step_hours = int(args.max_lead_days) * 24
    print(
        f"{len(model_dates)} model dates (MJJAS Mon/Thu {args.rt_year}), "
        f"{args.hindcast_years} hindcast years each, steps 6..{max_step_hours}h."
    )

    if not args.skip_download:
        try:
            import cdsapi
        except ImportError as exc:
            raise RuntimeError(
                "Missing cdsapi>=0.7.7. Install it and configure ~/.cdsapirc for ECDS."
            ) from exc
        client = cdsapi.Client()
        for model_date in model_dates:
            hdates = hindcast_dates(model_date, args.hindcast_years)
            for kind in ("cf", "pf"):
                target = downloads / f"model_{model_date.strftime('%Y%m%d')}_{kind}.grib"
                if target.exists() and target.stat().st_size > 0:
                    print(f"  exists, skipping: {target.name}")
                    continue
                print(f"  retrieving {target.name} ({len(hdates)} hdates)...")
                retrieve(client, model_date, hdates, kind, target, args.area, max_step_hours)

    labels = assemble_by_hdate(downloads, parts, raw_dir)
    print(f"Done: {len(labels)} init files in {raw_dir}, labels in init_list.txt")
    print("Next: submit submit_ens_ingest.slurm, then submit_ens_score_compare.slurm.")


if __name__ == "__main__":
    sys.exit(main())
