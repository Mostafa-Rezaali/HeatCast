#!/usr/bin/env python3
"""Recover init dates for legacy incremental chunks without rewriting them."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np


BASE_DATE = datetime(1981, 5, 1)


def validate_chunk_target_date(
    chunk_path: Path,
    init_time_index: int,
    target_center_time_index: int,
    time_values: Sequence[float],
) -> Tuple[int, int]:
    target_date = BASE_DATE + timedelta(days=float(time_values[int(target_center_time_index)]))
    with np.load(chunk_path, allow_pickle=False) as data:
        stored_year = int(np.asarray(data["year"]).item())
        stored_month = int(np.asarray(data["month"]).item())
    if stored_year != target_date.year or stored_month != target_date.month:
        raise RuntimeError(
            f"{chunk_path}: ordering recovery mismatch for init_time_index={init_time_index}; "
            f"stored target={stored_year:04d}-{stored_month:02d}, rebuilt target="
            f"{target_date.year:04d}-{target_date.month:02d}. Aborting fold; do not guess."
        )
    return stored_year, stored_month


def recover_fold(
    manifest: Mapping[str, object],
    chunks: Sequence[Path],
    test_indices: Sequence[int],
    time_values: np.ndarray,
    center_lead: int,
) -> Path:
    if len(chunks) != len(test_indices):
        raise RuntimeError(
            f"{manifest['run_name']}: chunk count={len(chunks)} does not match rebuilt "
            f"test-index count={len(test_indices)}."
        )
    sample_indices = []
    init_indices = []
    target_indices = []
    init_dates = []
    for sample_index, (chunk_path, init_t) in enumerate(zip(chunks, test_indices)):
        expected_name = f"sample_{sample_index:05d}.npz"
        if chunk_path.name != expected_name:
            raise RuntimeError(
                f"{manifest['run_name']}: expected ordered chunk {expected_name}, got {chunk_path.name}."
            )
        target_t = int(init_t) + int(center_lead)
        validate_chunk_target_date(chunk_path, int(init_t), target_t, time_values)
        init_date = BASE_DATE + timedelta(days=float(time_values[int(init_t)]))
        sample_indices.append(sample_index)
        init_indices.append(int(init_t))
        target_indices.append(target_t)
        init_dates.append(int(init_date.strftime("%Y%m%d")))

    sidecar = Path(manifest["root"]) / "incremental_arrays" / "init_dates.npz"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sidecar,
        sample_index=np.asarray(sample_indices, dtype=np.int32),
        init_time_index=np.asarray(init_indices, dtype=np.int32),
        target_center_time_index=np.asarray(target_indices, dtype=np.int32),
        init_date=np.asarray(init_dates, dtype=np.int32),
        source_fold=np.array(int(manifest["source_fold"]), dtype=np.int16),
        run_name=np.array(str(manifest["run_name"])),
        window_leads=np.asarray(manifest["window_leads"], dtype=np.int16),
    )
    return sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", default="exceedance_eval_incremental")
    parser.add_argument(
        "--run_names",
        default="cvfold0_dist_v2_normfix,cvfold1_dist_v2_normfix,cvfold2_dist_v2_normfix,cvfold3_dist_v2_normfix,cvfold4_dist_v2_normfix",
    )
    parser.add_argument("--window_leads", default="12,13,14,15,16,17,18")
    args = parser.parse_args()

    import cfm_mesh_train as cfm
    import exceedance_eval as ee
    import stitch_exceedance_folds as stitch

    run_names = ee.parse_str_list(args.run_names) if hasattr(ee, "parse_str_list") else tuple(
        value.strip() for value in args.run_names.split(",") if value.strip()
    )
    window_leads = ee.parse_int_list(args.window_leads)
    center_lead = ee.window_center_lead(window_leads)
    cfm.apply_extended_global_fields()
    shared_data = cfm.prepare_shared_data(cfm.Config, rank=0, world_size=1, ddp=False)
    time_values = np.asarray(shared_data["time_values"])

    for run_name in run_names:
        manifest, _, chunks = stitch.load_fold_inputs(Path(args.input_root), run_name, window_leads)
        fold = int(manifest["source_fold"])
        cfm.Config.CV_FOLD = fold
        cfm.Config.CV_TEST_OFFSETS = (fold,)
        cfm.Config.CV_VAL_OFFSETS = ((fold + 1) % int(cfm.Config.CV_STRIDE),)
        cfm.Config.MULTI_LEAD_TUBE = True
        cfm.Config.PREDICTION_LEADS = tuple(window_leads)
        runs = cfm.detect_continuous_runs(time_values)
        valid = cfm.build_valid_indices(
            runs,
            lead_time=max(window_leads),
            min_history=cfm.required_input_history(cfm.Config),
        )
        _, _, test_indices, _, _, test_years = cfm.build_crossval_split(valid, time_values)
        if set(int(value) for value in test_years) != set(manifest["test_years"]):
            raise RuntimeError(f"{run_name}: rebuilt test years do not match manifest test years.")
        sidecar = recover_fold(manifest, chunks, test_indices, time_values, center_lead)
        print(f"Recovered {len(chunks)} legacy init dates for fold {fold}: {sidecar}")


if __name__ == "__main__":
    main()
