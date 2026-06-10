from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from build_driver_tables import (
    BASE_DATE,
    parse_nino34_file,
    parse_rmm_file,
    soil_percentile_against_train,
)
from forecasts_of_opportunity import (
    DriverLookup,
    OpportunityStats,
    enso_stratum,
    mjo_stratum,
    stream_chunks,
)
from recover_chunk_init_dates import validate_chunk_target_date


def test_fake_rmm_and_nino_parsers(tmp_path: Path):
    rmm_path = tmp_path / "rmm.txt"
    rmm_path.write_text(
        "year month day RMM1 RMM2 phase amplitude\n"
        "2001 5 1 0.1 0.2 5 1.4\n"
        "2001 5 2 0.1 0.2 2 0.7\n",
        encoding="utf-8",
    )
    nino_path = tmp_path / "nino.txt"
    nino_path.write_text(
        "2001 5 27.1 26.3 0.8\n"
        "2001 12 24.0 25.0 -1.0\n",
        encoding="utf-8",
    )
    rmm = parse_rmm_file(rmm_path)
    nino = parse_nino34_file(nino_path)
    assert rmm[20010501] == (5, 1.4)
    assert mjo_stratum(*rmm[20010501]) == "phase_5"
    assert mjo_stratum(*rmm[20010502]) == "inactive"
    assert enso_stratum(nino[(2001, 5)]) == "el_nino"
    assert enso_stratum(nino[(2001, 12)]) == "la_nina"


def test_phase_five_cluster_has_positive_conditional_bss():
    phase_five = OpportunityStats()
    phase_five.update(
        probability=np.array([0.85] * 8 + [0.15] * 2),
        truth=np.array([1.0] * 8 + [0.0] * 2),
        base_rate=np.full(10, 0.05),
    )
    assert phase_five.metrics()["bss_conditional"] > 0.7


def test_legacy_recovery_rejects_shuffled_target_month(tmp_path: Path):
    chunk = tmp_path / "sample_00000.npz"
    np.savez_compressed(chunk, year=np.array(1981), month=np.array(7))
    target_day = (datetime(1981, 6, 15) - BASE_DATE).days
    with pytest.raises(RuntimeError, match="ordering recovery mismatch"):
        validate_chunk_target_date(chunk, 0, 0, np.array([target_day], dtype=float))


def test_soil_percentile_is_train_only_with_test_sentinel():
    train = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32)
    target = np.array([[2.0, 9999.0], [15.0, -9999.0]], dtype=np.float32)
    percentiles = soil_percentile_against_train(train, target)
    assert np.isclose(percentiles[0, 0], 200.0 / 3.0)
    assert np.isclose(percentiles[1, 0], 100.0 / 3.0)
    assert percentiles[0, 1] == 100.0
    assert percentiles[1, 1] == 0.0


class _IdentityModel:
    def predict_features(self, features):
        return np.asarray(features[:, 0], dtype=np.float32)


def test_driver_stream_uses_legacy_sidecar_and_builds_interactions(tmp_path: Path):
    chunk = tmp_path / "sample_00000.npz"
    np.savez_compressed(
        chunk,
        init_margin=np.array([0.1, 0.2, 0.8], dtype=np.float32),
        forecast_margin=np.zeros(3, dtype=np.float32),
        model_sigma=np.array([0.1, 0.5, 0.9], dtype=np.float32),
        truth=np.array([0, 0, 1], dtype=np.uint8),
        base_rate=np.full(3, 0.05, dtype=np.float32),
        year=np.array(2000),
        month=np.array(7),
        source_fold=np.array(0),
        init_time_index=np.array(-1),
    )
    soil_path = tmp_path / "soil.dat"
    soil = np.memmap(soil_path, mode="w+", dtype=np.float16, shape=(1, 3))
    soil[0] = np.array([10.0, 50.0, 90.0])
    soil.flush()
    lookup = DriverLookup(
        mjo_phase=np.array([5], dtype=np.int8),
        mjo_amplitude=np.array([1.5], dtype=np.float32),
        nino34=np.array([0.8], dtype=np.float32),
        sidecars={0: {0: 0}},
        soil_rows={0: {0: 0}},
        soil_memmaps={0: np.memmap(soil_path, mode="r", dtype=np.float16, shape=(1, 3))},
    )
    manifest = {"source_fold": 0, "test_years": {2000}}
    boundaries = {
        "confidence_edges": np.array([0.5]),
        "sigma_edges": np.linspace(0.1, 0.9, 9),
        "forecast_margin_edges": np.linspace(0.1, 0.9, 9),
        "sigma_bottom_tercile_edge": np.array(0.3),
    }
    global_stats, _, years = stream_chunks(
        [(manifest, _IdentityModel(), boundaries, [chunk])],
        land_count=3,
        region_land_masks={},
        confidence_percentiles=[90],
        driver_lookup=lookup,
        progress_every=100,
    )
    assert years == {2000}
    assert global_stats[("mjo_phase", "phase_5")].count == 3
    assert global_stats[("enso_state", "el_nino")].count == 3
    assert global_stats[("soil_moisture_tercile", "dry")].count == 1
    assert global_stats[("mjo_phase_x_low_sigma", "phase_5__bottom_sigma_tercile")].count == 1
