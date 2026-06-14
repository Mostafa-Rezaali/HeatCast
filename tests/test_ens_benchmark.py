from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

import ens_ingest as ingest
import ens_score
from ens_common import (
    apply_quantile_mapping,
    common_init_indices,
    fit_quantile_mapping,
    intersection_years,
    member_fraction_probability,
)
from download_ecmwf_s2s import hindcast_dates, mjjas_mon_thu, retrieve
from ens_ingest import load_init_list, load_native_daily_max, validate_ingested_output
from stitch_exceedance_folds import load_fold_inputs


def test_quantile_mapping_is_monotonic_and_reproduces_train_distribution():
    rng = np.random.default_rng(42)
    source = rng.normal(size=(6, 4, 20, 30)).astype(np.float32).reshape(24, 600)
    target = (1.5 * source + 0.75).astype(np.float32)
    levels = np.linspace(0.0, 1.0, 51)
    source_q, target_q = fit_quantile_mapping(source, target, levels)
    mapped = apply_quantile_mapping(source, source_q, target_q)
    assert np.all(np.diff(target_q, axis=0) >= -1e-6)
    assert np.mean(np.abs(np.mean(mapped, axis=0) - np.mean(target, axis=0))) < 0.1


def test_member_fraction_probability_uses_all_valid_members():
    members = np.array([
        [0.0, 4.0, np.nan],
        [2.0, 5.0, 3.0],
        [4.0, 6.0, 5.0],
        [6.0, 7.0, 7.0],
    ], dtype=np.float32)
    probability = member_fraction_probability(members, np.array([3.0, 5.5, 4.0], dtype=np.float32))
    assert np.allclose(probability, np.array([0.5, 0.5, 2.0 / 3.0], dtype=np.float32))


def test_chunk_schema_round_trip_through_stitch_loader(tmp_path: Path):
    root = tmp_path / "cvfold0_ens_synthetic" / "test" / "window_12-13-14-15-16-17-18"
    array_dir = root / "incremental_arrays"
    chunk_dir = array_dir / "test_chunks"
    chunk_dir.mkdir(parents=True)
    np.savez_compressed(
        array_dir / "manifest.npz",
        run_name=np.array("cvfold0_ens_synthetic"),
        source_fold=np.array(0, dtype=np.int16),
        target_mode=np.array("window"),
        window_leads=np.arange(12, 19, dtype=np.int16),
        train_years=np.array([2000], dtype=np.int16),
        calibration_years=np.array([2001], dtype=np.int16),
        test_years=np.array([2002], dtype=np.int16),
        calibration_split=np.array("val"),
        eval_split=np.array("test"),
        sample_count=np.array(1, dtype=np.int32),
        valid_cell_count=np.array(600, dtype=np.int64),
    )
    np.savez_compressed(
        array_dir / "calibration_pairs.npz",
        init_margin=np.linspace(0.0, 1.0, 600, dtype=np.float32),
        forecast_margin=np.linspace(-2.0, 2.0, 600, dtype=np.float32),
        model_sigma=np.ones(600, dtype=np.float32),
        truth=np.tile(np.array([0, 1], dtype=np.uint8), 300),
        base_rate=np.full(600, 0.05, dtype=np.float32),
        year=np.full(600, 2001, dtype=np.int16),
        source_fold=np.array(0, dtype=np.int16),
    )
    np.savez_compressed(
        chunk_dir / "sample_00000.npz",
        init_margin=np.linspace(0.0, 1.0, 600, dtype=np.float32),
        forecast_margin=np.linspace(-2.0, 2.0, 600, dtype=np.float32),
        model_sigma=np.ones(600, dtype=np.float32),
        truth=np.tile(np.array([0, 1], dtype=np.uint8), 300),
        base_rate=np.full(600, 0.05, dtype=np.float32),
        year=np.array(2002, dtype=np.int16),
        month=np.array(7, dtype=np.int8),
        source_fold=np.array(0, dtype=np.int16),
        init_time_index=np.array(123, dtype=np.int32),
        target_center_time_index=np.array(138, dtype=np.int32),
    )
    manifest, calibration, chunks = load_fold_inputs(
        tmp_path,
        "cvfold0_ens_synthetic",
        tuple(range(12, 19)),
    )
    assert manifest["sample_count"] == 1
    assert calibration["truth"].size == 600
    assert chunks == [chunk_dir / "sample_00000.npz"]


def test_intersection_logic_restricts_years_and_init_dates():
    assert intersection_years([1981, 1982, 1983], [1982, 1983, 1984]) == (1982, 1983)
    assert common_init_indices({3: "a", 7: "b", 9: "c"}, {2: "d", 7: "e", 9: "f"}) == (7, 9)


def test_s2s_download_dates_and_ingest_init_list(tmp_path: Path):
    model_dates = list(mjjas_mon_thu(2022))
    assert model_dates
    assert all(value.month in (5, 6, 7, 8, 9) and value.weekday() in (0, 3) for value in model_dates)
    hdates = hindcast_dates(model_dates[0], 20)
    assert len(hdates) == 20
    assert hdates[0].year == 2002
    init_list = tmp_path / "init_list.txt"
    init_list.write_text("20020502\n20020502\n20020506\n", encoding="utf-8")
    assert load_init_list(init_list) == ["20020502", "20020506"]


def test_s2s_retrieve_uses_current_ecds_dataset(tmp_path: Path):
    calls = []

    class Client:
        def retrieve(self, dataset, request, target):
            calls.append((dataset, request, target))

    retrieve(
        Client(),
        model_date=list(mjjas_mon_thu(2022))[0],
        hdates=[hindcast_dates(list(mjjas_mon_thu(2022))[0], 1)[0]],
        kind="pf",
        target=tmp_path / "out.grib",
        area="50/-125/24/-66",
        max_step_hours=24,
    )
    dataset, request, target = calls[0]
    assert dataset == "s2s-reforecasts"
    assert request["number"] == [str(number) for number in range(1, 11)]
    assert request["hdate"] == ["2021-05-02"]
    assert request["step"] == ["6", "12", "18", "24"]
    assert "dataset" not in request and "target" not in request
    assert target.endswith("out.grib")


def test_ingest_opens_and_combines_control_and_perturbed_grib_groups(monkeypatch, tmp_path: Path):
    calls = []
    steps = np.array([6, 12, 18, 24], dtype="timedelta64[h]")
    lat = np.array([25.0, 26.5], dtype=np.float32)
    lon = np.array([235.0, 236.5], dtype=np.float32)

    class FakeDataArray:
        def __init__(self, values, dims, coords):
            self.values = np.asarray(values)
            self.dims = tuple(dims)
            self.coords = {name: np.asarray(value) for name, value in coords.items()}

        def squeeze(self, drop=True):
            return self

        def expand_dims(self, dimensions):
            if isinstance(dimensions, str):
                name = dimensions
                values = np.asarray(self.coords[name]).reshape(-1)
                return FakeDataArray(self.values[None, ...], (name, *self.dims), self.coords)
            name, values = next(iter(dimensions.items()))
            coords = dict(self.coords)
            coords[name] = np.asarray(values)
            return FakeDataArray(self.values[None, ...], (name, *self.dims), coords)

        def transpose(self, *dims):
            order = [self.dims.index(name) for name in dims]
            return FakeDataArray(np.transpose(self.values, order), dims, self.coords)

        def __getitem__(self, name):
            return SimpleNamespace(values=self.coords[name], attrs={})

    class FakeDataset:
        def __init__(self, data):
            self.data = data
            self.data_vars = {"mx2t6": data}

        def __contains__(self, name):
            return name in self.data_vars

        def __getitem__(self, name):
            return self.data_vars[name]

        def close(self):
            pass

    def fake_open_dataset(path, engine=None, backend_kwargs=None):
        data_type = backend_kwargs["filter_by_keys"]["dataType"]
        calls.append((engine, data_type, backend_kwargs["indexpath"]))
        if data_type == "cf":
            values = np.arange(16, dtype=np.float32).reshape(4, 2, 2)
            return FakeDataset(
                FakeDataArray(
                    values,
                    ("step", "latitude", "longitude"),
                    {"number": np.array(0), "step": steps, "latitude": lat, "longitude": lon},
                )
            )
        values = 100 + np.arange(32, dtype=np.float32).reshape(2, 4, 2, 2)
        return FakeDataset(
            FakeDataArray(
                values,
                ("number", "step", "latitude", "longitude"),
                {"number": [1, 2], "step": steps, "latitude": lat, "longitude": lon},
            )
        )

    monkeypatch.setitem(sys.modules, "xarray", SimpleNamespace(open_dataset=fake_open_dataset))
    daily, source_lat, source_lon, members = load_native_daily_max(
        tmp_path / "mixed.grib",
        "mx2t6",
        max_lead=1,
        expected_members=3,
    )
    assert calls == [("cfgrib", "cf", ""), ("cfgrib", "pf", "")]
    assert daily.shape == (3, 1, 2, 2)
    assert np.array_equal(members, np.array([0, 1, 2]))
    assert np.array_equal(source_lat, lat)
    assert np.array_equal(source_lon, lon)
    assert np.array_equal(daily[:, 0, 0, 0], np.array([12.0, 112.0, 128.0], dtype=np.float32))


def test_ingest_worker_writes_atomic_resume_safe_output(monkeypatch, tmp_path: Path):
    land_mask = np.array([[True, False], [True, True]])
    target_lat = np.array([[25.0, 25.0], [26.0, 26.0]], dtype=np.float32)
    target_lon = np.array([[-100.0, -99.0], [-100.0, -99.0]], dtype=np.float32)
    native = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)

    monkeypatch.setattr(
        ingest,
        "load_native_daily_max",
        lambda *args: (native, target_lat[:, 0], target_lon[0], np.array([0])),
    )
    monkeypatch.setattr(
        ingest,
        "bilinear_regrid_regular",
        lambda values, *args: values.copy(),
    )
    ingest._initialize_ingest_worker(land_mask, target_lat, target_lon)
    output_path = tmp_path / "init_20010701.npz"
    assert ingest.ingest_one_init(
        "20010701",
        str(tmp_path / "raw.grib"),
        str(output_path),
        "mx2t6",
        3,
        1,
        123,
    ) == "20010701"
    assert output_path.exists()
    assert not list(tmp_path.glob("*.tmp.*"))
    with np.load(output_path) as saved:
        assert saved["t2max"].shape == (1, 3, 2, 2)
        assert np.all(np.isnan(saved["t2max"][:, :, 0, 1]))
        assert saved["init_time_index"].item() == 123
        assert saved["init_date"].item() == "20010701"


def test_ens_score_configures_extended_global_fields_before_loading_stats(monkeypatch):
    calls = []
    monkeypatch.setattr(ens_score.cfm, "apply_extended_global_fields", lambda: calls.append("extended"))
    ens_score.configure_fold(2, tuple(range(15, 29)), 5)
    assert calls == ["extended"]
    assert ens_score.cfm.Config.CV_FOLD == 2
    assert ens_score.cfm.Config.CV_TEST_OFFSETS == (2,)
    assert ens_score.cfm.Config.CV_VAL_OFFSETS == (3,)
    assert ens_score.cfm.Config.PREDICTION_LEADS == tuple(range(15, 29))


def test_ingest_resume_validator_rejects_corrupt_and_accepts_valid_metadata(tmp_path: Path):
    corrupt = tmp_path / "init_20010701.npz"
    corrupt.write_bytes(b"partial output")
    valid, reason = validate_ingested_output(corrupt, (1, 2, 3), expected_members=2)
    assert not valid
    assert "BadZipFile" in reason

    complete = tmp_path / "init_20010702.npz"
    np.savez_compressed(
        complete,
        t2max=np.zeros((2, 3, 1, 1), dtype=np.float32),
        leads=np.array([1, 2, 3], dtype=np.int16),
        members=np.array([0, 1], dtype=np.int16),
        init_date=np.array("20010702"),
        init_time_index=np.array(456, dtype=np.int32),
        variable=np.array("mx2t6"),
    )
    valid, reason = validate_ingested_output(
        complete,
        (1, 2, 3),
        expected_members=2,
        expected_label="20010702",
        expected_init_time_index=456,
        expected_variable="mx2t6",
    )
    assert valid, reason


def test_ens_score_reports_all_invalid_ingested_outputs(tmp_path: Path):
    (tmp_path / "init_20010701.npz").write_bytes(b"partial one")
    (tmp_path / "init_20010702.npz").write_bytes(b"partial two")
    try:
        ens_score.load_ingested_files(tmp_path, (15, 16))
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected invalid ENS outputs to stop scoring.")
    assert "Found 2 invalid ingested ENS outputs" in message
    assert "init_20010701.npz" in message
    assert "init_20010702.npz" in message
    assert "Rerun submit_ens_ingest.slurm" in message


def test_ens_quantile_mapping_requires_only_observed_valid_target_months():
    months = np.array([5] * 20 + [6] * 30 + [7] * 30 + [8] * 30 + [9] * 30, dtype=np.int16)
    files_by_init = {
        10: Path("early.npz"),
        45: Path("middle.npz"),
        105: Path("late.npz"),
    }
    required = ens_score.required_target_months_by_lead(files_by_init, months, (15, 28))
    assert required[15] == (6, 7, 9)
    assert required[28] == (6, 7, 9)
    assert 5 not in required[15]
    assert 5 not in required[28]


def test_ens_score_submission_runs_bounded_parallel_folds_before_compare():
    script = (Path(__file__).resolve().parents[1] / "submit_ens_score_compare.slurm").read_text(
        encoding="utf-8"
    )
    assert "FOLD_WORKERS=${FOLD_WORKERS:-2}" in script
    assert 'score_fold "$FOLD" &' in script
    assert "wait_for_fold_batch()" in script
    assert 'if [ "${#PIDS[@]}" -ge "$FOLD_WORKERS" ]; then' in script
    assert script.index("All ENS folds complete; starting pooled comparison") < script.index(
        '"$PY" -u ens_compare.py'
    )


def test_ens_score_lightweight_loader_uses_only_disk_heat_and_time_cache(tmp_path: Path):
    cache_dir = tmp_path / "data_cache"
    cache_dir.mkdir()
    heat = np.zeros((2, 3, 4), dtype=np.float32)
    time_values = np.arange(4, dtype=np.float64)
    np.save(cache_dir / "heat_index.npy", heat)
    np.save(cache_dir / "time_values.npy", time_values)
    config = SimpleNamespace(OUTPUT_DIR=str(tmp_path))
    shared = ens_score.load_ens_scoring_shared_data(config)
    assert set(shared) == {"heat_index", "time_values"}
    assert isinstance(shared["heat_index"], np.memmap)
    assert isinstance(shared["time_values"], np.memmap)
    assert shared["heat_index"].shape == heat.shape
    assert shared["time_values"].shape == time_values.shape
