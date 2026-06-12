from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from ens_common import (
    apply_quantile_mapping,
    common_init_indices,
    fit_quantile_mapping,
    intersection_years,
    member_fraction_probability,
)
from download_ecmwf_s2s import hindcast_dates, mjjas_mon_thu, retrieve
from ens_ingest import load_init_list, load_native_daily_max
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
                    {"step": steps, "latitude": lat, "longitude": lon},
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
