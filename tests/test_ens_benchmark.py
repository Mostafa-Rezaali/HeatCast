from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np

import ens_ingest as ingest
import ens_compare
import ens_score
import build_driver_tables
import forecasts_of_opportunity
from ens_common import (
    apply_quantile_mapping,
    common_init_indices,
    fit_quantile_mapping,
    intersection_years,
    member_fraction_probability,
)
from download_ecmwf_s2s import hindcast_dates, mjjas_mon_thu, parse_year_list, retrieve
from ens_ingest import find_raw_file, load_init_list, load_native_daily_max, normalize_rt_tag, validate_ingested_output
from stitch_exceedance_folds import load_fold_inputs
import build_paper_figures_extended as paper_ext
from figure_style import SINGLE_COLUMN_MM, figure_size, save_figure, setup_matplotlib


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
    assert parse_year_list("2022,2024,2022") == (2022, 2024)
    assert normalize_rt_tag("2024") == "rt2024"
    legacy = tmp_path / "ens_init_20020502.grib"
    tagged = tmp_path / "ens_init_20020502_rt2024.grib"
    legacy.touch()
    tagged.touch()
    assert find_raw_file(tmp_path, "20020502") == legacy
    assert find_raw_file(tmp_path, "20020502", "rt2024") == tagged


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
        "rt2024",
    ) == "20010701"
    assert output_path.exists()
    assert not list(tmp_path.glob("*.tmp.*"))
    with np.load(output_path) as saved:
        assert saved["t2max"].shape == (1, 3, 2, 2)
        assert np.all(np.isnan(saved["t2max"][:, :, 0, 1]))
        assert saved["init_time_index"].item() == 123
        assert saved["init_date"].item() == "20010701"
        assert saved["rt_tag"].item() == "rt2024"


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
        rt_tag=np.array("rt2024"),
    )
    valid, reason = validate_ingested_output(
        complete,
        (1, 2, 3),
        expected_members=2,
        expected_label="20010702",
        expected_init_time_index=456,
        expected_variable="mx2t6",
        expected_rt_tag="rt2024",
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
    assert "Rerun slurm/submit_ens_widen_cycles.slurm" in message


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


def test_ens_quantile_mapping_cache_is_keyed_by_exact_training_init_set():
    source = (Path(__file__).resolve().parents[1] / "src" / "ens_score.py").read_text(encoding="utf-8")
    assert 'mapping_init_indices=mapping_init_indices' in source
    assert '"mapping_init_indices" in data.files' in source
    assert 'np.array_equal(' in source
    assert 'data["mapping_init_indices"]' in source


def test_cycle_probabilities_merge_duplicates_without_double_counting():
    chunks = [
        {
            "init_margin": np.array([0.2, 0.4], dtype=np.float32),
            "forecast_margin": np.array([0.0, 1.0], dtype=np.float32),
        },
        {
            "init_margin": np.array([0.4, 0.8], dtype=np.float32),
            "forecast_margin": np.array([1.0, 0.0], dtype=np.float32),
        },
    ]
    raw, calibrated = ens_compare.merge_cycle_probabilities(chunks)
    assert np.allclose(raw, np.array([0.3, 0.6], dtype=np.float32))
    expected = 0.5 * (ens_compare.sigmoid(chunks[0]["forecast_margin"]) + ens_compare.sigmoid(chunks[1]["forecast_margin"]))
    assert np.allclose(calibrated, expected)


def test_cycle_run_templates_expand_per_fold():
    groups = ens_compare.resolve_ens_run_groups(
        ("cvfold{F}_ens_w34", "cvfold{F}_ens_w34_rt2024"),
        tuple(f"cvfold{fold}_heatcast" for fold in range(5)),
        Path("unused"),
        tuple(range(15, 29)),
    )
    assert groups[3] == ["cvfold3_ens_w34", "cvfold3_ens_w34_rt2024"]
    assert set(groups) == set(range(5))
    assert ens_compare.cycle_label(groups[3][0]) == "legacy"
    assert ens_compare.cycle_label(groups[3][1]) == "rt2024"
    union, overlap = ens_compare.cycle_year_union({
        "legacy": (1, 2, 3),
        "rt2024": (3, 4, 5),
    })
    assert union == {1, 2, 3, 4, 5}
    assert overlap == {3}


def test_cycle_widen_submission_rescores_legacy_and_merges_by_year():
    script = (Path(__file__).resolve().parents[1] / "slurm" / "submit_ens_widen_cycles.slurm").read_text(
        encoding="utf-8"
    )
    assert "INGEST_WORKERS=${INGEST_WORKERS:-16}" in script
    assert '--workers "$INGEST_WORKERS"' in script
    assert "WINDOW_LABEL=window_${LEADS//,/-}" in script
    assert "ens_score_metadata.json" in script
    assert "incremental_arrays/manifest.npz" in script
    assert "Skipping complete ENS score" in script
    assert "run_cycle \"\"" in script
    assert "run_cycle rt2024" in script
    assert "cvfold{F}_ens_w34,cvfold{F}_ens_w34_rt2024" in script
    assert "--emit_per_year" in script


def test_heatcast_ens_stack_opportunity_is_cross_fitted_and_paired():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src" / "ens_heatcast_stack_opportunity.py").read_text(encoding="utf-8")
    script = (root / "slurm" / "submit_ens_stack_opportunity.slurm").read_text(encoding="utf-8")
    assert "heatcast_ens_stack" in source
    assert "crossfit_excluding_fold" in source
    assert "if int(other) != int(fold)" in source
    assert "paired_chunk(" in source
    assert "merge_cycle_probabilities(ens_chunks)" in source
    assert "init_time_index" in source
    assert "opportunity_pair_bootstrap.csv" in source
    assert "heatcast_top10_confidence" in source
    assert "ThreadPoolExecutor(max_workers=fold_workers)" in source
    assert "--fold_workers" in source
    assert "robustness_by_month.csv" in source
    assert "robustness_by_region.csv" in source
    assert "robustness_leave_one_out.csv" in source
    assert "leave-one-" in source
    assert "--driver_table_dir" in source
    assert "driver_pair_bootstrap.csv" in source
    assert "driver_pair_parent_bootstrap.csv" in source
    assert "Paired driver-stratified Stack-vs-ENS tests" in source
    assert "generic_teleconnection" in source
    assert "alldata" in source
    assert "driver_family" in source
    assert "--mem=500G" in script
    assert "--gres=gpu" not in script
    assert "module load cuda" not in script
    assert "--partition=hpg-b200" not in script
    assert "cvfold{F}_ens_w34,cvfold{F}_ens_w34_rt2024" in script
    assert "--max_stack_samples_per_fold 500000" in script
    assert "FOLD_WORKERS=${FOLD_WORKERS:-5}" in script
    assert "DRIVER_ARGS=()" in script
    assert "DRIVER_TABLE_DIR" in script
    assert "data_cache/slow_driver_tables_w34_alldata" in script
    assert "Missing driver table" in script
    assert '--fold_workers "$FOLD_WORKERS"' in script
    assert "OMP_NUM_THREADS=1" in script


def test_paper_evidence_blocks_are_cpu_only_and_cover_required_sections():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src" / "build_paper_evidence_blocks.py").read_text(encoding="utf-8")
    script = (root / "slurm" / "submit_paper_evidence_blocks.slurm").read_text(encoding="utf-8")
    assert "mechanism_block.csv" in source
    assert "robustness_block.csv" in source
    assert "operational_block.csv" in source
    assert "paper_evidence_summary.md" in source
    assert "MJO/ENSO/soil" in source
    assert "paired_stack_vs_ens_driver" in source
    assert "driver_pair_parent_bootstrap.csv" in source
    assert "Stack minus ENS delta BSS" in source
    assert "--gres=gpu" not in script
    assert "module load cuda" not in script
    assert "--partition=hpg-b200" not in script
    assert "--mem=16G" in script
    assert "build_paper_evidence_blocks.py" in script


def test_teleconnection_stack_submission_is_cpu_only_and_explicit():
    root = Path(__file__).resolve().parents[1]
    script = (root / "slurm" / "submit_teleconnection_stack_analysis.slurm").read_text(encoding="utf-8")
    assert "--mem=500G" in script
    assert "--gres=gpu" not in script
    assert "module load cuda" not in script
    assert "--partition=hpg-b200" not in script
    assert "TELECONNECTION_INDEX_PATHS=${TELECONNECTION_INDEX_PATHS:?" in script
    assert "data_cache/slow_driver_tables_w34_teleconnections" in script
    assert "ens_heatcast_stack_opportunity_teleconnections" in script
    assert "--teleconnection_index_paths \"$TELECONNECTION_INDEX_PATHS\"" in script
    assert "--driver_table_dir \"$DRIVER_DIR\"" in script
    assert "TELECONNECTION STACK ANALYSIS COMPLETE" in script


def test_paper_figures_tables_package_is_cpu_only_and_records_claim_boundaries():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src" / "build_paper_figures_tables.py").read_text(encoding="utf-8")
    script = (root / "slurm" / "submit_paper_figures_journal.slurm").read_text(encoding="utf-8")
    assert "figure_1_headline_skill" in source
    assert "figure_2_headline_stack_minus_ens_ci" in source
    assert "figure_3_robustness" in source
    assert "figure_4_opportunity_and_driver_tests" in source
    assert "figure_5_probabilistic_scorecard" in source
    assert "figure_6_probability_threshold_operating_curves" in source
    assert "figure_7_opportunity_probability_metrics" in source
    assert "table_7_probability_threshold_operating_points" in source
    assert "table_8_opportunity_probability_metrics" in source
    assert "hit_rate_0.2" in source
    assert "false_alarm_ratio_0.2" in source
    assert "methods_text_draft.md" in source
    assert "narrative_and_claim_boundaries.md" in source
    assert "investigation_record.md" in source
    assert "reproducibility_manifest.json" in source
    assert "Do not say HeatCast alone beats ENS." in source
    assert "--gres=gpu" not in script
    assert "module load cuda" not in script
    assert "--partition=hpg-b200" not in script
    assert "--mem=64G" in script
    assert "build_paper_figures_tables.py" in script


def test_extended_paper_murphy_decomposition_reconstructs_brier():
    prob = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    truth = np.array([0, 0, 1, 1], dtype=np.float32)
    dec = paper_ext.murphy_decomposition(prob, truth, n_bins=2)
    assert abs(dec["brier"] - np.mean((prob - truth) ** 2)) < 1e-8
    assert abs(dec["murphy_reconstructed_brier"] - dec["brier"]) < 0.03
    assert dec["resolution"] > 0


def test_extended_paper_per_cell_auc_and_year_fold_map(tmp_path: Path):
    prob = np.array([[0.1, 0.7], [0.4, 0.2], [0.8, 0.9], [0.9, 0.1]], dtype=np.float32)
    truth = np.array([[0, 1], [0, 0], [1, 1], [1, 0]], dtype=np.float32)
    auc = paper_ext.auc_per_cell(prob, truth)
    assert np.allclose(auc, np.array([1.0, 1.0], dtype=np.float32))

    stack_dir = tmp_path / "stack"
    stack_dir.mkdir()
    paper_ext.write_csv(
        stack_dir / "heatcast_ens_stack_head_to_head.csv",
        [
            {"section": "coverage", "fold": 0, "intersection_years": "2001 2006"},
            {"section": "coverage", "fold": 2, "intersection_years": "2003 2008"},
        ],
    )
    assert paper_ext.year_to_fold_from_head_to_head(stack_dir) == {2001: 0, 2006: 0, 2003: 2, 2008: 2}


def test_extended_paper_flexible_ens_run_resolver_allows_mixed_templates(tmp_path: Path):
    for fold in range(2):
        root = tmp_path / f"cvfold{fold}_explicit" / "test" / "window_15-16"
        array_dir = root / "incremental_arrays"
        chunk_dir = array_dir / "test_chunks"
        chunk_dir.mkdir(parents=True)
        np.savez_compressed(
            array_dir / "manifest.npz",
            run_name=np.array(f"cvfold{fold}_explicit"),
            source_fold=np.array(fold, dtype=np.int16),
            target_mode=np.array("window"),
            window_leads=np.array([15, 16], dtype=np.int16),
            train_years=np.array([2000], dtype=np.int16),
            calibration_years=np.array([2001], dtype=np.int16),
            test_years=np.array([2002 + fold], dtype=np.int16),
            calibration_split=np.array("val"),
            eval_split=np.array("test"),
            sample_count=np.array(0, dtype=np.int32),
            valid_cell_count=np.array(0, dtype=np.int64),
        )
        np.savez_compressed(
            array_dir / "calibration_pairs.npz",
            init_margin=np.empty(0, dtype=np.float32),
            forecast_margin=np.empty(0, dtype=np.float32),
            model_sigma=np.empty(0, dtype=np.float32),
            truth=np.empty(0, dtype=np.float32),
            base_rate=np.empty(0, dtype=np.float32),
            year=np.empty(0, dtype=np.int16),
            source_fold=np.array(fold, dtype=np.int16),
        )
    groups = paper_ext.resolve_ens_run_groups_flexible(
        ("cvfold{F}_templated", "cvfold0_explicit", "cvfold1_explicit"),
        ("heat0", "heat1"),
        tmp_path,
        (15, 16),
    )
    assert groups[0] == ["cvfold0_templated", "cvfold0_explicit"]
    assert groups[1] == ["cvfold1_templated", "cvfold1_explicit"]


def test_extended_paper_submission_is_cpu_only_and_auditable():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src" / "build_paper_figures_extended.py").read_text(encoding="utf-8")
    script = (root / "slurm" / "submit_paper_figures_journal.slurm").read_text(encoding="utf-8")
    assert "figure_5_spatial_skill" in source
    assert "figure_6_reliability_decomposition" in source
    assert "figure_7_case_studies" in source
    assert "figure_8_per_lead_profile" in source
    assert "figure_9_teleconnection_ranking" in source
    assert "figure_9_opportunity_discard_curve" in source
    assert "figure_10a_leave_one_out_robustness" in source
    assert "table_7_stack_ablation_probability" in source
    assert "table_8_per_year_head_to_head" in source
    assert "table_9_teleconnection_ranking" in source
    assert "table_10_computational_cost_comparison" in source
    assert "reproducibility_manifest.json" in source
    assert "--gres=gpu" not in script
    assert "module load cuda" not in script
    assert "--mem=64G" in script
    assert "ENS_RUNS=${ENS_RUNS:-cvfold{F}" not in script
    assert "ENS_RUNS='cvfold{F}_ens_w34,cvfold{F}_ens_w34_rt2024'" in script
    assert "repo_integrity.py" in script


def test_journal_figure_style_contract_and_submission_wrapper():
    root = Path(__file__).resolve().parents[1]
    style = (root / "src" / "figure_style.py").read_text(encoding="utf-8")
    tables = (root / "src" / "build_paper_figures_tables.py").read_text(encoding="utf-8")
    extended = (root / "src" / "build_paper_figures_extended.py").read_text(encoding="utf-8")
    script = (root / "slurm" / "submit_paper_figures_journal.slurm").read_text(encoding="utf-8")
    assert "SYSTEM_COLORS" in style
    assert "save_figure" in style
    assert "svg.fonttype" in style and '"none"' in style
    assert "from figure_style import" in tables
    assert "from figure_style import" in extended
    for system_hex in ("#0072B2", "#D55E00", "#009E73", "#6F6F6F"):
        assert system_hex not in tables
        assert system_hex not in extended
    assert "--gres=gpu" not in script
    assert "module load cuda" not in script
    assert "--mem=64G" in script
    assert "submit_paper_figures_journal" in script or "paper_fig_journal" in script
    assert "figure_style.py" in script
    assert "--w34_log_glob" in script


def test_journal_style_svg_smoke_text_is_selectable_and_legible(tmp_path: Path):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=figure_size(SINGLE_COLUMN_MM, 50.0))
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("Probability")
    ax.set_ylabel("Observed frequency")
    save_figure(fig, tmp_path / "smoke")
    plt.close(fig)
    svg = (tmp_path / "smoke.svg").read_text(encoding="utf-8")
    assert "<text" in svg
    assert "<path" in svg
    sizes = [float(value) for value in __import__("re").findall(r"font-size:\s*([0-9.]+)px", svg)]
    assert not sizes or min(sizes) >= 5.0
    assert (tmp_path / "smoke.png").exists()


def test_generic_teleconnection_indices_parse_and_emit_strata(tmp_path: Path):
    pna_path = tmp_path / "pna.txt"
    pna_path.write_text(
        "1981 0.7 0.1 -0.2 0.0 1.0 -0.8 0.2 0.3 0.4 0.5 0.6 0.7\n",
        encoding="utf-8",
    )
    parsed = build_driver_tables.parse_monthly_index_file(pna_path)
    assert parsed[(1981, 1)] == 0.7
    assert parsed[(1981, 6)] == -0.8
    names = build_driver_tables.parse_named_index_paths(f"pna={pna_path},AO={pna_path}")
    assert set(names) == {"pna", "ao"}

    lookup = forecasts_of_opportunity.DriverLookup(
        mjo_phase=np.array([1, 2], dtype=np.int8),
        mjo_amplitude=np.array([1.2, 0.5], dtype=np.float32),
        nino34=np.array([0.0, 0.8], dtype=np.float32),
        teleconnection_names=("pna",),
        teleconnection_values=np.array([[0.7, -0.9]], dtype=np.float32),
        teleconnection_threshold=0.5,
        alldata_names=("ridge_index",),
        alldata_values=np.array([[-0.8, 0.2]], dtype=np.float32),
        alldata_threshold=0.5,
        sidecars={0: {}},
        soil_rows={0: {0: 0}},
        soil_memmaps={0: np.asarray([[10.0, 50.0, 90.0]], dtype=np.float16)},
    )
    chunk_path = tmp_path / "chunk_000000.npz"
    np.savez_compressed(chunk_path, init_time_index=np.array(0, dtype=np.int32))
    with np.load(chunk_path, allow_pickle=False) as data:
        _, strata = lookup.sample_strata(0, chunk_path, data)
    assert strata["tele_pna"]["positive"].tolist() == [True, True, True]
    assert strata["alldata_ridge_index"]["negative"].tolist() == [True, True, True]
    assert set(strata) >= {"mjo_phase", "enso_state", "soil_moisture_tercile", "tele_pna", "alldata_ridge_index"}


def _write_minimal_xlsx(path: Path) -> None:
    sheet_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1">
      <c r="A1" t="inlineStr"><is><t>Date</t></is></c>
      <c r="B1" t="inlineStr"><is><t>PNA like</t></is></c>
      <c r="C1" t="inlineStr"><is><t>Noise</t></is></c>
    </row>
    <row r="2"><c r="A2"><v>29707</v></c><c r="B2"><v>0.8</v></c><c r="C2"><v>1.0</v></c></row>
    <row r="3"><c r="A3"><v>29708</v></c><c r="B3"><v>-0.6</v></c><c r="C3"><v>2.0</v></c></row>
  </sheetData>
</worksheet>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"/>")
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def test_alldata_xlsx_drivers_align_without_openpyxl(tmp_path: Path):
    workbook = tmp_path / "AllData.xlsx"
    _write_minimal_xlsx(workbook)
    names, values = build_driver_tables.build_alldata_drivers(
        [0.0, 1.0],
        workbook,
        ("PNA like",),
    )
    assert names == ("pna_like",)
    assert values.shape == (1, 2)
    assert np.allclose(values[0], [0.8, -0.6])


def test_extended_paper_parses_w34_per_lead_logs(tmp_path: Path):
    leads = tuple(range(15, 29))
    parts0 = ", ".join(f"+{lead}:TAC={0.10 + lead / 1000:.3f}/MSE={0.40 + lead / 1000:.3f}" for lead in leads)
    parts1 = ", ".join(f"+{lead}:TAC={0.20 + lead / 1000:.3f}/MSE={0.50 + lead / 1000:.3f}" for lead in leads)
    (tmp_path / "w34_tube_all_1.log").write_text(
        "\n".join([
            "Run name: cvfold0_w34_dist_v1",
            f"  Per-lead diagnostics: {parts0}",
            "Run name: cvfold1_w34_dist_v1",
            f"  Per-lead diagnostics: {parts1}",
        ]),
        encoding="utf-8",
    )
    rows, detail = paper_ext.parse_w34_per_lead_logs(tmp_path, "w34_tube_all_*.log", leads)
    assert len(rows) == 14
    assert len(detail) == 28
    assert all(np.isfinite(float(row["tac_mean"])) for row in rows)
    assert rows[0]["lead"] == 15
    assert rows[-1]["lead"] == 28
    assert rows[0]["n_folds"] == 2


def test_extended_table8_replaces_fold2_sensitivity_with_base_rate_rows(tmp_path: Path):
    stack_dir = tmp_path / "stack"
    table_dir = tmp_path / "tables"
    stack_dir.mkdir()
    paper_ext.write_csv(
        stack_dir / "heatcast_ens_stack_head_to_head.csv",
        [{"section": "coverage", "fold": 0, "intersection_years": "2004 2014 2020"}],
    )
    rows = []
    for year, base_rate, ens_brier, stack_brier in (
        (2004, 0.005, 0.10, 0.11),
        (2014, 0.006, 0.09, 0.08),
        (2020, 0.060, 0.08, 0.06),
    ):
        rows.extend([
            {"year": year, "model": paper_ext.REFERENCE_MODEL, "brier": 0.12, "valid_count": 10, "bss_vs_monthly_climo": 0.0, "roc_auc": 0.5, "base_rate": base_rate},
            {"year": year, "model": paper_ext.ENS_MODEL, "brier": ens_brier, "valid_count": 10, "bss_vs_monthly_climo": 1 - ens_brier / 0.12, "roc_auc": 0.6},
            {"year": year, "model": paper_ext.STACK_MODEL, "brier": stack_brier, "valid_count": 10, "bss_vs_monthly_climo": 1 - stack_brier / 0.12, "roc_auc": 0.7},
        ])
    paper_ext.write_csv(stack_dir / "robustness_by_year.csv", rows)
    sources = {}
    paper_ext.write_table_8(stack_dir, table_dir, sources)
    out_rows = paper_ext.read_csv(table_dir / "table_8_per_year_head_to_head.csv")
    labels = {str(row["year"]) for row in out_rows}
    assert "fold2_removed" not in labels
    assert "median_year_delta" in labels
    assert "two_lowest_base_rate_years_removed" in labels
    assert any(row.get("sign_test_wins") == "2/3" for row in out_rows)


def test_s2s_downloader_uses_bounded_parallel_atomic_retrievals():
    source = (Path(__file__).resolve().parents[1] / "src" / "download_ecmwf_s2s.py").read_text(
        encoding="utf-8"
    )
    assert "ThreadPoolExecutor(max_workers=int(args.workers))" in source
    assert 'parser.add_argument(' in source and '"--workers"' in source
    assert 'partial = target.with_suffix(target.suffix + ".part")' in source
    assert "partial.replace(target)" in source
    assert "if not valid_grib(partial):" in source


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
