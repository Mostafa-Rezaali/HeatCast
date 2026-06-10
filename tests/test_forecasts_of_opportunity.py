from pathlib import Path

import numpy as np

from forecasts_of_opportunity import (
    OpportunityStats,
    confidence_selections,
    fit_boundaries,
    holm_adjust_pvalues,
    paired_year_block_bootstrap_interactions,
    retained_percent_from_confidence_stratum,
    roc_auc_from_hist,
    stream_chunks,
    year_block_bootstrap,
)


class IdentityModel:
    def predict_features(self, features):
        return np.asarray(features[:, 0], dtype=np.float32)


class SentinelMapping(dict):
    def __getitem__(self, key):
        if key == "test_sentinel":
            raise AssertionError("Boundary fitting touched test data.")
        return super().__getitem__(key)


def test_streamed_metrics_and_confidence_assignment():
    probability = np.array([0.1, 0.2, 0.8, 0.9])
    truth = np.array([0, 0, 1, 1])
    base = np.full(4, 0.5)
    stats = OpportunityStats()
    stats.update(probability, truth, base)
    metrics = stats.metrics()
    expected_brier = np.mean((probability - truth) ** 2)
    assert np.isclose(metrics["brier"], expected_brier)
    assert np.isclose(metrics["bss_unconditional"], 1.0 - expected_brier / 0.25)
    assert np.isclose(metrics["roc_auc"], 1.0)
    selections = dict(confidence_selections(np.array([0.1, 0.2, 0.3]), [90], [0.25]))
    assert selections["top_10pct_ge_p90"].tolist() == [False, False, True]


def test_conditional_bss_distinguishes_composition_from_skill():
    truth = np.array([1] * 8 + [0] * 2, dtype=float)
    probability = np.full(10, 0.8)
    stored_base = np.full(10, 0.05)
    stats = OpportunityStats()
    stats.update(probability, truth, stored_base)
    metrics = stats.metrics()
    assert metrics["bss_unconditional"] > 0.0
    assert np.isclose(metrics["bss_conditional"], 0.0)


def test_fit_boundaries_uses_only_calibration_mapping():
    calibration = SentinelMapping(
        init_margin=np.linspace(0.01, 0.99, 100),
        forecast_margin=np.linspace(-1.0, 1.0, 100),
        model_sigma=np.linspace(0.1, 1.0, 100),
        base_rate=np.full(100, 0.05),
        test_sentinel=np.array([1.0]),
    )
    boundaries = fit_boundaries(calibration, IdentityModel(), [50, 90])
    assert boundaries["confidence_edges"].shape == (2,)
    assert boundaries["sigma_edges"].shape == (9,)
    assert boundaries["forecast_margin_edges"].shape == (9,)


def test_year_bootstrap_is_reproducible():
    by_year = {}
    for year, probability in [(2000, 0.2), (2001, 0.8), (2002, 0.6)]:
        for stratum in ("all", "top_10pct_ge_p90"):
            stats = OpportunityStats()
            stats.update(
                np.array([probability, probability]),
                np.array([0.0, 1.0]),
                np.array([0.5, 0.5]),
            )
            by_year[("confidence", stratum, year)] = stats
    first = year_block_bootstrap(
        by_year, ["all", "top_10pct_ge_p90"], "top_10pct_ge_p90",
        [2000, 2001, 2002], reps=50, seed=42,
    )
    second = year_block_bootstrap(
        by_year, ["all", "top_10pct_ge_p90"], "top_10pct_ge_p90",
        [2000, 2001, 2002], reps=50, seed=42,
    )
    assert first == second


def test_paired_interaction_bootstrap_compares_both_parents():
    years = list(range(2000, 2008))
    top_stratum = "top_10pct_ge_p90"
    parent_keys = (
        ("confidence", top_stratum),
        ("low_sigma", "bottom_sigma_tercile"),
        ("mjo_phase", "phase_8"),
    )
    interaction_keys = (
        ("mjo_phase_x_top_confidence", f"phase_8__{top_stratum}"),
        ("mjo_phase_x_low_sigma", "phase_8__bottom_sigma_tercile"),
    )
    global_stats = {}
    by_year = {}
    truth = np.array([0.0, 1.0] * 50)
    base_rate = np.full(truth.shape, 0.5)
    parent_probability = np.full(truth.shape, 0.5)
    interaction_probability = np.where(truth > 0.5, 0.9, 0.1)
    for key in parent_keys + interaction_keys:
        global_stats[key] = OpportunityStats()
    for year in years:
        for key in parent_keys:
            stats = OpportunityStats()
            stats.update(parent_probability, truth, base_rate)
            global_stats[key].add(stats)
            by_year[(key[0], key[1], year)] = stats
        for key in interaction_keys:
            stats = OpportunityStats()
            stats.update(interaction_probability, truth, base_rate)
            global_stats[key].add(stats)
            by_year[(key[0], key[1], year)] = stats
    rows = paired_year_block_bootstrap_interactions(
        global_stats, by_year, years, top_stratum, reps=100, seed=42
    )
    required = {
        (axis, stratum, parent_kind)
        for axis, stratum in interaction_keys
        for parent_kind in ("selection_parent", "driver_parent")
    }
    unconditional = {
        (row["interaction_axis"], row["interaction_stratum"], row["parent_kind"]): row
        for row in rows
        if row["metric"] == "bss_unconditional"
    }
    assert required == set(unconditional)
    assert all(row["delta_ci_low"] > 0.0 for row in unconditional.values())
    assert all(np.isfinite(row["p_holm_mjo"]) for row in unconditional.values())


def test_holm_adjustment_is_step_down_and_never_smaller():
    raw = {"a": 0.01, "b": 0.02, "c": 0.20}
    adjusted = holm_adjust_pvalues(raw)
    assert adjusted == {"a": 0.03, "b": 0.04, "c": 0.20}
    assert all(adjusted[key] >= value for key, value in raw.items())


def test_histogram_auc_matches_known_separation():
    pos = np.zeros(8)
    neg = np.zeros(8)
    pos[-1] = 10
    neg[0] = 10
    assert np.isclose(roc_auc_from_hist(pos, neg), 1.0)


def test_confidence_stratum_retained_percent_parser():
    assert retained_percent_from_confidence_stratum("top_10pct_ge_p90") == 10
    assert retained_percent_from_confidence_stratum("top_1pct_ge_p99") == 1


def test_tiny_fake_fold_streams_three_full_land_chunks(tmp_path: Path):
    chunk_paths = []
    for index, year in enumerate((2000, 2001, 2002)):
        path = tmp_path / f"sample_{index:05d}.npz"
        probability = np.linspace(0.01, 0.99, 100, dtype=np.float32)
        truth = (probability > 0.5).astype(np.uint8)
        np.savez_compressed(
            path,
            init_margin=probability,
            forecast_margin=np.zeros(100, dtype=np.float32),
            model_sigma=np.ones(100, dtype=np.float32),
            truth=truth,
            base_rate=np.full(100, 0.5, dtype=np.float32),
            year=np.array(year, dtype=np.int16),
            month=np.array(7, dtype=np.int8),
            source_fold=np.array(0, dtype=np.int16),
        )
        chunk_paths.append(path)
    manifest = {"source_fold": 0, "test_years": {2000, 2001, 2002}}
    boundaries = {
        "confidence_edges": np.array([0.25]),
        "sigma_edges": np.linspace(0.1, 0.9, 9),
        "forecast_margin_edges": np.linspace(0.1, 0.9, 9),
    }
    regions = {"Synthetic": np.ones(100, dtype=bool)}
    global_stats, _, years = stream_chunks(
        [(manifest, IdentityModel(), boundaries, chunk_paths)],
        land_count=100,
        region_land_masks=regions,
        confidence_percentiles=[90],
        progress_every=1000,
    )
    assert years == {2000, 2001, 2002}
    metrics = global_stats[("confidence", "all")].metrics()
    assert metrics["n"] == 300
    assert metrics["roc_auc"] > 0.99
