#!/usr/bin/env python3
"""Fast, data-free repository contract audit for HeatCast.

This audit protects experiment intent and submission-script consistency. It is
deliberately independent of the external NetCDF datasets and GPU runtime.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


W34_LEADS = tuple(range(15, 29))
MJJAS_MONTHS = (5, 6, 7, 8, 9)
EMAIL = "mostafarezaali@ufl.edu"
CURRENT_SUBMISSIONS = (
    "submit_ens_stack_opportunity.slurm",
    "submit_ens_widen_cycles.slurm",
    "submit_export_w34_stack_netcdf.slurm",
    "submit_paper_evidence_blocks.slurm",
    "submit_paper_figures_journal.slurm",
    "submit_teleconnection_stack_analysis.slurm",
    "submit_w34_eval_stitch.slurm",
    "submit_w34_tube_all.slurm",
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def _text(root: Path, relative: str) -> str:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required repository file is missing: {relative}")
    return path.read_text(encoding="utf-8")


def _contains_all(text: str, tokens: Iterable[str]) -> tuple[bool, list[str]]:
    missing = [token for token in tokens if token not in text]
    return not missing, missing


def _shell_csv_variable(text: str, name: str) -> tuple[str, ...]:
    match = re.search(rf"(?m)^{re.escape(name)}=([^\n]+)$", text)
    if match is None:
        return ()
    value = match.group(1).strip().strip("\"'")
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _result(name: str, passed: bool, detail: str) -> CheckResult:
    return CheckResult(name=name, passed=bool(passed), detail=detail)


def _required_tokens_check(
    root: Path,
    name: str,
    relative: str,
    tokens: Iterable[str],
) -> CheckResult:
    text = _text(root, relative)
    passed, missing = _contains_all(text, tokens)
    detail = f"{relative}: all required tokens present" if passed else f"{relative}: missing {missing}"
    return _result(name, passed, detail)


def audit_repository(root: Path) -> list[CheckResult]:
    """Return all fast repository-contract checks."""
    root = root.resolve()
    results: list[CheckResult] = []

    cfm = _text(root, "cfm_mesh_train.py")
    exceed = _text(root, "exceedance_eval.py")
    ens_ingest = _text(root, "ens_ingest.py")
    ens_score = _text(root, "ens_score.py")
    mode = _text(root, "mode_dispatch.py")
    mesh = _text(root, "mesh_backbone.py")
    w34_train = _text(root, "submit_w34_tube_all.slurm")
    w34_eval = _text(root, "submit_w34_eval_stitch.slurm")

    month_literal = "MJJAS_MONTHS = (5, 6, 7, 8, 9)"
    results.append(_result(
        "target.month_specific_daily_exceedance",
        month_literal in cfm and month_literal in exceed
        and "def build_month_q95" in exceed
        and "truth_z > q95_z" not in exceed
        and "(field_z[valid] > threshold[valid])" in exceed,
        "MJJAS month-specific q95 builders and strict daily exceedance labels are present",
    ))

    results.append(_result(
        "evaluation.fold_safe_guards",
        all(token in exceed for token in (
            "Calibration/eval year overlap before fitting calibrator.",
            "Leakage check failed: evaluation split overlaps training years.",
            "Disjointness assert failed: calibration split",
            "Leakage assert failed: evaluation target year was in train years.",
        )),
        "Training, calibration, and evaluation overlap guards are present",
    ))

    results.append(_result(
        "distributional.mean_sigma_semantics",
        all(token in mode for token in (
            "mean = persistence + mean_raw",
            "sigma = F.softplus(sigma_raw) + float(floor)",
            "def gaussian_crps(",
        )),
        "Distributional mean uses persistence residual and sigma uses positive softplus floor",
    ))

    results.append(_result(
        "distributional.grid_refiner_mean_only",
        "mean_raw + self.grid_refiner(mean_raw)" in mesh
        and "torch.cat([mean_raw + self.grid_refiner(mean_raw), var_raw], dim=1)" in mesh,
        "Grid refiner is applied to the distributional mean while variance bypasses it",
    ))

    train_leads = tuple(int(value) for value in _shell_csv_variable(w34_train, "LEADS"))
    eval_leads = tuple(int(value) for value in _shell_csv_variable(w34_eval, "LEADS"))
    results.append(_result(
        "w34.identical_train_eval_leads",
        train_leads == W34_LEADS and eval_leads == W34_LEADS,
        f"W34 train leads={train_leads}; eval leads={eval_leads}",
    ))

    results.append(_required_tokens_check(
        root,
        "w34.training_contract",
        "submit_w34_tube_all.slurm",
        (
            "--gres=gpu:8",
            "--mem=500G",
            f"--mail-user={EMAIL}",
            "git pull --ff-only origin main",
            "--multi_lead_tube",
            "--tube_decode_chunk_size 2",
            "--distributional_head",
            "--crps_loss",
            "--sigma_floor 0.1",
            "--early_stop_metric tube_weekly7_tac",
            "--tube_loss_weekly_weight 0.20",
            'sbatch --parsable submit_w34_eval_stitch.slurm',
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "w34.evaluation_contract",
        "submit_w34_eval_stitch.slurm",
        (
            "--gres=gpu:1",
            "--mem=500G",
            f"--mail-user={EMAIL}",
            "--target_mode window",
            "--window_leads \"$LEADS\"",
            "--calibration_split val",
            "--eval_split test",
            "--calibrator platt",
            "--save_incremental_arrays",
            "--fit_mode cross_fitted",
            "--tube_decode_chunk_size 2",
            "CHECKPOINT=${CHECKPOINT_OVERRIDE:-best_monitor}",
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "opportunity.paired_parent_tests",
        "forecasts_of_opportunity.py",
        (
            "def paired_year_block_bootstrap_interactions(",
            "selection_parent",
            "driver_parent",
            "driver_interaction_paired_bootstrap.csv",
            "p_holm_mjo",
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "s2s.heatcast_ens_stack_opportunity_contract",
        "ens_heatcast_stack_opportunity.py",
        (
            "heatcast_ens_stack",
            "crossfit_excluding_fold",
            "paired_chunk(",
            "merge_cycle_probabilities",
            "init_time_index",
            "heatcast_top10_confidence",
            "opportunity_pair_bootstrap.csv",
            "ThreadPoolExecutor(max_workers=fold_workers)",
            "--fold_workers",
            "robustness_by_month.csv",
            "robustness_by_region.csv",
            "robustness_leave_one_out.csv",
            "Region robustness enabled",
            "--driver_table_dir",
            "driver_pair_bootstrap.csv",
            "driver_pair_parent_bootstrap.csv",
            "Paired driver-stratified Stack-vs-ENS tests",
            "generic_teleconnection",
            "driver_family",
            "alldata",
            "Cross-fit assert: PASS",
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "s2s.stack_opportunity_submission_contract",
        "submit_ens_stack_opportunity.slurm",
        (
            "--mem=500G",
            f"--mail-user={EMAIL}",
            "git pull --ff-only origin main",
            "ens_heatcast_stack_opportunity.py",
            "cvfold{F}_ens_w34,cvfold{F}_ens_w34_rt2024",
            "--bootstrap_reps 5000",
            "--max_stack_samples_per_fold 500000",
            "FOLD_WORKERS=${FOLD_WORKERS:-5}",
            "DRIVER_ARGS=()",
            "DRIVER_TABLE_DIR",
            "data_cache/slow_driver_tables_w34_alldata",
            "Missing driver table",
            '--fold_workers "$FOLD_WORKERS"',
            "OMP_NUM_THREADS=1",
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "s2s.teleconnection_stack_submission_contract",
        "submit_teleconnection_stack_analysis.slurm",
        (
            "--mem=500G",
            f"--mail-user={EMAIL}",
            "git pull --ff-only origin main",
            '"$PY" repo_integrity.py',
            "TELECONNECTION_INDEX_PATHS=${TELECONNECTION_INDEX_PATHS:?",
            "data_cache/slow_driver_tables_w34_teleconnections",
            "ens_heatcast_stack_opportunity_teleconnections",
            "--teleconnection_index_paths \"$TELECONNECTION_INDEX_PATHS\"",
            "--driver_table_dir \"$DRIVER_DIR\"",
            "--fold_workers \"$FOLD_WORKERS\"",
            "TELECONNECTION STACK ANALYSIS COMPLETE",
        ),
    ))
    tele_submit = _text(root, "submit_teleconnection_stack_analysis.slurm")
    results.append(_result(
        "s2s.teleconnection_stack_cpu_only_submission",
        "--gres=gpu" not in tele_submit
        and "module load cuda" not in tele_submit
        and "--partition=hpg-b200" not in tele_submit,
        "Teleconnection Stack-vs-ENS postprocessing is CPU-only and does not request B200 GPUs",
    ))
    stack_submit = _text(root, "submit_ens_stack_opportunity.slurm")
    results.append(_result(
        "s2s.stack_opportunity_cpu_only_submission",
        "--gres=gpu" not in stack_submit
        and "module load cuda" not in stack_submit
        and "--partition=hpg-b200" not in stack_submit,
        "Stack/opportunity paired postprocessing is CPU-only and does not request B200 GPUs",
    ))

    results.append(_required_tokens_check(
        root,
        "paper.evidence_blocks_contract",
        "build_paper_evidence_blocks.py",
        (
            "mechanism_block.csv",
            "robustness_block.csv",
            "operational_block.csv",
            "paper_evidence_summary.md",
            "generic teleconnection",
            "AllData",
            "paired_stack_vs_ens_driver",
            "driver_pair_parent_bootstrap.csv",
            "Stack minus ENS delta BSS",
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "paper.evidence_blocks_submission_contract",
        "submit_paper_evidence_blocks.slurm",
        (
            "--mem=16G",
            f"--mail-user={EMAIL}",
            "git pull --ff-only origin main",
            "build_paper_evidence_blocks.py",
            "paper_evidence_blocks/window_15-16-17-18-19-20-21-22-23-24-25-26-27-28",
        ),
    ))
    evidence_submit = _text(root, "submit_paper_evidence_blocks.slurm")
    results.append(_result(
        "paper.evidence_blocks_cpu_only_submission",
        "--gres=gpu" not in evidence_submit
        and "module load cuda" not in evidence_submit
        and "--partition=hpg-b200" not in evidence_submit,
        "Paper evidence block builder is CPU-only and does not request B200 GPUs",
    ))

    results.append(_required_tokens_check(
        root,
        "paper.figures_tables_contract",
        "build_paper_figures_tables.py",
        (
            "figure_1_headline_skill",
            "figure_2_headline_stack_minus_ens_ci",
            "figure_3_robustness",
            "figure_4_opportunity_and_driver_tests",
            "figure_5_probabilistic_scorecard",
            "figure_6_probability_threshold_operating_curves",
            "figure_7_opportunity_probability_metrics",
            "methods_text_draft.md",
            "narrative_and_claim_boundaries.md",
            "investigation_record.md",
            "reproducibility_manifest.json",
            "table_1_headline_model_metrics.csv",
            "table_6_operational_metrics.csv",
            "table_7_probability_threshold_operating_points.csv",
            "table_8_opportunity_probability_metrics.csv",
            "hit_rate_0.2",
            "false_alarm_ratio_0.2",
            "Do not say HeatCast alone beats ENS.",
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "paper.figures_extended_contract",
        "build_paper_figures_extended.py",
        (
            "figure_5_spatial_skill",
            "figure_6_reliability_decomposition",
            "figure_7_case_studies",
            "figure_8_per_lead_profile",
            "figure_9_teleconnection_ranking",
            "figure_9_opportunity_discard_curve",
            "figure_10a_leave_one_out_robustness",
            "table_7_stack_ablation_probability",
            "table_8_per_year_head_to_head",
            "table_9_teleconnection_ranking",
            "table_10_computational_cost_comparison",
            "murphy_decomposition",
            "auc_per_cell",
            "reproducibility_manifest.json",
            "source_entry",
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "paper.figures_journal_submission_contract",
        "submit_paper_figures_journal.slurm",
        (
            "--mem=64G",
            f"--mail-user={EMAIL}",
            "git pull --ff-only origin main",
            "figure_style.py",
            "build_paper_figures_tables.py",
            "build_paper_figures_extended.py",
            "--w34_log_glob",
            "OPENBLAS_NUM_THREADS=1",
        ),
    ))
    journal_submit = _text(root, "submit_paper_figures_journal.slurm")
    results.append(_result(
        "paper.figures_journal_cpu_only_submission",
        "--gres=gpu" not in journal_submit
        and "module load cuda" not in journal_submit
        and "--partition=hpg-b200" not in journal_submit,
        "Journal paper figure builder is CPU-only and does not request B200 GPUs",
    ))

    results.append(_result(
        "s2s.mixed_control_perturbed_grib_contract",
        all(token in ens_ingest for token in (
            'for data_type, default_member in (("cf", 0), ("pf", None)):',
            '"filter_by_keys": {"dataType": data_type}',
            '"indexpath": ""',
            "member_dim = _optional_dimension_name(data, member_candidates)",
            "data = data.expand_dims(member_dim)",
            "np.asarray(data[member_dim].values).reshape(-1)",
            "raw = np.concatenate([group[0] for group in groups], axis=0)",
            "member_values = np.concatenate([group[5] for group in groups])",
        )),
        "ENS ingestion opens and combines control and perturbed GRIB groups explicitly",
    ))

    ens_ingest_submission = _text(root, "submit_ens_widen_cycles.slurm")
    results.append(_result(
        "s2s.parallel_ingest_contract",
        all(token in ens_ingest for token in (
            "ProcessPoolExecutor",
            "multiprocessing.get_context(\"spawn\")",
            "def ingest_one_init(",
            "def _write_ingested_output(",
            "def validate_ingested_output(",
            "os.replace(temporary_path, output_path)",
            "Removing invalid existing output",
        ))
        and all(token in ens_ingest_submission for token in (
            "--cpus-per-task=32",
            "INGEST_WORKERS=${INGEST_WORKERS:-16}",
            '--workers "$INGEST_WORKERS"',
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
        )),
        "ENS ingestion uses bounded process parallelism and atomic resume-safe outputs",
    ))

    download_s2s = _text(root, "download_ecmwf_s2s.py")
    results.append(_result(
        "s2s.parallel_download_contract",
        "ThreadPoolExecutor(max_workers=int(args.workers))" in download_s2s
        and '"--workers"' in download_s2s
        and 'target.with_suffix(target.suffix + ".part")' in download_s2s
        and "partial.replace(target)" in download_s2s
        and "valid_grib(partial)" in download_s2s,
        "ENS downloading uses bounded parallel requests with validated atomic outputs",
    ))

    results.append(_result(
        "s2s.score_extended_global_contract",
        "def configure_fold(" in ens_score
        and "cfm.apply_extended_global_fields()" in ens_score
        and ens_score.index("cfm.apply_extended_global_fields()") < ens_score.index("norm_stats = ee.load_norm_stats()"),
        "ENS scoring applies the extended global-field configuration before loading fold norm stats",
    ))

    results.append(_result(
        "s2s.score_rejects_corrupt_ingest_contract",
        "validate_ingested_output(path, window_leads)" in ens_score
        and "Found {len(invalid_files)} invalid ingested ENS outputs" in ens_score
        and "Rerun submit_ens_widen_cycles.slurm" in ens_score,
        "ENS scoring rejects invalid ingested archives with a repair command",
    ))

    results.append(_result(
        "s2s.score_required_month_coverage_contract",
        "def required_target_months_by_lead(" in ens_score
        and "required_months = required_target_months_by_lead(" in ens_score
        and "for month in required_months[int(lead)]:" in ens_score
        and "int(years[target_t]) not in train_year_set" in ens_score,
        "ENS quantile mappings require only observed valid target months and remain target-year fold safe",
    ))

    results.append(_result(
        "s2s.score_qmap_init_fingerprint_contract",
        "mapping_init_indices=mapping_init_indices" in ens_score
        and '"mapping_init_indices" in data.files' in ens_score
        and 'data["mapping_init_indices"]' in ens_score
        and "np.array_equal(" in ens_score,
        "ENS quantile-mapping caches are invalidated when the exact training initialization set changes",
    ))

    results.append(_result(
        "s2s.score_lightweight_cache_contract",
        "def load_ens_scoring_shared_data(" in ens_score
        and '"heat_index": cache_dir / "heat_index.npy"' in ens_score
        and '"time_values": cache_dir / "time_values.npy"' in ens_score
        and "shared_data = load_ens_scoring_shared_data(cfm.Config)" in ens_score
        and "shared_data = cfm.prepare_shared_data" not in ens_score,
        "Parallel ENS folds use read-only heat/time disk memmaps and skip shared global predictor caches",
    ))

    ens_compare = _text(root, "ens_compare.py")
    results.append(_result(
        "s2s.multicycle_widening_contract",
        all(token in ens_score for token in (
            "--rt_tag",
            "downloaded S2S hdate initializations are authoritative",
            "quantile_cache_dir(cache_root, window_leads, rt_tag)",
        ))
        and all(token in ens_ingest for token in (
            "--rt_tag",
            "init_list_{rt_tag}.txt",
            "expected_rt_tag=rt_tag",
        ))
        and all(token in ens_compare for token in (
            "def merge_cycle_probabilities(",
            "def resolve_ens_run_groups(",
            "def per_year_comparison_rows(",
            "Bootstrap blocking assert: PASS (calendar year, never cycle).",
            "ens_heatcast_per_year.csv",
        )),
        "ENS cycles are bias-corrected separately, merged without duplicate-init weighting, and bootstrapped by year",
    ))

    results.append(_required_tokens_check(
        root,
        "s2s.multicycle_submission_contract",
        "submit_ens_widen_cycles.slurm",
        (
            "--mem=500G",
            "--gres=gpu:1",
            f"--mail-user={EMAIL}",
            "run_cycle \"\"",
            "run_cycle rt2024",
            "cvfold{F}_ens_w34,cvfold{F}_ens_w34_rt2024",
            "WINDOW_LABEL=window_${LEADS//,/-}",
            "Skipping complete ENS score",
            "incremental_arrays/manifest.npz",
            "--emit_per_year",
        ),
    ))

    for relative in CURRENT_SUBMISSIONS:
        text = _text(root, relative)
        results.append(_result(
            f"submission.preflight.{relative}",
            "git pull --ff-only origin main" in text
            and '"$PY" repo_integrity.py' in text,
            f"{relative}: pulls current code and runs repository integrity preflight",
        ))

    try:
        tracked = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        forbidden_suffixes = (".pth", ".pt", ".npy", ".npz", ".nc", ".pkl", ".grib", ".log", ".err")
        forbidden = sorted(path for path in tracked if path.lower().endswith(forbidden_suffixes))
        results.append(_result(
            "repository.no_tracked_runtime_artifacts",
            not forbidden,
            "No model/data/runtime artifacts tracked" if not forbidden else f"Tracked runtime artifacts: {forbidden}",
        ))
        tracked_submissions = tuple(sorted(path for path in tracked if path.endswith(".slurm")))
        results.append(_result(
            "repository.current_submission_set",
            tracked_submissions == CURRENT_SUBMISSIONS,
            (
                f"Tracked Slurm entry points are current: {tracked_submissions}"
                if tracked_submissions == CURRENT_SUBMISSIONS
                else f"Expected {CURRENT_SUBMISSIONS}; found {tracked_submissions}"
            ),
        ))
    except (OSError, subprocess.CalledProcessError) as exc:
        results.append(_result("repository.no_tracked_runtime_artifacts", False, f"git ls-files failed: {exc}"))

    workflow = _text(root, ".github/workflows/python-package.yml")
    results.append(_result(
        "ci.runs_integrity_and_pytest",
        "python repo_integrity.py" in workflow and "pytest" in workflow,
        "GitHub Actions runs the contract audit and pytest",
    ))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path(__file__).resolve().parent, type=Path)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    results = audit_repository(args.root)
    passed = sum(result.passed for result in results)
    if args.json:
        print(json.dumps({
            "passed": passed,
            "total": len(results),
            "checks": [asdict(result) for result in results],
        }, indent=2))
    else:
        for result in results:
            status = "PASS" if result.passed else "FAIL"
            print(f"[{status}] {result.name}: {result.detail}")
        print(f"\nRepository integrity: {passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
