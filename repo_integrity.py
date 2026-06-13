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
            "git pull --ff-only origin codex/tube_v1",
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
        "opportunity.slow_driver_submission_contract",
        "submit_slow_driver_opportunity.slurm",
        (
            "--mem=500G",
            "--gres=gpu:1",
            f"--mail-user={EMAIL}",
            "--driver_table_dir \"$DRIVER_DIR\"",
            "--bootstrap_axes mjo_phase,enso_state,soil_moisture_tercile",
            '"$PY" repo_integrity.py',
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "s2s.ingest_and_compare_contract",
        "submit_ens_score_compare.slurm",
        (
            "--mem=500G",
            "--gres=gpu:1",
            f"--mail-user={EMAIL}",
            "LEADS=15,16,17,18,19,20,21,22,23,24,25,26,27,28",
            "--bootstrap_reps 5000",
            '"$PY" repo_integrity.py',
        ),
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

    ens_ingest_submission = _text(root, "submit_ens_ingest.slurm")
    results.append(_result(
        "s2s.parallel_ingest_contract",
        all(token in ens_ingest for token in (
            "ProcessPoolExecutor",
            "multiprocessing.get_context(\"spawn\")",
            "def ingest_one_init(",
            "def _write_ingested_output(",
            "os.replace(temporary_path, output_path)",
        ))
        and all(token in ens_ingest_submission for token in (
            "--cpus-per-task=32",
            "--workers 8",
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
        )),
        "ENS ingestion uses bounded process parallelism and atomic resume-safe outputs",
    ))

    for relative in (
        "submit_w34_tube_all.slurm",
        "submit_w34_eval_stitch.slurm",
        "submit_ens_ingest.slurm",
        "submit_ens_score_compare.slurm",
        "submit_slow_driver_opportunity.slurm",
    ):
        text = _text(root, relative)
        results.append(_result(
            f"submission.preflight.{relative}",
            "git pull --ff-only origin codex/tube_v1" in text
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
