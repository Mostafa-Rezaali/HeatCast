#!/usr/bin/env python3
"""Build a compact meeting evidence pack for HeatCast role-fit discussions.

The pack is designed for a short technical meeting where the central argument is:
HeatCast already demonstrates leakage-safe extreme-event calibration, post-hoc
S2S ensemble correction, probabilistic verification, and ML+dynamical-model
stacking skill.

The script is intentionally read-only with respect to analysis products. It does
not recompute metrics. It copies existing CSV/figure/source evidence into a
dated folder, writes a concise briefing note, and creates a zip archive.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


WINDOW = "window_15-16-17-18-19-20-21-22-23-24-25-26-27-28"
OPP_WINDOW = "opportunity_15-16-17-18-19-20-21-22-23-24-25-26-27-28"


@dataclass(frozen=True)
class EvidenceFile:
    source: Path
    dest: Path
    why: str


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fnum(value: object) -> float | None:
    try:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        return float(text)
    except Exception:
        return None


def fmt(value: object, digits: int = 4, signed: bool = True) -> str:
    val = fnum(value)
    if val is None:
        return "NA"
    prefix = "+" if signed else ""
    return f"{val:{prefix}.{digits}f}"


def git_text(root: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def newest_existing(candidates: Iterable[Path]) -> Path | None:
    existing = [p for p in candidates if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def add_dir_files(root: Path, src_dir: Path, dest_prefix: Path, suffixes: set[str], why: str) -> list[EvidenceFile]:
    out: list[EvidenceFile] = []
    if not src_dir.exists():
        return out
    for src in sorted(p for p in src_dir.rglob("*") if p.is_file() and p.suffix.lower() in suffixes):
        rel = src.relative_to(src_dir)
        out.append(EvidenceFile(src, dest_prefix / rel, why))
    return out


def stack_dir(root: Path, preferred: str) -> Path:
    draft = root / "Draft_Paper"
    candidates = [
        draft / preferred / WINDOW,
        root / preferred / WINDOW,
    ]
    found = newest_existing(candidates)
    return found if found is not None else candidates[0]


def evidence_paths(root: Path) -> dict[str, Path]:
    draft = root / "Draft_Paper"
    return {
        "stack": stack_dir(root, "ens_heatcast_stack_opportunity"),
        "stack_tele": stack_dir(root, "ens_heatcast_stack_opportunity_teleconnections"),
        "opportunity": newest_existing(
            [
                draft / "exceedance_eval_incremental" / OPP_WINDOW,
                root / "exceedance_eval_incremental" / OPP_WINDOW,
            ]
        )
        or draft / "exceedance_eval_incremental" / OPP_WINDOW,
        "evidence": newest_existing(
            [
                draft / "paper_evidence_blocks" / WINDOW,
                root / "paper_evidence_blocks" / WINDOW,
            ]
        )
        or draft / "paper_evidence_blocks" / WINDOW,
        "fig_tables": newest_existing(
            [
                draft / "paper_figures_tables" / WINDOW,
                root / "paper_figures_tables" / WINDOW,
            ]
        )
        or draft / "paper_figures_tables" / WINDOW,
        "fig_ext": newest_existing(
            [
                draft / "paper_figures_extended" / WINDOW,
                root / "paper_figures_extended" / WINDOW,
            ]
        )
        or draft / "paper_figures_extended" / WINDOW,
    }


def extract_stack_headline(stack_csv: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(stack_csv)
    return {
        row.get("model", ""): row
        for row in rows
        if row.get("section") == "score" and row.get("model")
    }


def extract_bootstrap(stack_csv: Path) -> list[dict[str, str]]:
    return [row for row in read_csv_rows(stack_csv) if row.get("section") == "bootstrap"]


def selected_figures(paths: dict[str, Path]) -> list[EvidenceFile]:
    specs: list[tuple[Path, str]] = []
    fig_dir = paths["fig_tables"] / "figures"
    ext_fig_dir = paths["fig_ext"] / "figures"
    for name in [
        "figure_1_headline_skill",
        "figure_2_headline_stack_minus_ens_ci",
        "figure_3_robustness",
        "figure_4_opportunity_and_driver_tests",
        "figure_5_probabilistic_scorecard",
        "figure_6_probability_threshold_operating_curves",
        "figure_7_opportunity_probability_metrics",
    ]:
        specs.append((fig_dir / f"{name}.png", "paper figure supporting headline/robustness/opportunity claims"))
        specs.append((fig_dir / f"{name}.svg", "editable vector version of paper figure"))
    for name in [
        "figure_5_spatial_skill",
        "figure_6_reliability_decomposition",
        "figure_9_opportunity_discard_curve",
        "figure_10_per_year_base_rate",
    ]:
        specs.append((ext_fig_dir / f"{name}.png", "extended probabilistic evidence figure"))
        specs.append((ext_fig_dir / f"{name}.svg", "editable vector version of extended evidence figure"))
    return [
        EvidenceFile(src, Path("figures") / src.name, why)
        for src, why in specs
        if src.exists()
    ]


def selected_tables(paths: dict[str, Path]) -> list[EvidenceFile]:
    files: list[EvidenceFile] = []
    table_groups = {
        "stack_head_to_head": [
            paths["stack"] / "heatcast_ens_stack_head_to_head.csv",
            paths["stack"] / "opportunity_pair_bootstrap.csv",
            paths["stack"] / "opportunity_pair_summary.csv",
            paths["stack"] / "robustness_by_month.csv",
            paths["stack"] / "robustness_by_region.csv",
            paths["stack"] / "robustness_by_year.csv",
            paths["stack"] / "robustness_leave_one_out.csv",
            paths["stack"] / "robustness_region_bootstrap.csv",
            paths["stack"] / "driver_pair_bootstrap.csv",
            paths["stack"] / "driver_pair_parent_bootstrap.csv",
            paths["stack"] / "driver_pair_summary.csv",
        ],
        "stack_teleconnections": [
            paths["stack_tele"] / "driver_pair_bootstrap.csv",
            paths["stack_tele"] / "driver_pair_parent_bootstrap.csv",
            paths["stack_tele"] / "driver_pair_summary.csv",
            paths["stack_tele"] / "heatcast_ens_stack_head_to_head.csv",
        ],
        "heatcast_opportunity": [
            paths["opportunity"] / "opportunity_summary.csv",
            paths["opportunity"] / "driver_opportunity_summary.csv",
            paths["opportunity"] / "driver_interaction_paired_bootstrap.csv",
        ],
        "paper_evidence_blocks": [
            paths["evidence"] / "paper_evidence_summary.md",
            paths["evidence"] / "mechanism_block.csv",
            paths["evidence"] / "operational_block.csv",
            paths["evidence"] / "robustness_block.csv",
        ],
    }
    for group, sources in table_groups.items():
        for src in sources:
            if src.exists():
                files.append(EvidenceFile(src, Path("tables_and_summaries") / group / src.name, "numeric evidence table"))
    for sub in ["tables", "reproducibility"]:
        files.extend(
            add_dir_files(
                paths["fig_tables"],
                paths["fig_tables"] / sub,
                Path("tables_and_summaries") / f"paper_figures_tables_{sub}",
                {".csv", ".md", ".txt", ".json"},
                "paper figure/table source data",
            )
        )
    for sub in ["tables", "reproducibility"]:
        files.extend(
            add_dir_files(
                paths["fig_ext"],
                paths["fig_ext"] / sub,
                Path("tables_and_summaries") / f"paper_figures_extended_{sub}",
                {".csv", ".md", ".txt", ".json"},
                "extended figure/table source data",
            )
        )
    return files


def selected_code(root: Path) -> list[EvidenceFile]:
    code_specs = [
        ("exceedance_eval.py", "q95 exceedance target, leakage-safe calibration, reliability metrics"),
        ("ens_score.py", "post-hoc ENS quantile mapping and calibration"),
        ("ens_heatcast_stack_opportunity.py", "HeatCast+ENS stacking, paired bootstrap, opportunity tests"),
        ("forecasts_of_opportunity.py", "driver/opportunity stratification and paired parent comparisons"),
        ("cfm_mesh_train.py", "PyTorch/DDP training, CRPS/distributional head hooks, validation metrics"),
        ("mesh_backbone.py", "GraphCast-style mesh GNN backbone"),
        ("mode_dispatch.py", "persistence-residual distributional forecast semantics"),
        ("build_paper_figures_tables.py", "paper figure/table generation"),
        ("build_paper_figures_extended.py", "probabilistic figure generation"),
        ("repo_integrity.py", "contract checks proving leakage and workflow constraints are encoded"),
        ("submit_w34_tube_all.slurm", "GPU training workflow"),
        ("submit_w34_eval_stitch.slurm", "fold evaluation/stitch workflow"),
        ("submit_ens_widen_cycles.slurm", "ECMWF ENS ingestion/scoring workflow"),
        ("submit_ens_stack_opportunity.slurm", "CPU-only paired HeatCast+ENS stack/opportunity workflow"),
        ("submit_teleconnection_stack_analysis.slurm", "driver-stratified Stack-vs-ENS workflow"),
    ]
    out = []
    for rel, why in code_specs:
        src = root / rel
        if src.exists():
            out.append(EvidenceFile(src, Path("code_and_workflows") / rel, why))
    for rel in ["heatcast_specs.pdf", "README.md", "Model_Inputs.txt", "INTEGRITY_TEST_KIT.md"]:
        src = root / rel
        if src.exists():
            out.append(EvidenceFile(src, Path("code_and_workflows") / rel, "leave-behind/specification document"))
    return out


def write_brief(root: Path, out_dir: Path, paths: dict[str, Path]) -> None:
    stack_csv = paths["stack"] / "heatcast_ens_stack_head_to_head.csv"
    headline = extract_stack_headline(stack_csv)
    boot = extract_bootstrap(stack_csv)

    ens = headline.get("ens_calibrated", {})
    heat = headline.get("heatcast_C", {})
    stack = headline.get("heatcast_ens_stack", {})

    def boot_line(metric: str) -> str:
        row = next((r for r in boot if r.get("metric") == metric), {})
        if not row:
            return "not found"
        return (
            f"{fmt(row.get('point_estimate'))} "
            f"CI=[{fmt(row.get('ci_low'))},{fmt(row.get('ci_high'))}], "
            f"excludes_zero={row.get('ci_excludes_zero')}, "
            f"blocks={row.get('independent_year_blocks')}"
        )

    status = git_text(root, ["status", "--short"])
    commit = git_text(root, ["rev-parse", "HEAD"])

    brief = f"""# Meeting Evidence Pack: HeatCast Fit For Extreme-Event Calibration / S2S Bias Correction

Generated: {datetime.now().isoformat(timespec="seconds")}
Repo: `{root}`
Commit: `{commit}`

## 30-second position

HeatCast maps directly onto the role because it is already a leakage-safe,
probabilistic, extreme-event calibration and post-hoc S2S correction workflow:

1. It predicts month-specific W34 heat-exceedance risk above a fold-safe q95 threshold.
2. It calibrates rare-event probabilities with disjoint train/calibration/test years.
3. It ingests and bias-corrects ECMWF ENS reforecasts using train-only quantile mapping.
4. It proves additive ML value over an operational dynamical ensemble through a paired,
   cross-fitted HeatCast+ENS stack and year-block bootstrap.

## Lead with these numbers

Source: `tables_and_summaries/stack_head_to_head/heatcast_ens_stack_head_to_head.csv`

| Model | BSS vs climatology | AUC | Reliability slope | ECE | Brier |
|---|---:|---:|---:|---:|---:|
| ENS calibrated | {fmt(ens.get('bss_vs_monthly_climo'))} | {fmt(ens.get('roc_auc'), signed=False)} | {fmt(ens.get('reliability_slope'), signed=False)} | {fmt(ens.get('ece'), signed=False)} | {fmt(ens.get('brier'), signed=False)} |
| HeatCast-C | {fmt(heat.get('bss_vs_monthly_climo'))} | {fmt(heat.get('roc_auc'), signed=False)} | {fmt(heat.get('reliability_slope'), signed=False)} | {fmt(heat.get('ece'), signed=False)} | {fmt(heat.get('brier'), signed=False)} |
| HeatCast+ENS stack | {fmt(stack.get('bss_vs_monthly_climo'))} | {fmt(stack.get('roc_auc'), signed=False)} | {fmt(stack.get('reliability_slope'), signed=False)} | {fmt(stack.get('ece'), signed=False)} | {fmt(stack.get('brier'), signed=False)} |

Bootstrap headline:

- Stack minus ENS delta BSS: {boot_line('delta_bss_heatcast_ens_stack_minus_ens_calibrated')}
- Stack minus ENS delta AUC: {boot_line('delta_auc_heatcast_ens_stack_minus_ens_calibrated')}

Use this wording: **"HeatCast alone is competitive with ENS, but the stronger result is
that HeatCast adds independent calibrated information to ENS."**

## Evidence mapped to the job description

### 1. Extreme-event calibration targeted at heatwaves

Claim:
You built fold-safe q95 heat-exceedance calibration for W34 heat risk.

Evidence:
- `code_and_workflows/exceedance_eval.py`
- `tables_and_summaries/*/heatcast_ens_stack_head_to_head.csv`
- `figures/figure_6_reliability_decomposition.png`
- `figures/figure_5_probabilistic_scorecard.png`

What to say:
"The target is a pixelwise, month-specific q95 exceedance, with thresholds built
from train years only. Calibration and reporting are split by disjoint years, so
I can talk concretely about leakage in rare-event calibration."

### 2. Post-hoc correction of a dynamical S2S forecast system

Claim:
You already implemented the NASA-GEOS-style pattern using ECMWF ENS: ingest dynamical
reforecasts, bias-correct them, calibrate probabilities, and evaluate against the same
truth/thresholds as HeatCast.

Evidence:
- `code_and_workflows/ens_score.py`
- `code_and_workflows/submit_ens_widen_cycles.slurm`
- `tables_and_summaries/*/heatcast_ens_stack_head_to_head.csv`

What to say:
"The operational model changes from ECMWF ENS to GEOS, but the statistical layer is
the same: fold-safe reforecast bias correction, probability calibration, and paired
verification."

### 3. Probabilistic verification vocabulary

Claim:
You already report the metrics named in the role family: RMSE/ACC-like anomaly
correlation/TAC, CRPS/CRPSS-adjacent distributional training, BSS, reliability,
ECE, AUC, and year-block bootstrap CIs.

Evidence:
- `code_and_workflows/cfm_mesh_train.py`
- `code_and_workflows/exceedance_eval.py`
- `figures/figure_1_headline_skill.png`
- `figures/figure_2_headline_stack_minus_ens_ci.png`
- `figures/figure_6_reliability_decomposition.png`

### 4. ML adds value to an operational ensemble

Claim:
The strongest result is not "my ML model beats ENS everywhere"; it is that a
cross-fitted stack of HeatCast+ENS beats ENS with positive year-block CI.

Evidence:
- `code_and_workflows/ens_heatcast_stack_opportunity.py`
- `tables_and_summaries/*/opportunity_pair_bootstrap.csv`
- `figures/figure_2_headline_stack_minus_ens_ci.png`
- `figures/figure_3_robustness.png`

## Honest gaps and how to frame them

### Transformers

Do not claim HeatCast is transformer-based. It is a GraphCast-style mesh GNN with
temporal tube attention. Say:
"I have shipped graph neural weather models and temporal attention components. I have
not shipped a transformer bias-correction model yet, but the adaptation is architectural,
not conceptual: the data discipline, calibration, and verification are already in place."

### Dask/Zarr/Icechunk/cloud

Say:
"My current stack is NetCDF/memmap/SLURM on HiPerGator, not Dask/Zarr on cloud. The
underlying problem is the same: out-of-core climate arrays, reproducible preprocessing,
and fold-safe reforecast evaluation."

### Quantum-inspired training

Say:
"I have not worked on that yet, but I am comfortable reading a method paper and
turning it into tested code."

## One sharp question for Aodhan

"For NASA GEOS reforecasts, how are you planning to prevent calibration leakage across
limited hindcast years? In my ENS comparison, the small number of independent reforecast
years was the dominant uncertainty source, so I ended up using year-disjoint calibration
and whole-year block bootstrap."

## Files to open quickly

1. `code_and_workflows/heatcast_specs.pdf`
2. `figures/figure_2_headline_stack_minus_ens_ci.png`
3. `figures/figure_6_reliability_decomposition.png`
4. `tables_and_summaries/stack_head_to_head/heatcast_ens_stack_head_to_head.csv`
5. `tables_and_summaries/paper_evidence_blocks/paper_evidence_summary.md`

## Repo status at packaging time

```text
{status if status else "clean"}
```
"""
    (out_dir / "00_MEETING_BRIEF.md").write_text(brief, encoding="utf-8")

    questions = """# Meeting Questions

1. How many independent GEOS reforecast years are available for calibration and testing?
2. Will calibration be global, regional, grid-cell-wise, or regime-conditioned?
3. Are they prioritizing reliability/Brier/CRPSS, or deterministic ACC/RMSE?
4. Does the transformer correct fields directly, probabilities, or distributional parameters?
5. How do they handle extremes: threshold-specific calibration, tail-aware loss, or post-hoc recalibration?
6. Are they evaluating additive skill over the operational system, or standalone ML skill?
"""
    (out_dir / "01_QUESTIONS_TO_ASK.md").write_text(questions, encoding="utf-8")


def build_pack(root: Path, output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / f"heatcast_meeting_evidence_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = evidence_paths(root)
    evidence: list[EvidenceFile] = []
    evidence.extend(selected_figures(paths))
    evidence.extend(selected_tables(paths))
    evidence.extend(selected_code(root))

    copied: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    seen: set[tuple[Path, Path]] = set()
    for item in evidence:
        key = (item.source.resolve(), item.dest)
        if key in seen:
            continue
        seen.add(key)
        dst = out_dir / item.dest
        if copy_file(item.source, dst):
            copied.append(
                {
                    "source": str(item.source),
                    "dest": str(item.dest),
                    "why": item.why,
                    "bytes": str(dst.stat().st_size),
                }
            )
        else:
            missing.append({"source": str(item.source), "dest": str(item.dest), "why": item.why})

    write_brief(root, out_dir, paths)
    (out_dir / "evidence_manifest.json").write_text(
        json.dumps(
            {
                "generated": datetime.now().isoformat(timespec="seconds"),
                "root": str(root),
                "window": WINDOW,
                "paths": {k: str(v) for k, v in paths.items()},
                "copied": copied,
                "missing": missing,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
            zf.write(path, path.relative_to(out_dir.parent))
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--output_root", default="meeting_evidence", help="Directory where the evidence pack is written.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    output_root = (root / args.output_root).resolve()
    out_dir = build_pack(root, output_root)
    print(f"Meeting evidence folder: {out_dir}")
    print(f"Meeting evidence zip:    {out_dir.with_suffix('.zip')}")
    print(f"Open first:              {out_dir / '00_MEETING_BRIEF.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
