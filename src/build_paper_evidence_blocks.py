#!/usr/bin/env python3
"""Build manuscript evidence blocks from saved HeatCast/ENS verification CSVs.

The script is intentionally analysis-only: it reads existing stack/opportunity
CSV outputs and writes compact tables for (1) mechanism, (2) robustness, and
(3) operational relevance.  It does not load a model or retrain anything.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


WINDOW_LABEL = "window_15-16-17-18-19-20-21-22-23-24-25-26-27-28"
ENS_MODEL = "ens_calibrated"
HEATCAST_MODEL = "heatcast_C"
STACK_MODEL = "heatcast_ens_stack"


def read_csv(path: Path, required: bool = True) -> List[Dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def f(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def fmt(value: float, digits: int = 4) -> str:
    return "nan" if not math.isfinite(value) else f"{value:+.{digits}f}"


def rows_by_model(rows: Iterable[Mapping[str, str]]) -> Dict[str, Mapping[str, str]]:
    return {str(row.get("model", "")): row for row in rows}


def grouped_model_rows(rows: Iterable[Mapping[str, str]], group_key: str) -> Dict[str, Dict[str, Mapping[str, str]]]:
    groups: Dict[str, Dict[str, Mapping[str, str]]] = {}
    for row in rows:
        groups.setdefault(str(row[group_key]), {})[str(row["model"])] = row
    return groups


def delta_record(
    group_type: str,
    group_value: str,
    models: Mapping[str, Mapping[str, str]],
    candidate: str = STACK_MODEL,
    baseline: str = ENS_MODEL,
) -> Dict[str, object]:
    cand = models[candidate]
    base = models[baseline]
    return {
        "group_type": group_type,
        "group_value": group_value,
        "candidate_model": candidate,
        "baseline_model": baseline,
        "candidate_bss": f(cand["bss_vs_monthly_climo"]),
        "baseline_bss": f(base["bss_vs_monthly_climo"]),
        "delta_bss": f(cand["bss_vs_monthly_climo"]) - f(base["bss_vs_monthly_climo"]),
        "candidate_auc": f(cand["roc_auc"]),
        "baseline_auc": f(base["roc_auc"]),
        "delta_auc": f(cand["roc_auc"]) - f(base["roc_auc"]),
        "candidate_slope": f(cand["reliability_slope"]),
        "candidate_ece": f(cand["ece"]),
        "valid_count": f(cand["valid_count"]),
    }


def extract_headline(stack_dir: Path) -> Dict[str, object]:
    rows = read_csv(stack_dir / "heatcast_ens_stack_head_to_head.csv")
    score = rows_by_model(row for row in rows if row.get("section") == "score")
    boot = [row for row in rows if row.get("section") == "bootstrap"]
    out: Dict[str, object] = {}
    for model in (ENS_MODEL, HEATCAST_MODEL, STACK_MODEL):
        row = score[model]
        out[f"{model}_bss"] = f(row["bss_vs_monthly_climo"])
        out[f"{model}_auc"] = f(row["roc_auc"])
        out[f"{model}_slope"] = f(row["reliability_slope"])
        out[f"{model}_ece"] = f(row["ece"])
    for row in boot:
        metric = str(row["metric"])
        out[f"{metric}_estimate"] = f(row["point_estimate"])
        out[f"{metric}_ci_low"] = f(row["ci_low"])
        out[f"{metric}_ci_high"] = f(row["ci_high"])
        out[f"{metric}_excludes_zero"] = row["ci_excludes_zero"]
    return out


def build_robustness_block(stack_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for filename, group_type, group_key in (
        ("robustness_by_month.csv", "month", "month"),
        ("robustness_by_region.csv", "region", "region"),
        ("robustness_by_year.csv", "year", "year"),
        ("robustness_by_fold.csv", "fold", "fold"),
    ):
        for group, models in grouped_model_rows(read_csv(stack_dir / filename), group_key).items():
            if ENS_MODEL in models and STACK_MODEL in models:
                rows.append(delta_record(group_type, group, models))
    for row in read_csv(stack_dir / "robustness_leave_one_out.csv"):
        if row.get("candidate_model") == STACK_MODEL:
            rows.append({
                "group_type": f"leave_one_{row['dropped_group_type']}",
                "group_value": row["dropped_group_value"],
                "candidate_model": STACK_MODEL,
                "baseline_model": ENS_MODEL,
                "delta_bss": f(row["delta_bss_candidate_minus_baseline"]),
                "delta_auc": f(row["delta_auc_candidate_minus_baseline"]),
                "candidate_bss": f(row["candidate_bss"]),
                "baseline_bss": f(row["baseline_bss"]),
                "candidate_auc": f(row["candidate_roc_auc"]),
                "baseline_auc": f(row["baseline_roc_auc"]),
                "candidate_ece": f(row["candidate_ece"]),
                "baseline_ece": f(row["baseline_ece"]),
                "valid_count": f(row["valid_count"]),
            })
    return rows


def build_mechanism_block(stack_dir: Path, opportunity_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in read_csv(stack_dir / "opportunity_pair_bootstrap.csv"):
        if row.get("candidate_model") == STACK_MODEL and row.get("metric", "").startswith("delta_bss"):
            rows.append({
                "evidence_type": "paired_stack_vs_ens_opportunity",
                "axis": row["comparison_set"],
                "stratum": row["comparison_set"],
                "delta_bss": f(row["point_estimate"]),
                "ci_low": f(row["ci_low"]),
                "ci_high": f(row["ci_high"]),
                "ci_excludes_zero": row["ci_excludes_zero"],
                "interpretation": "Stack skill relative to ENS in an opportunity subset.",
            })
    for row in read_csv(stack_dir / "robustness_region_bootstrap.csv", required=False):
        if row.get("candidate_model") == STACK_MODEL and row.get("metric", "").startswith("delta_bss"):
            rows.append({
                "evidence_type": "paired_stack_vs_ens_region",
                "axis": "region",
                "stratum": str(row["comparison_set"]).replace("region_", ""),
                "delta_bss": f(row["point_estimate"]),
                "ci_low": f(row["ci_low"]),
                "ci_high": f(row["ci_high"]),
                "ci_excludes_zero": row["ci_excludes_zero"],
                "interpretation": "Regional Stack-vs-ENS incremental skill.",
            })

    for row in read_csv(stack_dir / "driver_pair_bootstrap.csv", required=False):
        if row.get("candidate_model") == STACK_MODEL and row.get("metric", "").startswith("delta_bss"):
            if ":" not in str(row.get("comparison_set", "")):
                continue
            axis, stratum = str(row.get("comparison_set", "")).split(":", 1)
            rows.append({
                "evidence_type": "paired_stack_vs_ens_driver",
                "axis": axis,
                "stratum": stratum,
                "delta_bss": f(row["point_estimate"]),
                "ci_low": f(row["ci_low"]),
                "ci_high": f(row["ci_high"]),
                "ci_excludes_zero": row["ci_excludes_zero"],
                "interpretation": "Driver-stratified Stack-vs-ENS incremental skill on identical samples.",
            })
    for row in read_csv(stack_dir / "driver_pair_parent_bootstrap.csv", required=False):
        if row.get("metric") == "delta_bss_stack_vs_ens_child_minus_parent":
            rows.append({
                "evidence_type": "paired_stack_vs_ens_driver_parent_comparison",
                "axis": row.get("interaction_axis", ""),
                "stratum": row.get("interaction_stratum", ""),
                "parent_kind": row.get("parent_kind", ""),
                "parent_axis": row.get("parent_axis", ""),
                "parent_stratum": row.get("parent_stratum", ""),
                "delta_bss": f(row.get("point_estimate")),
                "ci_low": f(row.get("ci_low")),
                "ci_high": f(row.get("ci_high")),
                "p_value": f(row.get("p_value")),
                "ci_excludes_zero": row.get("ci_excludes_zero", ""),
                "interpretation": "Whether a driver interaction improves Stack-vs-ENS delta beyond its parent selection or driver stratum.",
            })

    driver_summary = read_csv(opportunity_dir / "driver_opportunity_summary.csv", required=False)
    for row in driver_summary:
        axis = row.get("axis", "")
        if axis in {"mjo_phase", "enso_state", "soil_moisture_tercile"} or axis.startswith("tele_") or axis.startswith("alldata_"):
            rows.append({
                "evidence_type": "heatcast_driver_stratification",
                "axis": axis,
                "stratum": row.get("stratum", ""),
                "bss_conditional": f(row.get("bss_conditional")),
                "ci_low": f(row.get("bss_conditional_ci_low")),
                "ci_high": f(row.get("bss_conditional_ci_high")),
                "ci_excludes_zero": (
                    f(row.get("bss_conditional_ci_low")) > 0.0
                    if math.isfinite(f(row.get("bss_conditional_ci_low")))
                    else ""
                ),
                "interpretation": "HeatCast-only driver opportunity signal, including AllData covariates when present; use as mechanism evidence, not ENS-paired proof.",
            })
    for row in read_csv(opportunity_dir / "driver_interaction_paired_bootstrap.csv", required=False):
        rows.append({
            "evidence_type": "heatcast_driver_parent_comparison",
            "axis": row.get("axis", ""),
            "stratum": row.get("interaction_stratum", ""),
            "parent_axis": row.get("parent_axis", ""),
            "parent_stratum": row.get("parent_stratum", ""),
            "delta_bss": f(row.get("delta")),
            "ci_low": f(row.get("delta_ci_low")),
            "ci_high": f(row.get("delta_ci_high")),
            "p_value": f(row.get("p_value")),
            "p_holm_mjo": f(row.get("p_holm_mjo")),
            "interpretation": "Paired HeatCast opportunity subset versus parent stratum.",
        })
    return rows


def build_operational_block(stack_dir: Path) -> List[Dict[str, object]]:
    rows = read_csv(stack_dir / "heatcast_ens_stack_head_to_head.csv")
    score = rows_by_model(row for row in rows if row.get("section") == "score")
    out: List[Dict[str, object]] = []
    for model in (ENS_MODEL, HEATCAST_MODEL, STACK_MODEL):
        row = score[model]
        out.append({
            "model": model,
            "brier": f(row["brier"]),
            "bss": f(row["bss_vs_monthly_climo"]),
            "roc_auc": f(row["roc_auc"]),
            "reliability_slope": f(row["reliability_slope"]),
            "ece": f(row["ece"]),
            "base_rate": f(row["base_rate"]),
            "valid_count": f(row["valid_count"]),
            "operational_role": (
                "benchmark ensemble forecast" if model == ENS_MODEL
                else "standalone ML forecast" if model == HEATCAST_MODEL
                else "combined decision forecast"
            ),
        })
    for row in read_csv(stack_dir / "opportunity_pair_summary.csv"):
        if row.get("model") in {ENS_MODEL, STACK_MODEL}:
            out.append({
                "model": row["model"],
                "subset": row["subset"],
                "brier": f(row["brier"]),
                "bss": f(row["bss_vs_monthly_climo"]),
                "roc_auc": f(row["roc_auc"]),
                "reliability_slope": f(row["reliability_slope"]),
                "ece": f(row["ece"]),
                "base_rate": f(row["base_rate"]),
                "valid_count": f(row["valid_count"]),
                "operational_role": "opportunity/triage subset",
            })
    return out


def top_rows(rows: Sequence[Mapping[str, object]], key: str, reverse: bool = True, n: int = 5) -> List[Mapping[str, object]]:
    return sorted(rows, key=lambda row: f(row.get(key)), reverse=reverse)[:n]


def write_summary(path: Path, headline: Mapping[str, object], mechanism: Sequence[Mapping[str, object]], robustness: Sequence[Mapping[str, object]], operational: Sequence[Mapping[str, object]]) -> None:
    month = [r for r in robustness if r.get("group_type") == "month"]
    region = [r for r in robustness if r.get("group_type") == "region"]
    loo = [r for r in robustness if str(r.get("group_type", "")).startswith("leave_one_")]
    lines = [
        "# HeatCast ENS Paper Evidence Blocks",
        "",
        "## Headline",
        f"- ENS calibrated BSS: {fmt(f(headline['ens_calibrated_bss']))}; AUC: {fmt(f(headline['ens_calibrated_auc']))}",
        f"- HeatCast-C BSS: {fmt(f(headline['heatcast_C_bss']))}; AUC: {fmt(f(headline['heatcast_C_auc']))}",
        f"- HeatCast+ENS stack BSS: {fmt(f(headline['heatcast_ens_stack_bss']))}; AUC: {fmt(f(headline['heatcast_ens_stack_auc']))}",
        f"- Stack minus ENS delta BSS: {fmt(f(headline['delta_bss_heatcast_ens_stack_minus_ens_calibrated_estimate']))} "
        f"CI=[{fmt(f(headline['delta_bss_heatcast_ens_stack_minus_ens_calibrated_ci_low']))},"
        f"{fmt(f(headline['delta_bss_heatcast_ens_stack_minus_ens_calibrated_ci_high']))}]",
        f"- Stack minus ENS delta AUC: {fmt(f(headline['delta_auc_heatcast_ens_stack_minus_ens_calibrated_estimate']))} "
        f"CI=[{fmt(f(headline['delta_auc_heatcast_ens_stack_minus_ens_calibrated_ci_low']))},"
        f"{fmt(f(headline['delta_auc_heatcast_ens_stack_minus_ens_calibrated_ci_high']))}]",
        "",
        "## Robustness Block",
        "- Month deltas Stack-vs-ENS: "
        + ", ".join(f"{r['group_value']}={fmt(f(r['delta_bss']))}" for r in sorted(month, key=lambda r: int(r["group_value"]))),
        "- Top regional Stack-vs-ENS deltas: "
        + ", ".join(f"{r['group_value']}={fmt(f(r['delta_bss']))}" for r in top_rows(region, "delta_bss", n=4)),
        "- Leave-one-out minimum deltas: "
        + ", ".join(
            f"{kind.replace('leave_one_', '')}={fmt(min(f(r['delta_bss']) for r in loo if r['group_type'] == kind))}"
            for kind in sorted({str(r["group_type"]) for r in loo})
        ),
        "",
        "## Mechanism Block",
        "- Use paired Stack-vs-ENS regional, opportunity, and driver rows as primary mechanism evidence.",
        "- Treat HeatCast-only MJO/ENSO/soil/generic teleconnection/AllData rows as secondary context when paired driver rows are absent or sparse.",
        "- Strongest paired regional candidates are listed in mechanism_block.csv.",
        "",
        "## Operational Block",
        "- Use BSS, reliability slope, ECE, AUC, and top-confidence subset rows from operational_block.csv.",
        "- The operational claim should be probabilistic W34 heat-exceedance risk, not deterministic anomaly prediction.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stack_dir",
        default=f"ens_heatcast_stack_opportunity/{WINDOW_LABEL}",
        help="Directory containing HeatCast/ENS stack CSV outputs.",
    )
    parser.add_argument(
        "--opportunity_dir",
        default=f"exceedance_eval_incremental/opportunity_{WINDOW_LABEL}",
        help="Directory containing slow-driver/opportunity CSV outputs.",
    )
    parser.add_argument(
        "--output_dir",
        default=f"paper_evidence_blocks/{WINDOW_LABEL}",
    )
    args = parser.parse_args()

    stack_dir = Path(args.stack_dir)
    opportunity_dir = Path(args.opportunity_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    headline = extract_headline(stack_dir)
    mechanism = build_mechanism_block(stack_dir, opportunity_dir)
    robustness = build_robustness_block(stack_dir)
    operational = build_operational_block(stack_dir)

    write_csv(output_dir / "headline_block.csv", [headline])
    write_csv(output_dir / "mechanism_block.csv", mechanism)
    write_csv(output_dir / "robustness_block.csv", robustness)
    write_csv(output_dir / "operational_block.csv", operational)
    write_summary(output_dir / "paper_evidence_summary.md", headline, mechanism, robustness, operational)

    print("Paper evidence blocks complete")
    print(f"  stack_dir={stack_dir}")
    print(f"  opportunity_dir={opportunity_dir}")
    print(f"  output_dir={output_dir}")
    print(
        "  headline: "
        f"Stack BSS={fmt(f(headline['heatcast_ens_stack_bss']))}, "
        f"ENS BSS={fmt(f(headline['ens_calibrated_bss']))}, "
        f"delta_BSS={fmt(f(headline['delta_bss_heatcast_ens_stack_minus_ens_calibrated_estimate']))}"
    )
    print(f"  mechanism rows={len(mechanism)}")
    print(f"  robustness rows={len(robustness)}")
    print(f"  operational rows={len(operational)}")


if __name__ == "__main__":
    main()
