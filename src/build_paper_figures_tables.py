#!/usr/bin/env python3
"""Build paper-ready HeatCast/ENS figures, tables, methods, and audit notes.

This is an analysis-only manuscript packaging script. It reads the existing
W34 evidence CSVs, writes compact publication tables, generates figure panels,
and records the investigation trail and reproducibility metadata. It does not
load a model, read NetCDF data, or retrain anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from figure_style import (
    DOUBLE_COLUMN_MM,
    SINGLE_COLUMN_MM,
    figure_size,
    panel_label,
    save_figure,
    setup_matplotlib,
    system_color,
    system_label,
)


WINDOW_LABEL = "window_15-16-17-18-19-20-21-22-23-24-25-26-27-28"
ENS_MODEL = "ens_calibrated"
HEATCAST_MODEL = "heatcast_C"
STACK_MODEL = "heatcast_ens_stack"
REFERENCE_MODEL = "windowed_climatology"
PROBABILITY_THRESHOLDS = ("0.1", "0.2", "0.3", "0.5")
PROBABILITY_THRESHOLD_FIELDS = (
    "hit_rate_0.1",
    "false_alarm_ratio_0.1",
    "hit_rate_0.2",
    "false_alarm_ratio_0.2",
    "hit_rate_0.3",
    "false_alarm_ratio_0.3",
    "hit_rate_0.5",
    "false_alarm_ratio_0.5",
)


def read_csv(path: Path, required: bool = True) -> List[Dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("_No rows._\n", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row.keys()))
    lines = [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join("---" for _ in keys) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def f(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def fmt(value: object, digits: int = 4, signed: bool = True) -> str:
    value_f = f(value)
    if not math.isfinite(value_f):
        return "nan"
    sign = "+" if signed else ""
    return f"{value_f:{sign}.{digits}f}"


def rows_by_model(rows: Iterable[Mapping[str, str]]) -> Dict[str, Mapping[str, str]]:
    return {str(row.get("model", "")): row for row in rows}


def ensure_matplotlib():
    return setup_matplotlib()


def savefig(fig, path_base: Path) -> None:
    save_figure(fig, path_base)


def extract_head_to_head(stack_dir: Path) -> tuple[Dict[str, Mapping[str, str]], List[Mapping[str, str]]]:
    rows = read_csv(stack_dir / "heatcast_ens_stack_head_to_head.csv")
    scores = rows_by_model(row for row in rows if row.get("section") == "score")
    boot = [row for row in rows if row.get("section") == "bootstrap"]
    return scores, boot


def model_label(model: str) -> str:
    return system_label(model)


def build_headline_tables(stack_dir: Path, table_dir: Path) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    scores, boot = extract_head_to_head(stack_dir)
    score_rows: List[Dict[str, object]] = []
    for model in (REFERENCE_MODEL, "ens_raw_fraction", ENS_MODEL, HEATCAST_MODEL, STACK_MODEL):
        row = scores[model]
        score_rows.append({
            "model": model_label(model),
            "brier": fmt(row["brier"], signed=False),
            "bss": fmt(row["bss_vs_monthly_climo"]),
            "roc_auc": fmt(row["roc_auc"], signed=False),
            "reliability_slope": fmt(row["reliability_slope"], signed=False),
            "ece": fmt(row["ece"], signed=False),
            "valid_cells": int(f(row["valid_count"])),
        })
    bootstrap_rows: List[Dict[str, object]] = []
    for row in boot:
        bootstrap_rows.append({
            "metric": row["metric"],
            "estimate": fmt(row["point_estimate"]),
            "ci_low": fmt(row["ci_low"]),
            "ci_high": fmt(row["ci_high"]),
            "ci_excludes_zero": row["ci_excludes_zero"],
            "year_blocks": row.get("independent_year_blocks", ""),
        })
    write_csv(table_dir / "table_1_headline_model_metrics.csv", score_rows)
    write_markdown_table(table_dir / "table_1_headline_model_metrics.md", score_rows)
    write_csv(table_dir / "table_2_headline_bootstrap_deltas.csv", bootstrap_rows)
    write_markdown_table(table_dir / "table_2_headline_bootstrap_deltas.md", bootstrap_rows)
    return score_rows, bootstrap_rows


def build_robustness_tables(evidence_dir: Path, stack_dir: Path, table_dir: Path) -> List[Dict[str, object]]:
    robustness = read_csv(evidence_dir / "robustness_block.csv")
    rows: List[Dict[str, object]] = []
    for row in robustness:
        rows.append({
            "group_type": row.get("group_type", ""),
            "group_value": row.get("group_value", ""),
            "delta_bss_stack_minus_ens": fmt(row.get("delta_bss")),
            "delta_auc_stack_minus_ens": fmt(row.get("delta_auc")),
            "stack_bss": fmt(row.get("candidate_bss")),
            "ens_bss": fmt(row.get("baseline_bss")),
            "stack_auc": fmt(row.get("candidate_auc"), signed=False),
            "ens_auc": fmt(row.get("baseline_auc"), signed=False),
        })
    region_boot = read_csv(stack_dir / "robustness_region_bootstrap.csv", required=False)
    region_rows = []
    for row in region_boot:
        if row.get("candidate_model") == STACK_MODEL and row.get("metric", "").startswith("delta_bss"):
            region_rows.append({
                "region": str(row["comparison_set"]).replace("region_", ""),
                "delta_bss": fmt(row["point_estimate"]),
                "ci_low": fmt(row["ci_low"]),
                "ci_high": fmt(row["ci_high"]),
                "ci_excludes_zero": row["ci_excludes_zero"],
            })
    write_csv(table_dir / "table_3_robustness_by_group.csv", rows)
    write_markdown_table(table_dir / "table_3_robustness_by_group.md", rows)
    write_csv(table_dir / "table_4_region_bootstrap.csv", region_rows)
    write_markdown_table(table_dir / "table_4_region_bootstrap.md", region_rows)
    return rows


def build_mechanism_tables(evidence_dir: Path, table_dir: Path) -> List[Dict[str, object]]:
    mechanism = read_csv(evidence_dir / "mechanism_block.csv")
    rows: List[Dict[str, object]] = []
    for row in mechanism:
        rows.append({
            "evidence_type": row.get("evidence_type", ""),
            "axis": row.get("axis", ""),
            "stratum": row.get("stratum", ""),
            "parent_kind": row.get("parent_kind", ""),
            "parent_axis": row.get("parent_axis", ""),
            "parent_stratum": row.get("parent_stratum", ""),
            "delta_bss": fmt(row.get("delta_bss")),
            "ci_low": fmt(row.get("ci_low")),
            "ci_high": fmt(row.get("ci_high")),
            "ci_excludes_zero": row.get("ci_excludes_zero", ""),
            "p_value": fmt(row.get("p_value"), signed=False),
            "interpretation": row.get("interpretation", ""),
        })
    write_csv(table_dir / "table_5_mechanism_and_opportunity.csv", rows)
    write_markdown_table(table_dir / "table_5_mechanism_and_opportunity.md", rows)
    return rows


def build_operational_tables(evidence_dir: Path, table_dir: Path) -> List[Dict[str, object]]:
    operational = read_csv(evidence_dir / "operational_block.csv")
    rows: List[Dict[str, object]] = []
    for row in operational:
        rows.append({
            "model": model_label(row.get("model", "")),
            "subset": row.get("subset", "all") or "all",
            "role": row.get("operational_role", ""),
            "brier": fmt(row.get("brier"), signed=False),
            "bss": fmt(row.get("bss")),
            "roc_auc": fmt(row.get("roc_auc"), signed=False),
            "reliability_slope": fmt(row.get("reliability_slope"), signed=False),
            "ece": fmt(row.get("ece"), signed=False),
            "valid_cells": int(f(row.get("valid_count")) if math.isfinite(f(row.get("valid_count"))) else 0),
        })
    write_csv(table_dir / "table_6_operational_metrics.csv", rows)
    write_markdown_table(table_dir / "table_6_operational_metrics.md", rows)
    return rows


def build_threshold_operating_table(stack_dir: Path, table_dir: Path) -> List[Dict[str, object]]:
    scores, _ = extract_head_to_head(stack_dir)
    rows: List[Dict[str, object]] = []
    for model in (ENS_MODEL, HEATCAST_MODEL, STACK_MODEL):
        row = scores[model]
        for threshold in PROBABILITY_THRESHOLDS:
            rows.append({
                "model": model_label(model),
                "probability_threshold": threshold,
                "hit_rate": fmt(row.get(f"hit_rate_{threshold}"), signed=False),
                "false_alarm_ratio": fmt(row.get(f"false_alarm_ratio_{threshold}"), signed=False),
                "bss": fmt(row.get("bss_vs_monthly_climo")),
                "roc_auc": fmt(row.get("roc_auc"), signed=False),
                "pr_auc": fmt(row.get("pr_auc"), signed=False),
            })
    write_csv(table_dir / "table_7_probability_threshold_operating_points.csv", rows)
    write_markdown_table(table_dir / "table_7_probability_threshold_operating_points.md", rows)
    return rows


def build_opportunity_probability_table(stack_dir: Path, table_dir: Path) -> List[Dict[str, object]]:
    rows = read_csv(stack_dir / "opportunity_pair_summary.csv", required=False)
    by_subset_model = {(row.get("subset", ""), row.get("model", "")): row for row in rows}
    subset_labels = {
        "all": "All paired cases",
        "heatcast_top10_confidence": "Top 10% confidence",
        "heatcast_low_sigma_tercile": "Low-sigma tercile",
        "heatcast_top10_and_low_sigma": "Top confidence + low sigma",
    }
    output: List[Dict[str, object]] = []
    for subset, label in subset_labels.items():
        ens = by_subset_model.get((subset, ENS_MODEL))
        stack = by_subset_model.get((subset, STACK_MODEL))
        if not ens or not stack:
            continue
        output.append({
            "subset": label,
            "stack_bss": fmt(stack.get("bss_vs_monthly_climo")),
            "ens_bss": fmt(ens.get("bss_vs_monthly_climo")),
            "delta_bss_stack_minus_ens": fmt(f(stack.get("bss_vs_monthly_climo")) - f(ens.get("bss_vs_monthly_climo"))),
            "stack_roc_auc": fmt(stack.get("roc_auc"), signed=False),
            "ens_roc_auc": fmt(ens.get("roc_auc"), signed=False),
            "delta_roc_auc_stack_minus_ens": fmt(f(stack.get("roc_auc")) - f(ens.get("roc_auc"))),
            "stack_pr_auc": fmt(stack.get("pr_auc"), signed=False),
            "ens_pr_auc": fmt(ens.get("pr_auc"), signed=False),
            "stack_ece": fmt(stack.get("ece"), signed=False),
            "ens_ece": fmt(ens.get("ece"), signed=False),
            "delta_ece_stack_minus_ens": fmt(f(stack.get("ece")) - f(ens.get("ece"))),
            "valid_cells": int(f(stack.get("valid_count"))) if math.isfinite(f(stack.get("valid_count"))) else 0,
        })
    write_csv(table_dir / "table_8_opportunity_probability_metrics.csv", output)
    write_markdown_table(table_dir / "table_8_opportunity_probability_metrics.md", output)
    return output


def plot_headline(stack_dir: Path, fig_dir: Path) -> None:
    plt = ensure_matplotlib()
    scores, boot = extract_head_to_head(stack_dir)
    models = [ENS_MODEL, HEATCAST_MODEL, STACK_MODEL]
    labels = [model_label(m) for m in models]
    bss = [f(scores[m]["bss_vs_monthly_climo"]) for m in models]
    auc = [f(scores[m]["roc_auc"]) for m in models]
    slope = [f(scores[m]["reliability_slope"]) for m in models]
    ece = [f(scores[m]["ece"]) for m in models]

    fig, axes = plt.subplots(1, 3, figsize=figure_size(DOUBLE_COLUMN_MM, 70.0))
    colors = [system_color(model) for model in models]
    axes[0].bar(labels, bss, color=colors)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Brier skill score")
    axes[0].set_title("Probabilistic skill")
    axes[1].bar(labels, auc, color=colors)
    axes[1].set_ylim(0.5, max(0.76, max(auc) + 0.02))
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title("Discrimination")
    x = range(len(labels))
    axes[2].plot(x, slope, marker="o", label="slope", color="#B279A2")
    axes[2].plot(x, ece, marker="s", label="ECE", color="#E45756")
    axes[2].axhline(1.0, color="#B279A2", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[2].set_xticks(list(x), labels, rotation=0)
    axes[2].set_title("Calibration")
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelrotation=20)
    fig.suptitle("W34 heat-exceedance skill on identical ENS/HeatCast cases", y=1.03)
    savefig(fig, fig_dir / "figure_1_headline_skill")
    plt.close(fig)

    boot_rows = [row for row in boot if "heatcast_ens_stack_minus_ens_calibrated" in row.get("metric", "")]
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ylabels = []
    points = []
    lows = []
    highs = []
    for row in boot_rows:
        metric = "BSS" if "delta_bss" in row["metric"] else "AUC"
        ylabels.append(f"Stack - ENS {metric}")
        points.append(f(row["point_estimate"]))
        lows.append(f(row["ci_low"]))
        highs.append(f(row["ci_high"]))
    y = list(range(len(points)))
    ax.errorbar(points, y, xerr=[[p - lo for p, lo in zip(points, lows)], [hi - p for p, hi in zip(points, highs)]], fmt="o", color=system_color(STACK_MODEL), capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, ylabels)
    ax.set_xlabel("Difference versus calibrated ENS")
    ax.set_title("Year-block bootstrap confidence intervals")
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, fig_dir / "figure_2_headline_stack_minus_ens_ci")
    plt.close(fig)


def plot_robustness(evidence_dir: Path, stack_dir: Path, fig_dir: Path) -> None:
    plt = ensure_matplotlib()
    robustness = read_csv(evidence_dir / "robustness_block.csv")
    month = sorted([r for r in robustness if r.get("group_type") == "month"], key=lambda r: int(r["group_value"]))
    loo = [r for r in robustness if str(r.get("group_type", "")).startswith("leave_one_")]
    region_boot = [
        r for r in read_csv(stack_dir / "robustness_region_bootstrap.csv", required=False)
        if r.get("candidate_model") == STACK_MODEL and r.get("metric", "").startswith("delta_bss")
    ]
    region_boot = sorted(region_boot, key=lambda r: f(r["point_estimate"]))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    axes[0].bar([r["group_value"] for r in month], [f(r["delta_bss"]) for r in month], color="#72B7B2")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Target center month")
    axes[0].set_ylabel("Stack - ENS BSS")
    axes[0].set_title("Monthly robustness")

    labels = [str(r["comparison_set"]).replace("region_", "") for r in region_boot]
    points = [f(r["point_estimate"]) for r in region_boot]
    lows = [f(r["ci_low"]) for r in region_boot]
    highs = [f(r["ci_high"]) for r in region_boot]
    y = list(range(len(labels)))
    axes[1].errorbar(points, y, xerr=[[p - lo for p, lo in zip(points, lows)], [hi - p for p, hi in zip(points, highs)]], fmt="o", color=system_color(ENS_MODEL), capsize=2)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_yticks(y, labels)
    axes[1].set_xlabel("Stack - ENS BSS")
    axes[1].set_title("Regional bootstrap")

    loo_groups = ["leave_one_fold", "leave_one_month", "leave_one_year"]
    loo_vals = [
        min(f(r["delta_bss"]) for r in loo if r["group_type"] == group)
        for group in loo_groups
    ]
    axes[2].bar(["fold", "month", "year"], loo_vals, color=system_color(STACK_MODEL))
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Minimum leave-one-out BSS")
    axes[2].set_title("Dominance checks")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Stack-vs-ENS robustness", y=1.03)
    savefig(fig, fig_dir / "figure_3_robustness")
    plt.close(fig)


def plot_mechanism(stack_dir: Path, fig_dir: Path) -> None:
    plt = ensure_matplotlib()
    opportunity = [
        r for r in read_csv(stack_dir / "opportunity_pair_bootstrap.csv")
        if r.get("candidate_model") == STACK_MODEL and r.get("metric", "").startswith("delta_bss")
    ]
    driver_parent = [
        r for r in read_csv(stack_dir / "driver_pair_parent_bootstrap.csv", required=False)
        if r.get("metric") == "delta_bss_stack_vs_ens_child_minus_parent"
    ]
    selected_driver = [
        r for r in driver_parent
        if (
            r.get("interaction_axis") == "mjo_phase_x_top_confidence" and r.get("interaction_stratum") == "phase_8__top_10pct_ge_p90"
        )
        or (
            r.get("interaction_axis") == "mjo_phase_x_low_sigma" and r.get("interaction_stratum") == "phase_8__bottom_sigma_tercile"
        )
        or (
            r.get("interaction_axis") == "soil_moisture_tercile_x_top_confidence" and r.get("interaction_stratum") == "dry__top_10pct_ge_p90"
        )
    ]

    opportunity_order = {
        "all": 0,
        "heatcast_top10_confidence": 1,
        "heatcast_low_sigma_tercile": 2,
        "heatcast_top10_and_low_sigma": 3,
    }
    opportunity = sorted(opportunity, key=lambda row: opportunity_order.get(row["comparison_set"], 99))
    opportunity_labels = {
        "all": "All paired cases",
        "heatcast_top10_confidence": "Top 10% confidence",
        "heatcast_low_sigma_tercile": "Low-sigma tercile",
        "heatcast_top10_and_low_sigma": "Top confidence + low sigma",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True)
    labels = [opportunity_labels.get(r["comparison_set"], r["comparison_set"]) for r in opportunity]
    points = [f(r["point_estimate"]) for r in opportunity]
    lows = [f(r["ci_low"]) for r in opportunity]
    highs = [f(r["ci_high"]) for r in opportunity]
    y = list(range(len(labels)))
    axes[0].errorbar(points, y, xerr=[[p - lo for p, lo in zip(points, lows)], [hi - p for p, hi in zip(points, highs)]], fmt="o", color=system_color(STACK_MODEL), capsize=2)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel("Stack - ENS BSS")
    axes[0].set_title("Operational opportunity strata")

    def driver_label(row: Mapping[str, str]) -> str:
        axis = row["interaction_axis"]
        stratum = row["interaction_stratum"]
        parent = row["parent_kind"]
        if axis.startswith("soil_moisture_tercile") and stratum.startswith("dry__"):
            child = "Dry soil + top confidence"
        elif axis.endswith("_x_top_confidence") and stratum.startswith("phase_8__"):
            child = "MJO phase 8 + top confidence"
        elif axis.endswith("_x_low_sigma") and stratum.startswith("phase_8__"):
            child = "MJO phase 8 + low sigma"
        else:
            child = f"{axis}: {stratum}"
        parent_text = "selection parent" if parent.startswith("selection_parent") else "driver parent"
        return f"{child}\nvs {parent_text}"

    selected_driver = sorted(
        selected_driver,
        key=lambda row: (
            0 if row["interaction_axis"].startswith("soil_moisture") else 1,
            row["interaction_axis"],
            row["parent_kind"],
        ),
    )
    labels = [driver_label(r) for r in selected_driver]
    points = [f(r["point_estimate"]) for r in selected_driver]
    lows = [f(r["ci_low"]) for r in selected_driver]
    highs = [f(r["ci_high"]) for r in selected_driver]
    y = list(range(len(labels)))
    axes[1].errorbar(points, y, xerr=[[p - lo for p, lo in zip(points, lows)], [hi - p for p, hi in zip(points, highs)]], fmt="o", color=system_color(HEATCAST_MODEL), capsize=2)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_yticks(y, labels)
    axes[1].set_xlabel("Child-parent Stack-vs-ENS BSS")
    axes[1].set_title("Driver interaction tests")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Opportunity and driver evidence", y=1.02)
    savefig(fig, fig_dir / "figure_4_opportunity_and_driver_tests")
    plt.close(fig)


def plot_probability_scorecard(stack_dir: Path, fig_dir: Path) -> None:
    plt = ensure_matplotlib()
    scores, _ = extract_head_to_head(stack_dir)
    models = [ENS_MODEL, HEATCAST_MODEL, STACK_MODEL]
    metrics = [
        ("Brier", "brier", "lower"),
        ("BSS", "bss_vs_monthly_climo", "higher"),
        ("ROC-AUC", "roc_auc", "higher"),
        ("PR-AUC", "pr_auc", "higher"),
        ("Slope", "reliability_slope", "target_one"),
        ("ECE", "ece", "lower"),
    ]
    raw = [[f(scores[model].get(column)) for _, column, _ in metrics] for model in models]
    normalized: List[List[float]] = []
    for row_idx, _ in enumerate(models):
        normalized.append([])
        for col_idx, (_, _, direction) in enumerate(metrics):
            column = [raw[r][col_idx] for r in range(len(models))]
            finite = [value for value in column if math.isfinite(value)]
            if not finite or max(finite) == min(finite):
                normalized[row_idx].append(0.5)
                continue
            value = raw[row_idx][col_idx]
            if direction == "target_one":
                distances = [abs(item - 1.0) for item in finite]
                distance = abs(value - 1.0)
                score = 1.0 - ((distance - min(distances)) / (max(distances) - min(distances) + 1e-12))
            else:
                score = (value - min(finite)) / (max(finite) - min(finite))
            if direction == "lower":
                score = 1.0 - score
            normalized[row_idx].append(score)

    fig, ax = plt.subplots(figsize=(8.8, 3.4))
    image = ax.imshow(normalized, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics)), [name for name, _, _ in metrics])
    ax.set_yticks(range(len(models)), [model_label(model) for model in models])
    ax.set_title("Probabilistic performance scorecard")
    for i, model in enumerate(models):
        for j, (metric_name, _, _) in enumerate(metrics):
            value = raw[i][j]
            text = fmt(value, digits=3, signed=metric_name in {"BSS"})
            ax.text(j, i, text, ha="center", va="center", color="white" if normalized[i][j] < 0.45 else "black", fontsize=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Within-metric rank score")
    savefig(fig, fig_dir / "figure_5_probabilistic_scorecard")
    plt.close(fig)


def plot_threshold_operating_curves(stack_dir: Path, fig_dir: Path) -> None:
    plt = ensure_matplotlib()
    scores, _ = extract_head_to_head(stack_dir)
    thresholds = [float(threshold) for threshold in PROBABILITY_THRESHOLDS]
    models = [ENS_MODEL, HEATCAST_MODEL, STACK_MODEL]
    colors = {model: system_color(model) for model in models}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharex=True)
    for model in models:
        row = scores[model]
        hits = [f(row.get(f"hit_rate_{threshold:.1f}")) for threshold in thresholds]
        fars = [f(row.get(f"false_alarm_ratio_{threshold:.1f}")) for threshold in thresholds]
        axes[0].plot(thresholds, hits, marker="o", color=colors[model], label=model_label(model))
        axes[1].plot(thresholds, fars, marker="o", color=colors[model], label=model_label(model))
    axes[0].set_ylabel("Hit rate")
    axes[1].set_ylabel("False-alarm ratio")
    for ax in axes:
        ax.set_xlabel("Issued probability threshold")
        ax.set_xticks(thresholds)
        ax.set_ylim(-0.02, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Event detection")
    axes[1].set_title("False alarms")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Probability-threshold operating characteristics", y=1.03)
    savefig(fig, fig_dir / "figure_6_probability_threshold_operating_curves")
    plt.close(fig)


def plot_opportunity_probability_metrics(stack_dir: Path, fig_dir: Path) -> None:
    plt = ensure_matplotlib()
    rows = read_csv(stack_dir / "opportunity_pair_summary.csv", required=False)
    by_subset_model = {(row.get("subset", ""), row.get("model", "")): row for row in rows}
    subset_order = [
        ("all", "All"),
        ("heatcast_top10_confidence", "Top conf."),
        ("heatcast_low_sigma_tercile", "Low sigma"),
        ("heatcast_top10_and_low_sigma", "Top+low"),
    ]
    labels: List[str] = []
    delta_bss: List[float] = []
    delta_auc: List[float] = []
    delta_ece: List[float] = []
    for subset, label in subset_order:
        ens = by_subset_model.get((subset, ENS_MODEL))
        stack = by_subset_model.get((subset, STACK_MODEL))
        if not ens or not stack:
            continue
        labels.append(label)
        delta_bss.append(f(stack.get("bss_vs_monthly_climo")) - f(ens.get("bss_vs_monthly_climo")))
        delta_auc.append(f(stack.get("roc_auc")) - f(ens.get("roc_auc")))
        delta_ece.append(f(stack.get("ece")) - f(ens.get("ece")))

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), constrained_layout=True)
    x = list(range(len(labels)))
    axes[0].bar(x, delta_bss, color=system_color(STACK_MODEL))
    axes[0].set_ylabel("Stack - ENS BSS")
    axes[1].bar(x, delta_auc, color=system_color(ENS_MODEL))
    axes[1].set_ylabel("Stack - ENS ROC-AUC")
    axes[2].bar(x, delta_ece, color="#E45756")
    axes[2].set_ylabel("Stack - ENS ECE")
    axes[2].text(0.5, 0.93, "Lower is better", transform=axes[2].transAxes, ha="center", fontsize=8)
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Opportunity-regime probabilistic deltas", y=1.08)
    savefig(fig, fig_dir / "figure_7_opportunity_probability_metrics")
    plt.close(fig)


def write_methods(path: Path, git_commit: str) -> None:
    text = f"""# Methods Text Draft: HeatCast/ENS W34 Verification

## Target definition
The forecast target is probabilistic week-3-to-4 heat-exceedance risk over CONUS land grid cells. A heat exceedance is defined daily using month-specific 95th-percentile thresholds for MJJAS. Thresholds are estimated using training years only within each cross-validation fold. For W34 verification, daily lead predictions from days 15 through 28 are averaged to define a windowed-mean target and forecast, and the corresponding windowed threshold is also estimated from training-year windowed means only.

## Cross-validation and leakage control
All reported HeatCast quantities use leave-k-years-out folds. The model is trained on fold-specific training years, probability calibration and stacking are fit on validation years, and final scores are reported on held-out test years. Thresholds, base rates, quantile mappings, and calibration objects are fold-safe. Year-block bootstrap intervals resample whole calendar years, not individual grid cells.

## ECMWF S2S ENS benchmark
ECMWF S2S reforecasts are evaluated on exactly the same initialization dates, target years, target grid, land mask, W34 lead window, and exceedance thresholds used for HeatCast. ENS fields are regridded to the HeatCast/PRISM land grid for matched verification. Bias correction uses fold-safe, cycle-specific quantile mapping fit only on fold training years, and ENS probability calibration is fit on validation years. ENS is therefore a fair benchmark on a common verification grid, not a native-resolution comparison.

## HeatCast+ENS stack
The stack forecast is a validation-year logistic combination of calibrated ENS and HeatCast features. For each scored fold, the stacker excludes that fold from its fitting data. It is scored only on held-out test years. This tests whether HeatCast contributes incremental information beyond ENS, rather than whether HeatCast alone dominates ENS.

## Probabilistic diagnostics
The primary verification metrics are Brier skill score against fold-safe windowed climatology, reliability slope, expected calibration error, and paired ROC-AUC. Secondary operating diagnostics include PR-AUC and threshold-specific hit rates and false-alarm ratios at probability cutoffs 0.1, 0.2, 0.3, and 0.5. Opportunity-regime analyses are reported on the same paired HeatCast/ENS cases.

## Uncertainty
The primary uncertainty estimate is a paired year-block bootstrap over independent held-out calendar years. Reported confidence intervals therefore account for temporal dependence across grid cells within a year and avoid treating cell-days as independent replicates.

## Reproducibility
Package generated from git commit `{git_commit}` at {datetime.now(timezone.utc).isoformat()}.
"""
    path.write_text(text, encoding="utf-8")


def write_narrative(path: Path) -> None:
    text = """# Manuscript Narrative and Claim Boundaries

## Central claim
HeatCast alone is competitive with calibrated ECMWF S2S ENS but is not significantly better in the paired year-block bootstrap. The combined HeatCast+ENS stack is significantly better than calibrated ENS in both BSS and ROC-AUC. This supports the claim that HeatCast adds independent predictive information to ENS for W34 heat-exceedance risk.

## Operational claim
The result should be framed as calibrated probabilistic W34 heat-exceedance risk. The strongest operational evidence is the statistically significant Stack-vs-ENS improvement overall, in high-confidence forecasts, and in low-sigma forecasts.

## Spatial/physical claim
Regional gains are strongest and statistically resolved in the Great Plains, Midwest, and West. These can be discussed as the main spatial evidence. MJO phase-specific enhancement is not statistically resolved and should not be stated as a primary mechanism. Driver-stratified results should be used as exploratory context unless their paired Stack-vs-ENS parent comparisons exclude zero.

## Avoid these overclaims
- Do not say HeatCast alone beats ENS.
- Do not say MJO phase 8 explains the skill gain.
- Do not imply ENS is natively evaluated at 4 km; state that ENS is regridded and bias-corrected for matched verification.
- Do not describe the result as deterministic anomaly prediction skill. The paper target is probabilistic W34 exceedance risk.
"""
    path.write_text(text, encoding="utf-8")


def write_investigation_record(path: Path) -> None:
    text = """# HeatCast Investigation Record

1. Daily anomaly forecasting showed damped anomaly amplitudes and low temporal anomaly correlation despite visually smooth maps.
2. Weekly-average optimization was abandoned as a training target because it could reward smoothing and obscure daily anomaly failure.
3. Gradient-loss experiments on fold 2 did not materially improve daily TAC; gradient sharpness improved only slightly.
4. Stage-1 exceedance evaluation introduced fold-safe month-specific 95th-percentile thresholds, train-year-only base rates, reliability, Brier skill, and persistence/point-model/logistic baselines.
5. Stage-2 exceedance head and distributional CRPS experiments improved some uncertainty diagnostics but did not create a standalone HeatCast result clearly superior to ENS.
6. W34 tube training widened the target to leads 15-28 and shifted the scientific target to probabilistic windowed heat-exceedance risk.
7. ECMWF S2S ENS was downloaded, ingested, regridded, quantile-mapped, calibrated, and evaluated on the identical HeatCast fold/test cases.
8. Standalone HeatCast-C was competitive with calibrated ENS but not significantly better under paired year-block bootstrap.
9. A cross-fitted HeatCast+ENS stack significantly improved BSS and AUC over calibrated ENS.
10. Robustness checks showed positive Stack-vs-ENS BSS gain under leave-one-fold, leave-one-month, and leave-one-year tests.
11. Opportunity tests showed significant Stack-vs-ENS gains in high-confidence and low-sigma regimes.
12. Slow-driver tests did not resolve a significant MJO phase-8 enhancement; driver mechanism claims remain exploratory/contextual.
13. Paper-figure packaging added probability-focused panels for Brier/BSS/AUC/PR-AUC/calibration, threshold-specific hit and false-alarm behavior, and opportunity-regime probability metrics.

Scientific conclusion: the defensible Nature Communications narrative is incremental information and operational risk improvement, not standalone model dominance or a resolved MJO mechanism.
"""
    path.write_text(text, encoding="utf-8")


def git_output(args: Sequence[str], root: Path) -> str:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def write_reproducibility(
    path: Path,
    root: Path,
    stack_dir: Path,
    evidence_dir: Path,
    opportunity_dir: Path,
) -> str:
    commit = git_output(["rev-parse", "HEAD"], root)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_branch": git_output(["branch", "--show-current"], root),
        "git_status_short": git_output(["status", "--short"], root),
        "window_label": WINDOW_LABEL,
        "stack_dir": str(stack_dir),
        "evidence_dir": str(evidence_dir),
        "opportunity_dir": str(opportunity_dir),
        "source_tables": {
            "head_to_head": str(stack_dir / "heatcast_ens_stack_head_to_head.csv"),
            "opportunity_pair_bootstrap": str(stack_dir / "opportunity_pair_bootstrap.csv"),
            "driver_pair_bootstrap": str(stack_dir / "driver_pair_bootstrap.csv"),
            "driver_pair_parent_bootstrap": str(stack_dir / "driver_pair_parent_bootstrap.csv"),
            "mechanism_block": str(evidence_dir / "mechanism_block.csv"),
            "robustness_block": str(evidence_dir / "robustness_block.csv"),
            "operational_block": str(evidence_dir / "operational_block.csv"),
        },
        "slurm_scripts": sorted(str(path.name) for path in root.glob("submit_*.slurm")),
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return commit


def copy_reproducibility_files(root: Path, out_dir: Path, stack_dir: Path, evidence_dir: Path, opportunity_dir: Path) -> None:
    repro = out_dir / "reproducibility"
    repro.mkdir(parents=True, exist_ok=True)
    for script in root.glob("submit_*.slurm"):
        shutil.copy2(script, repro / script.name)
    for source in (
        stack_dir / "heatcast_ens_stack_head_to_head.csv",
        stack_dir / "opportunity_pair_summary.csv",
        stack_dir / "opportunity_pair_bootstrap.csv",
        stack_dir / "driver_pair_summary.csv",
        stack_dir / "driver_pair_bootstrap.csv",
        stack_dir / "driver_pair_parent_bootstrap.csv",
        stack_dir / "robustness_by_region.csv",
        stack_dir / "robustness_region_bootstrap.csv",
        stack_dir / "robustness_by_month.csv",
        stack_dir / "robustness_by_year.csv",
        stack_dir / "robustness_leave_one_out.csv",
        evidence_dir / "headline_block.csv",
        evidence_dir / "mechanism_block.csv",
        evidence_dir / "robustness_block.csv",
        evidence_dir / "operational_block.csv",
        evidence_dir / "paper_evidence_summary.md",
        opportunity_dir / "driver_opportunity_summary.csv",
        opportunity_dir / "driver_interaction_paired_bootstrap.csv",
    ):
        if source.exists():
            shutil.copy2(source, repro / source.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack_dir", default=f"ens_heatcast_stack_opportunity/{WINDOW_LABEL}")
    parser.add_argument("--evidence_dir", default=f"paper_evidence_blocks/{WINDOW_LABEL}")
    parser.add_argument("--opportunity_dir", default=f"exceedance_eval_incremental/opportunity_{WINDOW_LABEL}")
    parser.add_argument("--output_dir", default=f"paper_figures_tables/{WINDOW_LABEL}")
    args = parser.parse_args()

    root = Path.cwd()
    stack_dir = Path(args.stack_dir)
    evidence_dir = Path(args.evidence_dir)
    opportunity_dir = Path(args.opportunity_dir)
    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    text_dir = output_dir / "text"
    for directory in (fig_dir, table_dir, text_dir):
        directory.mkdir(parents=True, exist_ok=True)

    build_headline_tables(stack_dir, table_dir)
    build_robustness_tables(evidence_dir, stack_dir, table_dir)
    build_mechanism_tables(evidence_dir, table_dir)
    build_operational_tables(evidence_dir, table_dir)
    build_threshold_operating_table(stack_dir, table_dir)
    build_opportunity_probability_table(stack_dir, table_dir)
    plot_headline(stack_dir, fig_dir)
    plot_robustness(evidence_dir, stack_dir, fig_dir)
    plot_mechanism(stack_dir, fig_dir)
    plot_probability_scorecard(stack_dir, fig_dir)
    plot_threshold_operating_curves(stack_dir, fig_dir)
    plot_opportunity_probability_metrics(stack_dir, fig_dir)
    copy_reproducibility_files(root, output_dir, stack_dir, evidence_dir, opportunity_dir)
    commit = write_reproducibility(
        output_dir / "reproducibility_manifest.json",
        root,
        stack_dir,
        evidence_dir,
        opportunity_dir,
    )
    write_methods(text_dir / "methods_text_draft.md", commit)
    write_narrative(text_dir / "narrative_and_claim_boundaries.md")
    write_investigation_record(text_dir / "investigation_record.md")

    print("Paper figures/tables package complete")
    print(f"  output_dir={output_dir}")
    print(f"  figures={fig_dir}")
    print(f"  tables={table_dir}")
    print(f"  text={text_dir}")
    print(f"  reproducibility={output_dir / 'reproducibility'}")


if __name__ == "__main__":
    main()
