#!/usr/bin/env python3
"""Shared journal figure styling for HeatCast paper products."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple


MM_PER_INCH = 25.4
SINGLE_COLUMN_MM = 89.0
DOUBLE_COLUMN_MM = 183.0

SYSTEM_COLORS = {
    "windowed_climatology": "#6F6F6F",
    "monthly_climatology": "#6F6F6F",
    "ens_raw_fraction": "#9ECAE1",
    "ens_calibrated": "#0072B2",
    "heatcast_C": "#D55E00",
    "heatcast_ens_stack": "#009E73",
}

SYSTEM_LABELS = {
    "windowed_climatology": "Climatology",
    "monthly_climatology": "Climatology",
    "ens_raw_fraction": "ENS raw",
    "ens_calibrated": "ENS calibrated",
    "heatcast_C": "HeatCast-C",
    "heatcast_ens_stack": "HeatCast+ENS",
}

TELECONNECTION_COLORS = {
    "AO": "#0072B2",
    "NAO": "#009E73",
    "PNA": "#D55E00",
    "PDO": "#CC79A7",
}

GROUP_COLORS = {
    "fold": "#0072B2",
    "month": "#009E73",
    "year": "#D55E00",
}


def mm_to_inch(value_mm: float) -> float:
    return float(value_mm) / MM_PER_INCH


def figure_size(width_mm: float = DOUBLE_COLUMN_MM, height_mm: float = 90.0) -> Tuple[float, float]:
    return (mm_to_inch(width_mm), mm_to_inch(height_mm))


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "figure.titlesize": 9.0,
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.9,
        "patch.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "savefig.transparent": False,
    })
    import matplotlib.pyplot as plt

    return plt


def system_color(model: str, fallback: str = "#333333") -> str:
    return SYSTEM_COLORS.get(str(model), fallback)


def system_label(model: str) -> str:
    return SYSTEM_LABELS.get(str(model), str(model))


def teleconnection_color(name: str, fallback: str = "#333333") -> str:
    return TELECONNECTION_COLORS.get(str(name).upper(), fallback)


def group_color(name: str, fallback: str = "#666666") -> str:
    return GROUP_COLORS.get(str(name), fallback)


def panel_label(ax, label: str, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        str(label).lower(),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def save_figure(fig, path_base: Path, dpi: int = 600) -> None:
    path_base = Path(path_base)
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=int(dpi), bbox_inches="tight")
