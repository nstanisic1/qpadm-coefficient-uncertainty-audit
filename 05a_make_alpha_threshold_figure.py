#!/usr/bin/env python3
"""
05a_make_alpha_threshold_figure.py
Render Supplementary Figure S4 for the alpha-threshold sensitivity analysis.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


COL_WEAK = "#D55E00"
COL_MODERATE = "#0072B2"
COL_STRONG = "#009E73"
COL_COMPARATOR = "#5E3C99"
COL_GREY = "#8F8F8F"
COL_DARK = "#222222"
GRID = "#E3E3E3"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

FLOOR_LABELS = {
    0.05: "Weak floor",
    0.10: "Moderate floor",
    0.25: "Strong floor",
}
FLOOR_COLORS = {
    0.05: COL_WEAK,
    0.10: COL_MODERATE,
    0.25: COL_STRONG,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".", help="Project root")
    return ap.parse_args()


def _to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def load_inputs(base: Path):
    outdir = base / "results" / "alpha_threshold_sensitivity"
    p_worst = outdir / "alpha_threshold_worst_rho_summary.csv"
    p_dir = outdir / "directional_support_grid.csv"
    p_sum = outdir / "alpha_threshold_sensitivity_summary.csv"

    for p in [p_worst, p_dir, p_sum]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    worst = pd.read_csv(p_worst)
    directional = pd.read_csv(p_dir)
    summary = pd.read_csv(p_sum)
    return worst, directional, summary


def relpath(base: Path, path: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def extract_comparator_median(summary: pd.DataFrame) -> float:
    for metric in [
        "comparator_same_axis::median_alpha",
        "comparator_all_comparators::median_alpha",
        "comparator_related_axis::median_alpha",
    ]:
        hit = summary.loc[summary["metric"] == metric, "value"]
        if len(hit):
            return float(hit.iloc[0])
    return 0.414


def extract_survival_by_floor(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pat = re.compile(r"^floor_(0\.\d+)::highest_threshold_supported_all_rho$")
    for _, row in summary.iterrows():
        m = pat.match(str(row["metric"]))
        if m:
            rows.append({
                "prob_floor": float(m.group(1)),
                "highest_threshold_all_rho": float(row["value"]),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No survival rows found in alpha_threshold_sensitivity_summary.csv")
    return out.sort_values("prob_floor").reset_index(drop=True)


def prepare_worst(worst: pd.DataFrame) -> pd.DataFrame:
    w = worst.copy()
    for c in ["prob_floor", "threshold", "worst_rho_min_focal_prob"]:
        w[c] = pd.to_numeric(w[c], errors="raise")
    w["all_rho_threshold_survives"] = w["all_rho_threshold_survives"].map(_to_bool)
    w = w[w["prob_floor"].round(2).isin([0.05, 0.10, 0.25])].copy()

    sort_cols = [c for c in ["prob_floor", "threshold", "threshold_source", "threshold_label"] if c in w.columns]
    w = w.sort_values(sort_cols)
    w = w.groupby(["prob_floor", "threshold"], as_index=False).first()
    return w


def prepare_directional(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    out["panel_pretty"] = out["panel"].map({
        "main_I27297": "Main panel\n(I27297)",
        "ev13_sensitivity_I28388": "Nustar sensitivity\n(I28388)",
    }).fillna(out["panel"])

    out["directional_consistency_score"] = pd.to_numeric(
        out["directional_consistency_score"], errors="raise"
    )

    for strict_col in [
        "fraction_directionally_consistent_and_snp_bootstrap_interval_excludes_zero",
        "fraction_directionally_consistent_and_ci_excludes_zero",
        "fraction_directionally_consistent",
    ]:
        if strict_col in out.columns:
            break
    out["strict_support"] = pd.to_numeric(out[strict_col], errors="raise")
    return out[["panel_pretty", "directional_consistency_score", "strict_support"]]


def frac_label(v: float) -> str:
    if np.isclose(v, 1.0):
        return "4/4"
    if np.isclose(v, 0.75):
        return "3/4"
    return f"{v:.2f}"


def make_figure(base: Path):
    worst, directional, summary = load_inputs(base)
    worst = prepare_worst(worst)
    directional = prepare_directional(directional)
    survival = extract_survival_by_floor(summary)
    comparator_x = extract_comparator_median(summary)

    figdir = base / "figures"
    figdir.mkdir(exist_ok=True, parents=True)

    fig = plt.figure(figsize=(9.5, 9.2), dpi=200)
    gs = fig.add_gridspec(
    2, 2,
    height_ratios=[1.45, 1.08],
    hspace=0.50,
    wspace=0.55
    )

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    probability_curve = (
        worst.sort_values(["threshold", "prob_floor"])
        .groupby("threshold", as_index=False)
        .first()
        .sort_values("threshold")
    )
    curve_handle, = ax1.plot(
        probability_curve["threshold"],
        probability_curve["worst_rho_min_focal_prob"],
        color=COL_DARK,
        marker="o",
        markersize=5.5,
        linewidth=2.4,
        label="Worst-rho minimum focal probability",
        zorder=3
    )

    for floor in [0.05, 0.10, 0.25]:
        ax1.axhline(
            floor,
            color=FLOOR_COLORS[floor],
            linestyle=(0, (3, 3)),
            linewidth=1.0,
            alpha=0.55,
            zorder=1
        )

    ax1.axvline(
        comparator_x,
        color=COL_COMPARATOR,
        linestyle=(0, (2, 2)),
        linewidth=1.2,
        zorder=2
    )

    ax1.annotate(
        "Roman/late-antique\ncomparator median",
        xy=(comparator_x, 0.88),
        xytext=(comparator_x + 0.010, 0.93),
        ha="left",
        va="top",
        fontsize=8.8,
        color=COL_COMPARATOR,
        arrowprops=dict(
            arrowstyle="-",
            color=COL_COMPARATOR,
            lw=0.9,
            shrinkA=0,
            shrinkB=0
        )
    )

    label_positions = {
        0.25: 0.275,
        0.10: 0.122,
        0.05: 0.072,
    }
    for floor in [0.25, 0.10, 0.05]:
        ax1.text(
            0.495,
            label_positions[floor],
            FLOOR_LABELS[floor],
            color=FLOOR_COLORS[floor],
            ha="right",
            va="center",
            fontsize=8.6
        )

    ax1.set_xlim(0.18, 0.52)
    ax1.set_ylim(0.0, 1.02)
    ax1.set_xticks([0.20, 0.25, 0.333333, 0.40, 0.50])
    ax1.set_xticklabels(["0.20", "0.25", "0.33", "0.40", "0.50"])
    ax1.set_xlabel(
        r"Local-alpha threshold ($\alpha \geq t$)",
        fontsize=12.0,
        fontweight="bold"
    )
    ax1.set_ylabel(
        "Worst-rho minimum focal probability",
        fontsize=12.0,
        fontweight="bold"
    )
    ax1.set_title(
        "a  Alpha-threshold sensitivity",
        loc="left",
        fontsize=13.5,
        fontweight="bold",
        pad=10
    )
    ax1.grid(axis="y", color=GRID, linewidth=0.8)
    ax1.spines[["top", "right"]].set_visible(False)
    floor_handles = [
        Line2D([0], [0], color=FLOOR_COLORS[floor], lw=1.2, linestyle=(0, (3, 3)), label=FLOOR_LABELS[floor])
        for floor in [0.05, 0.10, 0.25]
    ]
    ax1.legend(
        [curve_handle] + floor_handles,
        ["Worst-rho probability"] + [FLOOR_LABELS[floor] for floor in [0.05, 0.10, 0.25]],
        ncol=2,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.005),
        fontsize=9.2,
        handlelength=2.0,
        columnspacing=1.4
    )

    x = np.arange(len(survival))
    bar_colors = [FLOOR_COLORS.get(round(v, 2), COL_GREY) for v in survival["prob_floor"]]

    bars = ax2.bar(
        x,
        survival["highest_threshold_all_rho"],
        color=bar_colors,
        edgecolor=COL_DARK,
        linewidth=0.6
    )

    ax2.axhline(
        comparator_x,
        color=COL_COMPARATOR,
        linestyle=(0, (2, 2)),
        linewidth=1.25,
        alpha=0.95
    )

    CMP_LABEL_X = 1.03

    ax2.text(
        CMP_LABEL_X,
        comparator_x + 0.006,
        "Comparator\nmedian",
        transform=ax2.get_yaxis_transform(),
        color=COL_COMPARATOR,
        ha="left",
        va="bottom",
        fontsize=8.6,
        linespacing=0.9,
        clip_on=False,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="none",
            alpha=0.90
        )
    )

    for rect, val in zip(bars, survival["highest_threshold_all_rho"]):
        ax2.text(
            rect.get_x() + rect.get_width() / 2,
            float(val) + 0.014,
            f"{float(val):.3f}",
            ha="center",
            va="bottom",
            fontsize=9.0,
            fontweight="bold"
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [FLOOR_LABELS.get(round(v, 2), str(v)) for v in survival["prob_floor"]],
        fontsize=10.0
    )
    ax2.set_ylim(0, 0.58)
    ax2.set_ylabel(
        "Highest supported threshold",
        fontsize=11.8,
        fontweight="bold"
    )
    ax2.set_title(
        "b  All-rho support",
        loc="left",
        fontsize=12.8,
        fontweight="bold",
        pad=10
    )
    ax2.grid(axis="y", color=GRID, linewidth=0.8)
    ax2.spines[["top", "right"]].set_visible(False)

    x = np.arange(len(directional))
    width = 0.34

    b1 = ax3.bar(
        x - width / 2,
        directional["directional_consistency_score"],
        width,
        color="#16A085",
        edgecolor=COL_DARK,
        linewidth=0.6,
        label="Descriptive directional score"
    )
    b2 = ax3.bar(
        x + width / 2,
        directional["strict_support"],
        width,
        color=COL_GREY,
        edgecolor=COL_DARK,
        linewidth=0.6,
        label="Directional + interval support"
    )

    for grp in [b1, b2]:
        for rect in grp:
            h = float(rect.get_height())
            ax3.text(
                rect.get_x() + rect.get_width() / 2,
                h + 0.015,
                frac_label(h),
                ha="center",
                va="bottom",
                fontsize=9.0,
                fontweight="bold"
            )

    ax3.set_xticks(x)
    ax3.set_xticklabels(directional["panel_pretty"], fontsize=10.0)
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel(
        "Support fraction / score",
        fontsize=11.8,
        fontweight="bold"
    )
    ax3.set_title(
        "c  Allele-sharing support",
        loc="left",
        fontsize=12.8,
        fontweight="bold",
        pad=10
    )
    ax3.grid(axis="y", color=GRID, linewidth=0.8)
    ax3.spines[["top", "right"]].set_visible(False)

    ax3.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.17),
        fontsize=8.6,
        ncol=1,
        handlelength=1.8,
        borderaxespad=0.0
    )

    fig.suptitle(
        "Local-alpha threshold sensitivity analysis",
        fontsize=18,
        fontweight="bold",
        y=0.985
    )

    fig.text(
        0.5,
        0.018,
        "Autosomal compatibility audit only: this figure does not infer paternal-lineage origin, continuity, or homeland.",
        ha="center",
        va="bottom",
        fontsize=9.8,
        color="#555555"
    )

    fig.subplots_adjust(
        top=0.88,
        bottom=0.11,
        left=0.09,
        right=0.98
    )

    out_png = figdir / "figureS4_alpha_threshold_sensitivity.png"
    out_pdf = figdir / "figureS4_alpha_threshold_sensitivity.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "script": "05a_make_alpha_threshold_figure.py",
        "outputs": [relpath(base, out_png), relpath(base, out_pdf)],
        "notes": {
            "panel_a": "Worst-rho minimum focal probability curve across pre-specified local-alpha thresholds; custom x-axis ticks mark the analytical thresholds evaluated in the sensitivity analysis.",
            "panel_b": "Highest threshold supported across all rho values by floor.",
            "panel_c": "Independent allele-sharing support in the main and Nustar sensitivity panels."
        }
    }

    with open(figdir / "figureS4_alpha_threshold_sensitivity_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print("Figure written:")
    print(f"  - {out_png}")
    print(f"  - {out_pdf}")


if __name__ == "__main__":
    args = parse_args()
    make_figure(Path(args.base).resolve())
