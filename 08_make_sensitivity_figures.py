#!/usr/bin/env python3
"""Draw the workflow, alpha intervals, and distance sensitivity figures."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd


INK = "#19252D"
GRAY = "#62737C"
BLUE = "#006889"
ORANGE = "#A54612"
GRID = "#DCE4E8"
FULL_WIDTH_INCHES = 170 / 25.4
PANELS = (("main", "Main", BLUE, "o"), ("ev13_sensitivity", "Nu\u0161tar", ORANGE, "s"))


def save(fig, outdir, name):
    fig.savefig(outdir / f"{name}.pdf", facecolor="white", metadata={"CreationDate": None})
    fig.savefig(outdir / f"{name}.png", dpi=300, facecolor="white")
    plt.close(fig)


def workflow(outdir):
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_INCHES, 2.55))
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.07, top=0.96)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    columns = ("Input", "Analysis", "Output")
    x_start = (0.01, 0.35, 0.69)
    for left, title in zip(x_start, columns):
        ax.text(left + 0.15, 0.93, title, ha="center", va="center", weight="semibold", color=INK)
    rows = (
        (0.53, BLUE, "#EDF6F8", (
            ("Published qpAdm results", "Coefficients and\nstandard errors"),
            ("Uncertainty propagation", "Correlation grid and\nsum-to-one covariance"),
            ("Croatia Iron Age share", "Median and 95% interval\nfor each focal individual"),
        )),
        (0.08, ORANGE, "#FCF3EC", (
            ("Public ENA alignments", "Two fixed panels of\nnine ancient individuals"),
            ("Complete-case allele calls", "Original calls, stricter QC\nand transversion subset"),
            ("Four distance contrasts", "Stability under SNP and\ngenomic block resampling"),
        )),
    )
    for bottom, color, background, cells in rows:
        for i, (left, (title, detail)) in enumerate(zip(x_start, cells)):
            box = FancyBboxPatch((left, bottom), 0.30, 0.33, boxstyle="round,pad=0.004,rounding_size=0.013",
                                 linewidth=0.85, edgecolor=color, facecolor=background)
            ax.add_patch(box)
            ax.text(left + 0.15, bottom + 0.225, title, ha="center", va="center",
                    color=color, weight="semibold", fontsize=9)
            ax.text(left + 0.15, bottom + 0.105, detail, ha="center", va="center", color=INK, fontsize=9)
            if i < 2:
                ax.annotate("", xy=(x_start[i + 1] - 0.005, bottom + 0.165), xytext=(left + 0.308, bottom + 0.165),
                            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.05, "shrinkA": 1, "shrinkB": 1})
    save(fig, outdir, "figure_analysis_workflow")


def alpha_intervals(base, outdir):
    folder = base / "results" / "alpha_covariance"
    summary = pd.read_csv(folder / "alpha_summary.csv")
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(FULL_WIDTH_INCHES, 3.12))
    fig.subplots_adjust(left=0.26, right=0.95, top=0.92, bottom=0.25, hspace=0.50)
    for ax, (individual, group) in zip(axes, summary.groupby("individual", sort=False)):
        group = pd.concat([group.loc[group["rho_setting"].eq("sum_to_one")],
                           group.loc[~group["rho_setting"].eq("sum_to_one")]])
        y = np.arange(len(group))[::-1]
        labels = []
        for height, row in zip(y, group.itertuples(index=False)):
            inferred = row.rho_setting == "sum_to_one"
            color = BLUE if inferred else GRAY
            labels.append(f"Sum-to-one ({row.rho:.3f})" if inferred else f"Assumed ({row.rho:.2f})")
            ax.hlines(height, row.lower_95, row.upper_95, color=color, lw=2.1 if inferred else 1.4, zorder=3)
            ax.plot(row.median_alpha, height, "o", color=color, markersize=5.4 if inferred else 4.5, zorder=4)
        ax.axvline(group["published_alpha"].iloc[0], color=GRAY, linestyle=":", linewidth=1.2, zorder=2)
        ax.set(yticks=y, yticklabels=labels, ylim=(-0.55, max(y) + 0.55), xlim=(0, 1))
        ax.set_title(individual, loc="left", weight="semibold", pad=7)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.grid(axis="x", color=GRID, linewidth=0.7)
        ax.tick_params(axis="y", length=0)
        for name in ("top", "right", "left"):
            ax.spines[name].set_visible(False)
    axes[-1].set_xlabel("Croatia Iron Age share of the modeled non-eastern component", labelpad=4)
    legend = (
        Line2D([0], [0], color=BLUE, marker="o", linewidth=2, label="Sum-to-one interval"),
        Line2D([0], [0], color=GRAY, linestyle=":", label="Ratio of published coefficients"),
    )
    fig.legend(handles=legend, loc="lower center", frameon=False, ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.025))
    save(fig, outdir, "figure_alpha_correlation_intervals")


def contrast_inputs(base):
    original = pd.read_csv(base / "results" / "block_bootstrap" / "contrast_summary.csv")
    snp = pd.read_csv(base / "baseline" / "snp_bootstrap_summary.csv")
    observed = original.loc[original["block_size_cm"].eq(5), ["panel", "metric", "observed"]]
    snp = snp.rename(columns={"ci_lower_95": "lower_95", "ci_upper_95": "upper_95"}).merge(observed, on=["panel", "metric"])
    filtered = pd.read_csv(base / "results" / "quality_control" / "filtered_block_bootstrap" / "contrast_summary.csv")
    transversions = pd.read_csv(base / "results" / "quality_control" / "transversions_block_bootstrap" / "contrast_summary.csv")
    tables = (("Original, SNP bootstrap", snp), ("Original, 5 cM blocks", original),
              ("Stricter QC, 5 cM blocks", filtered), ("Transversions, 5 cM blocks", transversions))
    return [(label, frame.loc[frame["block_size_cm"].eq(5)] if "block_size_cm" in frame else frame) for label, frame in tables]


def distance_sensitivity(base, outdir):
    conditions = contrast_inputs(base)
    mismatch_label = "Mismatch difference (\u00d71,000)"
    contrasts = (
        ("loo_statistic", "a  Leave-one-out contrast", mismatch_label, 1000),
        ("sirmium_gradient_r", "b  Distance from Sirmium", "Pearson correlation (r)", 1),
        ("ev13_difference", "c  E-V13 contrast", mismatch_label, 1000),
        ("homogenization_separation", "d  Between minus within", mismatch_label, 1000),
    )
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH_INCHES, 5.0))
    fig.subplots_adjust(left=0.28, right=0.955, top=0.91, bottom=0.19, wspace=0.34, hspace=0.54)
    for index, (ax, (metric, title, label, scale)) in enumerate(zip(axes.flat, contrasts)):
        y = np.arange(len(conditions))[::-1]
        for height, (_, frame) in zip(y, conditions):
            for offset, (panel, _, color, marker) in zip((0.11, -0.11), PANELS):
                row = frame.loc[frame["panel"].eq(panel) & frame["metric"].eq(metric)].iloc[0]
                ax.hlines(height + offset, scale * row.lower_95, scale * row.upper_95, color=color, linewidth=1.4)
                ax.plot(scale * row.observed, height + offset, color=color, marker=marker, markersize=4, linestyle="none")
        ax.axvline(0, color=GRAY, linewidth=0.95, linestyle="--", zorder=0)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set(yticks=y, ylim=(-0.45, max(y) + 0.45), xlabel=label)
        if scale != 1:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_yticklabels([condition for condition, _ in conditions] if index % 2 == 0 else [])
        ax.set_title(title, loc="left", weight="semibold", fontsize=10, pad=7)
        ax.tick_params(axis="y", length=0)
        for name in ("top", "right", "left"):
            ax.spines[name].set_visible(False)
    handles = [Line2D([0], [0], color=color, marker=marker, label=label) for _, label, color, marker in PANELS]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.045))
    save(fig, outdir, "figure_distance_sensitivity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path.cwd(), help="Repository root.")
    parser.add_argument("--outdir", type=Path, help="Output directory. Default: figures.")
    parser.add_argument("--figure", choices=("workflow", "alpha", "distance"), action="append", help="Figure to draw; defaults to all three.")
    args = parser.parse_args()
    base = args.base.resolve()
    outdir = args.outdir.resolve() if args.outdir else base / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42})
    selected = args.figure or ("workflow", "alpha", "distance")
    if "workflow" in selected:
        workflow(outdir)
    if "alpha" in selected:
        alpha_intervals(base, outdir)
    if "distance" in selected:
        distance_sensitivity(base, outdir)


if __name__ == "__main__":
    main()
