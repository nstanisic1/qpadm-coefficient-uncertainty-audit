#!/usr/bin/env python3
"""
Create publication-ready manuscript figures from saved outputs of Scripts 3 and 4.

This script is the dedicated figure-rendering stage of the reproducible
analysis pipeline. It reads saved outputs from:

    03_allele_sharing_panel.py
    04_identifiability_analysis.py

and creates publication-style figures without rerunning computational analyses.

Usage
-----
    python 05_make_figures.py --base .

Inputs
------
By default, this script expects:

    matrix_main/allele_sharing_summary.csv
    matrix_ev13_sensitivity/allele_sharing_summary.csv

    results/alpha_identifiability/alpha_summary.csv
    results/alpha_identifiability/alpha_draws.csv
    results/alpha_identifiability/comparators_normalized.csv

Outputs
-------
By default, this script writes:

    figures/figure1_alpha_density.png
    figures/figure1_alpha_density.pdf

    figures/figureS1_alpha_rho_sensitivity.png
    figures/figureS1_alpha_rho_sensitivity.pdf

    figures/figureS2_allele_sharing_contrasts_main.png
    figures/figureS2_allele_sharing_contrasts_main.pdf

    figures/figureS3_allele_sharing_contrasts_ev13_sensitivity.png
    figures/figureS3_allele_sharing_contrasts_ev13_sensitivity.pdf

    figures/figure_manifest.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


DEFAULT_BASE = Path.cwd()

COLOR_I15541 = "#D55E00"
COLOR_I15542 = "#009E73"
COLOR_SAME_AXIS = "#0072B2"
COLOR_RELATED_AXIS = "#8A8A8A"

ALLELE_SHARING_COLORS = {
    1: "#0072B2",
    2: "#D55E00",
    3: "#009E73",
    4: "#8A8A8A",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create publication-ready manuscript figures from saved outputs "
            "of the allele-sharing and alpha-identifiability analyses."
        )
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
        help="Project base directory. Default: current working directory.",
    )
    parser.add_argument(
        "--main-summary",
        type=Path,
        default=None,
        help=(
            "Path to matrix_main/allele_sharing_summary.csv. "
            "If omitted, the default path under --base is used."
        ),
    )
    parser.add_argument(
        "--sensitivity-summary",
        type=Path,
        default=None,
        help=(
            "Path to matrix_ev13_sensitivity/allele_sharing_summary.csv. "
            "If omitted, the default path under --base is used."
        ),
    )
    parser.add_argument(
        "--alpha-summary",
        type=Path,
        default=None,
        help=(
            "Path to results/alpha_identifiability/alpha_summary.csv. "
            "If omitted, the default path under --base is used."
        ),
    )
    parser.add_argument(
        "--alpha-draws",
        type=Path,
        default=None,
        help=(
            "Path to results/alpha_identifiability/alpha_draws.csv. "
            "If omitted, the default path under --base is used."
        ),
    )
    parser.add_argument(
        "--comparators",
        type=Path,
        default=None,
        help=(
            "Path to results/alpha_identifiability/comparators_normalized.csv. "
            "If omitted, the default path under --base is used."
        ),
    )
    parser.add_argument(
        "--figdir",
        type=Path,
        default=None,
        help="Directory for figure outputs. If omitted, '<base>/figures' is used.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    base = args.base.resolve()
    return {
        "base": base,
        "main_summary": (
            args.main_summary.resolve()
            if args.main_summary
            else base / "matrix_main" / "allele_sharing_summary.csv"
        ),
        "sensitivity_summary": (
            args.sensitivity_summary.resolve()
            if args.sensitivity_summary
            else base / "matrix_ev13_sensitivity" / "allele_sharing_summary.csv"
        ),
        "alpha_summary": (
            args.alpha_summary.resolve()
            if args.alpha_summary
            else base / "results" / "alpha_identifiability" / "alpha_summary.csv"
        ),
        "alpha_draws": (
            args.alpha_draws.resolve()
            if args.alpha_draws
            else base / "results" / "alpha_identifiability" / "alpha_draws.csv"
        ),
        "comparators": (
            args.comparators.resolve()
            if args.comparators
            else base / "results" / "alpha_identifiability" / "comparators_normalized.csv"
        ),
        "figdir": (
            args.figdir.resolve()
            if args.figdir
            else base / "figures"
        ),
    }


def configure_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 9.5
    plt.rcParams["ytick.labelsize"] = 9.5
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["figure.titlesize"] = 13


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} is empty: {path}")


def relpath(base: Path, path: Path | str) -> str:
    try:
        return Path(path).resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return str(path)


def validate_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def load_csv(path: Path, label: str) -> pd.DataFrame:
    require_file(path, label)
    return pd.read_csv(path)


def coerce_numeric(df: pd.DataFrame, columns: List[str], label: str) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="raise")
    return out


def normalize_axis_label(value: str) -> str:
    value_norm = str(value).strip().lower().replace("_", "-")
    if value_norm in {"same-axis", "same axis", "direct"}:
        return "same-axis"
    if value_norm in {"related-axis", "related axis", "contextual"}:
        return "related-axis"
    return str(value)


def load_alpha_inputs(paths: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alpha_summary = load_csv(paths["alpha_summary"], "Alpha summary CSV")
    validate_columns(
        alpha_summary,
        ["individual", "rho", "median_alpha", "ci_lower_95", "ci_upper_95"],
        "Alpha summary CSV",
    )
    alpha_summary = coerce_numeric(
        alpha_summary,
        ["rho", "median_alpha", "ci_lower_95", "ci_upper_95"],
        "Alpha summary CSV",
    )

    alpha_draws = load_csv(paths["alpha_draws"], "Alpha draws CSV")
    validate_columns(alpha_draws, ["individual", "rho"], "Alpha draws CSV")

    if "alpha_draw" in alpha_draws.columns:
        alpha_col = "alpha_draw"
    elif "alpha" in alpha_draws.columns:
        alpha_col = "alpha"
    else:
        raise ValueError("Alpha draws CSV must contain either 'alpha_draw' or 'alpha'.")

    alpha_draws = coerce_numeric(alpha_draws, ["rho", alpha_col], "Alpha draws CSV")
    alpha_draws = alpha_draws.rename(columns={alpha_col: "alpha"})

    comparators = load_csv(paths["comparators"], "Comparator CSV")
    validate_columns(
        comparators,
        ["individual", "site", "axis_type_normalized", "alpha_observed"],
        "Comparator CSV",
    )
    comparators["axis_type_normalized"] = comparators["axis_type_normalized"].map(normalize_axis_label)
    comparators = coerce_numeric(comparators, ["alpha_observed"], "Comparator CSV")

    for optional_col in ["ci_lower_95", "ci_upper_95", "se"]:
        if optional_col in comparators.columns:
            comparators[optional_col] = pd.to_numeric(comparators[optional_col], errors="coerce")

    return alpha_summary, alpha_draws, comparators


def load_allele_sharing_summary(path: Path, label: str) -> pd.DataFrame:
    df = load_csv(path, label)
    validate_columns(
        df,
        [
            "test_number",
            "test_name",
            "statistic_name",
            "statistic",
            "ci_lower_95",
            "ci_upper_95",
            "predicted_direction",
        ],
        label,
    )
    df = coerce_numeric(
        df,
        ["test_number", "statistic", "ci_lower_95", "ci_upper_95"],
        label,
    )
    if "p_value" in df.columns:
        df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    return df.sort_values("test_number").reset_index(drop=True)


def save_figure(fig: plt.Figure, output_base: Path) -> Dict[str, str]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def make_kde(values: np.ndarray, xmin: float = 0.0, xmax: float = 1.0, n_grid: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    clean_values = values[np.isfinite(values)]
    if len(clean_values) < 100:
        raise ValueError("Too few finite alpha draws for KDE.")
    grid = np.linspace(xmin, xmax, n_grid)
    kde = gaussian_kde(clean_values)
    density = kde(grid)
    return grid, density


def save_alpha_density_figure(
    alpha_summary: pd.DataFrame,
    alpha_draws: pd.DataFrame,
    comparators: pd.DataFrame,
    output_base: Path,
) -> Dict[str, str]:
    rho0_summary = alpha_summary[np.isclose(alpha_summary["rho"], 0.0)].copy()
    rho0_draws = alpha_draws[np.isclose(alpha_draws["rho"], 0.0)].copy()

    expected_individuals = ["I15541", "I15542"]
    for individual in expected_individuals:
        if individual not in set(rho0_draws["individual"]):
            raise ValueError(f"Alpha draws at rho=0 missing expected individual: {individual}")

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    fig.subplots_adjust(right=0.73)

    focal_specs = {
        "I15541": {"color": COLOR_I15541, "label": "I15541"},
        "I15542": {"color": COLOR_I15542, "label": "I15542"},
    }

    max_density = 0.0
    density_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for individual, spec in focal_specs.items():
        values = rho0_draws.loc[rho0_draws["individual"] == individual, "alpha"].to_numpy(dtype=float)
        grid, density = make_kde(values)
        density_cache[individual] = (grid, density)
        max_density = max(max_density, float(np.max(density)))

    for individual, spec in focal_specs.items():
        grid, density = density_cache[individual]
        ax.fill_between(grid, density, alpha=0.25, color=spec["color"], linewidth=0)
        ax.plot(grid, density, color=spec["color"], linewidth=2.2)

        row = rho0_summary.loc[rho0_summary["individual"] == individual]
        if len(row) != 1:
            raise ValueError(f"Alpha summary at rho=0 missing unique row for {individual}")

        median_alpha = float(row["median_alpha"].iloc[0])
        ax.axvline(
            median_alpha,
            color=spec["color"],
            linestyle=(0, (4, 2)),
            linewidth=1.5,
            alpha=0.95,
        )

    same_axis = comparators[comparators["axis_type_normalized"] == "same-axis"].copy()
    related_axis = comparators[comparators["axis_type_normalized"] == "related-axis"].copy()

    rug_height_same = 0.105 * max_density
    rug_height_related = 0.070 * max_density

    for _, row in related_axis.iterrows():
        val = float(row["alpha_observed"])
        ax.plot(
            [val, val],
            [0, rug_height_related],
            color=COLOR_RELATED_AXIS,
            linewidth=1.6,
            alpha=0.90,
            zorder=10,
        )

    for _, row in same_axis.iterrows():
        val = float(row["alpha_observed"])
        y_level = max_density * 0.115

        if pd.notnull(row.get("ci_lower_95")) and pd.notnull(row.get("ci_upper_95")):
            low = float(row["ci_lower_95"])
            high = float(row["ci_upper_95"])
            cap_h = max_density * 0.018

            ax.plot(
                [low, high],
                [y_level, y_level],
                color=COLOR_SAME_AXIS,
                linewidth=2.0,
                zorder=11,
            )
            ax.plot(
                [low, low],
                [y_level - cap_h, y_level + cap_h],
                color=COLOR_SAME_AXIS,
                linewidth=1.6,
                zorder=11,
            )
            ax.plot(
                [high, high],
                [y_level - cap_h, y_level + cap_h],
                color=COLOR_SAME_AXIS,
                linewidth=1.6,
                zorder=11,
            )

        ax.plot(
            [val],
            [y_level],
            marker="o",
            color=COLOR_SAME_AXIS,
            markersize=5.8,
            zorder=12,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max_density * 1.25 if max_density > 0 else 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel(
        r"$\alpha$ = Croatia_IA / (Croatia_IA + WestAnatolia_Roman_Byzantine)",
        fontweight="semibold",
    )
    ax.set_ylabel("Density", fontweight="semibold")
    ax.set_title(
        "Local-alpha uncertainty in published coefficients",
        fontweight="semibold",
        pad=10,
    )

    legend_handles: List[Line2D] = []
    for individual, spec in focal_specs.items():
        row = rho0_summary.loc[rho0_summary["individual"] == individual]
        median_alpha = float(row["median_alpha"].iloc[0])
        ci_low = float(row["ci_lower_95"].iloc[0])
        ci_high = float(row["ci_upper_95"].iloc[0])

        label = f"{individual}: median {median_alpha:.2f}; 95% propagation interval [{ci_low:.2f}, {ci_high:.2f}]"
        legend_handles.append(Line2D([0], [0], color=spec["color"], lw=2.2, label=label))

    legend_handles.extend(
        [
            Line2D([0], [0], color=COLOR_SAME_AXIS, lw=2.2, label="Same-axis comparator"),
            Line2D([0], [0], color=COLOR_RELATED_AXIS, lw=2.0, label="Related-axis comparators"),
        ]
    )

    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.00),
        borderaxespad=0.0,
        fontsize=8.7,
        handlelength=2.4,
    )

    return save_figure(fig, output_base)

def save_alpha_sensitivity_figure(
    alpha_summary: pd.DataFrame,
    output_base: Path,
) -> Dict[str, str]:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    focal_specs = {
        "I15541": {"color": COLOR_I15541, "label": "I15541", "offset": -0.010},
        "I15542": {"color": COLOR_I15542, "label": "I15542", "offset": 0.010},
    }

    for individual, spec in focal_specs.items():
        sub = alpha_summary.loc[alpha_summary["individual"] == individual].copy()
        if len(sub) == 0:
            raise ValueError(f"Alpha summary missing rows for {individual}")

        sub = sub.sort_values("rho", ascending=True)
        x = sub["rho"].to_numpy(dtype=float) + spec["offset"]
        y = sub["median_alpha"].to_numpy(dtype=float)
        y_low = sub["ci_lower_95"].to_numpy(dtype=float)
        y_high = sub["ci_upper_95"].to_numpy(dtype=float)

        yerr = np.vstack([y - y_low, y_high - y])

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o-",
            color=spec["color"],
            linewidth=1.8,
            markersize=5.5,
            capsize=0,
            label=spec["label"],
        )

    ax.set_xlim(-0.80, 0.05)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([-0.75, -0.50, -0.25, 0.0])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel(r"Assumed correlation $\rho$ between local and Anatolian components", fontweight="semibold")
    ax.set_ylabel(r"Median $\alpha$ (95% propagation interval)", fontweight="semibold")
    ax.set_title("Alpha sensitivity to Local-Anatolian correlation", fontweight="semibold", pad=10)
    ax.legend(frameon=False, loc="upper right")

    return save_figure(fig, output_base)


def shorten_test_label(test_name: str) -> str:
    mapping = {
        "Leave-one-out outlier diagnostic": "Leave-one-out outlier diagnostic",
        "Sirmium geographic gradient": "Sirmium geographic gradient",
        "E-V13 replication contrast": "E-V13 replication contrast",
        "Autosomal homogenization contrast": "Autosomal homogenization contrast",
    }
    return mapping.get(test_name, test_name)


def get_xlabel_for_test(statistic_name: str) -> str:
    mapping = {
        "S_LOO": r"Shift statistic ($S_{LOO}$)",
        "r": r"Correlation ($r$)",
        "D_EV13": r"Difference ($D_{EV13}$)",
        "Sep": "Separation statistic",
    }
    return mapping.get(str(statistic_name), str(statistic_name))


def get_xlim_for_test(test_number: int) -> Tuple[float, float]:
    if test_number == 1:
        return (-0.0025, 0.0005)
    if test_number == 2:
        return (0.4, 1.05)
    if test_number == 3:
        return (-0.001, 0.011)
    if test_number == 4:
        return (-0.0030, 0.0010)
    raise ValueError(f"Unsupported test_number: {test_number}")


def format_test_row_label(test_number: int, test_name: str) -> str:
    letters = {1: "A", 2: "B", 3: "C", 4: "D"}
    prefix = letters.get(test_number, str(test_number))
    return f"{prefix}. {shorten_test_label(test_name)}"


def figure_number(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}".replace("-", "\N{MINUS SIGN}")


def save_allele_sharing_contrast_figure(
    summary_df: pd.DataFrame,
    panel_name: str,
    output_base: Path,
) -> Dict[str, str]:
    panel_label_map = {
        "main": "Main panel (I27297)",
        "ev13_sensitivity": "Sensitivity panel (I28388)",
    }
    panel_display = panel_label_map.get(panel_name, panel_name)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(7.5, 6.8),
        sharey=False,
        gridspec_kw={"hspace": 0.62},
    )

    for ax, (_, row) in zip(axes, summary_df.iterrows()):
        test_number = int(row["test_number"])
        test_name = str(row["test_name"])
        statistic_name = str(row["statistic_name"])
        est = float(row["statistic"])
        lower = float(row["ci_lower_95"])
        upper = float(row["ci_upper_95"])
        color = ALLELE_SHARING_COLORS.get(test_number, "#333333")

        xmin, xmax = get_xlim_for_test(test_number)

        ax.axvline(
            x=0.0,
            color="#8A8A8A",
            linestyle=(0, (4, 2)),
            linewidth=1.0,
            alpha=0.80,
            zorder=1,
        )

        ax.errorbar(
            est,
            0.0,
            xerr=[[est - lower], [upper - est]],
            fmt="o",
            color=color,
            markersize=7.0,
            capsize=0,
            linewidth=2.0,
            zorder=5,
        )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.55, 0.55)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", left=False)

        ax.set_title(
            format_test_row_label(test_number, test_name),
            loc="left",
            fontsize=10.4,
            fontweight="semibold",
            pad=4,
        )
        ax.set_xlabel(get_xlabel_for_test(statistic_name), fontsize=9.4, fontweight="normal")

        decimals = 3 if test_number == 2 else 4
        text_str = (
            f"{figure_number(est, decimals)} "
            f"[{figure_number(lower, decimals)}, {figure_number(upper, decimals)}]"
        )

        ax.text(
            0.985,
            0.85,
            text_str,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=9.5,
            color=color,
            fontweight="normal",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
                pad=2,
            ),
            clip_on=False,
            zorder=20,
        )

    fig.suptitle(
        f"Independent allele-sharing contrasts: {panel_display}",
        fontsize=12.8,
        fontweight="semibold",
        y=0.992,
    )

    return save_figure(fig, output_base)


def write_manifest(manifest_path: Path, inputs: Dict[str, Path], outputs: Dict[str, Dict[str, str]]) -> None:
    base = inputs["base"]
    manifest = {
        "script": "05_make_figures.py",
        "timestamp_utc": utc_now(),
        "inputs": {key: relpath(base, value) for key, value in inputs.items() if key != "base"},
        "outputs": {
            figure: {kind: relpath(base, value) for kind, value in paths.items()}
            for figure, paths in outputs.items()
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> int:
    args = parse_args()
    paths = resolve_paths(args)
    paths["figdir"].mkdir(parents=True, exist_ok=True)

    configure_style()

    print("=" * 80)
    print("Manuscript figure generator")
    print(f"Base directory: {paths['base']}")
    print(f"Main allele-sharing summary: {paths['main_summary']}")
    print(f"Sensitivity allele-sharing summary: {paths['sensitivity_summary']}")
    print(f"Alpha summary: {paths['alpha_summary']}")
    print(f"Alpha draws: {paths['alpha_draws']}")
    print(f"Comparator CSV: {paths['comparators']}")
    print(f"Figure directory: {paths['figdir']}")
    print("=" * 80)

    try:
        main_summary = load_allele_sharing_summary(paths["main_summary"], "Main allele-sharing summary")
        sensitivity_summary = load_allele_sharing_summary(
            paths["sensitivity_summary"],
            "Sensitivity allele-sharing summary",
        )
        alpha_summary, alpha_draws, comparators = load_alpha_inputs(paths)

        outputs: Dict[str, Dict[str, str]] = {}

        outputs["figure1_alpha_density"] = save_alpha_density_figure(
            alpha_summary=alpha_summary,
            alpha_draws=alpha_draws,
            comparators=comparators,
            output_base=paths["figdir"] / "figure1_alpha_density",
        )

        outputs["figureS1_alpha_rho_sensitivity"] = save_alpha_sensitivity_figure(
            alpha_summary=alpha_summary,
            output_base=paths["figdir"] / "figureS1_alpha_rho_sensitivity",
        )

        outputs["figureS2_allele_sharing_contrasts_main"] = save_allele_sharing_contrast_figure(
            summary_df=main_summary,
            panel_name="main",
            output_base=paths["figdir"] / "figureS2_allele_sharing_contrasts_main",
        )

        outputs["figureS3_allele_sharing_contrasts_ev13_sensitivity"] = save_allele_sharing_contrast_figure(
            summary_df=sensitivity_summary,
            panel_name="ev13_sensitivity",
            output_base=paths["figdir"] / "figureS3_allele_sharing_contrasts_ev13_sensitivity",
        )

        manifest_path = paths["figdir"] / "figure_manifest.json"
        write_manifest(manifest_path, paths, outputs)
        outputs["figure_manifest"] = {"json": str(manifest_path)}

    except Exception as exc:
        print(f"[ERROR] Figure generation failed: {exc}")
        return 1

    print("\n" + "=" * 80)
    print("[OK] Figure generation completed successfully.")
    for figure_name, figure_paths in outputs.items():
        for fmt_key, fmt_path in figure_paths.items():
            print(f"{figure_name} ({fmt_key}): {fmt_path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
