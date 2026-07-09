#!/usr/bin/env python3
"""
Run a complete-case autosomal distance-space concordance audit.

This script reuses the strict complete-case pseudo-haploid 1240k matrices
constructed by 02_build_matrix.py and evaluates whether the full pairwise
mismatch-distance geometry preserves the same directional structure tested by
03_allele_sharing_panel.py.

The analysis is a descriptive robustness layer. It does not estimate ancestry
proportions, infer paternal-lineage origin, prove direct population continuity,
or identify a unique source population.

Usage
-----
Run both panels and render Figure 2:
    python 06_complete_case_distance_audit.py --base .

Run a reduced bootstrap validation:
    python 06_complete_case_distance_audit.py --base . --n-bootstrap 200

Run one panel only:
    python 06_complete_case_distance_audit.py --base . --panels main

Inputs
------
By default, this script expects the panel-specific matrices produced by
02_build_matrix.py:

    matrix_main/complete_case_9_sample_matrix.tsv
    matrix_ev13_sensitivity/complete_case_9_sample_matrix.tsv

Outputs
-------
By default, outputs are written to:

    results/distance_space_audit/observed_distance_metrics.csv
    results/distance_space_audit/distance_to_r1a_profile.csv
    results/distance_space_audit/bootstrap_directional_metrics.csv
    results/distance_space_audit/bootstrap_directional_summary.csv
    results/distance_space_audit/bootstrap_r1a_profile_summary.csv
    results/distance_space_audit/r1a_leaveone_reference_audit.csv
    results/distance_space_audit/i2a_partition_enumeration.csv
    results/distance_space_audit/projection_diagnostics.csv
    results/distance_space_audit/distance_audit_summary.csv
    results/distance_space_audit/distance_audit_run_metadata.json
    results/distance_space_audit/distance_audit_text_summary.txt

    figures/figure2_complete_case_distance_audit.png
    figures/figure2_complete_case_distance_audit.pdf
    figures/figure2_complete_case_distance_audit_manifest.json

Dependencies
------------
    numpy
    pandas
    scipy
    matplotlib
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


PANELS: Dict[str, Dict[str, object]] = {
    "main": {
        "display": "Main panel (I27297)",
        "matrix_dir": "matrix_main",
        "i2a_lowland": ["I1116", "I32299"],
        "i2a_highland": ["I15541", "I15542"],
        "ev13_highland": ["I15537"],
        "ev13_lowland": ["I27297"],
        "r1a_references": ["I26766", "I26748", "I35012"],
        "site_coords": {
            "I1116": (44.8889, 19.7494),
            "I32299": (44.7320, 21.2340),
            "I15541": (43.6450, 22.2650),
            "I15542": (43.6450, 22.2650),
            "I15537": (43.6450, 22.2650),
            "I27297": (44.7546, 20.2621),
        },
    },
    "ev13_sensitivity": {
        "display": "Nustar sensitivity (I28388)",
        "matrix_dir": "matrix_ev13_sensitivity",
        "i2a_lowland": ["I1116", "I32299"],
        "i2a_highland": ["I15541", "I15542"],
        "ev13_highland": ["I15537"],
        "ev13_lowland": ["I28388"],
        "r1a_references": ["I26766", "I26748", "I35012"],
        "site_coords": {
            "I1116": (44.8889, 19.7494),
            "I32299": (44.7320, 21.2340),
            "I15541": (43.6450, 22.2650),
            "I15542": (43.6450, 22.2650),
            "I15537": (43.6450, 22.2650),
            "I28388": (45.338156, 18.845028),
        },
    },
}

SIRMIUM_COORDS: Tuple[float, float] = (44.9764, 19.6122)

WEIGHTS: Dict[str, float] = {
    "loo": 0.9,
    "sirmium_gradient": 0.7,
    "ev13_replication": 0.6,
    "homogenization": 0.3,
}

VALID_ALLELES = {"A", "C", "G", "T"}
DEFAULT_BASE = Path.cwd()
DEFAULT_OUTDIR = Path("results") / "distance_space_audit"
DEFAULT_FIGDIR = Path("figures")
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_SEED = 42
DEFAULT_PROGRESS_INTERVAL = 100

COL_MAIN = "#0072B2"
COL_SENS = "#009E73"
COL_LOW_I2A = "#0072B2"
COL_HIGH_I2A = "#D55E00"
COL_HIGH_I2A_2 = "#009E73"
COL_HIGH_EV13 = "#CC79A7"
COL_LOW_EV13 = "#56B4E9"
COL_R1A = "#8A8A8A"
COL_DARK = "#222222"
COL_GRID = "#E3E3E3"
COL_GREY = "#8F8F8F"
COL_HILITE = "#D55E00"


@dataclass(frozen=True)
class OutputPaths:
    base: Path
    outdir: Path
    figdir: Path
    observed_metrics: Path
    r1a_profile: Path
    bootstrap_metrics: Path
    bootstrap_summary: Path
    bootstrap_r1a_profile_summary: Path
    r1a_leaveone: Path
    i2a_partitions: Path
    projection_diagnostics: Path
    audit_summary: Path
    metadata_json: Path
    text_summary: Path
    figure_png: Path
    figure_pdf: Path
    figure_manifest: Path


@dataclass(frozen=True)
class PanelData:
    panel: str
    panel_display: str
    matrix_path: Path
    df: pd.DataFrame
    samples: List[str]
    allele_array: np.ndarray
    n_snps: int
    groups: Dict[str, List[str]]
    site_coords: Dict[str, Tuple[float, float]]
    sample_to_idx: Dict[str, int]


@dataclass(frozen=True)
class DirectionalMetrics:
    panel: str
    r1a_condition: str
    r1a_references_used: str
    loo_statistic: float
    loo_direction_code: int
    loo_direction_ok: bool
    sirmium_gradient_r: float
    sirmium_direction_code: int
    sirmium_direction_ok: bool
    ev13_difference: float
    ev13_direction_code: int
    ev13_direction_ok: bool
    homogenization_separation: float
    homogenization_direction_code: int
    homogenization_direction_ok: bool
    i2a_highland_minus_lowland_distance_to_r1a: float
    i2a_distance_excess_direction_code: int
    i2a_distance_excess_direction_ok: bool
    weighted_directional_score: float
    n_pre_specified_directions_ok: int
    all_four_pre_specified_directions_ok: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def relpath(base: Path, path: Path | str) -> str:
    try:
        return Path(path).resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return str(path)


def parse_panel_list(raw: str) -> List[str]:
    panels = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not panels:
        raise argparse.ArgumentTypeError("Expected at least one panel name.")
    unknown = sorted(set(panels) - set(PANELS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown panel(s): {', '.join(unknown)}. Valid panels: {', '.join(sorted(PANELS))}"
        )
    return list(dict.fromkeys(panels))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a complete-case autosomal distance-space concordance audit "
            "for the main and/or Nustar sensitivity panels."
        )
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
        help="Project base directory. Default: current working directory.",
    )
    parser.add_argument(
        "--panels",
        type=parse_panel_list,
        default=parse_panel_list("main,ev13_sensitivity"),
        help="Comma-separated panel names to audit. Default: main,ev13_sensitivity.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for numeric audit outputs. Default: <base>/results/distance_space_audit.",
    )
    parser.add_argument(
        "--figdir",
        type=Path,
        default=None,
        help="Directory for figure outputs. Default: <base>/figures.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOTSTRAP,
        help=f"Number of SNP bootstrap replicates per panel. Default: {DEFAULT_N_BOOTSTRAP}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed. Default: {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=DEFAULT_PROGRESS_INTERVAL,
        help=f"Print progress every N bootstrap replicates. Default: {DEFAULT_PROGRESS_INTERVAL}.",
    )
    parser.add_argument(
        "--skip-figure",
        action="store_true",
        help="Write numeric outputs only; do not render Figure 2.",
    )
    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace) -> OutputPaths:
    base = args.base.resolve()
    outdir = args.outdir.resolve() if args.outdir else base / DEFAULT_OUTDIR
    figdir = args.figdir.resolve() if args.figdir else base / DEFAULT_FIGDIR
    return OutputPaths(
        base=base,
        outdir=outdir,
        figdir=figdir,
        observed_metrics=outdir / "observed_distance_metrics.csv",
        r1a_profile=outdir / "distance_to_r1a_profile.csv",
        bootstrap_metrics=outdir / "bootstrap_directional_metrics.csv",
        bootstrap_summary=outdir / "bootstrap_directional_summary.csv",
        bootstrap_r1a_profile_summary=outdir / "bootstrap_r1a_profile_summary.csv",
        r1a_leaveone=outdir / "r1a_leaveone_reference_audit.csv",
        i2a_partitions=outdir / "i2a_partition_enumeration.csv",
        projection_diagnostics=outdir / "projection_diagnostics.csv",
        audit_summary=outdir / "distance_audit_summary.csv",
        metadata_json=outdir / "distance_audit_run_metadata.json",
        text_summary=outdir / "distance_audit_text_summary.txt",
        figure_png=figdir / "figure2_complete_case_distance_audit.png",
        figure_pdf=figdir / "figure2_complete_case_distance_audit.pdf",
        figure_manifest=figdir / "figure2_complete_case_distance_audit_manifest.json",
    )


def configure_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["axes.labelsize"] = 10.5
    plt.rcParams["axes.titlesize"] = 11.5
    plt.rcParams["xtick.labelsize"] = 8.8
    plt.rcParams["ytick.labelsize"] = 8.8
    plt.rcParams["legend.fontsize"] = 8.7
    plt.rcParams["figure.titlesize"] = 15
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"


def deterministic_seed(base_seed: int, label: str) -> int:
    key = f"{base_seed}|{label}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    offset = int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32 - 1)
    return (base_seed + offset) % (2**32 - 1)


def build_sample_lists(panel_cfg: Dict[str, object]) -> Dict[str, List[str]]:
    i2a_lowland = list(panel_cfg["i2a_lowland"])
    i2a_highland = list(panel_cfg["i2a_highland"])
    ev13_highland = list(panel_cfg["ev13_highland"])
    ev13_lowland = list(panel_cfg["ev13_lowland"])
    r1a_references = list(panel_cfg["r1a_references"])

    all_i2a = i2a_lowland + i2a_highland
    all_non_r1a = all_i2a + ev13_highland + ev13_lowland
    all_targets = all_non_r1a + r1a_references

    return {
        "i2a_lowland": i2a_lowland,
        "i2a_highland": i2a_highland,
        "ev13_highland": ev13_highland,
        "ev13_lowland": ev13_lowland,
        "r1a_references": r1a_references,
        "all_i2a": all_i2a,
        "all_non_r1a": all_non_r1a,
        "all_targets": all_targets,
    }


def validate_columns(df: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required column(s): {', '.join(missing)}")


def validate_matrix(df: pd.DataFrame, target_samples: List[str], matrix_path: Path) -> None:
    if len(df) == 0:
        raise ValueError(f"Matrix has zero rows: {matrix_path}")

    validate_columns(df, ["chrom", "pos"], f"Matrix {matrix_path}")

    missing_samples = sorted(set(target_samples) - set(df.columns))
    if missing_samples:
        raise ValueError(
            f"Matrix {matrix_path} is missing required sample column(s): "
            + ", ".join(missing_samples)
        )

    sample_df = df[target_samples]
    if sample_df.isna().any().any():
        raise ValueError(f"Matrix {matrix_path} contains missing values in target sample columns.")

    invalid_by_sample: Dict[str, List[str]] = {}
    for sample in target_samples:
        observed = set(sample_df[sample].astype(str).unique())
        invalid = sorted(observed - VALID_ALLELES)
        if invalid:
            invalid_by_sample[sample] = invalid

    if invalid_by_sample:
        messages = [
            f"{sample}: {', '.join(values[:10])}"
            for sample, values in invalid_by_sample.items()
        ]
        raise ValueError(
            f"Matrix {matrix_path} contains invalid allele calls: " + "; ".join(messages)
        )


def load_panel_data(base: Path, panel_name: str) -> PanelData:
    cfg = PANELS[panel_name]
    groups = build_sample_lists(cfg)
    matrix_path = base / str(cfg["matrix_dir"]) / "complete_case_9_sample_matrix.tsv"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found for panel {panel_name}: {matrix_path}")

    df = pd.read_csv(matrix_path, sep="\t", dtype=str)
    samples = groups["all_targets"]
    validate_matrix(df, samples, matrix_path)
    df = df[["chrom", "pos"] + samples].copy()

    return PanelData(
        panel=panel_name,
        panel_display=str(cfg["display"]),
        matrix_path=matrix_path,
        df=df,
        samples=samples,
        allele_array=df[samples].to_numpy(),
        n_snps=len(df),
        groups=groups,
        site_coords=dict(cfg["site_coords"]),
        sample_to_idx={sample: idx for idx, sample in enumerate(samples)},
    )


def compute_distance_matrix(arr: np.ndarray) -> np.ndarray:
    """
    Compute pairwise pseudo-haploid mismatch distance across rows.
    Input array shape: (n_snps, n_samples).
    """
    if arr.shape[0] == 0:
        raise ValueError("Cannot compute distances on a zero-row matrix.")
    diff = arr[:, :, None] != arr[:, None, :]
    return diff.mean(axis=0)


def mean_dist_to_ref(sample_idx: int, ref_indices: List[int], dist_mat: np.ndarray) -> float:
    if not ref_indices:
        return float("nan")
    return float(np.mean([dist_mat[sample_idx, ref_idx] for ref_idx in ref_indices]))


def haversine(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return radius_km * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def idx_list(panel_data: PanelData, sample_names: Iterable[str]) -> List[int]:
    return [panel_data.sample_to_idx[sample] for sample in sample_names]


def direction_code(value: float, predicted: str) -> int:
    if not np.isfinite(value):
        return 0
    if predicted == "positive":
        return 1 if value > 0 else (-1 if value < 0 else 0)
    if predicted == "negative":
        return 1 if value < 0 else (-1 if value > 0 else 0)
    if predicted == "non-positive":
        return 1 if value <= 0 else -1
    raise ValueError(f"Unsupported predicted direction: {predicted}")


def compute_loo_shift(
    dist_mat: np.ndarray,
    sample_list: List[str],
    i2a_idx: List[int],
    r1a_idx: List[int],
    low_idx: List[int],
    high_idx: List[int],
) -> float:
    between = [dist_mat[i, r] for i in i2a_idx for r in r1a_idx]
    within_r1a = [
        dist_mat[r1, r2]
        for idx_r, r1 in enumerate(r1a_idx)
        for r2 in r1a_idx[idx_r + 1 :]
    ]
    if not between or not within_r1a:
        return float("nan")

    delta_base = float(np.mean(between) - np.mean(within_r1a))
    shifts: Dict[str, float] = {}

    for idx in i2a_idx:
        i2a_loo = [j for j in i2a_idx if j != idx]
        between_loo = [dist_mat[j, r] for j in i2a_loo for r in r1a_idx]
        delta_loo = float(np.mean(between_loo) - np.mean(within_r1a))
        shifts[sample_list[idx]] = delta_loo - delta_base

    high_shift = float(np.mean([shifts[sample_list[idx]] for idx in high_idx]))
    low_shift = float(np.mean([shifts[sample_list[idx]] for idx in low_idx]))
    return high_shift - low_shift


def compute_sirmium_gradient(
    dist_mat: np.ndarray,
    panel_data: PanelData,
    i2a_idx: List[int],
    r1a_idx: List[int],
) -> Tuple[float, List[str], List[float], List[float]]:
    auto_dists: List[float] = []
    geo_dists: List[float] = []
    used_samples: List[str] = []

    for idx in i2a_idx:
        sample = panel_data.samples[idx]
        if sample not in panel_data.site_coords:
            continue
        auto_dists.append(mean_dist_to_ref(idx, r1a_idx, dist_mat))
        geo_dists.append(haversine(panel_data.site_coords[sample], SIRMIUM_COORDS))
        used_samples.append(sample)

    if len(auto_dists) < 3:
        return float("nan"), used_samples, geo_dists, auto_dists

    r_value, _ = pearsonr(geo_dists, auto_dists)
    return float(r_value), used_samples, geo_dists, auto_dists


def compute_ev13_difference(
    dist_mat: np.ndarray,
    ev13_high_idx: List[int],
    ev13_low_idx: List[int],
    r1a_idx: List[int],
) -> float:
    if len(ev13_high_idx) != 1 or len(ev13_low_idx) != 1:
        return float("nan")
    high = mean_dist_to_ref(ev13_high_idx[0], r1a_idx, dist_mat)
    low = mean_dist_to_ref(ev13_low_idx[0], r1a_idx, dist_mat)
    return float(high - low)


def compute_homogenization(
    dist_mat: np.ndarray,
    i2a_idx: List[int],
    r1a_idx: List[int],
) -> float:
    within_i2a = [
        dist_mat[i, j]
        for idx_i, i in enumerate(i2a_idx)
        for j in i2a_idx[idx_i + 1 :]
    ]
    within_r1a = [
        dist_mat[r1, r2]
        for idx_r, r1 in enumerate(r1a_idx)
        for r2 in r1a_idx[idx_r + 1 :]
    ]
    between = [dist_mat[i, r] for i in i2a_idx for r in r1a_idx]
    if not within_i2a or not within_r1a or not between:
        return float("nan")
    return float(np.mean(between) - np.mean(within_i2a + within_r1a))


def compute_i2a_distance_excess(
    dist_mat: np.ndarray,
    i2a_low_idx: List[int],
    i2a_high_idx: List[int],
    r1a_idx: List[int],
) -> float:
    high_vals = [mean_dist_to_ref(idx, r1a_idx, dist_mat) for idx in i2a_high_idx]
    low_vals = [mean_dist_to_ref(idx, r1a_idx, dist_mat) for idx in i2a_low_idx]
    return float(np.mean(high_vals) - np.mean(low_vals))


def compute_directional_metrics(
    panel_data: PanelData,
    dist_mat: np.ndarray,
    r1a_samples: Optional[List[str]] = None,
    r1a_condition: str = "full",
) -> DirectionalMetrics:
    groups = panel_data.groups
    r1a_samples_used = r1a_samples if r1a_samples is not None else groups["r1a_references"]

    i2a_low_idx = idx_list(panel_data, groups["i2a_lowland"])
    i2a_high_idx = idx_list(panel_data, groups["i2a_highland"])
    i2a_idx = i2a_low_idx + i2a_high_idx
    ev13_high_idx = idx_list(panel_data, groups["ev13_highland"])
    ev13_low_idx = idx_list(panel_data, groups["ev13_lowland"])
    r1a_idx = idx_list(panel_data, r1a_samples_used)

    loo = compute_loo_shift(
        dist_mat,
        panel_data.samples,
        i2a_idx=i2a_idx,
        r1a_idx=r1a_idx,
        low_idx=i2a_low_idx,
        high_idx=i2a_high_idx,
    )
    sirmium_r, _, _, _ = compute_sirmium_gradient(dist_mat, panel_data, i2a_idx, r1a_idx)
    ev13_diff = compute_ev13_difference(dist_mat, ev13_high_idx, ev13_low_idx, r1a_idx)
    homog = compute_homogenization(dist_mat, i2a_idx, r1a_idx)
    i2a_excess = compute_i2a_distance_excess(dist_mat, i2a_low_idx, i2a_high_idx, r1a_idx)

    loo_code = direction_code(loo, "negative")
    sirmium_code = direction_code(sirmium_r, "positive")
    ev13_code = direction_code(ev13_diff, "positive")
    homog_code = direction_code(homog, "non-positive")
    i2a_excess_code = direction_code(i2a_excess, "positive")

    weighted_score = (
        loo_code * WEIGHTS["loo"]
        + sirmium_code * WEIGHTS["sirmium_gradient"]
        + ev13_code * WEIGHTS["ev13_replication"]
        + homog_code * WEIGHTS["homogenization"]
    ) / sum(WEIGHTS.values())

    ok_flags = [loo_code == 1, sirmium_code == 1, ev13_code == 1, homog_code == 1]

    return DirectionalMetrics(
        panel=panel_data.panel,
        r1a_condition=r1a_condition,
        r1a_references_used=";".join(r1a_samples_used),
        loo_statistic=float(loo),
        loo_direction_code=int(loo_code),
        loo_direction_ok=bool(loo_code == 1),
        sirmium_gradient_r=float(sirmium_r),
        sirmium_direction_code=int(sirmium_code),
        sirmium_direction_ok=bool(sirmium_code == 1),
        ev13_difference=float(ev13_diff),
        ev13_direction_code=int(ev13_code),
        ev13_direction_ok=bool(ev13_code == 1),
        homogenization_separation=float(homog),
        homogenization_direction_code=int(homog_code),
        homogenization_direction_ok=bool(homog_code == 1),
        i2a_highland_minus_lowland_distance_to_r1a=float(i2a_excess),
        i2a_distance_excess_direction_code=int(i2a_excess_code),
        i2a_distance_excess_direction_ok=bool(i2a_excess_code == 1),
        weighted_directional_score=float(weighted_score),
        n_pre_specified_directions_ok=int(sum(ok_flags)),
        all_four_pre_specified_directions_ok=bool(all(ok_flags)),
    )


def make_r1a_profile(panel_data: PanelData, dist_mat: np.ndarray) -> pd.DataFrame:
    groups = panel_data.groups
    r1a_idx = idx_list(panel_data, groups["r1a_references"])
    rows: List[Dict[str, object]] = []

    group_by_sample: Dict[str, str] = {}
    for s in groups["i2a_lowland"]:
        group_by_sample[s] = "Lowland I2a"
    for s in groups["i2a_highland"]:
        group_by_sample[s] = "Highland I2a"
    for s in groups["ev13_highland"]:
        group_by_sample[s] = "Highland E-V13"
    for s in groups["ev13_lowland"]:
        group_by_sample[s] = "Lowland E-V13"

    for sample in groups["all_non_r1a"]:
        idx = panel_data.sample_to_idx[sample]
        rows.append(
            {
                "panel": panel_data.panel,
                "panel_display": panel_data.panel_display,
                "sample": sample,
                "sample_group": group_by_sample.get(sample, "Other"),
                "mean_distance_to_r1a": mean_dist_to_ref(idx, r1a_idx, dist_mat),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_panel(
    panel_data: PanelData,
    n_bootstrap: int,
    seed: int,
    progress_interval: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(deterministic_seed(seed, panel_data.panel))
    metric_rows: List[Dict[str, object]] = []
    profile_rows: List[Dict[str, object]] = []

    print(f"[BOOT] {panel_data.panel}: {n_bootstrap} SNP-bootstrap replicates")
    for b in range(1, n_bootstrap + 1):
        boot_idx = rng.choice(panel_data.n_snps, size=panel_data.n_snps, replace=True)
        boot_arr = panel_data.allele_array[boot_idx]
        boot_dist = compute_distance_matrix(boot_arr)
        metrics = compute_directional_metrics(panel_data, boot_dist)
        row = asdict(metrics)
        row["bootstrap_id"] = b
        metric_rows.append(row)

        prof = make_r1a_profile(panel_data, boot_dist)
        prof["bootstrap_id"] = b
        profile_rows.extend(prof.to_dict(orient="records"))

        if progress_interval > 0 and (b % progress_interval == 0 or b == n_bootstrap):
            print(f"  -> {panel_data.panel}: replicate {b}/{n_bootstrap} complete")

    return pd.DataFrame(metric_rows), pd.DataFrame(profile_rows)


def summarize_bootstrap_metrics(boot_df: pd.DataFrame) -> pd.DataFrame:
    metric_specs = [
        ("loo_statistic", "Leave-one-out outlier diagnostic", "negative", "loo_direction_ok"),
        ("sirmium_gradient_r", "Sirmium geographic gradient", "positive", "sirmium_direction_ok"),
        ("ev13_difference", "E-V13 replication contrast", "positive", "ev13_direction_ok"),
        ("homogenization_separation", "Autosomal homogenization contrast", "non-positive", "homogenization_direction_ok"),
        ("i2a_highland_minus_lowland_distance_to_r1a", "I2a highland-minus-lowland R1a distance excess", "positive", "i2a_distance_excess_direction_ok"),
        ("weighted_directional_score", "Weighted directional score", "higher positive", "all_four_pre_specified_directions_ok"),
    ]
    rows: List[Dict[str, object]] = []
    for panel, sub in boot_df.groupby("panel", sort=False):
        for metric_col, metric_label, predicted, ok_col in metric_specs:
            vals = pd.to_numeric(sub[metric_col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                median = low = high = mean = np.nan
            else:
                median = float(np.median(vals))
                mean = float(np.mean(vals))
                low = float(np.quantile(vals, 0.025))
                high = float(np.quantile(vals, 0.975))
            ok_fraction = float(pd.to_numeric(sub[ok_col], errors="coerce").mean()) if ok_col in sub else np.nan
            rows.append(
                {
                    "panel": panel,
                    "metric": metric_col,
                    "metric_label": metric_label,
                    "predicted_direction": predicted,
                    "bootstrap_mean": mean,
                    "bootstrap_median": median,
                    "ci_lower_95": low,
                    "ci_upper_95": high,
                    "direction_preservation_fraction": ok_fraction,
                    "n_bootstrap": int(len(sub)),
                }
            )
        rows.append(
            {
                "panel": panel,
                "metric": "all_four_pre_specified_directions_ok",
                "metric_label": "All four pre-specified directions jointly preserved",
                "predicted_direction": "true",
                "bootstrap_mean": float(pd.to_numeric(sub["all_four_pre_specified_directions_ok"], errors="coerce").mean()),
                "bootstrap_median": np.nan,
                "ci_lower_95": np.nan,
                "ci_upper_95": np.nan,
                "direction_preservation_fraction": float(pd.to_numeric(sub["all_four_pre_specified_directions_ok"], errors="coerce").mean()),
                "n_bootstrap": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def summarize_bootstrap_r1a_profile(profile_boot: pd.DataFrame, observed_profile: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (panel, sample), sub in profile_boot.groupby(["panel", "sample"], sort=False):
        vals = pd.to_numeric(sub["mean_distance_to_r1a"], errors="coerce").dropna().to_numpy(dtype=float)
        obs_hit = observed_profile[(observed_profile["panel"].eq(panel)) & (observed_profile["sample"].eq(sample))]
        observed = float(obs_hit["mean_distance_to_r1a"].iloc[0]) if len(obs_hit) else np.nan
        group = str(obs_hit["sample_group"].iloc[0]) if len(obs_hit) else ""
        panel_display = str(obs_hit["panel_display"].iloc[0]) if len(obs_hit) else panel
        rows.append(
            {
                "panel": panel,
                "panel_display": panel_display,
                "sample": sample,
                "sample_group": group,
                "observed_mean_distance_to_r1a": observed,
                "bootstrap_mean": float(np.mean(vals)) if len(vals) else np.nan,
                "bootstrap_median": float(np.median(vals)) if len(vals) else np.nan,
                "ci_lower_95": float(np.quantile(vals, 0.025)) if len(vals) else np.nan,
                "ci_upper_95": float(np.quantile(vals, 0.975)) if len(vals) else np.nan,
                "n_bootstrap": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def r1a_leaveone_audit(panel_data: PanelData, dist_mat: np.ndarray) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    r1a_samples = panel_data.groups["r1a_references"]

    full = compute_directional_metrics(panel_data, dist_mat, r1a_samples=r1a_samples, r1a_condition="full")
    rows.append(asdict(full))

    for dropped in r1a_samples:
        keep = [sample for sample in r1a_samples if sample != dropped]
        metrics = compute_directional_metrics(
            panel_data,
            dist_mat,
            r1a_samples=keep,
            r1a_condition=f"drop_{dropped}",
        )
        row = asdict(metrics)
        row["dropped_r1a_reference"] = dropped
        rows.append(row)

    out = pd.DataFrame(rows)
    if "dropped_r1a_reference" not in out.columns:
        out["dropped_r1a_reference"] = ""
    out["panel_display"] = panel_data.panel_display
    return out


def enumerate_i2a_partitions(panel_data: PanelData, dist_mat: np.ndarray) -> pd.DataFrame:
    groups = panel_data.groups
    i2a_samples = groups["all_i2a"]
    observed_high = set(groups["i2a_highland"])
    r1a_idx = idx_list(panel_data, groups["r1a_references"])

    rows: List[Dict[str, object]] = []
    for combo in combinations(i2a_samples, 2):
        assigned_high = list(combo)
        assigned_low = [sample for sample in i2a_samples if sample not in assigned_high]
        high_idx = idx_list(panel_data, assigned_high)
        low_idx = idx_list(panel_data, assigned_low)
        high_vals = [mean_dist_to_ref(idx, r1a_idx, dist_mat) for idx in high_idx]
        low_vals = [mean_dist_to_ref(idx, r1a_idx, dist_mat) for idx in low_idx]
        statistic = float(np.mean(high_vals) - np.mean(low_vals))
        is_obs = set(assigned_high) == observed_high
        rows.append(
            {
                "panel": panel_data.panel,
                "panel_display": panel_data.panel_display,
                "assigned_highland_i2a": ";".join(assigned_high),
                "assigned_lowland_i2a": ";".join(assigned_low),
                "is_observed_geographic_partition": bool(is_obs),
                "mean_assigned_highland_distance_to_r1a": float(np.mean(high_vals)),
                "mean_assigned_lowland_distance_to_r1a": float(np.mean(low_vals)),
                "highland_minus_lowland_distance_to_r1a": statistic,
            }
        )

    out = pd.DataFrame(rows).sort_values("highland_minus_lowland_distance_to_r1a", ascending=False).reset_index(drop=True)
    out["rank_descending"] = np.arange(1, len(out) + 1)
    out["n_partitions"] = len(out)
    obs_rank = int(out.loc[out["is_observed_geographic_partition"], "rank_descending"].iloc[0])
    out["observed_partition_rank_descending"] = obs_rank
    return out


def classical_mds(dist_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = dist_mat.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H.dot(dist_mat ** 2).dot(H)
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    coords = evecs[:, :2] * np.sqrt(np.maximum(evals[:2], 0.0))
    return coords, evals


def projection_diagnostics(panel_data: PanelData, dist_mat: np.ndarray) -> Dict[str, object]:
    coords, evals = classical_mds(dist_mat)
    positive = evals[evals > 0]
    positive_sum = float(np.sum(positive)) if len(positive) else float("nan")
    axis1_positive_share = float(evals[0] / positive_sum) if positive_sum > 0 else float("nan")
    axis2_positive_share = float(evals[1] / positive_sum) if positive_sum > 0 else float("nan")
    axis12_positive_share = float((evals[0] + evals[1]) / positive_sum) if positive_sum > 0 else float("nan")

    tri = np.triu_indices(dist_mat.shape[0], k=1)
    full_d = dist_mat[tri]
    projected = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))[tri]

    pear = float(pearsonr(full_d, projected)[0]) if len(full_d) >= 3 else float("nan")
    spear = float(spearmanr(full_d, projected).correlation) if len(full_d) >= 3 else float("nan")
    denom = np.sum(full_d ** 2)
    stress = float(np.sqrt(np.sum((full_d - projected) ** 2) / denom)) if denom > 0 else float("nan")

    return {
        "panel": panel_data.panel,
        "panel_display": panel_data.panel_display,
        "n_samples": len(panel_data.samples),
        "n_snps": panel_data.n_snps,
        "axis1_positive_eigenvalue_share": axis1_positive_share,
        "axis2_positive_eigenvalue_share": axis2_positive_share,
        "axis1_axis2_positive_eigenvalue_share": axis12_positive_share,
        "shepard_pearson_r": pear,
        "shepard_spearman_r": spear,
        "projection_stress": stress,
        "note": "Projection diagnostics are reported for transparency; directional audit statistics are computed from full pairwise distances, not from the 2D projection.",
    }


def observed_metrics_row(panel_data: PanelData, dist_mat: np.ndarray, base: Path) -> Dict[str, object]:
    metrics = compute_directional_metrics(panel_data, dist_mat)
    row = asdict(metrics)
    row.update(
        {
            "panel_display": panel_data.panel_display,
            "matrix_path": relpath(base, panel_data.matrix_path),
            "matrix_sha256": sha256_file(panel_data.matrix_path),
            "n_samples": len(panel_data.samples),
            "n_snps": panel_data.n_snps,
            "sample_order": ";".join(panel_data.samples),
        }
    )
    return row


def make_audit_summary(
    observed_df: pd.DataFrame,
    boot_summary: pd.DataFrame,
    leaveone_df: pd.DataFrame,
    partition_df: pd.DataFrame,
    projection_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def add(panel: str, metric: str, value: object, note: str = "") -> None:
        rows.append({"panel": panel, "metric": metric, "value": value, "note": note})

    for _, row in observed_df.iterrows():
        panel = str(row["panel"])
        add(panel, "n_snps", int(row["n_snps"]), "strict complete-case SNP rows")
        add(panel, "weighted_directional_score_observed", float(row["weighted_directional_score"]), "weighted directional score following the allele-sharing analysis, recomputed from the full distance matrix")
        add(panel, "n_pre_specified_directions_ok_observed", int(row["n_pre_specified_directions_ok"]), "out of four pre-specified directional contrasts")
        add(panel, "i2a_highland_minus_lowland_distance_to_r1a", float(row["i2a_highland_minus_lowland_distance_to_r1a"]), "direct R1a-frame I2a landscape contrast")

        boot = boot_summary[(boot_summary["panel"].eq(panel)) & (boot_summary["metric"].eq("all_four_pre_specified_directions_ok"))]
        if len(boot):
            add(panel, "bootstrap_all_four_direction_fraction", float(boot["direction_preservation_fraction"].iloc[0]), "fraction of SNP bootstraps preserving all four directions")

        loo = leaveone_df[leaveone_df["panel"].eq(panel)]
        if len(loo):
            add(panel, "r1a_leaveone_min_weighted_score", float(loo["weighted_directional_score"].min()), "minimum score across full and leave-one-R1a frames")
            add(panel, "r1a_leaveone_min_direction_count", int(loo["n_pre_specified_directions_ok"].min()), "minimum number of directions preserved across full and leave-one-R1a frames")

        parts = partition_df[partition_df["panel"].eq(panel)]
        obs_part = parts[parts["is_observed_geographic_partition"]]
        if len(obs_part):
            add(panel, "observed_i2a_partition_rank_descending", int(obs_part["rank_descending"].iloc[0]), "rank of observed highland split among all 2-vs-2 I2a partitions by R1a-distance excess")
            add(panel, "observed_i2a_partition_n_partitions", int(obs_part["n_partitions"].iloc[0]), "number of balanced I2a partitions evaluated")

        proj = projection_df[projection_df["panel"].eq(panel)]
        if len(proj):
            add(panel, "projection_axis1_axis2_positive_eigenvalue_share", float(proj["axis1_axis2_positive_eigenvalue_share"].iloc[0]), "for transparency only; audit statistics use full distances")
            add(panel, "projection_stress", float(proj["projection_stress"].iloc[0]), "normalized residual stress of 2D projection")

    add("all", "scope_statement", "distance-space robustness audit only", "does not estimate ancestry proportions or infer paternal-lineage origin")
    return pd.DataFrame(rows)


def write_text_summary(path: Path, audit_summary: pd.DataFrame, boot_summary: pd.DataFrame, partition_df: pd.DataFrame, leaveone_df: pd.DataFrame) -> None:
    lines: List[str] = [
        "Complete-case autosomal distance-space concordance audit",
        "============================================================",
        "Scope: descriptive robustness audit only; not an ancestry estimator and not a paternal-lineage origin test.",
        "",
        "Observed and bootstrap directional support:",
    ]

    for panel in sorted(set(boot_summary["panel"])):
        all_four = boot_summary[(boot_summary["panel"].eq(panel)) & (boot_summary["metric"].eq("all_four_pre_specified_directions_ok"))]
        if len(all_four):
            frac = float(all_four["direction_preservation_fraction"].iloc[0])
            lines.append(f"  - {panel}: all-four-direction bootstrap preservation fraction = {frac:.3f}")
        loo = leaveone_df[leaveone_df["panel"].eq(panel)]
        if len(loo):
            lines.append(
                f"    leave-one-R1a minimum score = {float(loo['weighted_directional_score'].min()):.3f}; "
                f"minimum direction count = {int(loo['n_pre_specified_directions_ok'].min())}/4"
            )
        obs_part = partition_df[(partition_df["panel"].eq(panel)) & (partition_df["is_observed_geographic_partition"])]
        if len(obs_part):
            lines.append(
                f"    observed I2a geographic split rank = {int(obs_part['rank_descending'].iloc[0])}/"
                f"{int(obs_part['n_partitions'].iloc[0])} by highland-minus-lowland R1a-distance excess"
            )

    lines.extend(
        [
            "",
            "Summary:",
            "A complete-case distance-space audit visualizes and evaluates the pairwise mismatch geometry underlying the allele-sharing contrasts. The audit recomputes R1a-frame distances, SNP-bootstrap direction preservation, leave-one-R1a-reference robustness, and all balanced I2a partition contrasts. It supports the descriptive conclusion that the complete-case autosomal distance matrix is concordant with structured regional baselines under incomplete proxy sampling, while remaining non-identifying with respect to paternal-lineage origin, direct continuity, or homeland.",
            "",
            "Scope:",
            "This analysis does not distinguish locally retained Balkan-like ancestry from an unsampled eastern or northern source population carrying a similar autosomal profile. It tests concordance of the available complete-case distance geometry, not source uniqueness.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    outputs: OutputPaths,
    panels: List[PanelData],
) -> None:
    metadata = {
        "script": "06_complete_case_distance_audit.py",
        "timestamp_utc": utc_now(),
        "purpose": (
            "Complete-case autosomal distance-space concordance audit using the same strict "
            "pseudo-haploid 1240k matrices and directional contrast framework as the allele-sharing analysis."
        ),
        "settings": {
            "base": ".",
            "panels": args.panels,
            "n_bootstrap": int(args.n_bootstrap),
            "seed": int(args.seed),
            "progress_interval": int(args.progress_interval),
        },
        "inputs": {
            pdata.panel: {
                "matrix_path": relpath(outputs.base, pdata.matrix_path),
                "matrix_sha256": sha256_file(pdata.matrix_path),
                "n_snps": int(pdata.n_snps),
                "sample_order": pdata.samples,
                "groups": pdata.groups,
            }
            for pdata in panels
        },
        "outputs": {field: relpath(outputs.base, getattr(outputs, field)) for field in OutputPaths.__dataclass_fields__ if field not in {"base", "outdir", "figdir"}},
        "weights": WEIGHTS,
        "scope_statement": "Distance-space robustness audit only; not an ancestry estimator and not a paternal-lineage origin test.",
        "limitation_statement": (
            "The audit does not exclude an unsampled Balkan-like eastern/northern source and does not prove direct continuity. "
            "It evaluates concordance and robustness of the available complete-case distance geometry."
        ),
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def hide_spines(ax: plt.Axes, names: Sequence[str] = ("top", "right")) -> None:
    """Hide selected spines in a matplotlib-version-tolerant way."""
    for name in names:
        ax.spines[name].set_visible(False)


def group_color(sample: str, group: str) -> str:
    if group == "Lowland I2a":
        return COL_LOW_I2A
    if group == "Highland I2a":
        if sample == "I15542":
            return COL_HIGH_I2A_2
        return COL_HIGH_I2A
    if group == "Highland E-V13":
        return COL_HIGH_EV13
    if group == "Lowland E-V13":
        return COL_LOW_EV13
    return COL_GREY


def panel_color(panel: str) -> str:
    return COL_MAIN if panel == "main" else COL_SENS


def panel_marker(panel: str) -> str:
    return "o" if panel == "main" else "s"


def short_panel_label(panel: str) -> str:
    return "Main" if panel == "main" else "Nustar"


def make_figure(
    outputs: OutputPaths,
    r1a_profile_summary: pd.DataFrame,
    observed_metrics: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    leaveone_df: pd.DataFrame,
    partition_df: pd.DataFrame,
) -> None:
    configure_style()
    outputs.figdir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.1, 10.4), dpi=200)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.10, 1.0], width_ratios=[1.15, 1.0], hspace=0.40, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    subgs_d = gs[1, 1].subgridspec(1, 2, width_ratios=[0.92, 1.08], wspace=0.56)
    ax_d1 = fig.add_subplot(subgs_d[0, 0])
    ax_d2 = fig.add_subplot(subgs_d[0, 1])

    sample_order = ["I1116", "I32299", "I15541", "I15542", "I15537", "I27297", "I28388"]
    x_map = {sample: i for i, sample in enumerate(sample_order)}
    offsets = {"main": -0.11, "ev13_sensitivity": 0.11}

    group_spans = [
        (-0.5, 1.5, COL_LOW_I2A),
        (1.5, 3.5, COL_HIGH_I2A),
        (3.5, 4.5, COL_HIGH_EV13),
        (4.5, 6.5, COL_LOW_EV13),
    ]
    for xmin, xmax, color in group_spans:
        ax_a.axvspan(xmin, xmax, color=color, alpha=0.045, zorder=0)

    for _, row in r1a_profile_summary.iterrows():
        sample = str(row["sample"])
        if sample not in x_map:
            continue
        panel = str(row["panel"])
        group = str(row["sample_group"])
        x = x_map[sample] + offsets.get(panel, 0.0)
        y = float(row["observed_mean_distance_to_r1a"])
        lo = float(row["ci_lower_95"])
        hi = float(row["ci_upper_95"])
        ax_a.errorbar(
            x,
            y,
            yerr=[[y - lo], [hi - y]],
            fmt=panel_marker(panel),
            color=group_color(sample, group),
            markeredgecolor=COL_DARK,
            markeredgewidth=0.55,
            markersize=6.5,
            linewidth=1.3,
            capsize=0,
            alpha=0.95,
            zorder=4,
        )

    for boundary in [1.5, 3.5, 4.5]:
        ax_a.axvline(boundary, color="#DCDCDC", linewidth=0.9, linestyle=(0, (2, 3)), zorder=1)

    ax_a.set_xlim(-0.55, 6.45)
    ax_a.set_ylim(0.248, 0.266)
    ax_a.set_xticks([x_map[s] for s in sample_order])
    ax_a.set_xticklabels(["I1116", "I32299", "I15541", "I15542", "I15537", "I27297", "I28388"], rotation=45, ha="right")
    ax_a.set_ylabel("Mean distance to R1a frame", fontweight="semibold")
    ax_a.set_title("a  Distance to the R1a reference frame", loc="left", fontweight="semibold", pad=14)
    ax_a.grid(axis="y", color=COL_GRID, linewidth=0.8)
    hide_spines(ax_a)
    ax_a.text(0.5, 0.955, "Lowland I2a", transform=ax_a.get_xaxis_transform(), ha="center", va="top", fontsize=8.1, color=COL_LOW_I2A)
    ax_a.text(2.5, 0.955, "Highland I2a", transform=ax_a.get_xaxis_transform(), ha="center", va="top", fontsize=8.1, color=COL_HIGH_I2A)
    ax_a.text(4.0, 0.955, "Highland\nE-V13", transform=ax_a.get_xaxis_transform(), ha="center", va="top", fontsize=8.1, color=COL_HIGH_EV13, linespacing=0.9)
    ax_a.text(5.5, 0.955, "Lowland E-V13\ncomparators", transform=ax_a.get_xaxis_transform(), ha="center", va="top", fontsize=8.1, color=COL_LOW_EV13, linespacing=0.9)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=COL_DARK, label="Main panel", markersize=6.5),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor=COL_DARK, label="Nustar sensitivity", markersize=6.5),
    ]
    leg_a = ax_a.legend(
        handles=legend_handles,
        frameon=True,
        loc="lower right",
        bbox_to_anchor=(0.998, -0.012),
        borderaxespad=0.0,
        labelspacing=0.32,
        handlelength=1.55,
        fontsize=8.2,
    )
    leg_a.get_frame().set_facecolor("white")
    leg_a.get_frame().set_alpha(0.92)
    leg_a.get_frame().set_edgecolor("none")

    for panel in ["main", "ev13_sensitivity"]:
        sub_metric = observed_metrics[observed_metrics["panel"].eq(panel)]
        prof = r1a_profile_summary[(r1a_profile_summary["panel"].eq(panel)) & (r1a_profile_summary["sample_group"].isin(["Lowland I2a", "Highland I2a"]))].copy()
        if prof.empty:
            continue
        x_vals = []
        y_vals = []
        y_los = []
        y_his = []
        labels = []
        for _, row in prof.iterrows():
            sample = str(row["sample"])
            cfg = PANELS[panel]
            geo = haversine(dict(cfg["site_coords"])[sample], SIRMIUM_COORDS)
            x_vals.append(geo)
            y = float(row["observed_mean_distance_to_r1a"])
            y_vals.append(y)
            y_los.append(float(row["ci_lower_95"]))
            y_his.append(float(row["ci_upper_95"]))
            labels.append(sample)
        order = np.argsort(x_vals)
        x_arr = np.array(x_vals)[order]
        y_arr = np.array(y_vals)[order]
        lo_arr = np.array(y_los)[order]
        hi_arr = np.array(y_his)[order]
        labels_arr = np.array(labels)[order]
        ax_b.plot(x_arr, y_arr, color=panel_color(panel), linewidth=1.5, alpha=0.75, zorder=2)
        ax_b.errorbar(
            x_arr,
            y_arr,
            yerr=[y_arr - lo_arr, hi_arr - y_arr],
            fmt=panel_marker(panel),
            color=panel_color(panel),
            markeredgecolor=COL_DARK,
            markeredgewidth=0.5,
            markersize=6.3,
            capsize=0,
            linewidth=1.2,
            label=short_panel_label(panel),
            zorder=3,
        )
        if len(sub_metric):
            r_val = float(sub_metric["sirmium_gradient_r"].iloc[0])
            ax_b.text(
                0.02,
                0.94 - (0.075 if panel == "ev13_sensitivity" else 0.0),
                f"{short_panel_label(panel)} r = {r_val:.3f}",
                transform=ax_b.transAxes,
                ha="left",
                va="top",
                color=panel_color(panel),
                fontsize=8.7,
                fontweight="semibold",
            )
        if panel == "main":
            label_offsets = {
                "I1116": (3.0, 0.00008),
                "I32299": (2.0, 0.00022),
                "I15541": (2.6, -0.00010),
                "I15542": (2.6, 0.00022),
            }
            for x0, y0, lab in zip(x_arr, y_arr, labels_arr):
                dx, dy = label_offsets.get(str(lab), (2.0, 0.00010))
                ax_b.text(
                    x0 + dx,
                    y0 + dy,
                    str(lab),
                    fontsize=7.6,
                    va="center",
                    color=COL_DARK,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=0.8),
                    zorder=5,
                )

    ax_b.set_xlabel("Geographic distance from Sirmium (km)", fontweight="semibold")
    ax_b.set_ylabel("Mean distance to R1a frame", fontweight="semibold")
    ax_b.set_title("b  I2a R1a-frame distance follows the landscape gradient", loc="left", fontweight="semibold", pad=8)
    ax_b.grid(axis="both", color=COL_GRID, linewidth=0.8)
    hide_spines(ax_b)
    ax_b.legend(frameon=False, loc="lower right", fontsize=8.4)

    categories = [
        ("loo_statistic", "LOO"),
        ("sirmium_gradient_r", "Sirmium"),
        ("ev13_difference", "E-V13"),
        ("homogenization_separation", "Homog.\n(non-pos.)"),
        ("all_four_pre_specified_directions_ok", "All\n4"),
    ]
    x = np.arange(len(categories))
    width = 0.34
    for panel, offset in [("main", -width / 2), ("ev13_sensitivity", width / 2)]:
        vals = []
        for metric, _ in categories:
            hit = bootstrap_summary[(bootstrap_summary["panel"].eq(panel)) & (bootstrap_summary["metric"].eq(metric))]
            vals.append(float(hit["direction_preservation_fraction"].iloc[0]) if len(hit) else np.nan)
        bars = ax_c.bar(
            x + offset,
            vals,
            width=width,
            color=panel_color(panel),
            edgecolor=COL_DARK,
            linewidth=0.55,
            label=short_panel_label(panel),
            alpha=0.92,
        )
        for rect, val in zip(bars, vals):
            if np.isfinite(val):
                ax_c.text(
                    rect.get_x() + rect.get_width() / 2,
                    val + 0.012,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.6,
                    rotation=0,
                )
    ax_c.axhline(0.5, color="#BBBBBB", linestyle=(0, (3, 3)), linewidth=1.0, zorder=0)
    ax_c.set_ylim(0, 1.105)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([lab for _, lab in categories])
    ax_c.set_ylabel("Bootstrap direction-preservation fraction", fontweight="semibold")
    ax_c.set_title("c  SNP-bootstrap preservation of pre-specified directions", loc="left", fontweight="semibold", pad=8)
    ax_c.grid(axis="y", color=COL_GRID, linewidth=0.8)
    hide_spines(ax_c)
    ax_c.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.125),
        ncol=2,
        columnspacing=1.35,
        handlelength=1.7,
        handletextpad=0.55,
        fontsize=8.4,
    )

    condition_order = ["full", "drop_I26766", "drop_I26748", "drop_I35012"]
    panel_order = [p for p in ["main", "ev13_sensitivity"] if p in set(leaveone_df["panel"])]
    heat = np.full((len(condition_order), len(panel_order)), np.nan)
    for i, cond in enumerate(condition_order):
        for j, panel in enumerate(panel_order):
            hit = leaveone_df[(leaveone_df["panel"].eq(panel)) & (leaveone_df["r1a_condition"].eq(cond))]
            if len(hit):
                heat[i, j] = float(hit["n_pre_specified_directions_ok"].iloc[0])
    im = ax_d1.imshow(heat, vmin=0, vmax=4, cmap="YlGn", aspect="auto")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            if np.isfinite(heat[i, j]):
                txt_color = "white" if heat[i, j] >= 3 else COL_DARK
                ax_d1.text(j, i, f"{int(heat[i, j])}/4", ha="center", va="center", fontsize=8.5, color=txt_color, fontweight="semibold")
    ax_d1.set_xticks(np.arange(len(panel_order)))
    ax_d1.set_xticklabels([short_panel_label(p) for p in panel_order], rotation=45, ha="right")
    ax_d1.set_yticks(np.arange(len(condition_order)))
    ax_d1.set_yticklabels(["Full", "Drop\nI26766", "Drop\nI26748", "Drop\nI35012"])
    ax_d1.set_title("d1  Leave-one-R1a frame", loc="left", fontweight="semibold", pad=8, fontsize=10.4)
    ax_d1.tick_params(length=0)
    ax_d1.set_xticks(np.arange(-0.5, len(panel_order), 1), minor=True)
    ax_d1.set_yticks(np.arange(-0.5, len(condition_order), 1), minor=True)
    ax_d1.grid(which="minor", color="white", linewidth=1.2)
    for spine in ax_d1.spines.values():
        spine.set_visible(False)

    y_positions = {"main": 1.0, "ev13_sensitivity": 0.0}
    for panel in panel_order:
        sub = partition_df[partition_df["panel"].eq(panel)].copy()
        if sub.empty:
            continue
        y = y_positions[panel]
        jitter = np.linspace(-0.055, 0.055, len(sub))
        ax_d2.scatter(
            sub["highland_minus_lowland_distance_to_r1a"],
            np.full(len(sub), y) + jitter,
            s=34,
            color="#BDBDBD",
            edgecolor=COL_DARK,
            linewidth=0.35,
            alpha=0.85,
            zorder=2,
        )
        obs = sub[sub["is_observed_geographic_partition"]]
        if len(obs):
            obs_x = float(obs["highland_minus_lowland_distance_to_r1a"].iloc[0])
            rank = int(obs["rank_descending"].iloc[0])
            n_part = int(obs["n_partitions"].iloc[0])
            ax_d2.scatter(
                [obs_x],
                [y],
                marker="*",
                s=135,
                color=COL_HILITE,
                edgecolor=COL_DARK,
                linewidth=0.55,
                zorder=5,
            )
            ax_d2.text(
                0.97,
                y + 0.17,
                f"rank {rank}/{n_part}",
                transform=ax_d2.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=7.8,
                color=COL_HILITE,
                fontweight="semibold",
                clip_on=False,
            )
    ax_d2.axvline(0, color="#888888", linestyle=(0, (3, 3)), linewidth=1.0, zorder=1)
    ax_d2.set_ylim(-0.25, 1.25)
    ax_d2.set_yticks([1.0, 0.0])
    ax_d2.set_yticklabels(["Main", "Nustar"])
    ax_d2.set_xlabel("Assigned highland minus lowland\nmean distance to R1a", fontweight="semibold", fontsize=8.8)
    ax_d2.set_title("d2  I2a partition stress", loc="left", fontweight="semibold", pad=8, fontsize=10.4)
    ax_d2.grid(axis="x", color=COL_GRID, linewidth=0.8)
    partition_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=5.4,
            markerfacecolor="#BDBDBD",
            markeredgecolor=COL_DARK,
            markeredgewidth=0.35,
            label="Alternative 2-vs-2 split",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="*",
            linestyle="",
            markersize=8.8,
            markerfacecolor=COL_HILITE,
            markeredgecolor=COL_DARK,
            markeredgewidth=0.55,
            label="Observed geographic split",
        ),
    ]
    ax_d2.legend(
        handles=partition_handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.04),
        fontsize=7.4,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    hide_spines(ax_d2)

    fig.suptitle("Complete-case autosomal distance concordance audit", fontsize=16.5, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.012,
        "Distance-space robustness audit only: not an ancestry estimator and not a paternal-lineage origin or continuity test.",
        ha="center",
        va="bottom",
        fontsize=9.0,
        color="#555555",
    )
    fig.subplots_adjust(top=0.93, bottom=0.115, left=0.07, right=0.945)

    fig.savefig(outputs.figure_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs.figure_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    manifest = {
        "script": "06_complete_case_distance_audit.py",
        "timestamp_utc": utc_now(),
        "outputs": {
            "png": relpath(outputs.base, outputs.figure_png),
            "pdf": relpath(outputs.base, outputs.figure_pdf),
        },
        "panels": {
            "a": "Mean distance from each non-R1a sample to the R1a reference frame, with SNP-bootstrap 95% intervals.",
            "b": "I2a distance-to-R1a values plotted against geographic distance from Sirmium.",
            "c": "Fraction of SNP-bootstrap replicates preserving each pre-specified directional contrast and all four jointly.",
            "d1": "Leave-one-R1a-reference directional count audit.",
            "d2": "All balanced I2a partition enumeration; grey circles are alternative 2-vs-2 splits and the orange star marks the observed geographic split.",
        },
        "scope": "descriptive distance-space audit only; not an ancestry estimator or paternal-lineage origin test",
    }
    outputs.figure_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.n_bootstrap <= 0:
        print("[ERROR] --n-bootstrap must be positive")
        return 1

    outputs = resolve_output_paths(args)
    outputs.outdir.mkdir(parents=True, exist_ok=True)
    outputs.figdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Complete-case autosomal distance-space concordance audit")
    print(f"Base: {outputs.base}")
    print(f"Panels: {', '.join(args.panels)}")
    print(f"Output directory: {outputs.outdir}")
    print(f"Figure directory: {outputs.figdir}")
    print(f"Bootstrap replicates per panel: {args.n_bootstrap}")
    print(f"Seed: {args.seed}")
    print("=" * 80)

    try:
        panel_data_list: List[PanelData] = []
        observed_rows: List[Dict[str, object]] = []
        profile_parts: List[pd.DataFrame] = []
        bootstrap_metric_parts: List[pd.DataFrame] = []
        bootstrap_profile_parts: List[pd.DataFrame] = []
        leaveone_parts: List[pd.DataFrame] = []
        partition_parts: List[pd.DataFrame] = []
        projection_rows: List[Dict[str, object]] = []

        for panel in args.panels:
            pdata = load_panel_data(outputs.base, panel)
            panel_data_list.append(pdata)
            print(f"[OK] Loaded {panel}: {pdata.n_snps} complete-case SNPs; samples: {', '.join(pdata.samples)}")
            dist_mat = compute_distance_matrix(pdata.allele_array)

            observed_rows.append(observed_metrics_row(pdata, dist_mat, outputs.base))
            profile_parts.append(make_r1a_profile(pdata, dist_mat))
            leaveone_parts.append(r1a_leaveone_audit(pdata, dist_mat))
            partition_parts.append(enumerate_i2a_partitions(pdata, dist_mat))
            projection_rows.append(projection_diagnostics(pdata, dist_mat))

            boot_metrics, boot_profiles = bootstrap_panel(
                panel_data=pdata,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
                progress_interval=args.progress_interval,
            )
            bootstrap_metric_parts.append(boot_metrics)
            bootstrap_profile_parts.append(boot_profiles)

        observed_df = pd.DataFrame(observed_rows)
        r1a_profile_df = pd.concat(profile_parts, ignore_index=True)
        boot_metrics_df = pd.concat(bootstrap_metric_parts, ignore_index=True)
        boot_profile_df = pd.concat(bootstrap_profile_parts, ignore_index=True)
        boot_summary_df = summarize_bootstrap_metrics(boot_metrics_df)
        boot_r1a_profile_summary_df = summarize_bootstrap_r1a_profile(boot_profile_df, r1a_profile_df)
        leaveone_df = pd.concat(leaveone_parts, ignore_index=True)
        partition_df = pd.concat(partition_parts, ignore_index=True)
        projection_df = pd.DataFrame(projection_rows)
        audit_summary_df = make_audit_summary(observed_df, boot_summary_df, leaveone_df, partition_df, projection_df)

        observed_df.to_csv(outputs.observed_metrics, index=False)
        r1a_profile_df.to_csv(outputs.r1a_profile, index=False)
        boot_metrics_df.to_csv(outputs.bootstrap_metrics, index=False)
        boot_summary_df.to_csv(outputs.bootstrap_summary, index=False)
        boot_r1a_profile_summary_df.to_csv(outputs.bootstrap_r1a_profile_summary, index=False)
        leaveone_df.to_csv(outputs.r1a_leaveone, index=False)
        partition_df.to_csv(outputs.i2a_partitions, index=False)
        projection_df.to_csv(outputs.projection_diagnostics, index=False)
        audit_summary_df.to_csv(outputs.audit_summary, index=False)

        write_text_summary(outputs.text_summary, audit_summary_df, boot_summary_df, partition_df, leaveone_df)
        write_metadata(outputs.metadata_json, args, outputs, panel_data_list)

        if not args.skip_figure:
            make_figure(
                outputs=outputs,
                r1a_profile_summary=boot_r1a_profile_summary_df,
                observed_metrics=observed_df,
                bootstrap_summary=boot_summary_df,
                leaveone_df=leaveone_df,
                partition_df=partition_df,
            )

        print("\nObserved directional summary:")
        display_cols = [
            "panel",
            "n_snps",
            "loo_statistic",
            "sirmium_gradient_r",
            "ev13_difference",
            "homogenization_separation",
            "i2a_highland_minus_lowland_distance_to_r1a",
            "weighted_directional_score",
            "n_pre_specified_directions_ok",
        ]
        print(observed_df[display_cols].to_string(index=False))

        print("\nBootstrap direction-preservation summary:")
        print(
            boot_summary_df[
                ["panel", "metric", "direction_preservation_fraction", "bootstrap_median", "ci_lower_95", "ci_upper_95"]
            ].to_string(index=False)
        )

        print("\nOutputs written:")
        for path in [
            outputs.observed_metrics,
            outputs.r1a_profile,
            outputs.bootstrap_metrics,
            outputs.bootstrap_summary,
            outputs.bootstrap_r1a_profile_summary,
            outputs.r1a_leaveone,
            outputs.i2a_partitions,
            outputs.projection_diagnostics,
            outputs.audit_summary,
            outputs.metadata_json,
            outputs.text_summary,
        ]:
            print(f"  - {path}")
        if not args.skip_figure:
            print(f"  - {outputs.figure_png}")
            print(f"  - {outputs.figure_pdf}")
            print(f"  - {outputs.figure_manifest}")

    except Exception as exc:
        print(f"[ERROR] Distance-space audit failed: {exc}")
        return 1

    print("=" * 80)
    print("[OK] Complete-case distance-space audit complete")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
