#!/usr/bin/env python3
"""
Run the independent allele-sharing analysis for a selected sample panel.

This script implements four pre-specified contrasts on a strict complete-case
pseudo-haploid autosomal matrix:

1. Leave-one-out outlier diagnostic
2. Sirmium geographic gradient
3. E-V13 replication contrast
4. Autosomal homogenization contrast

For the main panel, the lowland E-V13 comparator is I27297.
For the E-V13 sensitivity panel, the lowland E-V13 comparator is I28388.

The pairwise autosomal distance used throughout is the mismatch proportion
across retained complete-case SNPs, equivalent to 1 - IBS for pseudo-haploid
single-base calls.

Default usage
-------------
Main panel:
    python 03_allele_sharing_panel.py --panel main --base .

E-V13 sensitivity panel:
    python 03_allele_sharing_panel.py --panel ev13_sensitivity --base .

Custom matrix path:
    python 03_allele_sharing_panel.py --panel main --matrix path\\to\\matrix.tsv

Inputs
------
By default, this script expects the panel-specific matrix produced by
02_build_matrix.py:

    matrix_main/complete_case_9_sample_matrix.tsv
    matrix_ev13_sensitivity/complete_case_9_sample_matrix.tsv

Outputs
-------
By default, outputs are written beside the input matrix:

    matrix_<panel>/allele_sharing_results.json
    matrix_<panel>/allele_sharing_summary.csv
    matrix_<panel>/pairwise_distance_matrix.tsv

Dependencies
------------
    numpy
    pandas
    scipy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations, permutations
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


PANELS: Dict[str, Dict[str, object]] = {
    "main": {
        "i2a_lowland": ["I1116", "I32299"],
        "i2a_highland": ["I15541", "I15542"],
        "ev13_highland": ["I15537"],
        "ev13_lowland": ["I27297"],
        "r1a_references": ["I26766", "I26748", "I35012"],
        "site_coords": {
            "I1116": (44.8889, 19.7494),      # Gomolava
            "I32299": (44.7320, 21.2340),     # Viminacium Svetinja
            "I15541": (43.6450, 22.2650),     # Timacum Minus Kuline
            "I15542": (43.6450, 22.2650),     # Timacum Minus Kuline
            "I15537": (43.6450, 22.2650),     # Timacum Minus Kuline
            "I27297": (44.7546, 20.2621),     # Kormadin
        },
    },
    "ev13_sensitivity": {
        "i2a_lowland": ["I1116", "I32299"],
        "i2a_highland": ["I15541", "I15542"],
        "ev13_highland": ["I15537"],
        "ev13_lowland": ["I28388"],
        "r1a_references": ["I26766", "I26748", "I35012"],
        "site_coords": {
            "I1116": (44.8889, 19.7494),       # Gomolava
            "I32299": (44.7320, 21.2340),      # Viminacium Svetinja
            "I15541": (43.6450, 22.2650),      # Timacum Minus Kuline
            "I15542": (43.6450, 22.2650),      # Timacum Minus Kuline
            "I15537": (43.6450, 22.2650),      # Timacum Minus Kuline
            "I28388": (45.338156, 18.845028),
        },
    },
}


SIRMIUM_COORDS: Tuple[float, float] = (44.9764, 19.6122)

WEIGHTS: Dict[str, float] = {
    "LOO": 0.9,
    "Gradient": 0.7,
    "EV13": 0.6,
    "Homogenization": 0.3,
}

DEFAULT_BASE = Path.cwd()
DEFAULT_N_PERM = 10_000
DEFAULT_N_BOOTSTRAP = 1_000
DEFAULT_SEED = 42
VALID_ALLELES = {"A", "C", "G", "T"}


@dataclass
class OutputPaths:
    base: Path
    matrix_path: Path
    output_json: Path
    output_csv: Path
    distance_matrix_tsv: Path


@dataclass
class TestSummaryRow:
    test_number: int
    test_name: str
    statistic_name: str
    statistic: Optional[float]
    ci_lower_95: Optional[float]
    ci_upper_95: Optional[float]
    p_value: Optional[float]
    predicted_direction: str
    observed_direction_code: int
    weight: float
    weighted_contribution: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the independent allele-sharing analysis for a selected "
            "complete-case pseudo-haploid panel."
        )
    )
    parser.add_argument(
        "--panel",
        choices=sorted(PANELS.keys()),
        default="main",
        help="Sample panel to analyze. Default: main.",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
        help="Project base directory. Default: current working directory.",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=None,
        help=(
            "Path to the complete-case matrix. If omitted, "
            "'<base>/matrix_<panel>/complete_case_9_sample_matrix.tsv' is used."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Path to the detailed output JSON. If omitted, "
            "'<matrix_dir>/allele_sharing_results.json' is used."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Path to compact output CSV summary. If omitted, "
            "'<matrix_dir>/allele_sharing_summary.csv' is used."
        ),
    )
    parser.add_argument(
        "--distance-matrix",
        type=Path,
        default=None,
        help=(
            "Path to pairwise distance matrix TSV. If omitted, "
            "'<matrix_dir>/pairwise_distance_matrix.tsv' is used."
        ),
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=DEFAULT_N_PERM,
        help=(
            "Permutation-count parameter recorded for reproducibility. Exact finite "
            "enumeration is used for implemented label tests. "
            f"Default: {DEFAULT_N_PERM}."
        ),
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOTSTRAP,
        help=f"Number of bootstrap replicates for SNP-bootstrap intervals. Default: {DEFAULT_N_BOOTSTRAP}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed. Default: {DEFAULT_SEED}.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
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


def default_matrix_path(base: Path, panel: str) -> Path:
    return base / f"matrix_{panel}" / "complete_case_9_sample_matrix.tsv"


def resolve_output_paths(args: argparse.Namespace) -> OutputPaths:
    base = args.base.resolve()
    matrix_path = args.matrix.resolve() if args.matrix else default_matrix_path(base, args.panel).resolve()
    matrix_dir = matrix_path.parent

    output_json = (
        args.output_json.resolve()
        if args.output_json
        else matrix_dir / "allele_sharing_results.json"
    )
    output_csv = (
        args.output_csv.resolve()
        if args.output_csv
        else matrix_dir / "allele_sharing_summary.csv"
    )
    distance_matrix_tsv = (
        args.distance_matrix.resolve()
        if args.distance_matrix
        else matrix_dir / "pairwise_distance_matrix.tsv"
    )

    return OutputPaths(
        base=base,
        matrix_path=matrix_path,
        output_json=output_json,
        output_csv=output_csv,
        distance_matrix_tsv=distance_matrix_tsv,
    )


def haversine(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Great-circle distance in kilometers between two (latitude, longitude) pairs.
    """
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


def build_sample_lists(panel_cfg: Dict[str, object]) -> Dict[str, List[str]]:
    i2a_lowland = list(panel_cfg["i2a_lowland"])
    i2a_highland = list(panel_cfg["i2a_highland"])
    ev13_highland = list(panel_cfg["ev13_highland"])
    ev13_lowland = list(panel_cfg["ev13_lowland"])
    r1a_references = list(panel_cfg["r1a_references"])

    all_i2a = i2a_lowland + i2a_highland
    all_targets = all_i2a + ev13_highland + ev13_lowland + r1a_references

    return {
        "i2a_lowland": i2a_lowland,
        "i2a_highland": i2a_highland,
        "ev13_highland": ev13_highland,
        "ev13_lowland": ev13_lowland,
        "r1a_references": r1a_references,
        "all_i2a": all_i2a,
        "all_targets": all_targets,
    }


def validate_matrix(df: pd.DataFrame, target_samples: List[str], matrix_path: Path) -> None:
    """
    Validate that the matrix is a strict complete-case allele matrix for targets.
    """
    if len(df) == 0:
        raise ValueError(f"Matrix has zero rows: {matrix_path}")

    required_metadata = ["chrom", "pos"]
    missing_metadata = [column for column in required_metadata if column not in df.columns]
    if missing_metadata:
        raise ValueError(
            f"Matrix is missing required metadata columns: {', '.join(missing_metadata)}"
        )

    missing_samples = sorted(set(target_samples) - set(df.columns))
    if missing_samples:
        raise ValueError(
            f"Matrix is missing required sample columns: {', '.join(missing_samples)}"
        )

    sample_df = df[target_samples]

    if sample_df.isna().any().any():
        raise ValueError("Matrix contains missing values in target sample columns.")

    invalid_values: Dict[str, List[str]] = {}
    for sample in target_samples:
        observed = set(sample_df[sample].astype(str).unique())
        invalid = sorted(observed - VALID_ALLELES)
        if invalid:
            invalid_values[sample] = invalid

    if invalid_values:
        messages = [
            f"{sample}: {', '.join(values[:10])}"
            for sample, values in invalid_values.items()
        ]
        raise ValueError(
            "Matrix contains invalid or missing allele calls in target samples: "
            + "; ".join(messages)
        )


def load_matrix(matrix_path: Path, target_samples: List[str]) -> pd.DataFrame:
    """
    Load and validate the complete-case matrix.
    """
    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

    df = pd.read_csv(matrix_path, sep="\t", dtype=str)
    validate_matrix(df, target_samples, matrix_path)

    ordered_columns = ["chrom", "pos"] + target_samples
    return df[ordered_columns].copy()


def compute_distance_matrix(df: pd.DataFrame, sample_list: List[str]) -> np.ndarray:
    """
    Compute pairwise IBS distance as mismatch proportion across retained rows.

    With pseudo-haploid complete-case calls, this is equivalent to 1 - IBS.
    """
    n_rows = len(df)
    if n_rows == 0:
        raise ValueError("Cannot compute distances on a zero-row matrix.")

    n = len(sample_list)
    dist_mat = np.zeros((n, n), dtype=float)

    for i, s1 in enumerate(sample_list):
        for j in range(i + 1, n):
            s2 = sample_list[j]
            diff = int((df[s1] != df[s2]).sum())
            dist = diff / n_rows
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist

    return dist_mat


def write_distance_matrix(
    dist_mat: np.ndarray,
    sample_list: List[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dist_df = pd.DataFrame(dist_mat, index=sample_list, columns=sample_list)
    dist_df.to_csv(output_path, sep="\t", index=True)


def mean_dist_to_ref(sample_idx: int, ref_indices: List[int], dist_mat: np.ndarray) -> float:
    if not ref_indices:
        return float("nan")
    return float(np.mean([dist_mat[sample_idx, ref_idx] for ref_idx in ref_indices]))


def permutation_test_pvalue(
    obs_stat: float,
    null_dist: np.ndarray,
    alternative: str = "greater",
) -> float:
    """
    Compute an exact one-sided permutation p-value.
    """
    if len(null_dist) == 0:
        return float("nan")

    if alternative == "greater":
        return float(np.sum(null_dist >= obs_stat) / len(null_dist))

    if alternative == "less":
        return float(np.sum(null_dist <= obs_stat) / len(null_dist))

    raise ValueError("alternative must be 'greater' or 'less'")


def two_sided_permutation_pvalue(obs_stat: float, null_dist: np.ndarray) -> float:
    """
    Compute an exact two-sided permutation p-value.

    This is based on the more extreme tail under the empirical null and is
    bounded at 1.
    """
    if len(null_dist) == 0:
        return float("nan")

    left = np.sum(null_dist <= obs_stat)
    right = np.sum(null_dist >= obs_stat)
    p = 2.0 * min(left, right) / len(null_dist)
    return float(min(1.0, p))


def bootstrap_ci(
    data: pd.DataFrame,
    stat_func: Callable[[pd.DataFrame], float],
    rng: np.random.Generator,
    n_bootstrap: int,
    alpha: float = 0.05,
) -> Tuple[float, float, float, int]:
    """
    Compute a SNP-bootstrap interval by resampling SNP rows with replacement.
    """
    n_rows = len(data)
    boot_stats: List[float] = []

    for _ in range(n_bootstrap):
        boot_indices = rng.choice(n_rows, size=n_rows, replace=True)
        boot_df = data.iloc[boot_indices]
        try:
            stat = stat_func(boot_df)
            if np.isfinite(stat):
                boot_stats.append(float(stat))
        except Exception:
            continue

    if len(boot_stats) < 10:
        return float("nan"), float("nan"), float("nan"), len(boot_stats)

    boot_stats_arr = np.array(boot_stats, dtype=float)
    ci_low = float(np.percentile(boot_stats_arr, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_stats_arr, 100 * (1 - alpha / 2)))
    return float(np.mean(boot_stats_arr)), ci_low, ci_high, len(boot_stats)


def compute_loo_shift(
    dist_mat: np.ndarray,
    sample_list: List[str],
    i2a_idx: List[int],
    r1a_idx: List[int],
    low_idx: List[int],
    high_idx: List[int],
) -> Tuple[float, Dict[str, float], float]:
    """
    Compute the leave-one-out summary statistic.

    S_LOO is the mean shift among highland I2a removals minus the mean shift
    among lowland I2a removals. Negative values are predicted by the structured
    regional ancestry model.
    """
    between = [dist_mat[i, r] for i in i2a_idx for r in r1a_idx]
    within_r1a = [
        dist_mat[r1, r2]
        for idx_r, r1 in enumerate(r1a_idx)
        for r2 in r1a_idx[idx_r + 1 :]
    ]

    delta_base = float(np.mean(between) - np.mean(within_r1a))

    shifts: Dict[str, float] = {}
    for idx in i2a_idx:
        i2a_loo = [j for j in i2a_idx if j != idx]
        between_loo = [dist_mat[j, r] for j in i2a_loo for r in r1a_idx]
        delta_loo = float(np.mean(between_loo) - np.mean(within_r1a))
        shifts[sample_list[idx]] = delta_loo - delta_base

    high_shift = float(np.mean([shifts[sample_list[idx]] for idx in high_idx]))
    low_shift = float(np.mean([shifts[sample_list[idx]] for idx in low_idx]))

    return high_shift - low_shift, shifts, delta_base


def test_loo_outlier(
    dist_mat: np.ndarray,
    sample_list: List[str],
    df_full: pd.DataFrame,
    i2a_low_idx: List[int],
    i2a_high_idx: List[int],
    r1a_idx: List[int],
    n_perm: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    i2a_idx = i2a_low_idx + i2a_high_idx
    obs_stat, shifts, delta_base = compute_loo_shift(
        dist_mat, sample_list, i2a_idx, r1a_idx, i2a_low_idx, i2a_high_idx
    )

    all_i2a = i2a_low_idx + i2a_high_idx
    n_low = len(i2a_low_idx)
    null_stats: List[float] = []

    for perm_low_tuple in combinations(all_i2a, n_low):
        perm_low = list(perm_low_tuple)
        perm_high = [idx for idx in all_i2a if idx not in perm_low]
        perm_stat, _, _ = compute_loo_shift(
            dist_mat, sample_list, i2a_idx, r1a_idx, perm_low, perm_high
        )
        null_stats.append(float(perm_stat))

    p_value = permutation_test_pvalue(
        obs_stat=obs_stat,
        null_dist=np.array(null_stats, dtype=float),
        alternative="less",
    )

    def stat_func(df_boot: pd.DataFrame) -> float:
        boot_dist = compute_distance_matrix(df_boot, sample_list)
        stat, _, _ = compute_loo_shift(
            boot_dist, sample_list, i2a_idx, r1a_idx, i2a_low_idx, i2a_high_idx
        )
        return float(stat)

    _, ci_low, ci_high, n_boot_valid = bootstrap_ci(
        df_full, stat_func=stat_func, rng=rng, n_bootstrap=n_bootstrap
    )

    direction = 1 if obs_stat < 0 else (-1 if obs_stat > 0 else 0)

    return {
        "statistic": float(obs_stat),
        "p_value": float(p_value),
        "ci_95": [ci_low, ci_high],
        "direction": int(direction),
        "n_permutations": int(len(null_stats)),
        "permutation_method": "exact_enumeration",
        "n_bootstrap_valid": int(n_boot_valid),
        "loo_shifts": shifts,
        "delta_base": float(delta_base),
    }


def compute_sirmium_correlation(
    dist_mat: np.ndarray,
    sample_list: List[str],
    i2a_idx: List[int],
    r1a_idx: List[int],
    site_coords: Dict[str, Tuple[float, float]],
) -> Tuple[float, List[float], List[float], List[str]]:
    auto_dists: List[float] = []
    geo_dists: List[float] = []
    used_samples: List[str] = []

    for idx in i2a_idx:
        sample = sample_list[idx]
        if sample not in site_coords:
            continue

        dist_to_r1a = mean_dist_to_ref(idx, r1a_idx, dist_mat)
        geo_dist = haversine(site_coords[sample], SIRMIUM_COORDS)

        auto_dists.append(float(dist_to_r1a))
        geo_dists.append(float(geo_dist))
        used_samples.append(sample)

    if len(auto_dists) < 3:
        return float("nan"), auto_dists, geo_dists, used_samples

    r_value, _ = pearsonr(geo_dists, auto_dists)
    return float(r_value), auto_dists, geo_dists, used_samples


def test_sirmium_gradient(
    dist_mat: np.ndarray,
    sample_list: List[str],
    df_full: pd.DataFrame,
    i2a_idx: List[int],
    r1a_idx: List[int],
    site_coords: Dict[str, Tuple[float, float]],
    n_perm: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    obs_r, auto_dists, geo_dists, used_samples = compute_sirmium_correlation(
        dist_mat, sample_list, i2a_idx, r1a_idx, site_coords
    )

    if not np.isfinite(obs_r):
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "ci_95": [float("nan"), float("nan")],
            "direction": 0,
            "n_permutations": 0,
            "permutation_method": "exact_enumeration",
            "n_bootstrap_valid": 0,
            "auto_dists": auto_dists,
            "geo_dists": geo_dists,
            "used_samples": used_samples,
        }

    null_r: List[float] = []
    for perm_auto in permutations(auto_dists):
        r_perm, _ = pearsonr(geo_dists, perm_auto)
        null_r.append(float(r_perm))

    p_value = permutation_test_pvalue(
        obs_stat=obs_r,
        null_dist=np.array(null_r, dtype=float),
        alternative="greater",
    )

    def stat_func(df_boot: pd.DataFrame) -> float:
        boot_dist = compute_distance_matrix(df_boot, sample_list)
        r_boot, _, _, _ = compute_sirmium_correlation(
            boot_dist, sample_list, i2a_idx, r1a_idx, site_coords
        )
        return float(r_boot)

    _, ci_low, ci_high, n_boot_valid = bootstrap_ci(
        df_full, stat_func=stat_func, rng=rng, n_bootstrap=n_bootstrap
    )

    direction = 1 if obs_r > 0 else (-1 if obs_r < 0 else 0)

    return {
        "statistic": float(obs_r),
        "p_value": float(p_value),
        "ci_95": [ci_low, ci_high],
        "direction": int(direction),
        "n_permutations": int(len(null_r)),
        "permutation_method": "exact_enumeration",
        "n_bootstrap_valid": int(n_boot_valid),
        "auto_dists": auto_dists,
        "geo_dists": geo_dists,
        "used_samples": used_samples,
    }


def test_ev13_replication(
    dist_mat: np.ndarray,
    sample_list: List[str],
    df_full: pd.DataFrame,
    ev13_high_idx: List[int],
    ev13_low_idx: List[int],
    r1a_idx: List[int],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    if len(ev13_high_idx) != 1 or len(ev13_low_idx) != 1:
        return {
            "statistic": float("nan"),
            "ci_95": [float("nan"), float("nan")],
            "direction": 0,
            "n_bootstrap_valid": 0,
            "highland_dist": float("nan"),
            "lowland_dist": float("nan"),
        }

    high_dist = mean_dist_to_ref(ev13_high_idx[0], r1a_idx, dist_mat)
    low_dist = mean_dist_to_ref(ev13_low_idx[0], r1a_idx, dist_mat)
    obs_stat = float(high_dist - low_dist)

    def stat_func(df_boot: pd.DataFrame) -> float:
        boot_dist = compute_distance_matrix(df_boot, sample_list)
        high_boot = mean_dist_to_ref(ev13_high_idx[0], r1a_idx, boot_dist)
        low_boot = mean_dist_to_ref(ev13_low_idx[0], r1a_idx, boot_dist)
        return float(high_boot - low_boot)

    _, ci_low, ci_high, n_boot_valid = bootstrap_ci(
        df_full, stat_func=stat_func, rng=rng, n_bootstrap=n_bootstrap
    )

    direction = 1 if obs_stat > 0 else (-1 if obs_stat < 0 else 0)

    return {
        "statistic": float(obs_stat),
        "ci_95": [ci_low, ci_high],
        "direction": int(direction),
        "n_bootstrap_valid": int(n_boot_valid),
        "highland_dist": float(high_dist),
        "lowland_dist": float(low_dist),
    }


def test_homogenization(
    dist_mat: np.ndarray,
    sample_list: List[str],
    df_full: pd.DataFrame,
    i2a_idx: List[int],
    r1a_idx: List[int],
    n_perm: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
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

    pooled_within = within_i2a + within_r1a
    obs_sep = float(np.mean(between) - np.mean(pooled_within))

    all_idx = i2a_idx + r1a_idx
    n_i2a = len(i2a_idx)
    null_sep: List[float] = []

    for perm_i2a_tuple in combinations(all_idx, n_i2a):
        perm_i2a = list(perm_i2a_tuple)
        perm_r1a = [idx for idx in all_idx if idx not in perm_i2a]

        perm_within_i2a = [
            dist_mat[i, j]
            for idx_i, i in enumerate(perm_i2a)
            for j in perm_i2a[idx_i + 1 :]
        ]
        perm_within_r1a = [
            dist_mat[r1, r2]
            for idx_r, r1 in enumerate(perm_r1a)
            for r2 in perm_r1a[idx_r + 1 :]
        ]
        perm_between = [dist_mat[i, r] for i in perm_i2a for r in perm_r1a]
        perm_sep = float(
            np.mean(perm_between) - np.mean(perm_within_i2a + perm_within_r1a)
        )
        null_sep.append(perm_sep)

    p_value = two_sided_permutation_pvalue(
        obs_stat=obs_sep,
        null_dist=np.array(null_sep, dtype=float),
    )

    def stat_func(df_boot: pd.DataFrame) -> float:
        boot_dist = compute_distance_matrix(df_boot, sample_list)

        within_i2a_b = [
            boot_dist[i, j]
            for idx_i, i in enumerate(i2a_idx)
            for j in i2a_idx[idx_i + 1 :]
        ]
        within_r1a_b = [
            boot_dist[r1, r2]
            for idx_r, r1 in enumerate(r1a_idx)
            for r2 in r1a_idx[idx_r + 1 :]
        ]
        between_b = [boot_dist[i, r] for i in i2a_idx for r in r1a_idx]
        return float(np.mean(between_b) - np.mean(within_i2a_b + within_r1a_b))

    _, ci_low, ci_high, n_boot_valid = bootstrap_ci(
        df_full, stat_func=stat_func, rng=rng, n_bootstrap=n_bootstrap
    )

    direction = 1 if obs_sep <= 0 else -1

    return {
        "statistic": float(obs_sep),
        "p_value": float(p_value),
        "ci_95": [ci_low, ci_high],
        "direction": int(direction),
        "n_permutations": int(len(null_sep)),
        "permutation_method": "exact_enumeration",
        "n_bootstrap_valid": int(n_boot_valid),
        "within_i2a_mean": float(np.mean(within_i2a)),
        "within_r1a_mean": float(np.mean(within_r1a)),
        "pooled_within_mean": float(np.mean(pooled_within)),
        "between_mean": float(np.mean(between)),
    }


def format_float(value: Optional[float], ndigits: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        if not np.isfinite(float(value)):
            return "NA"
    except Exception:
        return "NA"
    return f"{float(value):.{ndigits}f}"


def format_ci(ci: Optional[List[float]]) -> str:
    if ci is None or len(ci) != 2:
        return "NA"
    if any(not np.isfinite(float(x)) for x in ci):
        return "NA"
    return f"[{ci[0]:.6f}, {ci[1]:.6f}]"


def make_summary_rows(
    res1: Dict[str, object],
    res2: Dict[str, object],
    res3: Dict[str, object],
    res4: Dict[str, object],
) -> List[TestSummaryRow]:
    return [
        TestSummaryRow(
            test_number=1,
            test_name="Leave-one-out outlier diagnostic",
            statistic_name="S_LOO",
            statistic=float(res1["statistic"]),
            ci_lower_95=float(res1["ci_95"][0]),
            ci_upper_95=float(res1["ci_95"][1]),
            p_value=float(res1["p_value"]),
            predicted_direction="negative",
            observed_direction_code=int(res1["direction"]),
            weight=WEIGHTS["LOO"],
            weighted_contribution=float(res1["direction"] * WEIGHTS["LOO"]),
        ),
        TestSummaryRow(
            test_number=2,
            test_name="Sirmium geographic gradient",
            statistic_name="r",
            statistic=float(res2["statistic"]),
            ci_lower_95=float(res2["ci_95"][0]),
            ci_upper_95=float(res2["ci_95"][1]),
            p_value=float(res2["p_value"]),
            predicted_direction="positive",
            observed_direction_code=int(res2["direction"]),
            weight=WEIGHTS["Gradient"],
            weighted_contribution=float(res2["direction"] * WEIGHTS["Gradient"]),
        ),
        TestSummaryRow(
            test_number=3,
            test_name="E-V13 replication contrast",
            statistic_name="D_EV13",
            statistic=float(res3["statistic"]),
            ci_lower_95=float(res3["ci_95"][0]),
            ci_upper_95=float(res3["ci_95"][1]),
            p_value=None,
            predicted_direction="positive",
            observed_direction_code=int(res3["direction"]),
            weight=WEIGHTS["EV13"],
            weighted_contribution=float(res3["direction"] * WEIGHTS["EV13"]),
        ),
        TestSummaryRow(
            test_number=4,
            test_name="Autosomal homogenization contrast",
            statistic_name="Sep",
            statistic=float(res4["statistic"]),
            ci_lower_95=float(res4["ci_95"][0]),
            ci_upper_95=float(res4["ci_95"][1]),
            p_value=float(res4["p_value"]),
            predicted_direction="non-positive",
            observed_direction_code=int(res4["direction"]),
            weight=WEIGHTS["Homogenization"],
            weighted_contribution=float(res4["direction"] * WEIGHTS["Homogenization"]),
        ),
    ]


def write_summary_csv(summary_rows: List[TestSummaryRow], score: float, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(row) for row in summary_rows])
    df["directional_consistency_score"] = score
    df.to_csv(output_csv, index=False)


def print_summary(summary_rows: List[TestSummaryRow], score: float) -> None:
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    header = (
        f"{'Test':<38}"
        f"{'Statistic':<12}"
        f"{'Observed':<14}"
        f"{'95% SNP-bootstrap interval':<28}"
        f"{'P-value':<12}"
        f"{'Weight':<10}"
        f"{'Contribution':<12}"
    )
    print(header)
    print("-" * len(header))

    for row in summary_rows:
        p_value = "NA" if row.p_value is None else format_float(row.p_value, 4)
        ci = format_ci([row.ci_lower_95, row.ci_upper_95])
        print(
            f"{str(row.test_number) + '. ' + row.test_name:<38}"
            f"{row.statistic_name:<12}"
            f"{format_float(row.statistic):<14}"
            f"{ci:<28}"
            f"{p_value:<12}"
            f"{row.weight:<10.1f}"
            f"{row.weighted_contribution:+.2f}"
        )

    print("-" * len(header))
    print(f"Directional consistency score: {score:+.3f}")


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    panel_name = args.panel
    panel_cfg = PANELS[panel_name]
    groups = build_sample_lists(panel_cfg)
    paths = resolve_output_paths(args)

    print("=" * 80)
    print("Independent allele-sharing analysis")
    print(f"Panel: {panel_name}")
    print(f"Matrix: {paths.matrix_path}")
    print(f"Output JSON: {paths.output_json}")
    print(f"Output CSV: {paths.output_csv}")
    print(f"Distance matrix TSV: {paths.distance_matrix_tsv}")
    print("Permutation tests: exact enumeration")
    print("[INFO] Exact finite enumeration is used where applicable; --n-perm is recorded for reproducibility.")
    print(f"Bootstrap replicates: {args.n_bootstrap}")
    print(f"Random seed: {args.seed}")
    print("=" * 80)

    try:
        df_full = load_matrix(
            matrix_path=paths.matrix_path,
            target_samples=groups["all_targets"],
        )
        sample_list = groups["all_targets"]
        dist_mat = compute_distance_matrix(df_full, sample_list)
        write_distance_matrix(dist_mat, sample_list, paths.distance_matrix_tsv)

        n_snps = len(df_full)
        sample_to_idx = {sample: idx for idx, sample in enumerate(sample_list)}

        i2a_low_idx = [sample_to_idx[s] for s in groups["i2a_lowland"]]
        i2a_high_idx = [sample_to_idx[s] for s in groups["i2a_highland"]]
        i2a_idx = i2a_low_idx + i2a_high_idx
        ev13_high_idx = [sample_to_idx[s] for s in groups["ev13_highland"]]
        ev13_low_idx = [sample_to_idx[s] for s in groups["ev13_lowland"]]
        r1a_idx = [sample_to_idx[s] for s in groups["r1a_references"]]

        print("[OK] Matrix loaded and validated.")
        print(f"Samples: {', '.join(sample_list)}")
        print(f"Retained complete-case rows: {n_snps}")
        print(f"Pairwise distance matrix written to: {paths.distance_matrix_tsv}")

        print("\nRunning Test 1: leave-one-out outlier diagnostic")
        res1 = test_loo_outlier(
            dist_mat=dist_mat,
            sample_list=sample_list,
            df_full=df_full,
            i2a_low_idx=i2a_low_idx,
            i2a_high_idx=i2a_high_idx,
            r1a_idx=r1a_idx,
            n_perm=args.n_perm,
            n_bootstrap=args.n_bootstrap,
            rng=rng,
        )

        print("Running Test 2: Sirmium geographic gradient")
        res2 = test_sirmium_gradient(
            dist_mat=dist_mat,
            sample_list=sample_list,
            df_full=df_full,
            i2a_idx=i2a_idx,
            r1a_idx=r1a_idx,
            site_coords=panel_cfg["site_coords"],
            n_perm=args.n_perm,
            n_bootstrap=args.n_bootstrap,
            rng=rng,
        )

        print("Running Test 3: E-V13 replication contrast")
        res3 = test_ev13_replication(
            dist_mat=dist_mat,
            sample_list=sample_list,
            df_full=df_full,
            ev13_high_idx=ev13_high_idx,
            ev13_low_idx=ev13_low_idx,
            r1a_idx=r1a_idx,
            n_bootstrap=args.n_bootstrap,
            rng=rng,
        )

        print("Running Test 4: autosomal homogenization contrast")
        res4 = test_homogenization(
            dist_mat=dist_mat,
            sample_list=sample_list,
            df_full=df_full,
            i2a_idx=i2a_idx,
            r1a_idx=r1a_idx,
            n_perm=args.n_perm,
            n_bootstrap=args.n_bootstrap,
            rng=rng,
        )

        weighted_sum = (
            res1["direction"] * WEIGHTS["LOO"]
            + res2["direction"] * WEIGHTS["Gradient"]
            + res3["direction"] * WEIGHTS["EV13"]
            + res4["direction"] * WEIGHTS["Homogenization"]
        )
        directional_consistency_score = float(weighted_sum / sum(WEIGHTS.values()))

        summary_rows = make_summary_rows(res1, res2, res3, res4)
        print_summary(summary_rows, directional_consistency_score)
        write_summary_csv(summary_rows, directional_consistency_score, paths.output_csv)

        output = {
            "metadata": {
                "script": "03_allele_sharing_panel.py",
                "timestamp_utc": utc_now(),
                "panel": panel_name,
                "matrix_path": relpath(paths.base, paths.matrix_path),
                "matrix_sha256": file_sha256(paths.matrix_path),
                "n_rows": int(n_snps),
                "sample_order": sample_list,
                "permutation_method": "exact_enumeration",
                "n_permutations_requested": int(args.n_perm),
                "n_permutations_usage_note": (
                    "Recorded for reproducibility; implemented label tests use exact finite enumeration."
                ),
                "n_bootstrap": int(args.n_bootstrap),
                "seed": int(args.seed),
                "distance_definition": (
                    "Pairwise IBS distance computed as mismatch proportion across "
                    "strict complete-case pseudo-haploid SNP calls; equivalent to 1 - IBS."
                ),
                "sirmium_coordinates_lat_lon": list(SIRMIUM_COORDS),
            },
            "groups": {
                "i2a_lowland": groups["i2a_lowland"],
                "i2a_highland": groups["i2a_highland"],
                "ev13_highland": groups["ev13_highland"],
                "ev13_lowland": groups["ev13_lowland"],
                "r1a_references": groups["r1a_references"],
            },
            "site_coordinates_lat_lon": {
                sample: list(coords)
                for sample, coords in dict(panel_cfg["site_coords"]).items()
            },
            "weights": WEIGHTS,
            "summary_rows": [asdict(row) for row in summary_rows],
            "test_results": {
                "LOO": {
                    "statistic": res1["statistic"],
                    "p_value": res1["p_value"],
                    "ci_95": res1["ci_95"],
                    "direction": res1["direction"],
                    "weight": WEIGHTS["LOO"],
                    "details": {
                        "loo_shifts": res1["loo_shifts"],
                        "delta_base": res1["delta_base"],
                        "permutation_method": res1["permutation_method"],
                        "n_permutations": res1["n_permutations"],
                        "n_bootstrap_valid": res1["n_bootstrap_valid"],
                    },
                },
                "Gradient": {
                    "statistic": res2["statistic"],
                    "p_value": res2["p_value"],
                    "ci_95": res2["ci_95"],
                    "direction": res2["direction"],
                    "weight": WEIGHTS["Gradient"],
                    "details": {
                        "samples_used": res2["used_samples"],
                        "autosomal_distances_to_r1a": res2["auto_dists"],
                        "geographic_distances_to_sirmium_km": res2["geo_dists"],
                        "permutation_method": res2["permutation_method"],
                        "n_permutations": res2["n_permutations"],
                        "n_bootstrap_valid": res2["n_bootstrap_valid"],
                    },
                },
                "EV13": {
                    "statistic": res3["statistic"],
                    "ci_95": res3["ci_95"],
                    "direction": res3["direction"],
                    "weight": WEIGHTS["EV13"],
                    "details": {
                        "highland_distance_to_r1a": res3.get("highland_dist"),
                        "lowland_distance_to_r1a": res3.get("lowland_dist"),
                        "n_bootstrap_valid": res3["n_bootstrap_valid"],
                    },
                },
                "Homogenization": {
                    "statistic": res4["statistic"],
                    "p_value": res4["p_value"],
                    "ci_95": res4["ci_95"],
                    "direction": res4["direction"],
                    "weight": WEIGHTS["Homogenization"],
                    "details": {
                        "within_i2a_mean": res4["within_i2a_mean"],
                        "within_r1a_mean": res4["within_r1a_mean"],
                        "pooled_within_mean": res4["pooled_within_mean"],
                        "between_mean": res4["between_mean"],
                        "permutation_method": res4["permutation_method"],
                        "n_permutations": res4["n_permutations"],
                        "n_bootstrap_valid": res4["n_bootstrap_valid"],
                    },
                },
            },
            "directional_consistency_score": directional_consistency_score,
            "output_files": {
                "json": relpath(paths.base, paths.output_json),
                "csv": relpath(paths.base, paths.output_csv),
                "pairwise_distance_matrix": relpath(paths.base, paths.distance_matrix_tsv),
            },
        }

        paths.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(paths.output_json, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)

    except Exception as exc:
        print(f"[ERROR] Allele-sharing analysis failed: {exc}")
        return 1

    print("\n" + "=" * 80)
    print("[OK] Allele-sharing analysis complete")
    print(f"Detailed JSON: {paths.output_json}")
    print(f"Summary CSV: {paths.output_csv}")
    print(f"Pairwise distance matrix: {paths.distance_matrix_tsv}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
