#!/usr/bin/env python3
"""
Alpha-threshold sensitivity analysis for the autosomal audit.

This script is a conservative supplementary analysis that combines:

1. Alpha draws from 04_identifiability_analysis.py
   alpha = Croatia_IA / (Croatia_IA + WestAnatolia_Roman_Byzantine)
   within the modeled non-eastern component of the focal I-Y3120 individuals.

2. Directional summaries from 03_allele_sharing_panel.py for the main and
   E-V13 sensitivity panels.

Purpose
-------
This is an autosomal compatibility audit, not a paternal-lineage origin test.
It quantifies whether the coefficient-compatible local-alpha range remains
supported under increasingly strict local-alpha thresholds, including thresholds
anchored to Roman/late-antique comparators and sensitivity across the
Local-Anatolian correlation grid.

The analysis explicitly does not exclude unsampled-source alternatives. It
tests whether the available autosomal evidence rules out structured regional
compatibility; it does not prove continuity, homeland, or the origin of any
Y-chromosome lineage.

Usage
-----
    python 04a_alpha_threshold_sensitivity.py --base .

Pipeline order
--------------
    python 03_allele_sharing_panel.py --panel main --base .
    python 03_allele_sharing_panel.py --panel ev13_sensitivity --base .
    python 04_identifiability_analysis.py --base . --save-draws
    python 04a_alpha_threshold_sensitivity.py --base .
    python 05_make_figures.py --base .

Inputs expected by default
--------------------------
    results/alpha_identifiability/alpha_summary.csv
    results/alpha_identifiability/alpha_draws.csv
    results/alpha_identifiability/comparators_normalized.csv
    matrix_main/allele_sharing_summary.csv
    matrix_ev13_sensitivity/allele_sharing_summary.csv

Outputs
-------
    results/alpha_threshold_sensitivity/alpha_threshold_probabilities.csv
    results/alpha_threshold_sensitivity/comparator_anchored_thresholds.csv
    results/alpha_threshold_sensitivity/directional_support_grid.csv
    results/alpha_threshold_sensitivity/alpha_threshold_scenarios.csv
    results/alpha_threshold_sensitivity/alpha_threshold_worst_rho_summary.csv
    results/alpha_threshold_sensitivity/alpha_threshold_sensitivity_summary.csv
    results/alpha_threshold_sensitivity/figuredata_alpha_threshold_sensitivity.csv
    results/alpha_threshold_sensitivity/alpha_threshold_sensitivity_run_metadata.json
    results/alpha_threshold_sensitivity/alpha_threshold_text_summary.txt

Dependencies
------------
    numpy
    pandas
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_BASE = Path.cwd()
DEFAULT_ALPHA_DIR = Path("results") / "alpha_identifiability"
DEFAULT_OUTDIR = Path("results") / "alpha_threshold_sensitivity"
DEFAULT_THRESHOLDS = [0.20, 0.25, 1.0 / 3.0, 0.40, 0.50]
DEFAULT_PROB_FLOORS = [0.05, 0.10, 0.25]
DEFAULT_PRIMARY_RHO = 0.0
DEFAULT_FOCALS = ["I15541", "I15542"]
DEFAULT_MAIN_PANEL_LABEL = "main_I27297"
DEFAULT_SENS_PANEL_LABEL = "ev13_sensitivity_I28388"


@dataclass
class Paths:
    base: Path
    alpha_summary: Path
    alpha_draws: Path
    comparators: Path
    main_allele_summary: Path
    sensitivity_allele_summary: Path
    outdir: Path
    alpha_threshold_probabilities: Path
    comparator_anchored_thresholds: Path
    directional_support_grid: Path
    alpha_threshold_scenarios: Path
    alpha_threshold_worst_rho_summary: Path
    alpha_threshold_sensitivity_summary: Path
    figuredata: Path
    metadata_json: Path
    text_summary: Path


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


def parse_float_list(value: str) -> List[float]:
    out: List[float] = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated numeric value.")
    return out


def parse_str_list(value: str) -> List[str]:
    out = [part.strip() for part in str(value).split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated sample ID.")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine alpha uncertainty propagation and allele-sharing directionality "
            "into a local-alpha threshold sensitivity analysis."
        )
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="Project base directory. Default: current working directory.")
    parser.add_argument("--alpha-summary", type=Path, default=None, help="Path to alpha_summary.csv. Default: <base>/results/alpha_identifiability/alpha_summary.csv")
    parser.add_argument("--alpha-draws", type=Path, default=None, help="Path to alpha_draws.csv. Default: <base>/results/alpha_identifiability/alpha_draws.csv")
    parser.add_argument("--comparators", type=Path, default=None, help="Path to comparators_normalized.csv. Default: <base>/results/alpha_identifiability/comparators_normalized.csv")
    parser.add_argument("--main-allele-summary", type=Path, default=None, help="Path to main panel allele_sharing_summary.csv. Default: <base>/matrix_main/allele_sharing_summary.csv")
    parser.add_argument("--sensitivity-allele-summary", type=Path, default=None, help="Path to E-V13 sensitivity allele_sharing_summary.csv. Default: <base>/matrix_ev13_sensitivity/allele_sharing_summary.csv")
    parser.add_argument("--outdir", type=Path, default=None, help="Output directory. Default: <base>/results/alpha_threshold_sensitivity")
    parser.add_argument("--thresholds", type=parse_float_list, default=DEFAULT_THRESHOLDS, help="Comma-separated fixed alpha thresholds. Default: 0.20,0.25,0.333333,0.40,0.50")
    parser.add_argument("--prob-floors", type=parse_float_list, default=DEFAULT_PROB_FLOORS, help="Comma-separated probability floors for survival labels. Default: 0.05,0.10,0.25")
    parser.add_argument("--primary-rho", type=float, default=DEFAULT_PRIMARY_RHO, help="Primary rho used for headline summaries. Default: 0.0")
    parser.add_argument("--focal-individuals", type=parse_str_list, default=DEFAULT_FOCALS, help="Comma-separated focal sample IDs. Default: I15541,I15542")
    parser.add_argument("--main-panel-label", type=str, default=DEFAULT_MAIN_PANEL_LABEL, help="Label for main allele-sharing panel.")
    parser.add_argument("--sensitivity-panel-label", type=str, default=DEFAULT_SENS_PANEL_LABEL, help="Label for E-V13 sensitivity panel.")
    parser.add_argument("--no-comparator-thresholds", action="store_true", help="Do not add comparator-anchored alpha thresholds to the tested threshold grid.")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Paths:
    base = args.base.resolve()
    alpha_dir = base / DEFAULT_ALPHA_DIR
    outdir = args.outdir.resolve() if args.outdir else base / DEFAULT_OUTDIR
    return Paths(
        base=base,
        alpha_summary=args.alpha_summary.resolve() if args.alpha_summary else alpha_dir / "alpha_summary.csv",
        alpha_draws=args.alpha_draws.resolve() if args.alpha_draws else alpha_dir / "alpha_draws.csv",
        comparators=args.comparators.resolve() if args.comparators else alpha_dir / "comparators_normalized.csv",
        main_allele_summary=args.main_allele_summary.resolve() if args.main_allele_summary else base / "matrix_main" / "allele_sharing_summary.csv",
        sensitivity_allele_summary=args.sensitivity_allele_summary.resolve() if args.sensitivity_allele_summary else base / "matrix_ev13_sensitivity" / "allele_sharing_summary.csv",
        outdir=outdir,
        alpha_threshold_probabilities=outdir / "alpha_threshold_probabilities.csv",
        comparator_anchored_thresholds=outdir / "comparator_anchored_thresholds.csv",
        directional_support_grid=outdir / "directional_support_grid.csv",
        alpha_threshold_scenarios=outdir / "alpha_threshold_scenarios.csv",
        alpha_threshold_worst_rho_summary=outdir / "alpha_threshold_worst_rho_summary.csv",
        alpha_threshold_sensitivity_summary=outdir / "alpha_threshold_sensitivity_summary.csv",
        figuredata=outdir / "figuredata_alpha_threshold_sensitivity.csv",
        metadata_json=outdir / "alpha_threshold_sensitivity_run_metadata.json",
        text_summary=outdir / "alpha_threshold_text_summary.txt",
    )


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} is empty: {path}")


def validate_columns(df: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required column(s): {', '.join(missing)}")


def load_csv(path: Path, label: str) -> pd.DataFrame:
    require_file(path, label)
    return pd.read_csv(path)


def normalize_alpha_draws(draws: pd.DataFrame) -> pd.DataFrame:
    validate_columns(draws, ["individual", "rho"], "Alpha draws")
    if "alpha" in draws.columns:
        alpha_col = "alpha"
    elif "alpha_draw" in draws.columns:
        alpha_col = "alpha_draw"
    else:
        raise ValueError("Alpha draws must contain either 'alpha' or 'alpha_draw'.")

    out = draws.copy()
    out["individual"] = out["individual"].astype(str)
    out["rho"] = pd.to_numeric(out["rho"], errors="raise")
    out["alpha"] = pd.to_numeric(out[alpha_col], errors="raise")
    out = out[np.isfinite(out["alpha"]) & np.isfinite(out["rho"])].copy()
    out = out[(out["alpha"] >= 0.0) & (out["alpha"] <= 1.0)].copy()
    return out[["individual", "rho", "alpha"]]


def normalize_alpha_summary(summary: pd.DataFrame) -> pd.DataFrame:
    validate_columns(summary, ["individual", "rho"], "Alpha summary")
    out = summary.copy()
    out["individual"] = out["individual"].astype(str)
    out["rho"] = pd.to_numeric(out["rho"], errors="raise")
    for col in ["median_alpha", "ci_lower_95", "ci_upper_95", "valid_fraction"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def normalize_axis(value: object) -> str:
    v = str(value).strip().lower().replace("_", "-")
    if v in {"same-axis", "same axis", "direct"}:
        return "same-axis"
    if v in {"related-axis", "related axis", "contextual"}:
        return "related-axis"
    return v if v else "unspecified"


def normalize_comparators(comparators: pd.DataFrame) -> pd.DataFrame:
    if len(comparators) == 0:
        return comparators.copy()
    out = comparators.copy()
    if "alpha_observed" not in out.columns:
        if {"p_local", "p_anatolian"}.issubset(out.columns):
            local = pd.to_numeric(out["p_local"], errors="coerce")
            anat = pd.to_numeric(out["p_anatolian"], errors="coerce")
            denom = local + anat
            out["alpha_observed"] = np.where(denom > 0, local / denom, np.nan)
        else:
            raise ValueError("Comparator table must contain 'alpha_observed' or both 'p_local' and 'p_anatolian'.")

    out["individual"] = out.get("individual", pd.Series([""] * len(out))).astype(str)
    out["alpha_observed"] = pd.to_numeric(out["alpha_observed"], errors="coerce")

    if "axis_type_normalized" in out.columns:
        axis_col = "axis_type_normalized"
    elif "axis_type" in out.columns:
        axis_col = "axis_type"
    else:
        out["axis_type"] = "unspecified"
        axis_col = "axis_type"
    out["axis_type"] = out[axis_col].map(normalize_axis)
    return out[np.isfinite(out["alpha_observed"])].copy()


def summarize_comparator_floors(comparators: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = {
        "all_comparators": comparators,
        "same_axis": comparators[comparators["axis_type"].eq("same-axis")],
        "related_axis": comparators[comparators["axis_type"].eq("related-axis")],
    }
    for label, df in groups.items():
        vals = df["alpha_observed"].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            rows.append({"comparator_group": label, "n": 0, "min_alpha": np.nan, "median_alpha": np.nan, "max_alpha": np.nan})
        else:
            rows.append({
                "comparator_group": label,
                "n": int(len(vals)),
                "min_alpha": float(np.min(vals)),
                "median_alpha": float(np.median(vals)),
                "max_alpha": float(np.max(vals)),
            })
    return pd.DataFrame(rows)


def make_threshold_catalog(fixed_thresholds: List[float], comparator_floors: pd.DataFrame, include_comparators: bool) -> pd.DataFrame:
    rows = []
    for t in fixed_thresholds:
        rows.append({
            "threshold": float(t),
            "threshold_label": fixed_threshold_label(float(t)),
            "threshold_source": "fixed",
            "comparator_group": "",
            "comparator_statistic": "",
        })
    if include_comparators:
        for _, row in comparator_floors.iterrows():
            group = str(row["comparator_group"])
            if int(row.get("n", 0)) <= 0:
                continue
            for stat in ["min_alpha", "median_alpha"]:
                val = row.get(stat)
                if pd.notna(val) and np.isfinite(float(val)):
                    rows.append({
                        "threshold": float(val),
                        "threshold_label": f"{group}_{stat}",
                        "threshold_source": "comparator_anchored",
                        "comparator_group": group,
                        "comparator_statistic": stat,
                    })
    out = pd.DataFrame(rows)
    out["threshold_key"] = out["threshold"].map(lambda x: f"{x:.6f}")
    out = out.sort_values(["threshold", "threshold_source", "threshold_label"]).reset_index(drop=True)
    return out


def fixed_threshold_label(threshold: float) -> str:
    if np.isclose(threshold, 1 / 3, atol=1e-5):
        return "alpha_ge_one_third"
    return f"alpha_ge_{threshold:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def threshold_probabilities(draws: pd.DataFrame, alpha_summary: pd.DataFrame, threshold_catalog: pd.DataFrame, focals: List[str]) -> pd.DataFrame:
    rows = []
    focal_draws = draws[draws["individual"].isin(focals)].copy()
    if len(focal_draws) == 0:
        raise ValueError(f"No alpha draws found for focal individual(s): {', '.join(focals)}")

    unique_thresholds = threshold_catalog[["threshold", "threshold_key"]].drop_duplicates()
    for (individual, rho), sub in focal_draws.groupby(["individual", "rho"], sort=True):
        vals = sub["alpha"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        for _, th in unique_thresholds.iterrows():
            threshold = float(th["threshold"])
            rows.append({
                "individual": individual,
                "rho": float(rho),
                "threshold": threshold,
                "threshold_key": str(th["threshold_key"]),
                "n_draws": int(len(vals)),
                "prob_alpha_ge_threshold": float(np.mean(vals >= threshold)),
                "median_alpha_from_draws": float(np.median(vals)),
                "ci_lower_95_from_draws": float(np.quantile(vals, 0.025)),
                "ci_upper_95_from_draws": float(np.quantile(vals, 0.975)),
                "mean_alpha_from_draws": float(np.mean(vals)),
            })

    out = pd.DataFrame(rows)
    keep_cols = [c for c in ["individual", "rho", "median_alpha", "ci_lower_95", "ci_upper_95", "valid_fraction"] if c in alpha_summary.columns]
    if {"individual", "rho"}.issubset(keep_cols):
        out = out.merge(alpha_summary[keep_cols].drop_duplicates(["individual", "rho"]), on=["individual", "rho"], how="left")
    return out


def infer_snp_bootstrap_interval_excludes_null(row: pd.Series) -> bool:
    lo = pd.to_numeric(row.get("ci_lower_95", np.nan), errors="coerce")
    hi = pd.to_numeric(row.get("ci_upper_95", np.nan), errors="coerce")
    if not np.isfinite(lo) or not np.isfinite(hi):
        return False
    return bool((lo > 0.0) or (hi < 0.0))


def summarize_allele_panel(path: Path, panel_label: str) -> pd.DataFrame:
    df = load_csv(path, f"Allele-sharing summary ({panel_label})")
    validate_columns(df, ["test_name", "observed_direction_code", "weight", "weighted_contribution"], f"Allele-sharing summary ({panel_label})")

    out = df.copy()
    out["panel"] = panel_label
    out["observed_direction_code"] = pd.to_numeric(out["observed_direction_code"], errors="coerce")
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out["weighted_contribution"] = pd.to_numeric(out["weighted_contribution"], errors="coerce")
    if "statistic" in out.columns:
        out["statistic"] = pd.to_numeric(out["statistic"], errors="coerce")
    if "ci_lower_95" in out.columns:
        out["ci_lower_95"] = pd.to_numeric(out["ci_lower_95"], errors="coerce")
    if "ci_upper_95" in out.columns:
        out["ci_upper_95"] = pd.to_numeric(out["ci_upper_95"], errors="coerce")
    if "p_value" in out.columns:
        out["p_value"] = pd.to_numeric(out["p_value"], errors="coerce")

    out["directionally_consistent"] = out["observed_direction_code"] > 0
    out["snp_bootstrap_interval_excludes_zero"] = out.apply(infer_snp_bootstrap_interval_excludes_null, axis=1)
    out["directionally_consistent_and_snp_bootstrap_interval_excludes_zero"] = (
        out["directionally_consistent"] & out["snp_bootstrap_interval_excludes_zero"]
    )

    total_weight = out["weight"].sum(skipna=True)
    weighted_score = out["weighted_contribution"].sum(skipna=True) / total_weight if total_weight and np.isfinite(total_weight) else np.nan
    out.attrs["panel_summary"] = {
        "panel": panel_label,
        "n_tests": int(len(out)),
        "n_directionally_consistent": int(out["directionally_consistent"].sum()),
        "n_snp_bootstrap_interval_excludes_zero": int(out["snp_bootstrap_interval_excludes_zero"].sum()),
        "n_directionally_consistent_and_snp_bootstrap_interval_excludes_zero": int(out["directionally_consistent_and_snp_bootstrap_interval_excludes_zero"].sum()),
        "fraction_directionally_consistent": float(out["directionally_consistent"].mean()) if len(out) else np.nan,
        "fraction_directionally_consistent_and_snp_bootstrap_interval_excludes_zero": float(out["directionally_consistent_and_snp_bootstrap_interval_excludes_zero"].mean()) if len(out) else np.nan,
        "total_weight": float(total_weight) if np.isfinite(total_weight) else np.nan,
        "directional_consistency_score": float(weighted_score) if np.isfinite(weighted_score) else np.nan,
        "min_weighted_contribution": float(out["weighted_contribution"].min(skipna=True)) if len(out) else np.nan,
    }
    return out


def make_directional_support_grid(main_df: pd.DataFrame, sens_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([dict(main_df.attrs.get("panel_summary", {})), dict(sens_df.attrs.get("panel_summary", {}))])


def scenario_label_for_threshold(threshold: float) -> str:
    if threshold < 0.25:
        return "minimal_local_alpha_support"
    if threshold < 0.34:
        return "moderate_local_alpha_support"
    if threshold < 0.45:
        return "strong_local_alpha_support"
    return "very_strong_local_alpha_support"


def make_threshold_scenarios(alpha_probs: pd.DataFrame, directional_grid: pd.DataFrame, threshold_catalog: pd.DataFrame, prob_floors: List[float], primary_rho: float, focals: List[str]) -> pd.DataFrame:
    main_score = float(directional_grid["directional_consistency_score"].iloc[0]) if len(directional_grid) > 0 else np.nan
    sens_score = float(directional_grid["directional_consistency_score"].iloc[1]) if len(directional_grid) > 1 else np.nan
    min_panel_score = float(np.nanmin([main_score, sens_score])) if np.any(np.isfinite([main_score, sens_score])) else np.nan
    both_panels_positive = bool(np.isfinite(min_panel_score) and min_panel_score > 0)

    catalog_unique = threshold_catalog.drop_duplicates("threshold_key").copy()
    rows = []
    rho_values = sorted(alpha_probs["rho"].dropna().unique())
    for rho in rho_values:
        for _, th in catalog_unique.iterrows():
            threshold = float(th["threshold"])
            threshold_key = str(th["threshold_key"])
            sub = alpha_probs[(np.isclose(alpha_probs["rho"], rho)) & (alpha_probs["threshold_key"].eq(threshold_key))]
            focal_map = {row["individual"]: row["prob_alpha_ge_threshold"] for _, row in sub.iterrows()}
            probs = [float(focal_map.get(ind, np.nan)) for ind in focals]
            finite_probs = [x for x in probs if np.isfinite(x)]
            min_prob = float(np.min(finite_probs)) if finite_probs else np.nan
            mean_prob = float(np.mean(finite_probs)) if finite_probs else np.nan
            max_prob = float(np.max(finite_probs)) if finite_probs else np.nan

            for floor in prob_floors:
                both_focals_supported = bool(len(finite_probs) == len(focals) and min_prob >= floor)
                any_focal_supported = bool(len(finite_probs) > 0 and max_prob >= floor)
                threshold_survives = bool(both_focals_supported and both_panels_positive)
                rows.append({
                    "rho": float(rho),
                    "threshold": threshold,
                    "threshold_key": threshold_key,
                    "threshold_label": str(th["threshold_label"]),
                    "threshold_source": str(th["threshold_source"]),
                    "comparator_group": str(th.get("comparator_group", "")),
                    "comparator_statistic": str(th.get("comparator_statistic", "")),
                    "prob_floor": float(floor),
                    "min_focal_prob_alpha_ge_threshold": min_prob,
                    "mean_focal_prob_alpha_ge_threshold": mean_prob,
                    "max_focal_prob_alpha_ge_threshold": max_prob,
                    "both_focals_alpha_supported": both_focals_supported,
                    "any_focal_alpha_supported": any_focal_supported,
                    "main_directional_score": main_score,
                    "sensitivity_directional_score": sens_score,
                    "min_panel_directional_score": min_panel_score,
                    "both_panels_directionally_positive": both_panels_positive,
                    "threshold_survives": threshold_survives,
                    "scenario_label": scenario_label_for_threshold(threshold),
                    "is_primary_rho": bool(np.isclose(rho, primary_rho)),
                })
    return pd.DataFrame(rows)


def make_worst_rho_summary(scenarios: pd.DataFrame, prob_floors: List[float]) -> pd.DataFrame:
    rows = []
    unique = scenarios.drop_duplicates(["rho", "threshold_key", "prob_floor"]).copy()
    for (threshold_key, floor), sub in unique.groupby(["threshold_key", "prob_floor"], sort=False):
        sub = sub.copy()
        idx_min = sub["min_focal_prob_alpha_ge_threshold"].astype(float).idxmin()
        worst = sub.loc[idx_min]
        rows.append({
            "threshold_key": threshold_key,
            "threshold": float(worst["threshold"]),
            "threshold_label": str(worst["threshold_label"]),
        "threshold_source": str(worst["threshold_source"]),
        "comparator_group": str(worst.get("comparator_group", "")),
        "comparator_statistic": str(worst.get("comparator_statistic", "")),
        "prob_floor": float(floor),
        "worst_rho": float(worst["rho"]),
            "worst_rho_min_focal_prob": float(worst["min_focal_prob_alpha_ge_threshold"]),
            "all_rho_threshold_survives": bool(sub["threshold_survives"].all()),
            "any_rho_threshold_survives": bool(sub["threshold_survives"].any()),
            "n_rho_tested": int(sub["rho"].nunique()),
        })
    return pd.DataFrame(rows).sort_values(["prob_floor", "threshold"]).reset_index(drop=True)


def make_threshold_summary(directional_grid: pd.DataFrame, scenarios: pd.DataFrame, worst_rho: pd.DataFrame, comparator_floors: pd.DataFrame, primary_rho: float, threshold_catalog: pd.DataFrame, prob_floors: List[float], focals: List[str]) -> pd.DataFrame:
    rows = []
    def add(metric: str, value, note: str = "") -> None:
        rows.append({"metric": metric, "value": value, "note": note})

    add("primary_rho_requested", primary_rho, "rho value requested for headline summaries")
    add("available_rho_values", ";".join(f"{x:.6g}" for x in sorted(scenarios["rho"].unique())), "rho values available in alpha draws")
    add("focal_individuals", ";".join(focals), "focal I-Y3120 individuals evaluated")
    add("tested_thresholds", ";".join(sorted(threshold_catalog["threshold_key"].unique())), "fixed and comparator-anchored alpha thresholds evaluated")
    add("probability_floors", ";".join(f"{x:.6g}" for x in prob_floors), "probability floors for weak/moderate/strong survival")

    for _, row in directional_grid.iterrows():
        panel = str(row["panel"])
        add(f"{panel}::directional_consistency_score", float(row["directional_consistency_score"]), "descriptive weighted directional score from 03 allele-sharing contrasts")
        add(f"{panel}::n_directionally_consistent", int(row["n_directionally_consistent"]), "number of contrasts pointing in pre-specified direction")
        add(f"{panel}::n_tests", int(row["n_tests"]), "number of allele-sharing contrasts")
        add(f"{panel}::n_directionally_consistent_and_snp_bootstrap_interval_excludes_zero", int(row["n_directionally_consistent_and_snp_bootstrap_interval_excludes_zero"]), "directionally consistent contrasts with 95% SNP-bootstrap interval excluding zero")

    primary = scenarios[scenarios["is_primary_rho"] & scenarios["threshold_source"].eq("fixed")].copy()
    if len(primary) == 0:
        nearest = float(scenarios.iloc[(scenarios["rho"] - primary_rho).abs().argsort()].iloc[0]["rho"])
        primary = scenarios[np.isclose(scenarios["rho"], nearest) & scenarios["threshold_source"].eq("fixed")].copy()
        add("primary_rho_used", nearest, "nearest available rho used because requested rho was missing")
    else:
        add("primary_rho_used", primary_rho, "rho value used for fixed-threshold headline summaries")

    for floor in prob_floors:
        sub_floor = worst_rho[np.isclose(worst_rho["prob_floor"], floor)].copy()
        survived_all = sub_floor[sub_floor["all_rho_threshold_survives"]]
        max_all = float(survived_all["threshold"].max()) if len(survived_all) else np.nan
        add(f"floor_{floor:.6g}::highest_threshold_supported_all_rho", max_all, "highest threshold supported across every rho value")

        prim_floor = primary[np.isclose(primary["prob_floor"], floor)]
        survived_primary = prim_floor[prim_floor["threshold_survives"]]
        max_primary = float(survived_primary["threshold"].max()) if len(survived_primary) else np.nan
        add(f"floor_{floor:.6g}::highest_threshold_supported_primary_rho", max_primary, "highest fixed threshold supported at primary rho")

    for _, row in comparator_floors.iterrows():
        group = str(row["comparator_group"])
        add(f"comparator_{group}::n", int(row["n"]), "number of comparator individuals")
        add(f"comparator_{group}::min_alpha", row["min_alpha"], "minimum comparator alpha")
        add(f"comparator_{group}::median_alpha", row["median_alpha"], "median comparator alpha")

    add("scope_statement", "local-alpha threshold sensitivity only", "does not infer paternal-lineage origin, continuity, or homeland")
    add("unsampled_source_statement", "not resolved", "an unsampled Balkan-like eastern/northern source cannot be excluded with these inputs alone")
    return pd.DataFrame(rows)


def make_figuredata(scenarios: pd.DataFrame, worst_rho: pd.DataFrame) -> pd.DataFrame:
    base = scenarios.copy()
    base["threshold_percent"] = base["threshold"] * 100.0
    base["min_focal_prob_percent"] = base["min_focal_prob_alpha_ge_threshold"] * 100.0
    base["mean_focal_prob_percent"] = base["mean_focal_prob_alpha_ge_threshold"] * 100.0
    base["curve_type"] = np.where(base["is_primary_rho"], "primary_rho", "other_rho")
    keep = [
        "rho", "threshold", "threshold_key", "threshold_percent", "threshold_label", "threshold_source",
        "comparator_group", "comparator_statistic", "prob_floor", "scenario_label", "is_primary_rho", "curve_type",
        "min_focal_prob_alpha_ge_threshold", "mean_focal_prob_alpha_ge_threshold", "max_focal_prob_alpha_ge_threshold",
        "min_focal_prob_percent", "mean_focal_prob_percent", "both_focals_alpha_supported",
        "both_panels_directionally_positive", "threshold_survives", "main_directional_score", "sensitivity_directional_score",
    ]
    return base[keep]


def write_text_summary(path: Path, scenarios: pd.DataFrame, worst_rho: pd.DataFrame, directional_grid: pd.DataFrame, primary_rho: float, prob_floors: List[float]) -> None:
    main_score = float(directional_grid["directional_consistency_score"].iloc[0]) if len(directional_grid) else np.nan
    sens_score = float(directional_grid["directional_consistency_score"].iloc[1]) if len(directional_grid) > 1 else np.nan

    lines = [
        "Alpha-threshold sensitivity analysis",
        "============================================================",
        f"Primary rho value: {primary_rho}",
        f"Probability floors evaluated: {', '.join(f'{x:.3f}' for x in prob_floors)}",
        f"Main-panel descriptive directional score: {main_score:.3f}" if np.isfinite(main_score) else "Main-panel descriptive directional score: NA",
        f"Sensitivity-panel descriptive directional score: {sens_score:.3f}" if np.isfinite(sens_score) else "Sensitivity-panel descriptive directional score: NA",
        "",
        "Highest alpha thresholds supported across all rho assumptions:",
    ]
    for floor in prob_floors:
        sub = worst_rho[np.isclose(worst_rho["prob_floor"], floor) & worst_rho["all_rho_threshold_survives"]]
        max_thr = float(sub["threshold"].max()) if len(sub) else np.nan
        if np.isfinite(max_thr):
            lines.append(f"  - floor {floor:.3f}: alpha >= {max_thr:.3f}")
        else:
            lines.append(f"  - floor {floor:.3f}: no threshold was supported across all rho values")

    lines.extend([
        "",
        "Summary:",
        "This local-alpha threshold sensitivity analysis does not infer paternal-lineage origin. It evaluates whether the coefficient-compatible alpha range remains compatible with non-trivial local-alpha thresholds, including comparator-anchored thresholds and sensitivity across the Local-Anatolian correlation grid. The combined alpha and allele-sharing diagnostics remain a compatibility test for structured regional baselines under incomplete proxy sampling, not a proof of continuity or homeland.",
        "",
        "Scope:",
        "This analysis does not exclude unsampled-source alternatives. Excluding a Balkan-like eastern/northern source would require additional source-side ancient DNA or Y-stratified modern autosomal/IBD data.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata(path: Path, args: argparse.Namespace, paths: Paths, comparator_floors: pd.DataFrame, threshold_catalog: pd.DataFrame) -> None:
    input_paths = {
        "alpha_summary": paths.alpha_summary,
        "alpha_draws": paths.alpha_draws,
        "comparators": paths.comparators,
        "main_allele_summary": paths.main_allele_summary,
        "sensitivity_allele_summary": paths.sensitivity_allele_summary,
    }
    metadata = {
        "script": Path(__file__).name,
        "timestamp_utc": utc_now(),
        "base": ".",
        "settings": {
            "fixed_thresholds": args.thresholds,
            "prob_floors": args.prob_floors,
            "primary_rho": args.primary_rho,
            "focal_individuals": args.focal_individuals,
            "main_panel_label": args.main_panel_label,
            "sensitivity_panel_label": args.sensitivity_panel_label,
            "include_comparator_thresholds": not args.no_comparator_thresholds,
        },
        "inputs": {key: {"path": relpath(paths.base, value), "sha256": sha256_file(value)} for key, value in input_paths.items()},
        "outputs": {field: relpath(paths.base, getattr(paths, field)) for field in [
            "alpha_threshold_probabilities", "comparator_anchored_thresholds", "directional_support_grid",
            "alpha_threshold_scenarios", "alpha_threshold_worst_rho_summary", "alpha_threshold_sensitivity_summary", "figuredata", "text_summary",
        ]},
        "comparator_floors": comparator_floors.to_dict(orient="records"),
        "threshold_catalog": threshold_catalog.to_dict(orient="records"),
        "scope_statement": (
            "This analysis is a local-alpha threshold sensitivity audit for coefficient-layer under-resolution. "
            "It does not infer paternal-lineage origin, direct population continuity, or historical identity."
        ),
        "limitation_statement": (
            "The analysis cannot exclude an unsampled Balkan-like eastern/northern source. That requires additional "
            "source-side ancient DNA or Y-stratified modern autosomal/IBD data."
        ),
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    paths.outdir.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("04a ALPHA-THRESHOLD SENSITIVITY ANALYSIS")
    print("============================================================")
    print(f"Base: {paths.base}")
    print(f"Outdir: {paths.outdir}")
    print(f"Focal individuals: {', '.join(args.focal_individuals)}")
    print(f"Fixed thresholds: {', '.join(f'{x:.6g}' for x in args.thresholds)}")
    print(f"Probability floors: {', '.join(f'{x:.6g}' for x in args.prob_floors)}")

    alpha_summary = normalize_alpha_summary(load_csv(paths.alpha_summary, "Alpha summary"))
    alpha_draws = normalize_alpha_draws(load_csv(paths.alpha_draws, "Alpha draws"))
    comparators = normalize_comparators(load_csv(paths.comparators, "Comparators"))

    comparator_floors = summarize_comparator_floors(comparators)
    fixed_thresholds = sorted(set(float(x) for x in args.thresholds))
    prob_floors = sorted(set(float(x) for x in args.prob_floors))
    threshold_catalog = make_threshold_catalog(fixed_thresholds, comparator_floors, include_comparators=not args.no_comparator_thresholds)

    alpha_probs = threshold_probabilities(alpha_draws, alpha_summary, threshold_catalog, args.focal_individuals)
    main_panel = summarize_allele_panel(paths.main_allele_summary, args.main_panel_label)
    sensitivity_panel = summarize_allele_panel(paths.sensitivity_allele_summary, args.sensitivity_panel_label)
    directional_grid = make_directional_support_grid(main_panel, sensitivity_panel)

    scenarios = make_threshold_scenarios(
        alpha_probs=alpha_probs,
        directional_grid=directional_grid,
        threshold_catalog=threshold_catalog,
        prob_floors=prob_floors,
        primary_rho=args.primary_rho,
        focals=args.focal_individuals,
    )
    worst_rho = make_worst_rho_summary(scenarios, prob_floors)
    summary = make_threshold_summary(
        directional_grid=directional_grid,
        scenarios=scenarios,
        worst_rho=worst_rho,
        comparator_floors=comparator_floors,
        primary_rho=args.primary_rho,
        threshold_catalog=threshold_catalog,
        prob_floors=prob_floors,
        focals=args.focal_individuals,
    )
    figuredata = make_figuredata(scenarios, worst_rho)

    alpha_probs.to_csv(paths.alpha_threshold_probabilities, index=False)
    threshold_catalog.to_csv(paths.comparator_anchored_thresholds, index=False)
    directional_grid.to_csv(paths.directional_support_grid, index=False)
    scenarios.to_csv(paths.alpha_threshold_scenarios, index=False)
    worst_rho.to_csv(paths.alpha_threshold_worst_rho_summary, index=False)
    summary.to_csv(paths.alpha_threshold_sensitivity_summary, index=False)
    figuredata.to_csv(paths.figuredata, index=False)
    write_text_summary(paths.text_summary, scenarios, worst_rho, directional_grid, args.primary_rho, prob_floors)
    write_metadata(paths.metadata_json, args, paths, comparator_floors, threshold_catalog)

    print("\nDirectional support:")
    print(directional_grid.to_string(index=False))

    print("\nComparator-anchored thresholds:")
    print(threshold_catalog.to_string(index=False))

    primary = scenarios[scenarios["is_primary_rho"] & scenarios["threshold_source"].eq("fixed") & np.isclose(scenarios["prob_floor"], min(prob_floors))]
    if len(primary) > 0:
        print("\nPrimary-rho fixed-threshold scenarios at weakest probability floor:")
        cols = ["rho", "threshold", "prob_floor", "min_focal_prob_alpha_ge_threshold", "both_focals_alpha_supported", "both_panels_directionally_positive", "threshold_survives"]
        print(primary[cols].to_string(index=False))

    print("\nWorst-rho survival summary:")
    print(worst_rho[worst_rho["threshold_source"].eq("fixed")].to_string(index=False))

    print("\nOutputs written:")
    for p in [
        paths.alpha_threshold_probabilities,
        paths.comparator_anchored_thresholds,
        paths.directional_support_grid,
        paths.alpha_threshold_scenarios,
        paths.alpha_threshold_worst_rho_summary,
        paths.alpha_threshold_sensitivity_summary,
        paths.figuredata,
        paths.metadata_json,
        paths.text_summary,
    ]:
        print(f"  - {p}")
    print("============================================================")
    print("04a completed")
    print("============================================================")


if __name__ == "__main__":
    main()
