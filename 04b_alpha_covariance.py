#!/usr/bin/env python3
"""Propagate qpAdm coefficient uncertainty with sum-to-one covariance."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


import alpha_propagation as alpha


def sum_to_one_covariance(row):
    """Derive the covariance of three constrained coefficients from their SEs."""
    means = np.array([row.p_local, row.p_anatolian, row.p_eastern], dtype=float)
    if not np.isclose(means.sum(), 1.0, rtol=0, atol=1e-6):
        raise ValueError(f"{row.individual}: coefficient estimates must sum to one")

    se = np.array([row.se_local, row.se_anatolian, row.se_eastern], dtype=float)
    if not np.isfinite(se).all() or (se[:2] <= 0).any() or se[2] < 0:
        raise ValueError(f"{row.individual}: invalid coefficient standard errors")

    var_local, var_anatolian, var_eastern = se**2
    cov_local_anatolian = (var_eastern - var_local - var_anatolian) / 2
    rho = cov_local_anatolian / (se[0] * se[1])
    if not -1 - 1e-12 <= rho <= 1 + 1e-12:
        raise ValueError(f"{row.individual}: standard errors are incompatible with a sum-to-one covariance")

    return {
        "individual": row.individual,
        "rho_local_anatolian": float(np.clip(rho, -1, 1)),
        "var_local": var_local,
        "var_anatolian": var_anatolian,
        "var_eastern": var_eastern,
        "cov_local_anatolian": cov_local_anatolian,
        "cov_local_eastern": -var_local - cov_local_anatolian,
        "cov_anatolian_eastern": -var_anatolian - cov_local_anatolian,
    }


def compare_correlations(focal, config):
    covariances, summaries = [], []
    for _, row in focal.iterrows():
        covariance = sum_to_one_covariance(row)
        covariances.append(covariance)
        conditions = [("published_grid", rho) for rho in config.rho_grid]
        conditions.append(("sum_to_one", covariance["rho_local_anatolian"]))
        for setting, rho in conditions:
            result = alpha.simulate_alpha_distribution(row, rho, config)
            identifiers = {"individual": row.individual, "rho_setting": setting, "rho": rho}
            summaries.append({
                **identifiers,
                "published_alpha": row.p_local / (row.p_local + row.p_anatolian),
                "n_retained": result.n_valid_retained,
                "raw_acceptance_fraction": result.valid_fraction_raw,
                "median_alpha": result.median_alpha,
                "lower_95": result.ci_lower_95,
                "upper_95": result.ci_upper_95,
                "interval_width": result.ci_width,
            })

    return {
        "implied_covariance.csv": pd.DataFrame(covariances),
        "alpha_summary.csv": pd.DataFrame(summaries),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path.cwd(), help="Repository root.")
    parser.add_argument("--outdir", type=Path, help="Output directory. Default: results/alpha_covariance.")
    args = parser.parse_args()
    base = args.base.resolve()
    outdir = args.outdir.resolve() if args.outdir else base / "results" / "alpha_covariance"
    focal_path = base / "data" / "olalde_table6_focal.csv"
    focal = alpha.load_focal_table(focal_path)
    config = alpha.SimulationConfig()
    tables = compare_correlations(focal, config)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(outdir / name, index=False, float_format="%.12g", lineterminator="\n")

    sources = (focal_path, Path(__file__).resolve(), Path(alpha.__file__).resolve())
    metadata = {
        "source_sha256": {path.relative_to(base).as_posix(): alpha.sha256_file(path) for path in sources},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "base_seed": config.seed,
        "retained_draws_per_condition": config.n_sim,
        "batch_size": config.batch_size,
        "max_draw_multiplier": config.max_batch_multiplier,
        "compositional_tolerance": config.eps,
        "published_correlation_grid": config.rho_grid,
        "output_sha256": {name: alpha.sha256_file(outdir / name) for name in tables},
    }
    (outdir / "alpha_covariance_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
