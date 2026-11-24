#!/usr/bin/env python3
"""Generate the five supplementary tables from source extracts and allele calls."""

import argparse
import json
import platform
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

import alpha_propagation as alpha
import panel_distances as distance


covariance = import_module("04b_alpha_covariance")
block = import_module("07_block_bootstrap")

CONDITIONS = (
    ("original", "", "results/block_bootstrap"),
    ("filtered", "results/quality_control/filtered", "results/quality_control/filtered_block_bootstrap"),
    ("transversions", "results/quality_control/transversions", "results/quality_control/transversions_block_bootstrap"),
)
CONTRASTS = {
    "loo_statistic": "leave_one_out_shift",
    "sirmium_gradient_r": "sirmium_correlation",
    "ev13_difference": "ev13_comparator_difference",
    "homogenization_separation": "between_minus_within",
    block.ALL_FOUR: "joint_sign_fraction",
}
DIRECTION_RULES = {
    "loo_statistic": "< 0",
    "sirmium_gradient_r": "> 0",
    "ev13_difference": "> 0",
    "homogenization_separation": "<= 0",
    block.ALL_FOUR: "all four rules",
}


def kuline_context(models):
    same_model = models["first_proxy"].eq("Croatia_IA") & models["third_proxy"].eq("CEE_EarlyMedieval")
    config = alpha.SimulationConfig()
    values = []
    for _, row in models.loc[same_model].iterrows():
        rho = covariance.sum_to_one_covariance(row)["rho_local_anatolian"]
        sim = alpha.simulate_alpha_distribution(row, rho, config)
        values.append({
            "individual": row.individual,
            "reporting_role": "focal" if row.individual in {"I15541", "I15542"} else "same_site",
            "published_ratio": row.p_local / (row.p_local + row.p_anatolian),
            "implied_rho": rho, "raw_acceptance_fraction": sim.valid_fraction_raw,
            "n_retained": sim.n_valid_retained, "median_share": sim.median_alpha,
            "lower_95": sim.ci_lower_95, "upper_95": sim.ci_upper_95,
        })
    return models.merge(pd.DataFrame(values), how="left", on="individual", validate="one_to_one")


def two_source_context(models):
    keep = models["p_local"].between(0, 1) & models["p_anatolian"].between(0, 1) & models["p_value"].ge(0.05)
    eligible = models.loc[keep].copy()
    if not np.allclose(eligible["p_local"] + eligible["p_anatolian"], 1, atol=1e-6, rtol=0):
        raise ValueError("The reported two-source coefficients do not sum to one.")
    return eligible


def contrast_table(base):
    summaries = []
    for condition, _, summary_path in CONDITIONS:
        summary = pd.read_csv(base / summary_path / "contrast_summary.csv")
        if len(summary) != 20 or set(summary["metric"]) != CONTRASTS.keys():
            raise ValueError(f"Incomplete contrast summary for {condition}")
        summary.insert(0, "condition", condition)
        summary["direction_rule"] = summary["metric"].map(DIRECTION_RULES)
        summary["metric"] = summary["metric"].map(CONTRASTS)
        summary["n_preserved"] = summary.pop("n_preserved")
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def sirmium_influence(base):
    rows = []
    for condition, matrix_root, _ in CONDITIONS:
        for panel_key in distance.PANELS:
            panel = distance.load_panel_data(base / matrix_root, panel_key)
            pairwise = distance.compute_distance_matrix(panel.allele_array)
            i2a = distance.idx_list(panel, panel.groups["all_i2a"])
            refs = distance.idx_list(panel, panel.groups["r1a_references"])
            full, ids, km, mismatch = distance.compute_sirmium_gradient(pairwise, panel, i2a, refs)
            people = pd.DataFrame({"individual": ids, "km": km, "mismatch": mismatch})
            without_gomolava = people.loc[people["individual"].ne("I1116")]
            sites = people.groupby("km", as_index=False)["mismatch"].mean()
            earlier_refs = [i for i in refs if panel.samples[i] != "I35012"]
            omit_later, *_ = distance.compute_sirmium_gradient(pairwise, panel, i2a, earlier_refs)
            rows.append({
                "condition": condition, "panel": panel_key, "n_sites": panel.n_snps,
                "n_individuals": len(people), "n_geographic_sites": len(sites), "full_r": full,
                "omit_i1116_r": without_gomolava["km"].corr(without_gomolava["mismatch"]),
                "site_mean_r": sites["km"].corr(sites["mismatch"]), "omit_i35012_r": omit_later,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path.cwd(), help="Repository root.")
    args = parser.parse_args()
    base = args.base.resolve()
    tables = base / "results" / "supplementary_tables"
    tables.mkdir(parents=True, exist_ok=True)
    panel = base / "data/panel_sample_provenance.csv"
    kuline = base / "data/olalde_kuline_models.csv"
    fits = base / "data/olalde_two_source_fits.csv"
    outputs = {
        "table_s1_panel.csv": pd.read_csv(panel),
        "table_s2_kuline.csv": kuline_context(pd.read_csv(kuline)),
        "table_s3_two_source_fits.csv": two_source_context(pd.read_csv(fits)),
        "table_s4_block_results.csv": contrast_table(base),
        "table_s5_sirmium_influence.csv": sirmium_influence(base),
    }
    for name, frame in outputs.items():
        frame.to_csv(tables / name, index=False, float_format="%.12g", lineterminator="\n")
    inputs = [panel, kuline, fits, Path(__file__).resolve(),
              Path(covariance.__file__).resolve(), Path(distance.__file__).resolve(),
              Path(alpha.__file__).resolve(), Path(block.__file__).resolve()]
    for _, matrix_root, summary_path in CONDITIONS:
        inputs.append(base / summary_path / "contrast_summary.csv")
        inputs.extend(base / matrix_root / str(cfg["matrix_dir"]) / "complete_case_9_sample_matrix.tsv"
                      for cfg in distance.PANELS.values())
    metadata = {
        "source_sha256": {p.relative_to(base).as_posix(): distance.sha256_file(p) for p in inputs},
        "direction_rules": {CONTRASTS[name]: rule for name, rule in DIRECTION_RULES.items()},
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__},
        "output_sha256": {name: distance.sha256_file(tables / name) for name in outputs},
    }
    (tables / "supplementary_tables_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
