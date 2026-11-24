#!/usr/bin/env python3
"""Resample chromosome-specific genetic blocks for the four distance contrasts."""

import argparse
import json
import platform
from decimal import Decimal
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy


import panel_distances as distance
BLOCK_SIZES_CM = (5, 10)
METRICS = (
    ("loo_statistic", "loo_direction_ok"),
    ("sirmium_gradient_r", "sirmium_direction_ok"),
    ("ev13_difference", "ev13_direction_ok"),
    ("homogenization_separation", "homogenization_direction_ok"),
)
ALL_FOUR = "all_four_directions_preserved"


def panel_keys(panel):
    return [(int(chrom), int(pos)) for chrom, pos in panel.df[["chrom", "pos"]].itertuples(index=False, name=None)]


def load_genetic_positions(path, required):
    genetic_cm = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            _, chrom, morgans, pos, *_ = line.split()
            if not chrom.isdecimal():
                continue
            key = (int(chrom), int(pos))
            if key not in required:
                continue
            position = Decimal(morgans) * 100
            if not position.is_finite() or position < 0:
                raise ValueError(f"Invalid genetic position at chromosome {chrom}, position {pos}.")
            if key in genetic_cm and genetic_cm[key] != position:
                raise ValueError(f"Conflicting genetic positions at chromosome {chrom}, position {pos}.")
            genetic_cm[key] = position
    missing = required - genetic_cm.keys()
    if missing:
        raise ValueError(f"The genetic map is missing {len(missing)} matrix positions.")
    return genetic_cm


def block_mismatch_counts(panel, keys, genetic_cm, block_cm):
    blocks = np.array([(chrom, int(genetic_cm[chrom, pos] // block_cm)) for chrom, pos in keys])
    labels, block_index, site_counts = np.unique(blocks, axis=0, return_inverse=True, return_counts=True)
    pairs = np.array(list(combinations(range(len(panel.samples)), 2)))
    left, right = pairs.T
    allele = panel.allele_array
    mismatches = allele[:, left] != allele[:, right]
    mismatch_counts = np.zeros((len(labels), len(pairs)), dtype=np.int64)
    np.add.at(mismatch_counts, block_index, mismatches)
    return labels, site_counts, pairs, mismatch_counts


def sample_blocks(labels, site_counts, mismatch_counts, n_bootstrap, rng):
    sites = np.zeros(n_bootstrap, dtype=np.int64)
    mismatches = np.zeros((n_bootstrap, mismatch_counts.shape[1]), dtype=np.int64)
    for chromosome in np.unique(labels[:, 0]):
        mask = labels[:, 0] == chromosome
        n_blocks = int(mask.sum())
        multiplicities = rng.multinomial(n_blocks, np.full(n_blocks, 1 / n_blocks), size=n_bootstrap)
        sites += multiplicities @ site_counts[mask]
        mismatches += multiplicities @ mismatch_counts[mask]
    return sites, mismatches


def evaluate_contrasts(panel, counts, n_sites):
    groups = panel.groups
    non_kuline = distance.idx_list(panel, groups["i2a_non_kuline"])
    kuline = distance.idx_list(panel, groups["i2a_kuline"])
    i2a = non_kuline + kuline
    refs = distance.idx_list(panel, groups["r1a_references"])
    ev13_kuline, = distance.idx_list(panel, groups["ev13_kuline"])
    ev13_comparator, = distance.idx_list(panel, groups["ev13_non_kuline"])

    non_kuline_counts = counts[np.ix_(non_kuline, refs)]
    kuline_counts = counts[np.ix_(kuline, refs)]
    # An omission shifts the mean by (panel mean - omitted mean) / (n - 1).
    loo_numerator = (
        int(non_kuline_counts.sum()) * kuline_counts.size - int(kuline_counts.sum()) * non_kuline_counts.size
    )
    loo_denominator = non_kuline_counts.size * kuline_counts.size * (len(i2a) - 1) * n_sites
    ev13_numerator = int(counts[ev13_kuline, refs].sum()) - int(counts[ev13_comparator, refs].sum())
    between = counts[np.ix_(i2a, refs)]
    within_pairs = list(combinations(i2a, 2)) + list(combinations(refs, 2))
    left, right = zip(*within_pairs)
    within = counts[left, right]
    separation_numerator = int(between.sum()) * within.size - int(within.sum()) * between.size
    r, *_ = distance.compute_sirmium_gradient(counts / n_sites, panel, i2a, refs)
    values = (
        loo_numerator / loo_denominator,
        r,
        ev13_numerator / (len(refs) * n_sites),
        separation_numerator / (between.size * within.size * n_sites),
    )
    directions = (loo_numerator < 0, r > 0, ev13_numerator > 0, separation_numerator <= 0)
    row = {"panel": panel.panel, ALL_FOUR: all(directions)}
    for (metric, flag), value, direction in zip(METRICS, values, directions):
        row[metric] = value
        row[flag] = direction
    return row


def block_bootstrap(panel, keys, genetic_cm, block_cm, n_bootstrap, seed):
    labels, counts, pairs, mismatches = block_mismatch_counts(panel, keys, genetic_cm, block_cm)
    rng = np.random.default_rng(distance.deterministic_seed(seed, f"{panel.panel}_{block_cm}cM"))
    sites, sampled_mismatches = sample_blocks(labels, counts, mismatches, n_bootstrap, rng)
    left, right = pairs.T
    matrix = np.zeros((len(panel.samples), len(panel.samples)), dtype=np.int64)
    rows = []
    for replicate, (n_sites, counts) in enumerate(zip(sites, sampled_mismatches), 1):
        matrix[left, right] = matrix[right, left] = counts
        metrics = evaluate_contrasts(panel, matrix, int(n_sites))
        row = {"panel": panel.panel, "block_size_cm": block_cm, "replicate": replicate, "n_sites": n_sites}
        for value, direction in METRICS:
            row[value] = metrics[value]
            row[direction] = metrics[direction]
        row[ALL_FOUR] = metrics[ALL_FOUR]
        rows.append(row)
    return pd.DataFrame(rows), len(labels)


def summarize(replicates, observed, n_blocks, n_sites):
    common = {
        "panel": observed["panel"],
        "block_size_cm": int(replicates["block_size_cm"].iloc[0]),
        "n_nonempty_blocks": n_blocks,
        "n_panel_sites": n_sites,
        "n_replicates": len(replicates),
    }
    rows = []
    for metric, direction in METRICS:
        values = replicates[metric].dropna()
        n_preserved = int(replicates[direction].sum())
        rows.append({
            **common,
            "metric": metric,
            "observed": observed[metric],
            "observed_direction_preserved": observed[direction],
            "n_finite_replicates": len(values),
            "lower_95": values.quantile(0.025),
            "upper_95": values.quantile(0.975),
            "direction_preservation_fraction": n_preserved / len(replicates),
            "n_preserved": n_preserved,
        })
    n_preserved = int(replicates[ALL_FOUR].sum())
    rows.append({
        **common,
        "metric": ALL_FOUR,
        "observed_direction_preserved": observed[ALL_FOUR],
        "n_finite_replicates": int(replicates[[metric for metric, _ in METRICS]].notna().all(axis=1).sum()),
        "direction_preservation_fraction": n_preserved / len(replicates),
        "n_preserved": n_preserved,
    })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path.cwd(), help="Repository root.")
    parser.add_argument("--matrix-root", type=Path, help="Directory containing the two panel matrix directories.")
    parser.add_argument("--outdir", type=Path, help="Output directory. Default: results/block_bootstrap.")
    parser.add_argument("--replicates", type=int, default=5000, help="Replicates per panel and block size.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    args = parser.parse_args()
    if args.replicates <= 0:
        parser.error("--replicates must be positive")
    if args.matrix_root is not None and args.outdir is None:
        parser.error("--outdir is required with --matrix-root")

    base = args.base.resolve()
    matrix_root = args.matrix_root.resolve() if args.matrix_root else base
    outdir = args.outdir.resolve() if args.outdir else base / "results" / "block_bootstrap"
    map_path = base / "v62.0_1240k_public.snp"
    panels = [distance.load_panel_data(matrix_root, name) for name in distance.PANELS]
    keys = {panel.panel: panel_keys(panel) for panel in panels}
    required = set().union(*keys.values())
    if any(len(set(rows)) != len(rows) for rows in keys.values()):
        raise ValueError("Panel matrices contain duplicate chromosome and position pairs.")
    genetic_cm = load_genetic_positions(map_path, required)
    parts, summaries = [], []
    for panel in panels:
        observed = evaluate_contrasts(panel, distance.mismatch_counts(panel.allele_array), panel.n_snps)
        for block_cm in BLOCK_SIZES_CM:
            print(f"{panel.panel}: {block_cm} cM blocks, {args.replicates} replicates", flush=True)
            replicates, n_blocks = block_bootstrap(panel, keys[panel.panel], genetic_cm, block_cm, args.replicates, args.seed)
            parts.append(replicates)
            summaries.extend(summarize(replicates, observed, n_blocks, panel.n_snps))

    outdir.mkdir(parents=True, exist_ok=True)
    tables = {"contrast_summary.csv": pd.DataFrame(summaries), "replicates.csv": pd.concat(parts, ignore_index=True)}
    for name, frame in tables.items():
        frame.to_csv(outdir / name, index=False, float_format="%.12g", lineterminator="\n")
    sources = [map_path, Path(__file__).resolve(), Path(distance.__file__).resolve()]
    sources.extend(panel.matrix_path for panel in panels)
    metadata = {
        "source_sha256": {distance.relpath(base, path): distance.sha256_file(path) for path in sources},
        "software": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
        },
        "base_seed": args.seed,
        "replicates_per_condition": args.replicates,
        "block_sizes_cm": BLOCK_SIZES_CM,
        "resampling": "Sample nonempty blocks with replacement within chromosomes, retaining each chromosome's block count.",
        "output_sha256": {name: distance.sha256_file(outdir / name) for name in tables},
    }
    (outdir / "block_bootstrap_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
