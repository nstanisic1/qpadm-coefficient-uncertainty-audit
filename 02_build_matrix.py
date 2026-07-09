#!/usr/bin/env python3
"""
Build strict complete-case pseudo-haploid autosomal matrices for the
reproducible allele-sharing analysis.

This script constructs the complete-case allele matrix used by the independent
allele-sharing analysis. It proceeds in two stages:

1. Build a seven-sample core matrix from four I2a samples and three R1a
   reference samples by scanning all autosomal positions from the public 1240k
   SNP list by default.
2. Add the two E-V13 samples at exactly the SNP positions retained in the core
   matrix, then remove any row containing a missing call in any of the nine
   panel samples.

The final output is a strict complete-case pseudo-haploid matrix in which every
retained SNP has a non-missing consensus allele call for all nine samples.

Default usage
-------------
Main panel:
    python 02_build_matrix.py --panel main --base .

E-V13 sensitivity panel:
    python 02_build_matrix.py --panel ev13_sensitivity --base .

Force a clean rebuild:
    python 02_build_matrix.py --panel main --base . --force

Inputs
------
By default, this script expects:

    ena_1240k_all/
        <sample>.1240k.bam
        <sample>.1240k.bam.bai

    v62.0_1240k_public.snp

The BAM/BAI directory is the expected output of:

    python 01_download_bams.py --panel all --base .

Outputs
-------
For the main panel:

    matrix_main/core_7_sample_matrix.tsv
    matrix_main/complete_case_9_sample_matrix.tsv
    matrix_main/matrix_build_summary.json

For the E-V13 sensitivity panel:

    matrix_ev13_sensitivity/core_7_sample_matrix.tsv
    matrix_ev13_sensitivity/complete_case_9_sample_matrix.tsv
    matrix_ev13_sensitivity/matrix_build_summary.json

Dependencies
------------
    pandas
    bamnostic
"""

from __future__ import annotations

import argparse
import json
import os
from bisect import bisect_left
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import bamnostic as bs
import pandas as pd


PANELS: Dict[str, Dict[str, List[str]]] = {
    "main": {
        "core_samples": [
            "I1116",
            "I32299",
            "I15541",
            "I15542",
            "I26766",
            "I26748",
            "I35012",
        ],
        "extra_samples": [
            "I15537",
            "I27297",
        ],
    },
    "ev13_sensitivity": {
        "core_samples": [
            "I1116",
            "I32299",
            "I15541",
            "I15542",
            "I26766",
            "I26748",
            "I35012",
        ],
        "extra_samples": [
            "I15537",
            "I28388",
        ],
    },
}


BASE_ORDER = ("A", "C", "G", "T")
DEFAULT_BASE = Path.cwd()
DEFAULT_BAM_DIR = "ena_1240k_all"
DEFAULT_SNP_FILE = "v62.0_1240k_public.snp"
DEFAULT_BUDGET = "all"
DEFAULT_CHECKPOINT_INTERVAL = 50_000
DEFAULT_WORKERS = min(3, os.cpu_count() or 1)


@dataclass
class InputFileRecord:
    sample: str
    bam_path: str
    bai_path: str
    bam_size_bytes: int
    bai_size_bytes: int
    status: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a strict complete-case pseudo-haploid autosomal matrix for "
            "a selected allele-sharing analysis panel."
        )
    )
    parser.add_argument(
        "--panel",
        choices=sorted(PANELS.keys()),
        default="main",
        help="Sample panel to build. Default: main.",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
        help="Project base directory. Default: current working directory.",
    )
    parser.add_argument(
        "--bam-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing BAM/BAI files. If omitted, "
            f"'<base>/{DEFAULT_BAM_DIR}' is used."
        ),
    )
    parser.add_argument(
        "--snp-path",
        type=Path,
        default=None,
        help=(
            f"Path to the 1240k SNP file. If omitted, "
            f"'<base>/{DEFAULT_SNP_FILE}' is used."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Directory for panel-specific outputs. If omitted, "
            "'<base>/matrix_<panel>' is used."
        ),
    )
    parser.add_argument(
        "--budget",
        default=DEFAULT_BUDGET,
        help=(
            "Maximum number of autosomal SNP positions to scan during the initial core build. "
            "Use 'all' to scan all autosomal 1240k positions. Default: all."
        ),
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help=(
            "Number of scanned SNP positions between progress updates. "
            f"Default: {DEFAULT_CHECKPOINT_INTERVAL}."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel BAM workers. Default: {DEFAULT_WORKERS}.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing matrix outputs and checkpoints before rebuilding.",
    )
    return parser.parse_args()


def normalize_budget(raw: object) -> Optional[int]:
    value = str(raw).strip().lower()
    if value in {"all", "none", "full"}:
        return None
    out = int(value)
    if out <= 0:
        raise ValueError("--budget must be a positive integer or 'all'.")
    return out


def budget_label(max_snps: Optional[int]) -> object:
    return "all" if max_snps is None else int(max_snps)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def relpath(base: Path, path: Path | str | None) -> str | None:
    if path is None:
        return None
    try:
        return Path(path).resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path)


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    base = args.base.resolve()
    panel = args.panel

    bam_dir = args.bam_dir.resolve() if args.bam_dir else base / DEFAULT_BAM_DIR
    snp_path = args.snp_path.resolve() if args.snp_path else base / DEFAULT_SNP_FILE
    results_dir = args.results_dir.resolve() if args.results_dir else base / f"matrix_{panel}"

    return bam_dir, snp_path, results_dir


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def retained_autosomes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        chroms = pd.read_csv(path, sep="\t", usecols=["chrom"], dtype=str)["chrom"]
    except (FileNotFoundError, ValueError):
        return set()
    return set(chroms.astype(str).str.replace("chr", "", regex=False))


def clean_existing_outputs(results_dir: Path, preserve_core: bool = False) -> None:
    """
    Remove files produced by this script for a clean rebuild.
    """
    core_path = results_dir / "core_7_sample_matrix.tsv"
    expected_autosomes = {str(i) for i in range(1, 23)}
    for filename in [
        "core_7_sample_matrix.tsv",
        "core_7_sample_checkpoint.json",
        "complete_case_9_sample_matrix.tsv",
        "matrix_build_summary.json",
    ]:
        path = results_dir / filename
        if preserve_core and path == core_path and retained_autosomes(path) == expected_autosomes:
            continue
        remove_if_exists(path)


def validate_input_files(
    samples: List[str],
    base: Path,
    bam_dir: Path,
    snp_path: Path,
) -> List[InputFileRecord]:
    """
    Validate required BAM, BAI, and SNP inputs before starting the build.
    """
    records: List[InputFileRecord] = []

    if not snp_path.exists():
        raise FileNotFoundError(f"SNP file not found: {snp_path}")

    if snp_path.stat().st_size == 0:
        raise ValueError(f"SNP file is empty: {snp_path}")

    if not bam_dir.exists():
        raise FileNotFoundError(f"BAM directory not found: {bam_dir}")

    for sample in samples:
        bam_path = bam_dir / f"{sample}.1240k.bam"
        bai_path = bam_dir / f"{sample}.1240k.bam.bai"

        if not bam_path.exists():
            records.append(
                InputFileRecord(
                    sample=sample,
                    bam_path=relpath(base, bam_path),
                    bai_path=relpath(base, bai_path),
                    bam_size_bytes=0,
                    bai_size_bytes=0,
                    status="failed",
                    message="missing BAM file",
                )
            )
            continue

        if not bai_path.exists():
            records.append(
                InputFileRecord(
                    sample=sample,
                    bam_path=relpath(base, bam_path),
                    bai_path=relpath(base, bai_path),
                    bam_size_bytes=bam_path.stat().st_size,
                    bai_size_bytes=0,
                    status="failed",
                    message="missing BAI file",
                )
            )
            continue

        bam_size = bam_path.stat().st_size
        bai_size = bai_path.stat().st_size

        if bam_size == 0:
            status = "failed"
            message = "empty BAM file"
        elif bai_size == 0:
            status = "failed"
            message = "empty BAI file"
        else:
            status = "ok"
            message = "input files present"

        records.append(
            InputFileRecord(
                sample=sample,
                bam_path=relpath(base, bam_path),
                bai_path=relpath(base, bai_path),
                bam_size_bytes=int(bam_size),
                bai_size_bytes=int(bai_size),
                status=status,
                message=message,
            )
        )

    failed = [record for record in records if record.status != "ok"]
    if failed:
        message_lines = ["Input validation failed:"]
        for record in failed:
            message_lines.append(f"  {record.sample}: {record.message}")
        raise FileNotFoundError("\n".join(message_lines))

    return records


def load_snps(path: Path, max_snps: Optional[int]) -> pd.DataFrame:
    """
    Load autosomal positions from an Eigenstrat .snp file.
    """
    autosomes = {str(i) for i in range(1, 23)}
    rows: List[Dict[str, object]] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            chrom = parts[1]
            chrom_num = chrom.replace("chr", "").replace("CHR", "")
            if chrom_num not in autosomes:
                continue

            pos = int(parts[3])
            rows.append({"chrom": chrom, "pos": pos})

            if max_snps is not None and len(rows) >= max_snps:
                break

    if not rows:
        raise ValueError(f"No autosomal SNP positions loaded from {path}")

    return pd.DataFrame(rows)


def normalize_chrom(bam_refs: Iterable[str], snp_chrom: str) -> str:
    """
    Return the chromosome label matching the BAM header if possible.
    """
    bam_ref_set = set(bam_refs)

    if snp_chrom in bam_ref_set:
        return snp_chrom

    with_chr = f"chr{snp_chrom}"
    if with_chr in bam_ref_set:
        return with_chr

    if snp_chrom.startswith("chr"):
        without_chr = snp_chrom[3:]
        if without_chr in bam_ref_set:
            return without_chr

    return snp_chrom


def base_at_refpos_from_read(read: bs.AlignedSegment, target_pos_1based: int) -> Optional[str]:
    """
    Return the base aligned to a requested 1-based reference position.

    Insertions and soft clips advance read position only. Deletions and skipped
    regions advance reference position and yield no base at the queried site.
    """
    if read.seq is None:
        return None

    ref_pos = read.pos + 1
    read_pos = 0
    cigar = read.cigar

    if cigar is None:
        idx = target_pos_1based - ref_pos
        if 0 <= idx < len(read.seq):
            return read.seq[idx].upper()
        return None

    for op, length in cigar:
        if op in (0, 7, 8):
            if ref_pos <= target_pos_1based < ref_pos + length:
                idx = read_pos + (target_pos_1based - ref_pos)
                if 0 <= idx < len(read.seq):
                    return read.seq[idx].upper()
                return None
            ref_pos += length
            read_pos += length
        elif op in (1, 4):
            read_pos += length
        elif op in (2, 3):
            if ref_pos <= target_pos_1based < ref_pos + length:
                return None
            ref_pos += length

    return None


def reference_end_1based_exclusive(read: bs.AlignedSegment) -> int:
    if read.seq is None:
        return read.pos + 1

    if read.cigar is None:
        return read.pos + len(read.seq) + 1

    ref_len = 0
    for op, length in read.cigar:
        if op in (0, 2, 3, 7, 8):
            ref_len += length

    return read.pos + ref_len + 1


def collect_sample_chrom_calls(
    sample: str,
    bam_path: str,
    source_chrom: str,
    positions: List[int],
) -> Tuple[str, Dict[int, str]]:
    target_positions = sorted({int(pos) for pos in positions})
    if not target_positions:
        return sample, {}

    bam = bs.AlignmentFile(bam_path, "rb")
    try:
        bam_refs = list(bam.references)
        ref_lengths = dict(zip(bam.references, bam.lengths))
        bam_chrom = normalize_chrom(bam_refs, source_chrom)
        if bam_chrom not in ref_lengths:
            return sample, {}

        start = max(0, target_positions[0] - 1)
        stop = min(target_positions[-1], int(ref_lengths[bam_chrom]))
        counters: Dict[int, Counter[str]] = defaultdict(Counter)

        for read in bam.fetch(bam_chrom, start, stop):
            if read.flag & 4 or read.seq is None:
                continue

            ref_start = read.pos + 1
            ref_end = reference_end_1based_exclusive(read)
            lo = bisect_left(target_positions, ref_start)
            hi = bisect_left(target_positions, ref_end, lo)

            for idx in range(lo, hi):
                pos = target_positions[idx]
                base = base_at_refpos_from_read(read, pos)
                if base in BASE_ORDER:
                    counters[pos][base] += 1

        calls = {
            pos: counts.most_common(1)[0][0]
            for pos, counts in counters.items()
            if counts
        }
        return sample, calls
    finally:
        bam.close()


def collect_panel_chrom_calls(
    samples: List[str],
    bam_dir: Path,
    source_chrom: str,
    positions: List[int],
    workers: int,
) -> Dict[str, Dict[int, str]]:
    tasks = [
        (
            sample,
            str(bam_dir / f"{sample}.1240k.bam"),
            source_chrom,
            positions,
        )
        for sample in samples
    ]

    if workers <= 1 or len(tasks) == 1:
        return {
            sample: calls
            for sample, calls in (
                collect_sample_chrom_calls(*task)
                for task in tasks
            )
        }

    out: Dict[str, Dict[int, str]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(collect_sample_chrom_calls, *task) for task in tasks]
        for future in as_completed(futures):
            sample, calls = future.result()
            out[sample] = calls
    return out


def build_complete_rows(
    snps: pd.DataFrame,
    samples: List[str],
    bam_dir: Path,
    progress_interval: int,
    workers: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    processed = 0

    for source_chrom, group in snps.groupby("chrom", sort=False):
        positions = [int(pos) for pos in group["pos"]]
        panel_calls = collect_panel_chrom_calls(
            samples=samples,
            bam_dir=bam_dir,
            source_chrom=str(source_chrom),
            positions=positions,
            workers=workers,
        )

        chrom_rows = 0
        for pos in positions:
            record: Dict[str, object] = {"chrom": source_chrom, "pos": int(pos)}
            dropped = False
            for sample in samples:
                base = panel_calls[sample].get(pos)
                if base is None:
                    dropped = True
                    break
                record[sample] = base

            if not dropped:
                rows.append(record)
                chrom_rows += 1

        processed += len(positions)
        covered = ", ".join(
            f"{sample}={len(panel_calls[sample])}" for sample in samples
        )
        print(
            f"[CHROM] {source_chrom}: retained {chrom_rows}/{len(positions)} | "
            f"processed {processed}/{len(snps)} | {covered}"
        )

        if processed % progress_interval < len(positions):
            print(f"[ASSEMBLE] Retained rows: {len(rows)}")

    return rows


def validate_existing_core_matrix(core_path: Path, core_samples: List[str]) -> pd.DataFrame:
    """
    Load and validate an existing seven-sample core matrix.
    """
    df = pd.read_csv(core_path, sep="\t", dtype=str)

    required_columns = ["chrom", "pos"] + core_samples
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Existing core matrix is missing required columns: {', '.join(missing)}"
        )

    df = df[required_columns].copy()
    df["pos"] = df["pos"].astype(int)

    if len(df) == 0:
        raise ValueError(f"Existing core matrix has zero rows: {core_path}")

    return df


def extract_core_matrix(
    panel_name: str,
    core_samples: List[str],
    bam_dir: Path,
    snp_path: Path,
    results_dir: Path,
    budget: Optional[int],
    checkpoint_interval: int,
    workers: int,
) -> Tuple[Path, pd.DataFrame, pd.DataFrame]:
    """
    Build or load the seven-sample core matrix.
    """
    core_path = results_dir / "core_7_sample_matrix.tsv"
    checkpoint_path = results_dir / "core_7_sample_checkpoint.json"
    snps = load_snps(snp_path, budget)
    print(f"[INFO] Loaded {len(snps)} autosomal SNP positions from {snp_path.name}.")

    if core_path.exists():
        print(f"[INFO] Reusing existing core matrix: {core_path}")
        core_df = validate_existing_core_matrix(core_path, core_samples)
        print(f"[OK] Existing core matrix rows: {len(core_df)}")
        return core_path, core_df, snps

    calls = build_complete_rows(
        snps=snps,
        samples=core_samples,
        bam_dir=bam_dir,
        progress_interval=checkpoint_interval,
        workers=workers,
    )

    if not calls:
        raise RuntimeError("Core matrix build retained zero complete-case rows.")

    core_df = pd.DataFrame(calls)
    ordered_columns = ["chrom", "pos"] + core_samples
    core_df = core_df[ordered_columns].copy()
    core_df.to_csv(core_path, sep="\t", index=False)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"[OK] Saved core matrix: {core_path}")
    print(f"[OK] Core retained rows: {len(core_df)}")

    return core_path, core_df, snps


def add_extra_samples(
    core_df: pd.DataFrame,
    extra_samples: List[str],
    bam_dir: Path,
    workers: int,
) -> Tuple[pd.DataFrame, Dict[str, int], int]:
    """
    Add E-V13 samples at the exact SNP positions retained in the core matrix.
    """
    snp_positions = core_df[["chrom", "pos"]].copy()
    print(f"[INFO] Extracting extra samples at {len(snp_positions)} core-retained positions.")

    combined_df = core_df.copy()
    for sample in extra_samples:
        combined_df[sample] = "N"

    for source_chrom, group in snp_positions.groupby("chrom", sort=False):
        positions = [int(pos) for pos in group["pos"]]
        panel_calls = collect_panel_chrom_calls(
            samples=extra_samples,
            bam_dir=bam_dir,
            source_chrom=str(source_chrom),
            positions=positions,
            workers=workers,
        )
        mask = combined_df["chrom"].astype(str).eq(str(source_chrom))
        for idx, pos in combined_df.loc[mask, "pos"].items():
            for sample in extra_samples:
                base = panel_calls[sample].get(int(pos), "N")
                combined_df.at[idx, sample] = base
        covered = ", ".join(
            f"{sample}={len(panel_calls[sample])}" for sample in extra_samples
        )
        print(f"[EXTRA] {source_chrom}: {covered}")

    all_sample_cols = [column for column in combined_df.columns if column not in {"chrom", "pos"}]
    valid_rows = pd.Series(True, index=combined_df.index)

    for sample in all_sample_cols:
        valid_rows &= combined_df[sample] != "N"

    pre_complete_case_call_counts = {
        sample: int((combined_df[sample] != "N").sum())
        for sample in all_sample_cols
    }
    final_df = combined_df.loc[valid_rows].copy()

    print(
        f"[OK] Retained complete-case rows after adding extra samples: "
        f"{len(final_df)} from {len(combined_df)} core rows."
    )

    if len(final_df) == 0:
        raise RuntimeError("Final nine-sample complete-case matrix has zero rows.")

    return final_df, pre_complete_case_call_counts, int(len(combined_df))


def chrom_row_counts(df: pd.DataFrame) -> Dict[str, int]:
    chrom_values = df["chrom"].astype(str).str.replace("chr", "", regex=False)
    counts = chrom_values.value_counts()
    return {
        str(chrom): int(count)
        for chrom, count in sorted(counts.items(), key=lambda item: int(item[0]))
    }


def write_summary(
    summary_path: Path,
    panel_name: str,
    base: Path,
    bam_dir: Path,
    snp_path: Path,
    results_dir: Path,
    budget: Optional[int],
    checkpoint_interval: int,
    workers: int,
    snps: pd.DataFrame,
    core_samples: List[str],
    extra_samples: List[str],
    input_records: List[InputFileRecord],
    core_path: Path,
    output_path: Path,
    core_df: pd.DataFrame,
    final_df: pd.DataFrame,
    pre_complete_case_call_counts: Dict[str, int],
    n_rows_before_nine_sample_complete_case: int,
) -> None:
    all_samples = core_samples + extra_samples
    chrom_values = snps["chrom"].astype(str).str.replace("chr", "", regex=False)
    chromosome_counts = {
        str(chrom): int(count)
        for chrom, count in sorted(
            chrom_values.value_counts().items(),
            key=lambda item: int(item[0]),
        )
    }
    autosomes_observed = [int(chrom) for chrom in chromosome_counts]
    budget_mode = "all_autosomes" if budget is None else "budgeted_autosomes"

    summary = {
        "script": "02_build_matrix.py",
        "timestamp_utc": utc_now(),
        "panel": panel_name,
        "snp_file": relpath(base, snp_path),
        "bam_dir": relpath(base, bam_dir),
        "results_dir": relpath(base, results_dir),
        "snp_budget": budget_label(budget),
        "snp_scan_mode": "all_autosomal" if budget is None else "budgeted_autosomal",
        "snp_budget_requested": "all" if budget is None else int(budget),
        "budget_mode": budget_mode,
        "n_autosomal_targets_available": int(len(snps)) if budget is None else None,
        "n_autosomal_targets_scanned": int(len(snps)),
        "n_autosomal_targets_loaded": int(len(snps)),
        "n_autosomal_target_positions_loaded": int(len(snps)),
        "chromosome_counts_loaded": chromosome_counts,
        "per_chromosome_targets_loaded": chromosome_counts,
        "autosomes_observed_before_complete_case": autosomes_observed,
        "n_autosomes_represented": int(chrom_values.nunique()),
        "checkpoint_interval": int(checkpoint_interval),
        "workers": int(workers),
        "core_samples": core_samples,
        "extra_samples": extra_samples,
        "all_samples": all_samples,
        "call_method": {
            "reference_space": "public 1240k SNP coordinates as represented in the source BAM headers",
            "read_filters": {
                "unmapped_reads": "excluded",
                "duplicate_marked_reads": "included",
                "mapq_threshold": None,
                "base_quality_threshold": None,
                "terminal_base_trimming": None,
                "damage_aware_filtering": None,
                "transitions": "included",
                "transversion_only_sensitivity": "not performed",
            },
            "minimum_read_depth_per_call": 1,
            "alleles_counted": list(BASE_ORDER),
            "tie_breaking_rule": "maximum-count consensus; equal-count ties follow first observed base in coordinate-sorted BAM traversal",
            "requires_match_to_1240k_allele_set": False,
            "genotype_model": "pseudo-haploid consensus; no diploid genotype calling, imputation, or genotype likelihood model",
        },
        "n_core_rows": int(len(core_df)),
        "n_rows_before_nine_sample_complete_case": int(n_rows_before_nine_sample_complete_case),
        "n_final_complete_case_rows": int(len(final_df)),
        "calling_method_changed": False,
        "mapq_filter": "none",
        "base_quality_filter": "none",
        "damage_terminal_trim": "none",
        "diploid_calling": False,
        "imputation": False,
        "model_based_ancestry_estimation": False,
        "valid_bases_counted": list(BASE_ORDER),
        "tie_policy": "maximum-count consensus; equal-count ties follow first observed base in coordinate-sorted BAM traversal",
        "transitions_included": True,
        "biallelic_1240k_allele_restriction": False,
        "per_chromosome_core_rows_retained": chrom_row_counts(core_df),
        "per_chromosome_final_rows_retained": chrom_row_counts(final_df),
        "pre_complete_case_call_counts": pre_complete_case_call_counts,
        "pre_complete_case_call_counts_note": (
            "Counts are measured after adding extra samples at core-retained positions "
            "and before final nine-sample complete-case filtering."
        ),
        "final_complete_case_call_counts": {
            sample: int((final_df[sample] != "N").sum())
            for sample in all_samples
        },
        "core_rows_by_chromosome": chrom_row_counts(core_df),
        "final_complete_case_rows_by_chromosome": chrom_row_counts(final_df),
        "core_matrix": relpath(base, core_path),
        "final_matrix": relpath(base, output_path),
        "input_files": [asdict(record) for record in input_records],
    }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> int:
    args = parse_args()

    panel_name = args.panel
    panel = PANELS[panel_name]
    base = args.base.resolve()
    core_samples = panel["core_samples"]
    extra_samples = panel["extra_samples"]
    all_samples = core_samples + extra_samples
    max_snps = normalize_budget(args.budget)

    bam_dir, snp_path, results_dir = resolve_paths(args)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.force:
        print("[INFO] --force set; removing existing matrix outputs for this panel.")
        clean_existing_outputs(
            results_dir,
            preserve_core=panel_name == "ev13_sensitivity",
        )

    print("=" * 80)
    print("Complete-case pseudo-haploid matrix builder")
    print(f"Panel: {panel_name}")
    print(f"Core samples: {', '.join(core_samples)}")
    print(f"Extra samples: {', '.join(extra_samples)}")
    print(f"All samples: {', '.join(all_samples)}")
    print(f"BAM directory: {bam_dir}")
    print(f"SNP file: {snp_path}")
    print(f"Results directory: {results_dir}")
    print(f"SNP budget: {budget_label(max_snps)}")
    print(f"Checkpoint interval: {args.checkpoint_interval}")
    print(f"Workers: {args.workers}")
    print("=" * 80)

    try:
        input_records = validate_input_files(
            samples=all_samples,
            base=base,
            bam_dir=bam_dir,
            snp_path=snp_path,
        )
        print("[OK] Input preflight passed for all required BAM/BAI files and SNP file.")

        core_path, core_df, snps = extract_core_matrix(
            panel_name=panel_name,
            core_samples=core_samples,
            bam_dir=bam_dir,
            snp_path=snp_path,
            results_dir=results_dir,
            budget=max_snps,
            checkpoint_interval=args.checkpoint_interval,
            workers=args.workers,
        )

        final_df, pre_complete_case_call_counts, n_rows_before_nine_sample_complete_case = add_extra_samples(
            core_df=core_df,
            extra_samples=extra_samples,
            bam_dir=bam_dir,
            workers=args.workers,
        )

        ordered_columns = ["chrom", "pos"] + all_samples
        final_df = final_df[ordered_columns].copy()

        output_path = results_dir / "complete_case_9_sample_matrix.tsv"
        final_df.to_csv(output_path, sep="\t", index=False)

        summary_path = results_dir / "matrix_build_summary.json"
        write_summary(
            summary_path=summary_path,
            panel_name=panel_name,
            base=base,
            bam_dir=bam_dir,
            snp_path=snp_path,
            results_dir=results_dir,
            budget=max_snps,
            checkpoint_interval=args.checkpoint_interval,
            workers=args.workers,
            snps=snps,
            core_samples=core_samples,
            extra_samples=extra_samples,
            input_records=input_records,
            core_path=core_path,
            output_path=output_path,
            core_df=core_df,
            final_df=final_df,
            pre_complete_case_call_counts=pre_complete_case_call_counts,
            n_rows_before_nine_sample_complete_case=n_rows_before_nine_sample_complete_case,
        )

    except Exception as exc:
        print(f"[ERROR] Matrix build failed: {exc}")
        return 1

    print("\n" + "=" * 80)
    print("[OK] Matrix build completed successfully.")
    print(f"Core matrix: {core_path}")
    print(f"Final complete-case matrix: {output_path}")
    print(f"Matrix build summary: {summary_path}")
    print(f"Core retained rows: {len(core_df)}")
    print(f"Final retained complete-case rows: {len(final_df)}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
