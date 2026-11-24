#!/usr/bin/env python3
"""Recompute complete-case calls after stringent ancient-DNA read filtering."""

import argparse
import json
import platform
import re
import subprocess
from bisect import bisect_left
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256 as sha256_digest
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import bamnostic
import pandas as pd


matrix_builder = import_module("02_build_matrix")
MIN_MAPPING_QUALITY = 30
MIN_BASE_QUALITY = 20
TRIM_BASES = 2
EXCLUDED_FLAGS = 0x4 | 0x100 | 0x200 | 0x400 | 0x800
PURINES = frozenset("AG")
PYRIMIDINES = frozenset("CT")
CIGAR_OPERATIONS = {letter: i for i, letter in enumerate("MIDNSHP=X")}
CIGAR_PATTERN = re.compile(r"(\d+)([MIDNSHP=X])")


def sha256(path):
    digest = sha256_digest()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def passing_observations(read, positions):
    if read.flag & EXCLUDED_FLAGS or not MIN_MAPPING_QUALITY <= read.mapq < 255:
        return
    sequence = read.seq
    if not sequence or sequence == "*":
        return
    qualities = read.query_qualities
    if qualities is None:
        return
    reference = read.pos + 1
    query = 0
    for operation, length in read.cigar or [(0, len(sequence))]:
        if operation in (0, 7, 8):
            start = bisect_left(positions, reference)
            stop = bisect_left(positions, reference + length, start)
            for site_index in range(start, stop):
                pos = positions[site_index]
                offset = query + pos - reference
                if TRIM_BASES <= offset < len(sequence) - TRIM_BASES:
                    base = sequence[offset]
                    quality = qualities[offset] - 33 if isinstance(qualities, bytes) else qualities[offset]
                    if MIN_BASE_QUALITY <= quality < 255 and base in matrix_builder.BASE_ORDER:
                        yield pos, base
            reference += length
            query += length
        elif operation in (1, 4):
            query += length
        elif operation in (2, 3):
            reference += length


def samtools_reads(samtools, bam_path, ref, start, stop):
    region = f"{ref}:{start}-{stop}"
    command = [str(samtools), "view", "-F", str(EXCLUDED_FLAGS), "-q", str(MIN_MAPPING_QUALITY), str(bam_path), region]
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        for line in process.stdout:
            fields = line.split(b"\t", 11)
            if fields[9] == b"*" or fields[10].rstrip(b"\r\n") == b"*":
                continue
            operations = [(CIGAR_OPERATIONS[op], int(length)) for length, op in CIGAR_PATTERN.findall(fields[5].decode("ascii"))]
            yield SimpleNamespace(
                flag=int(fields[1]), pos=int(fields[3]) - 1, mapq=int(fields[4]), cigar=operations,
                seq=fields[9].decode("ascii"), query_qualities=fields[10].rstrip(b"\r\n"),
            )
        if process.wait() != 0:
            raise RuntimeError(f"SAMtools could not read {bam_path.name} at {region}.")


def quality_calls(sample, bam_path, sites, samtools=None):
    calls = {}
    with bamnostic.AlignmentFile(str(bam_path), "rb") as bam:
        lengths = dict(zip(bam.references, bam.lengths))
        for chrom, positions in sorted(sites.items(), key=lambda item: int(item[0])):
            ref = matrix_builder.normalize_chrom(bam.references, chrom)
            if ref not in lengths:
                raise ValueError(f"Chromosome {chrom} is absent from {bam_path.name}.")
            start, stop = positions[0], min(positions[-1], lengths[ref])
            reads = samtools_reads(samtools, bam_path, ref, start, stop) if samtools else bam.fetch(ref, start - 1, stop)
            counts = defaultdict(Counter)
            for read in reads:
                for pos, base in passing_observations(read, positions):
                    counts[pos][base] += 1
            calls.update({(chrom, pos): bases.most_common(1)[0][0] for pos, bases in counts.items()})
    index_path = bam_path.with_suffix(".bam.bai")
    return sample, calls, {bam_path.name: sha256(bam_path), index_path.name: sha256(index_path)}


def sample_targets(panels):
    target_positions = defaultdict(lambda: defaultdict(set))
    for samples, frame in panels.values():
        for chrom, rows in frame.groupby("chrom", sort=False):
            positions = set(rows["pos"])
            for sample in samples:
                target_positions[sample][chrom].update(positions)
    return {
        sample: {chrom: sorted(positions) for chrom, positions in chromosomes.items()}
        for sample, chromosomes in target_positions.items()
    }


def transversion_alleles(path, required):
    alleles = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            _, chrom, _, position, first, second, *_ = line.split()
            key = (chrom, int(position))
            if key not in required:
                continue
            pair = frozenset((first, second))
            if key in alleles and alleles[key] != pair:
                raise ValueError(f"Conflicting SNP alleles at {chrom}:{position}.")
            alleles[key] = pair
    if required - alleles.keys():
        raise ValueError("The SNP file is missing complete-case positions.")
    return {key: pair for key, pair in alleles.items() if pair & PURINES and pair & PYRIMIDINES}


def rebuild_panel(samples, original, calls, transversions):
    keys = list(original[["chrom", "pos"]].itertuples(index=False, name=None))
    filtered = original[["chrom", "pos"]].copy()
    for sample in samples:
        filtered[sample] = [calls[sample].get(key) for key in keys]
    filtered = filtered.dropna()
    selected = []
    for chrom, pos, *bases in filtered.itertuples(index=False, name=None):
        alleles = transversions.get((chrom, pos), ())
        selected.append(all(base in alleles for base in bases))
    return {"filtered": filtered, "transversions": filtered.loc[selected]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path.cwd(), help="Repository root.")
    parser.add_argument("--bam-dir", type=Path, help="Directory containing the ENA BAM and BAI input files.")
    parser.add_argument("--outdir", type=Path, help="Output directory. Default: results/quality_control.")
    parser.add_argument("--workers", type=int, default=3, help="Number of sample BAMs processed in parallel.")
    parser.add_argument("--samtools", type=Path, help="Executable path for indexed reads via SAMtools. Default: BAMnostic.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    base = args.base.resolve()
    bam_dir = args.bam_dir.resolve() if args.bam_dir else base / "ena_1240k_all"
    outdir = args.outdir.resolve() if args.outdir else base / "results" / "quality_control"
    samtools = args.samtools.resolve() if args.samtools else None
    reader = (
        subprocess.check_output([str(samtools), "--version-only"], text=True).strip() if samtools else None
    )
    snp_path = base / "v62.0_1240k_public.snp"
    panels, source_files = {}, [snp_path, Path(__file__).resolve(), Path(matrix_builder.__file__).resolve()]
    required = set()
    for panel, groups in matrix_builder.PANELS.items():
        path = base / f"matrix_{panel}" / "complete_case_9_sample_matrix.tsv"
        samples = groups["core_samples"] + groups["extra_samples"]
        frame = pd.read_csv(path, sep="\t", dtype={"chrom": str, "pos": int})[["chrom", "pos", *samples]]
        keys = set(frame[["chrom", "pos"]].itertuples(index=False, name=None))
        if frame.empty or len(keys) != len(frame) or frame[samples].isna().any().any():
            raise ValueError(f"Invalid complete-case input for {panel}.")
        panels[panel] = samples, frame
        required.update(keys)
        source_files.append(path)

    transversions = transversion_alleles(snp_path, required)
    targets_by_sample = sample_targets(panels)
    bam_paths = {sample: bam_dir / f"{sample}.1240k.bam" for sample in targets_by_sample}
    for path in bam_paths.values():
        if not path.is_file() or not path.with_suffix(".bam.bai").is_file():
            raise FileNotFoundError(f"BAM or BAI missing for {path.name}.")

    calls, bam_digests = {}, {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(quality_calls, sample, bam_paths[sample], sites, samtools) for sample, sites in targets_by_sample.items()]
        for task in as_completed(tasks):
            sample, by_site, digests = task.result()
            calls[sample] = by_site
            bam_digests.update(digests)
            print(f"{sample}: {len(by_site):,} sites with calls after filtering", flush=True)

    output_files, summary = [], []
    for panel, (samples, original) in panels.items():
        for variant, frame in rebuild_panel(samples, original, calls, transversions).items():
            if frame.empty:
                raise ValueError(f"No complete-case sites retained for {panel} ({variant}).")
            changed = frame[samples].ne(original.loc[frame.index, samples])
            summary.append({
                "panel": panel, "filter": variant, "n_original_sites": len(original),
                "n_retained_sites": len(frame), "n_autosomes": frame["chrom"].nunique(),
                "n_sites_with_changed_calls": int(changed.any(axis=1).sum()),
                "n_changed_calls": int(changed.to_numpy().sum()),
            })
            path = outdir / variant / f"matrix_{panel}" / "complete_case_9_sample_matrix.tsv"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, sep="\t", index=False, lineterminator="\n")
            output_files.append(path)

    summary_path = outdir / "quality_control_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False, lineterminator="\n")
    output_files.append(summary_path)
    metadata = {
        "source_sha256": {path.relative_to(base).as_posix(): sha256(path) for path in source_files},
        "bam_and_index_sha256": dict(sorted(bam_digests.items())),
        "software": {"python": platform.python_version(), "pandas": pd.__version__,
                     "bamnostic": bamnostic.__version__, "samtools": reader},
        "min_mapping_quality": MIN_MAPPING_QUALITY,
        "min_base_quality": MIN_BASE_QUALITY,
        "trimmed_read_end_bases": TRIM_BASES,
        "excluded_alignment_flags": f"0x{EXCLUDED_FLAGS:x}",
        "output_sha256": {path.relative_to(outdir).as_posix(): sha256(path) for path in output_files},
    }
    (outdir / "quality_control_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
