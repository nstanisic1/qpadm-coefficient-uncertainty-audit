#!/usr/bin/env python3
"""
Download and validate ENA 1240k BAM/BAI files for the reproducible
allele-sharing analysis.

This script retrieves the BAM and BAI files required for either the main
autosomal panel or the E-V13 sensitivity panel. It is designed for reproducible
pipeline execution: transfers are resumed when possible, completed files are
validated, and a panel-specific manifest is written for auditability.

Default usage
-------------
Download all BAM/BAI files required for both the main and sensitivity analyses:
    python 01_download_bams.py --panel all --base .

Targeted rerun for the main panel only:
    python 01_download_bams.py --panel main --base .

Targeted rerun for the E-V13 sensitivity panel only:
    python 01_download_bams.py --panel ev13_sensitivity --base .

Force clean re-download of all required files:
    python 01_download_bams.py --panel all --base . --force

Inputs
------
The default ENA metadata file is:

    PRJEB66422_read_run.tsv

It must contain at least the columns:

    sample_alias
    submitted_ftp

If present, the column `library_name` is used to exclude mitochondrial-only
libraries ending in `_MT`.

Outputs
-------
By default, files are written to:

    ena_1240k_all/
    ena_1240k_main/
    ena_1240k_ev13_sensitivity/

depending on the selected panel.

A manifest is written to:

    results/download_manifest_all.tsv
    results/download_manifest_main.tsv
    results/download_manifest_ev13_sensitivity.tsv

depending on the selected panel.

Dependencies
------------
    pandas
    bamnostic
"""

from __future__ import annotations

import argparse
import ftplib
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from urllib.error import URLError
from urllib.parse import urlparse

import bamnostic as bs
import pandas as pd


MAIN_PANEL = [
    "I1116",
    "I32299",
    "I15541",
    "I15542",
    "I26766",
    "I26748",
    "I35012",
    "I15537",
    "I27297",
]

EV13_SENSITIVITY_PANEL = [
    "I1116",
    "I32299",
    "I15541",
    "I15542",
    "I26766",
    "I26748",
    "I35012",
    "I15537",
    "I28388",
]

ALL_REQUIRED_SAMPLES = list(dict.fromkeys(MAIN_PANEL + EV13_SENSITIVITY_PANEL))

PANELS: Dict[str, list[str]] = {
    "all": ALL_REQUIRED_SAMPLES,
    "main": MAIN_PANEL,
    "ev13_sensitivity": EV13_SENSITIVITY_PANEL,
}


DEFAULT_BASE = Path.cwd()
DEFAULT_ENA_TSV = "PRJEB66422_read_run.tsv"
DEFAULT_CHUNK_SIZE = 1024 * 1024
DEFAULT_MAX_RESUME_ATTEMPTS = 50
DEFAULT_RETRY_SLEEP_SECONDS = 5
DEFAULT_BUSY_SLEEP_SECONDS = 15
DEFAULT_TIMEOUT_SECONDS = 60


@dataclass
class DownloadRecord:
    sample: str
    panel: str
    status: str
    message: str
    bam_path: str
    bai_path: str
    bam_url: str
    bai_url: str
    bam_size_bytes: Optional[int]
    bai_size_bytes: Optional[int]
    bam_local_size_bytes: Optional[int]
    bai_local_size_bytes: Optional[int]
    bam_valid: bool
    bai_valid: bool
    timestamp_utc: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download, resume, and validate ENA 1240k BAM/BAI files for a "
            "selected allele-sharing analysis panel."
        )
    )
    parser.add_argument(
        "--panel",
        choices=sorted(PANELS.keys()),
        default="all",
        help="Target sample panel to download. Default: all.",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
        help="Project base directory. Default: current working directory.",
    )
    parser.add_argument(
        "--ena-tsv",
        type=Path,
        default=None,
        help=(
            f"Path to ENA metadata TSV. If omitted, "
            f"'<base>/{DEFAULT_ENA_TSV}' is used."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help=(
            "Output directory for BAM/BAI files. If omitted, "
            "'<base>/ena_1240k_<panel>' is used."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Path to output manifest TSV. If omitted, "
            "'<base>/results/download_manifest_<panel>.tsv' is used."
        ),
    )
    parser.add_argument(
        "--max-resume-attempts",
        type=int,
        default=DEFAULT_MAX_RESUME_ATTEMPTS,
        help=f"Maximum resume attempts per file. Default: {DEFAULT_MAX_RESUME_ATTEMPTS}.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Read chunk size in bytes. Default: {DEFAULT_CHUNK_SIZE}.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Network timeout in seconds. Default: {DEFAULT_TIMEOUT_SECONDS}.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=int,
        default=DEFAULT_RETRY_SLEEP_SECONDS,
        help=f"Sleep time in seconds after ordinary transfer errors. Default: {DEFAULT_RETRY_SLEEP_SECONDS}.",
    )
    parser.add_argument(
        "--busy-sleep",
        type=int,
        default=DEFAULT_BUSY_SLEEP_SECONDS,
        help=f"Sleep time in seconds after ENA busy/unavailable responses. Default: {DEFAULT_BUSY_SLEEP_SECONDS}.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing BAM/BAI files for each target sample before downloading.",
    )
    return parser.parse_args()


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
    base: Path = args.base.resolve()
    panel: str = args.panel

    ena_tsv = args.ena_tsv.resolve() if args.ena_tsv else base / DEFAULT_ENA_TSV
    outdir = args.outdir.resolve() if args.outdir else base / f"ena_1240k_{panel}"
    manifest = (
        args.manifest.resolve()
        if args.manifest
        else base / "results" / f"download_manifest_{panel}.tsv"
    )

    return ena_tsv, outdir, manifest


def normalize_url(raw: str) -> str:
    """
    Return a usable URL. ENA submitted_ftp entries may omit the scheme.
    """
    value = str(raw).strip()
    if not value:
        return ""

    parsed = urlparse(value)
    if parsed.scheme in {"ftp", "http", "https"}:
        return value

    return f"ftp://{value}"


def parse_1240k_urls(submitted_ftp: str) -> Tuple[str, str]:
    """
    Extract the 1240k BAM and BAI URLs from an ENA submitted_ftp field.
    """
    bam_url = ""
    bai_url = ""

    for entry in str(submitted_ftp).split(";"):
        entry = normalize_url(entry)
        if not entry:
            continue

        lower = entry.lower()
        if lower.endswith(".1240k.bam"):
            bam_url = entry
        elif lower.endswith(".1240k.bam.bai"):
            bai_url = entry

    return bam_url, bai_url


def load_ena_table(ena_tsv: Path) -> pd.DataFrame:
    if not ena_tsv.exists():
        raise FileNotFoundError(f"ENA metadata TSV not found: {ena_tsv}")

    df = pd.read_csv(ena_tsv, sep="\t", dtype=str).fillna("")

    required = {"sample_alias", "submitted_ftp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "ENA metadata TSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    df["sample_alias_norm"] = df["sample_alias"].astype(str).str.strip()

    if "library_name" in df.columns:
        df["library_name_norm"] = df["library_name"].astype(str).str.strip()
    else:
        df["library_name_norm"] = ""

    return df


def build_ena_map(
    ena_df: pd.DataFrame,
    targets: Iterable[str],
) -> Dict[str, Dict[str, str]]:
    """
    Build sample -> BAM/BAI URL mapping for target samples.

    Mitochondrial-only libraries ending in `_MT` are excluded when library_name
    is available.
    """
    target_set = set(targets)
    mapping: Dict[str, Dict[str, str]] = {}

    for sample, subdf in ena_df.groupby("sample_alias_norm", sort=False):
        if sample not in target_set:
            continue

        sub_non_mt = subdf[~subdf["library_name_norm"].str.endswith("_MT")].copy()
        if sub_non_mt.empty:
            continue

        for _, row in sub_non_mt.iterrows():
            bam_url, bai_url = parse_1240k_urls(row.get("submitted_ftp", ""))
            if bam_url or bai_url:
                mapping[sample] = {"bam": bam_url, "bai": bai_url}
                break

    return mapping


def get_remote_file_size(url: str, timeout: int) -> Optional[int]:
    """
    Return remote file size in bytes when available.

    FTP SIZE is used for ftp:// URLs. For HTTP(S), Content-Length is used.
    The pipeline intentionally fails rather than downloading blind, because
    exact file-size checks are part of reproducible transfer validation.
    """
    parsed = urlparse(url)

    try:
        if parsed.scheme == "ftp":
            with ftplib.FTP(parsed.hostname, timeout=timeout) as ftp:
                ftp.login()
                size = ftp.size(parsed.path)
                return int(size) if size is not None else None

        if parsed.scheme in {"http", "https"}:
            request = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                length = response.headers.get("Content-Length")
                return int(length) if length is not None else None

    except Exception as exc:
        print(f"       [WARN] Could not determine remote size for {url}: {exc}")
        return None

    print(f"       [WARN] Unsupported URL scheme for size check: {url}")
    return None


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def download_with_resume(
    url: str,
    dest_path: Path,
    max_resume_attempts: int,
    chunk_size: int,
    timeout: int,
    retry_sleep: int,
    busy_sleep: int,
) -> Tuple[bool, Optional[int], str]:
    """
    Download a URL to dest_path using byte-range resume when possible.

    Returns:
        success, remote_size_bytes, message
    """
    if not url:
        return False, None, "missing URL"

    remote_size = get_remote_file_size(url, timeout=timeout)
    if remote_size is None:
        return False, None, "remote file size could not be determined"

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_resume_attempts + 1):
        local_size = dest_path.stat().st_size if dest_path.exists() else 0

        if local_size == remote_size:
            print(
                f"       [OK] {dest_path.name} already complete "
                f"({local_size / 1e6:.1f} MB)."
            )
            return True, remote_size, "complete"

        if local_size > remote_size:
            print(
                f"       [WARN] {dest_path.name} is larger than remote file; "
                "restarting."
            )
            dest_path.unlink()
            local_size = 0

        print(
            f"       -> Attempt {attempt}/{max_resume_attempts}: "
            f"{local_size / 1e6:.1f} MB / {remote_size / 1e6:.1f} MB"
        )

        request = urllib.request.Request(url)
        if local_size > 0:
            request.add_header("Range", f"bytes={local_size}-")

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                mode = "ab" if local_size > 0 else "wb"
                with open(dest_path, mode) as handle:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        handle.write(chunk)

            final_size = dest_path.stat().st_size if dest_path.exists() else 0
            if final_size == remote_size:
                print(f"       [OK] {dest_path.name} complete.")
                return True, remote_size, "complete"

            print(
                f"       [WARN] Size mismatch after attempt: "
                f"{final_size} local bytes vs {remote_size} remote bytes."
            )
            time.sleep(retry_sleep)

        except URLError as exc:
            msg = str(exc)
            if "550" in msg or "450" in msg:
                print(
                    f"       [WARN] ENA returned a transient availability error: {exc}. "
                    f"Sleeping for {busy_sleep} seconds."
                )
                time.sleep(busy_sleep)
            else:
                print(f"       [WARN] Transfer interrupted: {exc}. Retrying.")
                time.sleep(retry_sleep)

        except Exception as exc:
            print(f"       [WARN] Transfer error: {exc}. Retrying.")
            time.sleep(retry_sleep)

    return False, remote_size, f"exhausted {max_resume_attempts} download attempts"


def validate_bam_and_bai(bam_path: Path, bai_path: Path) -> Tuple[bool, bool, str]:
    """
    Validate that the BAM and BAI files exist and that the BAM can be opened.

    The downstream matrix-building script uses indexed random access. Therefore,
    a non-empty BAI is required here.
    """
    if not bam_path.exists():
        return False, False, "BAM file is missing"

    if not bai_path.exists():
        return False, False, "BAI file is missing"

    if bam_path.stat().st_size == 0:
        return False, False, "BAM file is empty"

    if bai_path.stat().st_size == 0:
        return False, False, "BAI file is empty"

    try:
        bam = bs.AlignmentFile(str(bam_path), "rb")
        references = list(bam.references)
        bam.close()

        if not references:
            return False, True, "BAM opened but contains no reference names"

        return True, True, "OK"

    except Exception as exc:
        return False, True, f"BAM validation failed: {exc}"


def make_record(
    base: Path,
    sample: str,
    panel: str,
    status: str,
    message: str,
    bam_path: Path,
    bai_path: Path,
    bam_url: str,
    bai_url: str,
    bam_size_bytes: Optional[int],
    bai_size_bytes: Optional[int],
    bam_valid: bool,
    bai_valid: bool,
) -> DownloadRecord:
    bam_local = bam_path.stat().st_size if bam_path.exists() else None
    bai_local = bai_path.stat().st_size if bai_path.exists() else None

    return DownloadRecord(
        sample=sample,
        panel=panel,
        status=status,
        message=message,
        bam_path=relpath(base, bam_path),
        bai_path=relpath(base, bai_path),
        bam_url=bam_url,
        bai_url=bai_url,
        bam_size_bytes=bam_size_bytes,
        bai_size_bytes=bai_size_bytes,
        bam_local_size_bytes=bam_local,
        bai_local_size_bytes=bai_local,
        bam_valid=bam_valid,
        bai_valid=bai_valid,
        timestamp_utc=utc_now(),
    )


def write_manifest(records: list[DownloadRecord], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(record) for record in records])
    df.to_csv(manifest_path, sep="\t", index=False)


def process_sample(
    sample: str,
    panel: str,
    urls: Optional[Dict[str, str]],
    outdir: Path,
    base: Path,
    args: argparse.Namespace,
) -> DownloadRecord:
    bam_path = outdir / f"{sample}.1240k.bam"
    bai_path = outdir / f"{sample}.1240k.bam.bai"

    if urls is None:
        return make_record(
            base=base,
            sample=sample,
            panel=panel,
            status="failed",
            message="no ENA 1240k BAM/BAI entry found",
            bam_path=bam_path,
            bai_path=bai_path,
            bam_url="",
            bai_url="",
            bam_size_bytes=None,
            bai_size_bytes=None,
            bam_valid=False,
            bai_valid=False,
        )

    bam_url = urls.get("bam", "")
    bai_url = urls.get("bai", "")

    print(f"\n[PROCESSING] {sample}")

    if not bam_url or not bai_url:
        message = "missing BAM URL" if not bam_url else "missing BAI URL"
        print(f"   [ERROR] {message}.")
        return make_record(
            base=base,
            sample=sample,
            panel=panel,
            status="failed",
            message=message,
            bam_path=bam_path,
            bai_path=bai_path,
            bam_url=bam_url,
            bai_url=bai_url,
            bam_size_bytes=None,
            bai_size_bytes=None,
            bam_valid=False,
            bai_valid=False,
        )

    if args.force:
        print("   [INFO] --force set; removing existing BAM/BAI files if present.")
        remove_if_exists(bam_path)
        remove_if_exists(bai_path)

    print("   Downloading BAM")
    bam_ok, bam_remote_size, bam_message = download_with_resume(
        url=bam_url,
        dest_path=bam_path,
        max_resume_attempts=args.max_resume_attempts,
        chunk_size=args.chunk_size,
        timeout=args.timeout,
        retry_sleep=args.retry_sleep,
        busy_sleep=args.busy_sleep,
    )

    if not bam_ok:
        print(f"   [ERROR] BAM download failed for {sample}: {bam_message}")
        remove_if_exists(bam_path)
        remove_if_exists(bai_path)
        return make_record(
            base=base,
            sample=sample,
            panel=panel,
            status="failed",
            message=f"BAM download failed: {bam_message}",
            bam_path=bam_path,
            bai_path=bai_path,
            bam_url=bam_url,
            bai_url=bai_url,
            bam_size_bytes=bam_remote_size,
            bai_size_bytes=None,
            bam_valid=False,
            bai_valid=False,
        )

    print("   Downloading BAI")
    bai_ok, bai_remote_size, bai_message = download_with_resume(
        url=bai_url,
        dest_path=bai_path,
        max_resume_attempts=args.max_resume_attempts,
        chunk_size=args.chunk_size,
        timeout=args.timeout,
        retry_sleep=args.retry_sleep,
        busy_sleep=args.busy_sleep,
    )

    if not bai_ok:
        print(f"   [ERROR] BAI download failed for {sample}: {bai_message}")
        remove_if_exists(bam_path)
        remove_if_exists(bai_path)
        return make_record(
            base=base,
            sample=sample,
            panel=panel,
            status="failed",
            message=f"BAI download failed: {bai_message}",
            bam_path=bam_path,
            bai_path=bai_path,
            bam_url=bam_url,
            bai_url=bai_url,
            bam_size_bytes=bam_remote_size,
            bai_size_bytes=bai_remote_size,
            bam_valid=False,
            bai_valid=False,
        )

    bam_valid, bai_valid, validation_message = validate_bam_and_bai(bam_path, bai_path)
    if not (bam_valid and bai_valid):
        print(f"   [ERROR] Validation failed for {sample}: {validation_message}")
        print("   [INFO] Removing BAM/BAI so the sample can be re-downloaded cleanly.")
        remove_if_exists(bam_path)
        remove_if_exists(bai_path)
        return make_record(
            base=base,
            sample=sample,
            panel=panel,
            status="failed",
            message=validation_message,
            bam_path=bam_path,
            bai_path=bai_path,
            bam_url=bam_url,
            bai_url=bai_url,
            bam_size_bytes=bam_remote_size,
            bai_size_bytes=bai_remote_size,
            bam_valid=bam_valid,
            bai_valid=bai_valid,
        )

    print(f"   [OK] {sample} BAM/BAI passed validation.")
    return make_record(
        base=base,
        sample=sample,
        panel=panel,
        status="ok",
        message="BAM/BAI downloaded and validated",
        bam_path=bam_path,
        bai_path=bai_path,
        bam_url=bam_url,
        bai_url=bai_url,
        bam_size_bytes=bam_remote_size,
        bai_size_bytes=bai_remote_size,
        bam_valid=True,
        bai_valid=True,
    )


def main() -> int:
    args = parse_args()
    panel_name: str = args.panel
    targets = PANELS[panel_name]
    base = args.base.resolve()

    ena_tsv, outdir, manifest_path = resolve_paths(args)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ENA 1240k BAM/BAI downloader")
    print(f"Panel: {panel_name}")
    print(f"Samples: {', '.join(targets)}")
    print(f"ENA metadata TSV: {ena_tsv}")
    print(f"Output directory: {outdir}")
    print(f"Manifest: {manifest_path}")
    print("=" * 80)

    try:
        ena_df = load_ena_table(ena_tsv)
        ena_map = build_ena_map(ena_df, targets)
    except Exception as exc:
        print(f"[ERROR] Could not initialize download map: {exc}")
        return 1

    records: list[DownloadRecord] = []

    for sample in targets:
        record = process_sample(
            sample=sample,
            panel=panel_name,
            urls=ena_map.get(sample),
            outdir=outdir,
            base=base,
            args=args,
        )
        records.append(record)

        write_manifest(records, manifest_path)

    ok_count = sum(record.status == "ok" for record in records)
    fail_count = len(records) - ok_count

    print("\n" + "=" * 80)
    print(f"Completed: {ok_count}/{len(targets)} samples downloaded and validated.")
    print(f"Manifest written to: {manifest_path}")

    if fail_count:
        print(f"[ERROR] {fail_count} sample(s) failed. See manifest for details.")
        print("=" * 80)
        return 1

    print("[OK] All requested samples are ready for matrix construction.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
