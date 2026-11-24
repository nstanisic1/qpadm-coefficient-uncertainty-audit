"""Complete-case panel definitions and pairwise mismatch distances."""

import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


PANELS = {
    "main": {"matrix_dir": "matrix_main", "ev13_comparator": "I27297"},
    "ev13_sensitivity": {"matrix_dir": "matrix_ev13_sensitivity", "ev13_comparator": "I28388"},
}
I2A_NON_KULINE = ("I1116", "I32299")
I2A_KULINE = ("I15541", "I15542")
R1A_REFERENCES = ("I26766", "I26748", "I35012")
SIRMIUM_COORDS = (44.9764, 19.6122)
I2A_COORDS = {
    "I1116": (44.8889, 19.7494),
    "I32299": (44.7320, 21.2340),
    "I15541": (43.6450, 22.2650),
    "I15542": (43.6450, 22.2650),
}


@dataclass(frozen=True)
class PanelData:
    panel: str
    matrix_path: Path
    df: pd.DataFrame
    samples: tuple[str, ...]
    allele_array: np.ndarray
    groups: dict[str, tuple[str, ...]]
    sample_to_idx: dict[str, int]

    @property
    def n_snps(self) -> int:
        return len(self.df)


def load_panel_data(base: Path, panel: str) -> PanelData:
    cfg = PANELS[panel]
    path = base / cfg["matrix_dir"] / "complete_case_9_sample_matrix.tsv"
    groups = {
        "i2a_non_kuline": I2A_NON_KULINE,
        "i2a_kuline": I2A_KULINE,
        "all_i2a": I2A_NON_KULINE + I2A_KULINE,
        "ev13_kuline": ("I15537",),
        "ev13_non_kuline": (cfg["ev13_comparator"],),
        "r1a_references": R1A_REFERENCES,
    }
    samples = groups["all_i2a"] + groups["ev13_kuline"] + groups["ev13_non_kuline"] + R1A_REFERENCES
    df = pd.read_csv(path, sep="\t", dtype=str)
    required = ["chrom", "pos", *samples]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {', '.join(sorted(missing))}.")
    df = df[required]
    calls = df[list(samples)]
    if df.empty or not calls.isin(("A", "C", "G", "T")).to_numpy().all():
        raise ValueError(f"Invalid complete-case allele calls for panel {panel}.")
    return PanelData(panel, path, df, samples, calls.to_numpy(), groups,
                     {sample: i for i, sample in enumerate(samples)})


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relpath(base: Path, path: Path) -> str:
    path = path.resolve()
    return path.relative_to(base).as_posix() if path.is_relative_to(base) else path.as_posix()


def deterministic_seed(base_seed: int, label: str) -> int:
    key = f"{base_seed}|{label}".encode("utf-8")
    offset = int.from_bytes(sha256(key).digest()[:8], "big") % (2**32 - 1)
    return (base_seed + offset) % (2**32 - 1)


def mismatch_counts(alleles: np.ndarray) -> np.ndarray:
    if not len(alleles):
        raise ValueError("Pairwise distances require at least one locus.")
    return np.count_nonzero(alleles[:, :, None] != alleles[:, None, :], axis=0)


def compute_distance_matrix(alleles: np.ndarray) -> np.ndarray:
    return mismatch_counts(alleles) / len(alleles)


def idx_list(panel: PanelData, names) -> list[int]:
    return [panel.sample_to_idx[sample] for sample in names]


def haversine(coord1, coord2) -> float:
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    return 6371.0 * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def compute_sirmium_gradient(distances: np.ndarray, panel: PanelData, i2a: list[int], refs: list[int]):
    names = [panel.samples[idx] for idx in i2a]
    km = [haversine(I2A_COORDS[name], SIRMIUM_COORDS) for name in names]
    mismatch = [float(np.mean([distances[idx, ref] for ref in refs])) for idx in i2a]
    r = float(pearsonr(km, mismatch).statistic) if len(set(mismatch)) > 1 else float("nan")
    return r, names, km, mismatch
