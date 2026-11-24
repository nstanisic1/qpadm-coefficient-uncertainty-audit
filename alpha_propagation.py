"""Propagate uncertainty in the ratio of two compositional coefficients."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    n_sim: int = 100_000
    seed: int = 20240421
    rho_grid: tuple[float, ...] = (0.0, -0.25, -0.50, -0.75)
    max_batch_multiplier: int = 25
    batch_size: int = 50_000
    eps: float = 1e-12


@dataclass(frozen=True)
class SimulationResult:
    n_valid_retained: int
    valid_fraction_raw: float
    median_alpha: float
    ci_lower_95: float
    ci_upper_95: float

    @property
    def ci_width(self) -> float:
        return self.ci_upper_95 - self.ci_lower_95


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_focal_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    coefficients = ["p_local", "p_anatolian", "p_eastern"]
    errors = ["se_local", "se_anatolian", "se_eastern"]
    required = {"individual", *coefficients, *errors}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing focal table columns: {', '.join(sorted(missing))}.")
    if table.empty or table["individual"].isna().any() or table["individual"].duplicated().any():
        raise ValueError("The focal table must contain one row per individual.")
    if not table[coefficients].apply(lambda col: col.between(0, 1)).to_numpy().all():
        raise ValueError("Focal coefficients must be in [0, 1].")
    if not np.isfinite(table[errors].to_numpy()).all() or (table[errors] < 0).any().any():
        raise ValueError("Focal standard errors must be finite and non-negative.")
    return table


def deterministic_seed(base_seed: int, individual: str, rho: float) -> int:
    key = f"{base_seed}|{individual}|{rho:.6f}".encode("utf-8")
    offset = int.from_bytes(sha256(key).digest()[:8], "big") % (2**32 - 1)
    return (base_seed + offset) % (2**32 - 1)


def simulate_alpha_distribution(row: pd.Series, rho: float, config: SimulationConfig) -> SimulationResult:
    mean = np.array([row.p_local, row.p_anatolian], dtype=float)
    se_local, se_anatolian = float(row.se_local), float(row.se_anatolian)
    cross_cov = rho * se_local * se_anatolian
    cov = np.array([[se_local**2, cross_cov], [cross_cov, se_anatolian**2]])
    rng = np.random.default_rng(deterministic_seed(config.seed, str(row.individual), float(rho)))
    batches = []
    n_generated = n_accepted = 0
    max_generated = config.n_sim * config.max_batch_multiplier
    while n_accepted < config.n_sim and n_generated < max_generated:
        batch_size = min(max(config.batch_size, config.n_sim - n_accepted), max_generated - n_generated)
        draws = rng.multivariate_normal(mean=mean, cov=cov, size=batch_size, check_valid="raise")
        total = draws[:, 0] + draws[:, 1]
        feasible = (
            np.isfinite(draws).all(axis=1)
            & ((draws >= 0) & (draws <= 1)).all(axis=1)
            & (total <= 1 + config.eps)
            & (total > config.eps)
        )
        batches.append(draws[feasible, 0] / total[feasible])
        n_generated += batch_size
        n_accepted += int(feasible.sum())
    if n_accepted < config.n_sim:
        raise RuntimeError(f"Only {n_accepted} feasible draws for {row.individual} at rho={rho}.")
    ratios = np.concatenate(batches)[:config.n_sim]
    return SimulationResult(
        len(ratios),
        n_accepted / n_generated,
        float(np.median(ratios)),
        float(np.quantile(ratios, 0.025)),
        float(np.quantile(ratios, 0.975)),
    )
