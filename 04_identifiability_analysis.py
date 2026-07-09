#!/usr/bin/env python3
"""
Run alpha identifiability and correlation-sensitivity analyses for the
published non-eastern ancestry component in focal medieval Balkan individuals.

This script quantifies how strongly the local-to-Anatolian composition of the
modeled non-eastern ancestry component is resolved under the published
three-way admixture estimates and standard errors.

Primary estimand
----------------
For each focal individual, the primary estimand is:

    alpha = p_local / (p_local + p_anatolian)

where:
    * p_local     = Croatia_IA contribution
    * p_anatolian = WestAnatolia_Roman_Byzantine contribution
    * p_eastern   = eastern-associated contribution in the published three-way fit

Scope
-----
This script does not reproduce qpAdm, reconstruct a full posterior, or infer a
unique unsampled source population. It performs a bounded Monte Carlo
uncertainty-propagation and correlation-sensitivity analysis using published
point estimates and standard errors.

Methodological design
---------------------
* Embedded Olalde-derived inputs are used by default and exported with the outputs.
* Structured CSV inputs may be supplied explicitly to test alternative values.
* The focal uncertainty model is a bivariate normal approximation on the local
  and Anatolian terms, evaluated across a grid of assumed correlations.
* Draws are rejected if they violate compositional bounds:
      0 <= local <= 1
      0 <= anatolian <= 1
      local + anatolian <= 1
* Alpha is computed only for valid draws with a positive denominator.
* Same-axis and related-axis comparator individuals are kept distinct.

Default usage
-------------
Run with embedded Olalde-derived focal and comparator tables:
    python 04_identifiability_analysis.py --base . --save-draws

Run with explicit external focal/comparator CSV files:
    python 04_identifiability_analysis.py ^
      --input data/olalde_table6_focal.csv ^
      --validation data/late_antique_validation.csv ^
      --outdir results/alpha_identifiability ^
      --n-sim 100000 ^
      --seed 20240421 ^
      --rho-grid 0,-0.25,-0.5,-0.75 ^
      --save-draws

Inputs
------
By default, the script uses embedded focal and comparator tables transcribed
from Olalde et al. Data S2 Table 6 and checked against the cited source table.
The embedded tables are printed to the console and written to the output
directory as focal_inputs_used.csv and comparators_used.csv.

External focal CSVs may be supplied with:
    individual, p_local, se_local, p_anatolian, se_anatolian, p_eastern

External comparator CSVs may be supplied with either:
    individual, alpha_observed
or:
    individual, p_local, p_anatolian

Recommended comparator columns also include:
    site, date, model_axis, axis_type, se, ci_lower_95, ci_upper_95, notes

Outputs
-------
    results/alpha_identifiability/focal_inputs_used.csv
    results/alpha_identifiability/comparators_used.csv
    results/alpha_identifiability/alpha_summary.csv
    results/alpha_identifiability/alpha_draws.csv        [when --save-draws is used]
    results/alpha_identifiability/alpha_run_metadata.json
    results/alpha_identifiability/comparators_normalized.csv

Figures are generated separately by:
    python 05_make_figures.py --base .

Dependencies
------------
    numpy
    pandas
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


REQUIRED_INPUT_COLUMNS = {
    "individual",
    "p_local",
    "se_local",
    "p_anatolian",
    "se_anatolian",
    "p_eastern",
}

REQUIRED_VALIDATION_COLUMNS_MINIMAL = {"individual", "alpha_observed"}
REQUIRED_VALIDATION_COLUMNS_COMPONENTS = {"individual", "p_local", "p_anatolian"}

DEFAULT_BASE = Path.cwd()
DEFAULT_OUTDIR = Path("results") / "alpha_identifiability"

DEFAULT_N_SIM = 100_000
DEFAULT_SEED = 20240421
DEFAULT_RHO_GRID = [0.0, -0.25, -0.50, -0.75]
DEFAULT_MIN_VALID_FRACTION = 0.65
DEFAULT_MAX_BATCH_MULTIPLIER = 25
DEFAULT_BATCH_SIZE = 50_000
DEFAULT_EPS = 1e-12


EMBEDDED_FOCAL_ROWS = [
    {
        "individual": "I15541",
        "site": "Timacum Kuline",
        "date": "800-1000 CE",
        "model_axis": "Croatia_IA + WestAnatolia_Roman_Byzantine + CEE_EarlyMedieval",
        "p_local": 0.122121647,
        "se_local": 0.154,
        "p_anatolian": 0.339512405,
        "se_anatolian": 0.084,
        "p_eastern": 0.538365948,
        "se_eastern": 0.090,
        "p_value": 0.370690746,
        "source": "Olalde et al. Data S2 Table 6; three-way model block, columns Prop1/SE1/Prop2/SE2/Prop3/SE3/Pval.",
        "notes": "Prop1=Croatia_IA; Prop2=WestAnatolia_Roman_Byzantine; Prop3=CEE_EarlyMedieval.",
    },
    {
        "individual": "I15542",
        "site": "Timacum Kuline",
        "date": "897-1021 cal CE",
        "model_axis": "Croatia_IA + WestAnatolia_Roman_Byzantine + CEE_EarlyMedieval",
        "p_local": 0.262317715,
        "se_local": 0.130,
        "p_anatolian": 0.192686361,
        "se_anatolian": 0.074,
        "p_eastern": 0.544995924,
        "se_eastern": 0.081,
        "p_value": 0.414848682,
        "source": "Olalde et al. Data S2 Table 6; three-way model block, columns Prop1/SE1/Prop2/SE2/Prop3/SE3/Pval.",
        "notes": "Prop1=Croatia_IA; Prop2=WestAnatolia_Roman_Byzantine; Prop3=CEE_EarlyMedieval.",
    },
]

EMBEDDED_COMPARATOR_ROWS = [
    {
        "individual": "I26897",
        "site": "Gardun",
        "date": "400-600 CE",
        "model_axis": "Croatia_IA + WestAnatolia_Roman_Byzantine",
        "axis_type": "same-axis",
        "p_local": 0.414358080,
        "p_anatolian": 0.585641920,
        "alpha_observed": 0.414358080,
        "se": 0.058,
        "p_value": 0.104610673,
        "source": "Olalde et al. Data S2 Table 6; two-way Croatia_IA + WestAnatolia_Roman_Byzantine model.",
        "notes": "Same-axis comparator for the focal alpha definition.",
    },
    {
        "individual": "I15546",
        "site": "Timacum Slog",
        "date": "380-410 CE",
        "model_axis": "Croatia_IA + Aegean_BA_IA",
        "axis_type": "related-axis",
        "p_local": 0.415233529,
        "p_anatolian": 0.584766471,
        "alpha_observed": 0.415233529,
        "se": 0.071,
        "p_value": 0.234582147,
        "source": "Olalde et al. Data S2 Table 6; two-way Croatia_IA + Aegean_BA_IA model.",
        "notes": "Related-axis comparator; non-eastern second source differs from focal WestAnatolia_Roman_Byzantine axis.",
    },
    {
        "individual": "I15544",
        "site": "Timacum Slog",
        "date": "261-418 cal CE",
        "model_axis": "Croatia_IA + Aegean_BA_IA",
        "axis_type": "related-axis",
        "p_local": 0.562622189,
        "p_anatolian": 0.437377811,
        "alpha_observed": 0.562622189,
        "se": 0.072,
        "p_value": 0.493955594,
        "source": "Olalde et al. Data S2 Table 6; two-way Croatia_IA + Aegean_BA_IA model.",
        "notes": "Related-axis comparator; non-eastern second source differs from focal WestAnatolia_Roman_Byzantine axis.",
    },
    {
        "individual": "I15555",
        "site": "Timacum Slog",
        "date": "410-450 CE",
        "model_axis": "Croatia_IA + Aegean_BA_IA",
        "axis_type": "related-axis",
        "p_local": 0.256623384,
        "p_anatolian": 0.743376616,
        "alpha_observed": 0.256623384,
        "se": 0.076,
        "p_value": 0.178757621,
        "source": "Olalde et al. Data S2 Table 6; two-way Croatia_IA + Aegean_BA_IA model.",
        "notes": "Related-axis comparator; non-eastern second source differs from focal WestAnatolia_Roman_Byzantine axis.",
    },
    {
        "individual": "I26703",
        "site": "Trogir",
        "date": "1-200 CE",
        "model_axis": "Croatia_IA + Aegean_BA_IA",
        "axis_type": "related-axis",
        "p_local": 0.789019916,
        "p_anatolian": 0.210980084,
        "alpha_observed": 0.789019916,
        "se": 0.071,
        "p_value": 0.0933642783,
        "source": "Olalde et al. Data S2 Table 6; two-way Croatia_IA + Aegean_BA_IA model.",
        "notes": "Earlier Roman-period related-axis comparator.",
    },
    {
        "individual": "I26708",
        "site": "Trogir",
        "date": "1-200 CE",
        "model_axis": "Croatia_IA + Aegean_BA_IA",
        "axis_type": "related-axis",
        "p_local": 0.370468972,
        "p_anatolian": 0.629531028,
        "alpha_observed": 0.370468972,
        "se": 0.068,
        "p_value": 0.169719176,
        "source": "Olalde et al. Data S2 Table 6; two-way Croatia_IA + Aegean_BA_IA model.",
        "notes": "Earlier Roman-period related-axis comparator.",
    },
]


@dataclass(frozen=True)
class SimulationConfig:
    n_sim: int
    seed: int
    rho_grid: List[float]
    min_valid_fraction: float
    max_batch_multiplier: int
    batch_size: int
    eps: float
    save_draws: bool


@dataclass
class SimulationResult:
    individual: str
    rho: float
    n_requested: int
    n_generated: int
    n_valid: int
    n_valid_raw: int
    n_valid_retained: int
    valid_fraction: float
    valid_fraction_raw: float
    valid_fraction_retained: float
    rejection_fraction: float
    mean_alpha: float
    median_alpha: float
    sd_alpha: float
    ci_lower_95: float
    ci_upper_95: float
    ci_width: float
    local_mean_valid: float
    anatolian_mean_valid: float
    draws: Optional[pd.DataFrame]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run alpha identifiability and Local-Anatolian correlation-sensitivity "
            "analysis for published focal admixture estimates."
        )
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
        help="Project base directory. Default: current working directory.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Optional CSV containing focal admixture means and standard errors. "
            "If omitted, embedded Olalde-derived focal inputs are used."
        ),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=None,
        help=(
            "Optional CSV containing same-axis and related-axis comparator alpha values. "
            "If omitted, embedded Olalde-derived comparator inputs are used."
        ),
    )
    parser.add_argument(
        "--no-comparators",
        action="store_true",
        help="Do not use embedded or external comparator values.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help=(
            "Directory for numeric outputs. If omitted, "
            f"'<base>/{DEFAULT_OUTDIR.as_posix()}' is used."
        ),
    )
    parser.add_argument(
        "--n-sim",
        type=int,
        default=DEFAULT_N_SIM,
        help=f"Number of valid draws requested per individual per rho. Default: {DEFAULT_N_SIM}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base random seed. Default: {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--rho-grid",
        type=str,
        default=",".join(str(x) for x in DEFAULT_RHO_GRID),
        help=(
            "Comma-separated Local-Anatolian correlation grid. "
            f"Default: {','.join(str(x) for x in DEFAULT_RHO_GRID)}."
        ),
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=DEFAULT_MIN_VALID_FRACTION,
        help=(
            "Warn if the retained valid-draw fraction falls below this value. "
            f"Default: {DEFAULT_MIN_VALID_FRACTION}."
        ),
    )
    parser.add_argument(
        "--max-batch-multiplier",
        type=int,
        default=DEFAULT_MAX_BATCH_MULTIPLIER,
        help=(
            "Maximum generated raw draws as a multiple of requested valid draws. "
            f"Default: {DEFAULT_MAX_BATCH_MULTIPLIER}."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Raw draw batch size for rejection sampling. Default: {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--save-draws",
        action="store_true",
        help="Save raw valid Monte Carlo draws to CSV.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_paths(args: argparse.Namespace) -> tuple[Optional[Path], Optional[Path], Path]:
    base = args.base.resolve()
    input_path = args.input.resolve() if args.input else None

    if args.no_comparators:
        validation_path: Optional[Path] = None
    elif args.validation is not None:
        validation_path = args.validation.resolve()
    else:
        validation_path = None

    outdir = args.outdir.resolve() if args.outdir else base / DEFAULT_OUTDIR
    return input_path, validation_path, outdir


def sha256_file(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relpath(base: Path, path: Optional[Path | str]) -> Optional[str]:
    if path is None:
        return None
    try:
        return Path(path).resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return str(path)


def parse_rho_grid(raw: str) -> List[float]:
    values: List[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        val = float(item)
        if val < -0.999 or val > 0.999:
            raise ValueError(f"Correlation rho must be between -0.999 and 0.999; got {val}")
        values.append(val)
    if not values:
        raise ValueError("rho grid is empty")
    return values


def normalize_axis_type(value: object) -> str:
    raw = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "same-axis": "same-axis",
        "sameaxis": "same-axis",
        "direct": "same-axis",
        "same": "same-axis",
        "related-axis": "related-axis",
        "relatedaxis": "related-axis",
        "contextual": "related-axis",
        "related": "related-axis",
        "unspecified": "unspecified",
        "": "unspecified",
        "nan": "unspecified",
    }
    return aliases.get(raw, raw)


def compute_bounded_ci(alpha: float, se: float) -> tuple[float, float]:
    lower = max(0.0, float(alpha) - 1.96 * float(se))
    upper = min(1.0, float(alpha) + 1.96 * float(se))
    return lower, upper


def embedded_focal_table() -> pd.DataFrame:
    """Return the embedded focal qpAdm values used by default."""
    return pd.DataFrame(EMBEDDED_FOCAL_ROWS)


def embedded_comparator_table() -> pd.DataFrame:
    """Return the embedded Roman-period and late-antique comparator values used by default."""
    df = pd.DataFrame(EMBEDDED_COMPARATOR_ROWS)
    ci = df.apply(lambda row: compute_bounded_ci(row["alpha_observed"], row["se"]), axis=1)
    df["ci_lower_95"] = [x[0] for x in ci]
    df["ci_upper_95"] = [x[1] for x in ci]
    df["axis_type_normalized"] = df["axis_type"].map(normalize_axis_type)
    return df


def validate_focal_table(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    missing = REQUIRED_INPUT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{source_label} focal table missing required columns: {', '.join(sorted(missing))}")

    if df.empty:
        raise ValueError(f"{source_label} focal table contains no focal individuals")

    numeric_cols = ["p_local", "se_local", "p_anatolian", "se_anatolian", "p_eastern"]
    optional_numeric = ["se_eastern", "p_value"]
    for col in numeric_cols + [c for c in optional_numeric if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    for row in df.itertuples(index=False):
        if not (0 <= row.p_local <= 1):
            raise ValueError(f"{row.individual}: p_local out of [0,1] range")
        if not (0 <= row.p_anatolian <= 1):
            raise ValueError(f"{row.individual}: p_anatolian out of [0,1] range")
        if not (0 <= row.p_eastern <= 1):
            raise ValueError(f"{row.individual}: p_eastern out of [0,1] range")
        if row.se_local < 0 or row.se_anatolian < 0:
            raise ValueError(f"{row.individual}: standard errors must be non-negative")
        total = row.p_local + row.p_anatolian + row.p_eastern
        if not np.isfinite(total) or abs(total - 1.0) > 0.02:
            raise ValueError(
                f"{row.individual}: focal proportions do not sum approximately to 1 "
                f"(sum={total:.6f})"
            )

    if df["individual"].duplicated().any():
        dupes = df.loc[df["individual"].duplicated(), "individual"].tolist()
        raise ValueError(f"Duplicate individual IDs in {source_label} focal table: {dupes}")

    return df


def load_input_table(path: Optional[Path]) -> tuple[pd.DataFrame, str]:
    if path is None:
        df = embedded_focal_table()
        return validate_focal_table(df, "embedded"), "embedded"

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    return validate_focal_table(df, str(path)), "external"


def validate_comparator_table(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"{source_label} comparator table is empty")

    cols = set(df.columns)
    if REQUIRED_VALIDATION_COLUMNS_MINIMAL.issubset(cols):
        df["alpha_observed"] = pd.to_numeric(df["alpha_observed"], errors="raise")
    elif REQUIRED_VALIDATION_COLUMNS_COMPONENTS.issubset(cols):
        df["p_local"] = pd.to_numeric(df["p_local"], errors="raise")
        df["p_anatolian"] = pd.to_numeric(df["p_anatolian"], errors="raise")
        denom = df["p_local"] + df["p_anatolian"]
        if (denom <= 0).any():
            raise ValueError(f"{source_label} comparator table has non-positive p_local + p_anatolian denominator")
        df["alpha_observed"] = df["p_local"] / denom
    else:
        raise ValueError(
            "Comparator table must contain either 'alpha_observed' or both "
            "'p_local' and 'p_anatolian'"
        )

    df["alpha_observed"] = pd.to_numeric(df["alpha_observed"], errors="raise")
    if ((df["alpha_observed"] < 0) | (df["alpha_observed"] > 1)).any():
        bad = df.loc[(df["alpha_observed"] < 0) | (df["alpha_observed"] > 1), "individual"].tolist()
        raise ValueError(f"Comparator alpha_observed outside [0,1] for: {bad}")

    if "axis_type" not in df.columns:
        df["axis_type"] = "unspecified"
    df["axis_type_normalized"] = df["axis_type"].map(normalize_axis_type)

    if "se" in df.columns:
        df["se"] = pd.to_numeric(df["se"], errors="raise")
        if "ci_lower_95" not in df.columns or "ci_upper_95" not in df.columns:
            ci = df.apply(lambda row: compute_bounded_ci(row["alpha_observed"], row["se"]), axis=1)
            df["ci_lower_95"] = [x[0] for x in ci]
            df["ci_upper_95"] = [x[1] for x in ci]

    recognized = {"same-axis", "related-axis", "unspecified"}
    unexpected = sorted(set(df["axis_type_normalized"]) - recognized)
    if unexpected:
        print(
            "[WARN] Comparator table contains unrecognized axis_type values after normalization: "
            + ", ".join(unexpected)
        )

    return df


def load_validation_table(path: Optional[Path], use_embedded: bool = True) -> tuple[Optional[pd.DataFrame], str]:
    if path is None:
        if not use_embedded:
            return None, "none"
        df = embedded_comparator_table()
        return validate_comparator_table(df, "embedded"), "embedded"

    if not path.exists():
        raise FileNotFoundError(f"Validation/comparator file not found: {path}")

    df = pd.read_csv(path)
    return validate_comparator_table(df, str(path)), "external"


def deterministic_seed(base_seed: int, individual: str, rho: float) -> int:
    """Derive a deterministic 32-bit seed from base seed, individual, and rho."""
    key = f"{base_seed}|{individual}|{rho:.6f}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    offset = int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32 - 1)
    return (base_seed + offset) % (2**32 - 1)


def make_covariance_matrix(se_local: float, se_anatolian: float, rho: float) -> np.ndarray:
    cov = np.array(
        [
            [se_local ** 2, rho * se_local * se_anatolian],
            [rho * se_local * se_anatolian, se_anatolian ** 2],
        ],
        dtype=float,
    )
    eigvals = np.linalg.eigvalsh(cov)
    if np.min(eigvals) < -1e-12:
        raise ValueError(f"Covariance matrix is not positive semidefinite for rho={rho}")
    return cov


def draw_bivariate_candidates(
    mean_local: float,
    mean_anatolian: float,
    se_local: float,
    se_anatolian: float,
    rho: float,
    n_draw: int,
    rng: np.random.Generator,
) -> np.ndarray:
    cov = make_covariance_matrix(se_local, se_anatolian, rho)
    mean = np.array([mean_local, mean_anatolian], dtype=float)
    draws = rng.multivariate_normal(mean=mean, cov=cov, size=n_draw, check_valid="raise")
    return draws


def simulate_alpha_distribution(row: pd.Series, rho: float, config: SimulationConfig) -> SimulationResult:
    valid_local: List[float] = []
    valid_anatolian: List[float] = []
    n_generated = 0
    max_generated = config.n_sim * config.max_batch_multiplier

    seed = deterministic_seed(config.seed, str(row["individual"]), float(rho))
    rng = np.random.default_rng(seed)

    while len(valid_local) < config.n_sim and n_generated < max_generated:
        remaining = config.n_sim - len(valid_local)
        batch_n = min(max(config.batch_size, remaining), max_generated - n_generated)
        draws = draw_bivariate_candidates(
            mean_local=float(row["p_local"]),
            mean_anatolian=float(row["p_anatolian"]),
            se_local=float(row["se_local"]),
            se_anatolian=float(row["se_anatolian"]),
            rho=rho,
            n_draw=batch_n,
            rng=rng,
        )
        n_generated += batch_n

        local = draws[:, 0]
        anatolian = draws[:, 1]
        valid_mask = (
            np.isfinite(local)
            & np.isfinite(anatolian)
            & (local >= 0.0)
            & (anatolian >= 0.0)
            & (local <= 1.0)
            & (anatolian <= 1.0)
            & ((local + anatolian) <= 1.0 + config.eps)
            & ((local + anatolian) > config.eps)
        )

        if np.any(valid_mask):
            valid_local.extend(local[valid_mask].tolist())
            valid_anatolian.extend(anatolian[valid_mask].tolist())

    if len(valid_local) < config.n_sim:
        raise RuntimeError(
            f"Only {len(valid_local)} valid draws retained for {row['individual']} at rho={rho}; "
            f"requested {config.n_sim}. Increase max-batch-multiplier or check input values."
        )

    n_valid_raw = len(valid_local)
    valid_local_arr = np.asarray(valid_local[: config.n_sim], dtype=float)
    valid_anat_arr = np.asarray(valid_anatolian[: config.n_sim], dtype=float)
    alpha = valid_local_arr / (valid_local_arr + valid_anat_arr)

    n_valid_retained = len(alpha)
    valid_fraction_raw = n_valid_raw / n_generated if n_generated > 0 else 0.0
    valid_fraction_retained = n_valid_retained / n_generated if n_generated > 0 else 0.0
    rejection_fraction = 1.0 - valid_fraction_raw

    draws_df = None
    if config.save_draws:
        draws_df = pd.DataFrame(
            {
                "individual": row["individual"],
                "rho": rho,
                "draw_id": np.arange(1, n_valid_retained + 1),
                "local_draw": valid_local_arr,
                "anatolian_draw": valid_anat_arr,
                "alpha_draw": alpha,
            }
        )

    return SimulationResult(
        individual=str(row["individual"]),
        rho=float(rho),
        n_requested=int(config.n_sim),
        n_generated=int(n_generated),
        n_valid=int(n_valid_retained),
        n_valid_raw=int(n_valid_raw),
        n_valid_retained=int(n_valid_retained),
        valid_fraction=float(valid_fraction_raw),
        valid_fraction_raw=float(valid_fraction_raw),
        valid_fraction_retained=float(valid_fraction_retained),
        rejection_fraction=float(rejection_fraction),
        mean_alpha=float(np.mean(alpha)),
        median_alpha=float(np.median(alpha)),
        sd_alpha=float(np.std(alpha, ddof=1)),
        ci_lower_95=float(np.quantile(alpha, 0.025)),
        ci_upper_95=float(np.quantile(alpha, 0.975)),
        ci_width=float(np.quantile(alpha, 0.975) - np.quantile(alpha, 0.025)),
        local_mean_valid=float(np.mean(valid_local_arr)),
        anatolian_mean_valid=float(np.mean(valid_anat_arr)),
        draws=draws_df,
    )


def result_to_dict(res: SimulationResult, rho_order: int) -> dict:
    return {
        "individual": res.individual,
        "rho": res.rho,
        "rho_order": rho_order,
        "n_requested": res.n_requested,
        "n_generated": res.n_generated,
        "n_valid": res.n_valid,
        "n_valid_raw": res.n_valid_raw,
        "n_valid_retained": res.n_valid_retained,
        "valid_fraction": res.valid_fraction,
        "valid_fraction_raw": res.valid_fraction_raw,
        "valid_fraction_retained": res.valid_fraction_retained,
        "rejection_fraction": res.rejection_fraction,
        "mean_alpha": res.mean_alpha,
        "median_alpha": res.median_alpha,
        "sd_alpha": res.sd_alpha,
        "ci_lower_95": res.ci_lower_95,
        "ci_upper_95": res.ci_upper_95,
        "ci_width": res.ci_width,
        "local_mean_valid": res.local_mean_valid,
        "anatolian_mean_valid": res.anatolian_mean_valid,
    }


def write_metadata_json(
    path: Path,
    base: Path,
    config: SimulationConfig,
    input_file: Optional[Path],
    validation_file: Optional[Path],
    focal_input_source: str,
    comparator_input_source: str,
    outdir: Path,
    summary_df: pd.DataFrame,
    warnings: List[str],
    output_files: dict,
) -> None:
    metadata = {
        "script": "04_identifiability_analysis.py",
        "timestamp_utc": utc_now(),
        "purpose": (
            "Bounded Monte Carlo uncertainty-propagation and correlation-sensitivity "
            "analysis for the local-to-Anatolian composition of the modeled non-eastern ancestry component."
        ),
        "focal_input_source": relpath(base, input_file) if input_file is not None else focal_input_source,
        "comparator_input_source": relpath(base, validation_file) if validation_file is not None else comparator_input_source,
        "input_file": relpath(base, input_file),
        "input_file_sha256": sha256_file(input_file),
        "validation_file": relpath(base, validation_file),
        "validation_file_sha256": sha256_file(validation_file),
        "outdir": relpath(base, outdir),
        "n_sim": config.n_sim,
        "seed": config.seed,
        "rho_grid": config.rho_grid,
        "min_valid_fraction": config.min_valid_fraction,
        "max_batch_multiplier": config.max_batch_multiplier,
        "batch_size": config.batch_size,
        "eps": config.eps,
        "save_draws": config.save_draws,
        "assumptions": [
            "Primary estimand is alpha = Local / (Local + Anatolian).",
            "Eastern-associated ancestry is not part of the alpha denominator because alpha is conditional on the modeled non-eastern component.",
            "Local and Anatolian uncertainty are evaluated under a sensitivity grid of assumed correlations.",
            "Draws are rejected if they violate compositional bounds or yield a non-positive Local + Anatolian denominator.",
            "Results are a sensitivity analysis, not a qpAdm replacement or a full posterior reconstruction.",
            "Figures are generated separately by 05_make_figures.py from the numeric outputs of this script.",
        ],
        "individuals": summary_df["individual"].drop_duplicates().tolist(),
        "warnings": warnings,
        "output_files": {key: relpath(base, value) for key, value in output_files.items()},
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def run_analysis(args: argparse.Namespace) -> int:
    input_path, validation_path, outdir = resolve_paths(args)
    rho_grid = parse_rho_grid(args.rho_grid)

    if args.n_sim <= 0:
        raise ValueError("--n-sim must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_batch_multiplier <= 0:
        raise ValueError("--max-batch-multiplier must be positive")

    config = SimulationConfig(
        n_sim=args.n_sim,
        seed=args.seed,
        rho_grid=rho_grid,
        min_valid_fraction=args.min_valid_fraction,
        max_batch_multiplier=args.max_batch_multiplier,
        batch_size=args.batch_size,
        eps=DEFAULT_EPS,
        save_draws=args.save_draws,
    )

    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Alpha identifiability analysis")
    print(f"Focal input source: {input_path if input_path is not None else 'embedded Olalde-derived table'}")
    if args.no_comparators:
        print("Comparator source: none (--no-comparators)")
    else:
        print(f"Comparator source: {validation_path if validation_path is not None else 'embedded Olalde-derived table'}")
    print(f"Output directory: {outdir}")
    print(f"n_sim: {config.n_sim}")
    print(f"rho_grid: {config.rho_grid}")
    print(f"seed: {config.seed}")
    print("=" * 80)

    focal_df, focal_input_source = load_input_table(input_path)
    validation_df, comparator_input_source = load_validation_table(
        validation_path, use_embedded=not args.no_comparators
    )

    focal_inputs_path = outdir / "focal_inputs_used.csv"
    focal_df.to_csv(focal_inputs_path, index=False)

    print("\nFocal inputs used:")
    focal_display_cols = [
        "individual", "site", "date", "p_local", "se_local",
        "p_anatolian", "se_anatolian", "p_eastern", "p_value",
    ]
    print(focal_df[[c for c in focal_display_cols if c in focal_df.columns]].to_string(index=False))

    if validation_df is not None:
        print("\nComparator inputs used:")
        comp_display_cols = [
            "individual", "site", "date", "axis_type_normalized", "model_axis",
            "alpha_observed", "se", "ci_lower_95", "ci_upper_95",
        ]
        print(validation_df[[c for c in comp_display_cols if c in validation_df.columns]].to_string(index=False))
        comparator_used_path = outdir / "comparators_used.csv"
        comparator_path = outdir / "comparators_normalized.csv"
        validation_df.to_csv(comparator_used_path, index=False)
        validation_df.to_csv(comparator_path, index=False)
    else:
        comparator_used_path = None
        comparator_path = None

    all_results: List[SimulationResult] = []
    raw_draws_parts: List[pd.DataFrame] = []
    warnings: List[str] = []

    for row in focal_df.to_dict(orient="records"):
        row_series = pd.Series(row)
        for rho in config.rho_grid:
            print(f"[RUN] {row_series['individual']} | rho={rho}")
            res = simulate_alpha_distribution(row_series, rho, config)
            all_results.append(res)
            if config.save_draws and res.draws is not None:
                raw_draws_parts.append(res.draws)
            if res.valid_fraction < config.min_valid_fraction:
                warnings.append(
                    f"Low valid fraction for {res.individual} at rho={rho}: {res.valid_fraction:.3f}"
                )

    rho_to_order = {rho: idx for idx, rho in enumerate(config.rho_grid)}
    summary_df = pd.DataFrame(
        [result_to_dict(res, rho_to_order[res.rho]) for res in all_results]
    ).sort_values(["individual", "rho_order"])

    summary_path = outdir / "alpha_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    draws_path = None
    if config.save_draws and raw_draws_parts:
        draws_df = pd.concat(raw_draws_parts, ignore_index=True)
        draws_path = outdir / "alpha_draws.csv"
        draws_df.to_csv(draws_path, index=False)

    metadata_path = outdir / "alpha_run_metadata.json"
    output_files = {
        "focal_inputs_used_csv": str(focal_inputs_path),
        "comparators_used_csv": str(comparator_used_path) if comparator_used_path is not None else None,
        "summary_csv": str(summary_path),
        "draws_csv": str(draws_path) if draws_path is not None else None,
        "metadata_json": str(metadata_path),
        "comparators_normalized_csv": str(comparator_path) if comparator_path is not None else None,
    }
    write_metadata_json(
        path=metadata_path,
        base=args.base.resolve(),
        config=config,
        input_file=input_path,
        validation_file=validation_path,
        focal_input_source=focal_input_source,
        comparator_input_source=comparator_input_source,
        outdir=outdir,
        summary_df=summary_df,
        warnings=warnings,
        output_files=output_files,
    )

    print("\n" + "=" * 80)
    print("[OK] Alpha identifiability analysis complete")
    print(f"Focal inputs used CSV: {focal_inputs_path}")
    if comparator_used_path is not None:
        print(f"Comparators used CSV: {comparator_used_path}")
    print(f"Summary CSV: {summary_path}")
    if draws_path is not None:
        print(f"Raw draws CSV: {draws_path}")
    if comparator_path is not None:
        print(f"Normalized comparator CSV: {comparator_path}")
    print(f"Metadata JSON: {metadata_path}")

    print("\nSummary:")
    display_cols = [
        "individual",
        "rho",
        "n_valid",
        "n_generated",
        "valid_fraction",
        "median_alpha",
        "ci_lower_95",
        "ci_upper_95",
    ]
    print(summary_df[display_cols].to_string(index=False))

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    print("=" * 80)
    return 0


def main() -> int:
    try:
        args = parse_args()
        return run_analysis(args)
    except Exception as exc:
        print(f"[ERROR] Alpha identifiability analysis failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
