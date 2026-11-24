# Alpha uncertainty under sum-to-one covariance

This analysis uses the three qpAdm estimates and standard errors in Olalde et al. Data S2, Table 6. Let `L`, `A`, and `E` be the Croatia IA, West Anatolia Roman/Byzantine, and eastern-associated coefficients. If the coefficients sum to one for each estimate, `E = 1 - L - A` and

```text
Cov(L, A) = [Var(E) - Var(L) - Var(A)] / 2
Cov(L, E) = -Var(L) - Cov(L, A)
Cov(A, E) = -Var(A) - Cov(L, A)
```

The input variances are the squares of the published standard errors. The resulting correlations are approximate because the standard errors were rounded. They describe the Gaussian approximation before applying compositional bounds. The covariance reconstructed this way does not recover the original jackknife replicates or their distribution.

The bounded Monte Carlo sampler in `alpha_propagation.py` compares four assumed correlations with the individual-specific sum-to-one correlations, retaining 100,000 draws per individual per setting. Each accepted draw has nonnegative local, Anatolian, and residual eastern fractions within numerical tolerance. Alpha is `L / (L + A)`, the Croatia IA share of the modeled non-eastern fraction. The intervals are the 2.5th and 97.5th percentiles of alpha after applying these bounds.

Run from the repository root with the analysis dependencies installed:

```text
python 04b_alpha_covariance.py --base .
```

The script writes three files to `results/alpha_covariance/`:

- `implied_covariance.csv` gives the variances, covariances, and implied correlation for each individual.
- `alpha_summary.csv` gives the published point ratio, retained-draw count, raw acceptance fraction, median, and 95% propagation interval for each setting.
- `alpha_covariance_metadata.json` records the software versions, simulation settings, and input and output checksums.

The supplementary Kuline table repeats this propagation for the two additional Kuline cases with the same three proxies. The separate two-source table reports all fits in the corresponding source workbook block with coefficients in [0,1] and model p at least 0.05. The two-source estimates are contextual and are not thresholds for the focal ratios. Feasible-draw intervals do not test whether a contribution is exactly zero.
