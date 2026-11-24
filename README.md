# qpAdm coefficient uncertainty and allele-sharing sensitivity in medieval Balkan ancient DNA

This repository reproduces two analyses of published ancient DNA data. The first estimates uncertainty in the Croatia IA share of the modeled non-eastern component for two Timacum Kuline individuals. The second describes four allele-sharing contrasts in two fixed nine-person panels and examines their sensitivity to read filtering, transversion restriction, and genomic block resampling.

## Analysis outputs

| Analysis | Results | Figure |
| --- | --- | --- |
| Study workflow | [Reproduction instructions](REPRODUCIBILITY.md) | [Workflow](figures/figure_analysis_workflow.pdf) |
| Coefficient uncertainty | [Propagation intervals](results/alpha_covariance/alpha_summary.csv) and [implied covariance](results/alpha_covariance/implied_covariance.csv) | [Share by correlation setting](figures/figure_alpha_correlation_intervals.pdf) |
| Read filtering | [Site retention and changed calls](results/quality_control/quality_control_summary.csv) | [Distance sensitivity](figures/figure_distance_sensitivity.pdf) |
| Genomic block resampling | Contrast tables for the [original](results/block_bootstrap/contrast_summary.csv), [filtered](results/quality_control/filtered_block_bootstrap/contrast_summary.csv), and [transversion](results/quality_control/transversions_block_bootstrap/contrast_summary.csv) matrices | [Distance sensitivity](figures/figure_distance_sensitivity.pdf) |

The manuscript's [supplementary CSV tables](results/supplementary_tables/) cover the panel, the same-model Kuline cases, the two-source context, the block resampling estimates, and the Sirmium influence checks.

The [coefficient methods](ALPHA_COVARIANCE.md) describe the covariance calculation. The [allele-sharing methods](DISTANCE_SENSITIVITY.md) describe the read filters and genomic block bootstrap. Run settings and file checksums accompany each set of results.

## Reproduction

This snapshot is version 1.1.0. Follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md) to install the pinned dependencies and reproduce the analyses from the included call matrices. The archive also includes the AADR v62.0 1240k target file and source extracts from Olalde et al. Data S2. See [DATA_USE.md](DATA_USE.md) for source and redistribution details.

## Baseline

[The original SNP-bootstrap summary](baseline/snp_bootstrap_summary.csv) provides the comparison for Figure 3. It retains the column labels and values of the separately published [v1.0.0 snapshot](https://github.com/nstanisic1/qpadm-coefficient-uncertainty-audit/releases/tag/v1.0.0).

The code is available under the [MIT License](LICENSE).
