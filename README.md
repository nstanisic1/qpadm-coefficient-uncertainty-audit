# A reproducible audit of qpAdm coefficient uncertainty and complete-case allele-sharing diagnostics

This repository contains the reproducible Python pipeline and derived outputs for the manuscript:

**A reproducible audit of qpAdm coefficient uncertainty and complete-case allele-sharing diagnostics in medieval Balkan ancient DNA**

The repository supports a secondary analysis of published ancient genomic data. It evaluates whether published qpAdm coefficient summaries and a strict complete-case allele-sharing panel leave the local/Anatolian source axis under-resolved in selected medieval Balkan individuals. The repository does not redistribute large primary BAM/BAI files; these are downloaded from ENA by the pipeline.

## Version

This repository contains version `v1.0.0` of the reproducible analysis.

Repository URL: https://github.com/nstanisic1/qpadm-coefficient-uncertainty-audit
GitHub release: https://github.com/nstanisic1/qpadm-coefficient-uncertainty-audit/releases/tag/v1.0.0
Archived release DOI: https://doi.org/10.5281/zenodo.21276853
Zenodo concept DOI: https://doi.org/10.5281/zenodo.21276852

## Pipeline overview

The reproducible workflow consists of eight Python stages:

1. `01_download_bams.py` - downloads and validates ENA BAM/BAI files, with restart/resume support where supported by the remote server.
2. `02_build_matrix.py` - builds strict complete-case pseudo-haploid 1240k matrices by scanning all autosomal 1240k positions by default.
3. `03_allele_sharing_panel.py` - runs four pre-specified independent allele-sharing contrasts.
4. `04_identifiability_analysis.py` - runs bounded Monte Carlo coefficient-layer alpha uncertainty analysis.
5. `04a_alpha_threshold_sensitivity.py` - tests increasingly strict local-alpha thresholds and directional-support floors.
6. `05_make_figures.py` - generates Figure 1 and Supplementary Figures S1-S3 from saved numeric outputs.
7. `05a_make_alpha_threshold_figure.py` - generates Supplementary Figure S4 from alpha-threshold sensitivity outputs.
8. `06_complete_case_distance_audit.py` - runs the complete-case distance-space concordance audit and renders Figure 2.

## Requirements

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

Required packages: numpy, pandas, scipy, matplotlib, bamnostic.

For a pinned environment, use `requirements.lock.txt` with Python 3.10.11 or create the Conda environment from `environment.yml`.

## Required input files

The repository includes:

- `PRJEB66422_read_run.tsv`: ENA read/run metadata used by `01_download_bams.py`.
- `v62.0_1240k_public.snp`: public AADR/1240k SNP coordinate resource, version v62.0, used by `02_build_matrix.py`.
- `data/olalde_table6_focal.csv` and `data/late_antique_validation.csv`: small extracted, non-restricted input/reference tables used for auditability and reproducibility.

Large BAM/BAI files are downloaded by `01_download_bams.py` into `ena_1240k_all/`, which is ignored by Git. Full third-party supplement files and manuscript/submission files are not redistributed in this repository.

## Third-party data and redistribution

The bundled `v62.0_1240k_public.snp` file is the public AADR/1240k SNP coordinate resource used to define target positions for complete-case matrix construction. It is included in this release to make the coordinate set fixed and auditable. Its SHA256 checksum is `1189e253b82058822fddabc4966c1448dea70373a96c40106e720bedd93b55bf`.

Primary BAM/BAI files remain external and are obtained from ENA using the accessions in `PRJEB66422_read_run.tsv`. Full third-party supplementary files from the source publication are not redistributed and should be obtained from the original publication or archive. Users remain responsible for following ENA, AADR/source-resource, and original-publication access and reuse terms.

The newly generated audit outputs in this repository are the complete-case matrices, allele-sharing summaries, coefficient-layer alpha outputs, alpha-threshold sensitivity outputs, distance-space audit outputs, figures, run metadata, manifests, and reproducibility documentation.

## Reproduction Workflow

Run all commands from the repository root:

```bash
python 01_download_bams.py --panel all --base .

python 02_build_matrix.py --panel main --base . --budget all --workers 1 --force
mkdir -p matrix_ev13_sensitivity
cp -f matrix_main/core_7_sample_matrix.tsv matrix_ev13_sensitivity/core_7_sample_matrix.tsv
python 02_build_matrix.py --panel ev13_sensitivity --base . --budget all --workers 1 --force

python 03_allele_sharing_panel.py --panel main --base . --seed 42 --n-bootstrap 1000
python 03_allele_sharing_panel.py --panel ev13_sensitivity --base . --seed 42 --n-bootstrap 1000

python 04_identifiability_analysis.py --base . --input data/olalde_table6_focal.csv --validation data/late_antique_validation.csv --seed 20240421 --n-sim 100000 --rho-grid 0,-0.25,-0.5,-0.75 --save-draws
python 04a_alpha_threshold_sensitivity.py --base .

python 05_make_figures.py --base .
python 05a_make_alpha_threshold_figure.py --base .
python 06_complete_case_distance_audit.py --base . --n-bootstrap 1000 --seed 42
```

On Windows PowerShell, replace the `mkdir/cp` step with:

```powershell
New-Item -ItemType Directory -Force .\matrix_ev13_sensitivity
Copy-Item .\matrix_main\core_7_sample_matrix.tsv .\matrix_ev13_sensitivity\core_7_sample_matrix.tsv -Force
```

## Final output locations

Main matrix and allele-sharing outputs:

- `matrix_main/complete_case_9_sample_matrix.tsv`
- `matrix_main/matrix_build_summary.json`
- `matrix_main/allele_sharing_results.json`
- `matrix_main/allele_sharing_summary.csv`
- `matrix_main/pairwise_distance_matrix.tsv`

E-V13 sensitivity matrix and allele-sharing outputs:

- `matrix_ev13_sensitivity/complete_case_9_sample_matrix.tsv`
- `matrix_ev13_sensitivity/matrix_build_summary.json`
- `matrix_ev13_sensitivity/allele_sharing_results.json`
- `matrix_ev13_sensitivity/allele_sharing_summary.csv`
- `matrix_ev13_sensitivity/pairwise_distance_matrix.tsv`

Alpha uncertainty outputs:

- `results/alpha_identifiability/focal_inputs_used.csv`
- `results/alpha_identifiability/comparators_used.csv`
- `results/alpha_identifiability/comparators_normalized.csv`
- `results/alpha_identifiability/alpha_summary.csv`
- `results/alpha_identifiability/alpha_draws.csv`
- `results/alpha_identifiability/alpha_run_metadata.json`

Alpha-threshold sensitivity outputs:

- `results/alpha_threshold_sensitivity/alpha_threshold_probabilities.csv`
- `results/alpha_threshold_sensitivity/comparator_anchored_thresholds.csv`
- `results/alpha_threshold_sensitivity/directional_support_grid.csv`
- `results/alpha_threshold_sensitivity/alpha_threshold_scenarios.csv`
- `results/alpha_threshold_sensitivity/alpha_threshold_worst_rho_summary.csv`
- `results/alpha_threshold_sensitivity/alpha_threshold_sensitivity_summary.csv`
- `results/alpha_threshold_sensitivity/figuredata_alpha_threshold_sensitivity.csv`
- `results/alpha_threshold_sensitivity/alpha_threshold_sensitivity_run_metadata.json`
- `results/alpha_threshold_sensitivity/alpha_threshold_text_summary.txt`

Distance-space audit outputs:

- `results/distance_space_audit/observed_distance_metrics.csv`
- `results/distance_space_audit/distance_to_r1a_profile.csv`
- `results/distance_space_audit/bootstrap_directional_metrics.csv`
- `results/distance_space_audit/bootstrap_directional_summary.csv`
- `results/distance_space_audit/bootstrap_r1a_profile_summary.csv`
- `results/distance_space_audit/r1a_leaveone_reference_audit.csv`
- `results/distance_space_audit/i2a_partition_enumeration.csv`
- `results/distance_space_audit/projection_diagnostics.csv`
- `results/distance_space_audit/distance_audit_summary.csv`
- `results/distance_space_audit/distance_audit_run_metadata.json`
- `results/distance_space_audit/distance_audit_text_summary.txt`

Figures:

- `figures/figure1_alpha_density.png` and `.pdf`
- `figures/figure2_complete_case_distance_audit.png` and `.pdf`
- `figures/figure2_complete_case_distance_audit_manifest.json`
- `figures/figureS1_alpha_rho_sensitivity.png` and `.pdf`
- `figures/figureS2_allele_sharing_contrasts_main.png` and `.pdf`
- `figures/figureS3_allele_sharing_contrasts_ev13_sensitivity.png` and `.pdf`
- `figures/figureS4_alpha_threshold_sensitivity.png` and `.pdf`
- `figures/figureS4_alpha_threshold_sensitivity_manifest.json`
- `figures/figure_manifest.json`

## Reproducibility notes

`03_allele_sharing_panel.py` uses exact enumeration for the allele-sharing permutation tests and fixed seed 42 by default for SNP-bootstrap intervals. `04_identifiability_analysis.py` uses fixed seed 20240421 by default for Monte Carlo coefficient-layer alpha uncertainty analysis. Output metadata records seeds, parameters, repository-relative paths, and relevant checksums where applicable.

The pipeline treats the focal panel as purposive rather than population-random. Distance-based results are interpreted as directional concordance diagnostics, not population-level parameter estimates.

## Large files and Git

The following files and folders are intentionally ignored by Git:

- downloaded ENA BAM/BAI files in `ena_1240k_all/`
- local manuscript/submission files in `manuscript/`, `submission/`, and `_submission/`
- local private working files excluded by `.gitignore`
- full third-party NIHMS supplementary files
- BAM, BAI, GENO, ANNO, IND, and partial download files

## License

This repository is released under the MIT License. See `LICENSE` for details.
