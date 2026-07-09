# Reproducibility

This repository contains the scripts, derived input tables, generated outputs, figures, and metadata required to reproduce the analyses for:

**A reproducible audit of qpAdm coefficient uncertainty and complete-case allele-sharing diagnostics in medieval Balkan ancient DNA**

## Version

This repository contains version `v1.0.0` of the reproducible analysis.

Repository URL: https://github.com/nstanisic1/qpadm-coefficient-uncertainty-audit
GitHub release: https://github.com/nstanisic1/qpadm-coefficient-uncertainty-audit/releases/tag/v1.0.0
Archived release DOI: https://doi.org/10.5281/zenodo.21276853
Zenodo concept DOI: https://doi.org/10.5281/zenodo.21276852

## Inputs

- `PRJEB66422_read_run.tsv` provides ENA metadata for BAM/BAI acquisition.
- `v62.0_1240k_public.snp` provides public 1240k autosomal target coordinates.
- `data/olalde_table6_focal.csv` and `data/late_antique_validation.csv` provide small derived tables used by the coefficient-layer audit.

Large ENA BAM/BAI files, manuscript/submission files, and full third-party supplementary files are not redistributed in this repository.

## Environment

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

Required packages are `numpy`, `pandas`, `scipy`, `matplotlib`, and `bamnostic`.

For a pinned environment, use `requirements.lock.txt` with Python 3.10.11 or create the Conda environment from `environment.yml`.

## Reproduction Workflow

Run the pipeline from the repository root:

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

On Windows PowerShell, replace the `mkdir/cp` lines with:

```powershell
New-Item -ItemType Directory -Force .\matrix_ev13_sensitivity
Copy-Item .\matrix_main\core_7_sample_matrix.tsv .\matrix_ev13_sensitivity\core_7_sample_matrix.tsv -Force
```

## Validation

Each analysis step writes machine-readable metadata beside its outputs. Metadata paths are repository-relative. The matrix summaries must report `snp_scan_mode` as `all_autosomal` and `n_autosomes_represented` as `22`.

To validate the scripts, run:

```bash
python -m py_compile 01_download_bams.py 02_build_matrix.py 03_allele_sharing_panel.py 04_identifiability_analysis.py 04a_alpha_threshold_sensitivity.py 05_make_figures.py 05a_make_alpha_threshold_figure.py 06_complete_case_distance_audit.py
```
