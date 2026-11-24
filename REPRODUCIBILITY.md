# Reproducibility

Run the commands from the repository root. The environment is specified in `requirements.lock.txt` and `environment.yml`. To use the pinned Python 3.10.11 and package versions, install the dependencies in an isolated environment:

```bash
python -m pip install -r requirements.lock.txt
```

## Inputs

- `PRJEB66422_read_run.tsv` lists the ENA accessions for primary BAM and BAI files.
- `v62.0_1240k_public.snp` provides the 1240k variant coordinates, genetic positions, and allele pairs from AADR v62.0.
- `data/olalde_table6_focal.csv` and `data/olalde_kuline_models.csv` contain the focal and same-site qpAdm model values extracted from Olalde et al. Data S2 Table 6 (BD:BJ). `data/olalde_two_source_fits.csv` contains the reported fits from Table 6 (Q:T) before the screen. `data/panel_sample_provenance.csv` records the sample metadata from Data S2 Table 1 and the ENA accession for each archived alignment.
- `matrix_main/` and `matrix_ev13_sensitivity/` contain the original complete-case matrices for the main and Nustar panels. The original individual-SNP bootstrap summary used for comparison is in `baseline/snp_bootstrap_summary.csv`.

Primary alignment files and complete third-party supplements are supplied by their original archives. See [Data use](DATA_USE.md).

## Run the sensitivity analyses

All commands below can run directly from the included inputs. The original and filtered matrices are supplied, and rebuilding alleles from BAM is not necessary to reproduce the revision tables and figures:

```bash
python 04b_alpha_covariance.py --base .
python 07_block_bootstrap.py --base .
python 07_block_bootstrap.py --base . --matrix-root results/quality_control/filtered --outdir results/quality_control/filtered_block_bootstrap
python 07_block_bootstrap.py --base . --matrix-root results/quality_control/transversions --outdir results/quality_control/transversions_block_bootstrap
python 09_supplementary_tables.py --base .
python 08_make_sensitivity_figures.py --base .
```

To rebuild the filtered matrices from the upstream alignments, use the SAMtools backend that generated the supplied matrices (SAMtools 1.25 with HTSlib 1.25):

```bash
python 01_download_bams.py --panel all --base .
python 02a_quality_control.py --base . --samtools /path/to/samtools
```

Run these commands before the block analyses. The QC script accepts `--bam-dir` for alignments outside `ena_1240k_all/`. If `--samtools` is omitted, the script uses BAMnostic. Both readers skip alignments without a sequence.

The [covariance methods](ALPHA_COVARIANCE.md) and [distance sensitivity methods](DISTANCE_SENSITIVITY.md) describe the calculations, settings, and interpretation.

To check the direction rules and replicate summaries, run `python -m unittest discover -s tests`.

## Original SNP-bootstrap comparison

Figure 3 reads the intervals from `baseline/snp_bootstrap_summary.csv`. The four direction rules were met in 996 of 1,000 replicates in the main panel and 998 of 1,000 in the Nustar panel. The full original workflow, including reconstruction of the original call matrices, is available in the [v1.0.0 snapshot](https://github.com/nstanisic1/qpadm-coefficient-uncertainty-audit/releases/tag/v1.0.0). It is not needed to reproduce the revision outputs from the supplied matrices.

## Verify outputs

`manifest_sha256.csv` lists the SHA-256 digest and byte count for each supplied file other than the manifest itself. Git attributes preserve the recorded bytes across checkout platforms. Run metadata records the inputs, software versions, settings, and output digests. The filtered and transversion matrix counts are in `results/quality_control/quality_control_summary.csv`.
