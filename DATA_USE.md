# Data Use

Large primary BAM/BAI files are downloaded from ENA and are not redistributed. Third-party Olalde supplementary files are not redistributed. The repository includes only derived non-restricted input tables, scripts, generated outputs, figures, run metadata, manifests, and documentation needed to reproduce the associated analysis.

Version `v1.0.0` corresponds to the reproducible-analysis repository snapshot at https://github.com/nstanisic1/qpadm-coefficient-uncertainty-audit.

The pipeline uses published ancient genomic data and public metadata for secondary analysis. Users should consult the original data providers and archives for primary-data access terms, sample metadata, and citation requirements.

## 1240k coordinate resource

The repository bundles `v62.0_1240k_public.snp`, the public AADR/1240k SNP coordinate resource used by `02_build_matrix.py` to define target autosomal positions. The bundled file fixes the coordinate version used for this audit and has SHA256 checksum `1189e253b82058822fddabc4966c1448dea70373a96c40106e720bedd93b55bf`.

The coordinate resource is redistributed here only as the fixed public target-position list required for reproducibility. Primary sequence data are not redistributed. Users should follow the access and reuse terms of ENA, AADR/source resources, and the original publications.

## Generated audit outputs

Newly generated outputs from this audit include the complete-case matrices in `matrix_main/` and `matrix_ev13_sensitivity/`, allele-sharing summaries, coefficient-layer alpha outputs, alpha-threshold sensitivity outputs, distance-space audit outputs, figures, run metadata, checksum manifests, and reproducibility documentation.
