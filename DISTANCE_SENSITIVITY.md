# Sensitivity of the allele-sharing contrasts

The two panels each contain nine fixed individuals. The four contrasts use pairwise mismatch distances on sites with a call in every individual in a panel. Block resampling assesses sensitivity to the sampled genomic regions while keeping the individuals and contrast definitions fixed.

## Genomic block bootstrap

The analysis uses the genetic position in column 3 of `v62.0_1240k_public.snp`, expressed in Morgans. Each SNP is assigned to a non-overlapping block on its chromosome. The primary block size is 5 cM and the sensitivity analysis uses 10 cM, sizes used for block jackknives in previous ancient-DNA analyses ([Reich et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3188841/), [Patterson et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3522152/)). Here each replicate samples the observed number of nonempty blocks within each chromosome with replacement. It keeps every site in each sampled block and divides the pooled mismatch count by the number of sampled sites, including repetitions. The same block sample is used for all four contrasts within a replicate.

The analysis uses 5,000 replicates for each panel and block size, with base seed 42. It reports percentile intervals and the fraction of replicates with each expected direction and with all four directions. At 5,000 replicates, the maximum Monte Carlo standard error of a reported preservation fraction is about 0.007. This is simulation precision for a fixed panel. It does not describe variation across individuals or populations.

The four exploratory contrasts share individuals and use the same distance matrix. The joint fraction counts the replicates meeting all four direction rules: leave-one-out shift < 0, Sirmium correlation > 0, E-V13 comparator difference > 0, and between-minus-within mismatch difference <= 0. Signs of the three mismatch contrasts are determined from integer mismatch counts before division. The leave-one-out contrast equals minus one third of the difference between the two I2a groups' mean distances to the R1a set. The Sirmium correlation reuses those four individual mean distances. Counts and fractions use the total number of replicates; an undefined contrast does not meet its direction rule.

## Read filtering and transversions

The filtered calls use the original alignments, panel membership, and most frequent base rule. The analysis requires mapping quality at least 30 and base quality at least 20, and ignores two bases at each read end. It excludes unmapped, secondary, supplementary, duplicate-marked, and QC-failed alignments. Missing mapping or base qualities are excluded. A site can be called from a single passing read, with ties resolved by the first observed base. These filters may remove observations and change calls. A site with a call for every individual after filtering must therefore be among the original complete-case sites. The script recomputes the calls at those sites rather than scanning loci that cannot pass.

The transversion subset starts from the filtered calls and retains only sites for which the coordinate file lists a purine and a pyrimidine and every retained call matches one of the two listed alleles. Reported differences from the original distance estimates can reflect changes in the set of retained sites as well as changes in called bases. The QC summary reports both the number of retained sites and the number with changed calls. These filters are applied to the archived alignments, whose upstream processing is described by [Olalde et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10752003/).

## Reproduction

See [Reproducibility](REPRODUCIBILITY.md#run-the-sensitivity-analyses) for the commands and inputs.

The block analysis saves a contrast table, replicate values, and run metadata for each matrix set. The QC stage saves the filtered matrices, site counts, and input and output checksums.

The Sirmium influence table uses four I2a individuals from three sites. For each call set it compares the full correlation with the correlation omitting Gomolava, the equal-weight site-level correlation, and the correlation recomputed without the later R1a reference I35012. All calculations within a given matrix use the same sites.

The supplied matrices were generated using SAMtools for indexed alignment retrieval. The QC metadata records the generating script digest and SAMtools version.

