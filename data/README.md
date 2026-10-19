# Data

Each directory holds the input of one fit as two tables.
`counts.csv` has one row per sample and measured pair with columns `sample`, `bait`, `prey`, `total` and `edited`, where `bait` is the ABE8e (TadA8e) fusion, `prey` is the MCP fusion, `total` is the number of reporter reads assigned to the pair and `edited` is the number of those reads scored as edited.
Rows with no reads are omitted, and no read floor has been applied.
`samples.csv` has one row per sample: descriptive columns parsed from the sample name, then binary design columns.
Variant constructs are written `GENE:A63E` (reference residue, position, substituted residue), with synonymous variants as `GENE:A63A` and stop-gain variants as `GENE:R12*`.
Treatment values `rapamycin_1h` and `rapamycin_3h` denote rapamycin added 1 or 3 hours before harvest, and `rapamycin_0h` denotes no rapamycin.

## Datasets

### rad51x9

Nine-protein RAD51 paralog matrix (RAD51, RAD51B, RAD51C, RAD51D, XRCC2, XRCC3, SFR1, GFP and mCherry) in every bait-prey orientation, from the condensed HCT116 readout of the October 2023 run and the U2OS all-position edit-rate readout of November 2023.
Two replicates per cell line at 0, 1 and 3 hours of rapamycin.
The workbooks give a summed edit fraction over reporter positions 3 and 11; `edited` is total reads times half that fraction, truncated to an integer.
Design columns are `1h`, `3h`, `HCT_x_1h` and `HCT_x_3h` (condition block) and `HCT` (cell block).
972 rows, 12 samples, 81 measured pairs.

### clinorf

ClinORF module screen of 20 proteins from homologous recombination, Fanconi anemia and cerebral cavernous malformation complexes plus GFP and mCherry, from the U2OS all-position edit-rate readout of November 2023.
U2OS cells, replicate A at 0, 1 and 3 hours of rapamycin and replicate B at 1 and 3 hours.
`edited` is total reads times half the summed edit fraction over reporter positions 3 and 11, rounded to the nearest integer.
Design columns are `1h` and `3h` (condition block).
2,177 rows, 5 samples, 436 measured pairs.

### zf_pom

CRBN lacking its DDB1-interaction domain fused to TadA8e, against an MCP library of 90 zinc-finger domains, from the `data_summary` sheet of the 100-domain zinc-finger pomalidomide screen.
HCT116 cells treated with pomalidomide at 300 nM or 3 µM, replicates 1 to 4; untreated samples and replicate 5 are not included.
`total` and `edited` are the workbook's read and edited-read columns as written; the workbook does not record which reporter position is scored.
Design columns are `POM_300_NM` and `POM_3_UM` (condition block); every sample has exactly one of them set, so no sample is untreated.
720 rows, 8 samples, 90 measured pairs.

### localization

Eight TadA8e-tag recorder lines (MEM, ERM, LD, RAB10, NES, ACT, NLS, NOLS) against an MCP-ORFeome library, from the per-run read-count and edit-fraction workbooks of runs 102822, 111522 and 011023.
HCT116 cells, three replicates per tag and run at 1 and 3 hours of rapamycin, with NES and NLS measured in two runs; samples labelled PSTI, which the source names an edit-enrichment step, are not included.
`edited` is total reads times the workbook's primary-position edit fraction, truncated to an integer; the workbooks do not record which reporter position is scored.
Each sample belongs to one tag, given in the `tag` column, and that tag is the bait of every row of the sample; `batch` is the run.
The design column is `3H` (condition block), with 1 hour as the reference.
229,928 rows, 60 samples, 30,811 measured pairs over 3,852 ORFs.

### dms_rad51d and dms_xrcc2

Deep mutational scans of RAD51D and XRCC2 on MCP, each measured against TadA8e fused to RAD51C, XRCC2 and XRCC3 (RAD51D library) or RAD51D, XRCC2 and XRCC3 (XRCC2 library), and to mCherry, from the RAD51D and XRCC2 saturation scans sequenced in August 2024.
HCT116 and U2OS cells, five replicates each at 0 and 3 hours of rapamycin.
`total` is the workbook's `Total_reads` and `edited` its `Pos11` column, reads edited at reporter position 11, each summed over the TadA8e clones of one bait.
Design columns are `3h` and `U2OS_x_3h` (condition block) and `U2OS` (cell block).
dms_rad51d: 141,515 rows, 20 samples, 7,083 measured pairs.
dms_xrcc2: 111,724 rows, 20 samples, 5,598 measured pairs.

### dms_rad51c and dms_xrcc3

Deep mutational scans of RAD51C and XRCC3 on MCP, each measured against TadA8e fused to the wild-type partner (XRCC3 and RAD51C respectively), from the `11rep_50+reads` sheet of the eleven-replicate RAD51C and XRCC3 saturation scan, which keeps variants with at least 50 reads.
HCT116 cells, eleven replicates over three runs given in the `batch` column, with no treatment.
`total` and `edited` are the sheet's read and edited-read columns as written, scored at reporter position 14.
There are no design columns.
dms_rad51c: 12,457 rows, 11 samples, 1,144 measured pairs.
dms_xrcc3: 13,963 rows, 11 samples, 1,312 measured pairs.

### commander

The ten COMMD proteins of the Commander complex plus GFP and mCherry in every bait-prey orientation, from the `data_summary` sheet of the Tol2 COMMD screen, restricted to pairs with at least 100 reads in at least 80% of samples and at most 50,000 reads in every sample.
HCT116 (six replicates) and U2OS (eight replicates), each without rapamycin and with 1 hour of rapamycin.
`edited` is total reads times the workbook's primary-position edit fraction computed on observations of at least 50 reads, rounded to the nearest integer; the workbook does not record which reporter position is scored.
Design columns are `Rap1h` (condition block) and `HCT` (cell block).
4,032 rows, 28 samples, 144 measured pairs.

### pulldown_rad51d and pulldown_xrcc2

Targeted RAD51D and XRCC2 variant sublibraries on MCP, measured against TadA8e fused to the wild-type partner (XRCC2 and RAD51D respectively), from the PPI Recorder arm of the HCT116 X2RD dataset, keeping variants with at least 50 reads in every sample.
HCT116 cells, four replicates, with no treatment.
`total` is the `Total_reads` sheet and `edited` the `Pos14` sheet, reads edited at reporter position 14.
There are no design columns.
pulldown_rad51d: 720 rows, 4 samples, 180 measured pairs.
pulldown_xrcc2: 592 rows, 4 samples, 148 measured pairs.

## Variant annotations

`annotations/variants.csv` annotates the 4,950 missense variants of the RAD51D, XRCC2, RAD51C and XRCC3 scans that carry predictor scores, one row per variant; synonymous and stop-gain variants are not annotated.
`alphamissense` and `alphamissense_class` are the AlphaMissense pathogenicity score and its three-class call (Cheng et al., Science 2023), used under CC BY 4.0.
`popEVE` is the `popEVE (unfiltered)` score from the published per-protein popEVE exports (Orenbuch et al. 2025); lower is more deleterious, and the scale differs between proteins.
`partner` is the wild-type partner of the scan, and `min_dist_to_partner_A` is the shortest heavy-atom distance in ångströms from the residue to the partner chain in PDB entry 9SVX, with ligands and nucleic acid excluded.
`interface_8A` is true where that distance is below 8 Å; both are blank where the residue is not resolved in the entry.
`conservation` is one minus the Shannon entropy of the residue's column in an HHblits multiple sequence alignment, divided by log2(20), with gaps and nonstandard residues ignored.
`clinvar_classification` is the ClinVar germline classification of the same protein change, retrieved through NCBI E-utilities in September 2026, with distinct classifications of one change joined by semicolons; it is blank where ClinVar has no record.

## Reporter measurements

These three tables measure the reporter on a plate instead of in a pooled screen, so they carry no bait and prey columns and no design matrix.

`reporter/rapamycin_positions.csv` is the editing of the MS2 reporter in HEK cells with and without rapamycin, from the HEK rapamycin test plate.
One row per sample and reporter adenosine: `treatment` is `none` or `rapamycin`, `replicate` is 1 or 2, `reporter_position` is the adenosine's position in the amplicon and `edit_index` its rank among the twenty, and `editing_percent` is the percentage of reads edited at that adenosine.
The MS2 hairpin covers positions 28 to 44 inclusive, which is `edit_index` 8 to 14.
80 rows over 4 samples and 20 positions.

`reporter/rapamycin_fold_change.csv` is the ratio of the two arms of the same experiment, from that plate's own fold-change table.
One row per reporter position and replicate, with `fold_change` the editing with rapamycin over the editing without it.
The values are the source's own, rounded to one decimal; each agrees with the ratio of the two percentages in `rapamycin_positions.csv` to within the rounding both tables carry.
40 rows.

`reporter/degrader_titration.csv` is the two degrader dose-response curves, from the pomalidomide and ARV-471 titration tables.
One row per compound, concentration and replicate: `compound` is pomalidomide or ARV-471, `partner` is the zinc finger or receptor the compound recruits (ZNF276 and ESR1), `concentration_uM` is the compound concentration in micromolar with 0 for the untreated well, and `editing_fraction` is the fraction of reads edited.
60 rows over 2 compounds, 10 concentrations and 3 replicates each.

## Library and benchmark annotations

`annotations/degrader_benchmarks.csv` holds two published measurements of zinc-finger degradation, one row per domain, from the `comparison data` sheet of the 100-domain zinc-finger pomalidomide screen.
`sievers_pom_fold_depletion` is the average pomalidomide fold depletion of a zinc-finger-GFP reporter from Sievers, Q. L. et al., Defining the human C2H2 zinc finger degrome targeted by thalidomide analogs through CRBN, Science 362, eaat0572 (2018), available for 86 of the 90 domains.
`nguyen_2024_l2fc_depletion` is a pomalidomide depletion measure derived from the zinc-finger degron reporter screen of Nguyen, T. M. et al., Proteolysis-targeting chimeras with reduced off-targets, Nature Chemistry 16, 218-228 (2024), available for 9 of them.
That screen reports eGFP normalised to mCherry on a relative fluorescence scale; the transformation to the values in this column is not recorded in the source workbook, which supplied them under this heading.
`gene` uses the truncated names the screen's own tables use, matching the `prey` column of `zf_pom/counts.csv`.
90 rows.

`annotations/zf_qc.csv` is the zinc-finger screen as sequenced, from the `data_summary` sheet of the 100-domain zinc-finger pomalidomide screen.
One row per domain, treatment arm and replicate, with `total_reads`, `edited_reads` and the `edit_fraction` the workbook records, which is the ratio of the two.
Arms are `NO_TREATMENT`, `POM_300_NM` and `POM_3_UM`, named as the source names them, and replicates are held to 1 to 4, the set the `zf_pom` fit carries.
No read floor is applied.
1,080 rows over 90 domains, 3 arms and 4 replicates.

`annotations/barcode_summary_zf.csv` summarises the MCP-zinc-finger barcode dictionary, from its 100-domain barcode-to-gene assignment.
One row per domain and barcode length, with `n_barcodes` the number of distinct barcodes of that length assigned to that domain.
The library holds 45,192 barcodes over 90 domains, at lengths of 12, 13, 14 and 15 nucleotides and no others.
357 rows.

`annotations/barcode_summary_orfeome.csv` summarises the MCP-ORFeome barcode dictionary, from the `ORF summary` sheet of the v8 L1-6 dictionary workbook.
One row per hORFeome v8.1 clone: `clone` is the clone identifier, `gene` its symbol, `orf_length_nt` the open reading frame's length in nucleotides and `n_barcodes` the number of distinct barcodes assigned to it.
The dictionary holds 42,249 barcodes over 3,569 clones.
3,569 rows.

`annotations/orfeome_length_coverage.csv` is the share of the hORFeome v8.1 clone collection that dictionary represents, binned by open reading frame length.
`n_represented` counts clones of `barcode_summary_orfeome.csv` in the bin and `n_horfeome_clones` counts distinct clones of the collection, taken from the `horf81_cloneInfo20120427` sheet of the hORFeome clone metadata workbook; `fraction_represented` is their ratio.
The bins are the dictionary's own edges with a closing 4,000 to 5,000 nucleotide bin and an open bin above it, so each of the 12,680 clones falls in exactly one.
The collection itself is a third-party annotation workbook and is not shipped. This table carries the per-bin counts alone.
15 rows.

`annotations/dictionary_cutoffs.csv` is the diagnostic the nine Dual NEBuilder barcode-to-variant dictionaries' cutoffs were read off, from the per-barcode support and accuracy table delivered with the Dual NEBuilder cutoff plots.
`support_reads` is the number of reads a barcode gives its best-supported variant and `total_reads` the barcode's reads in all; purity is `support_reads` over `total_reads`.
The delivered file is one row per barcode; this table carries one row per distinct combination of block, support reads and total reads with `n_barcodes` the number of barcodes on it, which keeps every point and its multiplicity and is a third of the size.
72,462 rows standing for 246,417 barcodes over 9 blocks.

`annotations/dictionary_cutoff_parameters.csv` is the cutoff each of those blocks took, from the Dual NEBuilder dictionary parameter table, with the retention it gives.
`support_cutoff` and `purity_cutoff` bound the accepted region, `n_retained` and `pct_retained` are the barcodes inside it, and `n_retained_standard_rule` and `pct_retained_standard_rule` are the same under the pipeline's standard threshold of at least 3 support reads and purity at least 0.9.
All four are computed over the full per-barcode table.
9 rows.

`annotations/localization_labels.csv` is the compartment class of each ORF of the localization library, one row per ORF.
`orf_class` is `CY` where the Surfaceome master table gives the protein a cytosol-facing N terminus, `NM` where the protein has no Surfaceome topology, carries a curated Gene Ontology term for the cytosol, nucleoplasm or nucleolus and carries no membrane term, and `excluded` otherwise.
The Surfaceome join is on UniProt accession rather than on gene symbol, which is what recovers ORFs renamed since that table was published; `join_route` records which field matched.
`topology`, `tm_domains`, `signal_peptide` and `surface_label` are the Surfaceome fields the class is read from, and the four `go_*` columns the Gene Ontology terms.
Surfaceome topology is from Bausch-Fluck, D. et al., The in silico human surfaceome, Proc. Natl Acad. Sci. USA 115, E10988-E10997 (2018); the Gene Ontology annotations are from the human GOA association file and the GO basic ontology.
3,852 rows, of which 454 are CY and 1,123 NM.

`annotations/known_interactions.csv` lists the previously reported interactions among the ClinORF library's twenty proteins, one row per protein pair.
Pairs are human binary physical interactions retrieved from the IntAct web service, restricted to the evidence types `direct interaction` and `physical association`; `publications`, `pubmed` and `methods` record the supporting evidence and `known` is true where two or more publications support the pair.
The retrieval is from `https://www.ebi.ac.uk/intact/ws/interaction/findInteractions`; the release version is not recorded with the table.
17 rows, 11 of them marked known.

`annotations/interface_area.csv` is the buried surface area of each protein pair resolved in three deposited structures, from the combined solvent-accessible surface area analysis.
`bsa_A2` is in square ångströms and is the largest value over the copy combinations of the two proteins, since a non-contacting copy pair reports zero; pairs the structure resolves without an interface therefore appear with a value of zero.
`structure` is the PDB accession: 9Q2A for the BCDX2 complex, 9SVX for the RAD51C-XRCC3 complex and 8P0W for the Commander complex.
9Q2A is an unpublished entry and is cited by accession only.
87 rows over 3 structures.

## Immunoprecipitation arms

`pulldown_arms/arms.csv` is the read counts of the targeted RAD51D and XRCC2 sublibraries over three sequencing arms of one pool, from the HCT116 X2RD dataset.
One row per variant and sample: `arm` is `PPI Recorder` for the unenriched arm, `anti-FLAG` for immunoprecipitation of the wild-type partner and `anti-myc` for immunoprecipitation of the variant, and the three count columns are the workbook's `Total_reads`, `Pos14` and `Any_edit_ct` sheets.
`total_reads` is the reads assigned to the variant, `edited_reads` those edited at reporter position 14 and `any_edit_reads` those edited at any position; the last can exceed `total_reads` in the source and is not an edited-read channel for a binomial likelihood.
Variants are kept where `total_reads` is at least 50 in every one of the eleven samples, which is the rule the figures read the pool under.
The PPI Recorder rows are the same counts as `pulldown_rad51d/counts.csv` and `pulldown_xrcc2/counts.csv`, repeated here so the three arms can be read together.
3,608 rows over 328 variants and 11 samples, 4 PPI Recorder replicates, 3 anti-FLAG and 4 anti-myc.
