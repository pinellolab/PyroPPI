# Example data

Two small datasets that run the pipeline end to end, each with the output it produces.
`vignettes/reads_to_counts` runs both and compares what it built against `expected_output/`.

## The two datasets

Both are RAD51C, one block each, and both are driven by the same cDNA sample.
`NR-TV_HCT_RAD51C_XRCC3_Rep1` spans the whole RAD51C ORF, so it covers either block.
The scripts are identical for the two; the step 3 cutoffs differ.

| | `Dual_NEBuilder` | `regular` |
|---|---|---|
| `--reference` | `RAD51C_blk_1` (residues 109-113) | `RAD51C_blk_3` (residues 232-236) |
| step 3 cutoffs | support >= 5, purity >= 0.10 | support >= 3, purity >= 0.90 |
| construct | several constructs per barcode | about one construct per barcode |
| dictionary barcodes | 62 | 75 |
| cDNA reads matched | 5,949 of 7,156 (83%) | 7,243 of 7,280 (99%) |

The `Dual_NEBuilder` cloning puts several constructs behind one barcode, so its purity cutoff sits at 0.10.
A barcode there is kept on a plurality of its reads.
On `regular` the barcodes sit at purity 1.0 and the 0.90 cutoff barely bites.

## Layout

```
Dual_NEBuilder/
  dict_build/        RAD51C_DMS_1_Dictionary_R{1,2}.fastq.gz
  cdna_edit_count/   NR-TV_HCT_RAD51C_XRCC3_Rep1__R1.fastq.gz
  expected_output/   the dictionary and the per-construct edit counts
regular/
  ... the same, for RAD51C_DMS_3
```

The cDNA half ships R1 only.
This sample does not use the R2 dictionary-contamination screen.

## Provenance

The dictionary reads come from the `multiPPIseq_DMS_dict_071526` build and the cDNA reads from `multiPPIseq_DMS_editRate_041726`, sample `NR-TV_HCT_RAD51C_XRCC3_Rep1`.
These are the two datasets paired in the production run the Methods describe.

They were subset to a few consecutive residues with strong editing signal: the dictionary read pairs for a set of well-covered barcodes, and the cDNA reads whose barcode, edited or not, matches one of them.
Each FASTQ holds roughly 7-8k reads.
