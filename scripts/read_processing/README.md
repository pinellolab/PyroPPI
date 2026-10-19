# From sequencing reads to a counts table

PyroPPI's input contract starts at `counts.csv`, one row per sample and measured pair with `total` and `edited` already counted.
This directory holds the step before it: the pipeline that turns raw PPI Recorder reads into those counts.
Whatever a barcode names is a *construct* here, whether the library is a mutational scan, an ORFeome or a set of engineered fusions.

`example_data/` holds two small datasets together with the output they produce.
`vignettes/reads_to_counts` runs both and compares what it built against those files.

## The two halves

The first half builds a dictionary saying which construct each DNA barcode names.
A dictionary read pair carries the barcodes in R1 and the construct sequence in R2.
That sequence is aligned to its reference, the mismatches it carries are matched against the construct library, and the reads of one barcode are collapsed to the construct most of them agree on.

    step1_extract_barcodes.py   paired FASTQ  ->  FASTA of constructs headed bc1_bc2_<n>
    bowtie2 | samtools sort     FASTA         ->  sorted BAM
    step2_map_constructs.py     BAM           ->  one construct call per read
    step3_build_dictionary.py   read calls    ->  barcode -> construct dictionary

The second half applies that dictionary to cDNA reads and counts edits.
A base editor can edit the barcode itself, so step 1 expands each barcode into every edited form it could take.
Step 2 anchors the edit template in each read, calls an edit at every position that can carry one, and credits the read to the barcode's construct.

    step1_expand_barcodes.py    dictionary    ->  edited barcode alternatives
    step2_count_edits.py        cDNA FASTQ    ->  <sample>.cDNA_parsed.csv

`<sample>.cDNA_parsed.csv` has one row per construct.
`Total_reads` is every read credited to it and `Primary_edit_ct` is how many of those carry the on-target edit at `Primary_pos`.
Those two columns are PyroPPI's `total` and `edited`.

The step 3 dictionary is a headerless CSV: barcode pair, construct id, reads supporting the call, reads in total, purity.
Purity is the share of a barcode's reads that agree on the call.

## What has to be installed

The cDNA half needs pandas, so it runs anywhere PyroPPI runs.

The dictionary half needs pysam, through the `reads` extra:

```sh
pip install -e '.[reads]'
```

It also needs **bowtie2** and **samtools**, two programs pip does not install.
Verified here with bowtie2 2.5.5 and samtools 1.21.

## The three inputs

Three files describe a library, and `ref/` holds one example of each from the RAD51-paralog scan this pipeline was built for.
Another library supplies its own.

A **reference** is one string naming both a contig in the FASTA and the rows of the construct library that belong to it.
A tiled ORF has one reference per tile; otherwise there is one per target.
`--reference` names it, and both dictionary steps require it.
Nothing is read from a file name.

**The reference FASTA** (`ref/reference_sequences.fa`) holds one contig per reference, carrying the sequence a construct is compared against.
Step 2 reads it through pysam, which needs a `samtools faidx` index beside it.
The copy here ships with its `.fai`; run `samtools faidx` again on a copy made elsewhere.
Build the bowtie2 index once:

```sh
bowtie2-build ref/reference_sequences.fa ref/reference_index
```

**The construct library** (`ref/construct_library.csv`) is a CSV of `construct_id,reference,sequence`.
`construct_id` is written through to the output untouched.
`sequence` is compared base for base against the reference contig, and a construct identical to its reference carries no mismatch to recognise it by, so it is dropped.

**The R2 fixed-sequence table** (`ref/r2_fixed_seq.csv`) is a CSV of `reference,r2_fixed_seq` giving the constant sequence that precedes the construct in R2.

## Read layout

The three inputs generalise to any library; the read layout does not.
R1 must be `[bc1][linker][bc2][constant 3' sequence]`.
R2 must be `[prefix][fixed sequence][construct sequence]`.
A cDNA read must be `[bc1][edit template][bc2][constant 3' sequence]`.
A library cloned another way needs its own extraction step.

Step 2 takes a sorted BAM of construct sequences whose read names are `bc1_bc2_<n>`, so a replacement step 1 that produces that FASTA joins the rest of the pipeline unchanged.

## Known pitfalls

Step 2 reuses a `.bai` found beside the BAM without checking it, and a stale index gives wrong answers with no warning.
Delete `*.bai` before running it.

Step 2 of the cDNA half searches for the edit template's head anchor from position zero, which falls inside bc1.
A barcode matching that anchor, or sitting one edit away from matching it, loses its reads silently.
Check a barcode design against the anchor before sequencing.
