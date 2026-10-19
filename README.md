# PyroPPI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23245780.svg)](https://doi.org/10.5281/zenodo.23245780)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)

PyroPPI is the model behind PPI Recorder, a pooled two-hybrid assay that reads protein interactions out of a CRISPR base editor.
Two proteins are expressed as fusions on one plasmid, one to the adenine base editor ABE8e and one to the RNA-binding protein MCP, together with a barcoded reporter RNA carrying an MS2 hairpin.
When the two proteins interact, the editor reaches the MCP-bound reporter and converts adenines to inosines, so the fraction of edited reporter reads measures how often the pair comes together.
Sequencing recovers the edited fraction for thousands of pairs at once, and the barcode says which pair each read belongs to.

The model turns those counts into an interaction propensity for each pair, on the editing-probability scale, and separates the propensity from what a drug or a cell line adds on top of it.
Pairs are partially pooled, so a pair with few reads is pulled toward the rest of the library rather than taking an extreme value on thin evidence.

Source code: https://github.com/pinellolab/PyroPPI

## System requirements

PyroPPI is a Python package and needs no non-standard hardware.
It runs on any machine with Python 3.10 or newer.
A GPU shortens the fits and is never required; the same code runs on a CPU and gives the same answers.
The fits reported in the manuscript were run on one NVIDIA A100-PCIE-40GB in single precision; no fit comes close to that card's 40 GB, so a smaller CUDA GPU is enough.

Dependencies, installed automatically by pip:

| Package | Required for |
|---|---|
| numpy, pandas, scipy | data handling and the statistics |
| jax, jaxlib, numpyro, optax | the model and its inference |
| tqdm | the progress bar |
| matplotlib, scikit-learn | the figure scripts, through the `figures` extra |
| jupyter | the vignette notebooks, through the `notebooks` extra |
| pysam | the barcode dictionary scripts, through the `reads` extra |

Tested on Ubuntu 22.04.5 LTS with Linux 5.15, Python 3.12.10, and numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, jax 0.11.2, jaxlib 0.11.2, numpyro 0.22.0, optax 0.2.8, tqdm 4.70.1, matplotlib 3.11.2, scikit-learn 1.9.1 and pytest 9.1.1.
macOS and Windows are untested.
No version is pinned, so pip resolves current releases.

## Installation

```sh
git clone https://github.com/pinellolab/PyroPPI.git
cd PyroPPI
pip install -e '.[figures]'
```

Installing into a fresh virtual environment takes about 50 seconds on a desktop with no cached packages, and about 30 seconds with a warm pip cache.
Drop the `[figures]` extra for the model alone, or use `'.[notebooks]'` to run the vignette notebooks.
Add `'.[reads]'` for the barcode dictionary scripts in `scripts/read_processing`, which also want bowtie2 and samtools, two external programs pip does not install; `scripts/read_processing/README.md` says what they are for.

That install brings the CPU-only build of jaxlib, so the fits run on the CPU.
Ask for the CUDA build alongside the package to fit on a GPU instead:

```sh
pip install -e '.[figures]' 'jax[cuda12]'
```

`jax[cuda13]` is the same request against a CUDA 13 driver.
Fitting then uses the GPU whenever JAX finds one, and falls back to the CPU otherwise.
Reading a saved fit does no fitting, so it needs no GPU under either install.

Check the install with the test suite, which takes about a minute:

```sh
pytest tests/ -q
```

## Demo

`data/rad51x9` ships with the package and is the smallest screen: nine proteins measured in both orientations across two cell lines and three rapamycin durations, 972 observations over 81 measured pairs.

```sh
python scripts/fit_all.py --dataset rad51x9 --results results
```

This takes about one minute on a desktop CPU and about 50 seconds on a GPU.
It writes `results/rad51x9/` holding the posterior draws and three tables.

Read the result:

```python
import pyroppi
from pyroppi import stats

fit = pyroppi.load_fit("results/rad51x9")
pairs = fit.pairs().sort_values("propensity_mean", ascending=False)
print(pairs.head(3)[["protein_a", "protein_b", "propensity_mean"]])
print(stats.control_threshold(pairs))
```

Expected output: the three highest-ranked pairs are RAD51D with XRCC2 at a propensity near 0.140, RAD51C with XRCC3 near 0.106, and RAD51B with RAD51C near 0.054, against a control threshold of 0.040 set by the GFP and mCherry pairs.
Those three are the known BCDX2 and CX3 contacts, and they are the only pairs the control test calls.

## Instructions for use

To run PyroPPI on your own screen, write two tables into one directory.

`counts.csv` takes one row per sample and measured pair:

| sample | bait | prey | total | edited |
|---|---|---|---|---|
| U2OS_RepA_0h | RAD51D | XRCC2 | 4183 | 512 |

`bait` is the base-editor fusion, `prey` is the MCP fusion, `total` is the reporter reads assigned to that pair in that sample, and `edited` is how many of them were edited.

`samples.csv` takes one row per sample, with a `sample` column matching the counts and one binary column per condition:

| sample | 3h | HCT |
|---|---|---|
| U2OS_RepA_0h | 0 | 0 |
| HCT_RepA_3h | 1 | 1 |

The sample whose condition columns are all zero is the reference, and every coefficient is read against it.

Then fit, naming which columns belong to the drug block and which to the cell block:

```python
import pyroppi

counts, samples = pyroppi.read_dataset("my_screen")
fit = pyroppi.fit(counts, samples, drug=["3h"], cell=["HCT"])
fit.pairs().to_csv("my_pairs.csv", index=False)
```

Leave `drug` and `cell` empty for a screen measured under one condition.
Pass `baseline=False` for a screen where every sample is treated and there is no untreated arm, as in the degrader vignette.

A fit answers three questions:

```python
fit.pairs()          # one row per pair: propensity and its interval
fit.effects()        # what each drug and cell covariate adds, per pair
fit.contributions()  # the fitted mean per observation, split into its three parts
fit.save("results/my_screen")
```

A pair measured in both orientations is collapsed to one canonical pair, so the result is symmetric.
Observations below `min_reads`, which defaults to 20, are censored out of the likelihood and stay in the table.
A pair left with no observation above the floor is dropped.

## Calling variants and interactions

`pyroppi.stats` holds the tests the analyses use.

```python
from pyroppi import stats

pairs = fit.pairs()
stats.control_test(pairs)                    # against the GFP and mCherry pairs
stats.synonymous_test(variants)              # against a gene's synonymous variants
```

Both are one-tailed z tests on the log-odds scale against an empirical null taken from the experiment itself.
The module also provides Holm and Benjamini-Hochberg correction, an area under the curve with its Mann-Whitney p value, a partial Spearman correlation, and two permutation tests that respect structure in the data: one that keeps runs of neighbouring residues intact, and one that permutes labels only between blocks of equal size.

## Repository layout

```
src/pyroppi/     the model, the data reader, the statistics and the figure style
scripts/         fit_all.py and one script per manuscript figure
scripts/read_processing/   the reads-to-counts pipeline, with its reference tables
vignettes/       four worked analyses as notebooks and scripts
data/            the counts and annotations every script reads
tests/
```

## Reproducing the manuscript

`scripts/fit_all.py` refits all eleven screens and writes them to `results/`.
The registry at the top of that file lists each screen with its covariates, so a reader can see the whole design at a glance.

```sh
python scripts/fit_all.py                    # all eleven
python scripts/fit_all.py --dataset rad51x9  # one
```

A default install is CPU only, so these are the times to expect, measured with `JAX_PLATFORMS=cpu`: roughly 1.0 min for rad51x9, 1.4 for clinorf, 0.9 for zf_pom, 1.6 for commander and 0.5 for pulldown_xrcc2.
The three large screens, the localization panel and the RAD51D and XRCC2 scans, take tens of minutes on a CPU.
`results/runtimes.csv` records GPU times instead, measured on one NVIDIA A100-PCIE-40GB, each fit between 31 and 66 seconds, so a reader who installed the CUDA build sees those.
Refitting all eleven writes about 630 MB into `results/`, most of it posterior draws.

Each figure script then draws the data panels of one manuscript figure and prints every statistic it reports.

```sh
python scripts/figure_3.py
python scripts/supp_figure_8.py
```

Cartoons, schematics and structure renders are not reproduced here, so the panel letters skip those positions.

Every script takes `--data` and `--output`, and those that read a fit also take `--results`.
Nothing is hard-coded to a machine, so the data and the fits can live anywhere.

## Vignettes

Each vignette is a notebook with its output and a script that produces the same figures.

`vignettes/rad51x9` fits a nine-protein matrix and calls interactions against the GFP and mCherry controls.
`vignettes/deep_mutational_scanning` fits saturation scans of RAD51D and XRCC2, calls variants against the synonymous null, and asks whether the called residues sit closer to the partner than the rest.
`vignettes/degrader_screen` fits a screen with no untreated arm, where pomalidomide induces the interaction, and compares the result against published degradation data.
`vignettes/reads_to_counts` starts at the reads and stops where the others begin: it builds a barcode-to-construct dictionary, counts edits per construct, and shapes the counts into `counts.csv` and `samples.csv`.
It runs on the two example datasets in `scripts/read_processing/example_data` and compares what it built against the output committed beside them.
Nothing is fitted there, so numpyro and jax go unused, but it needs pysam with bowtie2 and samtools on the PATH.

## Data

`data/README.md` records, for every table, the source experiment, the cell lines and conditions, the reporter position scored, how the edited counts were derived, and the citation for any third-party annotation.

## Counting the reads

The shipped datasets start at `counts.csv`, with `total` and `edited` already counted, because the counting is a separate pipeline.
`scripts/read_processing` holds that pipeline: four steps that turn dictionary reads into a barcode-to-construct dictionary, and two that apply the dictionary to cDNA reads and count edits per construct per position.
`scripts/read_processing/README.md` is the guide.
`example_data/` beside it holds two small datasets with the output they produce, and `vignettes/reads_to_counts` runs both.

## Licence

PyroPPI is released under the GNU Affero General Public License v3.0 or later, a licence approved by the Open Source Initiative.
The full text is in `LICENSE`.
