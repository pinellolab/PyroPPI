# PyroPPI

PyroPPI is the model behind PPI Recorder, a pooled two-hybrid assay that reads protein interactions out of a CRISPR base editor.
Two proteins are expressed as fusions on one plasmid, one to the adenine base editor ABE8e and one to the RNA-binding protein MCP, together with a barcoded reporter RNA carrying an MS2 hairpin.
When the two proteins interact, the editor reaches the MCP-bound reporter and converts adenines to inosines, so the fraction of edited reporter reads measures how often the pair comes together.
Sequencing recovers the edited fraction for thousands of pairs at once, and the barcode says which pair each read belongs to.

The model turns those counts into an interaction propensity for each pair, on the editing-probability scale, and separates the propensity from what a drug or a cell line adds on top of it.
Pairs are partially pooled, so a pair with few reads is pulled toward the rest of the library rather than taking an extreme value on thin evidence.

## Install

```sh
pip install -e .
```

Python 3.10 or newer.
The model runs on numpy, pandas, scipy, jax, numpyro, optax and tqdm.
Add the figure dependencies with `pip install -e '.[figures]'` and the notebook dependencies with `pip install -e '.[notebooks]'`.

That install brings the CPU-only build of jaxlib, so the fits run on the CPU.
Ask for the CUDA build alongside the package to fit on a GPU instead:

```sh
pip install -e '.[figures]' 'jax[cuda12]'
```

`jax[cuda13]` is the same request against a CUDA 13 driver.
Fitting then uses the GPU whenever JAX finds one, and falls back to the CPU otherwise.
Reading a saved fit does no fitting, so it needs no GPU under either install.

## Fitting a dataset

```python
import pyroppi

counts, samples = pyroppi.read_dataset("data/rad51x9")
fit = pyroppi.fit(
    counts,
    samples,
    drug=["1h", "3h", "HCT_x_1h", "HCT_x_3h"],
    cell=["HCT"],
)

fit.pairs()          # one row per pair: propensity and its interval
fit.effects()        # what each drug and cell covariate adds, per pair
fit.contributions()  # the fitted mean per observation, split into its three parts
fit.save("results/rad51x9")
```

`counts.csv` holds one row per sample and measured pair with columns `sample`, `bait`, `prey`, `total` and `edited`.
`samples.csv` holds one row per sample with the binary design columns named in `drug` and `cell`.
The reference condition is the sample whose design columns are all zero.
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
vignettes/       three worked analyses as notebooks and scripts
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
`results/runtimes.csv` records GPU times instead, each fit between 30 seconds and a little over a minute, so a reader who installed the CUDA build sees those.
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

## Data

`data/README.md` records, for every table, the source experiment, the cell lines and conditions, the reporter position scored, how the edited counts were derived, and the citation for any third-party annotation.
