"""Fit PyroPPI to the pooled zinc-finger degrader screen and write the figures.

The companion notebook `degrader_screen.ipynb` walks through the same analysis
with the explanation. This script is the version to run from a terminal. It
reads the dataset, fits the model without a pair baseline, ranks the 90
zinc-finger domains by their pomalidomide-dependent interaction propensity,
tests the enrichment of known degradation targets at the top of the ranking,
and benchmarks the whole ranking against two published degradation screens.
Four PNG figures are written beside this file.

    python degrader_screen.py                  # data from the repository
    python degrader_screen.py --data /path/to/zf_pom
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import pyroppi

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# Okabe-Ito, the colourblind-safe palette the manuscript figures use.
BLUE = "#0072B2"  # the 300 nM arm
VERMILLION = "#D55E00"  # the 3 uM arm, and the known degradation targets
GREY = "#6F6F6F"
PALE = "#D9D9D9"

DRUG = ["POM_300_NM", "POM_3_UM"]
BAIT = "CRBN"

# The 11 zinc-finger domains Sievers et al. (Science 362, eaat0572, 2018) call
# degraded on pomalidomide; 9 of them were cloned into this library. The names
# are the truncated ones the screen's own tables use: ZN276 is ZNF276, ZN517 is
# ZNF517, ZN582 is ZNF582, ZN653 is ZNF653, ZN654 is ZNF654, ZN787 is ZNF787
# and ZN827 is ZNF827.
SIEVERS_TARGETS = [
    "E4F1",
    "IKZF3",
    "ZN276",
    "ZN517",
    "ZN582",
    "ZN653",
    "ZN654",
    "ZN787",
    "ZN827",
]
TOP_N = 10

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def dose_colour(sample_or_column: str) -> str:
    """Blue for the 300 nM arm, vermillion for the 3 uM arm."""
    return BLUE if "300" in sample_or_column else VERMILLION


def observed_by_dose(counts: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """Observed edited fraction of each domain in each dose arm.

    Reads are pooled over the four replicates rather than the per-replicate
    fractions averaged, which is the quantity a binomial likelihood sees.
    """
    table = counts.merge(samples[["sample", "treatment"]], on="sample")
    totals = table.groupby(["prey", "treatment"])[["total", "edited"]].sum()
    return (totals["edited"] / totals["total"]).unstack()


def propensity_table(result, covariate: str) -> pd.DataFrame:
    """Interaction propensity of each domain under one dose indicator.

    The fit has no pair baseline, so the drug logit of a canonical pair is the
    whole log-odds of editing under that dose, and the sigmoid of it is the
    whole fitted propensity rather than an increment over an untreated arm.
    """
    effects = result.effects()
    table = effects[effects["covariate"] == covariate].copy()
    table["gene"] = np.where(
        table["protein_a"] == BAIT, table["protein_b"], table["protein_a"]
    )
    table["propensity"] = 1.0 / (1.0 + np.exp(-table["logit_mean"]))
    table = table.sort_values("propensity", ascending=False).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    return table[["gene", "rank", "logit_mean", "logit_sd", "propensity"]]


def correlate(x, y) -> dict:
    """Spearman and Pearson of two vectors, with the number of pairs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    spearman = stats.spearmanr(x, y)
    pearson = stats.pearsonr(x, y)
    return {
        "n": len(x),
        "rho": float(spearman.statistic),
        "rho_p": float(spearman.pvalue),
        "r": float(pearson.statistic),
        "r_p": float(pearson.pvalue),
    }


def plot_design(counts, samples, path):
    """The eight treated samples, and the two doses against each other."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.8, 2.8))

    order = list(samples.sort_values(["treatment", "replicate"])["sample"])
    fraction = counts["edited"] / counts["total"]
    rng = np.random.default_rng(0)
    for x, name in enumerate(order):
        rows = counts["sample"] == name
        jitter = rng.normal(0, 0.08, int(rows.sum()))
        left.scatter(
            x + jitter,
            fraction[rows] * 100,
            s=6,
            color=dose_colour(name),
            alpha=0.45,
            lw=0,
        )
    left.set_xticks(range(len(order)), order, rotation=90)
    left.set_ylabel("observed edited reads (%)")
    left.set_title("every sample is drug-treated", loc="left")

    wide = observed_by_dose(counts, samples)
    low = wide["pomalidomide_300nM"] * 100
    high = wide["pomalidomide_3uM"] * 100
    span = [0, max(low.max(), high.max()) * 1.05]
    right.plot(span, span, color=GREY, lw=0.6, ls="--")
    right.scatter(low, high, s=14, color=VERMILLION, alpha=0.7, lw=0)
    right.set_xlim(span)
    right.set_ylim(span)
    right.set_xlabel("300 nM pomalidomide (% edited)")
    right.set_ylabel("3 µM pomalidomide (% edited)")
    right.set_title("reads pooled over four replicates, one point per domain", loc="left")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_convergence(losses, path):
    """ELBO loss against step, with the last fifth enlarged beside it."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.5, 2.4))
    step = np.arange(1, len(losses) + 1)
    left.plot(step, losses, color=BLUE, lw=0.8)
    left.set_yscale("log")
    left.set_xlabel("SVI step")
    left.set_ylabel("ELBO loss (log scale)")
    left.set_title("whole fit", loc="left")

    tail = slice(int(0.8 * len(losses)), None)
    right.plot(step[tail], losses[tail], color=BLUE, lw=0.8)
    right.set_xlabel("SVI step")
    right.set_ylabel("ELBO loss")
    right.set_title("last fifth", loc="left")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ranking(high, low, path):
    """The ranked domains at 3 µM, and the top of the ranking at both doses."""
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1.35, 1]}
    )

    known = high["gene"].isin(SIEVERS_TARGETS)
    left.axvspan(0.5, TOP_N + 0.5, color=PALE, lw=0, zorder=0)
    left.scatter(
        high.loc[~known, "rank"],
        high.loc[~known, "propensity"] * 100,
        s=14,
        color=GREY,
        alpha=0.6,
        lw=0,
        label="other domain",
    )
    left.scatter(
        high.loc[known, "rank"],
        high.loc[known, "propensity"] * 100,
        s=22,
        color=VERMILLION,
        lw=0,
        label="Sievers degradation target",
    )
    for _, row in high[known].iterrows():
        left.annotate(
            row["gene"],
            (row["rank"], row["propensity"] * 100),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=6.5,
            color=VERMILLION,
        )
    left.set_xlabel(f"rank (shaded: top {TOP_N})")
    left.set_ylabel("POM-dependent interaction propensity (%)")
    left.set_title("90 zinc-finger domains at 3 µM", loc="left")
    left.legend(loc="upper right", fontsize=7)

    top = high.head(15).merge(low[["gene", "propensity"]], on="gene", suffixes=("", "_low"))
    y = np.arange(len(top))[::-1]
    right.hlines(y, top["propensity_low"] * 100, top["propensity"] * 100, color=PALE, lw=1.4)
    right.scatter(top["propensity_low"] * 100, y, s=18, color=BLUE, label="300 nM", zorder=3)
    right.scatter(top["propensity"] * 100, y, s=18, color=VERMILLION, label="3 µM", zorder=3)
    right.set_yticks(y, top["gene"])
    right.set_ylim(-0.8, len(top) - 0.2)
    right.set_xlabel("POM-dependent interaction propensity (%)")
    right.set_title("the top fifteen at both doses", loc="left")
    right.tick_params(axis="y", length=0)
    right.spines["left"].set_visible(False)
    right.legend(loc="lower right", fontsize=7)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_benchmarks(merged, sievers, nguyen, path):
    """Interaction propensity against each published degradation measurement."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.8, 3.0))

    panels = [
        (left, "sievers_pom_fold_depletion", "Sievers et al. fold depletion", sievers),
        (right, "nguyen_2024_l2fc_depletion", "Nguyen et al. depletion (log2 FC)", nguyen),
    ]
    for ax, column, xlabel, result in panels:
        part = merged.dropna(subset=[column])
        known = part["gene"].isin(SIEVERS_TARGETS)
        ax.scatter(
            part.loc[~known, column],
            part.loc[~known, "propensity"] * 100,
            s=14,
            color=GREY,
            alpha=0.6,
            lw=0,
        )
        ax.scatter(
            part.loc[known, column],
            part.loc[known, "propensity"] * 100,
            s=22,
            color=VERMILLION,
            lw=0,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("POM-dependent interaction propensity (%)")
        ax.set_title(
            f"n = {result['n']}\n"
            f"Spearman rho = {result['rho']:.2f}, P = {result['rho_p']:.2g}\n"
            f"Pearson r = {result['r']:.2f}, P = {result['r_p']:.2g}",
            loc="left",
            fontsize=7.5,
        )

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO / "data" / "zf_pom",
        help="directory holding counts.csv and samples.csv",
    )
    args = parser.parse_args()

    counts, samples = pyroppi.read_dataset(args.data)
    print(f"{len(counts)} observations, {len(samples)} samples")
    print(f"{counts['prey'].nunique()} zinc-finger domains against bait {BAIT}")
    print(samples.to_string(index=False), end="\n\n")
    print("every sample carries exactly one dose indicator:")
    print(samples[DRUG].sum(axis=1).value_counts().to_string(), end="\n\n")
    print(f"median reporter reads per observation: {counts['total'].median():,.0f}")
    print(f"observations below the 20-read floor: {(counts['total'] < 20).sum()}")
    wide = observed_by_dose(counts, samples)
    higher = (wide["pomalidomide_3uM"] > wide["pomalidomide_300nM"]).sum()
    print(f"domains edited more at 3 µM than at 300 nM: {higher} of {len(wide)}\n")

    benchmarks = pd.read_csv(REPO / "data" / "annotations" / "degrader_benchmarks.csv")

    start = time.perf_counter()
    result = pyroppi.fit(
        counts, samples, drug=DRUG, baseline=False, num_steps=60_000
    )
    runtime = time.perf_counter() - start
    print(f"\nfitted in {runtime:.0f} s")
    print(f"{len(result.canonical)} canonical pairs, final ELBO loss {result.losses[-1]:,.0f}")

    try:
        result.pairs()
    except ValueError as error:
        print(f"pairs() is unavailable here: {error}\n")

    high = propensity_table(result, "POM_3_UM")
    low = propensity_table(result, "POM_300_NM")
    print("top fifteen domains by POM-dependent interaction propensity at 3 µM")
    print(high.head(15).to_string(index=False), end="\n\n")

    doses = high.merge(low[["gene", "propensity"]], on="gene", suffixes=("", "_low"))
    rises = doses["propensity"] > doses["propensity_low"]
    print(f"fitted propensity higher at 3 µM: {rises.sum()} of {len(doses)} domains")
    print(f"among the top fifteen: {rises.head(15).sum()} of 15\n")

    present = [gene for gene in SIEVERS_TARGETS if gene in set(high["gene"])]
    top = list(high["gene"].head(TOP_N))
    overlap = [gene for gene in top if gene in present]
    p_enrichment = stats.hypergeom.sf(
        len(overlap) - 1, len(high), len(present), TOP_N
    )
    print(f"{len(present)} of the Sievers targets are in the library")
    print(f"top {TOP_N}: {', '.join(top)}")
    print(f"{len(overlap)} of them are Sievers targets: {', '.join(overlap)}")
    print(f"hypergeometric P = {p_enrichment:.3g}\n")

    merged = high.merge(benchmarks, on="gene", how="left")
    sievers_part = merged.dropna(subset=["sievers_pom_fold_depletion"])
    nguyen_part = merged.dropna(subset=["nguyen_2024_l2fc_depletion"])
    sievers = correlate(
        sievers_part["propensity"], sievers_part["sievers_pom_fold_depletion"]
    )
    nguyen = correlate(
        nguyen_part["propensity"], nguyen_part["nguyen_2024_l2fc_depletion"]
    )
    for name, result_ in (("Sievers", sievers), ("Nguyen", nguyen)):
        print(
            f"{name}: n = {result_['n']}, "
            f"Spearman rho = {result_['rho']:.3f} (P = {result_['rho_p']:.3g}), "
            f"Pearson r = {result_['r']:.3f} (P = {result_['r_p']:.3g})"
        )
    target_values = sievers_part.loc[
        sievers_part["gene"].isin(present), "sievers_pom_fold_depletion"
    ]
    rest_values = sievers_part.loc[
        ~sievers_part["gene"].isin(present), "sievers_pom_fold_depletion"
    ]
    print(
        f"\nSievers fold depletion: targets {target_values.min():.2f} to "
        f"{target_values.max():.2f}, every other domain "
        f"{rest_values.min():.2f} to {rest_values.max():.2f}"
    )

    plot_design(counts, samples, HERE / "screen_design.png")
    plot_convergence(result.losses, HERE / "convergence.png")
    plot_ranking(high, low, HERE / "ranked_domains.png")
    plot_benchmarks(merged, sievers, nguyen, HERE / "benchmarks.png")
    print(f"figures written to {HERE}")


if __name__ == "__main__":
    main()
