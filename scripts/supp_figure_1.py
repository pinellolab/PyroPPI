#!/usr/bin/env python
"""Supplementary figure 1: the zinc-finger library as built and as sequenced.

Panels a and b describe the MCP-zinc-finger barcode dictionary, panels c and d
the screen's editing readout and replicate agreement in each treatment arm, and
panels e and f compare the fitted pomalidomide effect against two published
degradation measurements.

    python supp_figure_1.py
    python supp_figure_1.py --data /path/to/data --results /path/to/results
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr

import pyroppi
from pyroppi.plotting import COLORS, figure, panel_letter, save, use_style

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

BAIT = "CRBN"
DOSE = "POM_3_UM"
REPLICATES = (1, 2, 3, 4)

# Workbook arm key, the name used on the page, and its colour. The untreated arm
# is named NO_TREATMENT in every file of this screen.
ARMS = (
    ("NO_TREATMENT", "none", COLORS["control"]),
    ("POM_300_NM", "300 nM", COLORS["pom_300"]),
    ("POM_3_UM", "3 µM", COLORS["pom_3"]),
)

# An edit fraction of exactly zero cannot sit on a log axis. Those observations
# are drawn at this floor rather than dropped, since a domain that never edits
# is a real measurement.
ZERO_FLOOR = 1.0e-4

BARCODE_LENGTHS = (12, 13, 14, 15)


def logistic(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def style(ax: plt.Axes, grid_axis: str | None = None) -> None:
    """Tick padding shared by every panel, with an optional value grid."""
    ax.tick_params(axis="both", pad=2)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, color=COLORS["grid"], lw=0.45, alpha=0.7)
        ax.set_axisbelow(True)


def draw_barcodes_per_domain(ax: plt.Axes, data: Path) -> pd.DataFrame:
    """Panel a: unique barcodes per zinc-finger domain, ranked, on a log axis."""
    frame = pd.read_csv(data / "annotations" / "barcode_summary_zf.csv")
    counts = frame.groupby("domain", as_index=False).n_barcodes.sum()
    counts = counts.sort_values("n_barcodes", ascending=False).reset_index(drop=True)
    counts.insert(0, "rank", np.arange(1, len(counts) + 1))
    median = float(counts.n_barcodes.median())

    ax.bar(counts["rank"], counts.n_barcodes, width=0.82, color=COLORS["gray"], lw=0)
    ax.axhline(median, color=COLORS["threshold"], lw=0.7, ls=(0, (2.0, 1.6)))
    ax.text(len(counts), median * 1.3, f"median {median:.1f}", ha="right", va="bottom",
            color=COLORS["threshold"])

    top, bottom = counts.iloc[0], counts.iloc[-1]
    ax.annotate(f"{top.domain} {int(top.n_barcodes):,}", xy=(1, top.n_barcodes),
                xytext=(4, top.n_barcodes * 1.4), ha="left", va="bottom")
    ax.annotate(f"{bottom.domain} {int(bottom.n_barcodes)}", xy=(len(counts), bottom.n_barcodes),
                xytext=(len(counts), 4.0), ha="right", va="bottom")

    ax.set_yscale("log")
    ax.set_ylim(2.0, 3200.0)
    ax.set_xlim(0.2, len(counts) + 0.8)
    ax.set_xticks([1, 30, 60, 90])
    ax.set_xlabel("MCP-ZF domain, ranked")
    ax.set_ylabel("Unique barcodes")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    style(ax, grid_axis="y")
    return counts


def draw_barcode_lengths(ax: plt.Axes, data: Path) -> pd.DataFrame:
    """Panel b: how many barcodes the library holds at each of its four lengths."""
    frame = pd.read_csv(data / "annotations" / "barcode_summary_zf.csv")
    lengths = frame.groupby("barcode_length_nt", as_index=False).n_barcodes.sum()
    observed = tuple(lengths.barcode_length_nt.astype(int))
    if observed != BARCODE_LENGTHS:
        raise ValueError(
            f"barcode lengths {observed} do not match {list(BARCODE_LENGTHS)}"
        )

    ax.bar(lengths.barcode_length_nt, lengths.n_barcodes, width=0.68,
           color=COLORS["gray"], lw=0)
    for row in lengths.itertuples():
        ax.text(row.barcode_length_nt, row.n_barcodes, f"{row.n_barcodes:,}",
                ha="center", va="bottom")

    ax.set_xticks(list(BARCODE_LENGTHS))
    ax.set_xlim(min(BARCODE_LENGTHS) - 0.7, max(BARCODE_LENGTHS) + 0.7)
    ax.set_ylim(0, lengths.n_barcodes.max() * 1.22)
    ax.set_xlabel("Barcode length (nt)")
    ax.set_ylabel("Barcodes")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    style(ax)
    return lengths


def read_qc(data: Path) -> pd.DataFrame:
    """The screen as sequenced, one row per domain, arm and replicate."""
    frame = pd.read_csv(data / "annotations" / "zf_qc.csv")
    return frame.loc[frame.replicate.isin(REPLICATES)]


def draw_edit_fraction(ax: plt.Axes, qc: pd.DataFrame) -> pd.DataFrame:
    """Panel c: per-domain edit fraction in each arm, every replicate drawn."""
    jitter = np.random.default_rng(0)
    rows = []
    for position, (arm, label, color) in enumerate(ARMS):
        values = qc.loc[qc.arm == arm, "edit_fraction"].to_numpy(dtype=float)
        drawn = np.where(values <= 0.0, ZERO_FLOOR, values)
        x = position + jitter.uniform(-0.18, 0.18, drawn.size)
        ax.scatter(x, drawn * 100.0, s=1.6, c=color, lw=0, alpha=0.55, zorder=2)
        median = float(np.median(values))
        ax.plot([position - 0.3, position + 0.3], [median * 100.0] * 2,
                color=COLORS["ink"], lw=1.1, zorder=4, solid_capstyle="butt")
        rows.append({"arm": arm, "label": label, "n_observations": int(values.size),
                     "n_zero": int((values <= 0.0).sum()),
                     "median_percent": median * 100.0,
                     "q1_percent": float(np.percentile(values, 25)) * 100.0,
                     "q3_percent": float(np.percentile(values, 75)) * 100.0})

    ax.set_yscale("log")
    ax.set_ylim(ZERO_FLOOR * 100.0 * 0.6, 30.0)
    ax.set_xticks(range(len(ARMS)), [label for _, label, _ in ARMS])
    ax.set_xlim(-0.55, len(ARMS) - 0.45)
    ax.set_xlabel("Pomalidomide")
    ax.set_ylabel("Edit fraction per domain (%)")
    style(ax, grid_axis="y")
    return pd.DataFrame(rows)


def draw_replicate_agreement(ax: plt.Axes, qc: pd.DataFrame) -> pd.DataFrame:
    """Panel d: the six pairwise replicate correlations in each arm, both measures.

    Pearson and Spearman are drawn together because they disagree. Pearson
    separates the arms while Spearman leaves them level, so the gain under
    treatment is carried by a few strongly edited domains.
    """
    rows = []
    offsets = {"Pearson": -0.16, "Spearman": 0.16}
    pairs = list(itertools.combinations(range(len(REPLICATES)), 2))

    for position, (arm, label, color) in enumerate(ARMS):
        matrix = qc.loc[qc.arm == arm].pivot(
            index="gene", columns="replicate", values="edit_fraction").to_numpy()
        measures = {
            "Pearson": [pearsonr(matrix[:, i], matrix[:, j])[0] for i, j in pairs],
            "Spearman": [spearmanr(matrix[:, i], matrix[:, j])[0] for i, j in pairs],
        }
        for measure, values in measures.items():
            x = position + offsets[measure]
            filled = measure == "Pearson"
            ax.scatter(np.full(len(values), x), values, s=7.0,
                       facecolors=color if filled else "none", edgecolors=color,
                       linewidth=0.5, zorder=3)
            ax.plot([x - 0.11, x + 0.11], [np.median(values)] * 2, color=COLORS["ink"],
                    lw=1.1, zorder=4, solid_capstyle="butt")
            for (i, j), value in zip(pairs, values):
                rows.append({"arm": arm, "label": label, "measure": measure,
                             "replicate_a": REPLICATES[i], "replicate_b": REPLICATES[j],
                             "correlation": float(value)})

    ax.axhline(0.0, color=COLORS["gray"], lw=0.5, zorder=1)
    ax.set_xticks(range(len(ARMS)), [label for _, label, _ in ARMS])
    ax.set_xlim(-0.55, len(ARMS) - 0.45)
    ax.set_ylim(-0.25, 0.72)
    ax.set_xlabel("Pomalidomide")
    ax.set_ylabel("Pairwise replicate correlation")
    # Marker fill is all that separates the two measures, so it is named here.
    ax.text(0.03, 0.97, "filled Pearson\nopen Spearman", transform=ax.transAxes,
            ha="left", va="top", color=COLORS["gray"], linespacing=1.35)
    style(ax, grid_axis="y")
    return pd.DataFrame(rows)


def read_propensity(results: Path, data: Path) -> pd.DataFrame:
    """The fitted pomalidomide effect of each domain, joined to the benchmarks.

    The zinc-finger fit is baseline-free, so the dose coefficient is the whole
    log-odds of editing rather than a shift on top of a pair term, and the
    logistic of it is the editing probability pomalidomide accounts for.
    """
    fit = pyroppi.load_fit(results / "zf_pom")
    effects = fit.effects()
    effects = effects.loc[effects.covariate == DOSE].copy()
    effects["gene"] = np.where(effects.protein_a == BAIT, effects.protein_b,
                               effects.protein_a)
    effects["propensity"] = logistic(effects.logit_mean.to_numpy(dtype=float))
    benchmarks = pd.read_csv(data / "annotations" / "degrader_benchmarks.csv")
    return effects.merge(benchmarks, on="gene", how="inner", validate="one_to_one")


def correlations(x: pd.Series, y: pd.Series) -> dict[str, float]:
    rho, rho_p = spearmanr(x, y)
    r, r_p = pearsonr(x, y)
    return {"n": int(len(x)), "spearman_rho": float(rho), "spearman_p": float(rho_p),
            "pearson_r": float(r), "pearson_p": float(r_p)}


def annotate(ax: plt.Axes, result: dict[str, float]) -> None:
    ax.text(0.97, 0.04,
            f"rho = {result['spearman_rho']:.2f} (p = {result['spearman_p']:.2g})\n"
            f"r = {result['pearson_r']:.2f} (p = {result['pearson_p']:.2g})\n"
            f"n = {result['n']}",
            transform=ax.transAxes, ha="right", va="bottom", linespacing=1.35)


def draw_scatter(ax: plt.Axes, x: pd.Series, y: pd.Series, x_label: str,
                 y_label: str, color: str) -> dict[str, float]:
    """One benchmark against another, with both correlations in the corner."""
    ax.scatter(x, y, s=11, color=color, edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    for axis, values in ((ax.xaxis, x), (ax.yaxis, y)):
        low, high = float(values.min()), float(values.max())
        margin = 0.12 * (high - low)
        (ax.set_xlim if axis is ax.xaxis else ax.set_ylim)(low - margin, high + margin)
        axis.set_major_locator(mticker.MaxNLocator(4))
    result = correlations(x, y)
    annotate(ax, result)
    style(ax)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data",
                        help="directory holding the dataset tables")
    parser.add_argument("--results", type=Path, default=REPO / "results",
                        help="directory holding the saved fits")
    parser.add_argument("--output", type=Path, default=REPO / "figures",
                        help="directory to write the figure into")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    use_style()
    fig = figure(183, 112)
    top = fig.add_gridspec(1, 3, left=0.065, right=0.99, top=0.93, bottom=0.60,
                           width_ratios=[1.35, 0.82, 0.83], wspace=0.40)
    bottom = fig.add_gridspec(1, 3, left=0.065, right=0.99, top=0.44, bottom=0.10,
                              width_ratios=[0.9, 1.05, 1.05], wspace=0.40)
    ax_a, ax_b, ax_c = (fig.add_subplot(top[i]) for i in range(3))
    ax_d, ax_e, ax_f = (fig.add_subplot(bottom[i]) for i in range(3))

    counts = draw_barcodes_per_domain(ax_a, args.data)
    lengths = draw_barcode_lengths(ax_b, args.data)
    qc = read_qc(args.data)
    edit = draw_edit_fraction(ax_c, qc)
    agreement = draw_replicate_agreement(ax_d, qc)

    propensity = read_propensity(args.results, args.data)
    nguyen = propensity.dropna(subset=["nguyen_2024_l2fc_depletion"])
    both = nguyen.dropna(subset=["sievers_pom_fold_depletion"])
    fitted = draw_scatter(ax_e, nguyen.propensity * 100,
                          nguyen.nguyen_2024_l2fc_depletion,
                          "Pom-dependent editing (%)", "Nguyen et al.\nZF-GFP depletion",
                          COLORS["pom_3"])
    published = draw_scatter(ax_f, both.sievers_pom_fold_depletion,
                             both.nguyen_2024_l2fc_depletion,
                             "Sievers et al.\nZF-GFP depletion",
                             "Nguyen et al.\nZF-GFP depletion", COLORS["gray"])

    for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d"),
                       (ax_e, "e"), (ax_f, "f")):
        panel_letter(ax, letter)

    fig.savefig(args.output / "supp_figure_1.png")
    save(fig, args.output / "supp_figure_1.pdf")

    print("panel a: barcodes per zinc-finger domain")
    print(f"  {len(counts)} domains, {int(counts.n_barcodes.sum()):,} barcodes, "
          f"median {counts.n_barcodes.median():.1f}")
    print(f"  most {counts.domain.iloc[0]} {int(counts.n_barcodes.iloc[0]):,}, "
          f"least {counts.domain.iloc[-1]} {int(counts.n_barcodes.iloc[-1])}", end="\n\n")

    print("panel b: barcodes at each length")
    print(lengths.to_string(index=False), end="\n\n")

    print(f"panel c: edit fraction per domain, replicates {REPLICATES}")
    print(edit.to_string(index=False), end="\n\n")

    print("panel d: pairwise replicate correlations")
    print(agreement.to_string(index=False))
    medians = agreement.groupby(["arm", "measure"]).correlation.median()
    print(medians.round(3).to_string(), end="\n\n")

    print(f"panel e: {DOSE} editing against Nguyen et al.")
    print(f"  {fitted}", end="\n\n")
    print("panel f: Sievers et al. against Nguyen et al.")
    print(f"  {published}", end="\n\n")
    print(nguyen[["gene", "propensity", "nguyen_2024_l2fc_depletion",
                  "sievers_pom_fold_depletion"]].sort_values(
        "nguyen_2024_l2fc_depletion").to_string(index=False))


if __name__ == "__main__":
    main()
