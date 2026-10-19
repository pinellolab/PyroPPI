"""Fit PyroPPI to the nine-protein RAD51 paralog matrix and write the figures.

Reads the dataset, fits the model, calls hits against the GFP and mCherry
controls, and writes four PNGs beside this file.
`rad51x9.ipynb` is the same analysis with the commentary.

    python rad51x9.py                  # data from the repository
    python rad51x9.py --data /path/to/rad51x9
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import pyroppi
from pyroppi import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# Okabe-Ito, the colourblind-safe palette the manuscript figures use.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
AMBER = "#E69F00"
GREY = "#6F6F6F"
PALE = "#D9D9D9"

HEAT = LinearSegmentedColormap.from_list(
    "ppi_recorder", ["#F7F7F7", "#9ECAE1", BLUE, "#08306B"]
)

# The nine proteins in the order the matrix figure reads, controls last.
ORDER = [
    "RAD51",
    "RAD51B",
    "RAD51C",
    "RAD51D",
    "XRCC2",
    "XRCC3",
    "SFR1",
    "GFP",
    "mCherry",
]

DRUG = ["1h", "3h", "HCT_x_1h", "HCT_x_3h"]
CELL = ["HCT"]
CONTROLS = ("GFP", "mCherry")

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


def label(row) -> str:
    """Name a canonical pair, writing a self-pair once."""
    a, b = row["protein_a"], row["protein_b"]
    return a if a == b else f"{a}-{b}"


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


def plot_matrix(pairs, path):
    """Symmetric nine-by-nine heatmap of interaction propensity."""
    propensity = {}
    for _, row in pairs.iterrows():
        propensity[(row.protein_a, row.protein_b)] = row.propensity_mean
        propensity[(row.protein_b, row.protein_a)] = row.propensity_mean
    grid = np.array([[propensity[(a, b)] for b in ORDER] for a in ORDER])

    called = set()
    for _, row in pairs[pairs.hit].iterrows():
        called.add((row.protein_a, row.protein_b))
        called.add((row.protein_b, row.protein_a))

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    image = ax.imshow(grid * 100, cmap=HEAT, vmin=0)
    ax.set_xticks(range(len(ORDER)), ORDER, rotation=90)
    ax.set_yticks(range(len(ORDER)), ORDER)
    ax.set_xticks(np.arange(len(ORDER) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(ORDER) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, a in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            if (a, b) in called:
                ax.plot(j, i, marker="o", ms=3.5, mfc="none", mec=VERMILLION, mew=1.0)

    bar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    bar.set_label("interaction propensity (% edited)")
    bar.outline.set_visible(False)
    ax.set_title("a ring marks a pair called against the controls", loc="left", pad=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ranked(pairs, threshold, path):
    """Every canonical pair ranked by propensity, with the control threshold."""
    ranked = pairs.sort_values("propensity_mean").reset_index(drop=True)
    is_control = ranked.protein_a.isin(CONTROLS) | ranked.protein_b.isin(CONTROLS)
    colour = np.where(is_control, GREY, np.where(ranked.hit, BLUE, PALE))

    fig, ax = plt.subplots(figsize=(4.4, 6.4))
    y = np.arange(len(ranked))
    ax.hlines(y, 0, ranked.propensity_mean * 100, color=colour, lw=1.0)
    ax.scatter(ranked.propensity_mean * 100, y, c=colour, s=14, zorder=3)
    ax.axvline(threshold * 100, color=VERMILLION, ls="--", lw=0.9)
    ax.text(
        threshold * 100,
        len(ranked) - 0.2,
        "  control threshold",
        color=VERMILLION,
        va="top",
        ha="left",
    )
    ax.set_yticks(y, [label(row) for _, row in ranked.iterrows()])
    ax.set_ylim(-1, len(ranked))
    ax.set_xlabel("interaction propensity (% edited)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BLUE, label="called pair"),
        plt.Line2D([], [], marker="o", ls="", color=PALE, label="other pair"),
        plt.Line2D([], [], marker="o", ls="", color=GREY, label="control pair"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_effects(effects, parts, sample, path):
    """The covariate shifts, and how one sample's fitted mean is composed."""
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.0, 3.4), gridspec_kw={"width_ratios": [1, 1.4]}
    )

    names = DRUG + CELL
    for x, name in enumerate(names):
        values = effects.loc[effects.covariate == name, "logit_mean"]
        jitter = np.random.default_rng(0).normal(0, 0.06, len(values))
        left.scatter(x + jitter, values, s=10, color=BLUE, alpha=0.5, lw=0)
        left.hlines(values.median(), x - 0.3, x + 0.3, color=VERMILLION, lw=1.4)
    left.axhline(0, color=GREY, lw=0.6)
    left.set_xticks(range(len(names)), names, rotation=45, ha="right")
    left.set_ylabel("shift in log-odds of editing")
    left.set_title("covariate shifts, one point per pair", loc="left")

    top = parts.sort_values("contrib_pair", ascending=False).head(12)[::-1]
    y = np.arange(len(top))
    pair = top.contrib_pair * 100
    drug = top.contrib_drug * 100
    cell = top.contrib_cell * 100
    right.barh(y, pair, color=BLUE, label="pair baseline")
    right.barh(y, drug, left=pair, color=AMBER, label="drug block (3 h)")
    right.barh(y, cell, left=pair + drug, color=GREEN, label="cell block (HCT116)")
    right.scatter(
        top.edited / top.total * 100, y, color="black", s=12, zorder=3, label="observed"
    )
    right.set_yticks(y, [f"{b}-{p}" for b, p in zip(top.bait, top.prey)])
    right.set_xlabel("edited reporter reads (%)")
    right.set_title(f"composition in {sample}", loc="left")
    right.set_xlim(0, max(top.mu.max(), (top.edited / top.total).max()) * 100 * 1.08)
    right.tick_params(axis="y", length=0)
    right.spines["left"].set_visible(False)
    right.legend(fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.42))

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO / "data" / "rad51x9",
        help="directory holding counts.csv and samples.csv",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=HERE,
        help="directory to write the PNG figures into",
    )
    parser.add_argument("--steps", type=int, default=30_000, help="SVI steps")
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)

    counts, samples = pyroppi.read_dataset(args.data)
    print(f"{len(counts)} observations, {len(samples)} samples")
    print(f"{counts.groupby(['bait', 'prey']).ngroups} measured pairs")
    print(samples.to_string(index=False), end="\n\n")

    start = time.perf_counter()
    result = pyroppi.fit(counts, samples, drug=DRUG, cell=CELL, num_steps=args.steps)
    print(f"\nfitted in {time.perf_counter() - start:.0f} s")
    print(f"{len(result.canonical)} canonical pairs")
    print(f"final ELBO loss {result.losses[-1]:,.0f}\n")

    pairs = result.pairs()
    tested = stats.control_test(pairs, controls=CONTROLS)
    threshold = stats.control_threshold(pairs, controls=CONTROLS, q=0.95)
    hits = tested[tested.p < 0.05].sort_values("p")

    pairs["pair"] = [label(row) for _, row in pairs.iterrows()]
    pairs["hit"] = pairs.index.isin(hits.index)
    ranked = pairs.sort_values("propensity_mean", ascending=False)
    columns = ["pair", "propensity_mean", "propensity_q05", "propensity_q95"]
    print("top ten canonical pairs by interaction propensity")
    print(ranked[columns].head(10).to_string(index=False), end="\n\n")

    print(f"control threshold: {threshold:.4f} ({threshold * 100:.2f}% edited)")
    print(f"{len(hits)} of {len(tested)} non-control pairs called")
    hits = hits.copy()
    hits["pair"] = [label(row) for _, row in hits.iterrows()]
    print(hits[["pair", "propensity_mean", "z", "p"]].to_string(index=False), end="\n\n")

    effects = result.effects()
    summary = effects.groupby("covariate").logit_mean.agg(["mean", "min", "max"])
    print(f"covariate shifts in log-odds, across {len(result.canonical)} canonical pairs")
    print(summary.loc[DRUG + CELL].to_string(), end="\n\n")

    contributions = result.contributions()
    sample = "HCT_RAD51x9_RepA_3h"
    parts = contributions[contributions["sample"] == sample]

    plot_convergence(result.losses, args.figures / "convergence.png")
    plot_matrix(pairs, args.figures / "matrix.png")
    plot_ranked(pairs, threshold, args.figures / "ranked_pairs.png")
    plot_effects(effects, parts, sample, args.figures / "effects.png")
    print(f"figures written to {args.figures}")


if __name__ == "__main__":
    main()
