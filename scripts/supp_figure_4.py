#!/usr/bin/env python
"""Supplementary Figure 4: rapamycin-induced proximity as a library control.

Both fusion halves carry a dimerisation domain, TadA8e-FRB and MCP-FKBP, so
rapamycin brings the editor and the reporter together whether or not the two
proteins of interest bind. The treated arm is a positive control every pair in
the library shares, and it is not used to score interactions: the propensities
in Figure 3 come from the untreated arm.

The panel plots the ClinORF fit's own 3 h drug term, the shift in the log-odds
of editing that the treatment window contributes to a pair over and above that
pair's baseline. Previously reported interactions carry a smaller shift than
other pairs, tested one-sided in that direction. Control pairs and self-pairs
are left out: neither is a candidate interaction.

Posterior spread is not drawn; the panel shows the spread across pairs instead.

    python supp_figure_4.py
    python supp_figure_4.py --data /path/to/data --results /path/to/results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from pyroppi import load_fit
from pyroppi.plotting import COLORS, figure, save, use_style

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

CONTROLS = ("GFP", "mCherry")
WINDOW = "3h"
GROUPS = (
    ("Previously reported", True, COLORS["hit"], 5.0),
    ("Other pairs", False, COLORS["control"], 2.0),
)


def reported_pairs(data: Path) -> set[frozenset[str]]:
    """The interactions already in the literature: two or more publications."""
    table = pd.read_csv(data / "annotations" / "known_interactions.csv")
    marked = table.loc[table["known"]]
    return {frozenset((row.protein_a, row.protein_b)) for row in marked.itertuples()}


def drug_terms(results: Path, data: Path) -> pd.DataFrame:
    """The 3 h drug term of every heterotypic ORF pair, with reported flagged."""
    effects = load_fit(results / "clinorf").effects()
    terms = effects.loc[(effects["block"] == "drug") & (effects["covariate"] == WINDOW)]
    control = terms.protein_a.isin(CONTROLS) | terms.protein_b.isin(CONTROLS)
    heterotypic = terms.loc[~control & (terms.protein_a != terms.protein_b)].copy()
    marked = reported_pairs(data)
    heterotypic["reported"] = [
        frozenset((row.protein_a, row.protein_b)) in marked
        for row in heterotypic.itertuples()
    ]
    return heterotypic.reset_index(drop=True)


def draw_response(ax: plt.Axes, terms: pd.DataFrame) -> dict:
    """One point per pair in each group, with the group median ruled across it."""
    jitter = np.random.default_rng(0)
    reported = terms["reported"].to_numpy()
    effect = terms["logit_mean"].to_numpy(dtype=float)
    p_value = float(
        mannwhitneyu(effect[reported], effect[~reported], alternative="less").pvalue
    )

    for position, (label, flag, colour, size) in enumerate(GROUPS):
        mask = reported if flag else ~reported
        x = position + jitter.uniform(-0.09, 0.09, int(mask.sum()))
        ax.scatter(x, effect[mask], s=size, c=colour, linewidth=0.0,
                   zorder=3 if flag else 2)
        ax.plot([position - 0.17, position + 0.17], [np.median(effect[mask])] * 2,
                color=COLORS["ink"], lw=1.0, zorder=4, solid_capstyle="butt")

    ax.annotate(f"P = {p_value:.1g}", (0.5, effect.max()), xytext=(0, 5),
                textcoords="offset points", ha="center", color=COLORS["ink"])
    ax.set_xticks(range(len(GROUPS)), [label for label, *_ in GROUPS])
    ax.set_xlim(-0.55, len(GROUPS) - 0.45)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("Rapamycin effect (log-odds)")
    ax.margins(y=0.18)

    return {
        "n_reported": int(reported.sum()),
        "n_other": int((~reported).sum()),
        "median_reported": float(np.median(effect[reported])),
        "median_other": float(np.median(effect[~reported])),
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data",
                        help="directory holding the dataset tables")
    parser.add_argument("--results", type=Path, default=REPO / "results",
                        help="directory holding the fitted results")
    parser.add_argument("--output", type=Path, default=REPO / "figures",
                        help="directory to write the figure into")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    terms = drug_terms(args.results, args.data)

    use_style()
    fig = figure(89, 62)
    ax = fig.add_axes([0.17, 0.12, 0.80, 0.84])
    summary = draw_response(ax, terms)

    fig.savefig(args.output / "supp_figure_4.png")
    save(fig, args.output / "supp_figure_4.pdf")

    print(f"ClinORF {WINDOW} drug term, heterotypic ORF pairs only")
    print(f"  previously reported: n = {summary['n_reported']}, "
          f"median {summary['median_reported']:+.3f}")
    print(f"  other pairs:         n = {summary['n_other']}, "
          f"median {summary['median_other']:+.3f}")
    print(f"  one-sided Mann-Whitney U, reported lower: P = {summary['p_value']:.3g}")
    print()
    print("the reported pairs and their terms")
    marked = terms.loc[terms["reported"]].sort_values("logit_mean")
    for row in marked.itertuples():
        print(f"  {row.protein_a}-{row.protein_b:<10s} {row.logit_mean:+.3f}")


if __name__ == "__main__":
    main()
