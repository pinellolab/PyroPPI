#!/usr/bin/env python
"""Supplementary Figure 3 data panels: the localization screen as built and as measured.

Panels a and b describe the MCP-ORFeome barcode dictionary, and panels c and d
the editing baseline of the eight TadA8e localization tags with the correction
the fit applies to it. A tag is a construct, and constructs do not all edit at
the same rate, so nothing can be compared across tags until that offset is
dealt with.

    python supp_figure_3.py
    python supp_figure_3.py --data /path/to/data --results /path/to/results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import pyroppi
from pyroppi.plotting import COLORS, figure, panel_letter, save, use_style

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

ORF_BINS = 22
# The last coverage bin is open above 5,000 nucleotides, so every hORFeome clone
# falls in exactly one bin. An axis cannot run to infinity, and the bin holds no
# represented clone, so it is drawn to a finite right edge.
LENGTH_OPEN_EDGE = 6000.0

TAGS = ["MEM", "ERM", "LD", "RAB10", "NES", "ACT", "NLS", "NOLS"]
TIMEPOINTS = [("raw_1h", "1 h"), ("raw_3h", "3 h")]
TIMEPOINT_COLORS = {"raw_1h": COLORS["grid"], "raw_3h": COLORS["rad51d"]}
# The pale fill of the 1 h box is too light to set type in, so its name takes
# the ink the rest of the figure gives neutral marks.
TIMEPOINT_TEXT = {"raw_1h": COLORS["gray"], "raw_3h": COLORS["rad51d"]}
RATE_TICKS = [0.0, 0.04, 0.08, 0.12, 0.16]
# Cut to the mass of both distributions. A few pairs reach past 0.6, so an axis
# holding every point would put all eight boxes in its left tenth. Every box is
# inside the view and it is the whiskers that meet the spine.
RATE_XLIM = (0.0, 0.16)
BOX_WIDTH = 0.34


# ---------------------------------------------------------------------------
# Panels a and b: the barcode dictionary
# ---------------------------------------------------------------------------


def draw_depth(ax: plt.Axes, cumulative_ax: plt.Axes, table: pd.DataFrame) -> dict:
    """Panel a: barcodes per ORF on a log axis, over a cumulative curve.

    The histogram says where the mass of the library is and the curve says how
    much of it sits at or below any given depth, which is what a reader deciding
    whether to trust a per-ORF measurement needs.
    """
    counts = table["n_barcodes"].to_numpy()
    ax.hist(counts, bins=np.logspace(0, np.log10(counts.max()), ORF_BINS + 1),
            color=COLORS["gray"], lw=0, zorder=2)

    q25, median, q75 = np.quantile(counts, [0.25, 0.5, 0.75])
    ax.axvspan(q25, q75, facecolor=COLORS["threshold"], alpha=0.10, lw=0, zorder=0)
    ax.axvline(median, color=COLORS["threshold"], lw=0.7, ls=(0, (2.0, 1.6)), zorder=3)
    ax.text(0.97, 0.78, f"median {median:.0f}\nIQR {q25:.0f} to {q75:.0f}",
            transform=ax.transAxes, ha="right", va="top",
            color=COLORS["threshold"], linespacing=1.2)

    ax.set_xscale("log")
    ax.set_xlim(1.0, counts.max() * 1.1)
    ax.set_xlabel("Barcodes per ORF")
    ax.set_ylabel("ORFs")
    ax.xaxis.set_major_locator(mticker.FixedLocator([1, 10, 100]))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.tick_params(axis="both", pad=2)

    order = np.sort(counts)
    cumulative_ax.step(order, 100.0 * np.arange(1, order.size + 1) / order.size,
                       where="post", color=COLORS["rad51d"], lw=0.9, zorder=4)
    cumulative_ax.set_xscale("log")
    cumulative_ax.set_xlim(*ax.get_xlim())
    cumulative_ax.set_ylim(0, 100)
    cumulative_ax.set_yticks([0, 50, 100])
    cumulative_ax.set_ylabel("Cumulative ORFs (%)", color=COLORS["rad51d"])
    cumulative_ax.tick_params(axis="y", colors=COLORS["rad51d"], pad=2)
    cumulative_ax.spines["right"].set_color(COLORS["rad51d"])
    for side in ("top", "left"):
        cumulative_ax.spines[side].set_visible(False)
    cumulative_ax.patch.set_visible(False)

    deepest = table.loc[table["n_barcodes"].idxmax()]
    return {"n_clones": int(counts.size), "n_barcodes": int(counts.sum()),
            "median": float(median), "q25": float(q25), "q75": float(q75),
            "maximum": int(counts.max()), "deepest_gene": str(deepest["gene"]),
            "below_2": int((counts < 2).sum()), "below_5": int((counts < 5).sum()),
            "below_10": int((counts < 10).sum())}


def draw_coverage(ax: plt.Axes, coverage: pd.DataFrame) -> int:
    """Panel b: the share of hORFeome clones the dictionary represents, by length.

    A step across the bin edges rather than a line through their midpoints: the
    bins are 100 nucleotides wide below a kilobase and a kilobase wide above it,
    and a line between midpoints would hide that.
    """
    edges = np.append(coverage["length_from_nt"].to_numpy(), LENGTH_OPEN_EDGE)
    values = np.append(100.0 * coverage["fraction_represented"].to_numpy(),
                       100.0 * coverage["fraction_represented"].iloc[-1])
    ax.step(edges, values, where="post", color=COLORS["gray"], lw=1.0, zorder=3)
    ax.fill_between(edges, 0.0, values, step="post", color=COLORS["gray"],
                    alpha=0.18, lw=0, zorder=2)

    ax.set_xlim(0, float(edges[-1]))
    ax.set_ylim(0, 40)
    ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000],
                  ["0", "1k", "2k", "3k", "4k", "5k"])
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_xlabel("ORF length (nt)")
    ax.set_ylabel("hORFeome clones\nrepresented (%)")
    ax.tick_params(axis="both", pad=2)
    return int(coverage["n_horfeome_clones"].sum())


# ---------------------------------------------------------------------------
# Panels c and d: the tag baseline
# ---------------------------------------------------------------------------


def tag_table(data: Path, results: Path) -> pd.DataFrame:
    """One row per labelled tag-ORF pair: the raw rates and the fitted effect.

    The raw rate of a pair at one timepoint is the mean of its per-sample edited
    fraction over the samples of that timepoint. The fitted effect is the drug
    contribution taken against the headroom above the pair's own baseline, so a
    tag that edits heavily at baseline and one that does not are put on a single
    scale. Every 3 h sample of a tag carries the same design row, so both terms
    are constant over those samples and averaging only pools replicates.

    The cohort is the pairs whose ORF the Surfaceome and Gene Ontology join
    places in one of the two compartment classes, which is the cohort the
    benchmark in Figure 2 is scored on.
    """
    counts, samples = pyroppi.read_dataset(data / "localization")
    counts["rate"] = counts["edited"] / counts["total"]
    treated = counts["sample"].map(samples.set_index("sample")["3H"]).astype(bool)
    counts["timepoint"] = np.where(treated, "raw_3h", "raw_1h")
    observed = counts.pivot_table(index=["bait", "prey"], columns="timepoint",
                                  values="rate", aggfunc="mean").reset_index()

    fit = pyroppi.load_fit(results / "localization")
    design = fit.samples
    three = set(design.loc[design["3H"] == 1, "sample"])
    parts = fit.contributions()
    parts = parts[parts["sample"].isin(three)]
    fitted = parts.groupby(["bait", "prey"]).agg(
        contrib_drug=("contrib_drug", "mean"),
        contrib_pair=("contrib_pair", "mean"),
    ).reset_index()
    fitted["drug_effect"] = fitted["contrib_drug"] / (1.0 - fitted["contrib_pair"])

    labels = pd.read_csv(data / "annotations" / "localization_labels.csv")
    merged = observed.merge(fitted[["bait", "prey", "drug_effect"]],
                            on=["bait", "prey"], how="inner")
    merged = merged.merge(labels[["orf", "orf_class"]], left_on="prey",
                          right_on="orf", how="left")
    merged = merged[merged["orf_class"].isin(["CY", "NM"])]
    return merged.dropna(subset=["raw_3h", "drug_effect"]).rename(columns={"bait": "tag"})


def box_row(ax: plt.Axes, values: np.ndarray, position: float, color: str,
            width: float = BOX_WIDTH) -> float:
    """One horizontal box with 1.5 IQR whiskers; returns the right whisker end."""
    artists = ax.boxplot([values], positions=[position], widths=[width],
                         orientation="horizontal", showfliers=False,
                         patch_artist=True, manage_ticks=False)
    for patch in artists["boxes"]:
        patch.set(facecolor=color, edgecolor=COLORS["gray"], linewidth=0.4)
    for key in ("whiskers", "caps"):
        for line in artists[key]:
            line.set(color=COLORS["gray"], linewidth=0.4)
    for line in artists["medians"]:
        line.set(color=COLORS["ink"], linewidth=0.8)
    return max(line.get_xdata().max() for line in artists["whiskers"])


def quartiles(tag: str, readout: str, values: np.ndarray) -> dict:
    return {"tag": tag, "readout": readout, "n": int(values.size),
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75))}


def draw_raw_rates(ax: plt.Axes, table: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Panel c: the raw editing rate of each tag at both timepoints."""
    rows, ends = [], {}
    for index, tag in enumerate(order):
        sub = table[table["tag"] == tag]
        ends[index] = {}
        for offset, (column, _) in zip((0.20, -0.20), TIMEPOINTS):
            values = sub[column].dropna().to_numpy(float)
            end = box_row(ax, values, index + offset, TIMEPOINT_COLORS[column])
            rows.append(quartiles(tag, column, values))
            ends[index][column] = (end, index + offset)

    # The two timepoints are named beside their own boxes rather than in a
    # legend. Two rows apart is about one line of type, so the row chosen is the
    # one that separates the two whisker ends furthest while keeping both inside
    # the view, which separates the names horizontally instead.
    reach = 0.85 * RATE_XLIM[1]
    usable = {index: row for index, row in ends.items()
              if max(end for end, _ in row.values()) < reach}
    chosen = max(usable, key=lambda index: abs(
        usable[index]["raw_3h"][0] - usable[index]["raw_1h"][0]))
    for column, name in TIMEPOINTS:
        end, y = usable[chosen][column]
        ax.text(end + 0.004, y, name, ha="left", va="center",
                color=TIMEPOINT_TEXT[column])

    ax.set_yticks(range(len(order)), order)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(*RATE_XLIM)
    ax.set_xticks(RATE_TICKS)
    ax.set_xlabel("Raw editing rate")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="both", pad=2)
    return pd.DataFrame(rows)


def draw_drug_effect(ax: plt.Axes, table: pd.DataFrame,
                     order: list[str]) -> pd.DataFrame:
    """Panel d: the fitted drug effect of the same pairs, on panel c's rows.

    The fit carries one drug level, with 1 h as the reference and 3 h as the
    effect, so this is the whole of the drug block and panel c's two boxes are
    the states it is measured between. It is a posterior mean, so a pair
    measured on a few dozen reads is shrunk toward its tag rather than reported
    as the hard zero its raw rate would be.
    """
    rows = []
    for index, tag in enumerate(order):
        values = table.loc[table["tag"] == tag, "drug_effect"].dropna().to_numpy(float)
        box_row(ax, values, index, COLORS["drug"], width=BOX_WIDTH * 1.5)
        rows.append(quartiles(tag, "drug_effect", values))
    ax.set_yticks(range(len(order)), [])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(*RATE_XLIM)
    ax.set_xticks(RATE_TICKS)
    ax.set_xlabel("Drug effect")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="both", pad=2)
    return pd.DataFrame(rows)


def report_span(summary: pd.DataFrame, readout: str, name: str) -> None:
    """Print the smallest and largest per-tag median of one readout."""
    medians = summary.loc[summary["readout"] == readout, "median"]
    low, high = medians.min() * 100, medians.max() * 100
    print(f"  {name}: per-tag medians {low:.2f}% and {high:.2f}%, "
          f"a span of {high - low:.2f} percentage points")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data",
                        help="directory holding the dataset and annotation tables")
    parser.add_argument("--results", type=Path, default=REPO / "results",
                        help="directory holding the fits")
    parser.add_argument("--output", type=Path, default=REPO / "figures",
                        help="directory to write the figure into")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    use_style()
    width, height = 183.0, 112.0
    fig = figure(width, height)

    def place(left: float, bottom: float, box_width: float, box_height: float):
        return fig.add_axes([left / width, bottom / height,
                             box_width / width, box_height / height])

    ax_a = place(17, 74, 56, 32)
    ax_b = place(110, 74, 56, 32)
    ax_c = place(17, 12, 64, 48)
    ax_d = place(101, 12, 64, 48)
    ax_a_cumulative = ax_a.twinx()

    annotations = args.data / "annotations"
    depth = draw_depth(ax_a, ax_a_cumulative,
                       pd.read_csv(annotations / "barcode_summary_orfeome.csv"))
    coverage = pd.read_csv(annotations / "orfeome_length_coverage.csv")
    n_horfeome = draw_coverage(ax_b, coverage)

    table = tag_table(args.data, args.results)
    order = list(table.groupby("tag")["raw_3h"].median()
                 .sort_values(ascending=False).index)
    raw = draw_raw_rates(ax_c, table, order)
    fitted = draw_drug_effect(ax_d, table, order)

    for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")):
        panel_letter(ax, letter)
    fig.savefig(args.output / "supp_figure_3.png")
    save(fig, args.output / "supp_figure_3.pdf")

    print("panel a: the MCP-ORFeome barcode dictionary")
    print(f"  {depth['n_barcodes']:,} barcodes over {depth['n_clones']:,} clones")
    print(f"  median {depth['median']:.0f}, IQR {depth['q25']:.0f} to "
          f"{depth['q75']:.0f}, deepest {depth['deepest_gene']} at "
          f"{depth['maximum']}")
    print(f"  {depth['below_2']} clones carry one barcode, {depth['below_5']} "
          f"fewer than 5 and {depth['below_10']} fewer than 10")
    print()

    print(f"panel b: coverage of {n_horfeome:,} hORFeome v8.1 clones by length bin")
    print(coverage.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print()

    summary = pd.concat([raw, fitted], ignore_index=True)
    print(f"panels c and d: {len(table):,} labelled pairs over "
          f"{table['prey'].nunique():,} ORFs and {len(order)} tags")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    for readout, name in (("raw_1h", "raw 1 h rate"), ("raw_3h", "raw 3 h rate"),
                          ("drug_effect", "drug effect")):
        report_span(summary, readout, name)
    negative = int((table["drug_effect"] < 0).sum())
    print(f"  the drug effect is signed: {negative} of {len(table):,} pairs are "
          f"below zero, the lowest at {table['drug_effect'].min():.4f}")


if __name__ == "__main__":
    main()
