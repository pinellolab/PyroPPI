#!/usr/bin/env python
"""Supplementary Figure 9: how each pulldown arm separates the control classes.

The pool holds no GFP or mCherry constructs, so the only control contrast it
offers is stop-gain against synonymous. Each sublibrary gets three strip columns,
one per channel, and a receiver operating characteristic panel that puts the
three curves on one square. A stop-gain is the positive class and a low score is
the positive call, so a curve above the diagonal means the channel orders the two
control classes in the expected direction.

Scores are raw. Centring each channel on its synonymous pool would put the
reference at zero by construction, and the reference is what these panels draw,
so every strip column carries the synonymous median and interquartile range
behind it instead.

    python supp_figure_9.py
    python supp_figure_9.py --output /tmp/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats as scipy_stats

from figure_5 import (
    DEFAULT_DATA,
    DEFAULT_OUTPUT,
    DEFAULT_RESULTS,
    build,
    place,
    sublibrary,
    write,
)
from pyroppi import stats
from pyroppi.plotting import COLORS, figure, panel_letter, use_style

GENES = ("RAD51D", "XRCC2")
POSITIVE, NEGATIVE = "stop", "synonymous"
CLASS_ORDER = ("synonymous", "missense", "stop")
CLASS_TICK = {"synonymous": "Syn.", "missense": "Missense", "stop": "Stop"}
CLASS_MARKER = {"synonymous": "D", "missense": "o", "stop": "s"}
CLASS_SIZE = {"synonymous": 7.0, "missense": 3.0, "stop": 8.0}
CLASS_COLOUR = {
    "synonymous": COLORS["synonymous"],
    "missense": COLORS["missense"],
    "stop": COLORS["stop_gain"],
}

# The three channels, in the order the columns take them, with the dash pattern
# each curve carries so the panel reads without colour.
CHANNELS = (
    ("propensity", "Interaction propensity", "Propensity", COLORS["hit"], ()),
    ("binding", "Binding (log$_2$ ratio)", "Binding", COLORS["alphamissense"], (0, (3.0, 1.6))),
    ("stability", "Stability (log$_2$ ratio)", "Stability", COLORS["gray"], (0, (4.0, 1.3, 1.0, 1.3))),
)

JITTER_SEED = 0
JITTER_WIDTH = 0.17
BOX_WIDTH = 0.52
# Kept clear at the top of every strip column for its two-line annotation, as a
# share of the drawn range, so the six columns put their statistics in one place.
ANNOTATION_HEADROOM = 0.26

WIDTH_MM = 183.0
HEIGHT_MM = 120.0
STRIP_WIDTH_MM = 34.0
STRIP_HEIGHT_MM = 42.0
ROC_SIZE_MM = 34.0
COLUMN_X_MM = (16.0, 58.0, 100.0, 142.0)
ROW_Y_MM = (66.0, 10.0)
LETTERS = {"RAD51D": ("a", "b", "c", "d"), "XRCC2": ("e", "f", "g", "h")}


def discrimination(frame: pd.DataFrame, gene: str) -> pd.DataFrame:
    """Area under the curve and the one-sided p for each channel of one sublibrary.

    The area is the chance a synonymous variant scores above a stop-gain, so 1.0
    means the channel orders the two classes in the expected direction and a
    value below 0.5 means it runs backwards. The p-value is the one-sided
    Mann-Whitney U for the stop-gains sitting below the synonymous pool.
    """
    rows = []
    for column, axis_label, short_label, _, _ in CHANNELS:
        synonymous = frame.loc[frame["class"].eq(NEGATIVE), column].to_numpy(float)
        stop_gain = frame.loc[frame["class"].eq(POSITIVE), column].to_numpy(float)
        # ``auroc`` returns the chance the first group scores above the second,
        # so the synonymous pool goes first and the area reads in the expected
        # direction. Its own p is two-sided and the one wanted here is not, so
        # the one-sided test is run beside it.
        area, _ = stats.auroc(synonymous, stop_gain)
        statistic, p_value = scipy_stats.mannwhitneyu(
            stop_gain, synonymous, alternative="less"
        )
        rows.append(
            {
                "gene": gene,
                "channel": short_label,
                "axis_label": axis_label,
                "column": column,
                "n_synonymous": int(synonymous.size),
                "n_stop_gain": int(stop_gain.size),
                "auroc": area,
                "mannwhitney_u": float(statistic),
                "p_one_sided_lower": float(p_value),
                "median_synonymous": float(np.median(synonymous)),
                "median_stop_gain": float(np.median(stop_gain)),
            }
        )
    return pd.DataFrame(rows)


def roc_curve(negative: np.ndarray, positive: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False- and true-positive rates over every threshold, both ends included.

    A stop-gain is the positive class and a low score is the positive call, so
    the sweep takes each distinct score in turn as the largest score called.
    """
    thresholds = np.unique(np.concatenate([negative, positive]))
    false_rate = [0.0] + [float(np.mean(negative <= t)) for t in thresholds] + [1.0]
    true_rate = [0.0] + [float(np.mean(positive <= t)) for t in thresholds] + [1.0]
    return np.asarray(false_rate), np.asarray(true_rate)


def coincident_sets(curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> list[str]:
    """One line per set of channels whose curves fall on exactly the same path.

    Two channels that separate the classes perfectly trace the same curve, and
    the one drawn second hides the one drawn first, so the panel says which
    channels are under each other.
    """
    paths: dict[tuple, list[str]] = {}
    for label, (false_rate, true_rate) in curves.items():
        paths.setdefault((tuple(false_rate), tuple(true_rate)), []).append(label)
    return [" and ".join(labels) + "\ncoincide" for labels in paths.values() if len(labels) > 1]


def draw_strip(ax: plt.Axes, frame: pd.DataFrame, column: str, label: str, found: pd.Series) -> None:
    """One channel's three variant classes, over the synonymous reference band."""
    rng = np.random.default_rng(JITTER_SEED)
    synonymous = frame.loc[frame["class"].eq(NEGATIVE), column].to_numpy(float)
    low, high = np.quantile(synonymous, [0.25, 0.75])
    ax.axhspan(low, high, facecolor=CLASS_COLOUR[NEGATIVE], alpha=0.15, edgecolor="none", zorder=0)
    ax.axhline(
        float(np.median(synonymous)),
        color=CLASS_COLOUR[NEGATIVE],
        lw=0.7,
        zorder=0.5,
    )

    for index, variant_class in enumerate(CLASS_ORDER):
        values = frame.loc[frame["class"].eq(variant_class), column].to_numpy(float)
        colour = CLASS_COLOUR[variant_class]
        parts = ax.boxplot(
            [values],
            positions=[index],
            widths=BOX_WIDTH,
            showfliers=False,
            patch_artist=True,
            manage_ticks=False,
            zorder=2,
        )
        parts["boxes"][0].set(facecolor="white", edgecolor=colour, linewidth=0.6)
        for key in ("whiskers", "caps"):
            for line in parts[key]:
                line.set(color=colour, linewidth=0.6)
        parts["medians"][0].set(color=colour, linewidth=1.1)
        ax.plot(
            index + rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, values.size),
            values,
            linestyle="none",
            marker=CLASS_MARKER[variant_class],
            markersize=np.sqrt(CLASS_SIZE[variant_class]),
            markerfacecolor=colour,
            markeredgecolor="white",
            markeredgewidth=0.2,
            alpha=0.85,
            zorder=3,
        )

    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels([CLASS_TICK[name] for name in CLASS_ORDER])
    ax.set_xlim(-0.62, len(CLASS_ORDER) - 0.38)
    ax.set_ylabel(label)
    ax.tick_params(axis="x", length=0)

    # Headroom is reserved before the annotation is placed, so the six columns
    # set their statistics in the same corner.
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, bottom + (top - bottom) / (1.0 - ANNOTATION_HEADROOM))
    ax.text(
        0.04,
        0.97,
        f"AUROC {found['auroc']:.2f}\nP = {found['p_one_sided_lower']:.2g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        linespacing=1.3,
    )


def draw_roc(ax: plt.Axes, frame: pd.DataFrame, table: pd.DataFrame) -> None:
    """The three channels' curves for one sublibrary, each named with its area."""
    found = table.set_index("column")
    handles = []
    curves = {}
    for column, _, short_label, colour, dashes in CHANNELS:
        negative = frame.loc[frame["class"].eq(NEGATIVE), column].to_numpy(float)
        positive = frame.loc[frame["class"].eq(POSITIVE), column].to_numpy(float)
        false_rate, true_rate = roc_curve(negative, positive)
        curves[short_label] = (false_rate, true_rate)
        ax.plot(
            false_rate,
            true_rate,
            color=colour,
            lw=1.0,
            linestyle=dashes or "-",
            solid_capstyle="round",
            zorder=3,
        )
        handles.append(
            Line2D(
                [],
                [],
                color=colour,
                lw=1.0,
                linestyle=dashes or "-",
                label=f"{short_label}  {found.loc[column, 'auroc']:.2f}",
            )
        )

    note = coincident_sets(curves)
    if note:
        ax.text(
            0.97,
            0.04,
            "\n".join(note),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=COLORS["gray"],
            linespacing=1.3,
            zorder=4,
        )

    ax.plot([0, 1], [0, 1], color=COLORS["grid"], lw=0.6, linestyle=":", zorder=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks((0.0, 0.5, 1.0))
    ax.set_yticks((0.0, 0.5, 1.0))
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(-0.02, -0.28),
        frameon=False,
        handlelength=1.8,
        handletextpad=0.5,
        borderpad=0.0,
        borderaxespad=0.0,
        labelspacing=0.3,
    )


def draw_all(frame: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    """Both sublibraries, one row each, three strip columns and a curve panel."""
    fig = figure(WIDTH_MM, HEIGHT_MM)
    tables = []
    for row, gene in enumerate(GENES):
        variants = sublibrary(frame, gene)
        table = discrimination(variants, gene)
        tables.append(table)
        indexed = table.set_index("column")
        letters = LETTERS[gene]
        for index, (column, axis_label, _, _, _) in enumerate(CHANNELS):
            ax = place(
                fig,
                (COLUMN_X_MM[index], ROW_Y_MM[row], STRIP_WIDTH_MM, STRIP_HEIGHT_MM),
            )
            draw_strip(ax, variants, column, axis_label, indexed.loc[column])
            panel_letter(ax, letters[index])
            if index == 0:
                # Centred rather than flush left, where the panel letter sits.
                ax.set_title(gene, loc="center", pad=3.0)
        ax = place(
            fig,
            (COLUMN_X_MM[3], ROW_Y_MM[row] + 8.0, ROC_SIZE_MM, ROC_SIZE_MM),
        )
        draw_roc(ax, variants, table)
        panel_letter(ax, letters[3])
    return fig, pd.concat(tables, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="data directory")
    parser.add_argument(
        "--results", type=Path, default=DEFAULT_RESULTS, help="directory holding the fits"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="directory to write the figure into"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    use_style()
    frame = build(args.data, args.results)
    fig, table = draw_all(frame)
    print("stop-gain against synonymous, one sublibrary and one channel per row\n")
    print(table.drop(columns=["axis_label"]).to_string(index=False), end="\n\n")
    write(fig, args.output, "supp_figure_9")


if __name__ == "__main__":
    main()
