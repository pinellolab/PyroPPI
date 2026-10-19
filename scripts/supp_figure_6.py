"""Supplementary Figure 6. Where the barcode dictionary cutoffs were set.

The four RAD51-paralog deep mutational scanning libraries were cloned in nine
Dual NEBuilder blocks, and a barcode-to-variant dictionary was built for each.
Every block took a support-read cutoff and a purity cutoff of its own, read off
the diagnostic this figure redraws: the reads a barcode gives its best-supported
variant against the share of that barcode's reads the same variant accounts for.

A Dual NEBuilder block splits a barcode's reads across many variants by
construction, so a barcode that is perfectly assignable still carries most of
its reads on other variants, and the pipeline's standard threshold of at least
3 support reads and purity at least 0.9 is far above what these blocks can meet.
Both retentions are printed, the block's own and the standard one.

The source table is a weighted collapse of the delivered per-barcode file, one
row per distinct combination of block, support reads and total reads, so every
count here is weighted by ``n_barcodes``.

    python scripts/supp_figure_6.py
    python scripts/supp_figure_6.py --output /path/to/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from pyroppi.plotting import COLORS, figure, save, use_style

REPOSITORY = Path(__file__).resolve().parent.parent

CUTOFFS = "annotations/dictionary_cutoffs.csv"
PARAMETERS = "annotations/dictionary_cutoff_parameters.csv"

# Reading order: the four scanned genes as the variant figures take them,
# blocks ascending within a gene. Nine blocks over three rows of three.
BLOCK_ORDER = [
    "RAD51C_DMS_1",
    "RAD51C_DMS_2",
    "RAD51D_DMS_7",
    "RAD51D_DMS_8",
    "XRCC2_DMS_6",
    "XRCC2_DMS_7",
    "XRCC3_DMS_1",
    "XRCC3_DMS_2",
    "XRCC3_DMS_3",
]
GRID_ROWS = 3
GRID_COLUMNS = 3

STANDARD_SUPPORT = 3
STANDARD_PURITY = 0.9

# Rounded out from the pooled table, where support reads reach 2,700 and purity
# bottoms out at 0.0135. Shared by all nine, so a block's cloud sits where its
# sequencing depth puts it rather than filling its own cell.
SUPPORT_LIMITS = (0.8, 3000.0)
PURITY_LIMITS = (0.009, 1.25)
# Purity is a share, so the accepted region stops at 1 even though the axis
# carries headroom above it.
PURITY_CEILING = 1.0
SUPPORT_TICKS = [1, 10, 100, 1000]
PURITY_TICKS = [0.01, 0.1, 1.0]
PURITY_TICK_LABELS = ["0.01", "0.1", "1"]

HEX_GRIDSIZE = 22
DENSITY_CMAP = "GnBu"
CUTOFF_COLOR = COLORS["threshold"]
ACCEPTED_ALPHA = 0.11


def load(data: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The weighted diagnostic and the cutoff each block took."""
    frame = pd.read_csv(data / CUTOFFS)
    frame["purity"] = frame["support_reads"] / frame["total_reads"]
    parameters = pd.read_csv(data / PARAMETERS).set_index("block")
    missing = set(BLOCK_ORDER) - set(frame["block"])
    if missing:
        raise ValueError(f"the diagnostic carries no rows for {sorted(missing)}")
    missing = set(BLOCK_ORDER) - set(parameters.index)
    if missing:
        raise ValueError(f"no cutoff is recorded for {sorted(missing)}")
    return frame, parameters


def retention(frame: pd.DataFrame, parameters: pd.DataFrame) -> pd.DataFrame:
    """Barcodes each block keeps, under its own cutoff and under the standard one.

    Every share is a weighted one, because a row of the diagnostic stands for
    ``n_barcodes`` barcodes.
    """
    rows = []
    for block in BLOCK_ORDER:
        values = frame[frame["block"] == block]
        cutoff = parameters.loc[block]
        weight = values["n_barcodes"].to_numpy(dtype=float)
        total = weight.sum()
        own = (
            values["support_reads"].ge(cutoff["support_cutoff"])
            & values["purity"].ge(cutoff["purity_cutoff"])
        ).to_numpy()
        standard = (
            values["support_reads"].ge(STANDARD_SUPPORT)
            & values["purity"].ge(STANDARD_PURITY)
        ).to_numpy()
        rows.append(
            {
                "block": block,
                "gene": str(cutoff["gene"]),
                "block_number": int(cutoff["block_number"]),
                "support_cutoff": int(cutoff["support_cutoff"]),
                "purity_cutoff": float(cutoff["purity_cutoff"]),
                "n_barcodes": int(total),
                "n_retained": int(weight[own].sum()),
                "pct_retained": 100.0 * weight[own].sum() / total,
                "n_retained_standard": int(weight[standard].sum()),
                "pct_retained_standard": 100.0 * weight[standard].sum() / total,
            }
        )
    return pd.DataFrame(rows)


def draw_block(ax, values: pd.DataFrame, row: pd.Series):
    """One block's density, its two cutoffs and the region they accept."""
    collection = ax.hexbin(
        values["support_reads"].to_numpy(dtype=float),
        values["purity"].to_numpy(dtype=float),
        C=values["n_barcodes"].to_numpy(dtype=float),
        reduce_C_function=np.sum,
        xscale="log",
        yscale="log",
        gridsize=HEX_GRIDSIZE,
        mincnt=1,
        cmap=DENSITY_CMAP,
        linewidths=0.0,
        zorder=2,
        extent=(
            np.log10(SUPPORT_LIMITS[0]),
            np.log10(SUPPORT_LIMITS[1]),
            np.log10(PURITY_LIMITS[0]),
            np.log10(PURITY_LIMITS[1]),
        ),
    )

    # The accepted region is drawn over the density rather than under it. A tint
    # under a filled hexagon is invisible exactly where the barcodes are.
    ax.fill_between(
        [row["support_cutoff"], SUPPORT_LIMITS[1]],
        row["purity_cutoff"],
        PURITY_CEILING,
        facecolor=CUTOFF_COLOR,
        alpha=ACCEPTED_ALPHA,
        edgecolor="none",
        zorder=4,
    )
    ax.plot(
        [row["support_cutoff"], row["support_cutoff"]],
        [row["purity_cutoff"], PURITY_CEILING],
        color=CUTOFF_COLOR,
        linewidth=0.7,
        linestyle=(0, (2.6, 1.6)),
        zorder=5,
    )
    ax.plot(
        [row["support_cutoff"], SUPPORT_LIMITS[1]],
        [row["purity_cutoff"], row["purity_cutoff"]],
        color=CUTOFF_COLOR,
        linewidth=0.7,
        linestyle=(0, (2.6, 1.6)),
        zorder=5,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*SUPPORT_LIMITS)
    ax.set_ylim(*PURITY_LIMITS)
    ax.set_xticks(SUPPORT_TICKS)
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([f"{t:,}" for t in SUPPORT_TICKS]))
    ax.set_yticks(PURITY_TICKS)
    ax.yaxis.set_major_formatter(mticker.FixedFormatter(PURITY_TICK_LABELS))
    ax.tick_params(which="minor", length=0)
    ax.set_title(f"{row['gene']} block {row['block_number']}", loc="left", pad=3)
    ax.set_axisbelow(True)
    ax.grid(color=COLORS["grid"], linewidth=0.45)
    ax.text(
        0.04,
        0.04,
        f"{row['pct_retained']:.1f}% of {row['n_barcodes']:,}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        zorder=6,
    )
    return collection


def build(frame: pd.DataFrame, table: pd.DataFrame):
    """The nine cells on one colour scale, with the scale in its own column."""
    use_style()
    fig = figure(183, 150)
    grid = fig.add_gridspec(
        GRID_ROWS,
        GRID_COLUMNS + 1,
        width_ratios=[1, 1, 1, 0.055],
        left=0.07,
        right=0.95,
        bottom=0.07,
        top=0.95,
        wspace=0.30,
        hspace=0.38,
    )

    collections = []
    rows = table.set_index("block")
    for index, block in enumerate(BLOCK_ORDER):
        ax = fig.add_subplot(grid[index // GRID_COLUMNS, index % GRID_COLUMNS])
        collections.append(draw_block(ax, frame[frame["block"] == block], rows.loc[block]))
        # Named down the left column and under the bottom row only. At this
        # width the two names set nine times over would be most of the ink.
        if index % GRID_COLUMNS == 0:
            ax.set_ylabel("Purity")
        if index >= len(BLOCK_ORDER) - GRID_COLUMNS:
            ax.set_xlabel("Support reads")

    # One scale across the nine, so a hexagon means the same number of barcodes
    # in every cell and the blocks can be read against each other.
    ceiling = max(float(c.get_array().max()) for c in collections)
    norm = LogNorm(vmin=1.0, vmax=ceiling)
    for collection in collections:
        collection.set_norm(norm)

    bar = fig.colorbar(collections[0], cax=fig.add_subplot(grid[1, GRID_COLUMNS]))
    bar.set_label("Barcodes per bin")
    bar.outline.set_linewidth(0.4)
    bar.ax.tick_params(length=2.0, width=0.4, pad=2)
    bar.ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    return fig, ceiling


def write(fig, directory: Path, stem: str) -> None:
    """Write the figure as PDF and PNG, then close it."""
    fig.savefig(directory / f"{stem}.png")
    save(fig, directory / f"{stem}.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=REPOSITORY / "data",
        help="directory holding annotations/ (default: the repository's data/)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=REPOSITORY / "results",
        help="directory to write this figure's retention table into",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "figures",
        help="directory to write the PDF and PNG into",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    products = args.results / "supp_figure_6"
    products.mkdir(parents=True, exist_ok=True)

    frame, parameters = load(args.data)
    table = retention(frame, parameters)
    table.to_csv(products / "retention.csv", index=False)

    print(
        f"{len(frame):,} rows standing for {int(frame['n_barcodes'].sum()):,} barcodes "
        f"over {frame['block'].nunique()} blocks"
    )
    print(
        f"support reads reach {int(frame['support_reads'].max()):,}, "
        f"purity bottoms out at {frame['purity'].min():.4f}"
    )
    print("\nretention under each block's own cutoff")
    columns = [
        "block",
        "support_cutoff",
        "purity_cutoff",
        "n_barcodes",
        "n_retained",
        "pct_retained",
        "n_retained_standard",
        "pct_retained_standard",
    ]
    print(table[columns].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(
        f"\nretention spans {table['pct_retained'].min():.1f}% and "
        f"{table['pct_retained'].max():.1f}% across the nine blocks"
    )
    print(
        f"under at least {STANDARD_SUPPORT} support reads and purity at least "
        f"{STANDARD_PURITY}, {int(table['n_retained_standard'].sum())} barcodes survive "
        f"out of {int(table['n_barcodes'].sum()):,}, and no block keeps more than "
        f"{int(table['n_retained_standard'].max())}"
    )

    fig, ceiling = build(frame, table)
    print(f"\nthe fullest bin holds {int(ceiling):,} barcodes")
    write(fig, args.output, "supp_figure_6")
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
