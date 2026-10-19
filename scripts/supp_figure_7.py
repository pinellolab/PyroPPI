#!/usr/bin/env python
"""Supplementary figure 7: the RAD51C and XRCC3 scans.

Panel a is figure 4 panel d and panel b is figure 4 panel e, drawn for the other
scanned pair. Neither gene carries an annotated Walker motif here, so the tracks
shade nothing and the motif tests are not run. Panel c of the manuscript figure
is a structure render and is not drawn here.

    python supp_figure_7.py
    python supp_figure_7.py --data /path/to/data --results /path/to/results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from figure_4 import (  # noqa: E402
    CX3,
    distance_rows,
    draw_distance_dumbbell,
    draw_position_tracks,
    load_scans,
    report_distance,
    report_tracks,
)
from pyroppi.plotting import figure, panel_letter, save, use_style  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data",
                        help="directory holding the dataset tables")
    parser.add_argument("--results", type=Path, default=REPO / "results",
                        help="directory holding the fits")
    parser.add_argument("--output", type=Path, default=REPO / "figures",
                        help="directory to write the figure into")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    tables = load_scans(args.data, args.results, CX3)

    use_style()
    fig = figure(183, 195)
    grid_a = fig.add_gridspec(1, 2, left=0.085, right=0.985, top=0.959, bottom=0.385,
                              wspace=0.20)
    by_gene = {
        gene: draw_position_tracks(fig, grid_a[0, column], tables[gene], gene, {})
        for column, gene in enumerate(CX3)
    }

    rows = distance_rows(tables, CX3)
    grid_b = fig.add_gridspec(1, 1, left=0.285, right=0.965, top=0.292, bottom=0.067)
    ax_b = fig.add_subplot(grid_b[0, 0])
    draw_distance_dumbbell(ax_b, rows)

    panel_letter(by_gene["RAD51C"]["score"], "a", dx=-0.16)
    panel_letter(ax_b, "b", dx=-0.42)

    fig.savefig(args.output / "supp_figure_7.png")
    save(fig, args.output / "supp_figure_7.pdf")

    report_tracks(by_gene, "panel a")
    report_distance(rows, "panel b")


if __name__ == "__main__":
    main()
