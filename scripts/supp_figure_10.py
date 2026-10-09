#!/usr/bin/env python
"""Supplementary Figure 10: Figure 5's data panels over the RAD51D sublibrary.

The same pool carries a RAD51D sublibrary and an XRCC2 sublibrary, measured
against the wild-type partner in each case. Figure 5 draws the XRCC2 half; this
figure draws the RAD51D half with the same five panels, the same axes and the
same statistics, so the two halves can be read against each other. Each colour
bar is set from the variants it covers, so the two halves print their own
limits. Panels b to f of Figure 5 are panels a to e here.

    python supp_figure_10.py
    python supp_figure_10.py --output /tmp/figures
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

from figure_5 import (
    DEFAULT_DATA,
    DEFAULT_OUTPUT,
    DEFAULT_RESULTS,
    build,
    draw_all,
    report,
    write,
)
from pyroppi.plotting import use_style

GENE = "RAD51D"
LETTERS = ("a", "b", "c", "d", "e")


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
    fig, results = draw_all(frame, GENE, LETTERS)
    report(GENE, results, LETTERS)
    write(fig, args.output, "supp_figure_10")


if __name__ == "__main__":
    main()
