#!/usr/bin/env python
"""Supplementary Figure 5: the Commander complex interaction screen.

Every pair of the ten COMMD subunits, drawn by Figure 3c's own drawer: one
square per pair, its area the buried surface area PDB entry 8P0W gives the pair
and its colour and printed number the interaction propensity PPI Recorder measures
for it. This is the screen the two pairs taken forward to deep mutational
scanning were chosen from.

The fit is the ring and the GFP and mCherry controls the ring is called against.
All 45 pairs the ten subunits can make were measured, so the panel covers the
ring completely. The diagonal is blank: the fit also estimates a homodimer pair
per subunit, which is not interpretable as an interaction here.

The panel prints a propensity per cell and carries no threshold, so which pairs
clear the controls is a statement in the legend. Those numbers are computed here
and printed: each pair's propensity, its distance above the control pairs in
standard deviations, and its one-sided P.

    python supp_figure_5.py
    python supp_figure_5.py --data /path/to/data --results /path/to/results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from figure_3 import add_axes_mm, draw_size_matrix, fold, size_matrix_colorbar

from pyroppi import load_fit, stats
from pyroppi.plotting import figure, save, use_style

CONTROLS = ("GFP", "mCherry")
STRUCTURE = "8P0W"
RING_ORDER = [f"COMMD{i}" for i in range(1, 11)]
# Figure 3c's low anchors, with the top one near this structure's largest
# interface. A size key has to be at the scale of the matrix it keys.
KEY_AREAS = (0.0, 1500.0, 5000.0)
ALPHA = 0.05

# The matrix, in millimetres on a 120 mm page. Ten names do not fit side by side
# under a matrix this wide, so the column names are set vertically.
MATRIX_X = 16.0
MATRIX_Y = 5.0
MATRIX_SIDE = 100.0
LABEL_ROTATION = 90.0


def interface_matrix(areas: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Buried surface area per pair, read off the structure's own rows."""
    rows = areas.loc[areas["structure"] == STRUCTURE]
    grid = pd.DataFrame(np.nan, index=order, columns=order, dtype=float)
    for row in rows.itertuples():
        if row.protein_a in grid.index and row.protein_b in grid.index:
            grid.loc[row.protein_a, row.protein_b] = float(row.bsa_A2)
            grid.loc[row.protein_b, row.protein_a] = float(row.bsa_A2)
    return grid


def ring_test(pairs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Each ring pair against the control pairs, and the null it is tested on.

    The null is a Gaussian on the control pairs' logit baselines, which is what
    ``control_test`` fits: a pair's z is its distance above the control mean in
    control standard deviations, and its P is the upper tail of that normal.
    """
    control = pairs.protein_a.isin(CONTROLS) | pairs.protein_b.isin(CONTROLS)
    null = pairs.loc[control, "logit_mean"].to_numpy(dtype=float)
    tested = stats.control_test(pairs, controls=CONTROLS)
    ring = tested.loc[tested.protein_a != tested.protein_b]
    ring = ring.sort_values("propensity_mean", ascending=False).reset_index(drop=True)
    return ring, {
        "n_control_pairs": int(null.size),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "threshold": stats.control_threshold(pairs, controls=CONTROLS),
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

    pairs = load_fit(args.results / "commander").pairs()
    areas = pd.read_csv(args.data / "annotations" / "interface_area.csv")
    propensity = fold(pairs, RING_ORDER)
    bsa = interface_matrix(areas, RING_ORDER)
    ring, null = ring_test(pairs)

    use_style()
    fig = figure(120, 122)
    ax = add_axes_mm(fig, MATRIX_X, MATRIX_Y, MATRIX_SIDE, MATRIX_SIDE)
    cax = size_matrix_colorbar(fig, MATRIX_X, MATRIX_Y, MATRIX_SIDE, len(RING_ORDER))
    # No self-pair is drawn, so every diagonal cell reads as no data.
    panel = draw_size_matrix(ax, cax, bsa, propensity, self_pairs=(),
                             key_areas=KEY_AREAS, label_rotation=LABEL_ROTATION)
    ax.tick_params(axis="x", pad=4.0)

    fig.savefig(args.output / "supp_figure_5.png")
    save(fig, args.output / "supp_figure_5.pdf")

    print(f"panel: {panel['n_proteins']} subunits, {len(panel['cells'])} pairs, "
          f"largest buried surface area {panel['bsa_vmax']:.0f} A^2, "
          f"square floor at {panel['floor_area']:.0f} A^2, "
          f"largest propensity {panel['propensity_vmax'] * 100:.2f}%")
    print(f"control null on {null['n_control_pairs']} pairs: "
          f"mean logit {null['null_mean']:.4f}, sd {null['null_sd']:.4f}, "
          f"95% level {null['threshold']:.4f} "
          f"({null['threshold'] * 100:.2f}% edited)")

    called = ring.loc[ring["p"] < ALPHA]
    print(f"{len(called)} of {len(ring)} ring pairs called at P < {ALPHA}")
    for row in called.itertuples():
        name = f"{row.protein_a}-{row.protein_b}"
        print(f"  rank {row.Index + 1}  {name:<18s} "
              f"propensity {row.propensity_mean * 100:5.2f}%  z {row.z:5.2f}  "
              f"P {row.p:.3g}")
    runner_up = ring.loc[ring["p"] >= ALPHA].head(1)
    for row in runner_up.itertuples():
        name = f"{row.protein_a}-{row.protein_b}"
        print(f"  next     {name:<18s} "
              f"propensity {row.propensity_mean * 100:5.2f}%  z {row.z:5.2f}  "
              f"P {row.p:.3g}")


if __name__ == "__main__":
    main()
