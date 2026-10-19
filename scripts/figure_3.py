#!/usr/bin/env python
"""Figure 3 data panels: the ClinORF matrix, the paralog pair matrix, hit calling.

Panel a is the interaction matrix over the twenty ClinORF proteins, panel c the
six RAD51 paralogs drawn one square per pair, and panel d those pairs ranked
against the GFP and mCherry control pairs. Panel b of the manuscript figure is a
cartoon and is not drawn here.

    python figure_3.py
    python figure_3.py --data /path/to/data --results /path/to/results
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

from pyroppi import load_fit, stats
from pyroppi.plotting import COLORS, figure, panel_letter, save, use_style

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

CONTROLS = ("GFP", "mCherry")

# The ClinORF library was built around three complexes. Panel a keeps that
# grouping, and the paralogs run in panel c's order so the two matrices of the
# same proteins can be read against each other.
PARALOG_ORDER = ["XRCC2", "RAD51D", "RAD51C", "RAD51B", "XRCC3", "RAD51"]
CLINORF_GROUPS = {
    "RAD51 paralogs": [*PARALOG_ORDER, "SFR1"],
    "Fanconi anemia": ["FANCC", "FANCE", "FANCG", "FAAP100", "FAAP24", "UBE2T",
                       "FANCD2", "CENPS", "CENPX"],
    "CCM": ["KRIT1", "CCM2", "PDCD10", "TLNRD1"],
}
CLINORF_ORDER = [name for group in CLINORF_GROUPS.values() for name in group]

# Panel c reads its buried surface area off two structures. A pair inside one
# block takes that block's structure; a pair across the two takes 9SVX where it
# resolves the pair and 9Q2A otherwise.
BCDX2_BLOCK = {"XRCC2", "RAD51D", "RAD51C", "RAD51B"}
BCDX2_STRUCTURE = "9Q2A"
CX3_BLOCK = {"RAD51C", "XRCC3", "RAD51"}
CX3_STRUCTURE = "9SVX"
# RAD51 against itself is the filament contact, the one diagonal cell panel c
# draws. SFR1 is measured but has no structure here, so panel d leaves it out.
PARALOG_SELF_PAIRS = ("RAD51",)
PARALOG_EXCLUDED = ("SFR1",)

PROPENSITY_LABEL = "Interaction propensity (% edited)"
# Panel a's scale stops at the 99th percentile while panel c's reaches its
# largest pair, so the same quantity is drawn over two different ranges on one
# page. Panel a's bar carries the clip on a second line, which keeps the two
# bars apart at 6 pt without crowding either of them.
CLIPPED_LABEL = PROPENSITY_LABEL + "\nclipped at {clip:.1f}, the 99th percentile"
BSA_LABEL = "Buried surface area (Å$^2$)"
CMAP = "Oranges"
NO_DATA = "0.92"
BLOCK_LINE = "0.45"
KNOWN_EDGE = "#111111"
KNOWN_WIDTH = 0.75

# The square's side as a share of its cell, at the smallest and largest areas.
# Area carries the number, so doubling a buried surface doubles the ink. The
# side is floored because the square also carries the colour: a pair that
# buries nothing still has a measured propensity.
SIDE_FLOOR = 0.17
SIDE_CEILING = 0.80
CELL_EDGE = "0.88"
GLYPH_EDGE = "0.45"
# Where the two keys sit in the blank upper triangle, as fractions of the
# matrix. The bar reaches further left because it sits in the top row, where
# only the first cell is drawn; the size key sits two rows down, where the
# triangle has eaten further across.
BAR_X = (0.20, 0.89)
KEY_X = (0.52, 0.89)
KEY_AREAS = (0.0, 1500.0, 4500.0)
KEY_INNER_EDGE = "0.65"
KEY_NUMBER_WIDTH = 0.45
KEY_LABEL_GAP = 0.09
KEY_TICK = 0.13
KEY_BASE = 1.98

# Where the ramp's low end is pulled to white, and how far. ColorBrewer's
# Oranges starts at #FFF5EB, so without this the whole bottom of the scale is
# one off-white tint in a matrix where most pairs sit at the null.
WHITE_SHARE = 0.16
WHITE_FADE = 0.75


def white_anchored(name: str, share: float = WHITE_SHARE,
                   fade: float = WHITE_FADE, levels: int = 256):
    """``name`` with its low end blended to white over ``share`` of the ramp."""
    base = plt.get_cmap(name)
    position = np.linspace(0.0, 1.0, levels)
    colours = np.asarray(base(position), dtype=float)
    weight = np.clip(fade * (1.0 - position / share), 0.0, 1.0)[:, None]
    colours[:, :3] = colours[:, :3] * (1.0 - weight) + weight
    return LinearSegmentedColormap.from_list(f"{name}_white", colours)


def add_axes_mm(fig: plt.Figure, x_mm: float, y_mm: float,
                width_mm: float, height_mm: float) -> plt.Axes:
    """Axes placed in millimetres, measured down and right of the top left."""
    page_w = fig.get_figwidth() * 25.4
    page_h = fig.get_figheight() * 25.4
    return fig.add_axes([x_mm / page_w, 1.0 - (y_mm + height_mm) / page_h,
                         width_mm / page_w, height_mm / page_h])


def fold(pairs: pd.DataFrame, order: Sequence[str],
         column: str = "propensity_mean") -> pd.DataFrame:
    """Symmetric matrix over ``order``, averaging a pair's two orientations."""
    order = list(order)
    total = pd.DataFrame(0.0, index=order, columns=order)
    count = pd.DataFrame(0, index=order, columns=order)
    for row in pairs.itertuples():
        a, b = row.protein_a, row.protein_b
        if a not in count.index or b not in count.index:
            continue
        value = float(getattr(row, column))
        for first, second in {(a, b), (b, a)}:
            total.loc[first, second] += value
            count.loc[first, second] += 1
    return total / count.replace(0, np.nan)


def interface_matrix(areas: pd.DataFrame, order: Sequence[str]) -> pd.DataFrame:
    """Buried surface area per pair, merged across the two paralog structures."""
    order = list(order)
    lookup = {
        (row.structure, frozenset((row.protein_a, row.protein_b))): float(row.bsa_A2)
        for row in areas.itertuples()
    }
    grid = pd.DataFrame(np.nan, index=order, columns=order, dtype=float)
    for first in order:
        for second in order:
            key = frozenset((first, second))
            ends = {first, second}
            if ends <= CX3_BLOCK:
                value = lookup.get((CX3_STRUCTURE, key), np.nan)
            elif ends <= BCDX2_BLOCK:
                value = lookup.get((BCDX2_STRUCTURE, key), np.nan)
            else:
                value = lookup.get((CX3_STRUCTURE, key),
                                   lookup.get((BCDX2_STRUCTURE, key), np.nan))
            grid.loc[first, second] = value
    return grid


def known_pairs(data: Path) -> set[frozenset[str]]:
    """The reported interactions panel a outlines: two or more publications."""
    table = pd.read_csv(data / "annotations" / "known_interactions.csv")
    marked = table.loc[table["known"]]
    return {frozenset((row.protein_a, row.protein_b)) for row in marked.itertuples()}


def draw_clinorf_matrix(ax: plt.Axes, cax: plt.Axes, grid: pd.DataFrame,
                        reported: set[frozenset[str]]) -> dict:
    """Panel a: lower-triangular interaction propensity over twenty proteins."""
    names = list(grid.index)
    n = len(names)
    values = grid.to_numpy(dtype=float)
    # One or two pairs run far above the rest, and on a scale that reaches them
    # every other pair above the controls is drawn at a fifth of the ramp.
    vmax = float(np.nanpercentile(values, 99))
    cmap = white_anchored(CMAP)
    norm = Normalize(0.0, vmax)

    shown = np.where(np.tri(n, dtype=bool), values, np.nan)
    mesh = ax.pcolormesh(np.arange(n + 1), np.arange(n + 1),
                         np.ma.masked_invalid(shown), cmap=cmap, norm=norm,
                         edgecolors="white", linewidth=0.3)
    # An unmeasured cell inside the triangle is grey. White is the low end of
    # the scale, so a blank cell would read as a measured zero.
    for row in range(n):
        for col in range(row + 1):
            if np.isnan(values[row, col]):
                ax.add_patch(Rectangle((col, row), 1, 1, facecolor=NO_DATA,
                                       edgecolor="white", linewidth=0.3, zorder=2))

    index_of = {name: i for i, name in enumerate(names)}
    marked = []
    for pair in sorted(reported, key=sorted):
        if not pair <= index_of.keys():
            continue
        first, second = sorted(pair, key=index_of.__getitem__)
        row, col = index_of[second], index_of[first]
        if np.isnan(values[row, col]):
            continue
        marked.append((first, second, float(values[row, col])))
        ax.add_patch(Rectangle((col, row), 1, 1, facecolor="none",
                               edgecolor=KNOWN_EDGE, linewidth=KNOWN_WIDTH, zorder=5))

    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(n) + 0.5, names, rotation=90)
    ax.set_yticks(np.arange(n) + 0.5, names)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(length=0, pad=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Each complex is bounded inside the triangle and named above its columns,
    # so the matrix reads as three complexes rather than twenty proteins.
    start = 0
    for group, members in CLINORF_GROUPS.items():
        width = sum(1 for name in members if name in index_of)
        if not width:
            continue
        end = start + width
        if end < n:
            ax.plot([end, end], [end, n], color=BLOCK_LINE, lw=0.6,
                    solid_capstyle="butt", zorder=4)
            ax.plot([0, end], [end, end], color=BLOCK_LINE, lw=0.6,
                    solid_capstyle="butt", zorder=4)
        ax.plot([start + 0.1, end - 0.1], [-3.1, -3.1], color=COLORS["ink"],
                lw=0.6, clip_on=False, solid_capstyle="butt")
        ax.text((start + end) / 2.0, -3.4, group, ha="center", va="bottom",
                color=COLORS["ink"], clip_on=False)
        start = end

    bar = ax.figure.colorbar(mesh, cax=cax, orientation="horizontal")
    cax.xaxis.set_label_position("top")
    bar.set_label(CLIPPED_LABEL.format(clip=vmax * 100), labelpad=2.0)
    bar.set_ticks([tick for tick in (0.0, 0.02, 0.04, 0.06) if tick <= vmax])
    bar.ax.set_xticklabels([f"{tick * 100:g}" for tick in bar.get_ticks()])
    bar.ax.tick_params(length=1.8, width=0.5, pad=1.5)
    bar.outline.set_linewidth(0.5)

    swatch = 0.9
    key_x, key_y = n * 0.52, n * 0.21
    ax.add_patch(Rectangle((key_x, key_y), swatch, swatch, facecolor="none",
                           edgecolor=KNOWN_EDGE, linewidth=KNOWN_WIDTH, zorder=3))
    ax.text(key_x + swatch + 0.5, key_y + swatch / 2.0, "Reported interaction",
            ha="left", va="center", color=COLORS["ink"])

    return {"n_proteins": n, "vmax": vmax, "marked": marked,
            "n_measured": int(np.isfinite(shown).sum())}


def _side(area: float, area_max: float) -> float:
    """The square's side in cell units, with its area proportional to ``area``."""
    if not np.isfinite(area) or area <= 0.0 or area_max <= 0.0:
        return SIDE_FLOOR
    return float(max(SIDE_FLOOR, SIDE_CEILING * np.sqrt(area / area_max)))


def size_matrix_colorbar(fig: plt.Figure, x_mm: float, y_mm: float,
                         side_mm: float, n: int) -> plt.Axes:
    """Axes for the size matrix's colourbar, in the blank upper triangle."""
    cell = side_mm / n
    left = x_mm + (n * BAR_X[0] + 0.5) * cell
    right = x_mm + (n * BAR_X[1] + 0.5) * cell
    top = y_mm + (-0.05 + 0.5) * cell
    bottom = y_mm + (0.15 + 0.5) * cell
    return add_axes_mm(fig, left, top, right - left, bottom - top)


def draw_size_matrix(ax: plt.Axes, cax: plt.Axes, bsa: pd.DataFrame,
                     propensity: pd.DataFrame, self_pairs: Sequence[str] = (),
                     key_areas: Sequence[float] = KEY_AREAS,
                     label_rotation: float = 0.0) -> dict:
    """Panel c: one square per pair, area the structure and colour the data.

    Agreement is a big dark square, and the two ways of disagreeing look
    different rather than merely shaded differently. ``self_pairs`` names the
    proteins whose diagonal cell carries a measurement; the rest read as no data.
    """
    order = list(propensity.index)
    n = len(order)
    areas = np.where(np.isfinite(bsa.to_numpy(dtype=float)), bsa.to_numpy(dtype=float), 0.0)
    values = propensity.to_numpy(dtype=float)

    measured = np.tril(np.ones((n, n), dtype=bool))
    for i, name in enumerate(order):
        measured[i, i] = name in set(self_pairs)
    area_max = float(np.nanmax(np.where(measured, areas, np.nan)))
    value_max = float(np.nanmax(np.where(measured, values, np.nan)))
    cmap = white_anchored(CMAP)
    norm = Normalize(0.0, value_max)

    cells = []
    for row in range(n):
        for col in range(row + 1):
            ax.add_patch(Rectangle(
                (col - 0.5, row - 0.5), 1.0, 1.0,
                facecolor="none" if measured[row, col] else NO_DATA,
                edgecolor=CELL_EDGE, lw=0.4, zorder=1.0))
            if not measured[row, col] or not np.isfinite(values[row, col]):
                continue
            value = float(values[row, col])
            side = _side(areas[row, col], area_max)
            shade = cmap(norm(value))
            ax.add_patch(Rectangle((col - side / 2.0, row - side / 2.0), side, side,
                                   facecolor=shade, edgecolor=GLYPH_EDGE, lw=0.5,
                                   zorder=2.0))
            # Inside the square where it fits, under it where it does not: a
            # number set over a floor-sized square covers the square.
            inside = side > 0.55
            luminance = 0.299 * shade[0] + 0.587 * shade[1] + 0.114 * shade[2]
            ax.text(col, row if inside else row + side / 2.0 + 0.14,
                    f"{value * 100:.1f}", ha="center",
                    va="center" if inside else "top",
                    color="white" if inside and luminance < 0.55 else "0.25",
                    zorder=3.0)
            cells.append({"protein_row": order[row], "protein_col": order[col],
                          "bsa_A2": float(areas[row, col]), "propensity": value})

    ax.set_xticks(np.arange(n), order, rotation=label_rotation)
    ax.set_yticks(np.arange(n), order)
    ax.tick_params(axis="both", length=0, pad=2.5)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)

    bar = ax.figure.colorbar(plt.cm.ScalarMappable(norm, cmap), cax=cax,
                             orientation="horizontal")
    cax.xaxis.set_label_position("top")
    bar.set_label(PROPENSITY_LABEL, labelpad=2.0, loc="left")
    bar.set_ticks([tick for tick in (0.0, 0.05, 0.10, 0.15) if tick <= value_max])
    bar.ax.set_xticklabels([f"{tick * 100:g}" for tick in bar.get_ticks()])
    bar.ax.tick_params(length=1.8, width=0.5, pad=1.5)
    bar.outline.set_linewidth(0.5)
    _draw_size_key(ax, n, area_max, key_areas)

    return {"n_proteins": n, "bsa_vmax": area_max, "propensity_vmax": value_max,
            "floor_area": (SIDE_FLOOR / SIDE_CEILING) ** 2 * area_max,
            "cells": pd.DataFrame(cells)}


def _draw_size_key(ax: plt.Axes, n: int, area_max: float,
                   key_areas: Sequence[float]) -> None:
    """The size key: the squares themselves, at the matrix's own scale.

    A size key is read by matching a square in the matrix against a square in
    the key, so a key drawn at any other scale is not one. They nest on a shared
    bottom left corner, which puts the comparison the key exists for inside it.
    """
    sides = [_side(area, area_max) for area in key_areas]
    left = n * KEY_X[0]
    outer = sides[-1]
    numbers = left + KEY_NUMBER_WIDTH
    edge = numbers + KEY_LABEL_GAP + KEY_TICK
    ax.text(left, KEY_BASE - outer - 0.26, BSA_LABEL, ha="left", va="bottom",
            color="0.15")
    for area, side in zip(key_areas, sides):
        top = KEY_BASE - side
        widest = side == outer
        ax.add_patch(Rectangle((edge, top), side, side, facecolor="none",
                               edgecolor=GLYPH_EDGE if widest else KEY_INNER_EDGE,
                               lw=0.5 if widest else 0.4, zorder=2.0))
        ax.plot([numbers + KEY_LABEL_GAP, edge], [top, top], color=KEY_INNER_EDGE,
                lw=0.4, solid_capstyle="butt", zorder=2.0)
        ax.text(numbers, top, f"{area:.0f}", ha="right", va="center", color="0.25",
                zorder=3.0)


def draw_ranked_pairs(ax: plt.Axes, ranked: pd.DataFrame, threshold: float) -> None:
    """Panel d: the paralog pairs ranked by propensity against the control level."""
    y = np.arange(len(ranked))
    hit = (ranked["p"] < 0.05).to_numpy()
    colour = np.where(hit, COLORS["hit"], COLORS["gray"])

    ax.barh(y, ranked["propensity_mean"] * 100, height=0.62, color=colour,
            edgecolor="none", zorder=3)
    ax.axvline(threshold * 100, color=COLORS["threshold"], lw=0.8,
               ls=(0, (2.5, 2.0)), zorder=4)
    ax.text(threshold * 100 + 0.15, -0.9, "control 95%", color=COLORS["threshold"],
            ha="left", va="bottom", clip_on=False)

    ax.set_yticks(y, [f"{row.protein_a}-{row.protein_b}" for row in ranked.itertuples()])
    ax.set_ylim(len(ranked) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlim(0, max(16.0, float(ranked["propensity_mean"].max()) * 100 * 1.08))
    ax.set_xlabel(PROPENSITY_LABEL)
    ax.spines["left"].set_visible(False)
    for label, called in zip(ax.get_yticklabels(), hit):
        if called:
            label.set_color(COLORS["hit"])


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

    clinorf = load_fit(args.results / "clinorf").pairs()
    paralogs = load_fit(args.results / "rad51x9").pairs()
    areas = pd.read_csv(args.data / "annotations" / "interface_area.csv")

    clinorf_grid = fold(clinorf, CLINORF_ORDER)
    propensity = fold(paralogs, PARALOG_ORDER)
    bsa = interface_matrix(areas, PARALOG_ORDER)

    threshold = stats.control_threshold(paralogs, controls=CONTROLS)
    tested = stats.control_test(paralogs, controls=CONTROLS)
    excluded = tested.protein_a.isin(PARALOG_EXCLUDED) | tested.protein_b.isin(PARALOG_EXCLUDED)
    ranked = tested.loc[~excluded].sort_values("propensity_mean", ascending=False)

    use_style()
    fig = figure(183, 180)
    ax_a = add_axes_mm(fig, 17, 24, 70, 70)
    cax_a = add_axes_mm(fig, 53.4, 31.0, 28.0, 1.6)
    ax_c = add_axes_mm(fig, 112, 24, 64, 64)
    cax_c = size_matrix_colorbar(fig, 112, 24, 64, len(PARALOG_ORDER))
    ax_d = add_axes_mm(fig, 38, 108, 100, 62)

    panel_a = draw_clinorf_matrix(ax_a, cax_a, clinorf_grid, known_pairs(args.data))
    panel_c = draw_size_matrix(ax_c, cax_c, bsa, propensity,
                               self_pairs=PARALOG_SELF_PAIRS)
    draw_ranked_pairs(ax_d, ranked, threshold)

    panel_letter(ax_a, "a", dx=-0.20, dy=1.243)
    panel_letter(ax_c, "c", dx=-0.219, dy=1.04)
    panel_letter(ax_d, "d", dx=-0.35, dy=1.04)

    fig.savefig(args.output / "figure_3.png")
    save(fig, args.output / "figure_3.pdf")

    print(f"panel a: {panel_a['n_proteins']} proteins, "
          f"{panel_a['n_measured']} measured pairs in the triangle, "
          f"colour scale topping out at {panel_a['vmax'] * 100:.2f}%, "
          f"the 99th percentile")
    print(f"  {len(panel_a['marked'])} reported interactions outlined")
    for first, second, value in sorted(panel_a["marked"], key=lambda row: -row[2]):
        print(f"    {first:8s} {second:8s} {value * 100:6.2f}%")
    print()

    print(f"panel c: {panel_c['n_proteins']} paralogs, "
          f"largest buried surface area {panel_c['bsa_vmax']:.0f} A^2, "
          f"square floor at {panel_c['floor_area']:.0f} A^2, "
          f"largest propensity {panel_c['propensity_vmax'] * 100:.2f}%")
    cells = panel_c["cells"].sort_values("propensity", ascending=False)
    print(cells.to_string(index=False, float_format=lambda v: f"{v:.4f}"), end="\n\n")

    print(f"panel d: control 95% level {threshold:.4f} "
          f"({threshold * 100:.2f}% edited), "
          f"{int((ranked['p'] < 0.05).sum())} of {len(ranked)} pairs called")
    for row in ranked.itertuples():
        name = f"{row.protein_a}-{row.protein_b}"
        print(f"  {name:<15s} propensity {row.propensity_mean * 100:5.2f}%  "
              f"z {row.z:6.2f}  P {row.p:9.3g}"
              + ("  called" if row.p < 0.05 else ""))


if __name__ == "__main__":
    main()
