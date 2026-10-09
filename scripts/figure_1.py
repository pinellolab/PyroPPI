#!/usr/bin/env python
"""Figure 1 data panels: the MS2 reporter and the two degrader titrations.

Panel b is the editing of each reporter adenosine with and without rapamycin,
panel c the ratio of those two arms, and panels e and f the pomalidomide and
ARV-471 dose responses. Panels a and d of the manuscript figure are cartoons and
are not drawn here.

    python figure_1.py
    python figure_1.py --data /path/to/data --output /path/to/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from pyroppi.plotting import COLORS, figure, panel_letter, save, use_style

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# The MS2 hairpin covers amplicon positions 28 to 44 inclusive, recorded in
# data/README.md. Panels b and c both mark that span.
MS2_HAIRPIN = (28, 44)

HEAT = LinearSegmentedColormap.from_list(
    "reporter", ["#F7F7F7", "#9ECAE1", COLORS["rad51d"], "#08306B"]
)

ARM_LABELS = {("none", 1): "no Rap 1", ("none", 2): "no Rap 2",
              ("rapamycin", 1): "Rap 1", ("rapamycin", 2): "Rap 2"}
ARM_ORDER = [("none", 1), ("none", 2), ("rapamycin", 1), ("rapamycin", 2)]

TITRATIONS = [
    ("pomalidomide", "ZNF276", "[Pomalidomide] (µM)", COLORS["pom_300"]),
    ("ARV-471", "ESR1", "[ARV-471] (µM)", COLORS["pom_3"]),
]

FOLD_TICKS = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]


def hairpin_columns(positions: list[int]) -> tuple[int, int]:
    """First and last column index the hairpin covers, in plotting order."""
    low, high = MS2_HAIRPIN
    inside = [i for i, p in enumerate(positions) if low <= p <= high]
    if not inside:
        raise ValueError(f"no reporter position falls in {MS2_HAIRPIN}")
    return min(inside), max(inside)


def mark_hairpin(ax: plt.Axes, positions: list[int]) -> None:
    """Bracket and name the hairpin columns above the axes."""
    first, last = hairpin_columns(positions)
    x0, x1 = first - 0.5, last + 0.5
    transform = ax.get_xaxis_transform()
    for xs, ys in (([x0, x1], [1.03, 1.03]), ([x0, x0], [1.0, 1.03]), ([x1, x1], [1.0, 1.03])):
        ax.plot(xs, ys, color=COLORS["ink"], lw=0.6, transform=transform, clip_on=False)
    ax.text((x0 + x1) / 2, 1.06, "MS2 hairpin", transform=transform,
            ha="center", va="bottom", color=COLORS["ink"], clip_on=False)


def read_positions(data: Path) -> tuple[pd.DataFrame, list[int]]:
    """The per-adenosine editing table and the positions in amplicon order."""
    frame = pd.read_csv(data / "reporter" / "rapamycin_positions.csv")
    order = frame[["edit_index", "reporter_position"]].drop_duplicates()
    order = order.sort_values("edit_index")
    return frame, order["reporter_position"].astype(int).tolist()


def draw_heatmap(ax: plt.Axes, cax: plt.Axes, frame: pd.DataFrame,
                 positions: list[int]) -> pd.DataFrame:
    """Panel b: editing of every reporter adenosine in each of the four samples."""
    grid = frame.pivot_table(index=["treatment", "replicate"], columns="edit_index",
                             values="editing_percent")
    grid = grid.reindex(ARM_ORDER)
    values = grid.to_numpy(dtype=float)

    mesh = ax.pcolormesh(np.arange(len(positions) + 1) - 0.5,
                         np.arange(len(ARM_ORDER) + 1) - 0.5,
                         values, cmap=HEAT, vmin=0.0, vmax=100.0, shading="flat")
    ax.invert_yaxis()
    ax.set_xlim(-0.5, len(positions) - 0.5)
    ax.set_xticks(np.arange(len(positions)), [str(position) for position in positions])
    ax.set_yticks(np.arange(len(ARM_ORDER)), [ARM_LABELS[k] for k in ARM_ORDER])
    ax.tick_params(axis="both", length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    mark_hairpin(ax, positions)

    bar = ax.figure.colorbar(mesh, cax=cax, ticks=[0, 25, 50, 75, 100])
    bar.set_label("Editing (%)", labelpad=2)
    bar.ax.tick_params(length=2.0, width=0.55)
    bar.outline.set_linewidth(0.55)

    table = pd.DataFrame(values, columns=positions)
    table.insert(0, "sample", [ARM_LABELS[k] for k in ARM_ORDER])
    return table


def draw_fold_change(ax: plt.Axes, data: Path, positions: list[int]) -> pd.DataFrame:
    """Panel c: editing with rapamycin over editing without it, per position."""
    frame = pd.read_csv(data / "reporter" / "rapamycin_fold_change.csv")
    index = {position: i for i, position in enumerate(positions)}
    first, last = hairpin_columns(positions)
    ax.axvspan(first - 0.5, last + 0.5, color=COLORS["grid"], alpha=0.45, lw=0, zorder=0)

    rows = []
    for position, group in frame.groupby("reporter_position"):
        x = index[int(position)]
        values = group["fold_change"].to_numpy(dtype=float)
        ax.plot([x, x], [values.min(), values.max()], color=COLORS["gray"], lw=0.9, zorder=2)
        ax.plot([x - 0.22, x + 0.22], [values.mean()] * 2, color=COLORS["ink"], lw=0.9, zorder=4)
        ax.scatter(np.full(values.size, x), values, s=5, color=COLORS["rad51d"],
                   lw=0, zorder=5)
        low, high = MS2_HAIRPIN
        rows.append({"reporter_position": int(position), "edit_index": x + 1,
                     "in_ms2_hairpin": low <= int(position) <= high,
                     "fold_change_mean": values.mean(),
                     "fold_change_min": values.min(), "fold_change_max": values.max()})

    ax.axhline(1, color=COLORS["threshold"], lw=0.6, ls=(0, (2.5, 2.5)), zorder=1)
    ax.set_yscale("log", base=2)
    ax.set_ylim(0.25, 64)
    ax.set_yticks(FOLD_TICKS, [f"{t:g}" for t in FOLD_TICKS])
    ax.set_xlim(-0.5, len(positions) - 0.5)
    ax.set_xticks(np.arange(len(positions)), [str(p) for p in positions])
    ax.set_xlabel("Reporter position")
    ax.set_ylabel("Editing fold change\n(Rap / no Rap)")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.45, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", pad=2)
    return pd.DataFrame(rows).sort_values("edit_index").reset_index(drop=True)


def draw_titration(ax: plt.Axes, data: Path, compound: str, partner: str,
                   x_label: str, color: str) -> pd.DataFrame:
    """Panels e and f: mean editing with a standard deviation band, log dose axis."""
    frame = pd.read_csv(data / "reporter" / "degrader_titration.csv")
    frame = frame.loc[frame["compound"] == compound]
    wide = frame.pivot(index="concentration_uM", columns="replicate",
                       values="editing_fraction")
    summary = pd.DataFrame({"mean": wide.mean(axis=1) * 100,
                            "sd": wide.std(axis=1, ddof=1) * 100,
                            "n": wide.notna().sum(axis=1)})
    summary.index.name = "concentration_uM"
    summary = summary.reset_index()

    dosed = summary.loc[summary["concentration_uM"] > 0]
    vehicle = float(summary.loc[summary["concentration_uM"] == 0, "mean"].iloc[0])
    x_left = float(dosed["concentration_uM"].min()) * 0.65
    x_right = float(dosed["concentration_uM"].max()) * 1.5

    ax.hlines(vehicle, x_left, x_right, color=color, lw=0.8, ls=(0, (2.5, 2.5)), zorder=2)
    ax.text(x_right, vehicle, "0 µM ", ha="right", va="bottom", color=color)
    ax.fill_between(dosed["concentration_uM"], dosed["mean"] - dosed["sd"],
                    dosed["mean"] + dosed["sd"], color=color, alpha=0.2, lw=0, zorder=1)
    ax.plot(dosed["concentration_uM"], dosed["mean"], marker="o", color=color,
            lw=1.0, ms=2.6, zorder=3)

    ax.set_xscale("log")
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Editing (%)")
    ax.set_title(f"{compound} and {partner}", loc="left", pad=3)
    ax.tick_params(axis="both", pad=2)
    summary.insert(0, "compound", compound)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data",
                        help="directory holding the dataset tables")
    parser.add_argument("--output", type=Path, default=REPO / "figures",
                        help="directory to write the figure into")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    use_style()
    fig = figure(183, 122)
    upper = fig.add_gridspec(2, 1, left=0.075, right=0.885, top=0.90, bottom=0.46,
                             height_ratios=[0.72, 1.0], hspace=0.45)
    ax_b = fig.add_subplot(upper[0])
    ax_c = fig.add_subplot(upper[1])
    lower = fig.add_gridspec(1, 2, left=0.075, right=0.985, top=0.33, bottom=0.085,
                             wspace=0.24)
    ax_e = fig.add_subplot(lower[0])
    ax_f = fig.add_subplot(lower[1])

    box = ax_b.get_position()
    cax = fig.add_axes([box.x1 + 0.012, box.y0, 0.010, box.height])

    frame, positions = read_positions(args.data)
    heat = draw_heatmap(ax_b, cax, frame, positions)
    fold = draw_fold_change(ax_c, args.data, positions)
    titrations = [draw_titration(ax, args.data, *spec)
                  for ax, spec in zip((ax_e, ax_f), TITRATIONS)]

    for ax, letter in ((ax_b, "b"), (ax_c, "c"), (ax_e, "e"), (ax_f, "f")):
        panel_letter(ax, letter)

    fig.savefig(args.output / "figure_1.png")
    save(fig, args.output / "figure_1.pdf")

    print("panel b: editing percent per sample and reporter position")
    print(heat.to_string(index=False), end="\n\n")

    low, high = MS2_HAIRPIN
    print(f"panel c: fold change per reporter position, hairpin {low} to {high}")
    print(fold.to_string(index=False), end="\n\n")
    for name, subset in (("hairpin", fold[fold.in_ms2_hairpin]),
                         ("flanking", fold[~fold.in_ms2_hairpin])):
        print(f"  {name}: {len(subset)} positions, "
              f"mean fold change {subset.fold_change_mean.min():.2f} to "
              f"{subset.fold_change_mean.max():.2f}, "
              f"replicate values {subset.fold_change_min.min():.2f} to "
              f"{subset.fold_change_max.max():.2f}")
    print()

    for summary in titrations:
        compound = summary["compound"].iloc[0]
        print(f"panels e and f: {compound} titration, mean and SD in editing percent")
        print(summary.to_string(index=False), end="\n\n")


if __name__ == "__main__":
    main()
