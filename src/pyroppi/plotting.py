"""Shared figure style for the manuscript panels.

Two type sizes are used throughout: 6 pt for body text and 8 pt bold for panel
letters. Colours are chosen to stay distinguishable in common forms of colour
vision deficiency.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

BODY_PT = 6
LETTER_PT = 8

SINGLE_COLUMN_MM = 89
DOUBLE_COLUMN_MM = 183
MAX_HEIGHT_MM = 247

COLORS = {
    "ink": "#111111",
    "black": "#222222",
    "gray": "#6F6F6F",
    "grid": "#D9D9D9",
    "synonymous": "#009E73",
    "missense": "#7F7F7F",
    "stop_gain": "#E69F00",
    "hit": "#D55E00",
    "alphamissense": "#009E73",
    "popeve": "#0072B2",
    "conservation": "#2A6F74",
    "walker_motif": "#CC79A7",
    "rad51d": "#0072B2",
    "xrcc2": "#DDCC77",
    "modelled": "#D55E00",
    "pom_300": "#0072B2",
    "pom_3": "#D55E00",
    "control": "#8E8E8E",
    "threshold": "#B84A4A",
    "pair": "#C96A2D",
    "drug": "#4E9A51",
    "cell": "#5A9FD4",
}

FONT_STACK = ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]


def use_style() -> None:
    """Apply the manuscript rcParams. Call once before building a figure."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "font.size": BODY_PT,
            "axes.labelsize": BODY_PT,
            "axes.titlesize": BODY_PT,
            "xtick.labelsize": BODY_PT,
            "ytick.labelsize": BODY_PT,
            "legend.fontsize": BODY_PT,
            "axes.linewidth": 0.65,
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "xtick.major.size": 2.7,
            "ytick.major.size": 2.7,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.5,
            "lines.linewidth": 0.9,
            "lines.markersize": 2.5,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.01,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def figure(width_mm: float, height_mm: float) -> plt.Figure:
    """Create a figure sized in millimetres, as journals specify widths."""
    if height_mm > MAX_HEIGHT_MM:
        raise ValueError(f"height {height_mm} mm exceeds the {MAX_HEIGHT_MM} mm page")
    return plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))


def panel_letter(ax: plt.Axes, letter: str, dx: float = -0.06, dy: float = 1.04) -> None:
    """Place a bold panel letter in the axes' top left corner."""
    ax.text(
        dx,
        dy,
        letter,
        transform=ax.transAxes,
        fontsize=LETTER_PT,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save(fig: plt.Figure, path) -> None:
    """Write a figure and close it."""
    fig.savefig(path)
    plt.close(fig)
