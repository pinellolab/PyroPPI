#!/usr/bin/env python
"""Supplementary figure 8: PyroPPI against the sequence predictors.

Panel a is the concordance of the calls, panel b the per-scan rank correlations
of the three scores, and panels c and d the position means of the propensity
against each predictor.

Panel b reads the three scores in a damage orientation, so that one number
compares the two predictors: the propensity and popEVE are negated and
AlphaMissense is taken as published, and a positive correlation then means the
two agree about which variants are damaging. Panels c and d plot the scores as
published, so their coefficients are the correlation of exactly what is drawn
and AlphaMissense reads negative there.

    python supp_figure_8.py
    python supp_figure_8.py --data /path/to/data --results /path/to/results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from figure_4 import AM_PATHOGENIC_MIN, SCANS, load_scans, missense  # noqa: E402
from pyroppi.plotting import BODY_PT, COLORS, figure, panel_letter, save, use_style  # noqa: E402

GENE_ORDER = ("RAD51D", "XRCC2", "RAD51C", "XRCC3")

# The bootstrap. One resample matrix per block, drawn in one call, so the three
# correlations of a block share their resamples and the difference between two
# of them is a paired difference rather than the gap between two intervals.
N_BOOT = 4000
BOOT_SEED = 20260914
BOOT_CHUNK = 256
CI_PERCENTILES = (2.5, 97.5)
POOLED_RANK = "pooled_within_gene_rank"

SERIES = (
    ("ppi_am", "PyroPPI and AlphaMissense", COLORS["alphamissense"], "o"),
    ("ppi_popeve", "PyroPPI and popEVE", COLORS["popeve"], "s"),
    ("am_popeve", "AlphaMissense and popEVE", COLORS["gray"], "^"),
)
SERIES_OFFSET = {"ppi_am": -0.26, "ppi_popeve": 0.0, "am_popeve": 0.26}

# Panels c and d: the predictor, its column, the drawn axis name and the colour.
PREDICTORS = (
    ("am", "alphamissense", "AlphaMissense score", COLORS["alphamissense"]),
    ("popeve", "popEVE", "popEVE score", COLORS["popeve"]),
)

SET_LABEL = {"ppi_recorder": "PyroPPI hit", "am": "AlphaMissense pathogenic"}
SET_COLOR = {"ppi_recorder": COLORS["hit"], "am": COLORS["alphamissense"]}


# --- panel a: the two call sets --------------------------------------------


def lens_area(distance: float, radius_a: float, radius_b: float) -> float:
    """Area shared by two circles whose centres are ``distance`` apart."""
    if distance >= radius_a + radius_b:
        return 0.0
    if distance <= abs(radius_a - radius_b):
        return float(np.pi * min(radius_a, radius_b) ** 2)
    term_a = radius_a**2 * np.arccos(
        (distance**2 + radius_a**2 - radius_b**2) / (2 * distance * radius_a)
    )
    term_b = radius_b**2 * np.arccos(
        (distance**2 + radius_b**2 - radius_a**2) / (2 * distance * radius_b)
    )
    wedge = 0.5 * np.sqrt(
        (-distance + radius_a + radius_b)
        * (distance + radius_a - radius_b)
        * (distance - radius_a + radius_b)
        * (distance + radius_a + radius_b)
    )
    return float(term_a + term_b - wedge)


def solve_geometry(counts: dict) -> dict:
    """Radii and centre distance that make all three drawn areas exact.

    One area unit stands for the whole scan, so a region of k variants takes
    area k over n. Two circles have exactly enough freedom for three counts: one
    radius per set and the centre distance for the overlap. The lens area is
    monotone in the distance between the two degenerate cases, so the distance
    is a bracketed root rather than a fit.
    """
    total = counts["n_variants"]
    radius = {
        key: float(np.sqrt(counts[f"n_{key}"] / total / np.pi))
        for key in ("ppi_recorder", "am")
    }
    target = counts["ppi_recorder_and_am"] / total
    low = abs(radius["am"] - radius["ppi_recorder"])
    high = radius["am"] + radius["ppi_recorder"]
    distance = float(
        optimize.brentq(
            lambda value: lens_area(value, radius["am"], radius["ppi_recorder"])
            - target,
            low + 1e-12,
            high - 1e-12,
            xtol=1e-14,
        )
    )
    return {
        "radius_ppi_recorder": radius["ppi_recorder"],
        "radius_am": radius["am"],
        "centre_distance": distance,
    }


def geometry_error(counts: dict, geometry: dict) -> float:
    """Largest gap, in variants, between a drawn area and the count it stands for."""
    total = counts["n_variants"]
    overlap = lens_area(
        geometry["centre_distance"],
        geometry["radius_am"],
        geometry["radius_ppi_recorder"],
    )
    drawn = {
        "ppi_recorder_and_am": overlap,
        "only_ppi_recorder": np.pi * geometry["radius_ppi_recorder"] ** 2 - overlap,
        "only_am": np.pi * geometry["radius_am"] ** 2 - overlap,
    }
    drawn["neither"] = 1.0 - sum(drawn.values())
    return max(abs(value - counts[name] / total) for name, value in drawn.items()) * total


def concordance(frame: pd.DataFrame) -> pd.DataFrame:
    """Region counts and the association, pooled over the scans and per scan."""
    gene = frame["gene"].to_numpy()
    calls = {
        "ppi_recorder": frame["hit"].to_numpy(dtype=bool),
        "am": (frame["alphamissense"] > AM_PATHOGENIC_MIN).to_numpy(dtype=bool),
    }

    records = []
    for scope in ("pooled", *GENE_ORDER):
        rows = np.ones(len(frame), dtype=bool) if scope == "pooled" else gene == scope
        ppi_recorder, am = calls["ppi_recorder"][rows], calls["am"][rows]
        record = {
            "scope": scope,
            "drawn": scope == "pooled",
            "n_variants": int(rows.sum()),
            "n_ppi_recorder": int(ppi_recorder.sum()),
            "n_am": int(am.sum()),
            "only_ppi_recorder": int((ppi_recorder & ~am).sum()),
            "ppi_recorder_and_am": int((ppi_recorder & am).sum()),
            "only_am": int((~ppi_recorder & am).sum()),
            "neither": int((~ppi_recorder & ~am).sum()),
        }
        odds_ratio, p_value = stats.fisher_exact(
            [
                [record["ppi_recorder_and_am"], record["only_ppi_recorder"]],
                [record["only_am"], record["neither"]],
            ]
        )
        record["odds_ratio"] = float(odds_ratio)
        record["fisher_exact_p"] = float(p_value)
        geometry = solve_geometry(record)
        record.update(geometry)
        record["max_area_error_in_variants"] = geometry_error(record, geometry)
        records.append(record)
    return pd.DataFrame(records)


def draw_euler(ax: plt.Axes, counts: pd.Series) -> None:
    """Two circles whose three drawn areas are in proportion to their counts."""
    radius_ppi_recorder = counts["radius_ppi_recorder"]
    radius_am = counts["radius_am"]
    distance = counts["centre_distance"]
    centre_ppi_recorder = -distance / 2.0
    centre_am = distance / 2.0

    for centre, radius, key in (
        (centre_ppi_recorder, radius_ppi_recorder, "ppi_recorder"),
        (centre_am, radius_am, "am"),
    ):
        ax.add_patch(
            Circle(
                (centre, 0.0),
                radius,
                facecolor=SET_COLOR[key],
                edgecolor=SET_COLOR[key],
                alpha=0.22,
                linewidth=0.7,
            )
        )

    left_edge = centre_ppi_recorder - radius_ppi_recorder
    lens_left = centre_am - radius_am
    lens_right = centre_ppi_recorder + radius_ppi_recorder
    right_edge = centre_am + radius_am
    anchors = (
        ((left_edge + lens_left) / 2.0, counts["only_ppi_recorder"]),
        ((lens_left + lens_right) / 2.0, counts["ppi_recorder_and_am"]),
        ((lens_right + right_edge) / 2.0, counts["only_am"]),
    )
    for x, value in anchors:
        ax.text(x, 0.0, f"{int(value):,}", ha="center", va="center", fontsize=BODY_PT)

    for centre, radius, key in (
        (centre_ppi_recorder, radius_ppi_recorder, "ppi_recorder"),
        (centre_am, radius_am, "am"),
    ):
        ax.text(
            centre,
            radius + 0.02,
            SET_LABEL[key],
            ha="center",
            va="bottom",
            fontsize=BODY_PT,
            color=SET_COLOR[key],
        )
    ax.text(
        0.0,
        -radius_am - 0.03,
        f"{int(counts['neither']):,} called by neither, of {int(counts['n_variants']):,}",
        ha="center",
        va="top",
        fontsize=BODY_PT,
        color=COLORS["gray"],
    )

    margin = 0.03
    ax.set_xlim(left_edge - margin, right_edge + margin)
    ax.set_ylim(-radius_am - 0.14, radius_am + 0.10)
    ax.set_aspect("equal")
    # Equal aspect shrinks the axes box inside its cell; held to the left of the
    # cell so the panel letter sits beside the diagram.
    ax.set_anchor("W")
    ax.axis("off")


# --- panel b: the rank correlations ----------------------------------------


def damage_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The three scores turned to point the same way, large meaning damaging."""
    out = frame[["gene", "variant", "position", "propensity_mean"]].copy()
    out["damage_ppi_recorder"] = -frame["propensity_mean"].to_numpy()
    out["damage_am"] = frame["alphamissense"].to_numpy()
    out["damage_popeve"] = -frame["popEVE"].to_numpy()
    return out.dropna(subset=["damage_ppi_recorder", "damage_am", "damage_popeve"])


def within_gene_ranks(frame: pd.DataFrame, columns) -> pd.DataFrame:
    """Rank each column inside its gene and scale to the open unit interval.

    Pooling raw scores would let a between-gene offset stand in for agreement,
    and popEVE carries one because its scale is per protein.
    """
    out = frame.copy()
    for column in columns:
        out[column] = out.groupby("gene")[column].transform(
            lambda values: stats.rankdata(values) / (len(values) + 1)
        )
    return out


def rowwise_pearson(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pearson of each row of ``left`` against the same row of ``right``."""
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    return np.einsum("ij,ij->i", left, right) / np.sqrt(
        np.einsum("ij,ij->i", left, left) * np.einsum("ij,ij->i", right, right)
    )


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(stats.rankdata(x), stats.rankdata(y))[0, 1])


def interval(draws: np.ndarray) -> tuple[float, float]:
    finite = draws[np.isfinite(draws)]
    low, high = np.percentile(finite, CI_PERCENTILES)
    return float(low), float(high)


def paired_block(ppi, am, popeve, rng) -> dict:
    """Three correlations, their intervals, and the popEVE-minus-AlphaMissense gap."""
    n = len(ppi)
    indices = rng.integers(0, n, size=(N_BOOT, n))
    pairs = (("ppi_am", 0, 1), ("ppi_popeve", 0, 2), ("am_popeve", 1, 2))
    columns = (ppi, am, popeve)
    drawn = {key: [] for key, *_ in pairs}
    # In slices of the resample matrix, since holding every resampled column and
    # its ranks at once costs gigabytes on a pooled block.
    for start in range(0, N_BOOT, BOOT_CHUNK):
        rows = indices[start : start + BOOT_CHUNK]
        # Spearman is Pearson on ranks, and each column is ranked once here
        # rather than once per correlation it appears in.
        ranks = [stats.rankdata(column[rows], axis=1) for column in columns]
        for key, left, right in pairs:
            drawn[key].append(rowwise_pearson(ranks[left], ranks[right]))
    drawn = {key: np.concatenate(parts) for key, parts in drawn.items()}

    out = {"n": n}
    for key, point in (
        ("ppi_am", spearman(ppi, am)),
        ("ppi_popeve", spearman(ppi, popeve)),
        ("am_popeve", spearman(am, popeve)),
    ):
        out[key] = point
        out[f"{key}_lo"], out[f"{key}_hi"] = interval(drawn[key])

    out["diff"] = spearman(ppi, popeve) - spearman(ppi, am)
    difference = drawn["ppi_popeve"] - drawn["ppi_am"]
    out["diff_lo"], out["diff_hi"] = interval(difference)
    finite = difference[np.isfinite(difference)]
    below = float((finite <= 0).mean())
    out["diff_p"] = float(min(1.0, 2 * min(below, 1 - below)))
    return out


def correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per scope: the four scans and the pooled within-gene ranks.

    The gene blocks are resampled first and in :data:`GENE_ORDER`, so a scan's
    own interval does not move with what is pooled after it.
    """
    rng = np.random.default_rng(BOOT_SEED)
    columns = ["damage_ppi_recorder", "damage_am", "damage_popeve"]
    blocks = [(gene, frame[frame["gene"] == gene]) for gene in GENE_ORDER]
    blocks.append((POOLED_RANK, within_gene_ranks(frame, columns)))

    rows = []
    for scope, block in blocks:
        estimates = paired_block(
            block["damage_ppi_recorder"].to_numpy(float),
            block["damage_am"].to_numpy(float),
            block["damage_popeve"].to_numpy(float),
            rng,
        )
        rows.append({"scope": scope, **estimates})
    return pd.DataFrame(rows)


def draw_correlation_forest(ax: plt.Axes, table: pd.DataFrame) -> None:
    """One row per scan, three correlations each, with their bootstrap intervals."""
    by_scope = table.set_index("scope")
    for index, gene in enumerate(GENE_ORDER):
        row = by_scope.loc[gene]
        for key, label, color, marker in SERIES:
            y = index + SERIES_OFFSET[key]
            ax.plot(
                [row[f"{key}_lo"], row[f"{key}_hi"]],
                [y, y],
                color=color,
                linewidth=0.9,
                solid_capstyle="butt",
            )
            ax.plot(
                row[key], y, marker=marker, markersize=2.6, color=color,
                markeredgecolor="white", markeredgewidth=0.3, linestyle="none",
            )
            if index == 0:
                # Named beside the top scan's own markers, so the panel carries
                # no legend. The axis runs past the widest interval to make the
                # room; its line stops where the data stop.
                ax.text(
                    0.95, y, label, ha="left", va="center", fontsize=BODY_PT, color=color
                )
    ax.axvline(0.0, color=COLORS["grid"], linewidth=0.5)
    ax.set_yticks(range(len(GENE_ORDER)), GENE_ORDER)
    ax.set_ylim(len(GENE_ORDER) - 0.5, -0.65)
    ax.set_xlim(-0.06, 1.62)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.spines["bottom"].set_bounds(-0.06, 0.9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Spearman rho, damage orientation")


# --- panels c and d: the position means ------------------------------------


def position_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per gene and residue: the propensity and each predictor, averaged."""
    complete = frame.dropna(subset=["propensity_mean", "alphamissense", "popEVE"])
    return (
        complete.groupby(["gene", "position"], as_index=False)
        .agg(
            n_variants=("variant", "size"),
            propensity_mean=("propensity_mean", "mean"),
            alphamissense=("alphamissense", "mean"),
            popEVE=("popEVE", "mean"),
        )
    )


def position_correlations(table: pd.DataFrame) -> pd.DataFrame:
    """Spearman and Pearson of the drawn position means, per gene and predictor."""
    rows = []
    for gene in GENE_ORDER:
        block = table[table["gene"] == gene]
        propensity = block["propensity_mean"].to_numpy(float)
        for key, column, _, _ in PREDICTORS:
            predictor = block[column].to_numpy(float)
            rho = stats.spearmanr(propensity, predictor)
            pearson = stats.pearsonr(propensity, predictor)
            rows.append(
                {
                    "gene": gene,
                    "predictor": key,
                    "n_positions": int(len(block)),
                    "spearman_rho": float(rho.statistic),
                    "spearman_p": float(rho.pvalue),
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                }
            )
    return pd.DataFrame(rows)


def draw_position_scatter(
    ax: plt.Axes, table: pd.DataFrame, found: pd.DataFrame, gene: str, key: str
) -> None:
    """One gene's residues against one predictor, with its two coefficients."""
    column, label, color = next(
        (col, lab, colour) for name, col, lab, colour in PREDICTORS if name == key
    )
    block = table[table["gene"] == gene]
    ax.scatter(
        block[column].to_numpy(float),
        block["propensity_mean"].to_numpy(float),
        s=1.8,
        color=color,
        alpha=0.7,
        linewidths=0,
        rasterized=True,
    )
    row = found[(found["gene"] == gene) & (found["predictor"] == key)].iloc[0]
    ax.text(
        0.03,
        0.03,
        f"rho = {row['spearman_rho']:.2f}, P = {row['spearman_p']:.1e}\n"
        f"r = {row['pearson_r']:.2f}, P = {row['pearson_p']:.1e}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=BODY_PT,
    )
    ax.set_title(gene, pad=2.0)
    ax.set_xlabel(label)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:g}")
    ax.margins(0.08, 0.22)


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

    tables = load_scans(args.data, args.results, SCANS)
    variants = pd.concat(
        [missense(tables[gene]) for gene in GENE_ORDER], ignore_index=True
    )

    regions = concordance(variants)
    forest = correlations(damage_frame(variants))
    positions = position_table(variants)
    found = position_correlations(positions)

    use_style()
    fig = figure(183, 170)
    grid_a = fig.add_gridspec(1, 1, left=0.055, right=0.400, top=0.958, bottom=0.645)
    ax_a = fig.add_subplot(grid_a[0, 0])
    draw_euler(ax_a, regions[regions["drawn"]].iloc[0])

    grid_b = fig.add_gridspec(1, 1, left=0.520, right=0.985, top=0.925, bottom=0.690)
    ax_b = fig.add_subplot(grid_b[0, 0])
    draw_correlation_forest(ax_b, forest)

    rows = {}
    for key, (letter, top, bottom) in zip(
        ("am", "popeve"), (("c", 0.585, 0.375), ("d", 0.272, 0.062))
    ):
        grid = fig.add_gridspec(1, 4, left=0.065, right=0.985, top=top, bottom=bottom,
                                wspace=0.40)
        axes = [fig.add_subplot(grid[0, column]) for column in range(4)]
        for ax, gene in zip(axes, GENE_ORDER):
            draw_position_scatter(ax, positions, found, gene, key)
        axes[0].set_ylabel("Interaction propensity (%)")
        rows[letter] = axes

    panel_letter(ax_a, "a", dx=0.0, dy=0.98)
    panel_letter(ax_b, "b", dx=-0.13)
    panel_letter(rows["c"][0], "c", dx=-0.38)
    panel_letter(rows["d"][0], "d", dx=-0.38)

    fig.savefig(args.output / "supp_figure_8.png")
    save(fig, args.output / "supp_figure_8.pdf")

    print("panel a: PyroPPI hits against AlphaMissense pathogenic calls")
    print(regions.to_string(index=False), end="\n\n")
    print(
        f"panel b: Spearman with percentile intervals of {N_BOOT:,} resamples at "
        f"seed {BOOT_SEED}, one resample shared by the three correlations of a scan"
    )
    print(forest.to_string(index=False), end="\n\n")
    pooled = forest[forest["scope"] == POOLED_RANK].iloc[0]
    print(
        f"pooled within-gene ranks: popEVE minus AlphaMissense "
        f"{pooled['diff']:+.4f} [{pooled['diff_lo']:+.4f}, {pooled['diff_hi']:+.4f}], "
        f"bootstrap P = {pooled['diff_p']:.4f}, n = {int(pooled['n']):,}"
    )
    individually = [
        row["scope"]
        for _, row in forest.iterrows()
        if row["scope"] in GENE_ORDER and (row["diff_lo"] > 0 or row["diff_hi"] < 0)
    ]
    print(f"scans whose own difference excludes zero: {individually or 'none'}", end="\n\n")
    print("panels c and d: position means, one residue per point")
    print(found.to_string(index=False), end="\n\n")
    print(
        "positions per scan: "
        + ", ".join(
            f"{gene} {int((positions['gene'] == gene).sum())}" for gene in GENE_ORDER
        )
    )


if __name__ == "__main__":
    main()
