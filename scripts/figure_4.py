#!/usr/bin/env python
"""Figure 4 data panels: the RAD51-paralog deep mutational scans.

Panel b is the interaction propensity of each variant class, panel c the same
propensity by AlphaMissense class, panel d the position tracks of RAD51D and
XRCC2, and panel e the distance of called variants to the partner chain. Panels
a, f and g of the manuscript figure are a cartoon and two structure renders and
are not drawn here.

Supplementary figure 7 draws panels d and e for RAD51C and XRCC3 and imports the
functions below rather than repeating them.

    python figure_4.py
    python figure_4.py --data /path/to/data --results /path/to/results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sstats

from pyroppi import load_fit
from pyroppi import stats as ps
from pyroppi.plotting import BODY_PT, COLORS, figure, panel_letter, save, use_style

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Each scan puts its variant library on MCP against one wild-type partner on
# TadA8e. The cognate partner is the one the panels read.
SCANS = {
    "RAD51D": ("dms_rad51d", "XRCC2"),
    "XRCC2": ("dms_xrcc2", "RAD51D"),
    "RAD51C": ("dms_rad51c", "XRCC3"),
    "XRCC3": ("dms_xrcc3", "RAD51C"),
}
DX2 = ("RAD51D", "XRCC2")
CX3 = ("RAD51C", "XRCC3")

ALPHA = 0.05
MIN_READS = 20

# AlphaMissense publishes its three classes at these two cuts (Cheng et al.,
# Science 2023).
AM_BENIGN_MAX = 0.34
AM_PATHOGENIC_MIN = 0.564
AM_CLASS_ORDER = ("likely_benign", "ambiguous", "likely_pathogenic")
AM_CLASS_LABELS = {
    "likely_benign": "Benign",
    "ambiguous": "Ambiguous",
    "likely_pathogenic": "Pathogenic",
}
AM_CLASS_PAIRS = (
    ("likely_benign", "ambiguous"),
    ("ambiguous", "likely_pathogenic"),
    ("likely_benign", "likely_pathogenic"),
)

CLASS_ORDER = ("synonymous", "missense", "stop")
CLASS_LABELS = {"synonymous": "Syn.", "missense": "Missense", "stop": "Stop"}
CLASS_PAIRS = (
    ("synonymous", "missense"),
    ("missense", "stop"),
    ("synonymous", "stop"),
)
CLASS_COLORS = {
    "synonymous": COLORS["synonymous"],
    "missense": COLORS["missense"],
    "stop": COLORS["stop_gain"],
}

WALKER_MOTIFS = {
    "RAD51D": (("Walker A", 107, 114), ("Walker B", 200, 210)),
    "XRCC2": (("Walker A", 48, 55), ("Walker B", 145, 149)),
}
POOLED_MOTIF = "Walker A and B"

INTERFACE_CUTOFF_A = 8.0
# Panel c's whiskers. At 1.5 IQR the AlphaMissense classes overlap so far that
# the descent in the median is under a tenth of the drawn range.
WHISKER_PERCENTILES = (10.0, 90.0)
TRACK_WINDOW = 5
HIT_DENSITY_WINDOW = 3
DISTANCE_DRAWS = 20_000
DISTANCE_SEED = 0
DISTANCE_MIN_POSITIONS = 3

DISTANCE_SLICES = (
    ("all", "All variants"),
    ("am_benign", "AlphaMissense benign"),
    ("am_pathogenic", "AlphaMissense pathogenic"),
)


# --- loading ---------------------------------------------------------------


def scan_table(data: Path, results: Path, gene: str) -> pd.DataFrame:
    """Every variant of one scan against its cognate partner, tested and annotated.

    ``Fit.pairs`` names the ends of a canonical pair in alphabetical order, so
    the variant can sit at either end.
    """
    dataset, partner = SCANS[gene]
    fitted = load_fit(results / dataset).pairs()
    variant_is_a = fitted["protein_a"].str.contains(":")
    fitted["variant"] = np.where(variant_is_a, fitted["protein_a"], fitted["protein_b"])
    fitted["against"] = np.where(variant_is_a, fitted["protein_b"], fitted["protein_a"])
    fitted = fitted[fitted["against"] == partner].reset_index(drop=True)

    table = pd.concat([fitted, ps.parse_variants(fitted["variant"])], axis=1)
    table = ps.synonymous_test(table, score="logit_mean", by="gene", alpha=ALPHA)

    annotations = pd.read_csv(data / "annotations" / "variants.csv")
    columns = [
        "variant",
        "alphamissense",
        "alphamissense_class",
        "popEVE",
        "min_dist_to_partner_A",
        "interface_8A",
        "conservation",
    ]
    table = table.merge(annotations[columns], on="variant", how="left")
    table["am_pathogenic"] = table["alphamissense"] > AM_PATHOGENIC_MIN
    table["am_benign"] = table["alphamissense"] < AM_BENIGN_MAX
    return table


def load_scans(data: Path, results: Path, genes) -> dict[str, pd.DataFrame]:
    """One tested table per gene, keyed by gene."""
    return {gene: scan_table(data, results, gene) for gene in genes}


def missense(table: pd.DataFrame) -> pd.DataFrame:
    """The missense rows of a scan."""
    return table[table["class"] == "missense"]


# --- shared drawing helpers ------------------------------------------------


def stars(p: float) -> str:
    """Asterisks at 0.05, 0.01, 0.001 and 0.0001, or ``ns``."""
    for cut, mark in ((1e-4, "****"), (1e-3, "***"), (1e-2, "**"), (5e-2, "*")):
        if p < cut:
            return mark
    return "ns"


def draw_boxes(ax: plt.Axes, values, colors, whis) -> float:
    """One box per group; returns the highest whisker cap."""
    artists = ax.boxplot(
        values,
        positions=np.arange(len(values)),
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        whis=whis,
        medianprops=dict(color=COLORS["ink"], linewidth=0.9),
        whiskerprops=dict(color=COLORS["ink"], linewidth=0.55),
        capprops=dict(color=COLORS["ink"], linewidth=0.55),
        boxprops=dict(linewidth=0.6, edgecolor=COLORS["ink"]),
    )
    for patch, color in zip(artists["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    return max(float(np.max(line.get_ydata())) for line in artists["whiskers"])


def add_brackets(ax: plt.Axes, brackets, top: float) -> None:
    """Stack significance brackets above the boxes and open the axes for them."""
    if not brackets:
        return
    low, high = ax.get_ylim()
    span = high - low
    step = 0.135 * span
    base = top + 0.08 * span
    for index, (left, right, text) in enumerate(brackets):
        y = base + index * step
        ax.plot(
            [left, left, right, right],
            [y - 0.022 * span, y, y, y - 0.022 * span],
            color=COLORS["ink"],
            linewidth=0.5,
            clip_on=False,
        )
        ax.text(
            (left + right) / 2, y, text, ha="center", va="bottom", fontsize=BODY_PT
        )
    ax.set_ylim(low, max(high, base + (len(brackets) - 1) * step + 0.11 * span))


def percent_axis(ax: plt.Axes) -> None:
    """Label a propensity axis in percent while the data stay a probability."""
    ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:g}")


# --- panel b: propensity by variant class ----------------------------------


def position_means(
    table: pd.DataFrame, group: str, value: str = "propensity_mean"
) -> pd.DataFrame:
    """Mean of ``value`` over the variants a gene has at one position in one group.

    The variants at a residue are not independent of each other, so a residue is
    the unit every test below runs on. A position carrying variants of two groups
    contributes a mean to each.
    """
    return table.groupby(["gene", group, "position"], as_index=False)[value].mean()


def class_tests(
    tables: dict[str, pd.DataFrame], group: str, pairs, value: str = "propensity_mean"
) -> pd.DataFrame:
    """Welch on per-position means for every gene and pair, Holm over all of them."""
    rows = []
    for gene, table in tables.items():
        means = position_means(table, group, value)
        for first, second in pairs:
            left = means.loc[means[group] == first, value].to_numpy(float)
            right = means.loc[means[group] == second, value].to_numpy(float)
            welch = sstats.ttest_ind(left, right, equal_var=False)
            rows.append(
                {
                    "gene": gene,
                    "group_a": first,
                    "group_b": second,
                    "n_positions_a": len(left),
                    "n_positions_b": len(right),
                    "mean_a": float(np.mean(left)),
                    "mean_b": float(np.mean(right)),
                    "welch_t": float(welch.statistic),
                    "p_raw": float(welch.pvalue),
                }
            )
    frame = pd.DataFrame(rows)
    frame["p_holm"] = ps.holm(frame["p_raw"])
    frame["stars"] = [stars(value) for value in frame["p_holm"]]
    return frame


def group_summary(table: pd.DataFrame, group: str, order, whis) -> pd.DataFrame:
    """Quartiles, whisker caps and counts of each drawn box."""
    rows = []
    positions = position_means(table, group)
    for name in order:
        values = table.loc[table[group] == name, "propensity_mean"].to_numpy(float)
        values = values[np.isfinite(values)]
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        if isinstance(whis, tuple):
            low, high = np.percentile(values, whis)
        else:
            spread = whis * (q3 - q1)
            inside = values[(values >= q1 - spread) & (values <= q3 + spread)]
            low, high = float(inside.min()), float(inside.max())
        rows.append(
            {
                "gene": table["gene"].iloc[0],
                "group": name,
                "n_variants": int(len(values)),
                "n_positions": int((positions[group] == name).sum()),
                "whisker_low": float(low),
                "q1": float(q1),
                "median": float(median),
                "q3": float(q3),
                "whisker_high": float(high),
            }
        )
    return pd.DataFrame(rows)


def draw_class_panel(
    axes, tables: dict[str, pd.DataFrame], group: str, order, labels, pairs, colors,
    whis, rotation: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One boxplot axes per gene, with the corrected brackets above each."""
    tests = class_tests(tables, group, pairs)
    summaries = []
    for ax, (gene, table) in zip(axes, tables.items()):
        values = [
            table.loc[table[group] == name, "propensity_mean"].to_numpy(float)
            for name in order
        ]
        top = draw_boxes(ax, values, [colors[name] for name in order], whis)
        if isinstance(whis, tuple):
            low = min(float(np.percentile(part, whis[0])) for part in values)
            high = max(float(np.percentile(part, whis[1])) for part in values)
            margin = 0.06 * (high - low)
            ax.set_ylim(low - margin, high + margin)
            top = high
        brackets = []
        for _, row in tests[tests["gene"] == gene].iterrows():
            if not isinstance(whis, tuple) and row["stars"] == "ns":
                continue
            brackets.append(
                (order.index(row["group_a"]), order.index(row["group_b"]), row["stars"])
            )
        add_brackets(ax, brackets, top)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(
            [labels[name] for name in order],
            rotation=rotation,
            ha="right" if rotation else "center",
            rotation_mode="anchor" if rotation else None,
        )
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_title(gene, pad=7.0)
        percent_axis(ax)
        summaries.append(group_summary(table, group, order, whis))
    return pd.concat(summaries, ignore_index=True), tests


def am_trend(table: pd.DataFrame) -> dict:
    """Ordered-class trend and the pathogenic-against-benign area, per gene.

    Both read the synonymous-standardised z rather than the propensity. They
    depend only on the ordering, and the z is the statistic the hit call uses.
    """
    classed = table[table["alphamissense_class"].isin(AM_CLASS_ORDER)]
    ranks = {name: index for index, name in enumerate(AM_CLASS_ORDER)}
    spearman = sstats.spearmanr(
        classed["alphamissense_class"].map(ranks).to_numpy(float),
        classed["z"].to_numpy(float),
    )
    benign = classed.loc[
        classed["alphamissense_class"] == "likely_benign", "z"
    ].to_numpy(float)
    pathogenic = classed.loc[
        classed["alphamissense_class"] == "likely_pathogenic", "z"
    ].to_numpy(float)
    area, area_p = ps.auroc(benign, pathogenic)
    return {
        "gene": table["gene"].iloc[0],
        "n_classed": int(len(classed)),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "auroc_pathogenic_below_benign": area,
        "auroc_p_two_sided": area_p,
    }


# --- panel d: position tracks ----------------------------------------------


def track_bounds(table: pd.DataFrame) -> tuple[int, int]:
    """First and last substituted residue of a library."""
    positions = missense(table)["position"]
    return int(positions.min()), int(positions.max())


def rolling_over_positions(series: pd.Series, low: int, high: int, window: int, how: str):
    """Reindex onto every residue of the window, then roll over the residues.

    Rolling over the measured rows alone would give a window of five measured
    residues rather than five residues, so a gap in the library would compress
    the smoothing instead of breaking it.
    """
    full = series.reindex(pd.RangeIndex(low, high + 1, name="position"))
    roller = full.rolling(window=window, center=True, min_periods=1)
    return roller.mean() if how == "mean" else roller.sum()


def score_track(table: pd.DataFrame) -> dict:
    """Everything the score track draws for one gene."""
    low, high = track_bounds(table)
    per_position = missense(table).groupby("position")["propensity_mean"].mean()
    reference = {}
    for name in ("synonymous", "stop"):
        values = table.loc[table["class"] == name, "propensity_mean"].to_numpy(float)
        reference[name] = np.percentile(values, [25, 50, 75])
    interface = (
        missense(table)
        .loc[lambda frame: frame["min_dist_to_partner_A"] < INTERFACE_CUTOFF_A, "position"]
        .drop_duplicates()
        .to_numpy(int)
    )
    smoothed = rolling_over_positions(per_position, low, high, TRACK_WINDOW, "mean")
    smoothed = smoothed.where(per_position.reindex(smoothed.index).notna())
    return {
        "low": low,
        "high": high,
        "per_position": per_position,
        "smoothed": smoothed,
        "reference": reference,
        "interface": interface,
    }


def hit_density(table: pd.DataFrame) -> dict:
    """Rolling hit counts per residue for PyroPPI and for AlphaMissense."""
    low, high = track_bounds(table)
    rows = missense(table)
    counts = {
        "PyroPPI": rows.groupby("position")["hit"].sum(),
        "AlphaMissense": rows.groupby("position")["am_pathogenic"].sum(),
    }
    return {
        name: rolling_over_positions(
            series.astype(float), low, high, HIT_DENSITY_WINDOW, "sum"
        ).fillna(0.0)
        for name, series in counts.items()
    }


def conservation_track(table: pd.DataFrame) -> pd.Series:
    """Smoothed conservation over the residues the library covers."""
    low, high = track_bounds(table)
    per_position = missense(table).groupby("position")["conservation"].mean()
    return rolling_over_positions(per_position, low, high, TRACK_WINDOW, "mean")


def motif_tests(tables: dict[str, pd.DataFrame], motifs: dict) -> pd.DataFrame:
    """Each Walker motif against the rest of its protein, corrected over the family.

    One row per motif plus one pooled row per protein, all in one family, because
    the pooled rows are quoted beside the individual motifs. The alternative is
    one-sided: a motif lower than the background it is tested against.
    """
    rows = []
    for gene, ranges in motifs.items():
        rows_gene = missense(tables[gene])
        inside = np.zeros(len(rows_gene), dtype=bool)
        for _, start, end in ranges:
            inside |= rows_gene["position"].between(start, end).to_numpy()
        background = rows_gene.loc[~inside, "propensity_mean"].to_numpy(float)
        spans = [
            (name, rows_gene["position"].between(start, end).to_numpy())
            for name, start, end in ranges
        ]
        spans.append((POOLED_MOTIF, inside))
        for name, mask in spans:
            values = rows_gene.loc[mask, "propensity_mean"].to_numpy(float)
            result = sstats.mannwhitneyu(values, background, alternative="less")
            rows.append(
                {
                    "gene": gene,
                    "motif": name,
                    "pooled": name == POOLED_MOTIF,
                    "n_motif": int(len(values)),
                    "n_background": int(len(background)),
                    "median_motif": float(np.median(values)),
                    "median_background": float(np.median(background)),
                    "mannwhitney_u": float(result.statistic),
                    "p_raw": float(result.pvalue),
                }
            )
    frame = pd.DataFrame(rows)
    frame["p_bh"] = ps.benjamini_hochberg(frame["p_raw"])
    frame["stars"] = [stars(value) for value in frame["p_bh"]]
    return frame


def shade_motifs(ax: plt.Axes, ranges) -> None:
    """Shade each motif's residues down a track."""
    for _, start, end in ranges:
        ax.axvspan(
            start - 0.5,
            end + 0.5,
            color=COLORS["walker_motif"],
            alpha=0.18,
            linewidth=0,
            zorder=0,
        )


def draw_score_track(ax: plt.Axes, track: dict, ranges, marks) -> None:
    """Per-position missense propensity with the two control references."""
    shade_motifs(ax, ranges)
    for name, color in (("synonymous", COLORS["synonymous"]), ("stop", COLORS["stop_gain"])):
        q1, median, q3 = track["reference"][name]
        ax.axhspan(q1, q3, color=color, alpha=0.12, linewidth=0, zorder=1)
        ax.axhline(median, color=color, linestyle="--", linewidth=0.7, zorder=2)
    ax.scatter(
        track["per_position"].index,
        track["per_position"].to_numpy(),
        s=1.6,
        color=COLORS["missense"],
        alpha=0.55,
        linewidths=0,
        zorder=3,
        rasterized=True,
    )
    ax.plot(
        track["smoothed"].index,
        track["smoothed"].to_numpy(),
        color=COLORS["ink"],
        linewidth=0.9,
        zorder=4,
    )
    low, high = ax.get_ylim()
    ax.set_ylim(low - 0.14 * (high - low), high)
    ax.vlines(
        track["interface"],
        0.02,
        0.09,
        transform=ax.get_xaxis_transform(),
        color=COLORS["gray"],
        linewidth=0.8,
        zorder=5,
    )
    ax.set_ylabel("Propensity (%)")
    percent_axis(ax)
    for name, start, end in ranges:
        ax.text(
            (start + end) / 2,
            1.01,
            f"{name} {marks.get(name, '')}".strip(),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=BODY_PT,
            color=COLORS["walker_motif"],
        )


def draw_hit_track(ax: plt.Axes, density: dict, ranges) -> dict:
    """Two lanes of rolling hit counts, each scaled to its own maximum."""
    shade_motifs(ax, ranges)
    peaks = {}
    for index, (name, color) in enumerate(
        (("AlphaMissense", COLORS["alphamissense"]), ("PyroPPI", COLORS["hit"]))
    ):
        series = density[name]
        peak = float(series.max())
        peaks[name] = peak
        baseline = float(index)
        scaled = baseline + series.to_numpy() / peak * 0.85 if peak > 0 else baseline
        ax.plot(
            series.index,
            scaled,
            color=color,
            linewidth=0.7,
            drawstyle="steps-mid",
        )
        ax.axhline(baseline, color=COLORS["grid"], linewidth=0.5, zorder=0)
        ax.text(
            0.02,
            baseline + 0.88,
            name,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="top",
            fontsize=BODY_PT,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.5},
            zorder=4,
        )
    ax.set_ylim(-0.1, 2.05)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylabel("Hit density")
    return peaks


def draw_conservation_track(ax: plt.Axes, series: pd.Series, gene: str, ranges) -> None:
    """Smoothed conservation along the chain."""
    shade_motifs(ax, ranges)
    ax.fill_between(
        series.index,
        0.0,
        series.to_numpy(),
        color=COLORS["conservation"],
        alpha=0.25,
        linewidth=0,
    )
    ax.plot(series.index, series.to_numpy(), color=COLORS["conservation"], linewidth=0.7)
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel("Conservation")
    ax.set_xlabel(f"{gene} residue")


def draw_position_tracks(
    fig: plt.Figure, spec, table: pd.DataFrame, gene: str, marks: dict, ranges=()
) -> dict:
    """Three stacked tracks for one gene; returns the created axes and the peaks."""
    grid = spec.subgridspec(3, 1, height_ratios=[1.0, 0.52, 0.40], hspace=0.14)
    top = fig.add_subplot(grid[0])
    middle = fig.add_subplot(grid[1], sharex=top)
    bottom = fig.add_subplot(grid[2], sharex=top)

    track = score_track(table)
    draw_score_track(top, track, ranges, marks)
    peaks = draw_hit_track(middle, hit_density(table), ranges)
    draw_conservation_track(bottom, conservation_track(table), gene, ranges)

    for ax in (top, middle):
        ax.tick_params(axis="x", labelbottom=False)
    bottom.set_xlim(track["low"] - 1, track["high"] + 1)
    return {"score": top, "hits": middle, "conservation": bottom, "peaks": peaks,
            "track": track}


# --- panel e: distance to the partner chain --------------------------------


def slice_mask(frame: pd.DataFrame, key: str) -> np.ndarray:
    """The three drawn slices of a group."""
    if key == "all":
        return np.ones(len(frame), dtype=bool)
    if key == "am_benign":
        return frame["am_benign"].to_numpy(dtype=bool)
    return frame["am_pathogenic"].to_numpy(dtype=bool)


def distance_contrast(frame: pd.DataFrame) -> dict | None:
    """Non-hit mean minus hit mean in angstroms, with the run-replacement p.

    A structure measures residues rather than variants, so the variants collapse
    to positions and a position contributes ``n_hit`` hit slots and the rest
    non-hit slots. Positive means the called variants sit closer to the partner.
    """
    profile = (
        frame.groupby(["gene", "position"])
        .agg(
            n_hit=("hit", "sum"),
            n_assayed=("hit", "size"),
            distance=("min_dist_to_partner_A", "first"),
        )
        .reset_index()
        .sort_values(["gene", "position"])
    )
    n_hit = profile["n_hit"].to_numpy(int)
    n_assayed = profile["n_assayed"].to_numpy(int)
    if min(int((n_hit > 0).sum()), int(((n_assayed - n_hit) > 0).sum())) < DISTANCE_MIN_POSITIONS:
        return None
    distance = profile["distance"].to_numpy(float)

    def gap(hit_slots, assayed_slots):
        other = assayed_slots - hit_slots
        return float(
            other @ distance / other.sum() - hit_slots @ distance / hit_slots.sum()
        )

    observed, p_value = ps.run_replacement_test(
        profile["gene"].to_numpy(),
        n_hit,
        n_assayed,
        gap,
        draws=DISTANCE_DRAWS,
        seed=DISTANCE_SEED,
    )
    other = n_assayed - n_hit
    return {
        "hit_distance": float(n_hit @ distance / n_hit.sum()),
        "nonhit_distance": float(other @ distance / other.sum()),
        "delta_nonhit_minus_hit": observed,
        "p_value": p_value,
        "n_positions": int(len(profile)),
        "n_hit_slots": int(n_hit.sum()),
        "n_nonhit_slots": int(other.sum()),
    }


def distance_rows(tables: dict[str, pd.DataFrame], genes) -> pd.DataFrame:
    """One row per group and slice, in the order the dumbbell draws them."""
    resolved = {
        gene: missense(table)[missense(table)["min_dist_to_partner_A"].notna()].copy()
        for gene, table in tables.items()
        if gene in genes
    }
    groups = [("Combined", pd.concat(resolved.values(), ignore_index=True))]
    groups += [(gene, resolved[gene]) for gene in genes]

    rows = []
    for group, frame in groups:
        for key, label in DISTANCE_SLICES:
            subset = frame[slice_mask(frame, key)]
            result = distance_contrast(subset)
            if result is None:
                continue
            rows.append({"group": group, "slice": key, "label": label, **result})
    return pd.DataFrame(rows)


def draw_distance_dumbbell(ax: plt.Axes, rows: pd.DataFrame) -> None:
    """One row per group and slice: non-hit mean, hit mean and the corrected gap."""
    order = list(dict.fromkeys(rows["group"]))
    pitch = len(DISTANCE_SLICES) + 1
    centres = {}
    for index, (_, row) in enumerate(rows.iterrows()):
        slice_index = [key for key, _ in DISTANCE_SLICES].index(row["slice"])
        group_index = order.index(row["group"])
        y = -(group_index * pitch + slice_index)
        centres.setdefault(row["group"], []).append(y)
        ax.plot(
            [row["nonhit_distance"], row["hit_distance"]],
            [y, y],
            color=COLORS["gray"],
            linewidth=0.7,
            zorder=1,
        )
        ax.scatter(
            row["nonhit_distance"], y, s=11, color=COLORS["gray"], zorder=2, linewidths=0
        )
        ax.scatter(
            row["hit_distance"], y, s=11, color=COLORS["hit"], zorder=3, linewidths=0
        )
        mark = stars(row["p_value"])
        text = f"{row['delta_nonhit_minus_hit']:+.1f} \u00c5"
        if mark != "ns":
            text = f"{text} {mark}"
        ax.text(
            max(row["nonhit_distance"], row["hit_distance"]) + 0.6,
            y,
            text,
            ha="left",
            va="center",
            fontsize=BODY_PT,
            color=COLORS["ink"] if mark != "ns" else COLORS["gray"],
        )
        ax.text(
            -0.012,
            y,
            row["label"],
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=BODY_PT,
        )

    for group, positions in centres.items():
        ax.text(
            -0.40,
            float(np.mean(positions)),
            group,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=BODY_PT,
            fontweight="bold",
        )
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(min(min(v) for v in centres.values()) - 0.8, 0.8)
    span = rows[["hit_distance", "nonhit_distance"]].to_numpy(float)
    ax.set_xlim(span.min() - 1.5, span.max() + 5.0)
    ax.set_xlabel("Mean distance to the partner chain (\u00c5)")
    ax.scatter([], [], s=11, color=COLORS["hit"], label="Called")
    ax.scatter([], [], s=11, color=COLORS["gray"], label="Not called")
    ax.legend(loc="lower right", handletextpad=0.3, borderpad=0.0, borderaxespad=0.2)


# --- reporting -------------------------------------------------------------


def print_frame(title: str, frame: pd.DataFrame) -> None:
    print(title)
    print(frame.to_string(index=False), end="\n\n")


def report_motifs(frame: pd.DataFrame) -> None:
    print_frame(
        "panel d: Walker motifs against the rest of the protein, one-sided "
        "Mann-Whitney, Benjamini-Hochberg over the six rows",
        frame,
    )


def report_distance(frame: pd.DataFrame, label: str) -> None:
    print_frame(
        f"{label}: mean distance to the partner chain, run-replacement null at "
        f"{DISTANCE_DRAWS:,} draws and seed {DISTANCE_SEED}",
        frame,
    )


def report_tracks(axes_by_gene: dict, label: str = "panel d") -> None:
    for gene, built in axes_by_gene.items():
        track = built["track"]
        print(
            f"{label} {gene}: residues {track['low']} to {track['high']}, "
            f"{len(track['per_position'])} substituted positions, "
            f"{len(track['interface'])} of them within {INTERFACE_CUTOFF_A:g} A of the "
            f"partner; smoothed propensity {track['smoothed'].min():.4f} to "
            f"{track['smoothed'].max():.4f}; peak rolling hit count "
            + ", ".join(f"{name} {peak:.0f}" for name, peak in built["peaks"].items())
            + f"; synonymous median {track['reference']['synonymous'][1]:.4f}, "
            f"stop-gain median {track['reference']['stop'][1]:.4f}"
        )
    print()


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
    classed = {
        gene: missense(table)[
            missense(table)["alphamissense_class"].isin(AM_CLASS_ORDER)
        ]
        for gene, table in tables.items()
    }

    use_style()
    fig = figure(183, 245)
    grid_b = fig.add_gridspec(2, 2, left=0.065, right=0.455, top=0.973, bottom=0.748,
                              wspace=0.42, hspace=0.70)
    grid_c = fig.add_gridspec(2, 2, left=0.575, right=0.985, top=0.973, bottom=0.748,
                              wspace=0.42, hspace=0.70)
    axes_b = [fig.add_subplot(grid_b[index // 2, index % 2]) for index in range(4)]
    axes_c = [fig.add_subplot(grid_c[index // 2, index % 2]) for index in range(4)]

    boxes_b, tests_b = draw_class_panel(
        axes_b, tables, "class", CLASS_ORDER, CLASS_LABELS, CLASS_PAIRS,
        CLASS_COLORS, 1.5,
    )
    axes_b[0].set_ylabel("Interaction propensity (%)")
    axes_b[2].set_ylabel("Interaction propensity (%)")

    neutral = {name: COLORS["grid"] for name in AM_CLASS_ORDER}
    boxes_c, tests_c = draw_class_panel(
        axes_c, classed, "alphamissense_class", AM_CLASS_ORDER, AM_CLASS_LABELS,
        AM_CLASS_PAIRS, neutral, WHISKER_PERCENTILES, rotation=30.0,
    )
    axes_c[0].set_ylabel("Interaction propensity (%)")
    axes_c[2].set_ylabel("Interaction propensity (%)")
    for ax in axes_c[2:]:
        ax.set_xlabel("AlphaMissense class")

    marks = motif_tests(tables, WALKER_MOTIFS)
    by_gene = {}
    grid_d = fig.add_gridspec(1, 2, left=0.085, right=0.985, top=0.655, bottom=0.287,
                              wspace=0.20)
    for column, gene in enumerate(DX2):
        gene_marks = dict(
            zip(
                marks.loc[(marks["gene"] == gene) & ~marks["pooled"], "motif"],
                marks.loc[(marks["gene"] == gene) & ~marks["pooled"], "stars"],
            )
        )
        by_gene[gene] = draw_position_tracks(
            fig, grid_d[0, column], tables[gene], gene, gene_marks, WALKER_MOTIFS[gene]
        )

    rows = distance_rows(tables, DX2)
    grid_e = fig.add_gridspec(1, 1, left=0.285, right=0.965, top=0.228, bottom=0.045)
    ax_e = fig.add_subplot(grid_e[0, 0])
    draw_distance_dumbbell(ax_e, rows)

    panel_letter(axes_b[0], "b", dx=-0.30)
    panel_letter(axes_c[0], "c", dx=-0.30)
    panel_letter(by_gene["RAD51D"]["score"], "d", dx=-0.16)
    panel_letter(ax_e, "e", dx=-0.42)

    fig.savefig(args.output / "figure_4.png")
    save(fig, args.output / "figure_4.pdf")

    print_frame("panel b: interaction propensity by variant class", boxes_b)
    print_frame(
        "panel b: Welch on per-position means, Holm over all twelve comparisons",
        tests_b,
    )
    print_frame("panel c: interaction propensity by AlphaMissense class", boxes_c)
    print_frame(
        "panel c: Welch on per-position means, Holm over all twelve comparisons",
        tests_c,
    )
    print_frame(
        "panel c: ordered-class trend and the pathogenic-against-benign area, on z",
        pd.DataFrame([am_trend(table) for table in classed.values()]),
    )
    report_tracks(by_gene)
    report_motifs(marks)
    report_distance(rows, "panel e")


if __name__ == "__main__":
    main()
