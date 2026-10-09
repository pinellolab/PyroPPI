#!/usr/bin/env python
"""Figure 5: the immunoprecipitation arms of the RAD51D and XRCC2 variant pool.

One pool of RAD51D and XRCC2 variants was sequenced three ways. The PPI Recorder
arm is unenriched and is the arm PyroPPI fits, so every variant carries an
interaction propensity. The anti-FLAG arm immunoprecipitates the wild-type
partner, which carries the FLAG tag; its contrast against the PPI Recorder arm
is called binding here. The anti-myc arm immunoprecipitates the variant itself,
which carries the myc tag; its contrast against the PPI Recorder arm is called
stability here.

Panels a and g of the manuscript figure are a cartoon and a structure render, so
this script draws panels b to f and keeps those letters. Supplementary Figure 10
reuses the same five drawings for the RAD51D sublibrary.

    python figure_5.py
    python figure_5.py --output /tmp/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy import stats as scipy_stats

import pyroppi
from pyroppi import stats
from pyroppi.plotting import COLORS, figure, panel_letter, save, use_style

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_DATA = REPO / "data"
DEFAULT_RESULTS = REPO / "results"
DEFAULT_OUTPUT = REPO / "figures"

# The three sequencing arms, named as arms.csv names them, with the short key
# each one's columns carry.
ARM_KEYS = {"PPI Recorder": "ppi_recorder", "anti-FLAG": "flag", "anti-myc": "myc"}
FIT_DIRS = {"RAD51D": "pulldown_rad51d", "XRCC2": "pulldown_xrcc2"}

# The one-sided nominal threshold, in synonymous standard deviations.
Z_THRESHOLD = -1.645
PLANE_LIMITS = (-8.5, 4.0)
BENIGN_CLASS = "likely_benign"

# Panels c and d take their colour limits from the variants they draw, at these
# percentiles, so each colour map spends its range where the points are. One
# rule serves both panels and both sublibraries.
COLOUR_PERCENTILES = (2.0, 98.0)

# One shape and one fill per variant class, so the three are told apart twice
# over and the distinction survives in greyscale.
CLASS_MARKS = (
    ("missense", "Missense", "o", 5.0, COLORS["missense"]),
    ("synonymous", "Synonymous", "D", 9.0, COLORS["synonymous"]),
    ("stop", "Stop-gain", "s", 10.0, COLORS["stop_gain"]),
)

# Panel e's resampling. Substitutions at one residue behave alike, so both the
# p-value and the interval move whole residues rather than single variants.
N_PERMUTATIONS = 10_000
N_BOOTSTRAP = 2_000
STATS_SEED = 0
MIN_GROUP = 5

# Panel f's columns, as (column, sign, label). Every column is oriented so a
# higher value means more tolerated, which makes a positive correlation the
# expected direction everywhere.
CORRELATION_SCORES = (
    ("propensity", +1, "Propensity"),
    ("binding_z", +1, "Binding"),
    ("stability_z", +1, "Stability"),
    ("alphamissense", -1, "−AlphaMissense"),
    ("conservation", -1, "−Conservation"),
)

WIDTH_MM = 183.0
HEIGHT_MM = 214.0
# Each panel's box in millimetres as (x, y, width, height), measured from the
# figure's bottom left corner.
PANEL_BOXES = {
    "recovery": (18.0, 156.0, 78.0, 50.0),
    "plane_propensity": (18.0, 76.0, 56.0, 56.0),
    "plane_alphamissense": (92.0, 76.0, 56.0, 56.0),
    "bar_propensity": (18.0, 136.5, 56.0, 2.4),
    "bar_alphamissense": (92.0, 136.5, 56.0, 2.4),
    "group_test": (18.0, 14.0, 54.0, 46.0),
    "correlations": (104.0, 16.0, 42.0, 42.0),
}


# --- loading and scoring ---------------------------------------------------


def arm_log2rpm(arms: pd.DataFrame) -> pd.DataFrame:
    """Mean log2 reads per million of each arm, one row per variant.

    Depth is summed over every variant of the pool, which is how the pool was
    physically sequenced, so the two sublibraries share one denominator per
    sample. Averaging in log space is a pseudocount-shifted geometric mean of
    reads per million, which absorbs the unequal replicate counts across arms.
    """
    counts = arms.pivot_table(
        index="variant", columns=["arm", "replicate"], values="total_reads"
    )
    log2rpm = np.log2(counts.div(counts.sum(axis=0), axis=1) * 1e6 + 1.0)
    return pd.DataFrame(
        {f"log2rpm_{key}": log2rpm[arm].mean(axis=1) for arm, key in ARM_KEYS.items()}
    )


def propensity_table(results_dir: Path) -> pd.DataFrame:
    """Interaction propensity of every variant, read off the two fits."""
    tables = []
    for directory in FIT_DIRS.values():
        pairs = pyroppi.load_fit(results_dir / directory).pairs()
        # The variant is whichever end of the pair carries a substitution; the
        # other end is the wild-type partner.
        variant = np.where(
            pairs.protein_a.str.contains(":"), pairs.protein_a, pairs.protein_b
        )
        tables.append(
            pd.DataFrame(
                {
                    "variant": variant,
                    "pair_logit": pairs.logit_mean.to_numpy(),
                    "propensity": pairs.propensity_mean.to_numpy(),
                }
            )
        )
    return pd.concat(tables, ignore_index=True)


def build(data_dir: Path, results_dir: Path) -> pd.DataFrame:
    """One row per variant: the arm scores, their z, the fit and the predictors."""
    arms = pd.read_csv(data_dir / "pulldown_arms" / "arms.csv")
    frame = arm_log2rpm(arms).reset_index()
    frame["binding"] = frame.log2rpm_flag - frame.log2rpm_ppi_recorder
    frame["stability"] = frame.log2rpm_myc - frame.log2rpm_ppi_recorder
    frame = pd.concat([frame, stats.parse_variants(frame.variant)], axis=1)
    frame = frame.merge(propensity_table(results_dir), on="variant", validate="1:1")

    for score in ("binding", "stability"):
        tested = stats.synonymous_test(frame, score=score, by="gene")
        frame[f"{score}_z"] = tested.z
        frame[f"{score}_p"] = tested.p
        frame[f"{score}_p_holm"] = tested.groupby("gene").p.transform(stats.holm)
    frame["pyroppi_hit"] = stats.synonymous_test(frame, score="pair_logit", by="gene").hit

    annotations = pd.read_csv(data_dir / "annotations" / "variants.csv")
    columns = ["variant", "alphamissense", "alphamissense_class", "conservation"]
    frame = frame.merge(annotations[columns], on="variant", how="left")
    frame["benign"] = frame.alphamissense_class.eq(BENIGN_CLASS)
    frame["residue"] = frame.gene + ":" + frame.position.astype(str)
    return frame


def sublibrary(frame: pd.DataFrame, gene: str) -> pd.DataFrame:
    """Every variant of one sublibrary."""
    return frame[frame.gene == gene].reset_index(drop=True)


def missense(frame: pd.DataFrame) -> pd.DataFrame:
    """The missense variants that carry a predictor score."""
    rows = frame[frame["class"].eq("missense") & frame.alphamissense.notna()]
    return rows.reset_index(drop=True)


def format_p(p_value: float) -> str:
    """A p-value at the resolution the test that produced it supports."""
    return "P < 0.001" if p_value < 0.001 else f"P = {p_value:.3f}"


# --- panel b ---------------------------------------------------------------


def draw_recovery(ax: plt.Axes, frame: pd.DataFrame, gene: str) -> pd.DataFrame:
    """Interaction propensity against the anti-FLAG arm's own log2 RPM.

    Both axes are raw measurements and neither is a ratio, so the two share no
    denominator. The partial correlation removes the PPI Recorder arm's log2
    RPM, which is the sequencing depth both axes are measured over.

    The three variant classes overlap across the whole cloud, so they are named
    in a strip under the axes rather than at their own points.
    """
    handles = []
    for variant_class, label, marker, size, colour in CLASS_MARKS:
        rows = frame[frame["class"].eq(variant_class)]
        ax.scatter(
            rows.log2rpm_flag,
            rows.propensity,
            s=size,
            marker=marker,
            facecolor=colour,
            edgecolor="white",
            linewidths=0.2,
            zorder=3,
        )
        handles.append(
            Line2D(
                [],
                [],
                marker=marker,
                ls="none",
                markersize=np.sqrt(size),
                markerfacecolor=colour,
                markeredgecolor="white",
                markeredgewidth=0.2,
                color=colour,
                label=f"{label} ({len(rows)})",
            )
        )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=len(handles),
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.6,
        borderpad=0.0,
        borderaxespad=0.0,
    )

    slope, intercept = np.polyfit(frame.log2rpm_flag, frame.propensity, 1)
    grid = np.linspace(frame.log2rpm_flag.min(), frame.log2rpm_flag.max(), 32)
    ax.plot(grid, slope * grid + intercept, color=COLORS["ink"], lw=0.8, zorder=4)

    rho, rho_p = scipy_stats.spearmanr(frame.log2rpm_flag, frame.propensity)
    r_value, r_p = scipy_stats.pearsonr(frame.log2rpm_flag, frame.propensity)
    partial_rho, partial_p = stats.partial_spearman(
        frame.log2rpm_flag, frame.propensity, frame.log2rpm_ppi_recorder
    )
    ax.text(
        0.03,
        0.97,
        f"{gene}  (n = {len(frame)})\n"
        f"Spearman ρ = {rho:+.2f}\nPearson r = {r_value:+.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        linespacing=1.3,
    )
    ax.set_xlabel("anti-FLAG arm (log$_2$ RPM)")
    ax.set_ylabel("Interaction propensity")
    return pd.DataFrame(
        [
            {
                "panel": "recovery",
                "gene": gene,
                "n": len(frame),
                "spearman_rho": float(rho),
                "spearman_p": float(rho_p),
                "pearson_r": float(r_value),
                "pearson_p": float(r_p),
                "partial_spearman_rho": partial_rho,
                "partial_spearman_p": partial_p,
                "slope": float(slope),
                "intercept": float(intercept),
            }
        ]
    )


# --- panels c and d --------------------------------------------------------


def draw_plane(
    ax: plt.Axes,
    cax: plt.Axes,
    frame: pd.DataFrame,
    colour_by: str,
    show_y_label: bool = True,
) -> dict:
    """Binding against stability for every annotated missense variant.

    Both axes are z against the same synonymous pool, so the plane is square and
    the dashed diagonal is where the two arms move together. ``colour_by`` is
    ``propensity`` for panel c or ``alphamissense`` for panel d.

    The shaded corner holds the variants whose binding z is at or below the
    threshold while their stability z stays above it.

    Returns the colour limits the panel took, which the caller reports.
    """
    low, high = PLANE_LIMITS
    ax.add_patch(
        Rectangle(
            (Z_THRESHOLD, low),
            high - Z_THRESHOLD,
            Z_THRESHOLD - low,
            facecolor=COLORS["grid"],
            alpha=0.45,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.plot(PLANE_LIMITS, PLANE_LIMITS, color=COLORS["gray"], lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.axvline(Z_THRESHOLD, color=COLORS["gray"], lw=0.5, ls=(0, (1, 1.6)), zorder=1)
    ax.axhline(Z_THRESHOLD, color=COLORS["gray"], lw=0.5, ls=(0, (1, 1.6)), zorder=1)

    if colour_by == "propensity":
        cmap, label = "viridis", "Interaction propensity"
    elif colour_by == "alphamissense":
        cmap, label = "magma_r", "AlphaMissense"
    else:
        raise ValueError(f"colour_by must be propensity or alphamissense, got {colour_by!r}")

    values = frame[colour_by].to_numpy(float)
    limits = tuple(float(edge) for edge in np.nanpercentile(values, COLOUR_PERCENTILES))

    mappable = ax.scatter(
        frame.stability_z,
        frame.binding_z,
        c=frame[colour_by],
        cmap=cmap,
        norm=Normalize(*limits),
        s=7,
        linewidths=0.2,
        edgecolors="white",
        zorder=3,
    )
    ax.set_xlim(*PLANE_LIMITS)
    ax.set_ylim(*PLANE_LIMITS)
    ax.set_xlabel("Stability (z score)")
    if show_y_label:
        ax.set_ylabel("Binding (z score)")

    bar = ax.figure.colorbar(mappable, cax=cax, orientation="horizontal")
    bar.set_ticks(list(limits))
    bar.set_ticklabels([f"{edge:.2f}" for edge in limits])
    bar.outline.set_linewidth(0.5)
    cax.xaxis.set_ticks_position("top")
    cax.tick_params(length=1.6, width=0.5, pad=1.0)
    cax.set_xlabel(label, labelpad=2.0)
    return {
        "panel": "plane",
        "colour_by": colour_by,
        "percentile_low": COLOUR_PERCENTILES[0],
        "percentile_high": COLOUR_PERCENTILES[1],
        "colour_low": limits[0],
        "colour_high": limits[1],
        "n_coloured": int(np.isfinite(values).sum()),
    }


# --- panel e ---------------------------------------------------------------


def group_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """The shift on each arm between benign variants PyroPPI calls and the rest.

    The two groups are exhaustive within the AlphaMissense-benign class, so no
    variant is set aside. Blocks are residues, so a residue whose substitutions
    all behave alike contributes one observation rather than several.
    """
    tested = frame[frame.benign]
    rows = []
    for column, arm in (("binding_z", "binding"), ("stability_z", "stability")):
        values = tested[column].to_numpy(float)
        groups = tested.pyroppi_hit.to_numpy(bool)
        blocks = tested.residue.to_numpy()
        shift, p_value = stats.block_permutation_test(
            values, groups, blocks, draws=N_PERMUTATIONS, seed=STATS_SEED
        )
        low, high, usable = stats.block_bootstrap_ci(
            values, groups, blocks, draws=N_BOOTSTRAP, seed=STATS_SEED, min_group=MIN_GROUP
        )
        rows.append(
            {
                "panel": "group test",
                "arm": arm,
                "column": column,
                "n_hit": int(groups.sum()),
                "n_nonhit": int((~groups).sum()),
                "n_residues": int(pd.unique(blocks).size),
                "median_hit": float(np.median(values[groups])),
                "median_nonhit": float(np.median(values[~groups])),
                "hodges_lehmann_shift": shift,
                "ci_low": low,
                "ci_high": high,
                "n_bootstrap_usable": usable,
                "permutation_p": p_value,
            }
        )
    return pd.DataFrame(rows)


def draw_group_test(ax: plt.Axes, frame: pd.DataFrame, table: pd.DataFrame) -> None:
    """The two AlphaMissense-benign groups, on each arm separately.

    Splitting the arms is the point of the panel: a group that moves on one arm
    and holds on the other is saying something different from a group that moves
    on both.
    """
    tested = frame[frame.benign]
    rng = np.random.default_rng(STATS_SEED)
    step = 2.6
    groups = (
        (tested[tested.pyroppi_hit], "PyroPPI hit", COLORS["hit"]),
        (tested[~tested.pyroppi_hit], "Non-hit", COLORS["gray"]),
    )
    columns = ["binding_z", "stability_z"]
    bracket = float(tested[columns].to_numpy().max()) + 0.8

    for index, column in enumerate(columns):
        base = index * step
        for offset, (rows, label, colour) in enumerate(groups):
            position = base + offset
            parts = ax.boxplot(
                [rows[column].to_numpy()],
                positions=[position],
                widths=0.6,
                showfliers=False,
                patch_artist=True,
                manage_ticks=False,
                zorder=2,
            )
            parts["boxes"][0].set(facecolor="white", edgecolor=colour, linewidth=0.6)
            for key in ("whiskers", "caps"):
                for line in parts[key]:
                    line.set(color=colour, linewidth=0.6)
            parts["medians"][0].set(color=colour, linewidth=1.2)
            ax.scatter(
                position + rng.normal(0, 0.07, len(rows)),
                rows[column],
                s=3.0,
                color=colour,
                alpha=0.7,
                linewidths=0,
                zorder=3,
            )
            if index == 0:
                ax.text(
                    position,
                    bracket + 1.5,
                    f"{label}\n({len(rows)})",
                    ha="center",
                    va="bottom",
                    color=colour,
                    linespacing=1.2,
                )

        p_value = float(table.loc[table.column.eq(column), "permutation_p"].iloc[0])
        ax.plot(
            [base, base, base + 1, base + 1],
            [bracket - 0.3, bracket, bracket, bracket - 0.3],
            color=COLORS["gray"],
            lw=0.5,
        )
        ax.text(base + 0.5, bracket + 0.15, format_p(p_value), ha="center", va="bottom")

    ax.axhline(0, color=COLORS["grid"], lw=0.5, zorder=0)
    ax.set_xticks([index * step + 0.5 for index in range(len(columns))])
    ax.set_xticklabels(["Binding", "Stability"])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.8, step + 1.8)
    ax.set_ylim(float(tested[columns].to_numpy().min()) - 0.6, bracket + 3.2)
    ax.set_ylabel("Arm enrichment (z score)")
    ax.spines["bottom"].set_visible(False)


# --- panel f ---------------------------------------------------------------


def correlation_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise-complete Spearman over the oriented columns, and the long table."""
    oriented = pd.DataFrame(
        {label: sign * frame[column] for column, sign, label in CORRELATION_SCORES}
    )
    names = list(oriented.columns)
    rho = pd.DataFrame(np.nan, index=names, columns=names)
    rows = []
    # The lower triangle only. The matrix is symmetric and the diagonal is one
    # by construction, so the other half carries no further information.
    for i, first in enumerate(names):
        for j, second in enumerate(names[:i]):
            pair = oriented[[first, second]].dropna()
            result = scipy_stats.spearmanr(pair[first], pair[second])
            rho.iloc[i, j] = rho.iloc[j, i] = result.statistic
            rows.append(
                {
                    "panel": "correlations",
                    "score": first,
                    "reference": second,
                    "spearman_rho": float(result.statistic),
                    "p_value": float(result.pvalue),
                    "n": len(pair),
                }
            )
    for name in names:
        rho.loc[name, name] = 1.0
    return rho, pd.DataFrame(rows)


def draw_correlations(ax: plt.Axes, rho: pd.DataFrame, gene: str, n: int) -> None:
    """The lower triangle of the correlation matrix, every cell carrying its value.

    One triangle is drawn because the matrix is symmetric. The shading is a
    reading aid; the number in each cell is the statistic.
    """
    names = list(rho.index)
    size = len(names)
    values = rho.to_numpy(float)
    extent = float(np.nanmax(np.abs(values[np.tril_indices(size, -1)])))
    cmap = plt.get_cmap("RdBu_r")
    norm = Normalize(-extent, extent)

    for row in range(size):
        for col in range(row):
            value = values[row, col]
            shade = cmap(norm(value))
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1.0,
                    1.0,
                    facecolor=shade,
                    edgecolor="white",
                    lw=0.5,
                    zorder=1,
                )
            )
            ink = "white" if abs(value) > 0.62 * extent else COLORS["ink"]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=ink, zorder=2)

    ax.set_xticks(np.arange(size - 1))
    ax.set_xticklabels(names[:-1], rotation=40, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(1, size))
    ax.set_yticklabels(names[1:])
    ax.tick_params(length=0, pad=1.5)
    ax.set_xlim(-0.6, size - 1.4)
    ax.set_ylim(size - 0.4, 0.4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{gene} missense (n = {n})", loc="left", pad=3.0)


# --- the figure ------------------------------------------------------------


def place(fig: plt.Figure, box: tuple[float, float, float, float]) -> plt.Axes:
    """Axes at a position given in millimetres from the figure's bottom left."""
    width, height = (size * 25.4 for size in fig.get_size_inches())
    x, y, w, h = box
    return fig.add_axes([x / width, y / height, w / width, h / height])


def draw_all(frame: pd.DataFrame, gene: str, letters: tuple[str, ...]) -> tuple:
    """The five data panels for one sublibrary, in the order the figure argues them."""
    variants = sublibrary(frame, gene)
    annotated = missense(variants)

    fig = figure(WIDTH_MM, HEIGHT_MM)
    axes = {name: place(fig, box) for name, box in PANEL_BOXES.items()}

    recovery = draw_recovery(axes["recovery"], variants, gene)
    colour_limits = [
        draw_plane(axes["plane_propensity"], axes["bar_propensity"], annotated, "propensity"),
        draw_plane(
            axes["plane_alphamissense"],
            axes["bar_alphamissense"],
            annotated,
            "alphamissense",
            show_y_label=False,
        ),
    ]
    groups = group_statistics(annotated)
    draw_group_test(axes["group_test"], annotated, groups)
    rho, correlations = correlation_matrix(annotated)
    draw_correlations(axes["correlations"], rho, gene, len(annotated))

    order = ("recovery", "plane_propensity", "plane_alphamissense", "group_test", "correlations")
    for letter, name in zip(letters, order):
        panel_letter(axes[name], letter)

    return fig, {
        "recovery": recovery,
        "colour_limits": pd.DataFrame(colour_limits),
        "groups": groups,
        "correlations": correlations,
        "rho": rho,
        "n_variants": len(variants),
        "n_missense": len(annotated),
    }


def write(fig: plt.Figure, output: Path, stem: str) -> None:
    """Write the figure as a PDF and a PNG, then close it."""
    fig.savefig(output / f"{stem}.pdf")
    save(fig, output / f"{stem}.png")
    print(f"wrote {output / stem}.pdf and {output / stem}.png")


def report(gene: str, results: dict, letters: tuple[str, ...]) -> None:
    """Print every statistic the panels carry."""
    print(f"\n{gene}: {results['n_variants']} variants, "
          f"{results['n_missense']} annotated missense\n")
    print(f"panel {letters[0]}: interaction propensity against the anti-FLAG arm")
    print(results["recovery"].to_string(index=False), end="\n\n")
    print(f"panels {letters[1]} and {letters[2]}: binding against stability for "
          f"{results['n_missense']} variants, z threshold {Z_THRESHOLD}, "
          f"plane limits {PLANE_LIMITS}")
    print(results["colour_limits"].to_string(index=False), end="\n\n")
    print(f"panel {letters[3]}: AlphaMissense-benign variants by PyroPPI call")
    print(results["groups"].to_string(index=False), end="\n\n")
    print(f"panel {letters[4]}: pairwise-complete Spearman correlations")
    print(results["correlations"].to_string(index=False), end="\n\n")


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
    letters = ("b", "c", "d", "e", "f")
    fig, results = draw_all(frame, "XRCC2", letters)
    report("XRCC2", results, letters)
    write(fig, args.output, "figure_5")


if __name__ == "__main__":
    main()
