"""Fit PyroPPI to a deep mutational scan and call the variants.

Runs the analysis of ``deep_mutational_scanning.ipynb`` end to end and writes
the figures beside this file as PNGs. RAD51D is the worked example and XRCC2
the comparison; each is scanned against the other, which is the cognate pair.
The RAD51D fit is run here, and the XRCC2 fit is read from the results
directory that ``scripts/fit_all.py`` writes.

    python deep_mutational_scanning.py
    python deep_mutational_scanning.py --data /path/to/data --steps 10000
    python deep_mutational_scanning.py --results /path/to/results
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

os.environ.setdefault("TQDM_DISABLE", "1")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from pyroppi import fit, load_fit, read_dataset
from pyroppi import stats as ps
from pyroppi.data import apply_read_floor
from pyroppi.plotting import COLORS

# The manuscript palette. Green, amber and grey name the three variant classes,
# vermilion marks a called variant, blue separates the two scans and pink marks
# a Walker motif.
SYN = COLORS["synonymous"]
STOP = COLORS["stop_gain"]
MIS = COLORS["missense"]
HIT = COLORS["hit"]
BLUE = "#0072B2"
WALKER = COLORS["walker_motif"]
CLASS_COLOR = {"synonymous": SYN, "missense": MIS, "stop": STOP}
ORDER = ["synonymous", "missense", "stop"]

# Each scan is a variant library on MCP against wild-type baits. Keeping the
# cognate partner alone leaves one measurement per variant.
SCANS = [
    {"gene": "RAD51D", "dataset": "dms_rad51d", "partner": "XRCC2"},
    {"gene": "XRCC2", "dataset": "dms_xrcc2", "partner": "RAD51D"},
]

# The P-loop and the Walker B motif of each RecA fold, in the numbering the
# library uses.
MOTIFS = {
    "RAD51D": {"Walker A": (107, 114), "Walker B": (200, 210)},
    "XRCC2": {"Walker A": (48, 55), "Walker B": (145, 149)},
}

DRUG = ["3h", "U2OS_x_3h"]
CELL = ["U2OS"]
MIN_READS = 20
ALPHA = 0.05
DEPTH_LABEL_AT = 3.45  # where the two depth histograms are well separated
DRAWS = 20_000
SEED = 0

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

mpl.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def variant_table(result, partner: str) -> pd.DataFrame:
    """Per-variant interaction propensity against one partner.

    ``Fit.pairs`` names the two ends of a canonical pair ``protein_a`` and
    ``protein_b`` in alphabetical order, so the variant can sit at either end.
    """
    pairs = result.pairs()
    variant_is_a = pairs["protein_a"].str.contains(":")
    pairs["variant"] = np.where(variant_is_a, pairs["protein_a"], pairs["protein_b"])
    pairs["against"] = np.where(variant_is_a, pairs["protein_b"], pairs["protein_a"])
    pairs = pairs[pairs["against"] == partner].reset_index(drop=True)
    return pd.concat([pairs, ps.parse_variants(pairs["variant"])], axis=1)


def box(ax, position, values, color, width=0.55):
    """One box at ``position`` with a coloured median and a matching face."""
    artists = ax.boxplot(
        [values],
        positions=[position],
        widths=width,
        showfliers=False,
        patch_artist=True,
    )
    artists["boxes"][0].set(facecolor=color, alpha=0.35, edgecolor=color, linewidth=1.0)
    for part in ("whiskers", "caps"):
        for line in artists[part]:
            line.set(color=color, linewidth=1.0)
    artists["medians"][0].set(color=color, linewidth=1.8)


def count_labels(ax, counts, colors):
    """Write the group sizes in a strip of blank axes below the boxes."""
    low, high = ax.get_ylim()
    ax.set_ylim(low - 0.13 * (high - low), high)
    for position, (n, color) in enumerate(zip(counts, colors)):
        ax.text(
            position,
            0.015,
            f"n={n:,}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=color,
        )


def distance_profile(missense: pd.DataFrame) -> pd.DataFrame:
    """One row per resolved residue: hit slots, assayed slots and distance."""
    resolved = missense[missense["min_dist_to_partner_A"].notna()]
    return (
        resolved.groupby("position")
        .agg(
            n_hit=("hit", "sum"),
            n_assayed=("hit", "size"),
            distance=("min_dist_to_partner_A", "first"),
        )
        .reset_index()
        .sort_values("position")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO / "data",
        help="directory holding the dataset directories",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO / "results",
        help="directory holding the fits written by scripts/fit_all.py",
    )
    parser.add_argument("--steps", type=int, default=60_000, help="number of SVI steps")
    args = parser.parse_args()
    print(f"data: {args.data}")
    print(f"results: {args.results}")

    # --- 1. the library and its read depth ---------------------------------
    scans = {}
    for scan in SCANS:
        scan["counts"], scan["samples"] = read_dataset(args.data / scan["dataset"])
        counts = scan["counts"]
        library = ps.parse_variants(counts["prey"].unique())
        scan["library"] = library
        missense = library[library["class"] == "missense"]
        print(
            f"{scan['gene']}: {len(counts):,} rows, {len(scan['samples'])} samples, "
            f"{counts['prey'].nunique():,} constructs over "
            f"{missense['position'].nunique()} substituted positions "
            f"({missense['position'].min()} to {missense['position'].max()}), "
            f"median depth {counts['total'].median():,.0f} reads"
        )
        print(
            "    "
            + ", ".join(
                f"{n:,} {name}" for name, n in library["class"].value_counts().items()
            )
            + f", baits {', '.join(sorted(counts['bait'].unique()))}"
        )
        scans[scan["gene"]] = scan

    figure, axes = plt.subplot_mosaic(
        [["RAD51D", "XRCC2"], ["depth", "depth"]],
        figsize=(7.0, 4.2),
        constrained_layout=True,
    )
    for gene, scan in scans.items():
        library = scan["library"]
        per_position = library[library["class"] == "missense"].groupby("position").size()
        axes[gene].bar(per_position.index, per_position.to_numpy(), width=1.0, color=MIS)
        axes[gene].set(
            xlabel=f"{gene} residue",
            ylabel="substitutions assayed",
            title=f"{gene}: {len(per_position)} positions reached",
        )
    # Shared bins, so the two histograms can be read against each other, and a
    # label placed on each curve where the two are well separated.
    depth = {gene: np.log10(scan["counts"]["total"]) for gene, scan in scans.items()}
    bins = np.linspace(
        min(values.min() for values in depth.values()),
        max(values.max() for values in depth.values()),
        71,
    )
    for gene, values in depth.items():
        colour = BLUE if gene == "RAD51D" else MIS
        heights, edges = np.histogram(values, bins=bins)
        axes["depth"].step(edges[:-1], heights, where="post", color=colour, linewidth=1.2)
        at = min(np.searchsorted(edges, DEPTH_LABEL_AT) - 1, len(heights) - 1)
        axes["depth"].text(
            DEPTH_LABEL_AT,
            heights[at],
            f" {gene}",
            color=colour,
            va="bottom",
            fontsize=7.5,
        )
    axes["depth"].axvline(np.log10(MIN_READS), color=HIT, linestyle="--", linewidth=1.2)
    axes["depth"].text(
        np.log10(MIN_READS),
        axes["depth"].get_ylim()[1] * 0.96,
        f" {MIN_READS}-read floor",
        color=HIT,
        va="top",
        fontsize=7.5,
    )
    axes["depth"].set(
        xlabel="total reporter reads (log$_{10}$)",
        ylabel="observations",
        title="read depth per observation",
    )
    figure.savefig(HERE / "library_and_depth.png")
    plt.close(figure)

    # --- 2. the read floor -------------------------------------------------
    floors = np.arange(0, 205, 5)
    figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), constrained_layout=True)
    for gene, scan in scans.items():
        counts = scan["counts"]
        measured = counts.groupby(["bait", "prey"]).ngroups
        kept = apply_read_floor(counts, MIN_READS)
        scan["kept"] = kept
        censored = int((~kept["used"]).sum())
        dropped = measured - kept.groupby(["bait", "prey"]).ngroups
        touched = int(
            kept.groupby(["bait", "prey"])["used"].apply(lambda used: (~used).any()).sum()
        )
        print(
            f"{gene}: {int((counts['total'] < MIN_READS).sum()):,} of {len(counts):,} "
            f"observations sit below the {MIN_READS}-read floor, in "
            f"{counts.loc[counts['total'] < MIN_READS, 'prey'].nunique()} of "
            f"{counts['prey'].nunique():,} constructs; {censored:,} of them are censored "
            f"inside {touched} measured pairs that keep other replicates, and {dropped} "
            f"measured pairs of {measured:,} lose every observation and are dropped"
        )
        colour = BLUE if gene == "RAD51D" else MIS
        # A pair is dropped entirely when its deepest observation is below the
        # floor, which is the rule ``apply_read_floor`` applies pair by pair.
        deepest = counts.groupby(["bait", "prey"])["total"].max().to_numpy()
        series = [
            (counts["total"].to_numpy()[None, :] < floors[:, None]).sum(axis=1),
            (deepest[None, :] < floors[:, None]).sum(axis=1),
        ]
        for axis, values in zip(axes, series):
            axis.plot(floors, values, color=colour, linewidth=1.3)
            axis.text(
                floors[-1],
                values[-1],
                f" {gene}",
                color=colour,
                va="center",
                fontsize=7.5,
            )
    for axis, label in zip(
        axes, ["observations below the floor", "measured pairs dropped entirely"]
    ):
        axis.axvline(MIN_READS, color=HIT, linestyle="--", linewidth=1.2)
        axis.set(xlabel="read floor", ylabel=label, xlim=(0, 232))
    axes[0].text(
        MIN_READS,
        axes[0].get_ylim()[1],
        f" {MIN_READS} reads",
        color=HIT,
        va="top",
        fontsize=7.5,
    )
    figure.savefig(HERE / "read_floor.png")
    plt.close(figure)

    # --- 3. the fit and its ELBO -------------------------------------------
    worked = scans["RAD51D"]
    start = time.perf_counter()
    worked["fit"] = fit(
        worked["counts"],
        worked["samples"],
        drug=DRUG,
        cell=CELL,
        baseline=True,
        min_reads=MIN_READS,
        num_steps=args.steps,
        seed=SEED,
    )
    worked["seconds"] = time.perf_counter() - start
    print(
        f"RAD51D: {args.steps:,} steps in {worked['seconds']:.0f} s, "
        f"final ELBO {worked['fit'].losses[-1]:,.0f}, "
        f"{len(worked['fit'].canonical):,} canonical pairs"
    )

    comparison = scans["XRCC2"]
    comparison["fit"] = load_fit(args.results / comparison["dataset"])
    print(
        f"XRCC2: read from {args.results / comparison['dataset']}, "
        f"{len(comparison['fit'].losses):,} steps, "
        f"final ELBO {comparison['fit'].losses[-1]:,.0f}, "
        f"drug={comparison['fit'].drug}, cell={comparison['fit'].cell}, "
        f"min_reads={comparison['fit'].min_reads}"
    )

    effects = worked["fit"].effects()
    summary = effects.groupby("covariate")["logit_mean"].agg(["mean", "min", "max"])
    print("RAD51D covariate shifts in log-odds, over the canonical pairs")
    print(summary.loc[DRUG + CELL].to_string())

    figure, axes = plt.subplots(2, 2, figsize=(7.0, 4.2), constrained_layout=True)
    for column, (gene, scan) in enumerate(scans.items()):
        losses = scan["fit"].losses
        step = np.arange(1, len(losses) + 1)
        tail = min(10_000, len(losses))
        axes[0, column].plot(step, losses, color=BLUE, linewidth=0.8)
        axes[0, column].set(
            xlabel="SVI step",
            ylabel="ELBO loss",
            yscale="log",
            title=f"{gene}: {len(losses):,} steps",
        )
        axes[1, column].plot(
            step[-tail:], losses[-tail:], color=BLUE, linewidth=0.3, alpha=0.5
        )
        smoothed = pd.Series(losses[-tail:]).rolling(500, center=True).mean()
        axes[1, column].plot(
            step[-tail:], smoothed, color=HIT, linewidth=1.4, label="500-step mean"
        )
        axes[1, column].set(
            xlabel="SVI step",
            ylabel="ELBO loss",
            title=f"{gene}: the last {tail:,} steps",
        )
        axes[1, column].legend(loc="lower left", fontsize=7)
    figure.savefig(HERE / "elbo.png")
    plt.close(figure)

    # --- 4. the variant table ----------------------------------------------
    for gene, scan in scans.items():
        scan["variants"] = variant_table(scan["fit"], scan["partner"])
        counted = scan["variants"]["class"].value_counts()
        print(
            f"{gene}: {len(scan['fit'].canonical):,} canonical pairs, "
            f"{len(scan['variants']):,} of them against {scan['partner']}: "
            + ", ".join(f"{n:,} {name}" for name, n in counted.items())
        )

    # --- 5. class separation ------------------------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), constrained_layout=True)
    for axis, (gene, scan) in zip(axes, scans.items()):
        table = scan["variants"]
        sizes = []
        for position, name in enumerate(ORDER):
            values = table.loc[table["class"] == name, "propensity_mean"]
            box(axis, position, values, CLASS_COLOR[name])
            sizes.append(len(values))
        count_labels(axis, sizes, [CLASS_COLOR[name] for name in ORDER])
        axis.set_xticks(range(len(ORDER)), ["synonymous", "missense", "stop-gain"])
        axis.set(
            ylabel=f"interaction propensity with {scan['partner']}",
            title=f"{gene} variants",
        )
    figure.savefig(HERE / "class_separation.png")
    plt.close(figure)

    for gene, scan in scans.items():
        medians = scan["variants"].groupby("class")["propensity_mean"].median()
        contrast = sstats.mannwhitneyu(
            scan["variants"].loc[scan["variants"]["class"] == "stop", "logit_mean"],
            scan["variants"].loc[scan["variants"]["class"] == "synonymous", "logit_mean"],
            alternative="less",
        )
        print(
            f"{gene} median propensity: "
            + ", ".join(f"{name} {medians[name]:.4f}" for name in ORDER)
            + f"; stop-gain below synonymous, Mann-Whitney p = {contrast.pvalue:.2g}"
        )

    # --- 6. hit calling -----------------------------------------------------
    threshold = sstats.norm.ppf(ALPHA)
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
    for axis, (gene, scan) in zip(axes, scans.items()):
        tested = ps.synonymous_test(
            scan["variants"], score="logit_mean", by="gene", alpha=ALPHA
        )
        scan["tested"] = tested
        missense = tested["class"] == "missense"
        stop = tested["class"] == "stop"
        synonymous = tested["class"] == "synonymous"
        scan["n_hit"] = int(tested.loc[missense, "hit"].sum())
        scan["n_missense"] = int(missense.sum())
        bh = ps.benjamini_hochberg(tested.loc[missense, "p"])
        print(
            f"{gene}: {scan['n_hit']} of {scan['n_missense']:,} missense called "
            f"({scan['n_hit'] / scan['n_missense']:.1%}), "
            f"{int(tested.loc[stop, 'hit'].sum())} of {int(stop.sum())} stop-gain, "
            f"{(tested.loc[synonymous, 'p'] < ALPHA).mean():.1%} of the "
            f"{int(synonymous.sum())}-variant synonymous null below p = {ALPHA}, "
            f"{int((bh < ALPHA).sum())} missense surviving Benjamini-Hochberg"
        )

        bins = np.linspace(tested["z"].min(), tested["z"].max(), 60)
        for name in ORDER:
            axis.hist(
                tested.loc[tested["class"] == name, "z"],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.3,
                color=CLASS_COLOR[name],
                label=name.replace("stop", "stop-gain"),
            )
        axis.axvline(threshold, color=HIT, linestyle="--", linewidth=1.2)
        axis.set(
            xlabel="z against the synonymous pool",
            ylabel="density",
            title=f"{gene}: {scan['n_hit']} of {scan['n_missense']:,} missense called",
        )
        axis.legend(loc="upper left", fontsize=7)
    figure.savefig(HERE / "hit_calling.png")
    plt.close(figure)

    # --- 7. the position track and the Walker motifs -----------------------
    annotations = pd.read_csv(args.data / "annotations" / "variants.csv")
    annotation_columns = [
        "variant",
        "alphamissense",
        "alphamissense_class",
        "popEVE",
        "min_dist_to_partner_A",
        "interface_8A",
        "conservation",
    ]
    for gene, scan in scans.items():
        tested = scan["tested"]
        scan["missense"] = tested[tested["class"] == "missense"].merge(
            annotations[annotation_columns], on="variant", how="left"
        )

    motif_rows = []
    for gene, scan in scans.items():
        missense = scan["missense"]
        for label, (low, high) in MOTIFS[gene].items():
            inside = missense["position"].between(low, high)
            result = sstats.mannwhitneyu(
                missense.loc[inside, "logit_mean"],
                missense.loc[~inside, "logit_mean"],
                alternative="less",
            )
            motif_rows.append(
                {
                    "gene": gene,
                    "motif": label,
                    "residues": f"{low}-{high}",
                    "n_variants": int(inside.sum()),
                    "n_hit": int(missense.loc[inside, "hit"].sum()),
                    "motif_median": missense.loc[inside, "propensity_mean"].median(),
                    "background_median": missense.loc[~inside, "propensity_mean"].median(),
                    "p": result.pvalue,
                }
            )
    motifs = pd.DataFrame(motif_rows)
    print("Walker motifs against the rest of the chain, one-sided Mann-Whitney U")
    print(motifs.to_string(index=False))

    figure, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), constrained_layout=True)
    for axis, (gene, scan) in zip(axes, scans.items()):
        tested = scan["tested"]
        track = scan["missense"].groupby("position").agg(
            propensity=("propensity_mean", "mean"), n_hit=("hit", "sum")
        )
        reference = tested.groupby("class")["propensity_mean"].median()

        axis.plot(track.index, track["propensity"], color=MIS, linewidth=0.9)
        called = track[track["n_hit"] > 0]
        axis.scatter(
            called.index,
            called["propensity"],
            s=10,
            color=HIT,
            zorder=3,
            label="position with a called variant",
        )
        axis.axhline(reference["synonymous"], color=SYN, linestyle="--", linewidth=1.1)
        axis.axhline(reference["stop"], color=STOP, linestyle="--", linewidth=1.1)
        low, high = axis.get_ylim()
        axis.set_ylim(low - 0.22 * (high - low), high + 0.10 * (high - low))

        for label, (first, last) in MOTIFS[gene].items():
            axis.axvspan(first - 0.5, last + 0.5, color=WALKER, alpha=0.22, linewidth=0)
            p = motifs.loc[
                (motifs["gene"] == gene) & (motifs["motif"] == label), "p"
            ].iloc[0]
            axis.text(
                (first + last) / 2,
                0.985,
                f"{label}\np = {p:.1e}",
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=6.5,
                color=WALKER,
            )
        axis.set(
            xlabel=f"{gene} residue",
            ylabel="mean missense propensity",
            title=f"{gene} against {scan['partner']}",
        )
        axis.title.set(ha="left", position=(0.0, 1.0))
        halo = dict(facecolor="white", edgecolor="none", pad=0.6)
        for level, label, colour, side in [
            (reference["synonymous"], "synonymous", SYN, "bottom"),
            (reference["stop"], "stop-gain", STOP, "top"),
        ]:
            axis.text(
                0.995,
                level,
                label,
                color=colour,
                fontsize=6.5,
                transform=axis.get_yaxis_transform(),
                ha="right",
                va=side,
                bbox=halo,
            )
    axes[0].legend(loc="lower left", fontsize=7)
    figure.savefig(HERE / "position_track.png")
    plt.close(figure)

    # --- 8. distance to the partner chain -----------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), constrained_layout=True)
    for axis, (gene, scan) in zip(axes, scans.items()):
        profile = distance_profile(scan["missense"])
        distance = profile["distance"].to_numpy(float)
        n_hit = profile["n_hit"].to_numpy(int)
        n_assayed = profile["n_assayed"].to_numpy(int)

        def gap(hit_slots, assayed_slots):
            other = assayed_slots - hit_slots
            return float(
                other @ distance / other.sum() - hit_slots @ distance / hit_slots.sum()
            )

        observed, p_value = ps.run_replacement_test(
            np.full(len(profile), gene),
            n_hit,
            n_assayed,
            gap,
            draws=DRAWS,
            seed=SEED,
        )
        other = n_assayed - n_hit
        hit_mean = float(n_hit @ distance / n_hit.sum())
        other_mean = float(other @ distance / other.sum())
        print(
            f"{gene}: over {len(profile)} resolved residues carrying {n_hit.sum()} hit "
            f"slots and {other.sum():,} non-hit slots, the called variants sit "
            f"{hit_mean:.2f} Å from {scan['partner']} and the rest {other_mean:.2f} Å, "
            f"a gap of {observed:+.2f} Å at run-replacement p = {p_value:.4f}"
        )

        resolved = scan["missense"][scan["missense"]["min_dist_to_partner_A"].notna()]
        sizes = []
        for position, (subset, colour) in enumerate(
            [(resolved[resolved["hit"]], HIT), (resolved[~resolved["hit"]], MIS)]
        ):
            values = subset["min_dist_to_partner_A"].to_numpy()
            box(axis, position, values, colour)
            axis.scatter(
                position,
                values.mean(),
                marker="D",
                s=16,
                color=colour,
                edgecolors="white",
                linewidths=0.6,
                zorder=4,
            )
            sizes.append(len(values))
        count_labels(axis, sizes, [HIT, MIS])
        axis.set_xticks([0, 1], ["called", "not called"])
        axis.set(
            ylabel=f"min. distance to {scan['partner']} (Å)",
            title=f"{gene}: gap {observed:+.2f} Å, p = {p_value:.4f}",
        )
    figure.savefig(HERE / "interface_distance.png")
    plt.close(figure)

    print(f"figures written to {HERE}")


if __name__ == "__main__":
    main()
