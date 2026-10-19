#!/usr/bin/env python
"""Figure 2 data panels: the degrader screen and the localization benchmark.

Panel c ranks the 90 zinc-finger domains by the interaction propensity
pomalidomide accounts for, panel d sets that quantity against the depletion
Sievers et al. report, and panel f scores how well each localization tag
separates ORFs whose N terminus faces the cytoplasm. Panels a, b and e of the
manuscript figure are cartoons and are not drawn here.

    python figure_2.py
    python figure_2.py --data /path/to/data --results /path/to/results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold

import pyroppi
from pyroppi.plotting import COLORS, figure, panel_letter, save, use_style
from pyroppi.stats import auroc, benjamini_hochberg

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# --- the zinc-finger degrader screen ---------------------------------------

# Every sample in this screen is treated, so the fit carries no pair baseline
# and the whole of a domain's signal sits in its drug coefficient. The 3 µM arm
# is the one the manuscript reports.
POM_DOSE = "POM_3_UM"
CRBN = "CRBN"
PROPENSITY_LABEL = "POM-dependent\ninteraction propensity"

# The nine domains of the 90 in the library that Sievers, Q. L. et al., Science
# 362, eaat0572 (2018) report as pomalidomide degradation targets. Keys are the
# truncated names the screen's own tables use, values the HGNC symbols.
SIEVERS_TARGETS = {
    "E4F1": "E4F1",
    "IKZF3": "IKZF3",
    "ZN276": "ZNF276",
    "ZN517": "ZNF517",
    "ZN582": "ZNF582",
    "ZN653": "ZNF653",
    "ZN654": "ZNF654",
    "ZN787": "ZNF787",
    "ZN827": "ZNF827",
}
TOP_RANKS = 10

# --- the localization benchmark --------------------------------------------

MEMBRANE_TAGS = ["MEM", "ERM", "LD", "RAB10"]
SOLUBLE_TAGS = ["NES", "ACT", "NLS", "NOLS"]
PANEL_TAGS = MEMBRANE_TAGS + SOLUBLE_TAGS

BOOTSTRAP_DRAWS = 2_000
SEED = 12345
CV_SPLITS = 5
CV_REPEATS = 10


def sigmoid(logit: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(logit, dtype=float)))


def logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def percent_ticks(axis) -> None:
    """Carry the unit on the tick labels, which keeps the axis label short."""
    axis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:g}%"))


# ---------------------------------------------------------------------------
# Panel c: ranked POM-dependent propensity
# ---------------------------------------------------------------------------


def pom_propensity(results: Path) -> pd.DataFrame:
    """Each zinc-finger domain with its propensity at the reported dose, ranked."""
    fit = pyroppi.load_fit(results / "zf_pom")
    effects = fit.effects()
    frame = effects[effects["covariate"] == POM_DOSE].copy()
    partner = np.where(frame["protein_a"] == CRBN, frame["protein_b"], frame["protein_a"])
    frame["domain"] = partner
    frame["propensity"] = sigmoid(frame["logit_mean"]) * 100.0
    frame = frame.sort_values("propensity", ascending=False).reset_index(drop=True)
    frame["rank"] = frame.index + 1
    frame["known"] = frame["domain"].isin(SIEVERS_TARGETS)
    frame["label"] = frame["domain"].map(SIEVERS_TARGETS)
    return frame[["domain", "label", "rank", "propensity", "known", "logit_mean"]]


def stacked_labels(anchors: list[float], gap: float, limit: float) -> list[float]:
    """Label positions at their own anchors, pushed apart only to clear ``gap``.

    Anchors must ascend. Labels keep that order, so no two leader lines cross.
    Even spacing is avoided: most of the labelled domains share the top few
    ranks, and spreading them evenly puts each label far from its own point.
    """
    placed: list[float] = []
    for anchor in anchors:
        placed.append(anchor if not placed else max(anchor, placed[-1] + gap))
    overflow = placed[-1] - limit
    if overflow > 0:
        placed = [position - overflow for position in placed]
        for i in range(len(placed) - 2, -1, -1):
            placed[i] = min(placed[i], placed[i + 1] - gap)
    return placed


def draw_ranked(ax: plt.Axes, frame: pd.DataFrame) -> None:
    """Panel c: every domain ranked, with the known targets named."""
    reach = float(frame["propensity"].max())
    ax.scatter(frame["propensity"], frame["rank"], s=5, color=COLORS["grid"],
               lw=0, zorder=2)
    known = frame[frame["known"]]
    ax.scatter(known["propensity"], known["rank"], s=8, color=COLORS["hit"],
               lw=0, zorder=4)

    ax.set_xlim(0, reach * 1.52)
    ax.set_ylim(len(frame) + 1.5, -2.5)
    ax.set_xlabel(PROPENSITY_LABEL)
    ax.set_ylabel("Rank")
    percent_ticks(ax.xaxis)
    ax.set_xticks([0, 1, 2, 3])
    ticks = [1] + list(range(20, len(frame) + 1, 20))
    ax.set_yticks(ticks)
    ax.tick_params(axis="both", pad=2)

    # One text row is about 1.3 times the type size, converted into rank units
    # through the drawn height of the axes.
    height_pt = ax.get_position().height * ax.figure.get_size_inches()[1] * 72.0
    gap = 1.3 * plt.rcParams["font.size"] / (height_pt / (len(frame) + 4.0))
    ordered = known.sort_values("rank")
    positions = stacked_labels(ordered["rank"].astype(float).tolist(), gap,
                               max(ax.get_ylim()))
    line_x, text_x = reach * 1.12, reach * 1.18
    for (_, row), y in zip(ordered.iterrows(), positions):
        ax.plot([row["propensity"], line_x], [row["rank"], y], color=COLORS["hit"],
                lw=0.4, zorder=3)
        ax.text(text_x, y, row["label"], ha="left", va="center", color=COLORS["hit"])


def target_enrichment(frame: pd.DataFrame, top: int = TOP_RANKS) -> dict:
    """Known targets in the top ``top`` ranks, against the hypergeometric null."""
    n_domains = len(frame)
    n_known = int(frame["known"].sum())
    found = int(frame.loc[frame["rank"] <= top, "known"].sum())
    p = float(stats.hypergeom.sf(found - 1, n_domains, n_known, top))
    return {"n_domains": n_domains, "n_known": n_known, "top": top,
            "found": found, "p": p}


# ---------------------------------------------------------------------------
# Panel d: propensity against the published depletion
# ---------------------------------------------------------------------------


def draw_benchmark(ax: plt.Axes, frame: pd.DataFrame, data: Path) -> dict:
    """Panel d: propensity against the depletion Sievers et al. report."""
    bench = pd.read_csv(data / "annotations" / "degrader_benchmarks.csv")
    merged = frame.merge(bench, left_on="domain", right_on="gene", how="left")
    paired = merged.dropna(subset=["sievers_pom_fold_depletion"])

    ax.scatter(paired["propensity"], paired["sievers_pom_fold_depletion"], s=5,
               color=COLORS["gray"], lw=0, zorder=3)
    rho, rho_p = stats.spearmanr(paired["propensity"],
                                 paired["sievers_pom_fold_depletion"])
    r, r_p = stats.pearsonr(paired["propensity"],
                            paired["sievers_pom_fold_depletion"])
    ax.text(0.97, 0.05, f"rho = {rho:.2f} (P = {rho_p:.2g})\nr = {r:.2f} (P = {r_p:.2g})",
            transform=ax.transAxes, ha="right", va="bottom", linespacing=1.4)

    # Cut to the drawn points on both axes. Neither zero is a reference the
    # panel is read against, and holding them spends the corner the annotation
    # sits in on blank paper.
    for setter, values in ((ax.set_xlim, paired["propensity"]),
                           (ax.set_ylim, paired["sievers_pom_fold_depletion"])):
        low, high = float(values.min()), float(values.max())
        setter(low - 0.06 * (high - low), high + 0.06 * (high - low))
    ax.set_xlabel(PROPENSITY_LABEL)
    ax.set_ylabel("Sievers et al. ZF-GFP\nfold depletion")
    percent_ticks(ax.xaxis)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.tick_params(axis="both", pad=2)
    return {"n": len(paired), "rho": float(rho), "rho_p": float(rho_p),
            "r": float(r), "r_p": float(r_p)}


# ---------------------------------------------------------------------------
# Panel f: separating cytoplasm-facing ORFs
# ---------------------------------------------------------------------------


def tag_readouts(results: Path, data: Path) -> pd.DataFrame:
    """One row per tag and ORF: the 3 h drug readout and the compartment class.

    The readout is the drug contribution taken against the headroom above the
    pair's own baseline, so a tag that edits heavily at baseline and one that
    does not are put on a single scale. Every 3 h sample of a tag carries the
    same design row, so both terms are constant over those samples and the
    averages below only pool replicates.
    """
    fit = pyroppi.load_fit(results / "localization")
    samples = fit.samples
    treated = set(samples.loc[samples["3H"] == 1, "sample"])
    parts = fit.contributions()
    parts = parts[parts["sample"].isin(treated)]
    frame = parts.groupby(["bait", "prey"]).agg(
        contrib_drug=("contrib_drug", "mean"),
        contrib_pair=("contrib_pair", "mean"),
    ).reset_index()
    frame["readout"] = frame["contrib_drug"] / (1.0 - frame["contrib_pair"])

    labels = pd.read_csv(data / "annotations" / "localization_labels.csv")
    frame = frame.merge(labels[["orf", "orf_class"]], left_on="prey",
                        right_on="orf", how="left")
    frame = frame[frame["orf_class"].isin(["CY", "NM"])]
    return frame.rename(columns={"bait": "tag", "prey": "orf_name"})


def bootstrap_interval(positive: np.ndarray, negative: np.ndarray,
                       rng: np.random.Generator, draws: int = BOOTSTRAP_DRAWS):
    """Percentile interval on the area, resampling inside each class.

    A resample's area is ``w_pos @ C @ w_neg`` over ``n_pos * n_neg``, where C
    compares every positive against every negative once and the two weight
    vectors count how often each observation was drawn. Working through C keeps
    every draw to one matrix product.
    """
    n_pos, n_neg = positive.size, negative.size
    difference = positive[:, None] - negative[None, :]
    comparison = (difference > 0).astype(float) + 0.5 * (difference == 0)
    rows = np.arange(draws)[:, None]
    weights = []
    for size, source in ((n_pos, positive), (n_neg, negative)):
        counts = np.zeros((draws, size))
        np.add.at(counts, (rows, rng.integers(0, size, size=(draws, size))), 1.0)
        weights.append(counts)
    areas = ((weights[0] @ comparison) * weights[1]).sum(axis=1) / (n_pos * n_neg)
    return float(np.percentile(areas, 2.5)), float(np.percentile(areas, 97.5))


def tag_areas(frame: pd.DataFrame) -> pd.DataFrame:
    """Area under the curve of each tag, with its interval and adjusted p."""
    rng = np.random.default_rng(SEED)
    rows = []
    for tag in PANEL_TAGS:
        sub = frame[frame["tag"] == tag]
        positive = sub.loc[sub["orf_class"] == "CY", "readout"].dropna().to_numpy()
        negative = sub.loc[sub["orf_class"] == "NM", "readout"].dropna().to_numpy()
        area, p = auroc(positive, negative)
        low, high = bootstrap_interval(positive, negative, rng)
        rows.append({"tag": tag, "auroc": area, "ci_low": low, "ci_high": high,
                     "p": p, "n_cy": positive.size, "n_nm": negative.size})
    table = pd.DataFrame(rows)
    table["q"] = benjamini_hochberg(table["p"].to_numpy())
    return table


def combined_score(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold class probability from an ORF's whole vector of tag readouts.

    Features are the eight readouts on the logit scale, taken as measured. The
    score is pooled over the held-out folds and averaged over the repeats, so it
    is comparable with the single-tag readouts, which are not fitted at all.
    """
    wide = frame.pivot(index="orf_name", columns="tag", values="readout")
    wide = wide.reindex(columns=PANEL_TAGS).dropna()
    classes = frame.drop_duplicates("orf_name").set_index("orf_name")["orf_class"]
    y = (classes.reindex(wide.index) == "CY").to_numpy().astype(int)
    x = logit(wide.to_numpy())

    splitter = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS,
                                       random_state=SEED)
    out_of_fold = np.zeros((CV_REPEATS, y.size))
    for index, (train, test) in enumerate(splitter.split(x, y)):
        model = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=400, early_stopping=True,
            validation_fraction=0.15, random_state=SEED,
        )
        model.fit(x[train], y[train])
        out_of_fold[index // CV_SPLITS, test] = model.predict_proba(x[test])[:, 1]
    return y, out_of_fold.mean(axis=0)


def shades(base: str, count: int) -> list[tuple[float, float, float]]:
    """Tints of one colour, palest first, so a class of tags reads as a family."""
    rgb = np.array(mcolors.to_rgb(base))
    return [tuple(rgb + (1.0 - rgb) * weight)
            for weight in np.linspace(0.45, 0.0, count)]


def draw_roc(ax: plt.Axes, frame: pd.DataFrame, areas: pd.DataFrame,
             y: np.ndarray, score: np.ndarray) -> float:
    """Panel f: one curve per tag, one for the eight read together, and chance."""
    palette = dict(zip(MEMBRANE_TAGS, shades(COLORS["hit"], len(MEMBRANE_TAGS))))
    palette.update(zip(SOLUBLE_TAGS, shades(COLORS["rad51d"], len(SOLUBLE_TAGS))))
    style = {tag: "-" for tag in MEMBRANE_TAGS}
    style.update({tag: (0, (3, 1.6)) for tag in SOLUBLE_TAGS})

    lookup = areas.set_index("tag")
    for tag in PANEL_TAGS:
        sub = frame[(frame["tag"] == tag) & frame["readout"].notna()]
        fpr, tpr, _ = roc_curve((sub["orf_class"] == "CY").astype(int), sub["readout"])
        ax.plot(fpr, tpr, color=palette[tag], ls=style[tag], lw=0.8,
                label=f"{tag} {lookup.loc[tag, 'auroc']:.2f}")

    combined = float(roc_auc_score(y, score))
    fpr, tpr, _ = roc_curve(y, score)
    ax.plot(fpr, tpr, color=COLORS["ink"], lw=1.3, zorder=4,
            label=f"8-tag model {combined:.2f}")
    ax.plot([0, 1], [0, 1], color=COLORS["gray"], lw=0.5, ls=":",
            label="chance 0.50")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.005)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.tick_params(axis="both", pad=2)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.17), ncol=3,
              handlelength=1.3, columnspacing=0.8, labelspacing=0.25,
              handletextpad=0.4, borderpad=0.0)
    return combined


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data",
                        help="directory holding the dataset and annotation tables")
    parser.add_argument("--results", type=Path, default=REPO / "results",
                        help="directory holding the fits")
    parser.add_argument("--output", type=Path, default=REPO / "figures",
                        help="directory to write the figure into")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    use_style()
    width, height = 183.0, 90.0
    fig = figure(width, height)

    def place(left: float, bottom: float, box_width: float, box_height: float):
        return fig.add_axes([left / width, bottom / height,
                             box_width / width, box_height / height])

    ax_c = place(14, 14, 36, 64)
    ax_d = place(66, 14, 46, 64)
    ax_f = place(131, 32, 46, 46)

    ranked = pom_propensity(args.results)
    draw_ranked(ax_c, ranked)
    enrichment = target_enrichment(ranked)
    benchmark = draw_benchmark(ax_d, ranked, args.data)

    readouts = tag_readouts(args.results, args.data)
    areas = tag_areas(readouts)
    y, score = combined_score(readouts)
    combined = draw_roc(ax_f, readouts, areas, y, score)

    for ax, letter in ((ax_c, "c"), (ax_d, "d"), (ax_f, "f")):
        panel_letter(ax, letter)
    fig.savefig(args.output / "figure_2.png")
    save(fig, args.output / "figure_2.pdf")

    print(f"panel c: {len(ranked)} zinc-finger domains at {POM_DOSE}, "
          "POM-dependent interaction propensity in percent")
    print(ranked.loc[ranked["known"], ["label", "rank", "propensity"]]
          .sort_values("rank").to_string(index=False))
    print(f"  {enrichment['found']} of the {enrichment['n_known']} known targets "
          f"rank in the top {enrichment['top']} of {enrichment['n_domains']}, "
          f"hypergeometric P = {enrichment['p']:.3g}")
    print()

    print(f"panel d: {benchmark['n']} domains with a Sievers et al. measurement")
    print(f"  Spearman rho = {benchmark['rho']:.3f}, P = {benchmark['rho_p']:.3g}")
    print(f"  Pearson r = {benchmark['r']:.3f}, P = {benchmark['r_p']:.3g}")
    print()

    print("panel f: separating cytoplasm-facing ORFs from non-membrane ORFs")
    print(areas.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"  8-tag model: area {combined:.4f} on {y.size} ORFs "
          f"complete on all eight tags, {int(y.sum())} of them CY")


if __name__ == "__main__":
    main()
