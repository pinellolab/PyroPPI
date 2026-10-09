"""Supplementary Figure 2. Checks on the PyroPPI model.

This is the one figure script that fits the model, because its three panels test
the model itself rather than an experiment. All three run on the nine-protein
RAD51 paralog matrix, the smallest dataset that carries both covariate blocks
and fits in about a minute.

    a  the observed editing rate against the prior and posterior predictive
    b  held-out prediction under a five-fold split of the observations
    c  the same held-out score with each covariate block removed

    python scripts/supp_figure_2.py
    python scripts/supp_figure_2.py --steps 5000 --seeds 1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import pearsonr

import pyroppi
from pyroppi.data import apply_read_floor
from pyroppi.model import (
    CELL_SIGMA_SCALE,
    DRUG_SIGMA_SCALE,
    PAIR_LOC,
    PAIR_SCALE,
    PAIR_SIGMA_SCALE,
)
from pyroppi.plotting import COLORS, figure, panel_letter, save, use_style

REPOSITORY = Path(__file__).resolve().parent.parent

DATASET = "rad51x9"
DRUG = ["1h", "3h", "HCT_x_1h", "HCT_x_3h"]
CELL = ["HCT"]
MIN_READS = 20

N_FOLDS = 5
N_DRAWS = 1_000
CV_DRAWS = 200
FOLD_SEED = 7
PREDICTIVE_SEED = 11
# The ablation is scored on the first of the same five folds, which holds the
# default run of seventeen fits to about a quarter of an hour.
ABLATION_FOLD = 0

# The four cells of the ablation, ordered by what each can express.
ABLATIONS = [
    ("Pair only", [], []),
    ("Pair and cell", [], CELL),
    ("Pair and drug", DRUG, []),
    ("Full model", DRUG, CELL),
]

FOLD_COLORS = [
    COLORS["popeve"],
    COLORS["hit"],
    COLORS["synonymous"],
    COLORS["stop_gain"],
    COLORS["walker_motif"],
]
PRIOR_COLOR = COLORS["gray"]
POSTERIOR_COLOR = COLORS["hit"]
OBSERVED_COLOR = COLORS["ink"]

GRID_POINTS = 400
NOMINAL = 0.90


# --- prediction ------------------------------------------------------------


def index_rows(fit, counts: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical row and the two design rows of each observation.

    Keyed by protein and sample name rather than by position, so a fit made on
    part of the observations can score rows it never saw.
    """
    lookup = {
        (row.protein_a, row.protein_b): index
        for index, row in enumerate(fit.canonical.itertuples())
    }
    ends = np.sort(counts[["bait", "prey"]].to_numpy(dtype=str), axis=1)
    canonical = np.array([lookup[(a, b)] for a, b in ends])

    sample_row = {name: i for i, name in enumerate(fit.samples["sample"])}
    sample = counts["sample"].map(sample_row).to_numpy(dtype=int)
    drug = fit.samples[fit.drug].to_numpy(dtype=float)[sample]
    cell = fit.samples[fit.cell].to_numpy(dtype=float)[sample]
    return canonical, drug, cell


def mu_draws(fit, counts: pd.DataFrame) -> np.ndarray:
    """Editing probability of each observation under each posterior draw, (D, N).

    Each block contributes the logits of the indicator columns the observation's
    sample switches on, taken at its canonical pair, and the three blocks are
    summed on the log-odds scale.
    """
    canonical, drug_design, cell_design = index_rows(fit, counts)
    pair_eta = fit.draws["pair_logit"][:, canonical]
    drug_eta = np.einsum("nj,djn->dn", drug_design, fit.draws["drug_logit"][:, :, canonical])
    cell_eta = np.einsum("nj,djn->dn", cell_design, fit.draws["cell_logit"][:, :, canonical])
    return expit(pair_eta + drug_eta + cell_eta)


def predicted_rate(fit, counts: pd.DataFrame) -> np.ndarray:
    """Posterior mean editing rate of each observation."""
    return mu_draws(fit, counts).mean(axis=0)


def prior_mu_draws(fit, counts: pd.DataFrame, n_draws: int, seed: int) -> np.ndarray:
    """Editing probability under draws from the priors, (D, N).

    The same three blocks as the fitted model, with every latent taken from the
    prior in ``pyroppi.model`` rather than from the guide. A half normal is
    drawn as the absolute value of a normal.
    """
    rng = np.random.default_rng(seed)
    n_canonical = len(fit.canonical)
    canonical, drug_design, cell_design = index_rows(fit, counts)

    alpha_pair = rng.normal(PAIR_LOC, PAIR_SCALE, size=(n_draws, 1))
    sigma_pair = np.abs(rng.normal(0.0, PAIR_SIGMA_SCALE, size=(n_draws, 1)))
    pair_logit = alpha_pair + sigma_pair * rng.normal(size=(n_draws, n_canonical))
    eta = pair_logit[:, canonical]

    blocks = (
        (fit.drug, DRUG_SIGMA_SCALE, drug_design),
        (fit.cell, CELL_SIGMA_SCALE, cell_design),
    )
    for names, sigma_scale, design in blocks:
        if not names:
            continue
        sigma = np.abs(rng.normal(0.0, sigma_scale, size=(n_draws, 1, 1)))
        alpha = rng.normal(size=(n_draws, len(names), 1))
        z = rng.normal(size=(n_draws, len(names), n_canonical))
        logit = alpha + sigma * z
        eta = eta + np.einsum("nj,djn->dn", design, logit[:, :, canonical])
    return expit(eta)


# --- folds -----------------------------------------------------------------


def assign_folds(counts: pd.DataFrame, n_folds: int, seed: int) -> np.ndarray:
    """Fold label of each observation, balanced within each canonical pair.

    The observations of one canonical pair are shuffled and then dealt round
    robin across the folds, so each pair is spread as evenly as its count allows
    and no training set loses a pair entirely. The deal starts at a random fold
    for each pair, so a remainder does not always land on the same one.
    """
    rng = np.random.default_rng(seed)
    ends = np.sort(counts[["bait", "prey"]].to_numpy(dtype=str), axis=1)
    key = np.array([f"{a}|{b}" for a, b in ends])
    fold = np.empty(len(key), dtype=int)
    for name in np.unique(key):
        rows = np.flatnonzero(key == name)
        start = int(rng.integers(n_folds))
        fold[rows[rng.permutation(rows.size)]] = (np.arange(rows.size) + start) % n_folds
    for held in range(n_folds):
        if set(key[fold != held]) != set(key):
            raise ValueError(f"fold {held} leaves a canonical pair out of its training set")
    return fold


def cross_validate(counts, samples, folds, drug, cell, steps, seed, which=None):
    """Fit each named training set and score its held-out rows and its own rows.

    Both splits are returned because a block that only fits noise raises the
    in-sample score and leaves the held-out score where it was.
    """
    which = sorted(set(folds)) if which is None else which
    predictions = []
    in_sample = []
    for held in which:
        train = counts[folds != held]
        test = counts[folds == held]
        fit = pyroppi.fit(
            train,
            samples,
            drug=drug,
            cell=cell,
            min_reads=MIN_READS,
            num_steps=steps,
            num_draws=CV_DRAWS,
            seed=seed,
        )
        frame = test[["sample", "bait", "prey", "total", "edited"]].copy()
        frame["fold"] = held
        frame["observed"] = frame["edited"] / frame["total"]
        frame["predicted"] = predicted_rate(fit, test)
        predictions.append(frame)
        observed_train = (train["edited"] / train["total"]).to_numpy()
        in_sample.append(
            {
                "fold": held,
                "pearson": float(pearsonr(observed_train, predicted_rate(fit, train)).statistic),
            }
        )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(in_sample)


def agreement(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """Pearson correlation and root mean squared error of a set of predictions."""
    r = float(pearsonr(observed, predicted).statistic)
    rmse = float(np.sqrt(np.mean((observed - predicted) ** 2)))
    return r, rmse


# --- panels ----------------------------------------------------------------


def ecdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Cumulative fraction of the positive values at or below each grid point."""
    clean = np.sort(values[np.isfinite(values) & (values > 0)])
    return np.searchsorted(clean, grid, side="right") / clean.size


def draw_predictive(ax, observed, posterior, prior) -> dict:
    """Panel a. Three cumulative distributions of the editing rate.

    Evaluated on a shared grid rather than at every point, because the two
    predictive sets each carry a thousand draws of every observation.
    """
    series = [
        ("Prior predictive", prior, PRIOR_COLOR, (0, (2.4, 1.4)), 0.9, 0.50, "right"),
        ("Posterior predictive", posterior, POSTERIOR_COLOR, "-", 1.0, 0.88, "left"),
        ("Observed", observed, OBSERVED_COLOR, "-", 1.2, 0.26, "left"),
    ]
    lowest = min(float(values[values > 0].min()) for _, values, *_ in series)
    highest = max(float(values.max()) for _, values, *_ in series)
    grid = np.logspace(np.log10(lowest), np.log10(highest), GRID_POINTS)

    summary = {}
    for name, values, colour, style, width, anchor, side in series:
        curve = ecdf(values, grid)
        ax.plot(grid, curve, color=colour, linestyle=style, linewidth=width)
        at = grid[int(np.searchsorted(curve, anchor))]
        ax.text(
            at * (1.7 if side == "left" else 0.6),
            anchor,
            name,
            color=colour,
            ha=side,
            va="center",
        )
        summary[name] = float(np.median(values))

    ax.set_xscale("log")
    ax.set_xlabel("Editing rate")
    ax.set_ylabel("Cumulative fraction\nof observations")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.5, 1.0])
    return summary


def draw_heldout(ax, table: pd.DataFrame) -> None:
    """Panel b. Held-out observed rate against the rate the model predicts."""
    limit = float(max(table["observed"].max(), table["predicted"].max())) * 1.08
    ax.plot([0, limit], [0, limit], color=COLORS["grid"], linewidth=0.7, zorder=1)
    for fold, rows in table.groupby("fold"):
        ax.scatter(
            rows["predicted"],
            rows["observed"],
            s=2.2,
            color=FOLD_COLORS[int(fold) % len(FOLD_COLORS)],
            linewidths=0.0,
            alpha=0.8,
            zorder=3,
        )
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_box_aspect(1)
    ax.set_xlabel("Predicted editing rate")
    ax.set_ylabel("Observed editing rate")

    # The fold a point was held out in is keyed by colouring its number, which
    # keeps the five labels on one line and needs no legend.
    ax.text(0.05, 0.96, "fold", transform=ax.transAxes, va="top", color=COLORS["gray"])
    for fold in range(N_FOLDS):
        ax.text(
            0.23 + 0.07 * fold,
            0.96,
            str(fold + 1),
            transform=ax.transAxes,
            va="top",
            color=FOLD_COLORS[fold],
        )

    r, rmse = agreement(table["observed"].to_numpy(), table["predicted"].to_numpy())
    ax.text(
        0.96,
        0.05,
        f"r = {r:.3f}\nRMSE = {rmse:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        linespacing=1.3,
    )


def draw_ablation(ax, table: pd.DataFrame) -> None:
    """Panel c. Held-out and in-sample agreement for each set of covariate blocks.

    The bar on a held-out point spans the inference seeds behind it, so a gap
    narrower than the seed spread cannot be read as real.
    """
    x = np.arange(len(table))
    ax.errorbar(
        x,
        table["heldout_mean"],
        yerr=np.vstack(
            [
                table["heldout_mean"] - table["heldout_min"],
                table["heldout_max"] - table["heldout_mean"],
            ]
        ),
        fmt="o",
        markersize=3.2,
        color=COLORS["ink"],
        elinewidth=0.7,
        capsize=1.6,
        zorder=3,
    )
    ax.scatter(
        x,
        table["insample_mean"],
        s=14,
        facecolor="white",
        edgecolor=COLORS["gray"],
        linewidths=0.7,
        zorder=4,
    )
    # The two splits are keyed by colouring their names in the corner the points
    # leave empty, which keeps every marker clear of type.
    ax.text(
        0.03,
        0.97,
        f"held out (fold {ABLATION_FOLD + 1})",
        transform=ax.transAxes,
        va="top",
        color=COLORS["ink"],
    )
    ax.text(0.03, 0.87, "in sample", transform=ax.transAxes, va="top", color=COLORS["gray"])

    floor = min(0.0, float(table["heldout_min"].min()) - 0.05)
    ax.set_xticks(x)
    ax.set_xticklabels(table["model"], rotation=30, ha="right")
    ax.set_xlim(-0.4, len(table) - 0.4)
    ax.set_ylim(floor, 1.0)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel("Agreement with the\nobserved rate (r)")


def build(predictive: dict, heldout: pd.DataFrame, ablation: pd.DataFrame):
    """Lay the three panels out across one double-column figure."""
    use_style()
    fig = figure(183, 62)
    axes = fig.subplots(1, 3)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.26, top=0.90, wspace=0.42)
    summary = draw_predictive(
        axes[0], predictive["observed"], predictive["posterior"], predictive["prior"]
    )
    draw_heldout(axes[1], heldout)
    draw_ablation(axes[2], ablation)
    for ax, letter in zip(axes, "abc"):
        panel_letter(ax, letter)
    return fig, summary


def write(fig, directory: Path, stem: str) -> None:
    """Write the figure as PDF and PNG, then close it."""
    fig.savefig(directory / f"{stem}.png")
    save(fig, directory / f"{stem}.pdf")


# --- script ----------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=REPOSITORY / "data",
        help="directory holding the dataset directories (default: the repository's data/)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=REPOSITORY / "results",
        help="directory to write this figure's refit tables into",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "figures",
        help="directory to write the PDF and PNG into",
    )
    parser.add_argument("--steps", type=int, default=30_000, help="SVI steps per fit")
    parser.add_argument(
        "--seeds", type=int, default=3, help="inference seeds behind each ablation point"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    products = args.results / "supp_figure_2"
    products.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    counts, samples = pyroppi.read_dataset(args.data / DATASET)
    scored = apply_read_floor(counts, MIN_READS)
    scored = scored[scored["used"]].reset_index(drop=True)
    print(f"{DATASET}: {len(counts)} observations, {len(scored)} at or above {MIN_READS} reads")
    print(f"{len(samples)} samples, {counts.groupby(['bait', 'prey']).ngroups} measured pairs")

    # Panel a. One fit on everything, then its two predictive distributions.
    full = pyroppi.fit(
        counts,
        samples,
        drug=DRUG,
        cell=CELL,
        min_reads=MIN_READS,
        num_steps=args.steps,
        num_draws=N_DRAWS,
        seed=0,
    )
    print(f"{len(full.canonical)} canonical pairs, final ELBO loss {full.losses[-1]:,.0f}")

    used = full.counts[full.counts["used"]].reset_index(drop=True)
    total = used["total"].to_numpy()
    observed_counts = used["edited"].to_numpy()
    rng = np.random.default_rng(PREDICTIVE_SEED)
    posterior_counts = rng.binomial(total[None, :], mu_draws(full, used))
    prior_counts = rng.binomial(
        total[None, :], prior_mu_draws(full, used, N_DRAWS, PREDICTIVE_SEED)
    )
    predictive = {
        "observed": observed_counts / total,
        "posterior": posterior_counts / total[None, :],
        "prior": prior_counts / total[None, :],
    }

    tail = (1.0 - NOMINAL) / 2.0
    low = np.quantile(posterior_counts, tail, axis=0)
    high = np.quantile(posterior_counts, 1.0 - tail, axis=0)
    coverage = float(np.mean((observed_counts >= low) & (observed_counts <= high)))
    print(
        f"\nnominal {NOMINAL:.0%} posterior predictive interval covers "
        f"{coverage:.1%} of {len(used)} observations"
    )
    print(f"read depth spans {total.min():,} to {total.max():,} reads")

    # Panel b. Five folds, one fit per training set.
    folds = assign_folds(scored, N_FOLDS, FOLD_SEED)
    print(f"\nfold sizes: {np.bincount(folds).tolist()}")
    heldout, insample = cross_validate(scored, samples, folds, DRUG, CELL, args.steps, 0)
    heldout.to_csv(products / "heldout_predictions.csv", index=False)
    r, rmse = agreement(heldout["observed"].to_numpy(), heldout["predicted"].to_numpy())
    print(f"held out over all {len(heldout)} observations: r = {r:.3f}, RMSE = {rmse:.4f}")
    for fold, rows in heldout.groupby("fold"):
        fold_r, fold_rmse = agreement(rows["observed"].to_numpy(), rows["predicted"].to_numpy())
        print(f"  fold {fold + 1}: n = {len(rows)}, r = {fold_r:.3f}, RMSE = {fold_rmse:.4f}")

    # Panel c. The same split, scored on its first fold for each set of blocks.
    # The full model at seed 0 is already fitted above, so it is reused.
    reference = heldout[heldout["fold"] == ABLATION_FOLD]
    rows = []
    for name, drug, cell in ABLATIONS:
        is_full = (drug, cell) == (DRUG, CELL)
        held, inside = [], []
        if is_full:
            held.append(agreement(reference["observed"].to_numpy(), reference["predicted"].to_numpy())[0])
            inside.append(insample.loc[insample["fold"] == ABLATION_FOLD, "pearson"].iloc[0])
        for seed in range(args.seeds):
            if is_full and seed == 0:
                continue
            table, fitted = cross_validate(
                scored, samples, folds, drug, cell, args.steps, seed, which=[ABLATION_FOLD]
            )
            held.append(agreement(table["observed"].to_numpy(), table["predicted"].to_numpy())[0])
            inside.append(float(fitted["pearson"].iloc[0]))
        rows.append(
            {
                "model": name,
                "n_seeds": len(held),
                "heldout_mean": float(np.mean(held)),
                "heldout_min": float(np.min(held)),
                "heldout_max": float(np.max(held)),
                "insample_mean": float(np.mean(inside)),
            }
        )
    ablation = pd.DataFrame(rows)
    ablation.to_csv(products / "ablation.csv", index=False)
    print(f"\nablation on fold {ABLATION_FOLD + 1}, Pearson r against the observed rate")
    print(ablation.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    fig, summary = build(predictive, heldout, ablation)
    print("\nmedian editing rate")
    for name, value in summary.items():
        print(f"  {name}: {value:.4f}")

    write(fig, args.output, "supp_figure_2")
    print(f"\nwritten to {args.output}")
    print(f"total runtime {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
