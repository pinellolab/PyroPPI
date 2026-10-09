"""The statistics reported from a fit: hit calling, nulls and effect sizes.

Every function here takes a result table and never touches the posterior draws,
so a statistic can be recomputed from a saved fit or from any table with the same
columns. Variant tests work on the logit scale, which is the scale the model adds
on, so a shift means the same thing wherever a variant's baseline sits.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

VARIANT = re.compile(r"^(?P<gene>.+):(?P<ref>[A-Z*])(?P<position>\d+)(?P<alt>[A-Z*])$")
SYNONYMOUS = "synonymous"
MISSENSE = "missense"
STOP = "stop"


# --- variants --------------------------------------------------------------


def parse_variants(variants: Sequence[str]) -> pd.DataFrame:
    """Split ``GENE:A63E`` names into gene, ref, position, alt and class."""
    names = pd.Series(variants, dtype=str)
    parts = names.str.extract(VARIANT)
    bad = names[parts.gene.isna()]
    if len(bad):
        raise ValueError(f"cannot parse variant name(s): {list(bad[:5])}")
    parts["position"] = parts.position.astype(int)
    parts["class"] = np.where(
        parts.ref == parts.alt, SYNONYMOUS, np.where(parts.alt == "*", STOP, MISSENSE)
    )
    parts.index = names.index
    return parts


def classify_variants(variants: Sequence[str]) -> pd.Series:
    """Class of each ``GENE:A63E`` name: synonymous, missense or stop."""
    return parse_variants(variants)["class"]


def synonymous_test(
    df: pd.DataFrame,
    score: str = "logit_mean",
    by: str | None = "gene",
    variant_class: str = "class",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """One-tailed test of each variant against its gene's synonymous pool.

    The synonymous variants of a gene are the null: a variant is called when its
    score sits below that pool by enough that the normal lower tail is under
    ``alpha``. ``score`` must name a column on the logit scale. Synonymous rows
    get a z and a p like any other, so the null can be inspected, but they are
    never flagged as hits.

    Pass ``by=None`` to pool every row into one null.
    """
    out = df.copy()
    if not out.index.is_unique:
        raise ValueError("the table needs a unique index; call reset_index first")
    z = pd.Series(np.nan, index=out.index, dtype=float)
    groups = [(None, out)] if by is None else out.groupby(by, sort=False)
    for name, part in groups:
        null = part.loc[part[variant_class] == SYNONYMOUS, score].to_numpy(float)
        null = null[np.isfinite(null)]
        if null.size < 2:
            raise ValueError(f"{name}: need at least 2 synonymous variants for a null")
        z.loc[part.index] = (part[score] - null.mean()) / null.std(ddof=1)
    out["z"] = z
    out["p"] = stats.norm.cdf(z)
    out["hit"] = (out["p"] < alpha) & (out[variant_class] != SYNONYMOUS)
    return out


# --- control pairs ---------------------------------------------------------


def _is_control(pairs: pd.DataFrame, controls: Sequence[str]) -> pd.Series:
    controls = set(controls)
    return pairs.protein_a.isin(controls) | pairs.protein_b.isin(controls)


def control_test(
    pairs: pd.DataFrame,
    score: str = "logit_mean",
    controls: Sequence[str] = ("GFP", "mCherry"),
) -> pd.DataFrame:
    """One-tailed test of each pair against the control pairs.

    Pairs with a control protein at either end are taken as non-interacting and
    their spread on the logit scale is the null. The returned table holds the
    remaining pairs with their z and upper-tail p, so a large score is small p.
    """
    control = _is_control(pairs, controls)
    null = pairs.loc[control, score].to_numpy(float)
    if null.size < 2:
        raise ValueError(f"need at least 2 control pairs; found {null.size}")
    null_sd = null.std(ddof=1)
    if not np.isfinite(null_sd) or null_sd == 0:
        raise ValueError("control pairs have no spread on the logit scale")
    out = pairs.loc[~control].copy()
    out["z"] = (out[score] - null.mean()) / null_sd
    out["p"] = stats.norm.sf(out["z"])
    return out


def control_threshold(
    pairs: pd.DataFrame,
    score: str = "logit_mean",
    controls: Sequence[str] = ("GFP", "mCherry"),
    q: float = 0.95,
) -> float:
    """The ``q`` quantile of the control null, as an interaction propensity."""
    control = _is_control(pairs, controls)
    null = pairs.loc[control, score].to_numpy(float)
    if null.size < 2:
        raise ValueError(f"need at least 2 control pairs; found {null.size}")
    cut = null.mean() + stats.norm.ppf(q) * null.std(ddof=1)
    return float(1.0 / (1.0 + np.exp(-cut)))


# --- multiple testing ------------------------------------------------------


def holm(p: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values. Non-finite entries stay non-finite."""
    return _adjust(p, "holm")


def benjamini_hochberg(p: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values. Non-finite entries stay non-finite."""
    return _adjust(p, "bh")


def _adjust(p: Sequence[float], method: str) -> np.ndarray:
    values = np.asarray(p, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    ordered = np.sort(values[finite])
    n = ordered.size
    if n == 0:
        return adjusted
    if method == "holm":
        # Each p scaled by the tests not yet rejected, then made non-decreasing.
        corrected = np.maximum.accumulate((n - np.arange(n)) * ordered)
    else:
        # Each p scaled by n over its rank, then made non-increasing from the top.
        corrected = np.minimum.accumulate((ordered * n / np.arange(1, n + 1))[::-1])[::-1]
    adjusted[finite] = np.minimum(corrected, 1.0)[np.argsort(np.argsort(values[finite]))]
    return adjusted


# --- discrimination --------------------------------------------------------


def auroc(positive: Sequence[float], negative: Sequence[float]) -> tuple[float, float]:
    """Area under the ROC curve and its two-sided Mann-Whitney p.

    The area is the Mann-Whitney U of ``positive`` over the number of pairs, so
    it reads as the chance a positive scores above a negative.
    """
    positive = np.asarray(positive, float)
    negative = np.asarray(negative, float)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if positive.size == 0 or negative.size == 0:
        return np.nan, np.nan
    result = stats.mannwhitneyu(positive, negative, alternative="two-sided")
    return float(result.statistic / (positive.size * negative.size)), float(result.pvalue)


def partial_spearman(
    x: Sequence[float], y: Sequence[float], covariate: Sequence[float]
) -> tuple[float, float]:
    """Spearman of x and y after removing the covariate from both.

    All three are ranked first, so the regression that removes the covariate is
    linear in ranks. The result is a rank correlation, and it differs from a
    partial Pearson computed on the raw values.
    """
    rank_x, rank_y, rank_z = (stats.rankdata(values) for values in (x, y, covariate))
    residual_x = rank_x - np.polyval(np.polyfit(rank_z, rank_x, 1), rank_z)
    residual_y = rank_y - np.polyval(np.polyfit(rank_z, rank_y, 1), rank_z)
    result = stats.spearmanr(residual_x, residual_y)
    return float(result.statistic), float(result.pvalue)


# --- the run-replacement null ----------------------------------------------
#
# Hits and structural features both run in contiguous stretches along a chain.
# Permuting a per-position label breaks the clustering of the hits and keeps the
# clustering of the feature, and against data with no signal it rejects at 11-28%
# where its nominal rate is 5%. Re-placing whole runs of hit-carrying positions
# holds both kinds of clustering fixed and is the null the manuscript reports.

_MAX_TRIES = 10_000


def _replace_runs(
    n_hit: np.ndarray, n_assayed: np.ndarray, rng: np.random.Generator, label: str = ""
) -> np.ndarray:
    """Slide the contiguous runs of hit-carrying positions along one chain.

    The number of runs, their lengths and the counts inside them are all kept;
    only where they sit changes. Two runs laid end to end would read as one
    longer run, so each of the R - 1 internal gaps reserves a position and the
    rest are spread over the R + 1 gaps uniformly (stars and bars). A placement
    that gives a position more hits than it assayed is redrawn.
    """
    flags = n_hit > 0
    if not flags.any():
        return n_hit.copy()
    edges = np.diff(np.r_[0, flags.astype(np.int8), 0])
    runs = [
        n_hit[start:end]
        for start, end in zip(np.where(edges == 1)[0], np.where(edges == -1)[0])
    ]
    n_runs = len(runs)
    free = len(n_hit) - sum(len(run) for run in runs) - (n_runs - 1)
    if free < 0:
        raise ValueError(f"{label}: the hit runs do not fit on the chain")
    for _ in range(_MAX_TRIES):
        rng.shuffle(runs)
        cuts = np.sort(rng.choice(free + n_runs, size=n_runs, replace=False))
        gaps = np.diff(np.r_[-1, cuts, free + n_runs]) - 1
        gaps[1:n_runs] += 1
        placed = []
        for gap, run in zip(gaps, runs):
            placed.append(np.zeros(gap, dtype=int))
            placed.append(run)
        placed.append(np.zeros(gaps[-1], dtype=int))
        candidate = np.concatenate(placed)
        if (candidate <= n_assayed).all():
            return candidate
    raise RuntimeError(
        f"{label}: no placement of {n_runs} hit runs over {len(n_hit)} positions kept "
        f"every position at or under its assayed count in {_MAX_TRIES} draws"
    )


def run_replacement_test(
    chain: Sequence,
    n_hit: np.ndarray,
    n_assayed: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    draws: int = 20_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Observed statistic and its one-tailed p under the run-replacement null.

    One row per position, ordered along each chain, with ``chain`` naming the
    chain a position belongs to; runs are re-placed within their own chain, so a
    chain never borrows another's length. ``statistic(n_hit, n_assayed)`` returns
    a number that is large when the hits are special. The p-value counts draws at
    or above the observation, with one added to both sides so an unbeaten
    observation reports 1 / (draws + 1) rather than zero.

    For the residue-distance contrast, ``statistic`` is the non-hit mean minus
    the hit mean of the distance to the partner chain::

        gap = lambda h, a: ((a - h) @ d) / (a - h).sum() - (h @ d) / h.sum()
    """
    chain = np.asarray(chain)
    n_hit = np.asarray(n_hit, dtype=int)
    n_assayed = np.asarray(n_assayed, dtype=int)
    observed = float(statistic(n_hit, n_assayed))
    rng = np.random.default_rng(seed)
    chains = [(name, np.where(chain == name)[0]) for name in pd.unique(chain)]

    null = np.empty(draws)
    for draw in range(draws):
        replaced = n_hit.copy()
        for name, rows in chains:
            replaced[rows] = _replace_runs(n_hit[rows], n_assayed[rows], rng, str(name))
        null[draw] = statistic(replaced, n_assayed)
    null = null[np.isfinite(null)]
    return observed, float((np.sum(null >= observed) + 1) / (null.size + 1))


# --- block permutation and bootstrap ---------------------------------------
#
# Variants at one residue are not independent of each other: ten substitutions at
# a residue that all behave alike carry about as much information as one. Both
# resamplings below move whole residues, so that structure is held fixed.


def hodges_lehmann(x: Sequence[float], y: Sequence[float]) -> float:
    """Median of all pairwise differences: the shift between two groups."""
    return float(np.median(np.subtract.outer(np.asarray(x), np.asarray(y))))


def _blocks_by_size(groups: np.ndarray, blocks: Sequence) -> dict:
    """Blocks grouped by size, as ``(rows, labels)`` arrays of shape (k, size).

    Only equal-sized blocks can swap labels one for one, so the permutation is
    stratified by size. ``rows`` indexes positions in the input arrays.
    """
    strata: dict[int, tuple[list, list]] = {}
    index = pd.Series(np.arange(len(groups)))
    for _, block in index.groupby(pd.Series(list(blocks), index=index.index), sort=True):
        rows, labels = strata.setdefault(len(block), ([], []))
        rows.append(block.to_numpy())
        labels.append(groups[block.to_numpy()])
    return {size: (np.array(r), np.array(l)) for size, (r, l) in strata.items()}


def block_permutation_test(
    values: Sequence[float],
    groups: Sequence[bool],
    blocks: Sequence,
    draws: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Hodges-Lehmann shift between the two groups and its two-sided p.

    ``groups`` is True for the group of interest and ``blocks`` names the residue
    each row belongs to. Each block's set of labels is reassigned to another block
    of the same size, so the null keeps the number of labelled rows per residue. A
    plain rank-sum test on the variants is anti-conservative here.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups, dtype=bool)
    rng = np.random.default_rng(seed)

    masks = np.empty((draws, len(values)), dtype=bool)
    for size, (rows, labels) in _blocks_by_size(groups, blocks).items():
        count = len(rows)
        # Which block's labels each block receives, then a shuffle inside the block.
        between = np.argsort(rng.random((draws, count)), axis=1)
        within = np.argsort(rng.random((draws, count, size)), axis=2)
        drawn = np.take_along_axis(labels[between], within, axis=2)
        masks[:, rows.ravel()] = drawn.reshape(draws, count * size)

    n_hit = int(groups.sum())
    n_rest = len(values) - n_hit
    null = np.empty(draws)
    # Every draw keeps the observed group sizes, so both gathers reshape cleanly.
    # Chunked only to bound the (draws, n_hit * n_rest) difference array.
    chunk = max(1, 20_000_000 // max(n_hit * n_rest, 1))
    for start in range(0, draws, chunk):
        part = masks[start : start + chunk]
        tiled = np.broadcast_to(values, part.shape)
        hit = tiled[part].reshape(len(part), n_hit)
        rest = tiled[~part].reshape(len(part), n_rest)
        null[start : start + chunk] = np.median(
            (hit[:, :, None] - rest[:, None, :]).reshape(len(part), -1), axis=1
        )

    observed = hodges_lehmann(values[groups], values[~groups])
    # +1 on both sides, so an unbeaten observation reports 1 / (draws + 1).
    return observed, float((np.count_nonzero(np.abs(null) >= abs(observed)) + 1) / (draws + 1))


def block_bootstrap_ci(
    values: Sequence[float],
    groups: Sequence[bool],
    blocks: Sequence,
    draws: int = 2_000,
    seed: int = 0,
    min_group: int = 5,
) -> tuple[float, float, int]:
    """95% interval on the Hodges-Lehmann shift, resampling whole residues.

    Draws that leave either group with fewer than ``min_group`` rows are skipped;
    the count of draws that were usable is returned alongside the interval.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups, dtype=bool)
    rng = np.random.default_rng(seed)
    index = pd.Series(np.arange(len(values)))
    block_rows = [
        rows.to_numpy()
        for _, rows in index.groupby(pd.Series(list(blocks), index=index.index), sort=True)
    ]

    shifts = []
    for choice in rng.integers(0, len(block_rows), (draws, len(block_rows))):
        rows = np.concatenate([block_rows[i] for i in choice])
        hit = groups[rows]
        if hit.sum() < min_group or (~hit).sum() < min_group:
            continue
        shifts.append(hodges_lehmann(values[rows][hit], values[rows][~hit]))
    if not shifts:
        return np.nan, np.nan, 0
    low, high = np.percentile(shifts, [2.5, 97.5])
    return float(low), float(high), len(shifts)
