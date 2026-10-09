import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp

from pyroppi.stats import (
    auroc,
    benjamini_hochberg,
    block_bootstrap_ci,
    block_permutation_test,
    _replace_runs,
    classify_variants,
    control_test,
    control_threshold,
    hodges_lehmann,
    holm,
    parse_variants,
    partial_spearman,
    run_replacement_test,
    synonymous_test,
)


# --- variants --------------------------------------------------------------


def test_classify_variants():
    names = ["RAD51D:A63A", "RAD51D:A63E", "XRCC2:R12*", "XRCC2:*12*"]
    assert list(classify_variants(names)) == ["synonymous", "missense", "stop", "synonymous"]


def test_parse_variants_splits_a_gene_name_containing_digits():
    parsed = parse_variants(["RAD51D:A63E"]).iloc[0]
    assert (parsed.gene, parsed.ref, parsed.position, parsed.alt) == ("RAD51D", "A", 63, "E")


def test_parse_variants_rejects_an_unparsable_name():
    with pytest.raises(ValueError, match="cannot parse"):
        parse_variants(["RAD51D-A63E"])


# --- the synonymous test ---------------------------------------------------


def simulate_scan(seed=0, n_synonymous=60, n_missense=200, n_broken=40, shift=-2.5):
    """A scan in which ``n_broken`` missense variants are shifted down by ``shift``.

    Synonymous variants are drawn from the same standard normal as the intact
    missense variants, so the only structure a test can find is the shift.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_synonymous):
        rows.append((f"G:A{i + 1}A", rng.normal()))
    for i in range(n_missense):
        broken = i < n_broken
        rows.append((f"G:A{i + 101}E", rng.normal() + (shift if broken else 0.0)))
    frame = pd.DataFrame(rows, columns=["variant", "logit_mean"])
    frame["gene"] = "G"
    frame["class"] = classify_variants(frame.variant).to_numpy()
    frame["broken"] = frame.variant.isin([f"G:A{i + 101}E" for i in range(n_broken)])
    return frame


def test_synonymous_test_recovers_the_shifted_variants():
    out = synonymous_test(simulate_scan())
    # The shift is large enough that most broken variants clear a one-tailed 0.05,
    # and the intact ones are called at about the nominal rate.
    assert out.broken[out.hit].sum() >= 25
    intact = out[(out["class"] == "missense") & ~out.broken]
    assert intact.hit.mean() < 0.12
    assert out[out.broken].hit.mean() > 4 * intact.hit.mean()


def test_synonymous_test_never_flags_the_null_class():
    out = synonymous_test(simulate_scan())
    synonymous = out[out["class"] == "synonymous"]
    assert not synonymous.hit.any()
    # They still carry a z, which is what makes the null inspectable.
    assert synonymous.z.notna().all()
    assert synonymous.z.mean() == pytest.approx(0.0, abs=1e-9)
    assert synonymous.z.std(ddof=1) == pytest.approx(1.0, abs=1e-9)


def test_synonymous_test_matches_an_explicit_z():
    frame = simulate_scan()
    out = synonymous_test(frame)
    null = frame.loc[frame["class"] == "synonymous", "logit_mean"]
    z = (frame.logit_mean - null.mean()) / null.std(ddof=1)
    assert np.allclose(out.z, z)
    assert np.allclose(out.p, sp.norm.cdf(z))


def test_synonymous_test_uses_each_genes_own_null():
    frame = pd.concat(
        [simulate_scan(seed=1).assign(gene="G1"), simulate_scan(seed=1).assign(gene="G2")],
        ignore_index=True,
    )
    frame.loc[frame.gene == "G2", "logit_mean"] += 5.0
    out = synonymous_test(frame)
    # A constant offset moves both the variants and their null, so nothing moves.
    assert np.allclose(
        out[out.gene == "G1"].z.to_numpy(), out[out.gene == "G2"].z.to_numpy()
    )


def test_synonymous_test_needs_a_null():
    frame = simulate_scan(n_synonymous=1)
    with pytest.raises(ValueError, match="at least 2 synonymous"):
        synonymous_test(frame)


# --- control pairs ---------------------------------------------------------


def control_pairs():
    rng = np.random.default_rng(0)
    rows = [("GFP", f"P{i}", rng.normal(-3.0, 0.4)) for i in range(12)]
    rows += [("mCherry", f"P{i}", rng.normal(-3.0, 0.4)) for i in range(12)]
    rows += [("P1", "P2", -1.0), ("P3", "P4", -3.1)]
    return pd.DataFrame(rows, columns=["protein_a", "protein_b", "logit_mean"])


def test_control_test_drops_the_controls_and_scores_upward():
    out = control_test(control_pairs())
    assert len(out) == 2
    null = control_pairs().logit_mean[:24]
    expected = (-1.0 - null.mean()) / null.std(ddof=1)
    assert out.z.iloc[0] == pytest.approx(expected)
    assert np.allclose(out.p, sp.norm.sf(out.z))
    # A strong pair sits above the null, a control-like pair does not.
    assert out.p.iloc[0] < 0.05 < out.p.iloc[1]


def test_control_threshold_is_a_propensity():
    pairs = control_pairs()
    cut = control_threshold(pairs, q=0.95)
    null = pairs.logit_mean[:24]
    expected = null.mean() + sp.norm.ppf(0.95) * null.std(ddof=1)
    assert cut == pytest.approx(1 / (1 + np.exp(-expected)))
    assert 0.0 < cut < 1.0


# --- multiple testing ------------------------------------------------------


def test_holm_matches_an_inline_reference():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.6])
    order = np.argsort(p)
    reference = np.empty(len(p))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        reference[index] = min(running, 1.0)
    assert np.allclose(holm(p), reference)
    # Holm is never smaller than the raw p and never larger than Bonferroni.
    assert np.all(holm(p) >= p)
    assert np.all(holm(p) <= np.minimum(p * len(p), 1.0))


def test_benjamini_hochberg_matches_scipy():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.6])
    assert np.allclose(benjamini_hochberg(p), sp.false_discovery_control(p, method="bh"))


def test_adjustment_carries_non_finite_entries_through():
    p = np.array([0.01, np.nan, 0.5])
    adjusted = holm(p)
    assert np.isnan(adjusted[1])
    # The nan is not counted as a test: two tests, so the smallest doubles.
    assert adjusted[0] == pytest.approx(0.02)


def test_adjustment_of_an_empty_family():
    assert np.isnan(holm([np.nan])).all()
    assert benjamini_hochberg([]).size == 0


# --- discrimination --------------------------------------------------------


def test_auroc_hand_computed():
    # Positives 3 and 4 beat negatives 1 and 2 four times out of four, and
    # positive 1.5 beats negative 1 only: 5 of 6 pairs.
    area, _ = auroc([3.0, 4.0, 1.5], [1.0, 2.0])
    assert area == pytest.approx(5 / 6)


def test_auroc_of_a_perfect_and_a_reversed_split():
    assert auroc([3.0, 4.0], [1.0, 2.0])[0] == pytest.approx(1.0)
    assert auroc([1.0, 2.0], [3.0, 4.0])[0] == pytest.approx(0.0)


def test_auroc_is_one_half_for_one_distribution():
    rng = np.random.default_rng(0)
    area, p = auroc(rng.normal(size=4000), rng.normal(size=4000))
    assert area == pytest.approx(0.5, abs=0.03)
    assert p > 0.05


def test_auroc_of_an_empty_group():
    assert np.isnan(auroc([], [1.0, 2.0])).all()


# --- partial spearman ------------------------------------------------------


def test_partial_spearman_equals_plain_spearman_against_noise():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    y = x + rng.normal(size=400)
    noise = rng.normal(size=400)
    plain = sp.spearmanr(x, y).statistic
    assert partial_spearman(x, y, noise)[0] == pytest.approx(plain, abs=0.02)


def test_partial_spearman_removes_a_shared_driver():
    rng = np.random.default_rng(0)
    driver = rng.normal(size=400)
    x = driver + 0.3 * rng.normal(size=400)
    y = driver + 0.3 * rng.normal(size=400)
    assert sp.spearmanr(x, y).statistic > 0.8
    rho, p = partial_spearman(x, y, driver)
    assert abs(rho) < 0.2
    assert p > 0.01


# --- the run-replacement null ----------------------------------------------


def distance_gap(distance):
    """Non-hit mean minus hit mean: large when the hits sit closer to the partner."""

    def statistic(n_hit, n_assayed):
        other = n_assayed - n_hit
        return (other @ distance) / other.sum() - (n_hit @ distance) / n_hit.sum()

    return statistic


def chain_profile(rng, length=120, n_runs=6, run_length=4):
    """One chain of positions, three variants each, with hits in contiguous runs.

    The runs are laid down end to end and then slid into place by the same
    mechanism the null uses, so the hits are placed independently of anything
    else on the chain.
    """
    n_assayed = np.full(length, 3)
    n_hit = np.zeros(length, dtype=int)
    for run in range(n_runs):
        start = run * (run_length + 1)
        n_hit[start : start + run_length] = 2
    return _replace_runs(n_hit, n_assayed, rng), n_assayed


def smooth_distance(rng, length=120, window=21):
    """A distance profile that runs in contiguous stretches, as a real chain does."""
    raw = rng.normal(20, 8, size=length + window - 1)
    return np.convolve(raw, np.ones(window) / window, "valid")


def test_run_replacement_null_holds_its_nominal_rate_without_signal():
    # Distance runs in contiguous stretches and so do the hits, but the two are
    # placed independently, so there is nothing for the test to find. A null that
    # broke the runs apart would read the hit clustering itself as evidence.
    rng = np.random.default_rng(0)
    chain = np.repeat(["A", "B"], 120)
    trials = 80
    rejections = 0
    for _ in range(trials):
        distance = np.concatenate([smooth_distance(rng), smooth_distance(rng)])
        parts = [chain_profile(rng), chain_profile(rng)]
        n_hit = np.concatenate([part[0] for part in parts])
        n_assayed = np.concatenate([part[1] for part in parts])
        _, p = run_replacement_test(
            chain, n_hit, n_assayed, distance_gap(distance), draws=200, seed=1
        )
        rejections += p < 0.05
    # 80 trials at a nominal 5% gives 4 rejections on average, with a standard
    # deviation just under 2; the naive position permutation fires at 11-28%.
    assert rejections / trials < 0.13


def test_run_replacement_null_finds_a_planted_effect():
    rng = np.random.default_rng(1)
    chain = np.repeat("A", 120)
    n_hit, n_assayed = chain_profile(rng)
    # The hit positions are put at 5 angstroms and everything else at 25.
    distance = np.where(n_hit > 0, 5.0, 25.0)
    observed, p = run_replacement_test(
        chain, n_hit, n_assayed, distance_gap(distance), draws=400, seed=0
    )
    assert observed > 10.0
    assert p < 0.01


def test_run_replacement_keeps_the_run_structure_and_the_counts():
    rng = np.random.default_rng(2)
    chain = np.repeat("A", 120)
    n_hit, n_assayed = chain_profile(rng)
    seen = []

    def record(replaced, _assayed):
        seen.append(replaced.copy())
        return 0.0

    run_replacement_test(chain, n_hit, n_assayed, record, draws=50, seed=0)

    def runs(a):
        pieces = np.split(a, np.where(np.diff(a > 0))[0] + 1)
        return sorted(len(piece) for piece in pieces if piece[0] > 0)

    for replaced in seen[1:]:
        assert replaced.sum() == n_hit.sum()
        assert runs(replaced) == runs(n_hit)
        assert (replaced <= n_assayed).all()
    # The runs actually move, or the null would be a single point.
    assert any(not np.array_equal(replaced, n_hit) for replaced in seen[1:])


def test_run_replacement_keeps_chains_apart():
    rng = np.random.default_rng(3)
    chain = np.repeat(["A", "B"], 120)
    n_hit_a, n_assayed_a = chain_profile(rng)
    n_hit = np.concatenate([n_hit_a, np.zeros(120, dtype=int)])
    n_assayed = np.concatenate([n_assayed_a, np.full(120, 3)])
    seen = []

    def record(replaced, _assayed):
        seen.append(replaced.copy())
        return 0.0

    run_replacement_test(chain, n_hit, n_assayed, record, draws=20, seed=0)
    # Chain B has no hits, so no hit may be placed on it.
    assert all(replaced[120:].sum() == 0 for replaced in seen)


# --- block permutation and bootstrap ---------------------------------------


def test_hodges_lehmann_is_the_median_pairwise_difference():
    assert hodges_lehmann([4.0, 6.0], [1.0, 2.0]) == pytest.approx(3.5)


def blocked_frame(rng, n_blocks=40, size=5, shift=0.0):
    """Residue blocks of equal size, each wholly hit or wholly non-hit.

    The block effect is much larger than the within-block spread, so the
    effective sample size is the number of blocks, not the number of rows.
    """
    values, groups, blocks = [], [], []
    for block in range(n_blocks):
        is_hit = block % 2 == 0
        level = rng.normal(0.0, 1.0) + (shift if is_hit else 0.0)
        values.extend(level + 0.05 * rng.normal(size=size))
        groups.extend([is_hit] * size)
        blocks.extend([block] * size)
    return np.array(values), np.array(groups), np.array(blocks)


def test_block_permutation_is_not_fooled_by_within_block_replication():
    rng = np.random.default_rng(0)
    values, groups, blocks = blocked_frame(rng)
    observed, p = block_permutation_test(values, groups, blocks, draws=1000, seed=0)
    # A rank-sum test over the 200 rows reads the block replication as evidence.
    assert p > 0.05
    assert abs(observed) < 1.5


def test_block_permutation_finds_a_real_block_shift():
    rng = np.random.default_rng(0)
    values, groups, blocks = blocked_frame(rng, n_blocks=40, shift=2.0)
    observed, p = block_permutation_test(values, groups, blocks, draws=1000, seed=0)
    assert observed > 1.0
    assert p < 0.01


def test_block_permutation_p_has_a_floor():
    rng = np.random.default_rng(0)
    values, groups, blocks = blocked_frame(rng, shift=50.0)
    _, p = block_permutation_test(values, groups, blocks, draws=200, seed=0)
    assert p == pytest.approx(1 / 201)


def test_block_bootstrap_ci_covers_the_shift():
    rng = np.random.default_rng(0)
    values, groups, blocks = blocked_frame(rng, n_blocks=40, shift=2.0)
    observed, _ = block_permutation_test(values, groups, blocks, draws=200, seed=0)
    low, high, used = block_bootstrap_ci(values, groups, blocks, draws=300, seed=0)
    assert low < observed < high
    assert used > 290


def test_block_bootstrap_ci_of_a_null_straddles_zero():
    rng = np.random.default_rng(0)
    values, groups, blocks = blocked_frame(rng)
    low, high, _ = block_bootstrap_ci(values, groups, blocks, draws=300, seed=0)
    assert low < 0.0 < high
