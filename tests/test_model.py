import numpy as np
import pandas as pd
import pytest

import pyroppi
from pyroppi.data import apply_read_floor, prepare


def simulate(seed=0):
    """Four proteins, six canonical pairs, each measured in both orientations."""
    rng = np.random.default_rng(seed)
    proteins = ["A", "B", "C", "D"]
    true_logit = {
        ("A", "B"): 0.5,
        ("A", "C"): -0.5,
        ("A", "D"): -1.5,
        ("B", "C"): -2.5,
        ("B", "D"): -3.5,
        ("C", "D"): -4.5,
    }
    drug_shift = 1.0
    samples = pd.DataFrame(
        {
            "sample": [f"s{i}" for i in range(6)],
            "treatment": ["none"] * 3 + ["drug"] * 3,
            "drug": [0, 0, 0, 1, 1, 1],
        }
    )
    rows = []
    for _, sample in samples.iterrows():
        for bait in proteins:
            for prey in proteins:
                if bait == prey:
                    continue
                eta = true_logit[tuple(sorted((bait, prey)))] + drug_shift * sample["drug"]
                total = int(rng.integers(300, 600))
                edited = int(rng.binomial(total, 1 / (1 + np.exp(-eta))))
                rows.append((sample["sample"], bait, prey, total, edited))
    counts = pd.DataFrame(rows, columns=["sample", "bait", "prey", "total", "edited"])
    return counts, samples, true_logit


def test_read_floor_censors_and_drops():
    counts = pd.DataFrame(
        {
            "sample": ["s1", "s2", "s1", "s2", "s1"],
            "bait": ["A", "A", "B", "B", "C"],
            "prey": ["B", "B", "C", "C", "D"],
            "total": [100, 5, 10, 0, 50],
            "edited": [10, 1, 2, 0, 5],
        }
    )
    kept = apply_read_floor(counts, min_reads=20)
    # A-B keeps both rows, one censored; B-C has nothing above the floor and is dropped.
    assert list(zip(kept["bait"], kept["prey"])) == [("A", "B"), ("A", "B"), ("C", "D")]
    assert kept["used"].tolist() == [True, False, True]


def test_orientations_share_a_canonical_pair():
    counts, samples, _ = simulate()
    data = prepare(counts, samples, ["drug"], [], min_reads=20)
    assert len(data.measured) == 12
    assert len(data.canonical) == 6
    ab = data.measured.query("(bait == 'A' and prey == 'B') or (bait == 'B' and prey == 'A')")
    assert ab["canonical_index"].nunique() == 1


def test_fit_ranks_pairs_and_round_trips(tmp_path):
    counts, samples, true_logit = simulate()
    fit = pyroppi.fit(counts, samples, drug=["drug"], num_steps=8_000, num_draws=200)

    pairs = fit.pairs()
    estimated = pairs.set_index(["protein_a", "protein_b"])["logit_mean"]
    truth = pd.Series(true_logit).reindex(estimated.index)
    assert estimated.rank().tolist() == truth.rank().tolist()
    assert np.all(pairs["propensity_q05"] <= pairs["propensity_mean"])
    assert np.all(pairs["propensity_mean"] <= pairs["propensity_q95"])

    effects = fit.effects()
    assert len(effects) == 6
    assert effects["logit_mean"].mean() == pytest.approx(1.0, abs=0.3)

    contributions = fit.contributions()
    parts = contributions[["contrib_pair", "contrib_drug", "contrib_cell"]].sum(axis=1)
    assert np.allclose(parts, contributions["mu"], atol=1e-5)
    assert np.allclose(contributions.query("sample == 's0'")["contrib_drug"], 0.0)

    fit.save(tmp_path)
    restored = pyroppi.load_fit(tmp_path)
    pd.testing.assert_frame_equal(restored.pairs(), pairs)
    pd.testing.assert_frame_equal(restored.effects(), effects)
    pd.testing.assert_frame_equal(restored.contributions(), contributions)
    assert np.allclose(restored.losses, fit.losses)


def test_fit_without_baseline():
    counts, samples, _ = simulate()
    treated = samples.query("drug == 1")
    counts = counts[counts["sample"].isin(treated["sample"])]
    fit = pyroppi.fit(
        counts, treated, drug=["drug"], baseline=False, num_steps=500, num_draws=50
    )
    contributions = fit.contributions()
    assert np.allclose(contributions["contrib_pair"], 0.0)
    assert np.allclose(contributions["contrib_drug"], contributions["mu"], atol=1e-5)
    with pytest.raises(ValueError):
        fit.pairs()
