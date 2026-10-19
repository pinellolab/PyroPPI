"""The PyroPPI model, its fit by variational inference, and the fitted results.

Each observation is the number of edited reporter reads out of the total reads
for one measured pair in one sample. The log-odds of editing is a sum of three
blocks, each estimated per canonical pair:

    pair   the interaction baseline of the pair
    drug   one shift per drug or condition indicator that is switched on
    cell   one shift per cell-type indicator that is switched on

The editing probability is reported as increments that add up to the fitted mean:

    a  = sigmoid(pair)
    b  = sigmoid(pair + drug)
    mu = sigmoid(pair + drug + cell)

    contrib_pair = a,  contrib_drug = b - a,  contrib_cell = mu - b
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
import pandas as pd
from scipy.special import expit
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_median
from tqdm.auto import tqdm

from .data import Prepared, prepare

PAIR_LOC = -2.0
PAIR_SCALE = 1.0
PAIR_SIGMA_SCALE = 0.7
DRUG_SIGMA_SCALE = 0.9
CELL_SIGMA_SCALE = 0.7
EPS = 1e-8


def compose(pair_eta, drug_eta, cell_eta, baseline: bool = True, xp=jnp, sigmoid=None):
    """Return mu and the pair, drug and cell increments of the editing probability.

    The blocks add on the log-odds scale, so each increment is what that block
    moves the editing probability once the earlier blocks are in place. Reading a
    saved fit passes ``xp=np`` and ``sigmoid=expit`` so no accelerator is needed.
    """
    sigmoid = jax.nn.sigmoid if sigmoid is None else sigmoid
    after_pair = sigmoid(pair_eta) if baseline else xp.zeros_like(pair_eta)
    after_drug = sigmoid(pair_eta + drug_eta)
    mu = xp.clip(sigmoid(pair_eta + drug_eta + cell_eta), EPS, 1.0 - EPS)
    return mu, after_pair, after_drug - after_pair, mu - after_drug


def _block_eta(name, design, sigma_scale, pair_plate, sample_index, canonical_index):
    """Log-odds shift of one covariate block for each observation.

    Each indicator column j has a level alpha[j] shared by all pairs, and each
    canonical pair c deviates from it by sigma * z[j, c]. An observation adds
    the logits of the columns switched on in its sample.
    """
    n_obs = sample_index.shape[0]
    n_cols = design.shape[1]
    if n_cols == 0:
        return jnp.zeros(n_obs)
    sigma = numpyro.sample(f"sigma_{name}", dist.HalfNormal(sigma_scale))
    with numpyro.plate(f"{name}_covariate", n_cols, dim=-2):
        alpha = numpyro.sample(f"alpha_{name}", dist.Normal(0.0, 1.0))
        with pair_plate:
            z = numpyro.sample(f"z_{name}", dist.Normal(0.0, 1.0))
    logit = alpha + sigma * z  # (J, C)
    return jnp.sum(design[sample_index] * logit[:, canonical_index].T, axis=-1)


def model(
    total,
    sample_index,
    canonical_index,
    drug_design,
    cell_design,
    n_canonical: int,
    baseline: bool = True,
    edited=None,
):
    """Hierarchical binomial model of edited reporter reads."""
    pair_plate = numpyro.plate("canonical_pair", n_canonical, dim=-1)
    if baseline:
        alpha_pair = numpyro.sample("alpha_pair", dist.Normal(PAIR_LOC, PAIR_SCALE))
        sigma_pair = numpyro.sample("sigma_pair", dist.HalfNormal(PAIR_SIGMA_SCALE))
        with pair_plate:
            z_pair = numpyro.sample("z_pair", dist.Normal(0.0, 1.0))
        pair_eta = (alpha_pair + sigma_pair * z_pair)[canonical_index]
    else:
        pair_eta = jnp.zeros(total.shape[0])

    drug_eta = _block_eta(
        "drug", drug_design, DRUG_SIGMA_SCALE, pair_plate, sample_index, canonical_index
    )
    cell_eta = _block_eta(
        "cell", cell_design, CELL_SIGMA_SCALE, pair_plate, sample_index, canonical_index
    )
    mu, _, _, _ = compose(pair_eta, drug_eta, cell_eta, baseline)

    with numpyro.plate("observation", total.shape[0]):
        numpyro.sample("edited", dist.Binomial(total_count=total, probs=mu), obs=edited)


def _block_draws(posterior, name, n_draws, n_canonical):
    """Per-covariate, per-pair logits (D, J, C) from the guide's latent draws."""
    if f"z_{name}" not in posterior:
        return np.zeros((n_draws, 0, n_canonical), dtype=np.float32)
    sigma = np.asarray(posterior[f"sigma_{name}"])[:, None, None]
    return np.asarray(posterior[f"alpha_{name}"] + sigma * posterior[f"z_{name}"])


@dataclass
class Fit:
    """A fitted PyroPPI model.

    ``draws`` holds posterior draws of the log-odds: ``pair_logit`` (D, C),
    ``drug_logit`` (D, J, C) and ``cell_logit`` (D, K, C), where C indexes the
    rows of ``canonical``, J the names in ``drug`` and K the names in ``cell``.
    """

    canonical: pd.DataFrame
    measured: pd.DataFrame
    samples: pd.DataFrame
    counts: pd.DataFrame
    drug: list[str]
    cell: list[str]
    baseline: bool
    min_reads: int
    losses: np.ndarray
    draws: dict[str, np.ndarray]

    def pairs(self) -> pd.DataFrame:
        """Interaction propensity of each canonical pair.

        The propensity is sigmoid of the pair logit, the editing probability in
        the reference condition.
        """
        if not self.baseline:
            raise ValueError("this fit has no pair baseline; use effects() instead")
        logit = self.draws["pair_logit"]
        propensity = 1.0 / (1.0 + np.exp(-logit))
        table = self.canonical.copy()
        table["logit_mean"] = logit.mean(axis=0)
        table["logit_sd"] = logit.std(axis=0)
        table["propensity_mean"] = propensity.mean(axis=0)
        table["propensity_q05"] = np.quantile(propensity, 0.05, axis=0)
        table["propensity_q95"] = np.quantile(propensity, 0.95, axis=0)
        return table

    def effects(self) -> pd.DataFrame:
        """Log-odds shift of each drug and cell covariate for each canonical pair."""
        tables = []
        for block, names in (("drug", self.drug), ("cell", self.cell)):
            logit = self.draws[f"{block}_logit"]
            for j, name in enumerate(names):
                table = self.canonical.copy()
                table.insert(0, "covariate", name)
                table.insert(0, "block", block)
                table["logit_mean"] = logit[:, j].mean(axis=0)
                table["logit_sd"] = logit[:, j].std(axis=0)
                tables.append(table)
        if not tables:
            columns = ["block", "covariate", "protein_a", "protein_b", "logit_mean", "logit_sd"]
            return pd.DataFrame(columns=columns)
        return pd.concat(tables, ignore_index=True)

    def contributions(self) -> pd.DataFrame:
        """Posterior mean of mu and its pair, drug and cell increments per observation.

        ``used`` marks the observations that entered the likelihood; the others
        fell below the read floor.
        """
        drug_design = self.samples[self.drug].to_numpy(dtype=np.float32)
        cell_design = self.samples[self.cell].to_numpy(dtype=np.float32)
        s = self.counts["sample_index"].to_numpy()
        c = self.counts["canonical_index"].to_numpy()
        n_draws = self.draws["pair_logit"].shape[0]
        width = max(len(self.drug), len(self.cell), 1)
        chunk = max(1, 20_000_000 // (n_draws * width))

        means = np.zeros((len(self.counts), 4))
        for start in range(0, len(self.counts), chunk):
            rows = slice(start, start + chunk)
            pair_eta = self.draws["pair_logit"][:, c[rows]]
            drug_eta = np.einsum(
                "nj,djn->dn", drug_design[s[rows]], self.draws["drug_logit"][:, :, c[rows]]
            )
            cell_eta = np.einsum(
                "nj,djn->dn", cell_design[s[rows]], self.draws["cell_logit"][:, :, c[rows]]
            )
            parts = compose(pair_eta, drug_eta, cell_eta, self.baseline, xp=np, sigmoid=expit)
            means[rows] = np.stack([p.mean(axis=0) for p in parts], axis=1)

        table = self.counts[["sample", "bait", "prey", "total", "edited", "used"]].copy()
        table[["mu", "contrib_pair", "contrib_drug", "contrib_cell"]] = means
        return table

    def save(self, directory: str | Path) -> None:
        """Write the result tables, posterior draws, loss trace and settings."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.baseline:
            self.pairs().to_csv(directory / "pairs.csv", index=False)
        self.effects().to_csv(directory / "effects.csv", index=False)
        self.contributions().to_csv(directory / "contributions.csv", index=False)
        self.samples.to_csv(directory / "samples.csv", index=False)
        np.savez(directory / "draws.npz", **self.draws)
        pd.DataFrame({"step": np.arange(1, len(self.losses) + 1), "loss": self.losses}).to_csv(
            directory / "losses.csv", index=False
        )
        settings = {
            "drug": self.drug,
            "cell": self.cell,
            "baseline": self.baseline,
            "min_reads": self.min_reads,
        }
        (directory / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")


def _make_fit(data: Prepared, drug, cell, baseline, min_reads, losses, draws) -> Fit:
    return Fit(
        canonical=data.canonical,
        measured=data.measured,
        samples=data.samples,
        counts=data.counts,
        drug=list(drug),
        cell=list(cell),
        baseline=baseline,
        min_reads=min_reads,
        losses=np.asarray(losses),
        draws=draws,
    )


def fit(
    counts: pd.DataFrame,
    samples: pd.DataFrame,
    drug: Sequence[str] = (),
    cell: Sequence[str] = (),
    baseline: bool = True,
    min_reads: int = 20,
    num_steps: int = 60_000,
    num_draws: int = 1_000,
    seed: int = 0,
) -> Fit:
    """Fit PyroPPI to a dataset by stochastic variational inference.

    ``drug`` and ``cell`` name the 0/1 indicator columns of ``samples`` that
    form the drug and cell blocks. Observations with fewer than ``min_reads``
    total reads are left out of the likelihood, and measured pairs with no
    observation left are dropped. Set ``baseline=False`` for a screen in which
    every sample is treated, so there is no untreated condition to estimate a
    pair baseline from.
    """
    if not baseline and not drug and not cell:
        raise ValueError("a fit without a pair baseline needs drug or cell columns")
    data = prepare(counts, samples, list(drug), list(cell), min_reads)
    used = data.counts[data.counts["used"]]
    n_canonical = len(data.canonical)
    inputs = dict(
        total=jnp.asarray(used["total"].to_numpy()),
        sample_index=jnp.asarray(used["sample_index"].to_numpy()),
        canonical_index=jnp.asarray(used["canonical_index"].to_numpy()),
        drug_design=jnp.asarray(data.drug_design, dtype=jnp.float32),
        cell_design=jnp.asarray(data.cell_design, dtype=jnp.float32),
        n_canonical=n_canonical,
        baseline=baseline,
    )
    edited = jnp.asarray(used["edited"].to_numpy())

    guide = AutoNormal(model, init_loc_fn=init_to_median, init_scale=0.1)
    optimizer = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(5e-4))
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=10))

    fit_key, draw_key = jax.random.split(jax.random.PRNGKey(seed))
    state = svi.init(fit_key, edited=edited, **inputs)
    update = jax.jit(lambda state: svi.update(state, edited=edited, **inputs))
    losses = []
    progress = tqdm(range(num_steps), desc="Fitting")
    for step in progress:
        state, loss = update(state)
        losses.append(loss)
        if (step + 1) % 1_000 == 0:
            progress.set_postfix(loss=f"{float(loss):.1f}")
    losses = np.asarray(jax.device_get(losses))
    posterior = guide.sample_posterior(
        draw_key, svi.get_params(state), sample_shape=(num_draws,), **inputs
    )

    if baseline:
        sigma = np.asarray(posterior["sigma_pair"])[:, None]
        alpha = np.asarray(posterior["alpha_pair"])[:, None]
        pair_logit = alpha + sigma * np.asarray(posterior["z_pair"])
    else:
        pair_logit = np.zeros((num_draws, n_canonical), dtype=np.float32)
    draws = {
        "pair_logit": pair_logit,
        "drug_logit": _block_draws(posterior, "drug", num_draws, n_canonical),
        "cell_logit": _block_draws(posterior, "cell", num_draws, n_canonical),
    }
    return _make_fit(data, drug, cell, baseline, min_reads, losses, draws)


def load_fit(directory: str | Path) -> Fit:
    """Restore a fit written by ``Fit.save``."""
    directory = Path(directory)
    settings = json.loads((directory / "settings.json").read_text())
    counts = pd.read_csv(
        directory / "contributions.csv", dtype={"sample": str, "bait": str, "prey": str}
    )
    samples = pd.read_csv(directory / "samples.csv", dtype={"sample": str})
    data = prepare(
        counts[["sample", "bait", "prey", "total", "edited"]],
        samples,
        settings["drug"],
        settings["cell"],
        settings["min_reads"],
    )
    with np.load(directory / "draws.npz") as archive:
        draws = {name: archive[name] for name in archive.files}
    losses = pd.read_csv(directory / "losses.csv")["loss"].to_numpy()
    return _make_fit(
        data,
        settings["drug"],
        settings["cell"],
        settings["baseline"],
        settings["min_reads"],
        losses,
        draws,
    )
