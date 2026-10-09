"""Reading a PPI Recorder dataset and preparing it for the model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COUNT_COLUMNS = ["sample", "bait", "prey", "total", "edited"]


def read_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ``counts.csv`` and ``samples.csv`` from a dataset directory.

    ``counts`` has one row per sample and measured pair, with the bait (ABE8e
    fusion), the prey (MCP fusion), the total reporter reads and the edited
    reads. ``samples`` has one row per sample with its descriptive columns and
    the 0/1 design indicator columns.
    """
    path = Path(path)
    counts = pd.read_csv(
        path / "counts.csv", dtype={"sample": str, "bait": str, "prey": str}
    )
    samples = pd.read_csv(path / "samples.csv", dtype={"sample": str})
    missing = set(COUNT_COLUMNS) - set(counts.columns)
    if missing:
        raise ValueError(f"counts.csv is missing columns: {sorted(missing)}")
    if "sample" not in samples.columns:
        raise ValueError("samples.csv has no 'sample' column")
    return counts, samples


@dataclass
class Prepared:
    """A dataset after the read floor, indexed for the model."""

    counts: pd.DataFrame  # kept rows, with sample_index, measured_index, canonical_index, used
    samples: pd.DataFrame  # one row per sample, in model order
    measured: pd.DataFrame  # bait, prey, canonical_index
    canonical: pd.DataFrame  # protein_a, protein_b
    drug_design: np.ndarray  # (S, J)
    cell_design: np.ndarray  # (S, K)


def apply_read_floor(counts: pd.DataFrame, min_reads: int) -> pd.DataFrame:
    """Mark which observations enter the likelihood and drop emptied pairs.

    An observation enters the likelihood when its total is at least
    ``min_reads`` (and above zero). Observations below the floor are kept in
    the table but censored, so they do not inform the fit. A measured pair left
    with no observation in the likelihood is dropped entirely, because with no
    data its estimate would be the prior alone.
    """
    counts = counts.copy()
    counts["used"] = (counts["total"] > 0) & (counts["total"] >= min_reads)
    survives = counts.groupby(["bait", "prey"])["used"].transform("any")
    return counts[survives].reset_index(drop=True)


def check_design(samples: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Return the named indicator columns as an (S, J) float array."""
    missing = [c for c in columns if c not in samples.columns]
    if missing:
        raise ValueError(f"samples.csv has no column(s): {missing}")
    design = samples[columns].to_numpy(dtype=float).reshape(len(samples), len(columns))
    if not np.isin(design, (0.0, 1.0)).all():
        raise ValueError(f"design columns must be 0/1: {columns}")
    return design


def prepare(
    counts: pd.DataFrame,
    samples: pd.DataFrame,
    drug: list[str],
    cell: list[str],
    min_reads: int,
) -> Prepared:
    """Apply the read floor and index samples, measured pairs and canonical pairs.

    A canonical pair is the unordered pair {bait, prey}, so a pair measured in
    both orientations shares one canonical pair. Canonical pairs are sorted by
    name.
    """
    if (counts["edited"] > counts["total"]).any() or (counts["edited"] < 0).any():
        raise ValueError("edited reads must lie between 0 and total reads")
    unknown = set(counts["sample"]) - set(samples["sample"])
    if unknown:
        raise ValueError(f"samples missing from samples.csv: {sorted(unknown)}")

    counts = apply_read_floor(counts, min_reads)
    if counts.empty:
        raise ValueError(f"no observation has at least {min_reads} reads")

    samples = samples[samples["sample"].isin(counts["sample"])].reset_index(drop=True)
    sample_index = pd.Series(np.arange(len(samples)), index=samples["sample"])

    measured = counts[["bait", "prey"]].drop_duplicates().sort_values(["bait", "prey"])
    measured = measured.reset_index(drop=True)
    ends = np.sort(measured[["bait", "prey"]].to_numpy(dtype=str), axis=1)
    unique, canonical_index = np.unique(ends, axis=0, return_inverse=True)
    measured["canonical_index"] = canonical_index.ravel()
    canonical = pd.DataFrame(unique, columns=["protein_a", "protein_b"])

    measured_index = pd.Series(
        np.arange(len(measured)),
        index=pd.MultiIndex.from_frame(measured[["bait", "prey"]]),
    )
    counts["sample_index"] = sample_index[counts["sample"]].to_numpy()
    counts["measured_index"] = measured_index[
        pd.MultiIndex.from_frame(counts[["bait", "prey"]])
    ].to_numpy()
    counts["canonical_index"] = measured["canonical_index"].to_numpy()[
        counts["measured_index"]
    ]

    return Prepared(
        counts=counts,
        samples=samples,
        measured=measured,
        canonical=canonical,
        drug_design=check_design(samples, drug),
        cell_design=check_design(samples, cell),
    )
