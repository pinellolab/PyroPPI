"""Fit PyroPPI to every dataset and write one directory of results per fit.

Each entry below names the dataset directory under ``--data``, the indicator
columns of its ``samples.csv`` that form the drug and cell blocks, whether the
fit estimates a pair baseline, and how many SVI steps it takes to converge.

    python scripts/fit_all.py
    python scripts/fit_all.py --dataset zf_pom
    python scripts/fit_all.py --data /path/to/data --results /path/to/results
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from pyroppi.data import read_dataset
from pyroppi.model import fit

REPOSITORY = Path(__file__).resolve().parent.parent

DATASETS = [
    {
        "name": "rad51x9",
        "drug": ["1h", "3h", "HCT_x_1h", "HCT_x_3h"],
        "cell": ["HCT"],
        "baseline": True,
        "steps": 30_000,
    },
    {
        "name": "clinorf",
        "drug": ["1h", "3h"],
        "cell": [],
        "baseline": True,
        "steps": 30_000,
    },
    {
        # Every sample is treated with pomalidomide, so there is no untreated
        # condition to estimate a pair baseline from.
        "name": "zf_pom",
        "drug": ["POM_300_NM", "POM_3_UM"],
        "cell": [],
        "baseline": False,
        "steps": 60_000,
    },
    {
        "name": "localization",
        "drug": ["3H"],
        "cell": [],
        "baseline": True,
        "steps": 60_000,
    },
    {
        "name": "dms_rad51d",
        "drug": ["3h", "U2OS_x_3h"],
        "cell": ["U2OS"],
        "baseline": True,
        "steps": 60_000,
    },
    {
        "name": "dms_xrcc2",
        "drug": ["3h", "U2OS_x_3h"],
        "cell": ["U2OS"],
        "baseline": True,
        "steps": 60_000,
    },
    {
        "name": "dms_rad51c",
        "drug": [],
        "cell": [],
        "baseline": True,
        "steps": 60_000,
    },
    {
        "name": "dms_xrcc3",
        "drug": [],
        "cell": [],
        "baseline": True,
        "steps": 60_000,
    },
    {
        "name": "commander",
        "drug": ["Rap1h"],
        "cell": ["HCT"],
        "baseline": True,
        "steps": 60_000,
    },
    {
        "name": "pulldown_rad51d",
        "drug": [],
        "cell": [],
        "baseline": True,
        "steps": 60_000,
    },
    {
        "name": "pulldown_xrcc2",
        "drug": [],
        "cell": [],
        "baseline": True,
        "steps": 60_000,
    },
]


def write_runtimes(path: Path, fitted: list[dict]) -> None:
    """Record each fit's runtime, keeping the rows of the datasets that did not run.

    A run of one dataset replaces that dataset's row and leaves the others
    standing, so the file always describes the fits in ``results/``. Rows are
    written in the registry's order, with any name the registry no longer holds
    kept at the end.
    """
    order = {dataset["name"]: index for index, dataset in enumerate(DATASETS)}
    rows = {row["dataset"]: row for row in fitted}
    if path.exists():
        previous = pd.read_csv(path).to_dict("records")
        rows = {row["dataset"]: row for row in previous} | rows
    ranked = sorted(rows.values(), key=lambda row: order.get(row["dataset"], len(order)))
    pd.DataFrame(ranked).to_csv(path, index=False)


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
        help="directory to write the fits into (default: the repository's results/)",
    )
    parser.add_argument("--dataset", help="fit this dataset only, instead of all of them")
    parser.add_argument("--steps", type=int, help="number of SVI steps, overriding the table")
    args = parser.parse_args()

    chosen = [d for d in DATASETS if args.dataset in (None, d["name"])]
    if not chosen:
        names = ", ".join(d["name"] for d in DATASETS)
        parser.error(f"unknown dataset {args.dataset!r}; choose one of: {names}")

    runtimes = []
    for dataset in chosen:
        name = dataset["name"]
        steps = args.steps or dataset["steps"]
        print(f"\n{name}: {steps} steps, drug={dataset['drug']}, cell={dataset['cell']}")

        counts, samples = read_dataset(args.data / name)
        start = time.perf_counter()
        result = fit(
            counts,
            samples,
            drug=dataset["drug"],
            cell=dataset["cell"],
            baseline=dataset["baseline"],
            num_steps=steps,
        )
        seconds = time.perf_counter() - start
        result.save(args.results / name)

        print(f"{name}: {seconds / 60:.1f} min, written to {args.results / name}")
        runtimes.append({"dataset": name, "steps": steps, "seconds": round(seconds, 1)})

    args.results.mkdir(parents=True, exist_ok=True)
    write_runtimes(args.results / "runtimes.csv", runtimes)


if __name__ == "__main__":
    main()
