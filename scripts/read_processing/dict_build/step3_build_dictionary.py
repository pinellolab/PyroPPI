"""Dictionary build step 3: one construct per barcode pair, with error correction.

Step 2's construct map is one row per read, `bc1_bc2[_suffix],construct_id`.
That is aggregated into counts of (bc1, bc2) -> construct -> reads, and then
each pair lends its counts to the pairs one substitution away from it, in
either barcode, that the data already contains: a barcode misread in one base
otherwise splits a construct's reads across two entries.

Each pair is finally resolved to its best-supported construct and written as
`bc1_bc2,construct_id,best_reads,total_reads,purity`, keeping only the pairs
that clear the support and purity thresholds.

This merges the former step3a (aggregate), step3b (neighbour map) and step3c
(redistribute) Perl scripts. The Perl step3c loaded the second barcode's
neighbours into the wrong hash; the bc1 and bc2 neighbour maps are separate
here.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASES = ("A", "C", "G", "T")

# bc1 -> bc2 -> construct -> read count, at any of the three stages below
BarcodeCounts = Mapping[str, Mapping[str, Mapping[str, int]]]


def open_text(path: str):
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rt", newline="")
    return open(path, newline="")


def open_out(path: str):
    if path.lower().endswith(".gz"):
        return gzip.open(path, "wt", newline="")
    return open(path, "w", newline="")


def write_raw_dict_csv(counts: BarcodeCounts, path: str) -> None:
    """Pre-redistribution counts as CSV: bc1, bc2, construct_id, read_count.

    Gzipped when the path ends with .gz, e.g. .csv.gz.
    """
    logger.info("Writing raw dictionary CSV: %s", path)
    with open_out(path) as fh:
        writer = csv.writer(fh, delimiter=",", lineterminator="\n")
        for bc1 in sorted(counts.keys()):
            for bc2 in sorted(counts[bc1].keys()):
                for construct in sorted(counts[bc1][bc2].keys()):
                    writer.writerow([bc1, bc2, construct, counts[bc1][bc2][construct]])


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a barcode→construct dictionary with HD1 neighbor redistribution "
            "(step 3 merged)."
        )
    )
    parser.add_argument(
        "--construct_map_in",
        required=True,
        metavar="PATH",
        help="Construct map CSV: read_id (bc1_bc2[_suffix]), construct_id; gzip ok",
    )
    parser.add_argument(
        "--out",
        "-o",
        required=True,
        help="Output filtered barcode→construct table as CSV (gzip if path ends with .gz)",
    )
    parser.add_argument(
        "--min_support_reads",
        type=int,
        default=3,
        help="Minimum read support for the best construct (default: 3)",
    )
    parser.add_argument(
        "--min_accuracy",
        type=float,
        default=0.9,
        help="Minimum best_construct_count / total_count (default: 0.9)",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="CPU workers for HD1 neighbor search (default: 1, single core)",
    )
    ignore_group = parser.add_mutually_exclusive_group()
    ignore_group.add_argument(
        "--ignore_bc1",
        action="store_true",
        help="Key all rows with bc1=\"\"; skip bc1 HD1 neighbors and redistribution along bc1",
    )
    ignore_group.add_argument(
        "--ignore_bc2",
        action="store_true",
        help="Key all rows with bc2=\"\"; skip bc2 HD1 neighbors and redistribution along bc2",
    )
    parser.add_argument(
        "--raw_dict_out",
        metavar="PATH",
        default=None,
        help="Optional raw dict CSV: bc1, bc2, construct_id, read_count. Use .csv.gz "
        "(or any path ending in .gz) for gzip",
    )
    return parser.parse_args()


def load_raw_dict(
    path: str,
    *,
    ignore_bc1: bool = False,
    ignore_bc2: bool = False,
) -> tuple[
    defaultdict[str, defaultdict[str, defaultdict[str, int]]],
    set[str],
    set[str],
]:
    """
    Step 3a equivalent: aggregate (bc1, bc2) -> construct -> count.
    Each CSV line: first field = read id with bc1_bc2 or bc1_bc2_<suffix>, second is
    the construct id.
    If ignore_bc1, bc1 is stored as "". If ignore_bc2, bc2 is stored as "".
    """
    counts: defaultdict[str, defaultdict[str, defaultdict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    bc1_set: set[str] = set()
    bc2_set: set[str] = set()
    total_lines = 0
    valid_lines = 0
    started = time.time()
    with open_text(path) as f:
        reader = csv.reader(f)
        for row in reader:
            total_lines += 1
            if len(row) < 2:
                continue
            read_id, construct = row[0], row[1]
            parts = read_id.split("_")
            if len(parts) < 2:
                continue
            bc1, bc2 = parts[0], parts[1]
            if ignore_bc1:
                bc1 = ""
            if ignore_bc2:
                bc2 = ""
            bc1_set.add(bc1)
            bc2_set.add(bc2)
            counts[bc1][bc2][construct] += 1
            valid_lines += 1
            if total_lines % 1_000_000 == 0:
                logger.info(
                    "Loaded CSV lines: %s (valid %s), %.2fs",
                    total_lines,
                    valid_lines,
                    time.time() - started,
                )
                started = time.time()
    logger.info(
        "CSV total lines: %s, valid: %s, unique bc1: %s, unique bc2: %s",
        total_lines,
        valid_lines,
        len(bc1_set),
        len(bc2_set),
    )
    return counts, bc1_set, bc2_set


def _neighbors_of(barcodes: Iterable[str], present: frozenset[str]) -> dict[str, set[str]]:
    """HD1 neighbors of each barcode, restricted to barcodes the data contains.

    Enumerating the 3L substitutions of a barcode and testing each for membership
    is cheaper than comparing every barcode against every other one.
    """
    neighbors: dict[str, set[str]] = {}
    for bc in barcodes:
        found: set[str] = set()
        for i, base in enumerate(bc):
            for alt in BASES:
                if alt == base:
                    continue
                candidate = bc[:i] + alt + bc[i + 1 :]
                if candidate in present:
                    found.add(candidate)
        neighbors[bc] = found
    return neighbors


def _neighbors_chunk_subset(args: tuple[frozenset[str], tuple[str, ...]]) -> dict[str, set[str]]:
    """Worker: HD1 neighbors for a subset of barcodes; membership uses full set for that length."""
    present, chunk = args
    return _neighbors_of(chunk, present)


def build_hd1_neighbors(barcodes: Iterable[str], jobs: int) -> dict[str, set[str]]:
    """Step 3b equivalent: map each barcode -> the set of its HD1 neighbors in the full set.

    Barcodes are bucketed by length first: a substitution never changes the
    length, so barcodes of different lengths can never be neighbors.
    """
    by_length: defaultdict[int, list[str]] = defaultdict(list)
    for bc in barcodes:
        by_length[len(bc)].append(bc)

    merged: dict[str, set[str]] = {}
    if not by_length:
        return merged

    jobs = max(1, jobs)
    chunk_tasks: list[tuple[frozenset[str], tuple[str, ...]]] = []

    for seqs in by_length.values():
        unique = tuple(sorted(set(seqs)))
        n_barcodes = len(unique)
        if n_barcodes == 0:
            continue
        present = frozenset(unique)
        # Small buckets: serial; large: split for worker pool
        if jobs == 1 or n_barcodes < 5000:
            merged.update(_neighbors_of(unique, present))
            continue

        n_chunks = min(jobs, max(1, n_barcodes // 2500))
        chunk_size = (n_barcodes + n_chunks - 1) // n_chunks
        for i in range(0, n_barcodes, chunk_size):
            chunk_tasks.append((present, unique[i : i + chunk_size]))

    if chunk_tasks:
        with mp.Pool(jobs) as pool:
            for part in pool.imap_unordered(
                _neighbors_chunk_subset,
                chunk_tasks,
                chunksize=max(1, len(chunk_tasks) // (jobs * 4)),
            ):
                merged.update(part)

    return merged


def redistribute_counts(
    raw_counts: BarcodeCounts,
    bc1_neighbors: Mapping[str, set[str]],
    bc2_neighbors: Mapping[str, set[str]],
) -> dict[str, dict[str, dict[str, int]]]:
    """Step 3c redistribution: add source counts onto HD1-neighbor pairs that exist in the dict.

    Counts are read from the untouched original and added to a copy, so that a
    pair never lends counts it was itself lent: every pair gives away exactly
    what it was observed with. Only neighbor pairs already present in the data
    receive anything, which is what keeps this from inventing barcodes.
    """
    accumulated: dict[str, dict[str, dict[str, int]]] = {
        bc1: {bc2: dict(constructs) for bc2, constructs in bc2_counts.items()}
        for bc1, bc2_counts in raw_counts.items()
    }
    raw_bc_pairs = 0
    alt_bc_pairs = 0
    started = time.time()

    for bc1, bc2_counts in raw_counts.items():
        bc1_alts = bc1_neighbors.get(bc1, set())
        for bc2, source in bc2_counts.items():
            raw_bc_pairs += 1
            bc2_alts = bc2_neighbors.get(bc2, set())

            for bc1_alt in bc1_alts:
                if bc2 in accumulated.get(bc1_alt, {}):
                    alt_bc_pairs += 1
                    for construct, count in source.items():
                        accumulated[bc1_alt][bc2][construct] = (
                            accumulated[bc1_alt][bc2].get(construct, 0) + count
                        )
                for bc2_alt in bc2_alts:
                    if bc1_alt in accumulated and bc2_alt in accumulated[bc1_alt]:
                        alt_bc_pairs += 1
                        for construct, count in source.items():
                            accumulated[bc1_alt][bc2_alt][construct] = (
                                accumulated[bc1_alt][bc2_alt].get(construct, 0) + count
                            )

            for bc2_alt in bc2_alts:
                if bc1 in accumulated and bc2_alt in accumulated[bc1]:
                    alt_bc_pairs += 1
                    for construct, count in source.items():
                        accumulated[bc1][bc2_alt][construct] = (
                            accumulated[bc1][bc2_alt].get(construct, 0) + count
                        )

            if raw_bc_pairs % 1_000_000 == 0:
                logger.info(
                    "Redistribute progress: raw_pairs=%s alt_ops=%s %.2fs",
                    raw_bc_pairs,
                    alt_bc_pairs,
                    time.time() - started,
                )
                started = time.time()

    logger.info(
        "Redistribution done: raw barcode pairs: %s, alt update ops: %s",
        raw_bc_pairs,
        alt_bc_pairs,
    )
    return accumulated


def filter_and_write(
    counts: BarcodeCounts,
    out_path: str,
    min_support_reads: int,
    min_accuracy: float,
) -> None:
    """Write high-confidence pairs as CSV: bc_pair, construct_id, best_ct, total_ct, purity.

    A pair whose top construct is tied with another is dropped outright, before
    the thresholds are applied and before it reaches the support distribution:
    its reads do not say which construct the barcode carries, and no amount of
    support would settle it.

    Reads identical to the reference carry no mismatch for step 2 to match on
    and never arrive here, so what is written out is purity among the reads that
    did carry a construct, not among all reads of that barcode pair.
    """
    total_bc_pairs = 0
    valid_bc_pairs = 0
    support_dist: defaultdict[int, int] = defaultdict(int)

    with open_out(out_path) as fh:
        writer = csv.writer(fh, delimiter=",", lineterminator="\n")
        for bc1 in sorted(counts.keys()):
            for bc2 in sorted(counts[bc1].keys()):
                total_bc_pairs += 1
                construct_counts = counts[bc1][bc2]
                best_construct = ""
                best_ct = 0
                total_ct = 0
                for construct in sorted(construct_counts.keys()):
                    count = construct_counts[construct]
                    total_ct += count
                    if count > best_ct:
                        best_construct = construct
                        best_ct = count

                n_tie = sum(1 for count in construct_counts.values() if count == best_ct)
                if n_tie != 1:
                    continue
                support_dist[best_ct] += 1
                accuracy = best_ct / total_ct if total_ct else 0.0
                if best_ct >= min_support_reads and accuracy >= min_accuracy:
                    valid_bc_pairs += 1
                    writer.writerow(
                        [
                            f"{bc1}_{bc2}",
                            best_construct,
                            best_ct,
                            total_ct,
                            accuracy,
                        ]
                    )

    logger.info(
        "Output: total barcode pairs: %s, passing filters: %s",
        total_bc_pairs,
        valid_bc_pairs,
    )
    logger.info("Barcode pair best-support read distribution (unique max):")
    for support in sorted(support_dist.keys()):
        logger.info("  %s\t%s", support, support_dist[support])


def main() -> None:
    args = parse_args()
    jobs = max(1, args.jobs)

    if not os.path.isfile(args.construct_map_in):
        logger.error("Input not found: %s", args.construct_map_in)
        sys.exit(1)

    logger.info("Loading the construct map: %s", args.construct_map_in)
    raw_counts, bc1_set, bc2_set = load_raw_dict(
        args.construct_map_in,
        ignore_bc1=args.ignore_bc1,
        ignore_bc2=args.ignore_bc2,
    )
    n_pairs = sum(len(bc2_counts) for bc2_counts in raw_counts.values())
    logger.info("Raw (bc1, bc2) pairs with data: %s", n_pairs)

    if args.raw_dict_out:
        write_raw_dict_csv(raw_counts, args.raw_dict_out)

    # An ignored coordinate is keyed as "" for every row, so it has no neighbors
    # and nothing to redistribute along.
    if args.ignore_bc1:
        logger.info("Skipping bc1 HD1 neighbors (--ignore_bc1)")
        bc1_neighbors: dict[str, set[str]] = {}
    else:
        logger.info(
            "Building HD1 neighbor map for %s bc1 barcodes (%s workers)", len(bc1_set), jobs
        )
        started = time.time()
        bc1_neighbors = build_hd1_neighbors(bc1_set, jobs)
        logger.info("bc1 HD1 map done in %.2fs", time.time() - started)

    if args.ignore_bc2:
        logger.info("Skipping bc2 HD1 neighbors (--ignore_bc2)")
        bc2_neighbors: dict[str, set[str]] = {}
    else:
        logger.info(
            "Building HD1 neighbor map for %s bc2 barcodes (%s workers)", len(bc2_set), jobs
        )
        started = time.time()
        bc2_neighbors = build_hd1_neighbors(bc2_set, jobs)
        logger.info("bc2 HD1 map done in %.2fs", time.time() - started)

    logger.info("Redistributing counts across HD1 neighbors...")
    started = time.time()
    accumulated = redistribute_counts(raw_counts, bc1_neighbors, bc2_neighbors)
    logger.info("Redistribution finished in %.2fs", time.time() - started)

    logger.info(
        "Writing filtered dictionary to %s (min_support_reads>=%s, min_accuracy>=%s)",
        args.out,
        args.min_support_reads,
        args.min_accuracy,
    )
    filter_and_write(
        accumulated,
        args.out,
        args.min_support_reads,
        args.min_accuracy,
    )


if __name__ == "__main__":
    main()
