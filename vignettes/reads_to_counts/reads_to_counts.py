"""Take raw sequencing reads through to PyroPPI's counts table.

The other vignettes start at `counts.csv`. This one produces it: a
barcode-to-construct dictionary out of dictionary reads, applied to cDNA reads
to count edits per construct, shaped into `counts.csv` and `samples.csv`.

Reads come from `scripts/read_processing/example_data`, two datasets subset
from the production RAD51C scan. Each ships the output the pipeline produces,
and the last step compares this run against it.

`reads_to_counts.ipynb` is the same run with the commentary.

Needs pysam, from the `reads` extra, with bowtie2 and samtools on the PATH.
Nothing is fitted, so numpyro and jax are unused.

    python reads_to_counts.py                      # everything in a temp directory
    python reads_to_counts.py --work /tmp/demo     # keep the intermediates
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PIPELINE = REPO / "scripts" / "read_processing"
EXAMPLES = PIPELINE / "example_data"

# Okabe-Ito, the colourblind-safe palette the manuscript figures use.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
AMBER = "#E69F00"
GREY = "#6F6F6F"
PALE = "#D9D9D9"

CLASS_COLOR = {"synonymous": GREEN, "missense": BLUE, "stop": VERMILLION}

# Read structure, shared by both datasets. The cDNA edit template is the same
# constant sequence that separates bc1 from bc2 in a dictionary R1.
# These values come from the production run described in the Methods.
LINKER = "TATCATACggccctgcagacatgggtgatcctcatgttggccTGTAGTACC"
POST_BC2_DICT = "ACCTTTCGACCT"
POST_BC2_CDNA = "ACCTTTCGACCTGCA"
PRIMARY_POS = 14  # the on-target adenine, 1-based in the edit template
EDIT_MODE = "T2C"
BC1_MIN, BC1_MAX = 12, 15

# The cDNA sample driving both datasets, and the partner it was measured
# against. One biological sample, parsed once per block.
SAMPLE = "NR-TV_HCT_RAD51C_XRCC3_Rep1"
CDNA_FASTQ = f"{SAMPLE}__R1.fastq.gz"
BAIT = "XRCC3"

# The two datasets differ in which block of RAD51C the library tiles and in the
# step 3 cutoffs their cloning calls for. The NEBuilder construct puts several
# constructs behind one barcode, so a barcode there is kept on a plurality of
# its reads and its purity cutoff sits at 0.10.
DATASETS = {
    "Dual_NEBuilder": {
        "reference": "RAD51C_blk_1",
        "prefix": "RAD51C_DMS_1",
        "min_support": 5,
        "min_purity": 0.10,
    },
    "regular": {
        "reference": "RAD51C_blk_3",
        "prefix": "RAD51C_DMS_3",
        "min_support": 3,
        "min_purity": 0.90,
    },
}

DICT_COLUMNS = ["barcode", "construct", "support", "reads", "purity"]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


# --------------------------------------------------------------------------
# running the pipeline scripts
# --------------------------------------------------------------------------


def run(command: list[str], log: Path, announce: str | None = None) -> str:
    """Run one pipeline step, tee its chatter to a log file, and return it."""
    print(f"  {announce or Path(command[1]).name}")
    done = subprocess.run(command, capture_output=True, text=True)
    chatter = (done.stdout or "") + (done.stderr or "")
    with open(log, "a") as sink:
        sink.write(f"$ {' '.join(command)}\n{chatter}\n")
    if done.returncode != 0:
        raise RuntimeError(f"{command[1]} failed; see {log}\n{chatter[-2000:]}")
    return chatter


def python_step(script: Path, *args: str, log: Path) -> str:
    return run([sys.executable, str(script), *args], log)


def requirements() -> dict[str, str | None]:
    """What this vignette needs beyond PyroPPI itself.

    pip does not install bowtie2 or samtools. pysam, which step 2 reads the BAM
    through, comes with the `reads` extra.
    """
    found: dict[str, str | None] = {
        name: shutil.which(name) for name in ("bowtie2", "bowtie2-build", "samtools")
    }
    found["pysam"] = (
        "installed" if importlib.util.find_spec("pysam") is not None else None
    )
    return found


def prepare_reference(work: Path, log: Path) -> tuple[Path, Path]:
    """Copy the reference tables aside and build the two indexes they need.

    Step 2 reads the FASTA through pysam, which wants a `samtools faidx` index
    beside it. The bowtie2 index is regenerable, so the repository omits it.
    """
    ref = work / "ref"
    ref.mkdir(parents=True, exist_ok=True)
    for name in ("reference_sequences.fa", "r2_fixed_seq.csv", "construct_library.csv"):
        shutil.copy(PIPELINE / "ref" / name, ref / name)
    fasta = ref / "reference_sequences.fa"
    run(["samtools", "faidx", str(fasta)], log, announce="samtools faidx")
    index = ref / "reference_index"
    run(["bowtie2-build", "--threads", "4", str(fasta), str(index)], log,
        announce="bowtie2-build")
    return ref, index


def build_dictionary(name: str, work: Path, ref: Path, index: Path) -> tuple[Path, dict]:
    """Dictionary reads to a barcode-to-construct table, in four steps.

    Pull bc1, bc2 and the construct sequence out of the read pair. Align the
    construct sequence to its reference. Call which library construct each read
    carries, from the mismatches it holds. Collapse the reads of one barcode to
    the construct most of them agree on, subject to the two cutoffs.
    """
    spec = DATASETS[name]
    data = EXAMPLES / name / "dict_build"
    out = work / name / "dictionary"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "pipeline.log"
    fasta = ref / "reference_sequences.fa"
    prefix = spec["prefix"]
    tally: dict[str, float] = {}

    oligos = out / f"{prefix}.fasta.gz"
    report = python_step(
        PIPELINE / "dict_build" / "step1_extract_barcodes.py",
        "--r1_fastq", str(data / f"{prefix}_Dictionary_R1.fastq.gz"),
        "--r2_fastq", str(data / f"{prefix}_Dictionary_R2.fastq.gz"),
        "--reference", spec["reference"],
        "--r1_linker", LINKER,
        "--r2_fixed_seq_lib", str(ref / "r2_fixed_seq.csv"),
        "--bc1_min_len", str(BC1_MIN), "--bc1_max_len", str(BC1_MAX),
        "--bc2_min_len", "2", "--bc2_max_len", "7",
        "--r1_post_bc2", POST_BC2_DICT,
        "--out", str(oligos),
        log=log,
    )
    tally.update(parse_step1_report(report))

    # Align, then sort. Step 2 reuses a .bai found beside the BAM without
    # checking it, so any left from an earlier run is deleted below.
    bam = out / f"{prefix}.bam"
    print("  bowtie2 | samtools sort")
    with open(log, "a") as sink:
        aligner = subprocess.Popen(
            ["bowtie2", "-p", "4", "--local", "--very-sensitive-local", "--no-unal",
             "-x", str(index), "-f", str(oligos)],
            stdout=subprocess.PIPE, stderr=sink,
        )
        view = subprocess.Popen(
            ["samtools", "view", "-bS", "-"],
            stdin=aligner.stdout, stdout=subprocess.PIPE, stderr=sink,
        )
        aligner.stdout.close()
        subprocess.run(["samtools", "sort", "-", "-o", str(bam)],
                       stdin=view.stdout, check=True, stderr=sink)
        view.stdout.close()
        for process in (aligner, view):
            if process.wait() != 0:
                raise RuntimeError(f"alignment failed; see {log}")
    for stale in out.glob("*.bai"):
        stale.unlink()

    mapped = out / f"{prefix}.construct_mapped.csv.gz"
    report = python_step(
        PIPELINE / "dict_build" / "step2_map_constructs.py",
        "--bam_in", str(bam),
        "--fasta_in", str(fasta),
        "--construct_lib", str(ref / "construct_library.csv"),
        "--reference", spec["reference"],
        "--out", str(mapped),
        log=log,
    )
    found = re.search(r"mapped to exactly one construct: (\d+)", report)
    tally["construct called"] = int(found.group(1)) if found else np.nan

    dictionary = out / "barcode_to_construct.csv.gz"
    python_step(
        PIPELINE / "dict_build" / "step3_build_dictionary.py",
        "--construct_map_in", str(mapped),
        "--raw_dict_out", str(out / "raw_dict.csv.gz"),
        "--out", str(dictionary),
        "--ignore_bc2",
        "--min_support_reads", str(spec["min_support"]),
        "--min_accuracy", str(spec["min_purity"]),
        "--jobs", "4",
        log=log,
    )
    return dictionary, tally


def count_edits(name: str, dictionary: Path, work: Path) -> tuple[Path, dict]:
    """Apply the dictionary to the cDNA sample and count edits per construct.

    A base editor can edit the barcode itself, so step 1 expands each barcode
    into every edited form it could take. Step 2 anchors the template in each
    read, calls an edit at every position that can carry one, and credits the
    read to the barcode's construct.
    """
    out = work / name / "cdna"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "pipeline.log"
    prefix = out / "alts"

    python_step(
        PIPELINE / "cdna_edit_count" / "step1_expand_barcodes.py",
        "--dict_in", str(dictionary),
        "--edit_mode", EDIT_MODE,
        "--out_prefix", str(prefix),
        "--ignore_bc2",
        log=log,
    )

    parsed = out / f"{SAMPLE}_.cDNA_parsed.csv"
    report = python_step(
        PIPELINE / "cdna_edit_count" / "step2_count_edits.py",
        "--cDNA_in", str(EXAMPLES / name / "cdna_edit_count" / CDNA_FASTQ),
        "--dict_in", str(dictionary),
        "--bc_alt_prefix", str(prefix),
        "--out", str(parsed),
        "--ignore_bc2",
        "--edit_mode", EDIT_MODE,
        "--template_seq", LINKER,
        "--regex_match_len", "8",
        "--r1_post_bc2", POST_BC2_CDNA,
        "--primary_pos", str(PRIMARY_POS),
        "--bc1_min_len", str(BC1_MIN), "--bc1_max_len", str(BC1_MAX),
        log=log,
    )
    return parsed, parse_report_line(report)


# --------------------------------------------------------------------------
# reading what the pipeline wrote
# --------------------------------------------------------------------------


def parse_step1_report(chatter: str) -> dict[str, float]:
    """Step 1 prints a two-line tab-separated report: a header and one row."""
    rows = [line.split("\t") for line in chatter.splitlines() if "\t" in line]
    if len(rows) < 2:
        return {}
    fields = dict(zip(rows[0], rows[1]))
    return {
        "dictionary reads": float(fields["Total_reads"]),
        "barcode + construct found": float(fields["both_found"]),
    }


def parse_report_line(chatter: str) -> dict[str, float]:
    """The cDNA step 2 report line: a path, then alternating key and value."""
    for line in chatter.splitlines():
        if line.startswith("Report_line"):
            fields = line.split("\t")[2:]
            return {k: float(v) for k, v in zip(fields[0::2], fields[1::2])}
    return {}


def read_dictionary(path: Path) -> pd.DataFrame:
    """Step 3 writes no header: barcode pair, construct, support, reads, purity."""
    table = pd.read_csv(path, header=None, names=DICT_COLUMNS)
    table["bc1"] = table.barcode.str.split("_").str[0]
    return table


def read_raw_dictionary(path: Path, min_support: int, min_purity: float) -> pd.DataFrame:
    """Every barcode's support and purity before the cutoffs ran.

    Step 3 writes this on the way past, one row per barcode and construct. The
    two cutoffs are computed from these counts.
    """
    raw = pd.read_csv(path, header=None, names=["bc1", "bc2", "construct", "reads"])
    grouped = raw.groupby("bc1").reads
    table = pd.DataFrame({"support": grouped.max(), "reads": grouped.sum()})
    table["purity"] = table.support / table.reads
    table["kept"] = (table.support >= min_support) & (table.purity >= min_purity)
    return table.reset_index()


def classify(construct: str) -> str:
    """synonymous, missense or stop, from a `GENE---R109Q`-style identifier."""
    change = construct.split("---")[-1]
    found = re.fullmatch(r"([A-Z])(\d+)([A-Z*])", change)
    if found is None:
        return "other"
    reference, _, alternate = found.groups()
    if alternate == "*":
        return "stop"
    return "synonymous" if alternate == reference else "missense"


# --------------------------------------------------------------------------
# the check against the committed expected output
# --------------------------------------------------------------------------


def lines_of(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def verify(label: str, produced: Path, expected: Path) -> bool:
    """Compare a file this run produced against the committed answer, line by line."""
    ours, theirs = lines_of(produced), lines_of(expected)
    if ours == theirs:
        print(f"  match    {label}: {len(ours)} rows identical to {expected.name}")
        return True
    print(f"  MISMATCH {label}: {len(ours)} rows produced, {len(theirs)} expected")
    for index, (mine, yours) in enumerate(zip(ours, theirs), start=1):
        if mine != yours:
            print(f"      row {index} produced: {mine[:120]}")
            print(f"      row {index} expected: {yours[:120]}")
            break
    return False


# --------------------------------------------------------------------------
# shaping
# --------------------------------------------------------------------------


def shape_counts(parsed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The parsed edit counts, as the two tables PyroPPI reads.

    `total` is every read credited to a construct. `edited` is how many of them
    carry the on-target edit, the column `Primary_edit_ct`. XRCC3 is the partner
    throughout, so a measured pair is one RAD51C construct against it.
    """
    counts = pd.DataFrame(
        {
            "sample": SAMPLE,
            "bait": BAIT,
            "prey": parsed.Outcome.str.replace("---", ":", regex=False),
            "total": parsed.Total_reads,
            "edited": parsed.Primary_edit_ct,
        }
    ).sort_values(["sample", "prey"], ignore_index=True)

    samples = pd.DataFrame(
        {
            "sample": [SAMPLE],
            "cell_line": ["HCT116"],
            "batch": ["041726"],
            "replicate": [1],
        }
    )
    return counts, samples


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def plot_attrition(tallies: dict[str, dict], path: Path) -> None:
    """Where the reads go, through each half of the pipeline."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.8))

    dictionary_steps = ["dictionary reads", "barcode + construct found", "construct called"]
    # Reads_barcode_matched counts barcode-to-construct matches. A barcode
    # mapping to several constructs credits each one, so that field can exceed
    # Reads_total and is left off this axis.
    cdna_steps = [
        "Reads_total",
        "Reads_all_edit_pos_found",
        "Reads_all_edit_pos_found_barcode_matched",
    ]
    cdna_labels = ["cDNA reads", "template parsed", "+ barcode matched"]
    width = 0.38

    for offset, (name, tally) in zip((-width / 2, width / 2), tallies.items()):
        x = np.arange(len(dictionary_steps))
        left.bar(x + offset, [tally.get(k, np.nan) for k in dictionary_steps],
                 width, label=name, color=BLUE if offset < 0 else AMBER)
        x = np.arange(len(cdna_steps))
        right.bar(x + offset, [tally.get(k, np.nan) for k in cdna_steps],
                  width, label=name, color=BLUE if offset < 0 else AMBER)

    left.set_xticks(np.arange(len(dictionary_steps)))
    left.set_xticklabels(["reads", "bc + construct", "construct called"],
                         rotation=20, ha="right")
    left.set_ylabel("reads")
    left.set_title("dictionary half")
    left.legend(fontsize=6)

    right.set_xticks(np.arange(len(cdna_steps)))
    right.set_xticklabels(cdna_labels, rotation=20, ha="right")
    right.set_ylabel("reads")
    right.set_title("cDNA half")
    right.legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_dictionary(raws: dict[str, pd.DataFrame], path: Path) -> None:
    """Support and purity for every barcode, with the cutoffs applied to them."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    rng = np.random.default_rng(0)

    for ax, (name, raw) in zip(axes, raws.items()):
        spec = DATASETS[name]
        for kept, color, label in ((True, BLUE, "kept"), (False, PALE, "filtered out")):
            part = raw[raw.kept == kept]
            if not len(part):
                continue
            spread = np.exp(rng.normal(0, 0.04, len(part)))
            ax.scatter(part.support * spread, part.purity, s=14, c=color,
                       edgecolor="white", linewidth=0.3, label=label)
        ax.axhline(spec["min_purity"], color=VERMILLION, lw=0.8, ls="--")
        ax.axvline(spec["min_support"], color=VERMILLION, lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("reads supporting the best construct")
        ax.set_title(f"{name}  (support >= {spec['min_support']}, "
                     f"purity >= {spec['min_purity']:g})")
        ax.legend(fontsize=6, loc="lower right")
    axes[0].set_ylabel("purity")
    axes[0].set_ylim(0, 1.06)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_edit_rates(counts: pd.DataFrame, path: Path) -> None:
    """The on-target edit rate per construct, and the depth it rests on."""
    table = counts[counts.total > 0].copy()
    table["rate"] = table.edited / table.total
    table["class"] = table.prey.str.replace(":", "---", regex=False).map(classify)
    table = table.sort_values("rate", ascending=False).reset_index(drop=True)

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.0))

    colors = [CLASS_COLOR.get(k, GREY) for k in table["class"]]
    left.bar(np.arange(len(table)), table.rate, color=colors)
    left.set_xticks(np.arange(len(table)))
    left.set_xticklabels(table.prey.str.split(":").str[-1], rotation=90, fontsize=5)
    left.set_ylabel("edited / total")
    left.set_title("on-target edit rate, per construct")
    for name, color in CLASS_COLOR.items():
        left.bar([], [], color=color, label=name)
    left.legend(fontsize=6)

    right.scatter(table.total, table.rate, s=18, c=colors,
                  edgecolor="white", linewidth=0.3)
    right.set_xscale("log")
    right.set_xlabel("reads credited to the construct")
    right.set_ylabel("edited / total")
    right.set_title("edit rate against read depth")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work",
        type=Path,
        default=None,
        help="directory for the intermediates (default: a temp directory)",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=HERE,
        help="directory to write the PNG figures into",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        choices=sorted(DATASETS),
        help="run one dataset instead of both",
    )
    return parser


def main():
    args = build_parser().parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)

    needed = requirements()
    missing = [name for name, found in needed.items() if found is None]
    if missing:
        print(f"not found on this machine: {', '.join(missing)}")
        print("pysam comes with `pip install -e '.[reads]'`; bowtie2 and samtools "
              "are external programs that pip does not install")
        raise SystemExit(1)

    holder = None
    if args.work is None:
        holder = tempfile.TemporaryDirectory(prefix="pyroppi-reads-")
        work = Path(holder.name)
    else:
        work = args.work
        work.mkdir(parents=True, exist_ok=True)

    chosen = [args.dataset] if args.dataset else list(DATASETS)

    try:
        print("preparing the reference")
        ref, index = prepare_reference(work, work / "reference.log")

        tallies: dict[str, dict] = {}
        raws: dict[str, pd.DataFrame] = {}
        parsed_tables = []
        agreed = True

        for name in chosen:
            spec = DATASETS[name]
            print(f"\n{name}: building the barcode dictionary")
            dictionary_path, tally = build_dictionary(name, work, ref, index)
            dictionary = read_dictionary(dictionary_path)
            raws[name] = read_raw_dictionary(
                dictionary_path.parent / "raw_dict.csv.gz",
                spec["min_support"], spec["min_purity"],
            )
            print(f"  {len(dictionary)} barcodes kept of {len(raws[name])} seen, "
                  f"over {dictionary.construct.nunique()} constructs")

            print(f"\n{name}: counting edits in the cDNA reads")
            parsed_path, cdna_tally = count_edits(name, dictionary_path, work)
            tally.update(cdna_tally)
            tallies[name] = tally
            matched = tally.get("Reads_all_edit_pos_found_barcode_matched", np.nan)
            total = tally.get("Reads_total", np.nan)
            print(f"  {matched:,.0f} of {total:,.0f} cDNA reads parsed and matched a "
                  f"dictionary barcode ({matched / total * 100:.1f}%)")

            # The pipeline's own output against the answer committed beside
            # the reads, produced by a separate run.
            expected = EXAMPLES / name / "expected_output"
            print(f"\n{name}: checking against expected_output/")
            agreed &= verify(
                "dictionary", dictionary_path,
                expected / f"{spec['prefix']}.barcode_to_variant.csv.gz",
            )
            agreed &= verify(
                "cDNA_parsed", parsed_path,
                expected / f"{SAMPLE}_.cDNA_parsed.csv",
            )

            parsed_tables.append(pd.read_csv(parsed_path))

        parsed = pd.concat(parsed_tables, ignore_index=True)
        counts, samples = shape_counts(parsed)
        dataset = work / "dataset"
        dataset.mkdir(exist_ok=True)
        counts.to_csv(dataset / "counts.csv", index=False)
        samples.to_csv(dataset / "samples.csv", index=False)

        print(f"\ncounts.csv and samples.csv written to {dataset}")
        print(f"{len(counts)} rows, {counts.groupby(['bait', 'prey']).ngroups} pairs, "
              f"{len(samples)} sample")
        print(counts.head(8).to_string(index=False))
        print(samples.to_string(index=False))

        print("\nfigures")
        plot_attrition(tallies, args.figures / "attrition.png")
        plot_dictionary(raws, args.figures / "dictionary.png")
        plot_edit_rates(counts, args.figures / "edit_rates.png")
        print(f"  written to {args.figures}")

        if agreed:
            print("\nEvery file produced here matches the committed expected output.")
        else:
            raise SystemExit("\nthe pipeline disagreed with expected_output/ above")
    finally:
        if holder is not None:
            holder.cleanup()


if __name__ == "__main__":
    main()
