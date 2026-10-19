"""cDNA parse step 1: every edited form each dictionary barcode can be read as.

The base editor also edits the barcodes, so a barcode in the dictionary may
reach the sequencer with any subset of its ref_n bases converted. Each bc1 and
bc2 in the dictionary is expanded into all of those forms and written to
`{out_prefix}.bc1_alts.txt` / `.bc2_alts.txt` as `alt<TAB>original`, which step
2 loads as its barcode lookup.
"""
import argparse
import logging
import re
import sys
from itertools import product

import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def edit_mode(text):
    """Validate an edit mode like A2G or C2T, and return it upper-cased.

    Any base the editor converts, to any other base: an adenine base editor is
    A2G read off the forward strand and T2C read off the reverse, and a
    cytosine base editor is C2T or G2A.
    """
    mode = text.upper()
    match = re.fullmatch(r'([ACGT])2([ACGT])', mode)
    if match is None:
        raise argparse.ArgumentTypeError(
            f'{text!r} is not an edit mode: expected <ref base>2<edited base> over ACGT'
        )
    if match.group(1) == match.group(2):
        raise argparse.ArgumentTypeError(
            f'{text!r} converts a base to itself'
        )
    return mode


def edit_bases(mode):
    """The base the editor converts, and what it converts it to."""
    return mode[0], mode[2]


def parse_args():
    """Command-line interface for the barcode alt expander."""
    parser = argparse.ArgumentParser(
        description='Parse cDNA data and generate alternative barcode files'
    )
    parser.add_argument(
        '--dict_in',
        required=True,
        help=(
            'Barcode dictionary: tab-separated .txt/.txt.gz, or comma-separated '
            '.csv/.csv.gz (e.g. barcode_to_variant from step3)'
        ),
    )
    parser.add_argument(
        '--edit_mode',
        required=True,
        type=edit_mode,
        help='Edit mode as <ref base>2<edited base>, e.g. A2G, T2C or C2T'
    )
    parser.add_argument(
        '--out_prefix',
        required=True,
        help='Output file prefix for {out_prefix}.bc1_alts.txt and {out_prefix}.bc2_alts.txt'
    )
    ig = parser.add_mutually_exclusive_group()
    ig.add_argument(
        '--ignore_bc1',
        action='store_true',
        help='Dictionary keys bc1 as ""; only expand bc2 barcodes (matches step3 --ignore_bc1)',
    )
    ig.add_argument(
        '--ignore_bc2',
        action='store_true',
        help='Dictionary keys bc2 as ""; only expand bc1 barcodes (matches step3 --ignore_bc2)',
    )
    return parser.parse_args()


def _dict_sep(path: str) -> str:
    """Comma for .csv / .csv.gz; tab for legacy .txt / .txt.gz / other."""
    lowered = path.lower()
    if lowered.endswith((".csv.gz", ".csv")):
        return ","
    return "\t"


def all_substitutions(seq, ref_n, alt_n, include_original=False):
    """
    For a given sequence, return the set of all unique sequences where any subset (including all)
    of the ref_n bases are changed to alt_n. If include_original is True, include the
    original sequence. If ref_n is not present, returns an empty set (or set containing
    the original if include_original).
    """
    # Find positions of ref_n in seq
    positions = [i for i, base in enumerate(seq) if base == ref_n]
    n = len(positions)
    out = set()

    if n == 0:
        if include_original:
            out.add(seq)
        return out

    # For each possible combination of positions
    # Each position can be swapped (True) or kept (False)
    for mask in product([False, True], repeat=n):
        if not any(mask) and not include_original:
            continue  # skip all ref_n kept (original sequence) unless include_original
        seq_list = list(seq)
        for j, swap in enumerate(mask):
            if swap:
                seq_list[positions[j]] = alt_n
        out.add(''.join(seq_list))
    return out


def _write_bc_alts(path: str, bc_list: list, ref_n: str, alt_n: str) -> None:
    """Write `bc_alt<TAB>bc` rows for every barcode, one barcode at a time.

    The first barcode truncates the file and the rest append, so an empty
    bc_list leaves the path untouched.
    """
    for i, bc in enumerate(bc_list):
        bc_alts = all_substitutions(bc, ref_n, alt_n, include_original=True)
        bc_alts = list(bc_alts)
        df_alts = pd.DataFrame({'bc_alt': bc_alts, 'bc': bc})
        mode = 'w' if i == 0 else 'a'
        df_alts.to_csv(path, mode=mode, header=False, index=False, sep='\t')


def main():
    """Expand every dictionary barcode into its edited forms and write both files."""
    args = parse_args()

    infile = args.dict_in
    out_prefix = args.out_prefix
    bc_edit_mode = args.edit_mode
    ignore_bc1 = args.ignore_bc1
    ignore_bc2 = args.ignore_bc2

    ref_n, alt_n = edit_bases(bc_edit_mode)

    # Read input: compression='infer' handles .gz; sep follows extension
    logger.info("Loading barcode dictionary %s...", infile)
    try:
        df_dict = pd.read_csv(
            infile,
            sep=_dict_sep(infile),
            header=None,
            compression="infer",
        )
    except OSError as e:
        logger.error("can't open %s: %s", infile, e)
        sys.exit(1)
    ncols = df_dict.shape[1]

    # Extract bc1 and bc2 barcodes
    if ncols == 5 and df_dict[0].astype(str).str.contains('_').all():
        df_dict.columns = [
            'barcode_pair', 'construct_id', 'support_reads', 'total_reads', 'accuracy',
        ]
        pair_parts = df_dict['barcode_pair'].astype(str).str.split('_', n=1, expand=True)
        if ignore_bc1:
            bc1_set = {""}
            bc2_set = set(pair_parts[1].fillna("").astype(str))
        elif ignore_bc2:
            bc1_set = set(pair_parts[0].fillna("").astype(str))
            bc2_set = {""}
        else:
            bc1_set = set(pair_parts[0].fillna("").astype(str))
            bc2_set = set(pair_parts[1].fillna("").astype(str))
    elif ncols == 7:
        df_dict.columns = [
            'bc1', 'bc1_construct', 'bc2', 'bc2_construct', 'support_reads', 'total_reads',
            'accuracy',
        ]
        if ignore_bc1:
            bc1_set = {""}
            bc2_set = set(df_dict['bc2'].fillna("").astype(str))
        elif ignore_bc2:
            bc1_set = set(df_dict['bc1'].fillna("").astype(str))
            bc2_set = {""}
        else:
            bc1_set = set(df_dict['bc1'].fillna("").astype(str))
            bc2_set = set(df_dict['bc2'].fillna("").astype(str))
    else:
        raise ValueError(
            f"Unexpected file format: {ncols} columns detected. Expected 5 or 7 columns."
        )

    if ignore_bc1:
        logger.info("bc1 ignored (--ignore_bc1): only bc2 alt expansion")
    elif ignore_bc2:
        logger.info("bc2 ignored (--ignore_bc2): only bc1 alt expansion")

    logger.info(
        "Dictionary entries: %s, unique bc1 barcodes: %s, unique bc2 barcodes: %s",
        len(df_dict), len(bc1_set), len(bc2_set),
    )

    bc1_alts_path = out_prefix + ".bc1_alts.txt"
    bc2_alts_path = out_prefix + ".bc2_alts.txt"

    # Barcodes may contain edits as well — infer all possible alt barcodes from the dictionary
    logger.info("Infering all possible bc1 barcodes...")
    _write_bc_alts(bc1_alts_path, sorted(bc1_set), ref_n, alt_n)

    # Under --ignore_bc2 every dictionary row keys bc2 as "", so this writes a
    # file holding a single tab. That is correct, and step2_count_edits.py
    # expects the file to exist; the same holds for bc1 under --ignore_bc1.
    logger.info("Infering all possible bc2 barcodes...")
    _write_bc_alts(bc2_alts_path, sorted(bc2_set), ref_n, alt_n)


if __name__ == "__main__":
    main()
