"""Dictionary build step 1: barcode pair and construct sequence out of each read pair.

R1 is `[bc1][linker][bc2][constant 3' sequence]` and R2 is
`[prefix][fixed sequence][construct sequence]`, so the two reads together say
which barcode pair carries which construct. The linker and the fixed sequence
are split on rather than matched by position, because bc1 and the R2 prefix
both vary in length.

Read pairs that yield both halves are written to a FASTA headed `bc1_bc2_<n>`,
which step 2 aligns against the reference sequences. `--reference` says which
reference this library belongs to, and selects its row in the fixed-sequence
tables.
"""
import argparse
import gzip
import logging
import sys
import time
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def hamming_distance(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("Strings must be of equal length for Hamming distance")
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def parse_args():
    """Command-line interface for the dictionary read-1/read-2 parser."""
    parser = argparse.ArgumentParser(
        description="Build a barcode dictionary, step 1 (optional bc2 length modes)"
    )
    parser.add_argument(
        "--r1_fastq",
        required=True,
        help="Input R1 fastq file path",
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference this library belongs to, selecting its row in the fixed-sequence "
        "tables",
    )
    parser.add_argument(
        "--r2_fastq",
        required=True,
        help="Input R2 fastq file path",
    )
    parser.add_argument(
        "--r1_linker",
        required=True,
        help="Constant R1 sequence between bc1 and bc2 (split anchor)",
    )
    parser.add_argument(
        "--r2_fixed_seq_lib",
        required=True,
        help="Table of the fixed sequence that precedes the construct in R2, as "
        "reference,r2_fixed_seq",
    )
    parser.add_argument(
        "--r1_barcode2_lib",
        default=None,
        help="Optional table of the expected bc2 per reference, in the same two-column "
        "layout as --r2_fixed_seq_lib",
    )
    parser.add_argument(
        "--bc1_min_len",
        type=int,
        default=12,
        help="Minimum length for barcode 1 (default: 12)",
    )
    parser.add_argument(
        "--bc1_max_len",
        type=int,
        default=15,
        help="Maximum length for barcode 1 (default: 15)",
    )
    parser.add_argument(
        "--bc2_set_len",
        type=int,
        default=None,
        help="Fixed length for barcode 2; not required if --r1_barcode2_lib has a row "
        "for this reference (length taken from library)",
    )
    parser.add_argument(
        "--bc2_min_len",
        type=int,
        default=None,
        help="Minimum length for barcode 2; not required if --r1_barcode2_lib matches "
        "(see --bc2_max_len, --r1_post_bc2)",
    )
    parser.add_argument(
        "--bc2_max_len",
        type=int,
        default=None,
        help="Maximum length for barcode 2; not required if --r1_barcode2_lib matches",
    )
    parser.add_argument(
        "--r1_post_bc2",
        default=None,
        help="Constant R1 sequence immediately 3' of bc2; required with "
        "--bc2_min_len/--bc2_max_len to delimit bc2",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output FASTA path; gzip-compressed if the path ends with .gz (case-insensitive)",
    )

    return parser.parse_args()


def open_fastq(path):
    """Open a fastq for reading, decompressing when the name ends in .gz."""
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def open_fasta_out(path):
    """Open the output FASTA for writing, gzip-compressed when asked for by name."""
    return gzip.open(path, "wt") if path.lower().endswith(".gz") else open(path, "w")


def lookup_fixed_seq(path, reference):
    """The fixed sequence this reference uses, from a two-column CSV.

    Returns None when the table has no row for this reference, which the two
    callers treat differently: the R2 fixed sequence is required, the R1
    barcode 2 sequence is optional.
    """
    fixed_seq = None
    with open(path) as f:
        next(f)  # header: reference, sequence
        for line in f:
            line = line.strip()
            if not line:
                continue
            row_reference, row_seq = line.split(",")
            if row_reference == reference:
                fixed_seq = row_seq.upper()
    return fixed_seq


def resolve_bc2_mode(args, r1_bc2_seq):
    """Decide how bc2 is delimited, and at what length, or exit.

    A barcode 2 library row for this reference pins the length exactly, so
    it overrides the lengths given on the command line and only warns about the
    disagreement. Failing that, an explicit range delimits bc2 by the sequence
    behind it, and --bc2_set_len takes a fixed number of bases.
    """
    has_range = args.bc2_min_len is not None and args.bc2_max_len is not None
    has_partial_range = (args.bc2_min_len is not None) ^ (args.bc2_max_len is not None)
    has_set = args.bc2_set_len is not None

    if has_partial_range:
        logger.error(
            "Both --bc2_min_len and --bc2_max_len must be provided together, "
            "or use --bc2_set_len instead."
        )
        sys.exit(1)

    if r1_bc2_seq is not None:
        lib_len = len(r1_bc2_seq)
        if args.bc2_set_len is not None and args.bc2_set_len != lib_len:
            logger.warning(
                "--bc2_set_len (%s) does not match length of r1 barcode2 library "
                "sequence (%s); using library length.",
                args.bc2_set_len,
                lib_len,
            )
        if has_range and not (args.bc2_min_len <= lib_len <= args.bc2_max_len):
            logger.warning(
                "Barcode2 library sequence length (%s) is outside "
                "--bc2_min_len/--bc2_max_len (%s-%s); using library length for extraction.",
                lib_len,
                args.bc2_min_len,
                args.bc2_max_len,
            )
        return "set", lib_len

    if has_range:
        # a range wins over --bc2_set_len when both are given
        if args.r1_post_bc2 is None:
            logger.error(
                "--r1_post_bc2 is required when using --bc2_min_len and --bc2_max_len."
            )
            sys.exit(1)
        if args.bc2_min_len > args.bc2_max_len:
            logger.error(
                "--bc2_min_len (%s) cannot be greater than --bc2_max_len (%s).",
                args.bc2_min_len,
                args.bc2_max_len,
            )
            sys.exit(1)
        return "range", args.bc2_set_len

    if has_set:
        return "set", args.bc2_set_len

    logger.error(
        "Must provide either --r1_barcode2_lib with a matching reference row, "
        "or --bc2_set_len, or both --bc2_min_len and --bc2_max_len."
    )
    sys.exit(1)


@dataclass(frozen=True)
class R1Layout:
    """How to cut bc1 and bc2 out of R1, and which barcodes to accept."""
    linker: str
    bc2_mode: str
    bc2_set_len: int | None
    bc2_min_len: int | None
    bc2_max_len: int | None
    r1_post_bc2: str | None
    bc1_min_len: int
    bc1_max_len: int
    r1_bc2_seq: str | None

    def barcodes(self, seq1):
        """The (bc1, bc2) this read carries, or None if it carries neither.

        bc1 is whatever precedes the linker, so the read is rejected unless the
        linker occurs exactly once. bc2 then follows, either as a fixed number of
        bases or up to the constant sequence behind it.
        """
        parts = seq1.split(self.linker)
        if len(parts) != 2:
            return None
        bc1, after_linker = parts

        if self.bc2_mode == "set":
            bc2 = after_linker[: self.bc2_set_len]
        else:
            if self.r1_post_bc2 not in after_linker:
                return None
            bc2 = after_linker.partition(self.r1_post_bc2)[0]
            if not (self.bc2_min_len <= len(bc2) <= self.bc2_max_len):
                return None

        if not (self.bc1_min_len <= len(bc1) <= self.bc1_max_len):
            return None

        if self.r1_bc2_seq is not None:
            # bc2 is a known constant for this reference, so a read within one
            # substitution of it is corrected back rather than discarded
            if len(bc2) != len(self.r1_bc2_seq) or hamming_distance(bc2, self.r1_bc2_seq) > 1:
                return None
            bc2 = self.r1_bc2_seq

        return bc1, bc2


def extract_construct(seq2, r2_fixed_seq):
    """The construct sequence behind the fixed sequence in R2, or None.

    Something has to precede the fixed sequence for the read to be the expected
    amplicon, so a read that begins with it is rejected.
    """
    parts = seq2.split(r2_fixed_seq)
    if len(parts) != 2:
        return None
    prefix, construct_seq = parts
    if not prefix:
        return None
    return construct_seq


def main():
    """Parse the read pairs and write the barcode-headed construct FASTA."""
    args = parse_args()

    infile = args.r1_fastq
    infile2 = args.r2_fastq
    outfile = args.out

    r1_linker = args.r1_linker.upper()
    r1_post_bc2 = args.r1_post_bc2.upper() if args.r1_post_bc2 else None

    reference = args.reference

    # The barcode 2 library is optional, but it is read first: when it has a row
    # for this reference it fixes the bc2 length that mode resolution needs.
    r1_bc2_seq = (
        lookup_fixed_seq(args.r1_barcode2_lib, reference)
        if args.r1_barcode2_lib is not None
        else None
    )
    bc2_mode, bc2_set_len = resolve_bc2_mode(args, r1_bc2_seq)

    r2_fixed_seq = lookup_fixed_seq(args.r2_fixed_seq_lib, reference)
    if r2_fixed_seq is None:
        logger.error("R2 fixed sequence not found for reference %s", reference)
        sys.exit(1)

    logger.info("Reference: %s", reference)
    logger.info("R1 linker (bc1|bc2): %s", r1_linker)
    logger.info("BC2 mode: %s", bc2_mode)
    if bc2_mode == "range":
        logger.info("R1 post-bc2: %s", r1_post_bc2)
        logger.info("BC2 length range: %s-%s", args.bc2_min_len, args.bc2_max_len)
    else:
        logger.info("BC2 set length: %s", bc2_set_len)
    logger.info("R2 fixed seq: %s", r2_fixed_seq)

    layout = R1Layout(
        linker=r1_linker,
        bc2_mode=bc2_mode,
        bc2_set_len=bc2_set_len,
        bc2_min_len=args.bc2_min_len,
        bc2_max_len=args.bc2_max_len,
        r1_post_bc2=r1_post_bc2,
        bc1_min_len=args.bc1_min_len,
        bc1_max_len=args.bc1_max_len,
        r1_bc2_seq=r1_bc2_seq,
    )

    total_ct = 0
    pass_ct = 0
    barcodes_found = 0
    oligo_found = 0
    start_time = time.time()

    with (
        open_fastq(infile) as f_in,
        open_fastq(infile2) as f_in2,
        open_fasta_out(outfile) as f_out,
    ):
        while True:
            # Read R1 (4 lines per read: ID, sequence, +, quality)
            id1 = f_in.readline()
            if not id1:
                break
            seq1 = f_in.readline()
            f_in.readline()  # Skip +
            f_in.readline()  # Skip quality

            # Read R2 (4 lines per read: ID, sequence, +, quality)
            id2 = f_in2.readline()
            if not id2:
                break
            seq2 = f_in2.readline()
            f_in2.readline()  # Skip +
            f_in2.readline()  # Skip quality

            total_ct += 1

            # The two halves are counted separately, so both are parsed even when
            # one has already failed: the pair of counters in the report line is
            # what tells a bad linker apart from a bad fixed sequence.
            barcodes = layout.barcodes(seq1.strip())
            if barcodes is not None:
                barcodes_found += 1

            construct_seq = extract_construct(seq2.strip(), r2_fixed_seq)
            if construct_seq is not None:
                oligo_found += 1

            if barcodes is not None and construct_seq is not None:
                bc1, bc2 = barcodes
                pass_ct += 1
                f_out.write(">" + bc1 + "_" + bc2 + "_" + str(pass_ct) + "\n")
                f_out.write(construct_seq + "\n")

            if total_ct % 1000000 == 0:
                elapsed = time.time() - start_time
                print(
                    f"Processed reads: {total_ct}, passed reads: {pass_ct}, time: {elapsed:.2f}"
                )

    print("File_name\tTotal_reads\tbarcodes_found\toligo_found\tboth_found")
    print(f"{infile}\t{total_ct}\t{barcodes_found}\t{oligo_found}\t{pass_ct}")


if __name__ == "__main__":
    main()
