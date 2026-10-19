"""Dictionary build step 2: which library construct each aligned read carries.

Step 1's FASTA of construct sequences, headed `bc1_bc2_<n>`, is aligned against
one reference sequence: the contig `--reference` names. A read and a library
construct are described in the same vocabulary -- the set of mismatches each
carries against that reference, written `<ref base><0-based position><alt
base>` -- and a read carries a construct when every one of the construct's
mismatches is among the read's own. Reads matching exactly one construct are
written as `barcode pair,construct_id`, which step 3 collapses into the
dictionary.

The construct library is a CSV of `construct_id,reference,sequence`, where
`reference` selects the rows belonging to this alignment and names the contig
they were aligned to.
"""
import argparse
import logging
import os
from collections import defaultdict

import pandas as pd
import pysam

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

LIBRARY_COLUMNS = ('construct_id', 'reference', 'sequence')


def load_library(path, reference):
    """The constructs belonging to one reference, as a CSV of LIBRARY_COLUMNS."""
    library = pd.read_csv(path)
    missing = [column for column in LIBRARY_COLUMNS if column not in library.columns]
    if missing:
        raise ValueError(
            f"Construct library file is missing the columns: {', '.join(missing)}"
        )
    return library.loc[library['reference'].astype(str) == reference, list(LIBRARY_COLUMNS)]


def parse_args():
    parser = argparse.ArgumentParser(
        description=('This script assigns each aligned dictionary read to the library '
                     'construct whose mismatches it carries.'),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--bam_in',
                        type=str,
                        required=True,
                        help=('BAM file of the aligned reads, aligned to the contig\n'
                              '--reference names.'))

    parser.add_argument('--fasta_in',
                        type=str,
                        required=True,
                        help=('Fasta file of the reference sequences, one contig per\n'
                              'reference.'))

    parser.add_argument('--construct_lib',
                        type=str,
                        required=True,
                        help=('Construct library as a CSV file, with the columns\n'
                              'construct_id, reference and sequence.'))

    parser.add_argument('--reference',
                        type=str,
                        required=True,
                        help=('Reference these reads were aligned against, naming both the\n'
                              'FASTA contig and the library rows that belong to it.'))

    parser.add_argument('--out',
                        type=str,
                        required=True,
                        help=('Output CSV path; compression is inferred from the extension\n'
                              '(e.g. .csv.gz → gzip).'))

    return parser.parse_args()


def construct_mismatches(ref_seq, construct_seq):
    """Mismatch tokens (`<ref><0-based ref position><alt>`) of one library construct.

    The construct is compared base for base against the reference, so it is
    first trimmed to the reference length; a shorter sequence is an error.
    """
    construct_seq = construct_seq[:len(ref_seq)]
    if len(construct_seq) != len(ref_seq):
        raise ValueError(
            f'Construct sequence is shorter ({len(construct_seq)}) than the '
            f'reference ({len(ref_seq)})'
        )
    return [
        f'{ref_base}{r_pos}{alt_base}'
        for r_pos, (ref_base, alt_base) in enumerate(zip(ref_seq, construct_seq))
        if ref_base != alt_base
    ]


def read_mismatches(read, ref_seq):
    """Mismatch tokens of one aligned read, in the same vocabulary as the library.

    Only reference positions the read actually covers are compared, and only
    where the reference base recorded in the alignment agrees with the reference
    oligo: a position whose two reference bases disagree carries no usable call
    and is skipped, as is one past the end of the reference oligo.
    """
    query_seq = read.query_sequence.upper()
    tokens = []
    for q_pos, r_pos, aligned_ref_base in read.get_aligned_pairs(with_seq=True, matches_only=True):
        if aligned_ref_base is None or r_pos >= len(ref_seq):
            continue
        ref_base = ref_seq[r_pos]
        if aligned_ref_base.upper() != ref_base:
            continue
        alt_base = query_seq[q_pos]
        if alt_base != ref_base:
            tokens.append(f'{ref_base}{r_pos}{alt_base}')
    return tokens


def index_constructs_by_token(construct_mismatch_sets):
    """Index constructs by one of their mismatch tokens, for candidate lookup.

    A read carries a construct only if *every* mismatch of that construct is
    among the read's own mismatches, so a construct filed under any single one
    of its tokens is found by every read that could possibly match it. Reads
    then test a handful of candidates instead of the whole library.
    """
    index = defaultdict(list)
    for construct_id, mismatches in construct_mismatch_sets:
        index[next(iter(mismatches))].append((construct_id, mismatches))
    return index


def matching_constructs(read_mismatch_set, construct_index):
    """Library constructs whose mismatches are all present in this read."""
    matches = []
    for token in read_mismatch_set:
        for construct_id, mismatches in construct_index.get(token, ()):
            if mismatches <= read_mismatch_set:
                matches.append(construct_id)
    return matches


def main():
    args = parse_args()
    bam_path = args.bam_in
    fasta_path = args.fasta_in
    lib_path = args.construct_lib

    # load reference sequences
    logger.info('Loading reference sequences: %s', fasta_path)
    fasta = pysam.FastaFile(fasta_path)

    # load bam file
    logger.info('Loading align reads bam file: %s', bam_path)
    if not os.path.exists(bam_path + '.bai'):
        pysam.index(bam_path)
    bam = pysam.AlignmentFile(bam_path, 'rb')

    reference = args.reference
    ref_seq = fasta.fetch(reference)
    logger.info('Reference: %s', reference)

    # load the construct library and keep this reference's rows
    logger.info('Loading construct library file: %s', lib_path)
    library = load_library(lib_path, reference)
    logger.info('Number of constructs for %s: %s', reference, len(library))

    # identify mismatches for each construct; a construct indistinguishable from
    # the reference carries no mismatch to match on and is left out. Reads that
    # carry no mismatch are likewise dropped and never reach step 3, so a
    # dictionary's purity is a purity among construct-bearing reads alone.
    logger.info('Identify mismatches in construct sequences ...')
    construct_mismatch_sets = []
    for construct_id, sequence in zip(library['construct_id'], library['sequence']):
        mismatches = construct_mismatches(ref_seq, sequence)
        if mismatches:
            construct_mismatch_sets.append((construct_id, frozenset(mismatches)))
    construct_index = index_constructs_by_token(construct_mismatch_sets)
    logger.info('Constructs carrying at least one mismatch: %s', len(construct_mismatch_sets))

    # identify mismatches in each read and match them against the library. A read
    # is kept only when exactly one construct accounts for its mismatches.
    logger.info('Identify mismatches in reads and match constructs ...')
    bc_pair = []
    construct_matched = []
    reads_seen = 0
    for read in bam.fetch():
        if read.reference_name == reference:
            mismatches = frozenset(read_mismatches(read, ref_seq))
            if mismatches:
                matches = matching_constructs(mismatches, construct_index)
                if len(matches) == 1:
                    bc_pair.append(read.query_name)
                    construct_matched.append(matches[0])

        reads_seen += 1
        if reads_seen % 10000 == 0:
            logger.info('Processed %s reads ...', reads_seen)

    logger.info('Processed %s reads in total.', reads_seen)
    logger.info('Reads mapped to exactly one construct: %s', len(bc_pair))

    logger.info('Writing reads mapped with constructs to %s', args.out)
    pd.DataFrame({'bc_pair': bc_pair, 'construct_id': construct_matched}).to_csv(
        args.out,
        header=False,
        index=False,
        compression='infer',
    )


if __name__ == '__main__':
    main()
