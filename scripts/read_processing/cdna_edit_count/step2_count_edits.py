"""cDNA parse step 2: per-construct edited-read counts for one cDNA fastq.

A cDNA read is `[bc1][edit template][bc2][constant 3' sequence]`, where bc1 and
bc2 identify the construct and the template carries the adenines (or thymines,
read from the other strand) the base editor can convert. The template is found
in the read by regex, with the editable positions wildcarded so that an edited
read still anchors; what precedes it is bc1 and what follows it is bc2.

The barcodes are then looked up through the alt forms step 1 expanded -- the
editor edits the barcodes too -- which gives the construct (`outcome`) the
read belongs to. Reads are tallied per outcome and written as one CSV row each:
total reads, reads carrying any edit, total edits, and a per-position count.
A tab-separated `Report_line` of read-level tallies goes to stdout, where the
caller picks it up.
"""
import argparse
import gzip
import logging
import re
import sys
import time
from dataclasses import dataclass
from itertools import islice

import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

COMPLEMENT = str.maketrans('ACGTN', 'TGCAN')

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

def build_parser():
    """Command-line interface for the cDNA edit counter."""
    parser = argparse.ArgumentParser(
        description='Parse cDNA data and analyze edits at all positions'
    )
    parser.add_argument(
        '--cDNA_in',
        required=True,
        help='cDNA R1.fastq file (can be gzipped)'
    )
    parser.add_argument(
        '--dict_in',
        required=True,
        help='Input dictionary file (can be gzipped)'
    )
    parser.add_argument(
        '--bc_alt_prefix',
        required=True,
        help='Prefix for *.bc1_alts.txt and *.bc2_alts.txt (from step1_expand_barcodes.py)',
    )
    parser.add_argument(
        '--out',
        required=True,
        help='Output file path'
    )
    ig = parser.add_mutually_exclusive_group()
    ig.add_argument(
        '--ignore_bc1',
        action='store_true',
        help='Use bc1="" for lookups (must match Step1_v2 --ignore_bc1)',
    )
    ig.add_argument(
        '--ignore_bc2',
        action='store_true',
        help='Use bc2="" for lookups (must match Step1_v2 --ignore_bc2)',
    )
    parser.add_argument(
        '--edit_mode',
        required=True,
        type=edit_mode,
        help='Edit mode as <ref base>2<edited base>, e.g. A2G, T2C or C2T'
    )
    parser.add_argument(
        '--template_seq',
        required=True,
        help='Template sequence'
    )
    parser.add_argument(
        '--regex_match_len',
        type=int,
        default=8,
        help='Length of template prefix/suffix used to build start/end regex anchors (default: 8)',
    )
    parser.add_argument(
        '--r1_post_bc2',
        default=None,
        help='Optional R1 sequence after bc2; first regex_match_len bases form an anchor regex. '
        'If set, bc2 is taken from the read between the end of cDNA and this anchor.',
    )
    parser.add_argument(
        '--cDNA_in_r2',
        default=None,
        help='cDNA R2.fastq (can be gzipped). Optional. When given, each read PAIR is screened for '
        'dictionary-library contamination before parsing: a genuine cDNA R2 is just the reverse '
        'mate of R1, so it must start with the reverse complement of --r1_post_bc2. '
        'Requires --r1_post_bc2.',
    )
    parser.add_argument(
        '--r2_anchor_max_offset',
        type=int,
        default=2,
        help='How far into R2 the reverse-complement anchor may start (default: 2)',
    )
    parser.add_argument(
        '--r2_fixed_seq_lib',
        default=None,
        help='Optional dictionary R2 fixed-sequence CSV (reference,r2_fixed_seq). Read pairs '
        'whose R2 contains any of these are dictionary reads and are discarded. Used with '
        '--cDNA_in_r2.',
    )
    parser.add_argument(
        '--primary_pos',
        type=int,
        default=None,
        help='1-based position IN THE EDIT TEMPLATE of the primary (on-target) edit. Surfaced '
        'in the output as Primary_pos / Primary_edit_ct (the per-construct read count edited at '
        'that position). Must be one of the editable positions (a ref-base site); otherwise a '
        'hard error. If omitted, defaults to the position edited in the most reads summed over '
        'all constructs -- logged, and NOT stable across samples, so pin it explicitly for '
        'any multi-sample comparison.',
    )
    parser.add_argument('--bc1_min_len', type=int, default=None,
                        help='Minimum bc1 length (inclusive)')
    parser.add_argument('--bc1_max_len', type=int, default=None,
                        help='Maximum bc1 length (inclusive)')
    parser.add_argument('--bc2_set_len', type=int, default=None,
                        help='Exact bc2 length (mutually exclusive with '
                             '--bc2_min_len/--bc2_max_len)')
    parser.add_argument('--bc2_min_len', type=int, default=None,
                        help='Minimum bc2 length; use with --bc2_max_len')
    parser.add_argument('--bc2_max_len', type=int, default=None,
                        help='Maximum bc2 length; use with --bc2_min_len')
    return parser


def reverse_complement(seq):
    return seq.upper().translate(COMPLEMENT)[::-1]


def _dict_sep(path: str) -> str:
    """Comma for .csv / .csv.gz; tab for legacy .txt / .txt.gz / other."""
    lowered = path.lower()
    if lowered.endswith((".csv.gz", ".csv")):
        return ","
    return "\t"


def load_dictionary(path):
    """Read the barcode dictionary and return parallel bc1, bc2 and outcome lists.

    Two layouts are accepted. Step 3 of the dictionary build writes five columns
    keyed by a joined `bc1_bc2` pair; a paired dictionary instead gives bc1 and
    bc2 their own columns and a construct for each, whose outcome is the two
    construct ids joined by `__`.
    """
    df_dict = pd.read_csv(path, sep=_dict_sep(path), header=None, compression="infer")
    ncols = df_dict.shape[1]

    if ncols == 5 and df_dict[0].astype(str).str.contains('_').all():
        df_dict.columns = [
            'barcode_pair', 'construct_id', 'support_reads', 'total_reads', 'accuracy',
        ]
        pair_parts = df_dict['barcode_pair'].astype(str).str.split('_', n=1, expand=True)
        bc1_list = pair_parts[0].fillna("").astype(str).tolist()
        bc2_list = pair_parts[1].fillna("").astype(str).tolist()
        outcome_list = df_dict['construct_id'].tolist()
    elif ncols == 7:
        df_dict.columns = [
            'bc1', 'bc1_construct', 'bc2', 'bc2_construct', 'support_reads', 'total_reads',
            'accuracy',
        ]
        bc1_list = df_dict['bc1'].fillna("").astype(str).tolist()
        bc2_list = df_dict['bc2'].fillna("").astype(str).tolist()
        outcome_list = (df_dict['bc1_construct'] + "__" + df_dict['bc2_construct']).tolist()
    else:
        raise ValueError(
            f"Unexpected file format: {ncols} columns detected. Expected 5 or 7 columns."
        )

    logger.info(
        "Dictionary entries: %s, bc1 barcodes: %s, bc2 barcodes: %s",
        len(df_dict), len(bc1_list), len(bc2_list),
    )
    return bc1_list, bc2_list, outcome_list


def index_barcode_pairs(bc1_list, bc2_list, outcome_list):
    """Index the dictionary for lookup: pair -> outcomes, and bc1 -> its bc2s.

    A (bc1, bc2) pair can map to MULTIPLE outcomes when the dictionary pools
    per-reference dicts (a bc1 reused for a different construct elsewhere), so
    each pair maps to the SET of all its outcomes -- a matching read is counted
    toward every construct involved.
    """
    pair_to_outcomes = {}
    for bc1, bc2, outcome in zip(bc1_list, bc2_list, outcome_list):
        pair_to_outcomes.setdefault((bc1, bc2), set()).add(outcome)

    bc1_to_bc2s = {}
    for bc1, bc2 in zip(bc1_list, bc2_list):
        bc1_to_bc2s.setdefault(bc1, set()).add(bc2)

    return pair_to_outcomes, bc1_to_bc2s


def load_bc_alts(path, label):
    """Map each edited barcode form back to the dictionary barcodes it could be.

    One `bc_alt<TAB>bc` row per line, as step 1 writes it. The same alt form can
    be reached from more than one dictionary barcode, so the values are sets.
    """
    handle = open_or_exit(path)
    logger.info("Loading all possible %s alt barcodes...", label)
    start_time = time.time()
    lines_read = 0
    bc_alts = {}
    with handle:
        for line in handle:
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                bc_alt, bc = parts[0], parts[1]
                bc_alts.setdefault(bc_alt, set()).add(bc)

            lines_read += 1
            if lines_read % 100000 == 0:
                logger.info(
                    "%s barcodes processed: %d, time: %.2f",
                    label, lines_read, time.time() - start_time,
                )
                start_time = time.time()

    logger.info("Total %s alt barcodes: %d", label, len(bc_alts))
    return bc_alts


def build_template_anchors(template_regex_str, regex_match_len):
    """Start and end anchors for the edit template, as compiled regexes.

    The head and tail substrings are taken from opposite ends of the template and
    are not allowed to overlap, so a short template with a long --regex_match_len
    is rejected rather than silently matching itself twice.
    """
    template_len = len(template_regex_str)
    head_len = min(regex_match_len, template_len)
    tail_len = min(regex_match_len, template_len - head_len)
    if head_len <= 0 or tail_len <= 0:
        raise ValueError(
            f"regex_match_len too large for template: template_len={template_len}, "
            f"head_len={head_len}, tail_len={tail_len}"
        )
    head_regex = re.compile(template_regex_str[:head_len], re.IGNORECASE)
    tail_regex = re.compile(template_regex_str[template_len - tail_len:], re.IGNORECASE)
    logger.info(
        'template_len=%d, regex_match_len=%d, head_regex_len=%d, tail_regex_len=%d',
        template_len, regex_match_len, head_len, tail_len,
    )
    return head_regex, tail_regex, tail_len


def build_post_bc2_regex(r1_post_bc2, ref_n, regex_match_len):
    """Anchor for the constant sequence 3' of bc2, which delimits bc2."""
    post_bc2_regex_str = r1_post_bc2.replace(ref_n, ".")[:regex_match_len]
    logger.info(
        'r1_post_bc2 anchor: first %d bases (escaped regex len=%d)',
        len(post_bc2_regex_str), len(post_bc2_regex_str),
    )
    return re.compile(post_bc2_regex_str, re.IGNORECASE)


# ---------------------------------------------------------------------------
# R2 contamination screen (optional; enabled by --cDNA_in_r2)
#
# A genuine cDNA read pair is a short barcode amplicon, so R2 is simply R1's
# reverse mate: [1nt][rc(r1_post_bc2)][rc(bc2)][rc(template)][rc(bc1)][adapter].
# A dictionary read pair instead carries the oligo on R2, behind a reference-specific
# fixed sequence. The two are mutually exclusive -- measured at a 0.00% collision
# rate over 13 gold-standard dictionary references (2026-07-22).
# ---------------------------------------------------------------------------


def build_r2_anchor_regex(r1_post_bc2, ref_n):
    """Anchor that identifies R2 as the reverse mate of R1.

    The anchor is derived from --r1_post_bc2 rather than hard-coded, and the
    edit-mode wildcarding is carried across the strand: a ref_n->alt_n edit on the
    forward strand reads as complement(ref_n)->complement(alt_n) on R2, so those
    positions must be wildcards here too. Matching the literal reverse complement
    would preferentially reject the most-edited reads and bias the edit rate down.
    """
    rc_ref_n = ref_n.translate(COMPLEMENT)
    anchor_str = reverse_complement(r1_post_bc2).replace(rc_ref_n, '.')
    return re.compile(anchor_str, re.IGNORECASE), rc_ref_n, anchor_str


def load_r2_fixed_seqs(path):
    """The dictionary R2 fixed sequences, deduplicated.

    The column is located by name from the header rather than by position, so a
    table carrying extra columns still works.
    """
    fixed_seqs = []
    with open(path) as fh:
        header = fh.readline().strip().split(',')
        try:
            column = header.index('r2_fixed_seq')
        except ValueError:
            raise SystemExit(
                f'{path}: no r2_fixed_seq column, found {header}'
            ) from None
        for line in fh:
            parts = line.strip().split(',')
            if len(parts) > column and parts[column]:
                fixed_seqs.append(parts[column].upper())
    return sorted(set(fixed_seqs))


def r2_verdict(seq2, fixed_seqs, anchor_regex, max_anchor_offset):
    """
    Classify an R2 read. Returns one of:
      'cDNA'      -- reverse mate of R1, keep
      'dict'      -- carries a dictionary reference fixed seq: library contamination
      'no_anchor' -- neither; adapter dimer / short insert / unparseable

    'dict' and 'no_anchor' are counted separately on purpose: on clean cDNA
    ~10-16% of reads legitimately lack the anchor (adapter dimers, which fail R1
    parsing anyway), so collapsing them into one counter would report that
    background as contamination.
    """
    for fixed_seq in fixed_seqs:
        if fixed_seq in seq2:
            return 'dict'
    anchor = anchor_regex.search(seq2)
    if anchor is None or anchor.start() > max_anchor_offset:
        return 'no_anchor'
    return 'cDNA'


@dataclass(frozen=True)
class ReadLayout:
    """Where the pieces of a cDNA read sit, and which barcode lengths are allowed.

    `bc2_dict_len` is the bc2 length the dictionary itself carries, used only
    when neither --bc2_set_len nor a --bc2_min_len/--bc2_max_len range is given.
    """
    head_regex: re.Pattern
    tail_regex: re.Pattern
    tail_len: int
    template_len: int
    ignore_bc1: bool
    ignore_bc2: bool
    post_bc2_regex: re.Pattern | None
    bc2_dict_len: int
    bc1_min_len: int | None
    bc1_max_len: int | None
    bc2_set_len: int | None
    bc2_min_len: int | None
    bc2_max_len: int | None

    def bc1_len_ok(self, length):
        if self.ignore_bc1:
            return True
        too_short = self.bc1_min_len is not None and length < self.bc1_min_len
        too_long = self.bc1_max_len is not None and length > self.bc1_max_len
        return not (too_short or too_long)

    def bc2_len_ok(self, length):
        if self.ignore_bc2:
            return True
        if self.bc2_set_len is not None:
            return length == self.bc2_set_len
        if self.bc2_min_len is not None:
            return self.bc2_min_len <= length <= self.bc2_max_len
        # Default behavior: enforce dictionary-derived bc2 length.
        return length == self.bc2_dict_len


def split_read(seq, layout):
    """Cut one read into (template, bc1, bc2), or None if it does not parse.

    The head anchor is *searched* for rather than matched at the start of the
    read: bc1 comes first and its length varies, so where the anchor hits is what
    tells bc1 and the template apart. The tail anchor is then checked at the far
    end of the template slice, which rejects a read whose head matched by chance.
    """
    head_match = layout.head_regex.search(seq)
    if not head_match:
        return None

    cDNA_start = head_match.start()
    cDNA_end = cDNA_start + layout.template_len
    if cDNA_end > len(seq):
        return None
    cDNA = seq[cDNA_start:cDNA_end]
    if not layout.tail_regex.match(cDNA[-layout.tail_len:]):
        return None

    bc1_read = "" if layout.ignore_bc1 else seq[:cDNA_start]
    if not layout.bc1_len_ok(len(bc1_read)):
        return None

    if layout.ignore_bc2:
        bc2_read = ""
    elif layout.post_bc2_regex is not None:
        # bc2 runs from the end of the template to the constant sequence behind it
        post_match = layout.post_bc2_regex.search(seq, cDNA_end)
        if not post_match:
            return None
        bc2_read = seq[cDNA_end:post_match.start()]
    else:
        if cDNA_end + layout.bc2_dict_len > len(seq):
            return None
        bc2_read = seq[cDNA_end:cDNA_end + layout.bc2_dict_len]

    if not layout.bc2_len_ok(len(bc2_read)):
        return None

    return cDNA, bc1_read, bc2_read


def call_edits(cDNA, edit_pos, ref_n, alt_n):
    """Call every editable position as edited (1) or not (0).

    Returns None if any editable position holds a base that is neither the
    reference nor the edited base: that is a sequencing or synthesis error rather
    than an edit, and the read is not scored at all.
    """
    edits = {}
    for pos in edit_pos:
        base = cDNA[pos - 1]  # edit_pos is 1-based
        if base == alt_n:
            edits[pos] = 1
        elif base == ref_n:
            edits[pos] = 0
        else:
            return None
    return edits


def lookup_outcomes(bc1_read, bc2_read, bc1_alts, bc2_alts, bc1_to_bc2s, pair_to_outcomes):
    """Outcomes for a read's barcodes, and how many dictionary pairs it matched.

    Both barcodes may be edited, so each is first resolved to the set of
    dictionary barcodes it could have come from; every combination of those that
    exists in the dictionary contributes its outcomes.
    """
    outcomes = set()
    pair_matches = 0
    if (bc1_read in bc1_alts) and (bc2_read in bc2_alts):
        bc2_originals = bc2_alts[bc2_read]
        for original_bc1 in bc1_alts[bc1_read]:
            for original_bc2 in bc2_originals & bc1_to_bc2s.get(original_bc1, frozenset()):
                outcomes.update(pair_to_outcomes[(original_bc1, original_bc2)])
                pair_matches += 1
    return outcomes, pair_matches


def resolve_primary_pos(primary_pos, edit_pos, result_dict):
    """The position reported as the on-target edit, resolved if it was not given.

    The default is the position edited in the most reads summed over all constructs
    (a column-sum argmax), ties breaking to the earliest position. Every passing
    read is length==template_len, so all edit positions are scored over the same
    reads and the totals are directly comparable.
    """
    if primary_pos is not None:
        logger.info('primary edit position: %s (user-specified)', primary_pos)
        return primary_pos
    if not edit_pos:
        logger.info('--primary_pos not given and the template has no editable position')
        return None

    pos_totals = {
        pos: sum(counts['edit_pos'].get(pos, 0) for counts in result_dict.values())
        for pos in edit_pos
    }
    primary_pos = min(edit_pos, key=lambda p: (-pos_totals[p], p))
    logger.info(
        '--primary_pos not given; defaulting to most-edited position %s (%s edits across all '
        'constructs). NOT stable across samples -- pin it explicitly for multi-sample '
        'comparison.',
        primary_pos, pos_totals[primary_pos],
    )
    return primary_pos


def write_counts(out_file, result_dict, edit_pos, primary_pos):
    """One row per outcome: totals, the primary position, and every position."""
    with open_or_exit(out_file, 'w') as f_out:
        f_out.write("Outcome,Total_reads,Is_edited,Any_edit_ct,Primary_pos,Primary_edit_ct")
        for pos in edit_pos:
            f_out.write(f",Pos{pos}")
        f_out.write("\n")

        for outcome in sorted(result_dict.keys()):
            counts = result_dict[outcome]
            primary_edit_ct = (
                counts['edit_pos'].get(primary_pos, 0) if primary_pos is not None else 0
            )
            f_out.write(
                f"{outcome},{counts['total']},{counts['is_edited']},{counts['any_edited_ct']},"
                f"{primary_pos},{primary_edit_ct}"
            )
            for pos in edit_pos:
                f_out.write(f",{counts['edit_pos'].get(pos, 0)}")
            f_out.write("\n")


def open_or_exit(path, mode='r'):
    """Open a path, or log why it could not be opened and exit.

    Every path here comes from the command line, so a bad one is a user error
    rather than something to raise a traceback over.
    """
    try:
        return open(path, mode)
    except OSError as e:
        logger.error("can't open %s: %s", path, e)
        sys.exit(1)


def open_fastq(path):
    """Open a fastq for reading, decompressing when the name ends in .gz."""
    if not path.endswith('.gz'):
        return open_or_exit(path)
    try:
        return gzip.open(path, 'rt')
    except OSError as e:
        logger.error("can't open %s: %s", path, e)
        sys.exit(1)


def main():
    """Count edits per construct over one cDNA fastq, and write the counts table."""
    parser = build_parser()
    args = parser.parse_args()
    if (args.bc2_min_len is None) ^ (args.bc2_max_len is None):
        parser.error('--bc2_min_len and --bc2_max_len must be given together')
    if args.bc2_set_len is not None and (
        args.bc2_min_len is not None or args.bc2_max_len is not None
    ):
        parser.error('use either --bc2_set_len or --bc2_min_len/--bc2_max_len, not both')

    out_file = args.out
    ref_n, alt_n = edit_bases(args.edit_mode)

    # The editable positions are wildcarded in every anchor, so that an edited
    # read still matches the sequence it was edited in.
    template = args.template_seq.upper()
    template_regex_str = template.replace(ref_n, ".")
    try:
        head_regex, tail_regex, tail_len = build_template_anchors(
            template_regex_str, args.regex_match_len
        )
    except ValueError as e:
        parser.error(str(e))

    r1_post_bc2 = args.r1_post_bc2.upper() if args.r1_post_bc2 else None
    post_bc2_regex = (
        build_post_bc2_regex(r1_post_bc2, ref_n, args.regex_match_len) if r1_post_bc2 else None
    )

    r2_anchor_regex = None
    r2_fixed_seqs = []
    if args.cDNA_in_r2:
        if not r1_post_bc2:
            logger.error('--cDNA_in_r2 requires --r1_post_bc2 (the R2 anchor is derived from it)')
            sys.exit(1)
        r2_anchor_regex, rc_ref_n, r2_anchor_str = build_r2_anchor_regex(r1_post_bc2, ref_n)
        logger.info(
            'R2 anchor: rc(%s) with %s wildcarded -> %s (allowed start offset 0-%d)',
            r1_post_bc2, rc_ref_n, r2_anchor_str, args.r2_anchor_max_offset,
        )
        if args.r2_fixed_seq_lib:
            r2_fixed_seqs = load_r2_fixed_seqs(args.r2_fixed_seq_lib)
            logger.info('R2 dictionary fixed seqs loaded: %d', len(r2_fixed_seqs))

    # Positions in the template that can carry an edit, 1-based.
    edit_pos = [i + 1 for i, base in enumerate(template) if base == ref_n]
    if args.primary_pos is not None and args.primary_pos not in edit_pos:
        logger.error(
            '--primary_pos %d is not an editable position. Editable (%s) positions in this '
            'template are: %s', args.primary_pos, args.edit_mode, edit_pos,
        )
        sys.exit(1)

    bc1_list, bc2_list, outcome_list = load_dictionary(args.dict_in)
    result_dict = {outcome: {
        'total': 0,
        'any_edited_ct': 0,
        'is_edited': 0,
        'edit_pos': dict.fromkeys(edit_pos, 0)
    } for outcome in set(outcome_list)}

    # derive bc2 barcode length (0 if all empty / ignore_bc2)
    bc2_dict_len = len(bc2_list[0]) if bc2_list else 0
    if not args.ignore_bc2:
        if not all(len(bc) == bc2_dict_len for bc in bc2_list):
            raise ValueError(f"bc2 barcodes have different lengths: {bc2_list}")
        logger.info("bc2 barcode length: %s", bc2_dict_len)
    else:
        logger.info("bc2 ignored: bc2 barcode length treated as 0")

    pair_to_outcomes, bc1_to_bc2s = index_barcode_pairs(bc1_list, bc2_list, outcome_list)
    bc1_alts = load_bc_alts(args.bc_alt_prefix + ".bc1_alts.txt", "bc1")
    bc2_alts = load_bc_alts(args.bc_alt_prefix + ".bc2_alts.txt", "bc2")

    layout = ReadLayout(
        head_regex=head_regex,
        tail_regex=tail_regex,
        tail_len=tail_len,
        template_len=len(template_regex_str),
        ignore_bc1=args.ignore_bc1,
        ignore_bc2=args.ignore_bc2,
        post_bc2_regex=post_bc2_regex,
        bc2_dict_len=bc2_dict_len,
        bc1_min_len=args.bc1_min_len,
        bc1_max_len=args.bc1_max_len,
        bc2_set_len=args.bc2_set_len,
        bc2_min_len=args.bc2_min_len,
        bc2_max_len=args.bc2_max_len,
    )

    cdna_handle = open_fastq(args.cDNA_in)
    cdna_r2_handle = open_fastq(args.cDNA_in_r2) if args.cDNA_in_r2 else None

    total_reads = 0
    reads_r2_screened = 0
    reads_r2_dict = 0
    reads_r2_no_anchor = 0
    reads_cDNA_pass = 0
    reads_barcode_found = 0
    reads_cDNA_pass_barcode_found = 0
    barcode_pair_seen = {}  # "bc1_bc2" -> 1 if any read with that pair matched the dictionary

    logger.info("Parsing cDNA data...")
    start_time = time.time()

    while True:
        r1_lines = list(islice(cdna_handle, 4))
        if len(r1_lines) < 4:
            break
        seq = r1_lines[1].rstrip('\n')
        total_reads += 1

        # Screen the read PAIR before parsing R1: discard dictionary-library reads.
        if cdna_r2_handle is not None:
            r2_lines = list(islice(cdna_r2_handle, 4))
            if len(r2_lines) < 4:
                logger.error(
                    'R2 exhausted at read %d while R1 continues -- the fastqs are not a '
                    'matched pair (truncated or mismatched file?). Aborting.', total_reads,
                )
                sys.exit(1)
            reads_r2_screened += 1
            verdict = r2_verdict(
                r2_lines[1].rstrip('\n'), r2_fixed_seqs, r2_anchor_regex, args.r2_anchor_max_offset
            )
            if verdict != 'cDNA':
                if verdict == 'dict':
                    reads_r2_dict += 1
                else:
                    reads_r2_no_anchor += 1
                continue

        parsed = split_read(seq, layout)
        if parsed is not None:
            cDNA, bc1_read, bc2_read = parsed

            edits = call_edits(cDNA, edit_pos, ref_n, alt_n)
            if edits is not None:
                reads_cDNA_pass += 1

            outcomes, pair_matches = lookup_outcomes(
                bc1_read, bc2_read, bc1_alts, bc2_alts, bc1_to_bc2s, pair_to_outcomes
            )
            # Counted per matching dictionary pair, not per read: an edited barcode
            # that resolves to several dictionary entries counts once for each.
            reads_barcode_found += pair_matches

            bc_key = f"{bc1_read}_{bc2_read}"
            # Keep the pair flagged if ANY read carrying it matched the dictionary
            barcode_pair_seen[bc_key] = max(
                barcode_pair_seen.get(bc_key, 0), 1 if pair_matches else 0
            )

            if edits is not None and pair_matches:
                reads_cDNA_pass_barcode_found += 1
                edit_ct = sum(edits.values())
                for outcome in outcomes:
                    counts = result_dict[outcome]
                    counts['total'] += 1
                    counts['any_edited_ct'] += edit_ct
                    if edit_ct > 0:
                        counts['is_edited'] += 1
                    for pos, edited in edits.items():
                        counts['edit_pos'][pos] += edited

        if total_reads % 1000000 == 0:
            logger.info(
                "cDNA reads parsed: %s, time: %.2f", total_reads, time.time() - start_time
            )
            start_time = time.time()

    cdna_handle.close()
    if cdna_r2_handle is not None:
        # R1 ended first: if R2 still has reads the two files are not a matched pair
        if list(islice(cdna_r2_handle, 4)):
            logger.error(
                'R1 exhausted at read %d but R2 has more reads -- the fastqs are not a '
                'matched pair. Aborting.', total_reads,
            )
            sys.exit(1)
        cdna_r2_handle.close()
        logger.info(
            'R2 screen: %d of %d pairs = dictionary contamination (%.2f%%); '
            '%d lacked the anchor (%.2f%%, adapter dimers etc.)',
            reads_r2_dict, reads_r2_screened,
            100.0 * reads_r2_dict / reads_r2_screened if reads_r2_screened else 0.0,
            reads_r2_no_anchor,
            100.0 * reads_r2_no_anchor / reads_r2_screened if reads_r2_screened else 0.0,
        )

    primary_pos = resolve_primary_pos(args.primary_pos, edit_pos, result_dict)
    write_counts(out_file, result_dict, edit_pos, primary_pos)

    # cDNA analysis report line
    # The two R2-screen fields are always emitted (0 when --cDNA_in_r2 was not used) so the
    # Report_line schema stays fixed; step 02 only surfaces them when the screen actually ran.
    # Reads_barcode_matched counts barcode-to-construct matches rather than reads: a barcode
    # that maps to more than one construct credits each of them, so the field can exceed
    # Reads_total and is not a read-attrition number. Reads_all_edit_pos_found_barcode_matched
    # is the one that counts reads.
    print(f"Report_line\t{out_file}\tReads_total\t{total_reads}\tReads_barcode_matched\t{reads_barcode_found}\tReads_all_edit_pos_found\t{reads_cDNA_pass}\tReads_all_edit_pos_found_barcode_matched\t{reads_cDNA_pass_barcode_found}\tBarcodes\t{len(barcode_pair_seen)}\tBarcodes_matched\t{sum(barcode_pair_seen.values())}\tReads_r2_screened\t{reads_r2_screened}\tReads_r2_dict_contam\t{reads_r2_dict}\tReads_r2_no_anchor\t{reads_r2_no_anchor}")


if __name__ == "__main__":
    main()
