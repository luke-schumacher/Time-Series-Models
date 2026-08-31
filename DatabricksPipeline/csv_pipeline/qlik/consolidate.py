#!/usr/bin/env python3
"""
Consolidate per-scanner CSVs into two flat files for Qlik manual upload.

Reads:
    data/real/exchange/DATA_*.csv         (from step 01 on Databricks)
    data/real/exam/DATA_*.csv             (from step 02 on Databricks)
    data/synthetic/exchange/DATA_*.csv    (from step 05, if present)
    data/synthetic/exam/DATA_*.csv        (from step 05, if present)

Writes:
    data/combined/exchange_combined.csv   — all exchange rows, tagged by DataSource
    data/combined/exam_combined.csv       — all exam rows, tagged by DataSource

Key columns added / renamed during consolidation:
  + DataSource         ('Real' or 'Synthetic'; first column) — drives the
                        side-by-side comparisons in Qlik charts
  * sample_idx →       ExchangeBlockID in the exchange file
  * sample_idx →       PatientVisitID  in the exam file
                        (prevents Qlik from spuriously auto-linking these
                        columns across tables — they mean different things)

  * All remaining columns are prefixed with 'Exch_' or 'Exam_' per kind,
    except the four fields left unprefixed on purpose so Qlik can
    associate the two tables through them:
        DataSource, SN, ExchangeBlockID, PatientVisitID
    Everything else (Age, Weight, Height, duration, predicted_mu,
    timediff, PatientID, …) would otherwise collide by name between the
    two files and cause Qlik to auto-create a synthetic key over every
    shared column, silently inflating every aggregate through the join.
    With the prefixes, the associative model cleanly links on exactly
    two fields (SN and DataSource) and nothing else.

Synthetic data is optional: if the synthetic folders are empty (e.g. step 05
has not been rerun yet), the output files will contain only real rows.  The
same Qlik dashboard can still be built and tested; rerun this script after
step 05 produces new synthetic CSVs to refresh.

Usage:
    cd DatabricksPipeline/csv_pipeline/qlik
    python consolidate.py
"""

import os
import sys
from glob import glob

import pandas as pd

# ---------------------------------------------------------------------------
# Sequence-family vocabulary.
#
# Step 03 classifies the raw `Sequence` string ('%SiemensSeq%\\fl3d_vibe') into
# one of 15 families before training; step 05 emits the family name directly
# ('vibe'). Real CSVs from step 02 keep the raw string, so without this the two
# sides never share a single Exam_Sequence value and every Qlik comparison on
# that dimension renders two disjoint bar sets.
#
# Prefer the canonical copy in ../config.py. The qlik/ folder is also shipped
# standalone (see README), where that import isn't available — hence the
# inline fallback. Keep the fallback in sync with config.py:136-160.
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import classify_sequence_type, ID_TO_SEQUENCE_TYPE
    _VOCAB_SOURCE = '../config.py'
except ImportError:
    SEQUENCE_TYPE_VOCAB = {
        'other': 0, 'scout': 1, 'localizer': 2, 'tse': 3, 'space': 4,
        'haste': 5, 'gre': 6, 'flash': 7, 'epi': 8, 'tfl': 9, 'tirm': 10,
        'vibe': 11, 'dixon': 12, 'swi': 13, 'medic': 14,
    }
    _SEQUENCE_TYPE_KEYS = [
        'localizer', 'scout', 'haste', 'space', 'tirm', 'vibe', 'dixon',
        'medic', 'swi', 'tfl', 'flash', 'tse', 'gre',
    ]
    ID_TO_SEQUENCE_TYPE = {v: k for k, v in SEQUENCE_TYPE_VOCAB.items()}

    def classify_sequence_type(raw):
        """Map a raw `Sequence` string to a SEQUENCE_TYPE_VOCAB id."""
        s = str(raw or '').lower()
        if not s:
            return SEQUENCE_TYPE_VOCAB['other']
        for key in _SEQUENCE_TYPE_KEYS:
            if key in s:
                return SEQUENCE_TYPE_VOCAB[key]
        if 'ep2d' in s or 'epi' in s or 'bold' in s or 'diff' in s or 'dwi' in s:
            return SEQUENCE_TYPE_VOCAB['epi']
        return SEQUENCE_TYPE_VOCAB['other']

    _VOCAB_SOURCE = 'inline fallback'


HERE         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(HERE, 'data')
COMBINED_DIR = os.path.join(DATA_DIR, 'combined')

# Which timestamp column to sort on per kind
TIME_COL = {
    'exchange': 'datetime',
    'exam':     'startTime',
}

# Columns to drop before writing. These exist in one side only (or carry a
# single constant value) and therefore mislead any Qlik comparison chart.
DROP_COLS = {
    'exchange': (),
    'exam':     (),
}

# FinishEvent → sourceID mapping. Step 02 doesn't emit sourceID on real exam
# rows, but FinishEvent encodes the same finish-marker semantic. Backfilling
# real sourceID from FinishEvent makes the column comparable across sides
# (synthetic emits MRI_MSR_104 for 'Successful' and MRI_MSR_34 for any abort).
FINISH_TO_SOURCEID = {
    'Successful':           'MRI_MSR_104',
    'Stopped by User':      'MRI_MSR_34',
    'Stopped by Scanner':   'MRI_MSR_34',
    'Stopped by ImageR':    'MRI_MSR_34',
    'Start MeasSys Failed': 'MRI_MSR_34',
}

# Rename sample_idx to something unambiguous per kind, so Qlik doesn't
# auto-link the exchange sample_idx to the exam sample_idx.
SAMPLE_IDX_RENAME = {
    'exchange': 'ExchangeBlockID',
    'exam':     'PatientVisitID',
}

# Prefix applied to every non-key column of each kind, so Qlik's
# associative model only links the two tables on the intentional keys.
COLUMN_PREFIX = {
    'exchange': 'Exch_',
    'exam':     'Exam_',
}

# Columns that survive the prefix pass unchanged. DataSource + SN are
# the two intentional join keys between Exchange and Exam. The two ID
# columns are already unique per kind after the sample_idx rename above
# and shouldn't be prefixed either (they'd look silly as Exch_ExchangeBlockID).
KEEP_UNPREFIXED = {'DataSource', 'SN', 'ExchangeBlockID', 'PatientVisitID'}


def _read_one(path: str, data_source: str) -> pd.DataFrame:
    """Read a single DATA_{serial}.csv and tag it with DataSource."""
    df = pd.read_csv(path, low_memory=False)
    df.insert(0, 'DataSource', data_source)
    return df


def _backfill_exam_timediff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 02 older runs didn't emit a `timediff` column for exam rows.
    Step 05 always emits one for synthetic. Without this backfill, the
    combined exam CSV ends up with `timediff` populated for synthetic
    but NaN for real — any inter-exam-gap chart in Qlik looks broken.

    If the column is already fully populated, this is a no-op. Otherwise,
    we compute it per (DataSource, SN) from the startTime deltas, which
    matches the semantic step 02/05 both use (seconds between this exam's
    startTime and the previous exam's startTime on the same scanner).
    """
    if 'timediff' in df.columns and df['timediff'].notna().all():
        return df
    if 'startTime' not in df.columns:
        return df

    if 'timediff' not in df.columns:
        df['timediff'] = float('nan')

    st = pd.to_datetime(df['startTime'], errors='coerce')
    # Stable sort on the null-only subset; compute per-(source, scanner) diff.
    needs_fill = df['timediff'].isna()
    if not needs_fill.any():
        return df

    order = df.index  # remember original order
    tmp = df.assign(_st=st).sort_values(['DataSource', 'SN', '_st'], kind='stable')
    filled = tmp.groupby(['DataSource', 'SN'])['_st'].diff().dt.total_seconds().fillna(0)
    tmp['_filled'] = filled
    tmp = tmp.loc[order]  # restore original row order
    df.loc[needs_fill, 'timediff'] = tmp.loc[needs_fill, '_filled'].values
    return df


def _backfill_exam_sourceid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 02 doesn't emit a `sourceID` column for real exam rows, but step 05
    does (always 'MRI_MSR_104' on success, 'MRI_MSR_34' on abort). To make
    the column comparable across Real and Synthetic in Qlik, derive it from
    FinishEvent on rows where it's missing.
    """
    if 'FinishEvent' not in df.columns:
        return df
    if 'sourceID' not in df.columns:
        df['sourceID'] = pd.NA
    needs_fill = df['sourceID'].isna()
    if not needs_fill.any():
        return df
    df.loc[needs_fill, 'sourceID'] = (
        df.loc[needs_fill, 'FinishEvent'].map(FINISH_TO_SOURCEID).fillna('MRI_MSR_34')
    )
    return df


def _normalize_exam_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align the two categorical columns whose *spelling* differs between the
    real and synthetic exam CSVs. Both are pure formatting mismatches — Qlik
    treats 'Head' and 'HEAD' as two unrelated dimension values, so a
    side-by-side chart on either column shows two disjoint sets of bars and
    reads as "the model generated nothing".

      BodyGroup  real 'Head' / synthetic 'HEAD'  -> upper-case both sides.
                 Upper-case is the target because the exchange table already
                 uses it (Exch_BodyGroup_from_text / _to_text), so after this
                 all three body-region columns share one vocabulary and can
                 cross-filter.

      Sequence   real '%SiemensSeq%\\fl3d_vibe' / synthetic 'vibe'  -> run the
                 canonical step-03 classifier over both sides. A plain prefix
                 strip is not enough: the synthetic vocabulary is coarser
                 ('vibe' covers fl3d_vibe, fl3d_ce and fl3d_rd), so only the
                 real classifier reproduces the families the model was
                 actually trained on. Idempotent on already-classified values,
                 which is why it is safe to apply to the synthetic side too.
    """
    if 'BodyGroup' in df.columns:
        df['BodyGroup'] = df['BodyGroup'].astype('string').str.upper()

    if 'Sequence' in df.columns:
        df['Sequence'] = df['Sequence'].map(
            lambda v: ID_TO_SEQUENCE_TYPE[classify_sequence_type(v)]
        )
    return df


def _add_exam_sequence_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a per-visit total measurement count, replicated to every row of the
    visit. Distinct from StepCount (which is the within-visit row index):
    SequenceCount tells you 'this patient visit had N total scans' on every
    row, which is what Qlik needs to slice a histogram of scans-per-visit.

    Grouping key is (DataSource, SN, sample_idx) — that's one patient visit
    on one scanner from one source. Done before SAMPLE_IDX_RENAME, since we
    still have the raw sample_idx column at this stage.
    """
    if 'sample_idx' not in df.columns:
        return df
    df['SequenceCount'] = df.groupby(
        ['DataSource', 'SN', 'sample_idx'], sort=False
    )['sample_idx'].transform('size')
    return df


def consolidate(kind: str) -> str | None:
    """
    Merge per-scanner CSVs for one kind (exchange or exam) into one file.
    Returns the output path, or None if no files were found.
    """
    print(f"\n{'=' * 64}")
    print(f"Consolidating {kind.upper()}")
    print('=' * 64)

    pieces = []
    totals = {'Real': 0, 'Synthetic': 0}

    for source_dir, label in (('real', 'Real'), ('synthetic', 'Synthetic')):
        folder = os.path.join(DATA_DIR, source_dir, kind)
        if not os.path.isdir(folder):
            print(f"  {label:<10} folder not found — skipping ({folder})")
            continue

        files = sorted(glob(os.path.join(folder, 'DATA_*.csv')))
        if not files:
            print(f"  {label:<10} no DATA_*.csv files in {folder}")
            continue

        for path in files:
            serial = os.path.basename(path).replace('DATA_', '').replace('.csv', '')
            df = _read_one(path, label)
            pieces.append(df)
            totals[label] += len(df)
            print(f"  {label:<10} {serial}: {len(df):>8,} rows, {len(df.columns):>3} cols")

    if not pieces:
        print("  Nothing to consolidate. Skipping.")
        return None

    # Concatenate — pandas takes the union of columns, filling missing cells
    # with NaN.  This handles exam files with scanner-specific coil columns
    # gracefully.
    combined = pd.concat(pieces, ignore_index=True, sort=False)

    # Back-fill exam-side timediff from startTime when step 02 didn't emit it
    # (older real CSVs). No-op once 02_exam_preprocessing.py's timediff
    # patch has been rerun on Databricks and the fresh files are downloaded.
    if kind == 'exam':
        combined = _backfill_exam_timediff(combined)
        # Backfill sourceID on real (step 02 doesn't emit it; derive from
        # FinishEvent so the column is comparable in Qlik). Run before the
        # sample_idx rename so SequenceCount can group on the raw column.
        combined = _backfill_exam_sourceid(combined)
        combined = _add_exam_sequence_count(combined)
        # Align BodyGroup casing and Sequence vocabulary across the two
        # sides so Qlik can actually compare them on those dimensions.
        combined = _normalize_exam_labels(combined)

    # Drop misleading single-side / constant-valued columns
    to_drop = [c for c in DROP_COLS[kind] if c in combined.columns]
    if to_drop:
        combined = combined.drop(columns=to_drop)
        print(f"  (dropped {len(to_drop)} misleading col{'s' if len(to_drop)!=1 else ''}: {to_drop})")

    # Disambiguate sample_idx
    if 'sample_idx' in combined.columns:
        combined = combined.rename(columns={'sample_idx': SAMPLE_IDX_RENAME[kind]})

    # Sort for human-readable ordering (and to keep git diffs stable if you
    # ever want to inspect the combined files). Done before the prefix pass
    # so we can still reference the unprefixed timestamp column.
    sort_cols = ['DataSource']
    if 'SN' in combined.columns:
        sort_cols.append('SN')
    if TIME_COL[kind] in combined.columns:
        sort_cols.append(TIME_COL[kind])
    combined = combined.sort_values(sort_cols, kind='stable').reset_index(drop=True)

    # Prefix every non-key column so Qlik only auto-links the two tables
    # on SN and DataSource. This is the whole reason we bother with a
    # consolidation script instead of uploading per-scanner files.
    prefix = COLUMN_PREFIX[kind]
    combined = combined.rename(columns={
        c: (c if c in KEEP_UNPREFIXED else f'{prefix}{c}')
        for c in combined.columns
    })

    # Reorder columns so a human opening the CSV sees meaningful fields
    # first. Without this, the exam file leads with ~60 mostly-False coil
    # columns and looks empty in a spreadsheet preview. Order:
    #   1. The unprefixed keys (DataSource, SN, ExchangeBlockID/PatientVisitID)
    #   2. All prefixed non-coil columns (preserves original discovery order)
    #   3. Coil columns last (anything containing '_#' — e.g. Exam_#0_BC)
    key_order  = ['DataSource', 'SN', 'ExchangeBlockID', 'PatientVisitID']
    keys_first = [c for c in key_order if c in combined.columns]
    remaining  = [c for c in combined.columns if c not in keys_first]
    non_coil   = [c for c in remaining if '_#' not in c]
    coils      = [c for c in remaining if '_#' in c]
    combined = combined[keys_first + non_coil + coils]

    os.makedirs(COMBINED_DIR, exist_ok=True)
    out_path = os.path.join(COMBINED_DIR, f'{kind}_combined.csv')
    combined.to_csv(out_path, index=False)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\n  → {out_path}")
    print(f"    {len(combined):>8,} rows, {len(combined.columns)} cols, {size_mb:.1f} MB")
    print(f"    Real:      {totals['Real']:>8,} rows")
    print(f"    Synthetic: {totals['Synthetic']:>8,} rows")
    return out_path


def main() -> int:
    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: data folder not found at {DATA_DIR}")
        print("       Run this script from DatabricksPipeline/csv_pipeline/qlik/")
        return 1

    outputs = []
    for kind in ('exchange', 'exam'):
        out = consolidate(kind)
        if out:
            outputs.append(out)

    if not outputs:
        print("\nNo output files produced.")
        print("Check that data/real/ and/or data/synthetic/ contain DATA_*.csv files.")
        print("See fetch_from_dbfs.md for how to populate them.")
        return 1

    print(f"\n{'=' * 64}")
    print("DONE — upload these two files to Qlik:")
    print('=' * 64)
    for p in outputs:
        print(f"  {p}")
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
