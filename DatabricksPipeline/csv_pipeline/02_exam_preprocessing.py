# Databricks notebook source
# Databricks notebook — Exam CSV Preprocessing
#
# Follows Patient_Exam_updated_with_new_columns_March2026.ipynb workflow exactly.
# Loops over all 10 serial numbers and saves one CSV per serial to:
#   /dbfs/FileStore/csv_pipeline/exam/DATA_{serial}.csv
#
# Each row in the output represents one completed measurement.
# Runtime: << 1 hour for 10 serials over a 1-month window.

# COMMAND ----------
%pip install openpyxl

# COMMAND ----------
%run ./config

# COMMAND ----------

import re
import os
import gc
import time
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
import logging

logger = logging.getLogger("csv_pipeline.exam")
os.makedirs(EXAM_OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {EXAM_OUTPUT_DIR}")

# ---------------------------------------------------------------------------
# Arrow for toPandas(). This is the memory fix, not a tuning knob.
#
# Without Arrow, toPandas() materialises every row as a Python Row object on
# the driver before it builds the DataFrame. This loop pulls 56k-190k rows per
# serial, so the peak is several times the size of the frame that survives,
# and it is paid twice per serial (eventlog + examination_workflow). That is
# what restarted the driver on 2026-08-31 after 11 of 21 serials.
#
# fallback.enabled keeps a column type Arrow cannot handle from becoming an
# error: PySpark retries that conversion on the old path instead of failing.
# ---------------------------------------------------------------------------
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")

# ---- Lightweight section timing (see step 03 header for rationale) ----
_TIMINGS = []
def _timeit(label, t0):
    dt = time.perf_counter() - t0
    _TIMINGS.append((label, dt))
    print(f"[timing] step02  {label:<20} {dt:9.1f}s")
    return time.perf_counter()

# COMMAND ----------
# =============================================================================
# HELPER FUNCTIONS  (faithful port of Patient_Exam_updated_with_new_columns_March2026.ipynb)
# =============================================================================

def ptab(df, first_ptab):
    """
    Populate the PTAB column from MRI_FRR_257 events.
    Rows before the first FRR_257 event receive first_ptab.
    """
    start    = True
    df['PTAB'] = np.nan
    for idx, row in df.iterrows():
        if start and idx > 0:
            df.at[idx, 'PTAB'] = first_ptab
        if row['sourceID'] == 'MRI_FRR_257':
            df.at[idx, 'PTAB'] = row['text'].split()[-1]
            start = False
    return df


def expand_coils(coils_str, group):
    """
    Parse compact coil notation into a list of prefixed coil IDs.

    Args:
        coils_str: e.g. 'HC1-4,NC1,2'
        group:     receiver group string, e.g. '0' or '1'

    Returns:
        list of strings like ['#0_HC1', '#0_HC2', '#0_HC3', '#0_HC4', '#0_NC1', '#0_NC2']
    """
    connected_coil = coils_str.split(',')
    sc   = []
    coil = ''   # last seen alphabetic prefix

    for elem in connected_coil:
        if not elem:
            continue
        # Strip trailing dash or period (data artefacts)
        if elem.endswith('-') or elem.endswith('.'):
            elem = elem[:-1]
        if not elem.isnumeric():
            elem = elem.strip()
            coil = elem[:2]     # first two chars are always the letter prefix
            if '-' in elem:
                st     = elem[2:]
                st_sp  = st.split('-')
                if len(st_sp) > 1:
                    try:
                        for i in range(int(st_sp[0]), int(st_sp[1]) + 1):
                            sc.append(f'#{group}_{coil}{i}')
                    except ValueError:
                        pass
            else:
                sc.append(f'#{group}_{elem}')
        else:
            sc.append(f'#{group}_{coil}{elem}')
    return sc


def get_seq(text):
    """Extract Sequence and Protocol names from an MRI_MSR_100 message."""
    sequence_type = re.search(r"Sequence: '(.*)'", text).group(1)
    protocol      = re.search(r"Protocol: '(.*)',", text).group(1)
    return sequence_type, protocol


def get_body_patient(text):
    """
    Extract BodyPart and anonymised PatientID from an MRI_EXU_95 message.

    Returns:
        body        — body part string (or NaN on parse failure)
        patient_id  — anonymised patient ID string (or NaN on parse failure)
        flag_patient — True if patient_id was successfully parsed
    """
    body = np.nan
    try:
        body = re.search(r'with body part < (.*) >', text).group(1)
        if ' ' in body:     # may contain contrast-agent suffix e.g. '<BRAIN> <CONTRAST>'
            try:
                body = re.search(r'with body part < (.*) > <', text).group(1)
            except AttributeError:
                pass
    except AttributeError:
        pass

    try:
        patient_id   = re.search(r'Anonymised Patient ID < (.*) >', text).group(1)
        flag_patient = True
    except AttributeError:
        patient_id   = np.nan
        flag_patient = False

    return body, patient_id, flag_patient


def data_transformation(df_):
    """
    Core ETL: identify measurement boundaries in the event log and extract
    per-measurement features.

    Measurement start:  MRI_MSR_100
    Measurement end:    MRI_MSR_104 (Successful), MRI_MSR_34 (Stopped by User),
                        MRI_MSR_26 (Scanning Error), MRI_MSR_22/24/25/40 (other stops)
    Connected coils:    MRI_MSR_101 (individual coil element rows)
    DOT workflow:       MRI_EXU_89 (DOT examination open)
    Patient info:       MRI_EXU_95

    Returns:
        df_out    — input DataFrame with additional feature columns attached
        coilList  — list of unique coil binary column names seen in this scanner
    """
    # Drop adjustment and service sequences
    df_  = df_[~df_['text'].str.contains('AdjustSeq|ServiceSeq', case=False)]
    df_  = df_.reset_index(drop=True)

    df   = pd.DataFrame(index=range(df_.shape[0]), columns=['Sequence'])
    df['Sequence'] = 'False'

    finish_events         = ['MRI_MSR_104', 'MRI_MSR_18', 'MRI_FRR_18', 'MRI_MSR_21',
                             'MRI_MSR_34',  'MRI_MSR_26', 'MRI_MSR_22', 'MRI_MSR_40',
                             'MRI_MSR_25',  'MRI_MSR_24']
    finish_events_reduced = ['MRI_MSR_104', 'MRI_MSR_34', 'MRI_MSR_26',
                             'MRI_MSR_22',  'MRI_MSR_40', 'MRI_MSR_25', 'MRI_MSR_24']

    sc           = []       # coil ID list for current measurement
    coilList     = []       # all unique coil column names across this scanner
    start        = False    # inside a measurement?
    flag_dot     = False    # DOT workflow active?
    startEXU     = False    # MRI_EXU_95 found after MRI_MSR_100?
    outEXU       = False    # MRI_EXU_95 found just before MRI_MSR_100?
    conCoils     = ''       # ConnectedCoils string (from MRI_CCS_1008)
    body         = np.nan
    patient_id   = np.nan
    flag_patient = False
    datetime_start = None
    start_index    = 0
    start_index_prev = 0

    for i, row in df_.iterrows():

        if row.sourceID == 'MRI_EXU_89':
            flag_dot = True

        if row.sourceID == 'MRI_CCS_1008':
            coil_ids = re.findall(r"CoilID\s+'(\w+)'", row.text)
            conCoils = '-'.join(coil_ids)

        if row.sourceID == 'MRI_MSR_100':
            start = True
            try:
                seq, protocol = get_seq(row.text)
            except (AttributeError, TypeError):
                start = False
                continue
            datetime_start   = row.datetime
            start_index_prev = start_index
            start_index      = i
            startEXU         = False

        if start and row.sourceID == 'MRI_EXU_95':
            body, patient_id, flag_patient = get_body_patient(row.text)
            startEXU = True

        if start and row.sourceID == 'MRI_MSR_101':
            s = row.text
            try:
                group         = re.search(r'#(\d):', s).group(1)
                selected_coil = re.search(r": '(.*)'\)", s).group(1)
                expanded      = expand_coils(selected_coil, group)
                if not sc:
                    sc = expanded
                else:
                    sc.extend(expanded)
            except (AttributeError, TypeError):
                pass

        if row.sourceID in finish_events:
            if start and row.sourceID in finish_events_reduced:
                df_.at[i, 'startTime'] = datetime_start

                for elem in sc:
                    if elem not in df_.columns:
                        coilList.append(elem)
                    df_.at[i, elem] = True

                df.at[i, 'Sequence'] = seq
                df.at[i, 'Protocol'] = protocol

                # FinishEvent label
                finish_map = {
                    'MRI_MSR_104': 'Successful',
                    'MRI_MSR_34':  'Stopped by User',
                    'MRI_MSR_26':  'Scanning Error',
                    'MRI_MSR_22':  'Start MeasSys Failed',
                    'MRI_MSR_24':  'Stopped by Scanner',
                    'MRI_MSR_25':  'Stopped by ImageR',
                    'MRI_MSR_40':  'Stopped by CoilChange',
                }
                df.at[i, 'FinishEvent'] = finish_map.get(row.sourceID, 'Unknown')

                # BodyPart / PatientID — try to recover if MRI_EXU_95 was missed
                if not startEXU:
                    for offset in (-2, -1):
                        k = start_index + offset
                        if 0 <= k < len(df_) and df_['sourceID'].iloc[k] == 'MRI_EXU_95':
                            outEXU = True
                            break
                    if not outEXU:
                        # Last resort: check time distance to previous start
                        try:
                            curr_t = pd.to_datetime(df_['datetime'].iloc[start_index])
                            prev_t = pd.to_datetime(df_['datetime'].iloc[start_index_prev])
                            if (curr_t - prev_t) <= pd.Timedelta(minutes=4) and pd.notna(body):
                                startEXU = True
                        except (ValueError, TypeError, KeyError):
                            pass

                if outEXU:
                    try:
                        body, patient_id, flag_patient = get_body_patient(df_.iloc[k].text)
                        startEXU = True
                    except Exception:
                        pass
                    outEXU = False

                if startEXU:
                    df_.at[i, 'BodyPart'] = body
                    if flag_patient:
                        df_.at[i, 'PatientID'] = patient_id
                    else:
                        df_.at[i, 'MissingPatientID'] = True
                else:
                    df_.at[i, 'MissingBodyPart']  = True
                    df_.at[i, 'MissingPatientID'] = True

                df_.at[i, 'ConnectedCoils'] = conCoils if conCoils else np.nan
                if not conCoils:
                    df_.at[i, 'MissingConCoils'] = True

                df_.at[i, 'DOT'] = flag_dot

                df.fillna(False, inplace=True)

            start    = False
            startEXU = False
            sc       = []

        if row.sourceID == 'MRI_MPT_1005':
            flag_dot = False

    df_out = pd.concat([df_, df], axis=1)
    return df_out, coilList


def measurement_time(df):
    """
    Add endTime and duration (seconds) columns.
    Sorts measurements by datetime; drops rows where Sequence == 'False'.
    """
    df['datetime'] = pd.to_datetime(df['datetime'])
    df_sorted = df.sort_values(by='datetime')
    df_sorted = df_sorted.drop(df_sorted[df_sorted['Sequence'] == 'False'].index)
    df_sorted.rename(columns={'datetime': 'endTime'}, inplace=True)
    df_sorted['startTime'] = pd.to_datetime(df_sorted['startTime'])
    df_sorted['duration']  = (df_sorted['endTime'] -
                               df_sorted['startTime']).dt.total_seconds()
    return df_sorted


def order_cols(df, joint_coil_list):
    """
    Re-order columns: binary coil columns first, then metadata, then timestamps.
    """
    df_out = pd.DataFrame()
    joint_coil_list = list(set(joint_coil_list))

    existing_coils  = [c for c in joint_coil_list if c in df.columns]
    move_coils      = pd.concat([df.pop(x) for x in existing_coils], axis=1)
    df_out          = pd.concat([df_out, move_coils], axis=1)

    for col_name in ['BodyPart', 'Sequence', 'ConnectedCoils', 'DOT',
                     'PatientID', 'Protocol', 'SN', 'FinishEvent', 'sourceID']:
        if col_name in df.columns:
            df_out = pd.concat([df_out, df.pop(col_name)], axis=1)

    if any(c.startswith('Missing') for c in df.columns):
        missing_cols = [c for c in df.columns if 'Missing' in c]
        move_missing = pd.concat([df.pop(x) for x in missing_cols], axis=1)
        df_out       = pd.concat([df_out, move_missing], axis=1)

    df_out = df_out.fillna(False)

    for col_name in ['startTime', 'duration', 'endTime']:
        if col_name in df.columns:
            series = df.pop(col_name)
            if col_name == 'duration':
                series = series.astype('int64')
            df_out = pd.concat([df_out, series], axis=1)

    return df_out


def match_bp(df_info, df_bp):
    """Merge body part group information from the body mapping Excel."""
    df_info = df_info.copy()
    df_bp   = df_bp.copy()
    df_info['bp_lower'] = df_info['BodyPart'].astype(str).str.lower()
    df_bp['bp_lower']   = df_bp['BodyPart'].astype(str).str.lower()
    df_bp.drop(columns='BodyPart', inplace=True)
    merged = pd.merge(df_info, df_bp, on='bp_lower', how='left')
    merged['BodyGroup'] = merged['BodyGroup'].fillna('Unknown')
    merged.drop(columns='bp_lower', inplace=True)
    return merged

# COMMAND ----------
# =============================================================================
# CELL: Load body group mapping from Excel
# =============================================================================

df_bp = pd.read_excel(BODY_GROUP_MAPPING_PATH)
if 'MRType' in df_bp.columns:
    df_bp.drop(columns='MRType', inplace=True)
print(f"Body mapping rows: {len(df_bp)}")

# COMMAND ----------
# =============================================================================
# CELL: Per-serial processing loop
# =============================================================================

_t = time.perf_counter()
# Per-serial outcome, printed as a summary after the loop.
#
# WHY THIS EXISTS. A serial that produces no exam CSV disappears silently: the
# skip is one line in the middle of a 21-scanner run, and by the time it
# matters the log has scrolled. It matters more than it looks, because
# 03_build_preprocessed_pkl.py skips serials with no exam CSV when it builds
# customer_schedules (03:913-916), and csv_pipeline_seqparams/07 generates one
# synthetic scanner per customer_schedules key — so a scanner missing HERE is a
# scanner missing from the synthetic half of the Qlik comparison, three
# notebooks downstream, with nothing at that point to say why.
#
# That is exactly what happened to 155687 in the 2026-08-31 run: 20 synthetic
# scanners against 21 configured, traced back to this loop.
_OUTCOMES = {}

# ---------------------------------------------------------------------------
# RESUME. The written CSVs are the checkpoint.
#
# On 2026-08-31 the driver was restarted by Databricks partway through serial
# 155687, after 11 serials had already been written. A restarted kernel loses
# every Python name but none of the files, so without this the only way
# forward was to re-run all 11 — each one a full eventlog scan plus a
# six-figure-row toPandas().
#
# A serial whose CSV already exists WITH ROWS IN IT is skipped and its row
# count recorded, so the coverage summary below still reports on all 21 after
# a resumed run. A zero-row CSV is NOT treated as done: that is the 155687
# failure mode this notebook exists to surface, and skipping it would hide it.
#
# Set RESUME=0 to force a full rebuild.
# ---------------------------------------------------------------------------
RESUME = os.environ.get('RESUME', '1') != '0'

# Columns this notebook is currently expected to emit. A CSV on DBFS that is
# missing any of them was written by OLDER code and must be rebuilt, however
# many rows it has.
#
# WHY. The first version of this resume check asked only "does a non-empty CSV
# exist", and the 2026-08-31 re-run happily skipped nine serials whose files
# dated from 2026-08-04 — before commit 28ed58b added `sourceID`. The bundle
# went out with nine scanners whose finish codes had to be reconstructed from
# FinishEvent downstream, which folds every abort into MRI_MSR_34. Resuming
# must never silently mix two schemas.
#
# ADD TO THIS LIST whenever you add a column to the output. That is what makes
# stale files rebuild instead of surviving.
_SCHEMA_MARKERS = ('sourceID', 'timediff', 'customer_idx', 'sample_idx',
                   'predicted_mu', 'BodyGroup', 'StepCount', 'PTAB')


def _existing_output(path):
    """(row_count, missing_columns) for an already-written CSV, else None.

    Rows are counted by newline rather than parsed: these files carry up to
    90+ coil columns and the only questions being asked are whether this
    serial finished and whether the current code wrote it.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            header = handle.readline()
            if not header:
                return (0, list(_SCHEMA_MARKERS))
            cols = {c.strip() for c in header.rstrip('\n').split(',')}
            return (sum(1 for _ in handle), [m for m in _SCHEMA_MARKERS
                                             if m not in cols])
    except OSError:
        return None

if RESUME:
    print("RESUME=1 — serials with a non-empty CSV already on DBFS are skipped.")

for serial_number in SERIAL_NUMBERS:
    print(f"\n{'='*60}")
    print(f"Processing serial: {serial_number}")
    print(f"{'='*60}")
    _OUTCOMES[serial_number] = 'no eventlog rows in window'

    csv_path = f"{EXAM_OUTPUT_DIR}/DATA_{serial_number}.csv"
    _prev    = _existing_output(csv_path) if RESUME else None
    if _prev:
        _rows, _missing = _prev
        if _missing:
            print(f"  Existing CSV is from older code (missing "
                  f"{', '.join(_missing)}) — rebuilding.")
        elif _rows:
            _OUTCOMES[serial_number] = _rows
            print(f"  Already built ({_rows:,} rows) — skipping. "
                  f"Set RESUME=0 to rebuild.")
            continue

    # -------------------------------------------------------------------------
    # Load eventlog for this serial
    # -------------------------------------------------------------------------
    result_spark = (
        spark.read.table(EVENTLOG_TABLE)
        .select(
            F.date_format(F.col("EventDateTime"), "yyyy-MM-dd HH:mm:ss").alias("EventDateTime"),
            "SerialNumber",
            "MessageIdentification",
            "TimeZoneOffset",
            "Message",
            "Line",
        )
        .filter(F.col("SerialNumber").cast("long") == int(serial_number))
        .withColumn(
            "AdjustedEventDateTime",
            F.timestamp_seconds(
                F.unix_timestamp(F.col("EventDateTime")) + F.col("TimeZoneOffset")
            ),
        )
        .filter(F.to_date(F.col("AdjustedEventDateTime")).between(DATE_START, DATE_END))
        .filter(
            F.col("MessageIdentification").startswith("MRI_MSR") |
            F.col("MessageIdentification").isin(EXAM_EXTRA_SOURCE_TYPES)
        )
        .select(
            F.date_format(F.col("AdjustedEventDateTime"), "yyyy-MM-dd HH:mm:ss").alias("AdjustedEventDateTime"),
            "MessageIdentification",
            "Message",
            "Line",
        )
        .orderBy(F.col("AdjustedEventDateTime").asc())
    )
    result_spark = result_spark.toDF("datetime", "sourceID", "text", "Line")
    pandas_df    = result_spark.toPandas()

    if pandas_df.empty:
        print(f"  No events found — skipping.")
        continue

    print(f"  Events loaded: {len(pandas_df):,}")

    # -------------------------------------------------------------------------
    # Extract PTAB from MRI_FRR_257 events
    # -------------------------------------------------------------------------
    pandas_df          = ptab(pandas_df, 0)
    pandas_df['PTAB']  = pandas_df['PTAB'].ffill().bfill()

    # -------------------------------------------------------------------------
    # Main ETL
    # -------------------------------------------------------------------------
    df_ini, coil_list = data_transformation(pandas_df)

    # Keep only rows that are measurement-end events with a valid FinishEvent
    finish_keep = ['MRI_MSR_104', 'MRI_MSR_34', 'MRI_MSR_26',
                   'MRI_MSR_22',  'MRI_MSR_40', 'MRI_MSR_25', 'MRI_MSR_24']
    df_select = df_ini[
        df_ini['sourceID'].isin(finish_keep) & (df_ini['FinishEvent'] != False)
    ]
    df_pre = df_select.copy()
    # `text` is the raw log line and has served its purpose by here. `sourceID`
    # is KEPT: it is the actual finish code this measurement ended on
    # (MRI_MSR_104 on success, MRI_MSR_34/26/22/40/25/24 on the various aborts),
    # and step 05/07 emit exactly that column on the synthetic side. Dropping it
    # left the real half of the Qlik comparison with no sourceID at all, so
    # qlik/consolidate.py had to reconstruct one from FinishEvent through a
    # 5-entry map that folds every abort code into MRI_MSR_34 — an invented
    # value standing in for one we had and threw away.
    df_pre.drop(columns=['text'], axis=1, inplace=True)

    # -------------------------------------------------------------------------
    # Compute duration and sort
    # -------------------------------------------------------------------------
    df_sorted = measurement_time(df_pre)
    df_sorted.fillna(False, inplace=True)
    df_sorted['SN'] = str(serial_number)

    df_sorted['FinishEvent'] = df_sorted['FinishEvent'].fillna(False)
    df_sorted = df_sorted[df_sorted['FinishEvent'] != False]

    # -------------------------------------------------------------------------
    # Order columns
    # -------------------------------------------------------------------------
    df = order_cols(df_sorted, coil_list)

    # Restore PTAB (order_cols does not handle it)
    if 'PTAB' in df_sorted.columns:
        df['PTAB'] = df_sorted['PTAB'].values

    # -------------------------------------------------------------------------
    # PatientID / BodyPart forward-fill corrections (same as notebook)
    # -------------------------------------------------------------------------
    df['startTime'] = pd.to_datetime(df['startTime'])
    df['endTime']   = pd.to_datetime(df['endTime'])

    s1  = df['PatientID'].replace({'False': np.nan})
    s1b = df['BodyPart'].replace({'False': np.nan})

    # Fill localizer / scout rows that have no PatientID yet
    mask1 = (
        (df['PatientID'] == 'False') &
        (df['Sequence'].str.contains('Scout', case=False) |
         df['Protocol'].str.contains('localizer|Scout', case=False))
    )
    df.loc[mask1, 'PatientID'] = s1.bfill()
    df.loc[mask1, 'BodyPart']  = s1b.bfill()

    s3  = s1.ffill()
    s3b = s1b.ffill()
    mask2 = s3.eq(s1.bfill())
    df.loc[mask2, 'PatientID'] = s3
    df.loc[mask2, 'BodyPart']  = s3b

    df['PatientID'] = df['PatientID'].ffill()
    df['BodyPart']  = df['BodyPart'].ffill()

    mask3 = (df['PatientID'] == 'False') & \
            (df['ConnectedCoils'] == df['ConnectedCoils'].shift())
    df.loc[mask3, 'PatientID'] = df['PatientID'].shift()
    df.loc[mask3, 'BodyPart']  = df['BodyPart'].shift()

    df['PatientID'] = df['PatientID'].ffill()
    df['BodyPart']  = df['BodyPart'].ffill()

    mask4 = (
        (df['PatientID'] == 'False') &
        ((df['endTime'].shift(-1) - df['startTime']) <= pd.Timedelta(minutes=3))
    )
    df.loc[mask4, 'PatientID'] = df['PatientID'].shift(-1)
    df.loc[mask4, 'BodyPart']  = df['BodyPart'].shift(-1)

    # Drop zero-duration successful measurements and outliers > 4000 s
    mask5     = (df['duration'] == 0) & (df['FinishEvent'] == 'Successful')
    df        = df[~mask5]
    df_drop   = df.drop(df[df['duration'] > 4000].index)
    df        = df_drop.copy()

    # -------------------------------------------------------------------------
    # Add BodyGroup from Excel mapping
    # -------------------------------------------------------------------------
    df = match_bp(df, df_bp)

    # -------------------------------------------------------------------------
    # Pause time and step count
    # -------------------------------------------------------------------------
    df['pauseTime'] = (df['startTime'].shift(-1) - df['endTime']).dt.total_seconds()
    df['StepCount'] = df.groupby('PatientID').cumcount() + 1
    df.loc[df['PatientID'] != df['PatientID'].shift(-1), 'pauseTime'] = 0

    # timediff — seconds between this exam's startTime and the previous
    # exam's startTime on the same scanner. Produced here so it matches
    # the column that step 05 writes for synthetic CSVs; without it the
    # qlik combined file has timediff populated for synthetic but NaN
    # for real, which makes any inter-exam-gap chart look broken.
    df = df.sort_values('startTime').reset_index(drop=True)
    df['timediff'] = df['startTime'].diff().dt.total_seconds().fillna(0)

    # -------------------------------------------------------------------------
    # Merge examination_workflow for Age, Weight, Height, Direction
    # -------------------------------------------------------------------------
    start_int = int(DATE_START.replace('-', ''))
    end_int   = int(DATE_END.replace('-', ''))

    try:
        exam_df = spark.read.table(EXAMINATION_TABLE)
        exam_filtered = (
            exam_df
            .filter(F.col("SerialNumber").cast("long") == int(serial_number))
            .filter(
                (F.col("Year").cast("int") * 10000 +
                 F.col("Month").cast("int") * 100 +
                 F.col("Day").cast("int")).between(start_int, end_int)
            )
            .withColumn("PatientId", F.col("WorkflowValues")["PatientId"])
            .withColumn("Age",       F.col("WorkflowValues")["Age"])
            .withColumn("Weight",    F.col("WorkflowValues")["Weight"])
            .withColumn("Height",    F.col("WorkflowValues")["Height"])
            .withColumn("Direction", F.col("WorkflowValues")["Direction"])
            .select("PatientId", "Age", "Weight", "Height", "Direction")
            .dropDuplicates(["PatientId"])
        )
        exam_pdf = exam_filtered.toPandas()
        df = df.merge(exam_pdf, left_on='PatientID', right_on='PatientId', how='left')
        if 'PatientId' in df.columns:
            df.drop(columns=['PatientId'], inplace=True)
        print(f"  Examination data merged: {len(exam_pdf):,} patients")
    except Exception as e:
        print(f"  Warning: examination_workflow merge failed: {e}")
        for col_name in ['Age', 'Weight', 'Height', 'Direction']:
            df[col_name] = np.nan

    # -------------------------------------------------------------------------
    # Add schema columns to match 05_generate_synthetic_data.py output, then save
    # -------------------------------------------------------------------------
    df['customer_idx'] = SERIAL_NUMBERS.index(serial_number)

    # sample_idx — increments each time PatientID changes (one patient = one sample)
    df['sample_idx'] = (
        df['PatientID'] != df['PatientID'].shift(1, fill_value='__START__')
    ).cumsum() - 1

    # Model columns — not available at preprocessing time
    df['predicted_mu']     = float('nan')
    df['predicted_sigma']  = float('nan')
    df['sampled_duration'] = float('nan')

    # csv_path was resolved at the top of this iteration, by the resume check.
    df.to_csv(csv_path, index=False, header=True)
    _OUTCOMES[serial_number] = len(df)
    print(f"  Saved {len(df):,} rows → {csv_path}")

    # -------------------------------------------------------------------------
    # Release the driver before the next serial
    # -------------------------------------------------------------------------
    # None of these are read after the CSV is written, but Python only frees a
    # name when it is rebound, and the loop rebinds them at different points —
    # so the biggest ones (pandas_df at up to 190k rows, df_ini after coil
    # expansion at 90+ columns) stay live across the iteration boundary and
    # overlap with the next serial's load. Dropping them explicitly means the
    # driver holds one serial at a time instead of two.
    for _name in ('pandas_df', 'result_spark', 'df_ini', 'df_select', 'df_pre',
                  'df_sorted', 'df_drop', 'df', 'exam_df', 'exam_filtered',
                  'exam_pdf', 's1', 's1b', 's3', 's3b',
                  'mask1', 'mask2', 'mask3', 'mask4', 'mask5'):
        globals().pop(_name, None)
    gc.collect()

print("\nExam preprocessing complete.")
_timeit('per-serial loop', _t)

# COMMAND ----------
# =============================================================================
# CELL: Coverage summary — which scanners actually produced a CSV
# =============================================================================

def _has_rows(r):
    # A 0-row CSV is as damaging as a missing one: step 03 still writes a
    # customer_schedules entry for it, but an empty one, and seqparams/07 then
    # generates no days for that scanner. Both count as "no data" here.
    return isinstance(r, int) and r > 0

_ok      = [s for s, r in _OUTCOMES.items() if _has_rows(r)]
_no_data = [s for s, r in _OUTCOMES.items() if not _has_rows(r)]

print(f"{'serial':>8}  outcome")
print("-" * 40)
for _serial in SERIAL_NUMBERS:
    _r = _OUTCOMES[_serial]
    print(f"{_serial:>8}  {f'{_r:,} rows' if _has_rows(_r) else _r}")

print(f"\n{len(_ok)}/{len(SERIAL_NUMBERS)} scanners produced an exam CSV.")
if _no_data:
    print(f"NO DATA: {_no_data}")
    print("Each of these will also be absent from customer_schedules in step 03, "
          "and therefore from the synthetic output of seqparams/07. Either widen "
          "DATE_START..DATE_END for them or drop them from SERIAL_NUMBERS, so the "
          "real and synthetic halves of the Qlik comparison cover the same set.")

print("\n" + "-"*60)
print("  TIMING BREAKDOWN")
print("-"*60)
for _label, _dt in _TIMINGS:
    print(f"[timing] step02  {_label:<20} {_dt:9.1f}s")
print(f"[timing] step02  {'TOTAL':<20} {sum(d for _, d in _TIMINGS):9.1f}s")

# COMMAND ----------
# =============================================================================
# CELL: One zip, one download link
#
# The alternative is one browser click per scanner. At 21 scanners that is 21
# downloads whose only distinguishing feature is the serial in the filename,
# and the browser silently renames every collision to `DATA_x(1).csv` — which
# is how the 2026-08-31 pull ended up with 40 files in one flat folder,
# exchange and exam distinguishable only by a `(1)` suffix.
#
# Written to local disk first, then copied to FileStore: zipfile through the
# /dbfs FUSE mount is unreliable at this size, and a truncated archive fails at
# unzip time rather than here, where the cause is still on screen.
# =============================================================================

import zipfile
import datetime as _date

_stamp     = _date.date.today().isoformat()
_zip_name  = f"exam_csvs_{_stamp}.zip"
_local_zip = f"/tmp/{_zip_name}"
if os.path.exists(_local_zip):
    os.remove(_local_zip)

_written = 0
with zipfile.ZipFile(_local_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as _zf:
    for _serial in SERIAL_NUMBERS:
        _src = f"{EXAM_OUTPUT_DIR}/DATA_{_serial}.csv"
        if not os.path.exists(_src):
            continue
        # arcname keeps the exam/ folder, so this zip and step 01's unzip into
        # qlik/data/real/ side by side without either overwriting the other —
        # the two halves share filenames and differ only by directory.
        _zf.write(_src, arcname=f"exam/DATA_{_serial}.csv")
        _written += 1
    _zf.writestr("exam/MANIFEST.txt", "\n".join([
        f"generated : {_date.datetime.now().isoformat(timespec='seconds')}",
        "source    : csv_pipeline/02_exam_preprocessing.py",
        f"window    : {DATE_START} .. {DATE_END}  (TZ +{TIMEZONE_OFFSET_HOURS}h)",
        f"scanners  : {_written} of {len(SERIAL_NUMBERS)} configured",
        f"serials   : {', '.join(str(x) for x in _ok)}",
        f"no data   : {', '.join(str(x) for x in _no_data) or '(none)'}",
    ]) + "\n")

_size_mb = os.path.getsize(_local_zip) / (1024 * 1024)
dbutils.fs.mkdirs("dbfs:/FileStore/qlik_bundles")
dbutils.fs.cp(f"file:{_local_zip}", f"dbfs:/FileStore/qlik_bundles/{_zip_name}")
print(f"Zipped {_written} exam CSV(s), {_size_mb:.1f} MB "
      f"-> /dbfs/FileStore/qlik_bundles/{_zip_name}")

displayHTML(f'''
<h3>Exam CSVs — {_zip_name}</h3>
<p>{_written} of {len(SERIAL_NUMBERS)} scanners &nbsp;|&nbsp; {_size_mb:,.1f} MB
   &nbsp;|&nbsp; {DATE_START} &rarr; {DATE_END}</p>
<p style="font-size:1.15em;">
  <a href="/files/qlik_bundles/{_zip_name}" download><b>&#11015; Download {_zip_name}</b></a>
</p>
<pre style="background:#f6f6f6; padding:10px; border-radius:4px;">
cd DatabricksPipeline/csv_pipeline/qlik
unzip -o ~/Downloads/{_zip_name} -d data/real/
python consolidate.py
</pre>
''')
