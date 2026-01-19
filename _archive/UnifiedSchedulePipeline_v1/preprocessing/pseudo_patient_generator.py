"""
Pseudo-Patient Entity Generator
Creates pseudo-patient entities to represent machine idle states

This module separates pause modeling from core examination/exchange models by introducing
pseudo-patient entities that represent the machine's idle state. This allows modeling
transitions (Patient → Pseudo-Patient → Patient) while keeping core models time-independent.
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import PAUSE_DETECTION_THRESHOLD_MINUTES, IDLE_STATE_ENCODING

# ============================================================================
# ENTITY TYPE DEFINITIONS
# ============================================================================

ENTITY_TYPES = {
    'REAL_PATIENT': 0,
    'PSEUDO_PATIENT_PAUSE': 1,   # Inter-session pause
    'PSEUDO_PATIENT_START': 2,   # Start-of-day idle (future use)
    'PSEUDO_PATIENT_END': 3      # End-of-day idle (future use)
}


# ============================================================================
# PSEUDO-PATIENT FEATURE GENERATION
# ============================================================================

def create_pseudo_patient_features(segment_df, pipeline='seqofseq', gap_duration_seconds=None):
    """
    Generate conditioning features for pseudo-patient entities

    Pseudo-patients represent machine idle states. They inherit machine-specific features
    (SystemType, Country) since the machine remains constant, but anatomical features
    (BodyPart, BodyGroup) and patient demographics are set to special IDLE values.

    Args:
        segment_df: DataFrame representing a pause segment
        pipeline: 'seqofseq' or 'pxchange'
        gap_duration_seconds: Duration of the pause (optional, for metadata)

    Returns:
        modified_df: DataFrame with pseudo-patient conditioning features applied
    """
    df = segment_df.copy()

    if pipeline == 'seqofseq':
        # SeqofSeq pseudo-patient features
        # Keep: SystemType_encoded, Country_encoded (machine-specific)
        # Set to IDLE: BodyPart_encoded, Group_encoded (anatomical)

        if 'BodyPart_encoded' in df.columns:
            df['BodyPart_encoded'] = IDLE_STATE_ENCODING

        if 'Group_encoded' in df.columns:
            df['Group_encoded'] = IDLE_STATE_ENCODING

        # SystemType and Country are inherited from the machine context
        # (will be carried over from the previous patient sequence)

    elif pipeline == 'pxchange':
        # PXChange pseudo-patient features
        # Set to 0/IDLE: Age, Weight, Height, BodyGroup_from, BodyGroup_to, PTAB

        if 'Age' in df.columns:
            df['Age'] = 0.0

        if 'Weight' in df.columns:
            df['Weight'] = 0.0

        if 'Height' in df.columns:
            df['Height'] = 0.0

        if 'BodyGroup_from' in df.columns:
            df['BodyGroup_from'] = IDLE_STATE_ENCODING

        if 'BodyGroup_to' in df.columns:
            df['BodyGroup_to'] = IDLE_STATE_ENCODING

        if 'PTAB' in df.columns:
            df['PTAB'] = 0

    else:
        raise ValueError(f"Unknown pipeline: {pipeline}. Must be 'seqofseq' or 'pxchange'")

    # Set entity_type to PSEUDO_PATIENT_PAUSE
    df['entity_type'] = ENTITY_TYPES['PSEUDO_PATIENT_PAUSE']

    # Add pause metadata if provided
    if gap_duration_seconds is not None:
        df['pause_duration_seconds'] = gap_duration_seconds

    return df


# ============================================================================
# SEQUENCE SEGMENTATION
# ============================================================================

def split_sequence_at_pauses(patient_df, pause_threshold_minutes=None,
                             datetime_column='datetime',
                             sequence_id_column='PatientID',
                             pipeline='seqofseq'):
    """
    Segment a patient sequence into:
    1. Patient sessions (real medical sequences)
    2. Pause transitions (pseudo-patient idle periods)

    Each pause boundary creates a segmentation point, resulting in:
    - Multiple patient session segments (with REAL_PATIENT entity_type)
    - Multiple pause segments (with PSEUDO_PATIENT_PAUSE entity_type)

    Args:
        patient_df: DataFrame with one patient's sequence data
        pause_threshold_minutes: Time gap threshold (default from config)
        datetime_column: Column name for datetime
        sequence_id_column: Column name for patient/sequence ID
        pipeline: 'seqofseq' or 'pxchange'

    Returns:
        segments: List of DataFrames, each with entity_type label
    """
    if pause_threshold_minutes is None:
        pause_threshold_minutes = PAUSE_DETECTION_THRESHOLD_MINUTES

    pause_threshold_seconds = pause_threshold_minutes * 60

    # Make a copy and ensure datetime type
    df = patient_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_column]):
        df[datetime_column] = pd.to_datetime(df[datetime_column])

    # Sort by datetime
    df = df.sort_values(datetime_column).reset_index(drop=True)

    # Calculate time gaps between consecutive events
    df['time_gap_seconds'] = df[datetime_column].diff().dt.total_seconds()

    # Identify pause boundaries
    pause_boundaries = []
    for idx in range(1, len(df)):
        gap = df.loc[idx, 'time_gap_seconds']
        if pd.notna(gap) and gap > pause_threshold_seconds:
            pause_boundaries.append({
                'split_idx': idx,
                'gap_seconds': gap,
                'start_time': df.loc[idx - 1, datetime_column],
                'end_time': df.loc[idx, datetime_column]
            })

    # No pauses found - entire sequence is one real patient session
    if not pause_boundaries:
        df['entity_type'] = ENTITY_TYPES['REAL_PATIENT']
        df = df.drop('time_gap_seconds', axis=1, errors='ignore')
        return [df]

    # Split into segments
    segments = []
    start_idx = 0

    for boundary in pause_boundaries:
        split_idx = boundary['split_idx']
        gap_seconds = boundary['gap_seconds']

        # 1. Add patient session segment (before pause)
        session_df = df.iloc[start_idx:split_idx].copy()
        session_df['entity_type'] = ENTITY_TYPES['REAL_PATIENT']
        session_df = session_df.drop('time_gap_seconds', axis=1, errors='ignore')
        segments.append(session_df)

        # 2. Create pseudo-patient pause segment
        # Use the last event of previous session as template
        pause_template = df.iloc[split_idx - 1].copy()

        # Create a single-row DataFrame for the pause
        pause_row = pause_template.to_frame().T

        # Set the sequence/sourceID to IDLE
        if pipeline == 'seqofseq' and 'Sequence' in pause_row.columns:
            pause_row['Sequence'] = 'IDLE'
        elif pipeline == 'pxchange' and 'sourceID' in pause_row.columns:
            pause_row['sourceID'] = 'IDLE'

        # Set duration to the gap duration
        if 'duration' in pause_row.columns:
            pause_row['duration'] = gap_seconds
        elif 'step_duration' in pause_row.columns:
            pause_row['step_duration'] = gap_seconds
        elif 'timediff' in pause_row.columns:
            pause_row['timediff'] = gap_seconds

        # Set datetime to the start of the pause (end of previous event)
        pause_row[datetime_column] = boundary['start_time']

        # Apply pseudo-patient features
        pause_df = create_pseudo_patient_features(pause_row, pipeline=pipeline,
                                                   gap_duration_seconds=gap_seconds)
        pause_df = pause_df.drop('time_gap_seconds', axis=1, errors='ignore')

        segments.append(pause_df)

        # Update start index for next patient session
        start_idx = split_idx

    # Add final patient session (after last pause)
    final_session_df = df.iloc[start_idx:].copy()
    final_session_df['entity_type'] = ENTITY_TYPES['REAL_PATIENT']
    final_session_df = final_session_df.drop('time_gap_seconds', axis=1, errors='ignore')
    segments.append(final_session_df)

    return segments


# ============================================================================
# PAUSE CLASSIFICATION (Future Use)
# ============================================================================

def classify_pause_type(gap_start_time, gap_duration_seconds, day_of_week=None):
    """
    Classify pause type based on timing and duration

    Future enhancement to distinguish:
    - Lunch breaks (11-13:00, ~45 min)
    - Shift changes (8am, 4pm, ~15 min)
    - Overnight periods (8pm-7am, ~10 hours)
    - Random gaps (unpredictable)

    Args:
        gap_start_time: datetime when pause starts
        gap_duration_seconds: Duration of the pause
        day_of_week: Day of week (0=Monday, 6=Sunday)

    Returns:
        pause_type: str ('lunch', 'shift_change', 'overnight', 'random')
    """
    gap_duration_minutes = gap_duration_seconds / 60
    gap_duration_hours = gap_duration_seconds / 3600

    # Extract hour of day when pause starts
    hour_of_day = gap_start_time.hour

    # Overnight: long duration (>6 hours) and starts in evening
    if gap_duration_hours > 6 and (hour_of_day >= 18 or hour_of_day <= 6):
        return 'overnight'

    # Lunch break: mid-day (11-14:00) and moderate duration (30-90 min)
    if 11 <= hour_of_day <= 14 and 30 <= gap_duration_minutes <= 90:
        return 'lunch'

    # Shift change: short duration (5-30 min) and at typical shift times
    if 5 <= gap_duration_minutes <= 30 and hour_of_day in [7, 8, 15, 16]:
        return 'shift_change'

    # Random/unclassified
    return 'random'


# ============================================================================
# TRANSITION METADATA (For Evaluation)
# ============================================================================

def create_transition_metadata(segments, original_sequence_id):
    """
    Store metadata linking segmented sequences back to original sequence

    This is useful for evaluation and debugging to understand how sequences
    were split and to reconstruct the original timeline.

    Args:
        segments: List of segment DataFrames
        original_sequence_id: ID of the original patient sequence

    Returns:
        segments: List of segments with added metadata columns
    """
    for seg_idx, segment in enumerate(segments):
        segment['original_sequence_id'] = original_sequence_id
        segment['segment_index'] = seg_idx
        segment['total_segments'] = len(segments)

    return segments


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_segment_statistics(segments):
    """
    Compute statistics about segmented sequences

    Args:
        segments: List of segment DataFrames

    Returns:
        stats: Dictionary with segmentation statistics
    """
    total_segments = len(segments)
    real_patient_segments = sum(1 for s in segments if s['entity_type'].iloc[0] == ENTITY_TYPES['REAL_PATIENT'])
    pseudo_patient_segments = sum(1 for s in segments if s['entity_type'].iloc[0] == ENTITY_TYPES['PSEUDO_PATIENT_PAUSE'])

    total_events = sum(len(s) for s in segments)
    real_patient_events = sum(len(s) for s in segments if s['entity_type'].iloc[0] == ENTITY_TYPES['REAL_PATIENT'])
    pseudo_patient_events = sum(len(s) for s in segments if s['entity_type'].iloc[0] == ENTITY_TYPES['PSEUDO_PATIENT_PAUSE'])

    return {
        'total_segments': total_segments,
        'real_patient_segments': real_patient_segments,
        'pseudo_patient_segments': pseudo_patient_segments,
        'real_patient_ratio': real_patient_segments / total_segments if total_segments > 0 else 0,
        'total_events': total_events,
        'real_patient_events': real_patient_events,
        'pseudo_patient_events': pseudo_patient_events
    }


def validate_segmentation(segments, pipeline='seqofseq'):
    """
    Validate that segmentation is correct

    Checks:
    1. All segments have entity_type
    2. Real patient segments don't have IDLE tokens
    3. Pseudo-patient segments only have IDLE tokens
    4. Conditioning features are properly set

    Args:
        segments: List of segment DataFrames
        pipeline: 'seqofseq' or 'pxchange'

    Returns:
        is_valid: bool
        errors: List of error messages
    """
    errors = []

    for seg_idx, segment in enumerate(segments):
        # Check entity_type exists
        if 'entity_type' not in segment.columns:
            errors.append(f"Segment {seg_idx}: Missing entity_type column")
            continue

        entity_type = segment['entity_type'].iloc[0]

        # Check for IDLE tokens
        if pipeline == 'seqofseq':
            sequence_col = 'Sequence'
            has_idle = (segment[sequence_col] == 'IDLE').any() if sequence_col in segment.columns else False
        elif pipeline == 'pxchange':
            sequence_col = 'sourceID'
            has_idle = (segment[sequence_col] == 'IDLE').any() if sequence_col in segment.columns else False
        else:
            errors.append(f"Segment {seg_idx}: Unknown pipeline {pipeline}")
            continue

        # Validation rules
        if entity_type == ENTITY_TYPES['REAL_PATIENT']:
            if has_idle:
                errors.append(f"Segment {seg_idx}: Real patient segment contains IDLE tokens")

        elif entity_type == ENTITY_TYPES['PSEUDO_PATIENT_PAUSE']:
            if not has_idle:
                errors.append(f"Segment {seg_idx}: Pseudo-patient segment missing IDLE tokens")

            # Check conditioning features
            if pipeline == 'seqofseq':
                if 'BodyPart_encoded' in segment.columns:
                    if not (segment['BodyPart_encoded'] == IDLE_STATE_ENCODING).all():
                        errors.append(f"Segment {seg_idx}: BodyPart_encoded not set to IDLE")

            elif pipeline == 'pxchange':
                if 'Age' in segment.columns:
                    if not (segment['Age'] == 0.0).all():
                        errors.append(f"Segment {seg_idx}: Age not set to 0")

    is_valid = len(errors) == 0
    return is_valid, errors


# ============================================================================
# MAIN ENTRY POINT (For Testing)
# ============================================================================

if __name__ == "__main__":
    print("Pseudo-Patient Generator Module")
    print(f"Entity Types: {ENTITY_TYPES}")
    print(f"Idle State Encoding: {IDLE_STATE_ENCODING}")
    print(f"Pause Detection Threshold: {PAUSE_DETECTION_THRESHOLD_MINUTES} minutes")
    print("\nThis module is designed to be imported by other scripts.")
    print("See preprocess_all_data.py for usage examples.")
