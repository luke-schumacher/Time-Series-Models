"""
Pause Event Detection and Injection
Identifies temporal gaps in MRI sequences and inserts PAUSE tokens
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import PAUSE_DETECTION_THRESHOLD_MINUTES, PAUSE_DURATION_MIN_SECONDS, PAUSE_DURATION_MAX_SECONDS


def identify_pause_events(sequence_df, pause_threshold_minutes=None, datetime_column='datetime',
                           sourceID_column='sourceID', duration_column='step_duration'):
    """
    Identify pause events in a sequence based on time gaps

    Args:
        sequence_df: DataFrame with sequence data (must have datetime column)
        pause_threshold_minutes: Time gap threshold to trigger pause detection (default from config)
        datetime_column: Name of datetime column
        sourceID_column: Name of sourceID/sequence column
        duration_column: Name of duration column

    Returns:
        modified_df: DataFrame with PAUSE events inserted
    """
    if pause_threshold_minutes is None:
        pause_threshold_minutes = PAUSE_DETECTION_THRESHOLD_MINUTES

    pause_threshold_seconds = pause_threshold_minutes * 60

    # Make a copy to avoid modifying original
    df = sequence_df.copy()

    # Ensure datetime column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_column]):
        df[datetime_column] = pd.to_datetime(df[datetime_column])

    # Sort by datetime
    df = df.sort_values(datetime_column).reset_index(drop=True)

    # Calculate time gaps between consecutive events
    df['time_gap_seconds'] = df[datetime_column].diff().dt.total_seconds()

    # Identify where pauses should be inserted
    pause_locations = []

    for idx in range(1, len(df)):
        gap = df.loc[idx, 'time_gap_seconds']

        if gap > pause_threshold_seconds:
            # Insert a pause event
            pause_event = df.loc[idx].copy()
            pause_event[sourceID_column] = 'PAUSE'
            pause_event[duration_column] = min(gap, PAUSE_DURATION_MAX_SECONDS)

            # Store insertion location and pause event
            pause_locations.append((idx, pause_event))

    # Insert pauses from end to beginning (to preserve indices)
    rows_list = df.to_dict('records')
    for insert_idx, pause_event in reversed(pause_locations):
        rows_list.insert(insert_idx, pause_event.to_dict())

    # Recreate DataFrame
    modified_df = pd.DataFrame(rows_list).reset_index(drop=True)

    # Drop temporary column
    if 'time_gap_seconds' in modified_df.columns:
        modified_df = modified_df.drop('time_gap_seconds', axis=1)

    num_pauses = len(pause_locations)
    print(f"  Inserted {num_pauses} PAUSE events (threshold: {pause_threshold_minutes} min)")

    return modified_df


def inject_pauses_pxchange(raw_file_path, output_file_path=None, dataset_id=None):
    """
    Process PXChange raw data and inject PAUSE tokens

    Args:
        raw_file_path: Path to raw PXChange CSV file
        output_file_path: Path to save processed file (optional)
        dataset_id: Dataset identifier (optional)

    Returns:
        processed_df: DataFrame with PAUSE events
    """
    print(f"Processing PXChange file: {raw_file_path}")

    # Load raw data
    df = pd.read_csv(raw_file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Group by PatientID and process each patient's sequence
    processed_sequences = []

    for patient_id in df['PatientId'].unique():
        patient_df = df[df['PatientId'] == patient_id].copy()
        patient_df = patient_df.sort_values('datetime').reset_index(drop=True)

        # Detect and insert pauses
        patient_df_with_pauses = identify_pause_events(
            patient_df,
            pause_threshold_minutes=PAUSE_DETECTION_THRESHOLD_MINUTES,
            datetime_column='datetime',
            sourceID_column='sourceID',
            duration_column='timediff' if 'timediff' in patient_df.columns else 'step_duration'
        )

        processed_sequences.append(patient_df_with_pauses)

    # Combine all sequences
    processed_df = pd.concat(processed_sequences, ignore_index=True)

    # Add dataset_id if provided
    if dataset_id is not None:
        processed_df['dataset_id'] = dataset_id

    # Save if output path provided
    if output_file_path:
        processed_df.to_csv(output_file_path, index=False)
        print(f"Saved processed file to: {output_file_path}")

    return processed_df


def inject_pauses_seqofseq(raw_file_path, output_file_path=None, dataset_id=None):
    """
    Process SeqofSeq raw data and inject PAUSE tokens

    Args:
        raw_file_path: Path to raw SeqofSeq CSV file
        output_file_path: Path to save processed file (optional)
        dataset_id: Dataset identifier (optional)

    Returns:
        processed_df: DataFrame with PAUSE events
    """
    print(f"Processing SeqofSeq file: {raw_file_path}")

    # Load raw data
    df = pd.read_csv(raw_file_path)
    df['startTime'] = pd.to_datetime(df['startTime'])
    df['endTime'] = pd.to_datetime(df['endTime'])

    # Group by PatientID and process each patient's sequence
    processed_sequences = []

    for patient_id in df['PatientID'].unique():
        patient_df = df[df['PatientID'] == patient_id].copy()
        patient_df = patient_df.sort_values('startTime').reset_index(drop=True)

        # Detect and insert pauses
        patient_df_with_pauses = identify_pause_events(
            patient_df,
            pause_threshold_minutes=PAUSE_DETECTION_THRESHOLD_MINUTES,
            datetime_column='startTime',
            sourceID_column='Sequence',
            duration_column='duration'
        )

        processed_sequences.append(patient_df_with_pauses)

    # Combine all sequences
    processed_df = pd.concat(processed_sequences, ignore_index=True)

    # Add dataset_id if provided
    if dataset_id is not None:
        processed_df['dataset_id'] = dataset_id

    # Save if output path provided
    if output_file_path:
        processed_df.to_csv(output_file_path, index=False)
        print(f"Saved processed file to: {output_file_path}")

    return processed_df


def batch_process_pxchange_files(input_dir, output_dir, file_pattern='*.csv'):
    """
    Batch process multiple PXChange files

    Args:
        input_dir: Directory containing raw CSV files
        output_dir: Directory to save processed files
        file_pattern: Glob pattern for files to process

    Returns:
        summary: Dictionary with processing statistics
    """
    import glob

    os.makedirs(output_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(input_dir, file_pattern))
    print(f"Found {len(csv_files)} files to process")

    summary = {
        'files_processed': 0,
        'total_sequences': 0,
        'total_pauses_inserted': 0
    }

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        dataset_id = filename.replace('.csv', '')
        output_path = os.path.join(output_dir, f"{dataset_id}_with_pauses.csv")

        try:
            # Process file
            df_before = pd.read_csv(file_path)
            processed_df = inject_pauses_pxchange(file_path, output_path, dataset_id)

            # Update statistics
            num_pauses = len(processed_df) - len(df_before)
            summary['files_processed'] += 1
            summary['total_pauses_inserted'] += num_pauses

            print(f"  {filename}: {len(df_before)} → {len(processed_df)} rows (+{num_pauses} pauses)\n")

        except Exception as e:
            print(f"  Error processing {filename}: {e}\n")

    print(f"\nBatch processing complete:")
    print(f"  Files processed: {summary['files_processed']}")
    print(f"  Total pauses inserted: {summary['total_pauses_inserted']}")

    return summary


def batch_process_seqofseq_files(input_dir, output_dir, file_pattern='*.csv'):
    """
    Batch process multiple SeqofSeq files

    Args:
        input_dir: Directory containing raw CSV files
        output_dir: Directory to save processed files
        file_pattern: Glob pattern for files to process

    Returns:
        summary: Dictionary with processing statistics
    """
    import glob

    os.makedirs(output_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(input_dir, file_pattern))
    print(f"Found {len(csv_files)} files to process")

    summary = {
        'files_processed': 0,
        'total_sequences': 0,
        'total_pauses_inserted': 0
    }

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        dataset_id = filename.replace('.csv', '')
        output_path = os.path.join(output_dir, f"{dataset_id}_with_pauses.csv")

        try:
            # Process file
            df_before = pd.read_csv(file_path)
            processed_df = inject_pauses_seqofseq(file_path, output_path, dataset_id)

            # Update statistics
            num_pauses = len(processed_df) - len(df_before)
            summary['files_processed'] += 1
            summary['total_pauses_inserted'] += num_pauses

            print(f"  {filename}: {len(df_before)} → {len(processed_df)} rows (+{num_pauses} pauses)\n")

        except Exception as e:
            print(f"  Error processing {filename}: {e}\n")

    print(f"\nBatch processing complete:")
    print(f"  Files processed: {summary['files_processed']}")
    print(f"  Total pauses inserted: {summary['total_pauses_inserted']}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Inject PAUSE tokens into MRI sequence data')
    parser.add_argument('--pipeline', type=str, choices=['pxchange', 'seqofseq'], required=True,
                        help='Which pipeline to process')
    parser.add_argument('--input', type=str, required=True,
                        help='Input file or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file or directory')
    parser.add_argument('--batch', action='store_true',
                        help='Process entire directory (batch mode)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Pause detection threshold in minutes (default: from config)')

    args = parser.parse_args()

    # Override threshold if provided
    if args.threshold:
        PAUSE_DETECTION_THRESHOLD_MINUTES = args.threshold

    if args.batch:
        # Batch processing
        if args.pipeline == 'pxchange':
            batch_process_pxchange_files(args.input, args.output)
        else:
            batch_process_seqofseq_files(args.input, args.output)
    else:
        # Single file processing
        if args.pipeline == 'pxchange':
            inject_pauses_pxchange(args.input, args.output)
        else:
            inject_pauses_seqofseq(args.input, args.output)

    print("\nPause injection complete!")
