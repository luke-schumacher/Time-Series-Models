"""
Ground Truth Schedule Loader
Loads real customer schedules for validation testing

NOTE: This is a scaffold implementation. Will be fully implemented when ground truth data is available.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_ground_truth_schedule(schedule_path):
    """
    Load real customer schedule from file

    Expected format:
        schedule_id, date, patient_id, start_time, end_time,
        body_part, scan_sequence, duration

    Args:
        schedule_path: Path to ground truth schedule CSV file

    Returns:
        schedule_df: DataFrame with real schedule data
    """
    if not os.path.exists(schedule_path):
        raise FileNotFoundError(f"Ground truth schedule not found: {schedule_path}")

    df = pd.read_csv(schedule_path)

    # Validate required columns
    required_columns = ['schedule_id', 'date', 'patient_id', 'start_time', 'body_part', 'duration']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Parse datetime columns
    if 'start_time' in df.columns:
        df['start_time'] = pd.to_datetime(df['start_time'])
    if 'end_time' in df.columns:
        df['end_time'] = pd.to_datetime(df['end_time'])
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.date

    print(f"Loaded ground truth schedule: {schedule_path}")
    print(f"  Schedule ID: {df['schedule_id'].iloc[0] if 'schedule_id' in df.columns else 'N/A'}")
    print(f"  Date: {df['date'].iloc[0] if 'date' in df.columns else 'N/A'}")
    print(f"  Events: {len(df)}")
    print(f"  Patients: {df['patient_id'].nunique() if 'patient_id' in df.columns else 'N/A'}")

    return df


def extract_conditioning_from_schedule(schedule_df):
    """
    Extract conditioning features from real schedule to use for generation

    Analyzes the real schedule and extracts features that can be used as
    conditioning input for synthetic schedule generation.

    Args:
        schedule_df: DataFrame with real schedule

    Returns:
        conditioning_dict: Dictionary with conditioning features
            - date: Date of schedule
            - day_of_week: Day of week (0=Monday)
            - num_patients: Number of patients
            - body_parts: List of body parts scanned
            - typical_start_hour: Typical start time
            - typical_end_hour: Typical end time
    """
    conditioning = {}

    if 'date' in schedule_df.columns:
        schedule_date = pd.to_datetime(schedule_df['date'].iloc[0])
        conditioning['date'] = schedule_date
        conditioning['day_of_week'] = schedule_date.weekday()

    if 'patient_id' in schedule_df.columns:
        conditioning['num_patients'] = schedule_df['patient_id'].nunique()

    if 'body_part' in schedule_df.columns:
        conditioning['body_parts'] = schedule_df['body_part'].unique().tolist()
        conditioning['body_part_distribution'] = schedule_df['body_part'].value_counts().to_dict()

    if 'start_time' in schedule_df.columns:
        start_times = pd.to_datetime(schedule_df['start_time'])
        conditioning['typical_start_hour'] = start_times.dt.hour.min()
        conditioning['typical_end_hour'] = start_times.dt.hour.max()
        conditioning['session_start_times'] = start_times.dt.hour.unique().tolist()

    if 'duration' in schedule_df.columns:
        conditioning['total_duration_minutes'] = schedule_df['duration'].sum() / 60
        conditioning['avg_scan_duration_minutes'] = schedule_df['duration'].mean() / 60

    # TODO: Extract more features when ground truth format is finalized
    # - Machine type
    # - Geographic location
    # - Protocol information

    print(f"\nExtracted conditioning features:")
    for key, value in conditioning.items():
        if isinstance(value, list):
            print(f"  {key}: {len(value)} items")
        elif isinstance(value, dict):
            print(f"  {key}: {len(value)} keys")
        else:
            print(f"  {key}: {value}")

    return conditioning


def load_all_ground_truth_schedules(ground_truth_dir):
    """
    Load all ground truth schedules from a directory

    Args:
        ground_truth_dir: Directory containing ground truth schedule CSV files

    Returns:
        schedules: List of (schedule_path, schedule_df) tuples
    """
    if not os.path.exists(ground_truth_dir):
        raise FileNotFoundError(f"Ground truth directory not found: {ground_truth_dir}")

    csv_files = [f for f in os.listdir(ground_truth_dir) if f.endswith('.csv')]

    if not csv_files:
        print(f"No ground truth schedules found in: {ground_truth_dir}")
        return []

    schedules = []
    for filename in csv_files:
        file_path = os.path.join(ground_truth_dir, filename)
        try:
            schedule_df = load_ground_truth_schedule(file_path)
            schedules.append((file_path, schedule_df))
        except Exception as e:
            print(f"ERROR loading {filename}: {e}")

    print(f"\nLoaded {len(schedules)} ground truth schedules")
    return schedules


def validate_ground_truth_format(schedule_df):
    """
    Validate that ground truth schedule has correct format

    Args:
        schedule_df: DataFrame to validate

    Returns:
        is_valid: bool
        errors: List of error messages
    """
    errors = []

    # Check required columns
    required_columns = ['schedule_id', 'date', 'patient_id', 'start_time', 'body_part', 'duration']
    for col in required_columns:
        if col not in schedule_df.columns:
            errors.append(f"Missing required column: {col}")

    # Check data types
    if 'duration' in schedule_df.columns:
        if not pd.api.types.is_numeric_dtype(schedule_df['duration']):
            errors.append("Duration column must be numeric")

    # Check for null values in critical columns
    critical_columns = ['patient_id', 'start_time', 'duration']
    for col in critical_columns:
        if col in schedule_df.columns and schedule_df[col].isnull().any():
            errors.append(f"Null values found in {col}")

    # Check temporal ordering
    if 'start_time' in schedule_df.columns:
        start_times = pd.to_datetime(schedule_df['start_time'])
        if not start_times.is_monotonic_increasing:
            errors.append("Start times are not monotonically increasing")

    is_valid = len(errors) == 0
    return is_valid, errors


# ============================================================================
# EXAMPLE GROUND TRUTH FORMAT (For Reference)
# ============================================================================

def create_example_ground_truth_schedule(output_path=None):
    """
    Create an example ground truth schedule file for reference

    This shows the expected format for ground truth data.
    """
    example_data = {
        'schedule_id': ['EXAMPLE_001'] * 8,
        'date': ['2024-01-15'] * 8,
        'patient_id': ['P001', 'P001', 'P001', 'P002', 'P002', 'P003', 'P003', 'P003'],
        'start_time': [
            '2024-01-15 07:30:00',
            '2024-01-15 07:45:00',
            '2024-01-15 08:00:00',
            '2024-01-15 08:30:00',
            '2024-01-15 08:45:00',
            '2024-01-15 09:15:00',
            '2024-01-15 09:30:00',
            '2024-01-15 09:50:00'
        ],
        'end_time': [
            '2024-01-15 07:45:00',
            '2024-01-15 08:00:00',
            '2024-01-15 08:15:00',
            '2024-01-15 08:45:00',
            '2024-01-15 09:00:00',
            '2024-01-15 09:30:00',
            '2024-01-15 09:50:00',
            '2024-01-15 10:05:00'
        ],
        'body_part': ['Brain', 'Brain', 'Spine', 'Knee', 'Knee', 'Brain', 'Brain', 'Spine'],
        'scan_sequence': ['T1', 'T2', 'T1', 'T1', 'T2', 'T1', 'FLAIR', 'T2'],
        'duration': [900, 900, 900, 900, 900, 900, 1200, 900]  # seconds
    }

    df = pd.DataFrame(example_data)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Example ground truth schedule saved to: {output_path}")

    return df


# ============================================================================
# MAIN ENTRY POINT (For Testing)
# ============================================================================

if __name__ == "__main__":
    print("Ground Truth Schedule Loader (Scaffold)")
    print("=" * 70)
    print("\nThis module loads real customer schedules for validation.")
    print("Expected format: CSV with columns:")
    print("  - schedule_id, date, patient_id, start_time, end_time,")
    print("    body_part, scan_sequence, duration")
    print("\nTo create an example:")
    print("  python ground_truth_loader.py --create-example <output_path>")
    print("\nTo test loading:")
    print("  python ground_truth_loader.py --load <schedule_path>")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--create-example', type=str, help='Create example schedule file')
    parser.add_argument('--load', type=str, help='Load and validate a schedule file')
    args = parser.parse_args()

    if args.create_example:
        create_example_ground_truth_schedule(args.create_example)
    elif args.load:
        schedule_df = load_ground_truth_schedule(args.load)
        is_valid, errors = validate_ground_truth_format(schedule_df)
        print(f"\nValidation: {'✓ PASS' if is_valid else '✗ FAIL'}")
        if errors:
            for error in errors:
                print(f"  - {error}")
        conditioning = extract_conditioning_from_schedule(schedule_df)
    else:
        print("\nUse --create-example or --load to test the loader.")
