"""
Sequence Segmentation for MRI Scheduling
Replaces pause_injection.py with pseudo-patient architecture

Instead of injecting PAUSE tokens into sequences, this module segments sequences
at pause boundaries and creates pseudo-patient entities to represent idle states.

This maintains the same high-level API as pause_injection.py for backward compatibility.
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import PAUSE_DETECTION_THRESHOLD_MINUTES, PAUSE_DURATION_MIN_SECONDS, PAUSE_DURATION_MAX_SECONDS
from preprocessing.pseudo_patient_generator import (
    split_sequence_at_pauses,
    create_transition_metadata,
    get_segment_statistics,
    validate_segmentation,
    ENTITY_TYPES
)

# Import preprocessing utilities
from sklearn.preprocessing import LabelEncoder


# ============================================================================
# ENCODING HELPER FUNCTIONS
# ============================================================================

def apply_seqofseq_preprocessing(df):
    """
    Apply encoding and sequence grouping to segmented SeqofSeq data

    Adds the columns needed by the data loader:
    - sequence_id, protocol_id (encoded from Sequence/Protocol)
    - BodyPart_encoded, SystemType_encoded, Country_encoded, Group_encoded
    - sequence_idx (for grouping sequences)
    - seq_position, seq_length

    Args:
        df: Dataframe with segmented sequences (has entity_type column)

    Returns:
        df: Dataframe with all encoding columns added
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'SeqofSeq_Pipeline'))
    from SeqofSeq_Pipeline.config import SPECIAL_TOKENS, IDLE_TOKEN_ID, IDLE_STATE_ENCODING

    df = df.copy()

    # Build vocabulary for Sequence column
    sequence_vocab = SPECIAL_TOKENS.copy()
    unique_sequences = df[df['entity_type'] == 0]['Sequence'].unique()  # Only from real patients
    next_id = max(sequence_vocab.values()) + 1
    for seq in sorted(unique_sequences):
        if pd.notna(seq) and seq not in sequence_vocab:
            sequence_vocab[seq] = next_id
            next_id += 1

    # Encode sequences
    df['sequence_id'] = df['Sequence'].map(sequence_vocab)
    df['sequence_id'] = df['sequence_id'].fillna(SPECIAL_TOKENS['UNK'])

    # For IDLE rows (pseudo-patients), use IDLE token
    idle_mask = (df['entity_type'] == 1)
    df.loc[idle_mask, 'sequence_id'] = IDLE_TOKEN_ID

    # Build vocabulary for Protocol column
    protocol_vocab = SPECIAL_TOKENS.copy()
    unique_protocols = df[df['entity_type'] == 0]['Protocol'].unique()
    next_id = max(protocol_vocab.values()) + 1
    for prot in sorted(unique_protocols):
        if pd.notna(prot) and prot not in protocol_vocab:
            protocol_vocab[prot] = next_id
            next_id += 1

    df['protocol_id'] = df['Protocol'].map(protocol_vocab)
    df['protocol_id'] = df['protocol_id'].fillna(SPECIAL_TOKENS['UNK'])
    df.loc[idle_mask, 'protocol_id'] = SPECIAL_TOKENS['UNK']

    # Encode categorical conditioning features
    conditioning_features = ['BodyPart', 'Systemtype', 'Country', 'BodyGroup']

    for feature in conditioning_features:
        encoded_col = f"{feature}_encoded" if feature != 'BodyGroup' else "Group_encoded"

        if feature in df.columns:
            le = LabelEncoder()
            # Only fit on real patient data
            real_mask = (df['entity_type'] == 0) & df[feature].notna()

            if real_mask.sum() > 0:
                # Fit encoder on real patient data
                le.fit(df.loc[real_mask, feature].astype(str))

                # Apply to real patients
                df.loc[real_mask, encoded_col] = le.transform(df.loc[real_mask, feature].astype(str))

                # For pseudo-patients, use IDLE encoding
                df.loc[idle_mask, encoded_col] = IDLE_STATE_ENCODING

                # Handle any remaining NaN
                df[encoded_col] = df[encoded_col].fillna(-1)
            else:
                df[encoded_col] = -1
        else:
            df[encoded_col] = -1

    # Create sequence_idx by enumerating unique segment groups
    # Use original_sequence_id + segment_index as the grouping key
    if 'original_sequence_id' in df.columns and 'segment_index' in df.columns:
        df['seg_group'] = df['original_sequence_id'].astype(str) + '_' + df['segment_index'].astype(str)
        unique_groups = df['seg_group'].unique()
        seg_idx_map = {group: idx for idx, group in enumerate(unique_groups)}
        df['sequence_idx'] = df['seg_group'].map(seg_idx_map)
        df = df.drop('seg_group', axis=1)
    else:
        # Fallback: just enumerate rows
        df['sequence_idx'] = range(len(df))

    # Add seq_position and seq_length
    for seq_idx in df['sequence_idx'].unique():
        seq_mask = df['sequence_idx'] == seq_idx
        seq_len = seq_mask.sum()
        df.loc[seq_mask, 'seq_position'] = range(seq_len)
        df.loc[seq_mask, 'seq_length'] = seq_len

    return df


def apply_pxchange_preprocessing(df):
    """
    Apply encoding and sequence grouping to segmented PXChange data

    Adds the columns needed by the data loader:
    - sourceID (encoded from SourceId)
    - Age, Weight, Height, BodyGroup_from, BodyGroup_to, PTAB (encoded)
    - sequence_idx (for grouping sequences)
    - seq_position, seq_length

    Args:
        df: Dataframe with segmented sequences (has entity_type column)

    Returns:
        df: Dataframe with all encoding columns added
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'PXChange_Refactored'))
    from PXChange_Refactored.config import SOURCEID_VOCAB, IDLE_TOKEN_ID, IDLE_STATE_ENCODING

    df = df.copy()

    # For PXChange, the sourceID is already in the file
    # Just need to encode it using the vocab
    if 'SourceId' in df.columns:
        df['sourceID'] = df['SourceId'].map(SOURCEID_VOCAB)
        df['sourceID'] = df['sourceID'].fillna(SOURCEID_VOCAB.get('UNK', 17))

        # For IDLE rows (pseudo-patients), use IDLE token
        idle_mask = (df['entity_type'] == 1)
        df.loc[idle_mask, 'sourceID'] = IDLE_TOKEN_ID

    # Encode conditioning features
    # For Age, Weight, Height: use actual values for real patients, 0 for pseudo-patients
    numeric_features = ['Age', 'Weight', 'Height', 'PTAB']
    for feature in numeric_features:
        if feature not in df.columns:
            df[feature] = 0
        idle_mask = (df['entity_type'] == 1)
        df.loc[idle_mask, feature] = 0

    # For BodyGroup_from/to: encode categorical
    for feature in ['BodyGroup_from', 'BodyGroup_to']:
        if feature in df.columns:
            le = LabelEncoder()
            real_mask = (df['entity_type'] == 0) & df[feature].notna()

            if real_mask.sum() > 0:
                le.fit(df.loc[real_mask, feature].astype(str))
                df.loc[real_mask, feature] = le.transform(df.loc[real_mask, feature].astype(str))

                idle_mask = (df['entity_type'] == 1)
                df.loc[idle_mask, feature] = IDLE_STATE_ENCODING
                df[feature] = df[feature].fillna(-1)
            else:
                df[feature] = -1
        else:
            df[feature] = -1

    # Create sequence_idx
    if 'original_sequence_id' in df.columns and 'segment_index' in df.columns:
        df['seg_group'] = df['original_sequence_id'].astype(str) + '_' + df['segment_index'].astype(str)
        unique_groups = df['seg_group'].unique()
        seg_idx_map = {group: idx for idx, group in enumerate(unique_groups)}
        df['sequence_idx'] = df['seg_group'].map(seg_idx_map)
        df = df.drop('seg_group', axis=1)
    else:
        df['sequence_idx'] = range(len(df))

    # Add seq_position and seq_length
    for seq_idx in df['sequence_idx'].unique():
        seq_mask = df['sequence_idx'] == seq_idx
        seq_len = seq_mask.sum()
        df.loc[seq_mask, 'seq_position'] = range(seq_len)
        df.loc[seq_mask, 'seq_length'] = seq_len

    return df


# ============================================================================
# MAIN SEGMENTATION FUNCTIONS
# ============================================================================

def segment_pxchange_file(raw_file_path, output_file_path=None, dataset_id=None):
    """
    Process PXChange raw data and segment sequences at pause boundaries WITH full preprocessing

    Replaces inject_pauses_pxchange() with new pseudo-patient architecture.

    This function now does:
    1. Segment sequences at pause boundaries
    2. Create pseudo-patient entities
    3. Apply full preprocessing (encoding categorical features, building vocabularies)

    Args:
        raw_file_path: Path to raw PXChange CSV file
        output_file_path: Path to save processed file (optional)
        dataset_id: Dataset identifier (optional)

    Returns:
        processed_df: DataFrame with segmented sequences and entity_type labels
        stats: Dictionary with segmentation statistics
    """
    print(f"Segmenting PXChange file: {raw_file_path}")

    # Load raw data
    df = pd.read_csv(raw_file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Track statistics
    stats = {
        'total_patients': 0,
        'total_segments': 0,
        'real_patient_segments': 0,
        'pseudo_patient_segments': 0,
        'total_pauses': 0,
        'validation_errors': []
    }

    # Group by PatientID and segment each patient's sequence
    all_segments = []

    for patient_id in df['PatientId'].unique():
        patient_df = df[df['PatientId'] == patient_id].copy()
        patient_df = patient_df.sort_values('datetime').reset_index(drop=True)

        # Segment at pause boundaries
        segments = split_sequence_at_pauses(
            patient_df,
            pause_threshold_minutes=PAUSE_DETECTION_THRESHOLD_MINUTES,
            datetime_column='datetime',
            sequence_id_column='PatientId',
            pipeline='pxchange'
        )

        # Add transition metadata
        segments = create_transition_metadata(segments, patient_id)

        # Validate segmentation
        is_valid, errors = validate_segmentation(segments, pipeline='pxchange')
        if not is_valid:
            print(f"  WARNING: Validation errors for Patient {patient_id}:")
            for error in errors:
                print(f"    - {error}")
            stats['validation_errors'].extend(errors)

        # Update statistics
        seg_stats = get_segment_statistics(segments)
        stats['total_patients'] += 1
        stats['total_segments'] += seg_stats['total_segments']
        stats['real_patient_segments'] += seg_stats['real_patient_segments']
        stats['pseudo_patient_segments'] += seg_stats['pseudo_patient_segments']
        stats['total_pauses'] += seg_stats['pseudo_patient_segments']

        all_segments.extend(segments)

    # Combine all segments
    processed_df = pd.concat(all_segments, ignore_index=True)

    # Add dataset_id if provided
    if dataset_id is not None:
        processed_df['dataset_id'] = dataset_id

    # Print statistics
    print(f"  Processed {stats['total_patients']} patients")
    print(f"  Created {stats['total_segments']} segments:")
    print(f"    - Real patient segments: {stats['real_patient_segments']}")
    print(f"    - Pseudo-patient segments: {stats['pseudo_patient_segments']} (pauses)")
    if stats['total_segments'] > 0:
        pause_ratio = stats['pseudo_patient_segments'] / stats['total_segments'] * 100
        print(f"    - Pause ratio: {pause_ratio:.2f}%")

    # Save if output path provided
    if output_file_path:
        processed_df.to_csv(output_file_path, index=False)
        print(f"  Saved segmented file to: {output_file_path}")

    return processed_df, stats


def segment_seqofseq_file(raw_file_path, output_file_path=None, dataset_id=None):
    """
    Process SeqofSeq raw data and segment sequences at pause boundaries

    Replaces inject_pauses_seqofseq() with new pseudo-patient architecture.

    Args:
        raw_file_path: Path to raw SeqofSeq CSV file
        output_file_path: Path to save processed file (optional)
        dataset_id: Dataset identifier (optional)

    Returns:
        processed_df: DataFrame with segmented sequences and entity_type labels
        stats: Dictionary with segmentation statistics
    """
    print(f"Segmenting SeqofSeq file: {raw_file_path}")

    # Load raw data
    df = pd.read_csv(raw_file_path)
    df['startTime'] = pd.to_datetime(df['startTime'])
    df['endTime'] = pd.to_datetime(df['endTime'])

    # Track statistics
    stats = {
        'total_patients': 0,
        'total_segments': 0,
        'real_patient_segments': 0,
        'pseudo_patient_segments': 0,
        'total_pauses': 0,
        'validation_errors': []
    }

    # Group by PatientID and segment each patient's sequence
    all_segments = []

    for patient_id in df['PatientID'].unique():
        patient_df = df[df['PatientID'] == patient_id].copy()
        patient_df = patient_df.sort_values('startTime').reset_index(drop=True)

        # Segment at pause boundaries
        segments = split_sequence_at_pauses(
            patient_df,
            pause_threshold_minutes=PAUSE_DETECTION_THRESHOLD_MINUTES,
            datetime_column='startTime',
            sequence_id_column='PatientID',
            pipeline='seqofseq'
        )

        # Add transition metadata
        segments = create_transition_metadata(segments, patient_id)

        # Validate segmentation
        is_valid, errors = validate_segmentation(segments, pipeline='seqofseq')
        if not is_valid:
            print(f"  WARNING: Validation errors for Patient {patient_id}:")
            for error in errors:
                print(f"    - {error}")
            stats['validation_errors'].extend(errors)

        # Update statistics
        seg_stats = get_segment_statistics(segments)
        stats['total_patients'] += 1
        stats['total_segments'] += seg_stats['total_segments']
        stats['real_patient_segments'] += seg_stats['real_patient_segments']
        stats['pseudo_patient_segments'] += seg_stats['pseudo_patient_segments']
        stats['total_pauses'] += seg_stats['pseudo_patient_segments']

        all_segments.extend(segments)

    # Combine all segments
    processed_df = pd.concat(all_segments, ignore_index=True)

    # Add dataset_id if provided
    if dataset_id is not None:
        processed_df['dataset_id'] = dataset_id

    # IMPORTANT: Apply full preprocessing (encoding) to segmented data
    print("  Applying encoding and sequence grouping...")
    processed_df = apply_seqofseq_preprocessing(processed_df)

    # Print statistics
    print(f"  Processed {stats['total_patients']} patients")
    print(f"  Created {stats['total_segments']} segments:")
    print(f"    - Real patient segments: {stats['real_patient_segments']}")
    print(f"    - Pseudo-patient segments: {stats['pseudo_patient_segments']} (pauses)")
    if stats['total_segments'] > 0:
        pause_ratio = stats['pseudo_patient_segments'] / stats['total_segments'] * 100
        print(f"    - Pause ratio: {pause_ratio:.2f}%")

    # Save if output path provided
    if output_file_path:
        processed_df.to_csv(output_file_path, index=False)
        print(f"  Saved segmented file to: {output_file_path}")

    return processed_df, stats


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def batch_segment_pxchange_files(input_dir, output_dir, file_pattern='*.csv'):
    """
    Batch segment multiple PXChange files

    Args:
        input_dir: Directory containing raw CSV files
        output_dir: Directory to save segmented files
        file_pattern: Glob pattern for files to process

    Returns:
        summary: Dictionary with overall processing statistics
    """
    import glob

    os.makedirs(output_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(input_dir, file_pattern))
    print(f"Found {len(csv_files)} PXChange files to process\n")

    summary = {
        'files_processed': 0,
        'total_patients': 0,
        'total_segments': 0,
        'total_pauses': 0,
        'files_with_errors': 0,
        'validation_errors': []
    }

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        dataset_id = filename.replace('.csv', '')
        output_path = os.path.join(output_dir, f"{dataset_id}_segmented.csv")

        try:
            # Segment file
            processed_df, stats = segment_pxchange_file(file_path, output_path, dataset_id)

            # Update summary
            summary['files_processed'] += 1
            summary['total_patients'] += stats['total_patients']
            summary['total_segments'] += stats['total_segments']
            summary['total_pauses'] += stats['total_pauses']

            if stats['validation_errors']:
                summary['files_with_errors'] += 1
                summary['validation_errors'].extend(stats['validation_errors'])

            print(f"  {filename}: {stats['total_segments']} segments created\n")

        except Exception as e:
            print(f"  ERROR processing {filename}: {e}\n")
            summary['files_with_errors'] += 1

    print(f"\nBatch segmentation complete:")
    print(f"  Files processed: {summary['files_processed']}")
    print(f"  Total patients: {summary['total_patients']}")
    print(f"  Total segments created: {summary['total_segments']}")
    print(f"  Total pauses detected: {summary['total_pauses']}")
    print(f"  Files with validation errors: {summary['files_with_errors']}")

    return summary


def batch_segment_seqofseq_files(input_dir, output_dir, file_pattern='*.csv'):
    """
    Batch segment multiple SeqofSeq files

    Args:
        input_dir: Directory containing raw CSV files
        output_dir: Directory to save segmented files
        file_pattern: Glob pattern for files to process

    Returns:
        summary: Dictionary with overall processing statistics
    """
    import glob

    os.makedirs(output_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(input_dir, file_pattern))
    print(f"Found {len(csv_files)} SeqofSeq files to process\n")

    summary = {
        'files_processed': 0,
        'total_patients': 0,
        'total_segments': 0,
        'total_pauses': 0,
        'files_with_errors': 0,
        'validation_errors': []
    }

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        dataset_id = filename.replace('.csv', '')
        output_path = os.path.join(output_dir, f"{dataset_id}_segmented.csv")

        try:
            # Segment file
            processed_df, stats = segment_seqofseq_file(file_path, output_path, dataset_id)

            # Update summary
            summary['files_processed'] += 1
            summary['total_patients'] += stats['total_patients']
            summary['total_segments'] += stats['total_segments']
            summary['total_pauses'] += stats['total_pauses']

            if stats['validation_errors']:
                summary['files_with_errors'] += 1
                summary['validation_errors'].extend(stats['validation_errors'])

            print(f"  {filename}: {stats['total_segments']} segments created\n")

        except Exception as e:
            print(f"  ERROR processing {filename}: {e}\n")
            summary['files_with_errors'] += 1

    print(f"\nBatch segmentation complete:")
    print(f"  Files processed: {summary['files_processed']}")
    print(f"  Total patients: {summary['total_patients']}")
    print(f"  Total segments created: {summary['total_segments']}")
    print(f"  Total pauses detected: {summary['total_pauses']}")
    print(f"  Files with validation errors: {summary['files_with_errors']}")

    return summary


# ============================================================================
# MAIN ENTRY POINT (Command Line Interface)
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Segment MRI sequences at pause boundaries (replaces pause injection with pseudo-patient architecture)'
    )
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
        print(f"Using custom pause threshold: {args.threshold} minutes\n")

    if args.batch:
        # Batch processing
        if args.pipeline == 'pxchange':
            summary = batch_segment_pxchange_files(args.input, args.output)
        else:
            summary = batch_segment_seqofseq_files(args.input, args.output)

        # Save summary report
        summary_path = os.path.join(args.output, 'segmentation_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Sequence Segmentation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Pipeline: {args.pipeline}\n")
            f.write(f"Files processed: {summary['files_processed']}\n")
            f.write(f"Total patients: {summary['total_patients']}\n")
            f.write(f"Total segments: {summary['total_segments']}\n")
            f.write(f"Total pauses: {summary['total_pauses']}\n")
            f.write(f"Files with errors: {summary['files_with_errors']}\n\n")

            if summary['validation_errors']:
                f.write("Validation Errors:\n")
                for error in summary['validation_errors']:
                    f.write(f"  - {error}\n")

        print(f"\nSummary saved to: {summary_path}")

    else:
        # Single file processing
        if args.pipeline == 'pxchange':
            processed_df, stats = segment_pxchange_file(args.input, args.output)
        else:
            processed_df, stats = segment_seqofseq_file(args.input, args.output)

    print("\nSequence segmentation complete!")
    print("\nNote: This replaces pause_injection.py with pseudo-patient architecture.")
    print("Sequences are now segmented at pause boundaries with entity_type labels.")
