"""
PXChange Data Loader for UnifiedSchedulePipeline

Loads all segmented PXChange files and creates training/validation dataloaders.
Reuses MRISequenceDataset from PXChange_Refactored pipeline.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

# Add PXChange to path
pxchange_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'PXChange_Refactored')
sys.path.insert(0, pxchange_dir)

from preprocessing.data_loader import MRISequenceDataset
from PXChange_Refactored.config import RANDOM_SEED, SOURCEID_VOCAB, PSEUDO_PATIENT_TOKENS


def load_all_pxchange_segments(segmented_data_dir=None):
    """
    Load all segmented PXChange files and combine into single DataFrame.

    Args:
        segmented_data_dir: Path to preprocessed_segmented directory
                          If None, uses default PXChange_Refactored/data/preprocessed_segmented/

    Returns:
        combined_df: Combined DataFrame with all segments
        num_files: Number of files loaded
    """
    if segmented_data_dir is None:
        segmented_data_dir = os.path.join(pxchange_dir, 'data', 'preprocessed_segmented')

    if not os.path.exists(segmented_data_dir):
        raise ValueError(f"Segmented data directory not found: {segmented_data_dir}\n"
                        "Please run preprocessing first: python preprocessing/sequence_segmentation.py")

    # Find all *_segmented.csv files
    pattern = os.path.join(segmented_data_dir, '*_segmented.csv')
    segmented_files = glob.glob(pattern)

    if not segmented_files:
        raise ValueError(f"No segmented files found in {segmented_data_dir}\n"
                        "Expected files matching pattern: *_segmented.csv")

    print(f"[PXChange Loader] Found {len(segmented_files)} segmented files")

    # Load all files
    dfs = []
    for file_path in segmented_files:
        try:
            df = pd.read_csv(file_path)
            machine_id = os.path.basename(file_path).replace('_segmented.csv', '')
            df['machine_id'] = machine_id

            # Create SeqOrder from segment info
            if 'original_sequence_id' in df.columns and 'segment_index' in df.columns:
                df['SeqOrder'] = (
                    machine_id + '_' +
                    df['original_sequence_id'].astype(str) + '_' +
                    df['segment_index'].astype(str)
                )
                num_seqs = len(df['SeqOrder'].unique())
            else:
                print(f"  [WARNING] {machine_id}: Missing segmentation columns, skipping")
                continue

            dfs.append(df)
            print(f"  [OK] Loaded {machine_id}: {num_seqs} sequences")
        except Exception as e:
            print(f"  [ERROR] Failed to load {file_path}: {e}")

    if not dfs:
        raise ValueError("No data could be loaded from segmented files")

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Create unique sequence identifier from original_sequence_id + segment_index + machine_id
    combined_df['SeqOrder'] = (
        combined_df['machine_id'].astype(str) + '_' +
        combined_df['original_sequence_id'].astype(str) + '_' +
        combined_df['segment_index'].astype(str)
    )

    # Statistics
    num_sequences = len(combined_df['SeqOrder'].unique())
    num_real = len(combined_df[combined_df['entity_type'] == 0]['SeqOrder'].unique())
    num_pseudo = len(combined_df[combined_df['entity_type'] != 0]['SeqOrder'].unique())

    print(f"\n[PXChange Loader] Combined statistics:")
    print(f"  Total files: {len(dfs)}")
    print(f"  Total sequences: {num_sequences}")
    print(f"    - Real patient sequences: {num_real}")
    print(f"    - Pseudo-patient sequences: {num_pseudo}")
    print(f"    - Pseudo-patient ratio: {100 * num_pseudo / num_sequences:.1f}%")
    print(f"  Total events: {len(combined_df)}")

    return combined_df, len(dfs)


def encode_categorical_features(df):
    """
    Encode categorical features that may be strings.

    Args:
        df: DataFrame with raw segmented data

    Returns:
        df: DataFrame with encoded features
    """
    print(f"\n[PXChange Loader] Encoding categorical features...")

    # Encode BodyGroup_from and BodyGroup_to if they're strings
    if df['BodyGroup_from'].dtype == 'object':
        le_from = LabelEncoder()
        # Handle NaN values
        df['BodyGroup_from_encoded'] = df['BodyGroup_from'].fillna('UNKNOWN')
        df['BodyGroup_from_encoded'] = le_from.fit_transform(df['BodyGroup_from_encoded'])
        print(f"  BodyGroup_from: {len(le_from.classes_)} unique values")
    else:
        df['BodyGroup_from_encoded'] = df['BodyGroup_from']

    if df['BodyGroup_to'].dtype == 'object':
        le_to = LabelEncoder()
        df['BodyGroup_to_encoded'] = df['BodyGroup_to'].fillna('UNKNOWN')
        df['BodyGroup_to_encoded'] = le_to.fit_transform(df['BodyGroup_to_encoded'])
        print(f"  BodyGroup_to: {len(le_to.classes_)} unique values")
    else:
        df['BodyGroup_to_encoded'] = df['BodyGroup_to']

    # Encode Position and Direction if they exist
    if 'Position' in df.columns and df['Position'].dtype == 'object':
        le_pos = LabelEncoder()
        df['Position_encoded'] = df['Position'].fillna('UNKNOWN')
        df['Position_encoded'] = le_pos.fit_transform(df['Position_encoded'])
        print(f"  Position: {len(le_pos.classes_)} unique values")
    elif 'Position' in df.columns:
        df['Position_encoded'] = df['Position']

    if 'Direction' in df.columns and df['Direction'].dtype == 'object':
        le_dir = LabelEncoder()
        df['Direction_encoded'] = df['Direction'].fillna('UNKNOWN')
        df['Direction_encoded'] = le_dir.fit_transform(df['Direction_encoded'])
        print(f"  Direction: {len(le_dir.classes_)} unique values")
    elif 'Direction' in df.columns:
        df['Direction_encoded'] = df['Direction']

    # Ensure numeric types for conditioning features
    for col in ['Age', 'Weight', 'Height', 'PTAB', 'entity_type']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Replace original columns with encoded versions
    df['BodyGroup_from'] = df['BodyGroup_from_encoded']
    df['BodyGroup_to'] = df['BodyGroup_to_encoded']

    # Add 'Step' column if it doesn't exist (needed by MRISequenceDataset)
    if 'Step' not in df.columns:
        df['Step'] = df.groupby('SeqOrder').cumcount()
        print(f"  Added 'Step' column for sequence ordering")

    # Encode sourceID from string to integer using vocabulary
    if df['sourceID'].dtype == 'object':
        # Create combined vocabulary
        vocab = {**SOURCEID_VOCAB, **PSEUDO_PATIENT_TOKENS}

        def encode_sourceid(s):
            if pd.isna(s) or s == '':
                return vocab.get('PAD', 0)
            return vocab.get(str(s), vocab.get('UNK', 17))

        df['sourceID'] = df['sourceID'].apply(encode_sourceid)
        print(f"  Encoded sourceID using vocabulary (vocab size: {len(vocab)})")

    return df


def create_pxchange_dataloaders(batch_size=32, validation_split=0.2,
                                segmented_data_dir=None, num_workers=0):
    """
    Create training and validation dataloaders for PXChange models.

    Args:
        batch_size: Batch size for dataloaders
        validation_split: Fraction of data for validation (default: 0.2)
        segmented_data_dir: Path to segmented data directory (default: auto-detect)
        num_workers: Number of workers for dataloaders (default: 0 for Windows)

    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        conditioning_scaler: Fitted StandardScaler for conditioning features
        metadata: Dict with statistics and info
    """
    # Load all segmented data
    combined_df, num_files = load_all_pxchange_segments(segmented_data_dir)

    # Encode categorical features
    combined_df = encode_categorical_features(combined_df)

    # Split sequences into train/val
    from sklearn.model_selection import train_test_split
    seq_orders = combined_df['SeqOrder'].unique()
    train_seqs, val_seqs = train_test_split(
        seq_orders,
        test_size=validation_split,
        random_state=RANDOM_SEED
    )

    train_df = combined_df[combined_df['SeqOrder'].isin(train_seqs)]
    val_df = combined_df[combined_df['SeqOrder'].isin(val_seqs)]

    print(f"\n[PXChange Loader] Train/Val split:")
    print(f"  Training sequences: {len(train_seqs)} ({len(train_df)} events)")
    print(f"  Validation sequences: {len(val_seqs)} ({len(val_df)} events)")

    # Create datasets using existing MRISequenceDataset
    train_dataset = MRISequenceDataset(train_df, fit_scaler=True)
    val_dataset = MRISequenceDataset(val_df, conditioning_scaler=train_dataset.conditioning_scaler)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Prepare metadata
    metadata = {
        'num_files': num_files,
        'num_sequences': len(seq_orders),
        'num_train_sequences': len(train_seqs),
        'num_val_sequences': len(val_seqs),
        'num_train_events': len(train_df),
        'num_val_events': len(val_df),
        'vocab_size': 19,  # From PXChange config
        'conditioning_dim': 7  # Age, Weight, Height, BodyGroup_from, BodyGroup_to, PTAB, entity_type
    }

    print(f"\n[PXChange Loader] Dataloaders created:")
    print(f"  Training: {len(train_dataset)} sequences ({len(train_loader)} batches)")
    print(f"  Validation: {len(val_dataset)} sequences ({len(val_loader)} batches)")
    print(f"  Batch size: {batch_size}")

    return train_loader, val_loader, train_dataset.conditioning_scaler, metadata


if __name__ == "__main__":
    # Test data loading
    print("=" * 70)
    print("Testing PXChange Data Loader")
    print("=" * 70)

    try:
        train_loader, val_loader, scaler, metadata = create_pxchange_dataloaders(batch_size=16)

        # Test batch
        print("\n" + "=" * 70)
        print("Testing batch retrieval")
        print("=" * 70)
        batch = next(iter(train_loader))
        print("\nSample batch:")
        for key, value in batch.items():
            print(f"  {key}: {value.shape}")

        print("\n[SUCCESS] PXChange data loader working correctly!")

    except Exception as e:
        print(f"\n[ERROR] Data loader test failed: {e}")
        import traceback
        traceback.print_exc()
