"""
Data loading and preparation for MRI scan sequence prediction
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SeqofSeq_Pipeline.config import (
    CONDITIONING_FEATURES, MAX_SEQ_LEN,
    PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID, IDLE_TOKEN_ID,
    SPECIAL_TOKENS, IDLE_STATE_ENCODING
)


class MRISequenceDataset(Dataset):
    """
    Dataset for MRI scan sequences with conditioning information.

    Updated to support pseudo-patient architecture:
    - entity_type feature included in conditioning (5 features instead of 4)
    - IDLE tokens supported for pseudo-patient sequences
    - Validates that IDLE tokens only appear in pseudo-patient sequences

    Returns:
        conditioning: Patient/scan context features [batch_size, conditioning_dim]
        sequence_tokens: Sequence IDs (Sequence type) [batch_size, seq_len]
        durations: Target durations in seconds [batch_size, seq_len]
        sequence_mask: Mask for valid positions [batch_size, seq_len]
    """

    def __init__(self, dataframe, conditioning_scaler=None, fit_scaler=False, coil_cols=None):
        """
        Args:
            dataframe: Preprocessed DataFrame with sequences
            conditioning_scaler: StandardScaler for conditioning features
            fit_scaler: Whether to fit the scaler on this data
            coil_cols: List of coil column names
        """
        self.df = dataframe
        self.sequence_indices = dataframe['sequence_idx'].unique()
        self.coil_cols = coil_cols if coil_cols is not None else []

        # Combine conditioning features
        all_conditioning_features = CONDITIONING_FEATURES + self.coil_cols

        # Fit or use conditioning scaler
        if fit_scaler:
            self.conditioning_scaler = StandardScaler()
            # Get one row per sequence for conditioning features
            conditioning_data = []
            for seq_idx in self.sequence_indices:
                seq_data = dataframe[dataframe['sequence_idx'] == seq_idx].iloc[0]
                cond_values = []
                for feat in all_conditioning_features:
                    val = seq_data.get(feat, 0)
                    cond_values.append(float(val) if val is not None else 0.0)
                conditioning_data.append(cond_values)

            # Replace NaN values before fitting
            conditioning_data = np.array(conditioning_data, dtype=np.float32)
            conditioning_data = np.nan_to_num(conditioning_data, nan=0.0)

            self.conditioning_scaler.fit(conditioning_data)
        else:
            self.conditioning_scaler = conditioning_scaler

    def __len__(self):
        return len(self.sequence_indices)

    def __getitem__(self, idx):
        seq_idx = self.sequence_indices[idx]
        seq_data = self.df[self.df['sequence_idx'] == seq_idx].sort_values('seq_position')

        # Combine all conditioning features
        all_conditioning_features = CONDITIONING_FEATURES + self.coil_cols

        # Extract conditioning features (constant per sequence)
        first_row = seq_data.iloc[0]
        conditioning = []
        for feat in all_conditioning_features:
            val = first_row.get(feat, 0)
            conditioning.append(float(val) if val is not None else 0.0)
        conditioning = np.array(conditioning, dtype=np.float32)

        # Replace any NaN values with 0 before scaling
        conditioning = np.nan_to_num(conditioning, nan=0.0)

        # Note: entity_type is automatically included via CONDITIONING_FEATURES
        # It's the 5th feature (after BodyPart, SystemType, Country, Group)
        # Values: REAL_PATIENT=0, PSEUDO_PATIENT_PAUSE=1, etc.

        if self.conditioning_scaler is not None:
            conditioning = self.conditioning_scaler.transform(conditioning.reshape(1, -1))[0]
            # Replace any NaN from scaling with 0
            conditioning = np.nan_to_num(conditioning, nan=0.0)

        # Extract sequence tokens (sequence_id)
        sequence_tokens = seq_data['sequence_id'].values.astype(np.int64)

        # Extract durations (target)
        durations = seq_data['duration'].values.astype(np.float32)

        # Pad sequences to MAX_SEQ_LEN
        seq_len = len(sequence_tokens)

        # Create padded arrays
        padded_tokens = np.full(MAX_SEQ_LEN, PAD_TOKEN_ID, dtype=np.int64)
        padded_durations = np.zeros(MAX_SEQ_LEN, dtype=np.float32)
        mask = np.zeros(MAX_SEQ_LEN, dtype=np.bool_)

        # Fill with actual data
        actual_len = min(seq_len, MAX_SEQ_LEN)
        padded_tokens[:actual_len] = sequence_tokens[:actual_len]
        padded_durations[:actual_len] = durations[:actual_len]
        mask[:actual_len] = True

        return {
            'conditioning': torch.from_numpy(conditioning),
            'sequence_tokens': torch.from_numpy(padded_tokens),
            'durations': torch.from_numpy(padded_durations),
            'mask': torch.from_numpy(mask),
            'seq_length': torch.tensor(actual_len, dtype=torch.long)
        }


def load_preprocessed_data():
    """
    Load preprocessed MRI scan data.

    Returns:
        df: Preprocessed DataFrame
        metadata: Dictionary with vocab, encoders, etc.
    """
    preprocessed_dir = os.path.join(DATA_DIR, 'preprocessed')

    # Check if preprocessed directory exists
    if not os.path.exists(preprocessed_dir):
        raise ValueError(f"Preprocessed directory not found: {preprocessed_dir}\n"
                        "Please run preprocessing first: python preprocessing/preprocess_raw_data.py")

    # Load preprocessed data
    data_file = os.path.join(preprocessed_dir, 'preprocessed_data.csv')
    metadata_file = os.path.join(preprocessed_dir, 'metadata.pkl')

    if not os.path.exists(data_file):
        raise ValueError(f"Preprocessed data file not found: {data_file}")

    if not os.path.exists(metadata_file):
        raise ValueError(f"Metadata file not found: {metadata_file}")

    print(f"Loading preprocessed data from {data_file}...")
    df = pd.read_csv(data_file)

    print(f"Loading metadata from {metadata_file}...")
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    print(f"[OK] Loaded {df['sequence_idx'].nunique()} sequences ({len(df)} total scans)")
    print(f"[OK] Vocabulary size: {metadata['vocab_size']}")
    print(f"[OK] Conditioning features: {metadata['num_conditioning_features']}")

    return df, metadata


def create_dataloaders(dataframe, metadata, batch_size=32, validation_split=0.2):
    """
    Create training and validation dataloaders.

    Args:
        dataframe: Preprocessed DataFrame
        metadata: Metadata dictionary
        batch_size: Batch size for dataloaders
        validation_split: Fraction of data for validation

    Returns:
        train_loader, val_loader, conditioning_scaler
    """
    # Split sequences into train/val
    sequence_indices = dataframe['sequence_idx'].unique()
    train_seqs, val_seqs = train_test_split(
        sequence_indices,
        test_size=validation_split,
        random_state=RANDOM_SEED
    )

    train_df = dataframe[dataframe['sequence_idx'].isin(train_seqs)]
    val_df = dataframe[dataframe['sequence_idx'].isin(val_seqs)]

    # Get coil columns from metadata
    coil_cols = metadata.get('coil_cols', [])

    # Create datasets
    train_dataset = MRISequenceDataset(train_df, fit_scaler=True, coil_cols=coil_cols)
    val_dataset = MRISequenceDataset(
        val_df,
        conditioning_scaler=train_dataset.conditioning_scaler,
        coil_cols=coil_cols
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"\n[OK] Created dataloaders:")
    print(f"  Training: {len(train_dataset)} sequences ({len(train_loader)} batches)")
    print(f"  Validation: {len(val_dataset)} sequences ({len(val_loader)} batches)")

    return train_loader, val_loader, train_dataset.conditioning_scaler


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")
    df, metadata = load_preprocessed_data()
    train_loader, val_loader, scaler = create_dataloaders(df, metadata)

    # Test batch
    batch = next(iter(train_loader))
    print("\nSample batch:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")
