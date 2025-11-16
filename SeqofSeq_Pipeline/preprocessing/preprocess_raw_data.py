"""
Preprocessing module to convert raw MRI scan data to training format
"""
import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from config import (
    DATA_DIR, SPECIAL_TOKENS, COIL_FEATURES_PREFIX,
    DATA_CONFIG, CONDITIONING_FEATURES
)


def load_raw_data(data_file='176625.csv'):
    """Load raw MRI scan data from CSV."""
    file_path = os.path.join(DATA_DIR, data_file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path)
    print(f"[OK] Loaded {len(df)} scans from {data_file}")
    return df


def extract_coil_features(df):
    """Extract coil configuration features (boolean columns)."""
    coil_cols = []
    for col in df.columns:
        if any(col.startswith(prefix) for prefix in COIL_FEATURES_PREFIX):
            coil_cols.append(col)

    if len(coil_cols) > 0:
        # Handle NaN values by filling with 0 (False)
        coil_features = df[coil_cols].fillna(0).astype(int).values
        print(f"[OK] Extracted {len(coil_cols)} coil features")
        return coil_features, coil_cols
    else:
        print("[WARN] No coil features found")
        return None, []


def build_vocabulary(df, column='Sequence'):
    """Build vocabulary from unique values in a column."""
    # Start with special tokens
    vocab = SPECIAL_TOKENS.copy()

    # Add unique values from the column
    unique_values = df[column].unique()
    next_id = max(vocab.values()) + 1

    for value in sorted(unique_values):
        if pd.notna(value) and value not in vocab:
            vocab[value] = next_id
            next_id += 1

    # Create reverse mapping
    id_to_token = {v: k for k, v in vocab.items()}

    print(f"[OK] Built vocabulary with {len(vocab)} tokens for '{column}'")
    return vocab, id_to_token


def encode_categorical_features(df, features):
    """Encode categorical features using LabelEncoder."""
    encoders = {}
    encoded_df = df.copy()

    for feature in features:
        # Extract the base feature name (remove '_encoded' suffix if present)
        base_feature = feature.replace('_encoded', '')

        if base_feature in df.columns:
            le = LabelEncoder()
            # Handle missing values
            valid_mask = df[base_feature].notna()
            encoded_df[feature] = -1  # Default for missing

            if valid_mask.sum() > 0:
                encoded_df.loc[valid_mask, feature] = le.fit_transform(
                    df.loc[valid_mask, base_feature].astype(str)
                )

            encoders[base_feature] = le
            print(f"[OK] Encoded {base_feature}: {len(le.classes_)} unique values")
        else:
            print(f"[WARN] Feature '{base_feature}' not found in data")
            encoded_df[feature] = 0

    return encoded_df, encoders


def group_scans_into_sequences(df):
    """Group individual scans into sequences by PatientID."""
    if 'PatientID' not in df.columns:
        raise ValueError("PatientID column is required to group scans into sequences")

    # Sort by PatientID and startTime
    df = df.sort_values(['PatientID', 'startTime']).reset_index(drop=True)

    # Group by PatientID
    sequences = []
    for patient_id, group in df.groupby('PatientID'):
        # Filter out sequences that are too short or too long
        seq_len = len(group)
        if seq_len < DATA_CONFIG['min_sequence_length']:
            continue
        if seq_len > DATA_CONFIG['max_sequence_length']:
            # Split into multiple sequences
            for i in range(0, seq_len, DATA_CONFIG['max_sequence_length']):
                sub_group = group.iloc[i:i + DATA_CONFIG['max_sequence_length']]
                if len(sub_group) >= DATA_CONFIG['min_sequence_length']:
                    sequences.append(sub_group)
        else:
            sequences.append(group)

    print(f"[OK] Created {len(sequences)} sequences from {df['PatientID'].nunique()} patients")
    return sequences


def preprocess_mri_data(data_file='176625.csv', save_preprocessed=True):
    """
    Complete preprocessing pipeline for MRI scan data.

    Returns:
        preprocessed_df: DataFrame with preprocessed sequences
        vocab: Vocabulary mapping for sequences
        encoders: Label encoders for categorical features
        metadata: Additional metadata
    """
    print(f"\n{'='*70}")
    print("PREPROCESSING MRI SCAN DATA")
    print(f"{'='*70}\n")

    # Load raw data
    df = load_raw_data(data_file)

    # Parse datetime columns
    print("Parsing datetime columns...")
    df['startTime'] = pd.to_datetime(df['startTime'])
    df['endTime'] = pd.to_datetime(df['endTime'])

    # Remove outliers in duration
    if DATA_CONFIG['duration_outlier_std'] is not None:
        duration_mean = df['duration'].mean()
        duration_std = df['duration'].std()
        threshold = duration_mean + DATA_CONFIG['duration_outlier_std'] * duration_std

        original_len = len(df)
        df = df[df['duration'] <= threshold]
        print(f"[OK] Removed {original_len - len(df)} duration outliers (>{threshold:.0f}s)")

    # Build vocabulary for sequences
    sequence_vocab, sequence_id_to_token = build_vocabulary(df, 'Sequence')

    # Build vocabulary for protocols (optional, for additional context)
    protocol_vocab, protocol_id_to_token = build_vocabulary(df, 'Protocol')

    # Encode sequences
    df['sequence_id'] = df['Sequence'].map(sequence_vocab)
    df['protocol_id'] = df['Protocol'].map(protocol_vocab)

    # Handle unknown sequences/protocols
    df['sequence_id'] = df['sequence_id'].fillna(SPECIAL_TOKENS['UNK'])
    df['protocol_id'] = df['protocol_id'].fillna(SPECIAL_TOKENS['UNK'])

    # Extract coil features
    coil_features, coil_cols = extract_coil_features(df)

    # Encode categorical conditioning features
    print("\nEncoding conditioning features...")
    df, encoders = encode_categorical_features(df, CONDITIONING_FEATURES)

    # Group scans into sequences
    print("\nGrouping scans into sequences...")
    sequences = group_scans_into_sequences(df)

    # Create preprocessed DataFrame
    preprocessed_data = []
    for seq_idx, seq_df in enumerate(sequences):
        patient_id = seq_df['PatientID'].iloc[0]

        # Add sequence-level information
        seq_df = seq_df.copy()
        seq_df['sequence_idx'] = seq_idx
        seq_df['seq_position'] = range(len(seq_df))
        seq_df['seq_length'] = len(seq_df)

        preprocessed_data.append(seq_df)

    preprocessed_df = pd.concat(preprocessed_data, ignore_index=True)

    # Save preprocessed data
    if save_preprocessed:
        preprocessed_dir = os.path.join(DATA_DIR, 'preprocessed')
        os.makedirs(preprocessed_dir, exist_ok=True)

        # Save DataFrame
        output_file = os.path.join(preprocessed_dir, 'preprocessed_data.csv')
        preprocessed_df.to_csv(output_file, index=False)
        print(f"\n[OK] Saved preprocessed data to {output_file}")

        # Save vocabularies and encoders
        metadata = {
            'sequence_vocab': sequence_vocab,
            'sequence_id_to_token': sequence_id_to_token,
            'protocol_vocab': protocol_vocab,
            'protocol_id_to_token': protocol_id_to_token,
            'encoders': encoders,
            'coil_cols': coil_cols,
            'vocab_size': len(sequence_vocab),
            'num_conditioning_features': len(CONDITIONING_FEATURES) + len(coil_cols)
        }

        metadata_file = os.path.join(preprocessed_dir, 'metadata.pkl')
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"[OK] Saved metadata to {metadata_file}")

    print(f"\n{'='*70}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*70}\n")
    print(f"Total sequences: {preprocessed_df['sequence_idx'].nunique()}")
    print(f"Total scans: {len(preprocessed_df)}")
    print(f"Avg sequence length: {preprocessed_df.groupby('sequence_idx').size().mean():.1f}")
    print(f"Vocabulary size: {len(sequence_vocab)}")
    print(f"Conditioning features: {len(CONDITIONING_FEATURES) + len(coil_cols)}")

    return preprocessed_df, metadata


if __name__ == "__main__":
    # Run preprocessing
    preprocess_mri_data()
