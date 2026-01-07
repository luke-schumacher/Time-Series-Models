"""
SeqofSeq Data Loader for UnifiedSchedulePipeline

Loads all segmented SeqofSeq files and creates training/validation dataloaders.
Reuses MRISequenceDataset from SeqofSeq_Pipeline.
Handles metadata generation if not available.
"""

import os
import sys
import glob
import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

# Add SeqofSeq to path
seqofseq_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'SeqofSeq_Pipeline')
sys.path.insert(0, seqofseq_dir)

from preprocessing.data_loader import MRISequenceDataset
from config import RANDOM_SEED


def load_all_seqofseq_segments(segmented_data_dir=None):
    """
    Load all segmented SeqofSeq files and combine into single DataFrame.

    Args:
        segmented_data_dir: Path to preprocessed_segmented directory
                          If None, uses default SeqofSeq_Pipeline/data/preprocessed_segmented/

    Returns:
        combined_df: Combined DataFrame with all segments
        num_files: Number of files loaded
    """
    if segmented_data_dir is None:
        segmented_data_dir = os.path.join(seqofseq_dir, 'data', 'preprocessed_segmented')

    if not os.path.exists(segmented_data_dir):
        raise ValueError(f"Segmented data directory not found: {segmented_data_dir}\n"
                        "Please run preprocessing first: python preprocessing/sequence_segmentation.py")

    # Find all *_segmented.csv files
    pattern = os.path.join(segmented_data_dir, '*_segmented.csv')
    segmented_files = glob.glob(pattern)

    if not segmented_files:
        raise ValueError(f"No segmented files found in {segmented_data_dir}\n"
                        "Expected files matching pattern: *_segmented.csv")

    print(f"[SeqofSeq Loader] Found {len(segmented_files)} segmented files")

    # Load all files
    dfs = []
    for file_path in segmented_files:
        try:
            df = pd.read_csv(file_path)
            file_id = os.path.basename(file_path).replace('_segmented.csv', '')
            df['file_id'] = file_id
            dfs.append(df)
            print(f"  [OK] Loaded {file_id}: {len(df['sequence_idx'].unique())} sequences")
        except Exception as e:
            print(f"  [ERROR] Failed to load {file_path}: {e}")

    if not dfs:
        raise ValueError("No data could be loaded from segmented files")

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Renumber sequence indices to avoid conflicts across files
    combined_df['sequence_idx'] = combined_df.groupby(['file_id', 'sequence_idx']).ngroup()

    # Statistics
    num_sequences = len(combined_df['sequence_idx'].unique())
    num_real = len(combined_df[combined_df['entity_type'] == 0]['sequence_idx'].unique())
    num_pseudo = len(combined_df[combined_df['entity_type'] != 0]['sequence_idx'].unique())

    print(f"\n[SeqofSeq Loader] Combined statistics:")
    print(f"  Total files: {len(dfs)}")
    print(f"  Total sequences: {num_sequences}")
    print(f"    - Real patient sequences: {num_real}")
    print(f"    - Pseudo-patient sequences: {num_pseudo}")
    print(f"    - Pseudo-patient ratio: {100 * num_pseudo / num_sequences:.1f}%")
    print(f"  Total scans: {len(combined_df)}")

    return combined_df, len(dfs)


def generate_metadata_from_data(df):
    """
    Generate metadata if not available from preprocessing.

    Args:
        df: Combined DataFrame

    Returns:
        metadata: Dict with vocab, encoders, feature info
    """
    print("\n[SeqofSeq Loader] Generating metadata from data...")

    # Build vocabulary from sequence_id column
    unique_sequences = df['sequence_id'].unique()
    vocab_size = len(unique_sequences) + 5  # +5 for special tokens

    # Find coil columns
    coil_cols = [col for col in df.columns if col.startswith('#0_') or col.startswith('#1_')]

    # Calculate conditioning dimension
    num_base_features = 5  # BodyPart, SystemType, Country, Group, entity_type
    num_conditioning_features = num_base_features + len(coil_cols)

    metadata = {
        'vocab_size': vocab_size,
        'num_conditioning_features': num_conditioning_features,
        'coil_cols': coil_cols,
        'num_sequences': len(df['sequence_idx'].unique()),
        'generated_from_data': True
    }

    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Conditioning features: {num_conditioning_features} ({num_base_features} base + {len(coil_cols)} coil)")

    return metadata


def load_or_generate_metadata(metadata_path=None, df=None):
    """
    Load metadata from file or generate from data if not available.

    Args:
        metadata_path: Path to metadata.pkl file
        df: DataFrame to generate metadata from if file not found

    Returns:
        metadata: Dict with vocab, encoders, feature info
    """
    if metadata_path is None:
        metadata_path = os.path.join(seqofseq_dir, 'data', 'preprocessed', 'metadata.pkl')

    # Try to load existing metadata
    if os.path.exists(metadata_path):
        print(f"[SeqofSeq Loader] Loading metadata from {metadata_path}...")
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            print(f"  [OK] Loaded metadata: vocab_size={metadata['vocab_size']}, "
                  f"conditioning_dim={metadata['num_conditioning_features']}")
            return metadata
        except Exception as e:
            print(f"  [WARNING] Failed to load metadata: {e}")
            print(f"  [INFO] Will generate metadata from data")

    # Generate metadata from data
    if df is None:
        raise ValueError("Cannot generate metadata without DataFrame")

    return generate_metadata_from_data(df)


def create_seqofseq_dataloaders(batch_size=32, validation_split=0.2,
                                segmented_data_dir=None, metadata_path=None,
                                num_workers=0):
    """
    Create training and validation dataloaders for SeqofSeq models.

    Args:
        batch_size: Batch size for dataloaders
        validation_split: Fraction of data for validation (default: 0.2)
        segmented_data_dir: Path to segmented data directory (default: auto-detect)
        metadata_path: Path to metadata.pkl (default: auto-detect or generate)
        num_workers: Number of workers for dataloaders (default: 0 for Windows)

    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        conditioning_scaler: Fitted StandardScaler for conditioning features
        metadata: Dict with vocabulary, features, and statistics
    """
    # Load all segmented data
    combined_df, num_files = load_all_seqofseq_segments(segmented_data_dir)

    # Load or generate metadata
    metadata = load_or_generate_metadata(metadata_path, combined_df)

    # Split sequences into train/val
    from sklearn.model_selection import train_test_split
    sequence_indices = combined_df['sequence_idx'].unique()
    train_seqs, val_seqs = train_test_split(
        sequence_indices,
        test_size=validation_split,
        random_state=RANDOM_SEED
    )

    train_df = combined_df[combined_df['sequence_idx'].isin(train_seqs)]
    val_df = combined_df[combined_df['sequence_idx'].isin(val_seqs)]

    print(f"\n[SeqofSeq Loader] Train/Val split:")
    print(f"  Training sequences: {len(train_seqs)} ({len(train_df)} scans)")
    print(f"  Validation sequences: {len(val_seqs)} ({len(val_df)} scans)")

    # Create datasets using existing MRISequenceDataset
    coil_cols = metadata.get('coil_cols', [])
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

    # Update metadata with split info
    metadata.update({
        'num_files': num_files,
        'num_sequences': len(sequence_indices),
        'num_train_sequences': len(train_seqs),
        'num_val_sequences': len(val_seqs),
        'num_train_scans': len(train_df),
        'num_val_scans': len(val_df)
    })

    print(f"\n[SeqofSeq Loader] Dataloaders created:")
    print(f"  Training: {len(train_dataset)} sequences ({len(train_loader)} batches)")
    print(f"  Validation: {len(val_dataset)} sequences ({len(val_loader)} batches)")
    print(f"  Batch size: {batch_size}")

    return train_loader, val_loader, train_dataset.conditioning_scaler, metadata


if __name__ == "__main__":
    # Test data loading
    print("=" * 70)
    print("Testing SeqofSeq Data Loader")
    print("=" * 70)

    try:
        train_loader, val_loader, scaler, metadata = create_seqofseq_dataloaders(batch_size=16)

        # Test batch
        print("\n" + "=" * 70)
        print("Testing batch retrieval")
        print("=" * 70)
        batch = next(iter(train_loader))
        print("\nSample batch:")
        for key, value in batch.items():
            print(f"  {key}: {value.shape}")

        print("\n[SUCCESS] SeqofSeq data loader working correctly!")

    except Exception as e:
        print(f"\n[ERROR] Data loader test failed: {e}")
        import traceback
        traceback.print_exc()
