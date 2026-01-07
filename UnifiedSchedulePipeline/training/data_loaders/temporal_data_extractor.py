"""
Temporal Data Extractor for UnifiedSchedulePipeline

Extracts daily session patterns from PXChange segmented data and creates
augmented training data for the temporal schedule model.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add UnifiedSchedulePipeline to path
unified_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, os.path.join(unified_dir, 'UnifiedSchedulePipeline'))

from config import TEMPORAL_DATA_DIR, TEMPORAL_TRAINING_CONFIG, RANDOM_SEED
from datetime_model.temporal_features import extract_temporal_features, features_to_array


class TemporalDataset(Dataset):
    """
    PyTorch Dataset for temporal training data

    Returns:
        temporal_features: [12] features
        session_count: scalar
        session_start_times: [max_sessions] start times (seconds from midnight)
        num_valid_sessions: scalar (how many sessions are valid, rest is padding)
    """

    def __init__(self, dataframe, feature_scaler=None, fit_scaler=False, max_sessions=20):
        self.df = dataframe
        self.max_sessions = max_sessions

        # Fit or use feature scaler
        if fit_scaler:
            self.feature_scaler = StandardScaler()
            features = np.stack(self.df['temporal_features'].values)
            self.feature_scaler.fit(features)
        else:
            self.feature_scaler = feature_scaler

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Extract features
        features = row['temporal_features'].copy()
        if self.feature_scaler is not None:
            features = self.feature_scaler.transform(features.reshape(1, -1))[0]

        # Extract target session count
        session_count = row['num_sessions']

        # Extract and pad start times
        start_times = row['session_start_times']
        padded_start_times = np.zeros(self.max_sessions, dtype=np.float32)
        num_valid = min(len(start_times), self.max_sessions)
        padded_start_times[:num_valid] = start_times[:num_valid]

        return {
            'temporal_features': torch.from_numpy(np.array(features, dtype=np.float32)),
            'session_count': torch.tensor(session_count, dtype=torch.long),
            'session_start_times': torch.from_numpy(padded_start_times),
            'num_valid_sessions': torch.tensor(num_valid, dtype=torch.long)
        }


def extract_temporal_patterns_from_segments(segmented_data_dir=None, measurement_start_sourceID='MRI_MSR_104'):
    """
    Extract daily temporal patterns from PXChange segmented data.

    Note: Using MRI_MSR_104 (measurement start) instead of MRI_EXU_95 as the session marker
    based on the PXChange vocabulary.

    Args:
        segmented_data_dir: Path to preprocessed_segmented directory
        measurement_start_sourceID: SourceID that indicates measurement start

    Returns:
        daily_summaries_df: DataFrame with temporal patterns per date/machine
    """
    # Default to PXChange segmented data
    if segmented_data_dir is None:
        pxchange_dir = os.path.join(unified_dir, 'PXChange_Refactored')
        segmented_data_dir = os.path.join(pxchange_dir, 'data', 'preprocessed_segmented')

    if not os.path.exists(segmented_data_dir):
        raise ValueError(f"Segmented data directory not found: {segmented_data_dir}")

    print(f"[Temporal Extractor] Extracting patterns from: {segmented_data_dir}")

    # Find all segmented files
    pattern = os.path.join(segmented_data_dir, '*_segmented.csv')
    segmented_files = glob.glob(pattern)

    if not segmented_files:
        raise ValueError(f"No segmented files found in {segmented_data_dir}")

    print(f"[Temporal Extractor] Found {len(segmented_files)} segmented files\n")

    all_daily_summaries = []

    for file_path in segmented_files:
        try:
            # Extract machine ID from filename
            filename = os.path.basename(file_path)
            machine_id = filename.replace('_segmented.csv', '')

            # Load data
            df = pd.read_csv(file_path)

            # Check if datetime column exists
            if 'datetime' not in df.columns:
                print(f"  {filename}: No datetime column, skipping")
                continue

            # Parse datetime
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df = df.dropna(subset=['datetime'])
            df = df.sort_values('datetime')

            # Filter for measurement start events (only in real patient sequences)
            measurement_starts = df[
                (df['sourceID'].astype(str) == measurement_start_sourceID) &
                (df['entity_type'] == 0)  # Real patient only
            ].copy()

            if len(measurement_starts) == 0:
                print(f"  {filename}: No measurement start events found, skipping")
                continue

            # Group by date
            measurement_starts['date'] = measurement_starts['datetime'].dt.date

            for date, day_events in measurement_starts.groupby('date'):
                # Extract start times (seconds from midnight)
                start_times = [
                    (t.hour * 3600 + t.minute * 60 + t.second)
                    for t in day_events['datetime']
                ]

                # Sort start times
                start_times = sorted(start_times)

                # Count sessions (for now, each measurement start is a session)
                # Could add gap-based session detection if needed
                num_sessions = len(start_times)

                # Extract day info
                day_of_week = date.weekday()
                day_of_year = date.timetuple().tm_yday

                all_daily_summaries.append({
                    'date': date,
                    'day_of_week': day_of_week,
                    'day_of_year': day_of_year,
                    'num_sessions': num_sessions,
                    'session_start_times': start_times,
                    'machine_id': machine_id
                })

            print(f"  {filename}: {len(measurement_starts)} measurement starts across {len(measurement_starts['date'].unique())} days")

        except Exception as e:
            print(f"  [ERROR] {filename}: {e}")

    daily_summaries_df = pd.DataFrame(all_daily_summaries)

    if len(daily_summaries_df) == 0:
        raise ValueError("No daily summaries could be extracted from segmented data")

    print(f"\n[Temporal Extractor] Summary statistics:")
    print(f"  Total daily summaries: {len(daily_summaries_df)}")
    print(f"  Date range: {daily_summaries_df['date'].min()} to {daily_summaries_df['date'].max()}")
    print(f"  Average sessions per day: {daily_summaries_df['num_sessions'].mean():.2f}")
    print(f"  Session count range: {daily_summaries_df['num_sessions'].min()}-{daily_summaries_df['num_sessions'].max()}")
    print(f"  Unique machines: {daily_summaries_df['machine_id'].nunique()}")

    return daily_summaries_df


def create_augmented_temporal_dataset(daily_summaries_df, augmentation_factor=50, num_components=3):
    """
    Generate augmented temporal training data from real patterns.

    Args:
        daily_summaries_df: DataFrame from extract_temporal_patterns_from_segments()
        augmentation_factor: How many synthetic samples per real sample
        num_components: Number of Gaussian components for timing distribution

    Returns:
        augmented_df: DataFrame with temporal features and targets
    """
    print(f"\n[Temporal Extractor] Creating augmented dataset...")
    print(f"  Real samples: {len(daily_summaries_df)}")
    print(f"  Augmentation factor: {augmentation_factor}")
    print(f"  Target samples: {len(daily_summaries_df) * augmentation_factor}")

    # Fit session count distribution (Poisson)
    session_counts = daily_summaries_df['num_sessions'].values
    lambda_poisson = np.mean(session_counts)
    print(f"\n  Session count distribution: Poisson(λ={lambda_poisson:.2f})")

    # Fit timing distribution (Mixture of Gaussians)
    all_start_times = []
    for start_times in daily_summaries_df['session_start_times']:
        all_start_times.extend(start_times)

    X_times = np.array(all_start_times).reshape(-1, 1)
    gmm = GaussianMixture(n_components=num_components, random_state=RANDOM_SEED)
    gmm.fit(X_times)

    print(f"\n  Timing distribution: Mixture of {num_components} Gaussians")
    for i in range(num_components):
        mean_seconds = gmm.means_[i, 0]
        std_seconds = np.sqrt(gmm.covariances_[i, 0, 0])
        weight = gmm.weights_[i]
        print(f"    Component {i}: mean={mean_seconds/3600:.2f}h, std={std_seconds/3600:.2f}h, weight={weight:.3f}")

    # Generate augmented data
    augmented_data = []

    for _ in range(augmentation_factor):
        for _, row in daily_summaries_df.iterrows():
            # Add some noise to temporal features
            date = row['date']
            date_obj = datetime.combine(date, datetime.min.time())

            # Perturb date slightly (±7 days)
            date_offset = np.random.randint(-7, 8)
            perturbed_date = date_obj + timedelta(days=date_offset)

            # Extract temporal features
            machine_id_encoded = hash(str(row['machine_id'])) % 1000  # Simple encoding
            typical_load = row['num_sessions']

            features = extract_temporal_features(
                datetime_obj=perturbed_date,
                machine_id=machine_id_encoded,
                typical_load=typical_load
            )

            # Convert to array
            feature_array = features_to_array(features)

            # Sample session count from Poisson
            num_sessions = np.random.poisson(lambda_poisson)
            num_sessions = np.clip(num_sessions, 1, 25)  # Constrain to reasonable range

            # Sample start times from GMM
            start_times_samples = gmm.sample(num_sessions)[0].flatten()
            start_times_samples = np.clip(start_times_samples, 7*3600, 19*3600)  # 7 AM - 7 PM
            start_times = sorted(start_times_samples.tolist())

            augmented_data.append({
                'temporal_features': feature_array,
                'num_sessions': num_sessions,
                'session_start_times': start_times,
                'machine_id': row['machine_id'],
                'date': perturbed_date.date()
            })

    augmented_df = pd.DataFrame(augmented_data)
    print(f"\n[Temporal Extractor] Augmented dataset created: {len(augmented_df)} samples")

    return augmented_df


def prepare_temporal_training_data(segmented_data_dir=None, augmentation_factor=50,
                                   batch_size=32, validation_split=0.2,
                                   save_dir=None):
    """
    Complete pipeline: extract patterns, augment, and create dataloaders.

    Args:
        segmented_data_dir: Path to PXChange segmented data
        augmentation_factor: Augmentation factor
        batch_size: Batch size for dataloaders
        validation_split: Validation split fraction
        save_dir: Directory to save processed data (default: TEMPORAL_DATA_DIR)

    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        feature_scaler: Fitted StandardScaler
        metadata: Dict with statistics
    """
    if save_dir is None:
        save_dir = TEMPORAL_DATA_DIR

    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("TEMPORAL DATA PREPARATION PIPELINE")
    print("=" * 70)

    # Step 1: Extract real patterns
    daily_summaries_df = extract_temporal_patterns_from_segments(segmented_data_dir)

    # Save real summaries
    real_summaries_path = os.path.join(save_dir, 'real_daily_summaries.csv')
    daily_summaries_df.to_csv(real_summaries_path, index=False)
    print(f"\n[Temporal Extractor] Saved real summaries to: {real_summaries_path}")

    # Step 2: Create augmented dataset
    augmented_df = create_augmented_temporal_dataset(
        daily_summaries_df,
        augmentation_factor=augmentation_factor
    )

    # Save augmented data
    augmented_path = os.path.join(save_dir, 'augmented_temporal_data.pkl')
    with open(augmented_path, 'wb') as f:
        pickle.dump(augmented_df, f)
    print(f"[Temporal Extractor] Saved augmented data to: {augmented_path}")

    # Step 3: Create train/val split
    indices = np.arange(len(augmented_df))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=validation_split,
        random_state=RANDOM_SEED
    )

    train_df = augmented_df.iloc[train_indices].reset_index(drop=True)
    val_df = augmented_df.iloc[val_indices].reset_index(drop=True)

    print(f"\n[Temporal Extractor] Train/Val split:")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")

    # Step 4: Create datasets
    train_dataset = TemporalDataset(train_df, fit_scaler=True, max_sessions=20)
    val_dataset = TemporalDataset(
        val_df,
        feature_scaler=train_dataset.feature_scaler,
        max_sessions=20
    )

    # Step 5: Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # Metadata
    metadata = {
        'num_real_samples': len(daily_summaries_df),
        'num_augmented_samples': len(augmented_df),
        'augmentation_factor': augmentation_factor,
        'num_train_samples': len(train_df),
        'num_val_samples': len(val_df),
        'temporal_feature_dim': 12,
        'max_sessions': 20
    }

    # Save metadata
    metadata_path = os.path.join(save_dir, 'temporal_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"[Temporal Extractor] Saved metadata to: {metadata_path}")

    print(f"\n[Temporal Extractor] Dataloaders created:")
    print(f"  Training: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"  Validation: {len(val_dataset)} samples ({len(val_loader)} batches)")
    print(f"  Batch size: {batch_size}")

    print("\n" + "=" * 70)
    print("[SUCCESS] Temporal data preparation complete!")
    print("=" * 70)

    return train_loader, val_loader, train_dataset.feature_scaler, metadata


if __name__ == "__main__":
    # Test temporal data extraction
    print("=" * 70)
    print("Testing Temporal Data Extractor")
    print("=" * 70)

    try:
        train_loader, val_loader, scaler, metadata = prepare_temporal_training_data(
            augmentation_factor=10,  # Small for testing
            batch_size=16
        )

        # Test batch
        print("\n" + "=" * 70)
        print("Testing batch retrieval")
        print("=" * 70)
        batch = next(iter(train_loader))
        print("\nSample batch:")
        for key, value in batch.items():
            print(f"  {key}: {value.shape}")

        print("\n[SUCCESS] Temporal data extractor working correctly!")

    except Exception as e:
        print(f"\n[ERROR] Temporal extractor test failed: {e}")
        import traceback
        traceback.print_exc()
