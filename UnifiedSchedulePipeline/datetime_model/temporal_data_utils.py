"""
Temporal Data Utilities
Extract temporal patterns from PXChange data and generate augmented training data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from sklearn.mixture import GaussianMixture
import glob
import pickle
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import TEMPORAL_TRAINING_CONFIG, TEMPORAL_DATA_DIR, PXCHANGE_DIR
from datetime_model.temporal_features import TemporalFeatureExtractor


def extract_temporal_patterns_from_pxchange(pxchange_data_dir, measurement_start_sourceID='MRI_EXU_95',
                                              session_gap_hours=4):
    """
    Extract daily temporal patterns from PXChange raw data

    Args:
        pxchange_data_dir: Directory containing PXChange raw CSV files
        measurement_start_sourceID: SourceID that indicates measurement start
        session_gap_hours: Gap (in hours) to consider as new session

    Returns:
        daily_summaries: DataFrame with columns:
            - date, day_of_week, day_of_year, num_sessions, session_start_times, machine_id
    """
    print(f"Extracting temporal patterns from: {pxchange_data_dir}")

    csv_files = glob.glob(os.path.join(pxchange_data_dir, '*.csv'))
    print(f"Found {len(csv_files)} CSV files\n")

    all_daily_summaries = []

    for file_path in csv_files:
        try:
            # Extract machine ID from filename
            filename = os.path.basename(file_path)
            machine_id = filename.replace('.csv', '')

            # Load data
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')

            # Filter for measurement start events
            measurement_starts = df[df['sourceID'] == measurement_start_sourceID].copy()

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

                # Calculate gaps to identify separate sessions
                if len(start_times) > 1:
                    gaps = np.diff(start_times)
                    session_gap_threshold = session_gap_hours * 3600

                    # Count sessions (separated by large gaps)
                    num_sessions = 1 + np.sum(gaps > session_gap_threshold)
                else:
                    num_sessions = 1

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

            print(f"  {filename}: Extracted {len(measurement_starts)} measurement starts across {len(measurement_starts['date'].unique())} days")

        except Exception as e:
            print(f"  Error processing {filename}: {e}")

    daily_summaries_df = pd.DataFrame(all_daily_summaries)

    print(f"\nTotal daily summaries extracted: {len(daily_summaries_df)}")
    print(f"Average sessions per day: {daily_summaries_df['num_sessions'].mean():.2f}")
    print(f"Session count range: {daily_summaries_df['num_sessions'].min()}-{daily_summaries_df['num_sessions'].max()}")

    return daily_summaries_df


def fit_timing_distribution(start_times_list, num_components=3):
    """
    Fit Gaussian Mixture Model to start times

    Args:
        start_times_list: List of lists of start times (seconds from midnight)
        num_components: Number of Gaussian components

    Returns:
        gmm: Fitted GaussianMixture model
    """
    # Flatten all start times
    all_start_times = []
    for start_times in start_times_list:
        all_start_times.extend(start_times)

    if len(all_start_times) == 0:
        return None

    # Reshape for sklearn
    X = np.array(all_start_times).reshape(-1, 1)

    # Fit GMM
    gmm = GaussianMixture(n_components=num_components, random_state=42)
    gmm.fit(X)

    print(f"\nFitted GMM with {num_components} components:")
    for i in range(num_components):
        mean_seconds = gmm.means_[i, 0]
        std_seconds = np.sqrt(gmm.covariances_[i, 0, 0])
        weight = gmm.weights_[i]

        mean_hours = mean_seconds / 3600
        std_hours = std_seconds / 3600

        print(f"  Component {i}: mean={mean_hours:.2f}h, std={std_hours:.2f}h, weight={weight:.3f}")

    return gmm


def augment_temporal_data(real_daily_summaries, augmentation_factor=50, num_components=3):
    """
    Generate synthetic daily schedules based on observed patterns

    Args:
        real_daily_summaries: DataFrame with real daily summaries
        augmentation_factor: How many synthetic samples per real sample
        num_components: Number of Gaussian components for timing distribution

    Returns:
        augmented_df: DataFrame with both real and synthetic daily schedules
    """
    print(f"\nAugmenting temporal data (factor={augmentation_factor})...")

    # Fit session count distribution per day of week
    session_count_by_dow = {}

    for dow in range(7):
        dow_data = real_daily_summaries[real_daily_summaries['day_of_week'] == dow]

        if len(dow_data) > 0:
            counts = dow_data['num_sessions'].values
            mean_count = np.mean(counts)
            std_count = np.std(counts) if len(counts) > 1 else 1.0

            session_count_by_dow[dow] = {'mean': mean_count, 'std': std_count}
        else:
            # Default to overall mean if no data for this day of week
            session_count_by_dow[dow] = {
                'mean': real_daily_summaries['num_sessions'].mean(),
                'std': real_daily_summaries['num_sessions'].std()
            }

    # Fit timing distribution (GMM) per day of week
    timing_gmm_by_dow = {}

    for dow in range(7):
        dow_data = real_daily_summaries[real_daily_summaries['day_of_week'] == dow]

        if len(dow_data) > 0:
            start_times_list = dow_data['session_start_times'].tolist()
            gmm = fit_timing_distribution(start_times_list, num_components)
            timing_gmm_by_dow[dow] = gmm
        else:
            # Use overall distribution
            all_start_times = real_daily_summaries['session_start_times'].tolist()
            timing_gmm_by_dow[dow] = fit_timing_distribution(all_start_times, num_components)

    # Generate synthetic samples
    synthetic_samples = []
    machines = real_daily_summaries['machine_id'].unique()

    for _ in range(len(real_daily_summaries) * augmentation_factor):
        # Random day of week
        dow = np.random.randint(0, 7)

        # Random day of year (consistent with day of week)
        base_day = dow  # Start from a Monday (day 0 = Monday)
        day_of_year = base_day + np.random.randint(0, 52) * 7  # Pick a random week

        # Sample session count
        count_params = session_count_by_dow[dow]
        num_sessions = int(np.round(np.random.normal(count_params['mean'], count_params['std'])))
        num_sessions = max(1, min(num_sessions, 25))  # Clamp to reasonable range

        # Sample start times from GMM
        gmm = timing_gmm_by_dow[dow]
        if gmm is not None:
            start_times_raw = gmm.sample(num_sessions)[0].flatten()
            start_times = np.clip(start_times_raw, 21600, 72000)  # 6 AM to 8 PM
            start_times = sorted(start_times.tolist())
        else:
            # Fallback: uniform distribution
            start_times = sorted(np.random.uniform(21600, 72000, num_sessions).tolist())

        # Random machine
        machine_id = np.random.choice(machines)

        # Create synthetic date
        synthetic_date = datetime(2024, 1, 1) + timedelta(days=int(day_of_year))

        synthetic_samples.append({
            'date': synthetic_date.date(),
            'day_of_week': dow,
            'day_of_year': day_of_year,
            'num_sessions': num_sessions,
            'session_start_times': start_times,
            'machine_id': machine_id,
            'is_synthetic': True
        })

    # Combine real and synthetic
    real_daily_summaries['is_synthetic'] = False
    augmented_df = pd.concat([
        real_daily_summaries,
        pd.DataFrame(synthetic_samples)
    ], ignore_index=True)

    print(f"Generated {len(synthetic_samples)} synthetic samples")
    print(f"Total samples: {len(augmented_df)}")
    print(f"Real: {(~augmented_df['is_synthetic']).sum()}, Synthetic: {augmented_df['is_synthetic'].sum()}")

    return augmented_df


def create_temporal_training_dataset(augmented_daily_summaries, save_path=None):
    """
    Create training dataset with features and targets

    Args:
        augmented_daily_summaries: DataFrame with daily summaries
        save_path: Path to save the dataset (optional)

    Returns:
        dataset: Dictionary with 'features', 'targets'
    """
    print("\nCreating temporal training dataset...")

    feature_extractor = TemporalFeatureExtractor()

    features_list = []
    targets_list = []

    for idx, row in augmented_daily_summaries.iterrows():
        # Extract features
        date = row['date']
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d').date()

        machine_id = int(row['machine_id']) if isinstance(row['machine_id'], (int, float, str)) else 0
        typical_load = row['num_sessions']  # Use actual count as typical load

        features = feature_extractor.extract(
            date=date,
            machine_id=machine_id,
            typical_load=typical_load
        )

        # Create target
        num_sessions = row['num_sessions']
        start_times = row['session_start_times']

        # Pad or truncate start times to fixed length (max 20)
        max_sessions = 20
        padded_start_times = np.full(max_sessions, -1.0)
        padded_start_times[:min(num_sessions, max_sessions)] = start_times[:max_sessions]

        target = {
            'num_sessions': num_sessions,
            'start_times': padded_start_times
        }

        features_list.append(features)
        targets_list.append(target)

    features_array = np.array(features_list)
    targets_dict = {
        'num_sessions': np.array([t['num_sessions'] for t in targets_list]),
        'start_times': np.array([t['start_times'] for t in targets_list])
    }

    dataset = {
        'features': features_array,
        'targets': targets_dict,
        'metadata': {
            'feature_names': feature_extractor.get_feature_names(),
            'num_samples': len(features_list),
            'max_sessions': 20
        }
    }

    print(f"Dataset created:")
    print(f"  Features shape: {features_array.shape}")
    print(f"  Targets (num_sessions) shape: {targets_dict['num_sessions'].shape}")
    print(f"  Targets (start_times) shape: {targets_dict['start_times'].shape}")

    # Save if path provided
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"  Saved to: {save_path}")

    return dataset


def prepare_temporal_training_data(pxchange_data_dir=None, output_dir=None, augmentation_factor=50):
    """
    Complete pipeline: extract patterns, augment, and create training dataset

    Args:
        pxchange_data_dir: Directory with PXChange raw data (default: from config)
        output_dir: Directory to save outputs (default: from config)
        augmentation_factor: Augmentation factor

    Returns:
        dataset: Training dataset dictionary
    """
    if pxchange_data_dir is None:
        pxchange_data_dir = os.path.join(PXCHANGE_DIR, 'data')

    if output_dir is None:
        output_dir = TEMPORAL_DATA_DIR

    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("TEMPORAL TRAINING DATA PREPARATION PIPELINE")
    print("="*80)

    # Step 1: Extract patterns
    daily_summaries = extract_temporal_patterns_from_pxchange(pxchange_data_dir)

    # Save real summaries
    summaries_path = os.path.join(output_dir, 'real_daily_summaries.csv')
    daily_summaries.to_csv(summaries_path, index=False)
    print(f"\nSaved real summaries to: {summaries_path}")

    # Step 2: Augment
    augmented_summaries = augment_temporal_data(
        daily_summaries,
        augmentation_factor=augmentation_factor
    )

    # Save augmented summaries
    augmented_path = os.path.join(output_dir, 'augmented_daily_summaries.csv')
    augmented_summaries.to_csv(augmented_path, index=False)
    print(f"Saved augmented summaries to: {augmented_path}")

    # Step 3: Create training dataset
    dataset_path = os.path.join(output_dir, 'temporal_training_dataset.pkl')
    dataset = create_temporal_training_dataset(augmented_summaries, dataset_path)

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)

    return dataset


def load_temporal_training_dataset(dataset_path=None):
    """Load pre-processed temporal training dataset"""
    if dataset_path is None:
        dataset_path = os.path.join(TEMPORAL_DATA_DIR, 'temporal_training_dataset.pkl')

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    print(f"Loaded temporal training dataset from: {dataset_path}")
    print(f"  Features shape: {dataset['features'].shape}")
    print(f"  Num samples: {dataset['metadata']['num_samples']}")

    return dataset


if __name__ == "__main__":
    # Run the complete pipeline
    dataset = prepare_temporal_training_data(
        augmentation_factor=TEMPORAL_TRAINING_CONFIG['augmentation_factor']
    )

    print("\nDataset summary:")
    print(f"  Total samples: {dataset['metadata']['num_samples']}")
    print(f"  Feature dimensions: {len(dataset['metadata']['feature_names'])}")
    print(f"  Session count range: {dataset['targets']['num_sessions'].min()}-{dataset['targets']['num_sessions'].max()}")
