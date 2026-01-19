"""
Temporal Feature Engineering
Transforms datetime information into features for the temporal schedule model
"""
import numpy as np
import pandas as pd
from datetime import datetime, time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import TEMPORAL_FEATURES


def cyclical_encode(value, max_value):
    """
    Encode a cyclical feature using sine and cosine

    Args:
        value: Current value (e.g., day of year)
        max_value: Maximum value in the cycle (e.g., 365 for days)

    Returns:
        sin_encoded, cos_encoded: Tuple of sine and cosine encodings
    """
    angle = 2 * np.pi * value / max_value
    return np.sin(angle), np.cos(angle)


def extract_temporal_features(date=None, datetime_obj=None, machine_id=None, typical_load=None):
    """
    Extract temporal features from a date/datetime object

    Args:
        date: datetime.date object
        datetime_obj: datetime.datetime object (if provided, overrides date)
        machine_id: Machine/system identifier
        typical_load: Typical daily load (average sessions per day for this day type)

    Returns:
        features_dict: Dictionary of temporal features
    """
    if datetime_obj is not None:
        dt = datetime_obj
    elif date is not None:
        dt = datetime.combine(date, time(7, 0))  # Default to 7 AM
    else:
        dt = datetime.now()

    # Day of year (1-365/366)
    day_of_year = dt.timetuple().tm_yday
    max_days = 366 if (dt.year % 4 == 0 and dt.year % 100 != 0) or (dt.year % 400 == 0) else 365
    day_of_year_sin, day_of_year_cos = cyclical_encode(day_of_year, max_days)

    # Day of week (0=Monday, 6=Sunday)
    day_of_week = dt.weekday()
    day_of_week_sin, day_of_week_cos = cyclical_encode(day_of_week, 7)

    # Day of month (1-31)
    day_of_month = dt.day

    # Week of year (1-53)
    week_of_year = dt.isocalendar()[1]

    # Is weekend
    is_weekend = 1 if day_of_week >= 5 else 0

    # Time of day (if datetime provided)
    hour = dt.hour
    is_morning = 1 if 6 <= hour < 12 else 0
    is_afternoon = 1 if 12 <= hour < 18 else 0
    is_evening = 1 if 18 <= hour < 22 else 0

    # Machine ID (encoded)
    machine_id_encoded = machine_id if machine_id is not None else 0

    # Typical daily load (default to average if not provided)
    typical_daily_load = typical_load if typical_load is not None else 12.0

    features = {
        'day_of_year_sin': day_of_year_sin,
        'day_of_year_cos': day_of_year_cos,
        'day_of_week_sin': day_of_week_sin,
        'day_of_week_cos': day_of_week_cos,
        'day_of_month': day_of_month,
        'week_of_year': week_of_year,
        'is_weekend': is_weekend,
        'is_morning': is_morning,
        'is_afternoon': is_afternoon,
        'is_evening': is_evening,
        'machine_id_encoded': machine_id_encoded,
        'typical_daily_load': typical_daily_load
    }

    return features


def features_to_array(features_dict, feature_names=None):
    """
    Convert features dictionary to numpy array

    Args:
        features_dict: Dictionary of features
        feature_names: List of feature names in desired order (default: from config)

    Returns:
        features_array: Numpy array of shape [num_features]
    """
    if feature_names is None:
        feature_names = TEMPORAL_FEATURES

    return np.array([features_dict[name] for name in feature_names])


def batch_extract_features(dates, machine_ids=None, typical_loads=None):
    """
    Extract features for multiple dates

    Args:
        dates: List of date or datetime objects
        machine_ids: List of machine IDs (optional, same length as dates)
        typical_loads: List of typical loads (optional, same length as dates)

    Returns:
        features_array: Numpy array of shape [num_dates, num_features]
    """
    if machine_ids is None:
        machine_ids = [None] * len(dates)

    if typical_loads is None:
        typical_loads = [None] * len(dates)

    features_list = []

    for date, machine_id, typical_load in zip(dates, machine_ids, typical_loads):
        features = extract_temporal_features(
            datetime_obj=date if isinstance(date, datetime) else None,
            date=date if not isinstance(date, datetime) else None,
            machine_id=machine_id,
            typical_load=typical_load
        )
        features_list.append(features_to_array(features))

    return np.array(features_list)


def encode_session_start_time(hour, minute=0, second=0):
    """
    Encode session start time into cyclical features

    Args:
        hour: Hour of day (0-23)
        minute: Minute (0-59)
        second: Second (0-59)

    Returns:
        seconds_from_midnight, hour_sin, hour_cos: Encoded time features
    """
    seconds_from_midnight = hour * 3600 + minute * 60 + second

    # Cyclical encoding for hour
    hour_sin, hour_cos = cyclical_encode(hour, 24)

    return seconds_from_midnight, hour_sin, hour_cos


def decode_session_start_time(seconds_from_midnight):
    """
    Decode seconds from midnight to hour:minute:second

    Args:
        seconds_from_midnight: Seconds since midnight

    Returns:
        hour, minute, second: Time components
    """
    total_seconds = int(seconds_from_midnight)
    hour = total_seconds // 3600
    minute = (total_seconds % 3600) // 60
    second = total_seconds % 60

    return hour, minute, second


def calculate_typical_load(daily_summaries_df, day_of_week=None, machine_id=None):
    """
    Calculate typical daily load (average sessions per day)

    Args:
        daily_summaries_df: DataFrame with 'day_of_week', 'machine_id', 'num_sessions' columns
        day_of_week: Specific day of week to calculate for (optional)
        machine_id: Specific machine to calculate for (optional)

    Returns:
        typical_load: Average number of sessions
    """
    filtered = daily_summaries_df.copy()

    if day_of_week is not None:
        filtered = filtered[filtered['day_of_week'] == day_of_week]

    if machine_id is not None:
        filtered = filtered[filtered['machine_id'] == machine_id]

    if len(filtered) == 0:
        return 12.0  # Default

    return filtered['num_sessions'].mean()


class TemporalFeatureExtractor:
    """
    Class for extracting and managing temporal features
    """

    def __init__(self, feature_names=None):
        """
        Initialize feature extractor

        Args:
            feature_names: List of feature names (default: from config)
        """
        self.feature_names = feature_names if feature_names is not None else TEMPORAL_FEATURES
        self.num_features = len(self.feature_names)

    def extract(self, date=None, datetime_obj=None, machine_id=None, typical_load=None):
        """Extract features and return as array"""
        features = extract_temporal_features(date, datetime_obj, machine_id, typical_load)
        return features_to_array(features, self.feature_names)

    def batch_extract(self, dates, machine_ids=None, typical_loads=None):
        """Extract features for multiple dates"""
        return batch_extract_features(dates, machine_ids, typical_loads)

    def get_feature_names(self):
        """Get list of feature names"""
        return self.feature_names.copy()

    def get_num_features(self):
        """Get number of features"""
        return self.num_features


def create_feature_dataframe(dates, machine_ids=None, typical_loads=None):
    """
    Create DataFrame with temporal features

    Args:
        dates: List of dates
        machine_ids: List of machine IDs
        typical_loads: List of typical loads

    Returns:
        features_df: DataFrame with all temporal features
    """
    extractor = TemporalFeatureExtractor()
    features_array = extractor.batch_extract(dates, machine_ids, typical_loads)

    features_df = pd.DataFrame(features_array, columns=extractor.get_feature_names())

    # Add original date for reference
    features_df['date'] = dates

    return features_df


# =============================================================================
# Example usage and testing
# =============================================================================

def test_temporal_features():
    """Test temporal feature extraction"""
    print("Testing Temporal Feature Extraction\n")

    # Test single date
    test_date = datetime(2024, 6, 15, 14, 30)  # June 15, 2024, 2:30 PM (Saturday)

    features = extract_temporal_features(datetime_obj=test_date, machine_id=141049, typical_load=15)

    print("Date:", test_date)
    print("Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")

    # Test cyclical encoding
    print("\nCyclical Encoding Test:")
    for day in [0, 91, 182, 273, 364]:  # Different days of year
        sin_val, cos_val = cyclical_encode(day, 365)
        print(f"  Day {day}: sin={sin_val:.3f}, cos={cos_val:.3f}")

    # Test batch extraction
    print("\nBatch Extraction Test:")
    test_dates = [
        datetime(2024, 1, 15),  # Monday, Winter
        datetime(2024, 4, 15),  # Monday, Spring
        datetime(2024, 7, 15),  # Monday, Summer
        datetime(2024, 10, 15), # Tuesday, Fall
    ]

    features_array = batch_extract_features(test_dates, machine_ids=[141049]*4)
    print(f"  Shape: {features_array.shape}")
    print(f"  Sample features (first date): {features_array[0]}")

    # Test feature extractor class
    print("\nFeature Extractor Class Test:")
    extractor = TemporalFeatureExtractor()
    print(f"  Number of features: {extractor.get_num_features()}")
    print(f"  Feature names: {extractor.get_feature_names()}")

    # Test DataFrame creation
    print("\nDataFrame Creation Test:")
    df = create_feature_dataframe(test_dates, machine_ids=[141049]*4)
    print(df.head())

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    test_temporal_features()
