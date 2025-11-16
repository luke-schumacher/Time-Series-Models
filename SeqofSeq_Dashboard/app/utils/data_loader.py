"""
Data loading and processing utilities for the SeqofSeq Dashboard
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List


class DataLoader:
    """Handle loading and caching of generated sequence data"""

    def __init__(self, data_path: str):
        """
        Initialize the data loader

        Args:
            data_path: Path to the generated sequences CSV file
        """
        self.data_path = Path(data_path)
        self._data_cache: Optional[pd.DataFrame] = None

    def load_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load the generated sequences data

        Args:
            force_reload: Force reload from disk even if cached

        Returns:
            DataFrame with loaded data
        """
        if self._data_cache is None or force_reload:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")

            self._data_cache = pd.read_csv(self.data_path)

        return self._data_cache.copy()

    def get_summary_stats(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate summary statistics from the data

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with summary statistics
        """
        return {
            'total_samples': df['sample_id'].nunique(),
            'total_sequences': len(df),
            'unique_patients': df['patient_id'].nunique(),
            'unique_sequence_types': df['sequence_name'].nunique(),
            'avg_sequence_length': df.groupby('sample_id').size().mean(),
            'total_duration': df['predicted_duration'].sum(),
            'avg_duration': df['predicted_duration'].mean(),
            'min_duration': df['predicted_duration'].min(),
            'max_duration': df['predicted_duration'].max(),
        }

    def get_sequence_type_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistics per sequence type

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with per-sequence-type statistics
        """
        stats = df.groupby('sequence_name').agg({
            'predicted_duration': ['count', 'mean', 'std', 'min', 'max'],
            'sample_id': 'nunique'
        }).round(2)

        stats.columns = ['count', 'mean_duration', 'std_duration',
                        'min_duration', 'max_duration', 'num_samples']
        stats = stats.reset_index()
        stats = stats.sort_values('count', ascending=False)

        return stats

    def get_patient_sequences(self, df: pd.DataFrame, patient_id: str) -> pd.DataFrame:
        """
        Get all sequences for a specific patient

        Args:
            df: Input DataFrame
            patient_id: Patient identifier

        Returns:
            Filtered DataFrame for the patient
        """
        return df[df['patient_id'] == patient_id].sort_values(['sample_id', 'step'])

    def get_sample_sequences(self, df: pd.DataFrame, sample_id: int) -> pd.DataFrame:
        """
        Get sequence for a specific sample

        Args:
            df: Input DataFrame
            sample_id: Sample identifier

        Returns:
            Filtered DataFrame for the sample
        """
        return df[df['sample_id'] == sample_id].sort_values('step')

    def calculate_cumulative_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cumulative duration for sequences

        Args:
            df: Input DataFrame (should be for a single sample/sequence)

        Returns:
            DataFrame with cumulative duration column added
        """
        df = df.copy()
        df['cumulative_duration'] = df['predicted_duration'].cumsum()
        return df

    def get_sequence_transitions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sequence transitions (which sequences follow which)

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with transition counts
        """
        transitions = []

        for sample_id in df['sample_id'].unique():
            sample_df = df[df['sample_id'] == sample_id].sort_values('step')
            sequences = sample_df['sequence_name'].tolist()

            for i in range(len(sequences) - 1):
                transitions.append({
                    'from_sequence': sequences[i],
                    'to_sequence': sequences[i + 1],
                })

        if transitions:
            transition_df = pd.DataFrame(transitions)
            transition_counts = transition_df.groupby(
                ['from_sequence', 'to_sequence']
            ).size().reset_index(name='count')
            return transition_counts.sort_values('count', ascending=False)
        else:
            return pd.DataFrame(columns=['from_sequence', 'to_sequence', 'count'])

    def get_duration_distribution(self, df: pd.DataFrame, bins: int = 30) -> Dict:
        """
        Calculate duration distribution for histograms

        Args:
            df: Input DataFrame
            bins: Number of histogram bins

        Returns:
            Dictionary with histogram data
        """
        counts, bin_edges = np.histogram(df['predicted_duration'], bins=bins)

        return {
            'counts': counts.tolist(),
            'bin_edges': bin_edges.tolist(),
            'mean': df['predicted_duration'].mean(),
            'median': df['predicted_duration'].median(),
            'std': df['predicted_duration'].std(),
        }

    def get_sequence_length_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate distribution of sequence lengths

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with sequence length counts
        """
        lengths = df.groupby('sample_id').size().reset_index(name='sequence_length')
        length_dist = lengths.groupby('sequence_length').size().reset_index(name='count')
        return length_dist.sort_values('sequence_length')
