"""
SeqofSeq-specific Dataset Class
Handles SeqofSeq's label-encoded conditioning features + coil features
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# Import from SeqofSeq config using importlib to avoid conflicts
import sys
import os
import importlib.util

seqofseq_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'SeqofSeq_Pipeline')
config_path = os.path.join(seqofseq_dir, 'config.py')

# Load SeqofSeq config directly
spec = importlib.util.spec_from_file_location("seqofseq_config", config_path)
seqofseq_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seqofseq_config)

MAX_SEQ_LEN = seqofseq_config.MAX_SEQ_LEN
PAD_TOKEN_ID = seqofseq_config.PAD_TOKEN_ID
CONDITIONING_FEATURES = seqofseq_config.CONDITIONING_FEATURES


class SeqofSeqDataset(Dataset):
    """
    Dataset for SeqofSeq MRI sequences with label-encoded conditioning + coil features.

    Returns batches with:
        conditioning: Label-encoded + coil features [batch_size, conditioning_dim]
        sequence_tokens: Symbolic sequence (sourceID) [batch_size, seq_len]
        sequence_features: Minimal sequence features [batch_size, seq_len, 2]
        step_durations: Target durations [batch_size, seq_len]
        mask: Valid position mask [batch_size, seq_len]
    """

    def __init__(self, dataframe, metadata, conditioning_scaler=None, fit_scaler=False):
        """
        Args:
            dataframe: Preprocessed DataFrame with sequences
            metadata: Metadata dict with coil_cols and feature info
            conditioning_scaler: StandardScaler for conditioning features
            fit_scaler: Whether to fit the scaler on this data
        """
        self.df = dataframe
        self.seq_orders = dataframe['SeqOrder'].unique()
        self.metadata = metadata

        # Use only core CONDITIONING_FEATURES (skip coil columns for simplicity)
        self.all_conditioning_features = list(CONDITIONING_FEATURES)

        # Fit or use conditioning scaler
        if fit_scaler:
            self.conditioning_scaler = StandardScaler()
            # Extract conditioning for each sequence
            conditioning_data = []
            for seq_order in self.seq_orders:
                seq_data = dataframe[dataframe['SeqOrder'] == seq_order].iloc[0]
                cond_values = []
                for feat in self.all_conditioning_features:
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
        return len(self.seq_orders)

    def __getitem__(self, idx):
        seq_order = self.seq_orders[idx]
        seq_data = self.df[self.df['SeqOrder'] == seq_order].sort_values('Step')

        # Extract conditioning features from first row
        first_row = seq_data.iloc[0]
        conditioning = []
        for feat in self.all_conditioning_features:
            val = first_row.get(feat, 0)
            conditioning.append(float(val) if val is not None else 0.0)
        conditioning = np.array(conditioning, dtype=np.float32)

        # Replace any NaN values with 0 before scaling
        conditioning = np.nan_to_num(conditioning, nan=0.0)

        if self.conditioning_scaler is not None:
            conditioning = self.conditioning_scaler.transform(conditioning.reshape(1, -1))[0]
            # Replace any NaN from scaling with 0
            conditioning = np.nan_to_num(conditioning, nan=0.0)

        # Extract sequence tokens
        sequence_tokens = seq_data['sourceID'].values.astype(np.int64)

        # Extract minimal sequence features (Position/Direction set to 0)
        # SeqofSeq doesn't have these, so we use placeholder zeros
        sequence_features = np.zeros((len(seq_data), 2), dtype=np.float32)
        if 'Position_encoded' in seq_data.columns:
            sequence_features[:, 0] = seq_data['Position_encoded'].values
        if 'Direction_encoded' in seq_data.columns:
            sequence_features[:, 1] = seq_data['Direction_encoded'].values

        # Extract step durations
        if 'step_duration' in seq_data.columns:
            step_durations = seq_data['step_duration'].values.astype(np.float32)
        else:
            # Calculate from timediff if not present
            step_durations = np.zeros(len(seq_data), dtype=np.float32)
            if 'timediff' in seq_data.columns:
                timediff = seq_data['timediff'].values
                step_durations[1:] = np.diff(timediff)
                step_durations = np.clip(step_durations, 0, None)

        # Pad sequences to MAX_SEQ_LEN
        actual_len = min(len(seq_data), MAX_SEQ_LEN)

        # Create padded arrays
        padded_tokens = np.full(MAX_SEQ_LEN, PAD_TOKEN_ID, dtype=np.int64)
        padded_features = np.zeros((MAX_SEQ_LEN, 2), dtype=np.float32)
        padded_durations = np.zeros(MAX_SEQ_LEN, dtype=np.float32)
        mask = np.zeros(MAX_SEQ_LEN, dtype=np.bool_)

        # Fill with actual data
        padded_tokens[:actual_len] = sequence_tokens[:actual_len]
        padded_features[:actual_len] = sequence_features[:actual_len]
        padded_durations[:actual_len] = step_durations[:actual_len]
        mask[:actual_len] = True

        return {
            'conditioning': torch.from_numpy(conditioning.astype(np.float32)),
            'sequence_tokens': torch.from_numpy(padded_tokens),
            'sequence_features': torch.from_numpy(padded_features),
            'step_durations': torch.from_numpy(padded_durations),
            'mask': torch.from_numpy(mask),
            'seq_length': torch.tensor(actual_len, dtype=torch.long)
        }
