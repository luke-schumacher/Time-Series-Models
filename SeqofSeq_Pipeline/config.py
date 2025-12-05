"""
Configuration file for the SeqofSeq Pipeline
MRI Scan Sequence and Duration Prediction System
"""
import os

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
VISUALIZATION_DIR = os.path.join(BASE_DIR, 'visualizations')

# Create directories if they don't exist
for directory in [OUTPUT_DIR, MODEL_SAVE_DIR, VISUALIZATION_DIR, DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Sequence configuration
MAX_SEQ_LEN = 64  # Maximum sequence length for padding

# Conditioning features (patient/scan context)
CONDITIONING_FEATURES = [
    'BodyPart_encoded',      # Which body part is being scanned
    'SystemType_encoded',    # Type of MRI system
    'Country_encoded',       # Geographic location
    'Group_encoded'          # Scan group/category
]

# Coil features (boolean indicators for which coils are connected)
# These will be extracted from columns starting with '#0_' or '#1_'
COIL_FEATURES_PREFIX = ['#0_', '#1_']

# Target features to predict
TARGET_FEATURES = {
    'sequence': 'Sequence',      # The MRI sequence type
    'protocol': 'Protocol',      # Specific protocol name
    'duration': 'duration'       # Duration in seconds
}

# ============================================================================
# VOCABULARY CONFIGURATION
# ============================================================================

# Will be built from the data, but reserve special tokens
SPECIAL_TOKENS = {
    'PAD': 0,
    'START': 1,
    'END': 2,
    'UNK': 3,
    'PAUSE': 4
}

# Token ID constants for easy access
PAD_TOKEN_ID = 0
START_TOKEN_ID = 1
END_TOKEN_ID = 2
UNK_TOKEN_ID = 3
PAUSE_TOKEN_ID = 4
VOCAB_SIZE = None  # Will be set after preprocessing

# Vocabulary will be built from unique Sequences in the data
# Estimated: ~30 unique sequences + special tokens = ~35 vocab size

# ============================================================================
# MODEL ARCHITECTURE - CONDITIONAL SEQUENCE GENERATOR
# ============================================================================

SEQUENCE_MODEL_CONFIG = {
    'vocab_size': None,          # Will be set after data analysis
    'd_model': 256,              # Model dimension
    'nhead': 8,                  # Number of attention heads
    'num_encoder_layers': 6,     # Encoder depth
    'num_decoder_layers': 6,     # Decoder depth
    'dim_feedforward': 1024,     # FFN dimension
    'dropout': 0.15,
    'max_seq_len': MAX_SEQ_LEN,
    'conditioning_dim': None     # Will be set after feature extraction
}

# Training configuration for sequence model
SEQUENCE_TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.0001,
    'warmup_steps': 2000,
    'label_smoothing': 0.1,
    'gradient_clip': 1.0,
    'early_stopping_patience': 15,
    'validation_split': 0.2,
    'save_best_only': True
}

# Sampling configuration for sequence generation
SEQUENCE_SAMPLING_CONFIG = {
    'temperature': 1.0,      # Sampling temperature (higher = more random)
    'top_k': 10,             # Top-k sampling
    'top_p': 0.9,            # Nucleus sampling
    'max_length': MAX_SEQ_LEN
}

# ============================================================================
# MODEL ARCHITECTURE - CONDITIONAL DURATION PREDICTOR
# ============================================================================

DURATION_MODEL_CONFIG = {
    'd_model': 256,              # Model dimension
    'nhead': 8,                  # Number of attention heads
    'num_encoder_layers': 6,     # Number of encoder layers
    'num_cross_attention_layers': 4,  # Cross-attention layers
    'dim_feedforward': 1024,     # FFN dimension
    'dropout': 0.15,
    'max_seq_len': MAX_SEQ_LEN,
    'conditioning_dim': None,    # Will be set after feature extraction
    'sequence_feature_dim': None, # Will be set after vocab building
    'output_heads': 2            # μ and σ predictions (for Gamma distribution)
}

# Training configuration for duration model
DURATION_TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.0001,
    'warmup_steps': 2000,
    'gradient_clip': 1.0,
    'early_stopping_patience': 15,
    'validation_split': 0.2,
    'min_sigma': 1.0,            # Minimum standard deviation for duration
    'save_best_only': True
}

# Sampling configuration for duration generation
DURATION_SAMPLING_CONFIG = {
    'distribution': 'gamma',     # 'gamma', 'normal', or 'lognormal'
    'num_samples': 1,            # Number of samples to draw per step
    'clip_negative': True,       # Clip negative values to 0
    'min_duration': 0.0,         # Minimum allowed duration
    'max_duration': 2000.0       # Maximum allowed duration (in seconds)
}

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

# For sequence model: cross-entropy with label smoothing
SEQUENCE_LOSS = 'cross_entropy'

# For duration model: negative log-likelihood of Gamma distribution
DURATION_LOSS = 'gamma_nll'  # Options: 'gamma_nll', 'gaussian_nll', 'mse'

# ============================================================================
# EVALUATION METRICS
# ============================================================================

METRICS = {
    'sequence': ['accuracy', 'perplexity', 'top_k_accuracy'],
    'duration': ['mae', 'rmse', 'mape', 'r2']
}

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

DATA_CONFIG = {
    'min_sequence_length': 3,     # Minimum number of scans in a sequence
    'max_sequence_length': MAX_SEQ_LEN,
    'group_by_patient': True,     # Group scans by PatientID
    'sort_by_time': True,         # Sort by startTime
    'remove_incomplete': True,    # Remove sequences with missing data
    'duration_outlier_std': 3.0   # Remove durations beyond N std devs
}

# ============================================================================
# RANDOM SEEDS
# ============================================================================

RANDOM_SEED = 42

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

USE_GPU = True  # Set to False to force CPU usage
