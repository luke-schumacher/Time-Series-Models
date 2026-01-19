"""
Unified Schedule Pipeline Configuration
Combines PXChange, SeqofSeq, and Temporal models for complete daily MR machine schedule prediction
"""
import os
import sys

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')

# Links to existing projects
PXCHANGE_DIR = os.path.join(BASE_DIR, '..', 'PXChange_Refactored')
SEQOFSEQ_DIR = os.path.join(BASE_DIR, '..', 'SeqofSeq_Pipeline')

# Add existing projects to Python path for imports
sys.path.insert(0, PXCHANGE_DIR)
sys.path.insert(0, SEQOFSEQ_DIR)

# Output directories
TEMPORAL_DATA_DIR = os.path.join(DATA_DIR, 'temporal_training_data')
GENERATED_SCHEDULES_DIR = os.path.join(OUTPUT_DIR, 'generated_schedules')
EVENT_TIMELINES_DIR = os.path.join(OUTPUT_DIR, 'event_timelines')
PATIENT_SESSIONS_DIR = os.path.join(OUTPUT_DIR, 'patient_sessions')

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, MODEL_SAVE_DIR, TEMPORAL_DATA_DIR,
                  GENERATED_SCHEDULES_DIR, EVENT_TIMELINES_DIR, PATIENT_SESSIONS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# PAUSE HANDLING CONFIGURATION (Updated for Pseudo-Patient Architecture)
# ============================================================================
# Sequence segmentation replaces pause token injection
# Pauses are now represented as pseudo-patient entities (separate training examples)

PAUSE_DETECTION_THRESHOLD_MINUTES = 5  # Gap to trigger sequence segmentation
PAUSE_DURATION_MIN_SECONDS = 60       # Minimum pause duration
PAUSE_DURATION_MAX_SECONDS = 600      # Maximum pause duration (for capping)

# Pseudo-patient entity types
# These represent different machine idle states
ENTITY_TYPES = {
    'REAL_PATIENT': 0,              # Actual patient examination/exchange sequence
    'PSEUDO_PATIENT_PAUSE': 1,      # Inter-session pause (gap between patients)
    'PSEUDO_PATIENT_START': 2,      # Start-of-day idle state (future use)
    'PSEUDO_PATIENT_END': 3         # End-of-day idle state (future use)
}

# Special encoding for anatomical features when machine is idle
# Used for BodyPart, BodyGroup when entity_type is PSEUDO_PATIENT_*
IDLE_STATE_ENCODING = -1

# ============================================================================
# TEMPORAL MODEL CONFIGURATION
# ============================================================================
TEMPORAL_MODEL_CONFIG = {
    'temporal_feature_dim': 12,      # Number of input temporal features
    'd_model': 128,                  # Model embedding dimension
    'nhead': 4,                      # Number of attention heads
    'num_encoder_layers': 4,         # Transformer encoder layers
    'dim_feedforward': 512,          # FFN hidden dimension
    'dropout': 0.1,
    'max_sessions': 20,              # Maximum sessions per day
    'max_session_gap_hours': 4,      # Maximum gap between sessions (hours)

    # Session count prediction (Poisson distribution)
    'session_count_min': 1,
    'session_count_max': 25,

    # Session timing prediction (Mixture of Gaussians)
    'num_gaussian_components': 3,    # Morning, afternoon, evening modes
    'time_encoding_dim': 32          # Dimension for time embeddings
}

TEMPORAL_TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'early_stopping_patience': 15,
    'validation_split': 0.2,
    'augmentation_factor': 50,       # How many synthetic samples per real sample
    'scheduler_patience': 5,         # For ReduceLROnPlateau
    'scheduler_factor': 0.5
}

# Temporal features to extract/use
TEMPORAL_FEATURES = [
    'day_of_year_sin',
    'day_of_year_cos',
    'day_of_week_sin',
    'day_of_week_cos',
    'day_of_month',
    'week_of_year',
    'is_weekend',
    'is_morning',
    'is_afternoon',
    'is_evening',
    'machine_id_encoded',
    'typical_daily_load'
]

# ============================================================================
# GENERATION CONFIGURATION
# ============================================================================
GENERATION_CONFIG = {
    'default_day_of_week': 3,        # Wednesday
    'default_machine_id': 141049,
    'seed': 42,
    'temperature_temporal': 0.8,     # Temperature for temporal predictions
    'temperature_sequences': 1.0,    # Temperature for sequence generation
    'enable_validation': True,       # Enable constraint validation
    'enable_adjustment': True,       # Enable automatic adjustments for violations
    'max_retries': 3,                # Max retries if generation fails
    'session_buffer_seconds': 60     # Minimum gap between sessions
}

# ============================================================================
# OUTPUT FORMATTING CONFIGURATION
# ============================================================================
OUTPUT_CONFIG = {
    'generate_event_timeline': True,
    'generate_patient_sessions': True,
    'output_format': 'csv',          # 'csv', 'json', 'parquet'
    'timestamp_format': 'both',      # 'datetime', 'seconds', 'both'
    'include_metadata': True,
    'decimal_places': 2
}

# Event timeline CSV columns
EVENT_TIMELINE_COLUMNS = [
    'event_id',
    'timestamp',
    'datetime',
    'event_type',
    'session_id',
    'patient_id',
    'sourceID',
    'scan_sequence',
    'body_part',
    'duration',
    'cumulative_time'
]

# Patient sessions CSV columns
PATIENT_SESSION_COLUMNS = [
    'session_id',
    'patient_id',
    'session_start_time',
    'session_end_time',
    'session_duration',
    'num_events',
    'num_scans',
    'num_pauses',
    'body_parts',
    'scan_sequences'
]

# ============================================================================
# VALIDATION THRESHOLDS
# ============================================================================
VALIDATION_CONFIG = {
    # Daily schedule constraints
    'min_total_duration_hours': 6,
    'max_total_duration_hours': 14,
    'typical_start_time_seconds': 25200,  # 7:00 AM
    'typical_end_time_seconds': 68400,    # 7:00 PM

    # Session constraints
    'min_sessions': 3,
    'max_sessions': 25,
    'min_session_duration_minutes': 10,
    'max_session_duration_minutes': 90,

    # Gap constraints
    'min_inter_session_gap_minutes': 2,
    'max_inter_session_gap_minutes': 120,

    # Timing constraints
    'allow_evening_sessions': False,      # Sessions must end before 8 PM
    'require_lunch_break': True,          # Should have a longer gap around noon
    'lunch_break_start_hour': 12,
    'lunch_break_min_duration_minutes': 30,

    # Event constraints
    'max_consecutive_pauses': 2,
    'min_scan_duration_seconds': 10,
    'max_scan_duration_seconds': 500
}

# ============================================================================
# EVALUATION METRICS CONFIGURATION
# ============================================================================
METRICS_CONFIG = {
    'compute_schedule_quality': True,
    'compute_temporal_accuracy': True,
    'compute_constraint_satisfaction': True,

    'quality_metrics': [
        'total_duration_hours',
        'num_sessions',
        'avg_session_duration_minutes',
        'avg_inter_session_gap_minutes',
        'num_scans',
        'num_pauses',
        'pause_ratio'
    ],

    'temporal_metrics': [
        'mae_session_count',
        'kl_divergence_timing',
        'timing_distribution_accuracy'
    ],

    'constraint_metrics': [
        'duration_violations',
        'gap_violations',
        'overlap_violations',
        'sequencing_violations',
        'constraint_satisfaction_rate'
    ]
}

# ============================================================================
# MODEL PATHS
# ============================================================================
MODEL_PATHS = {
    'temporal': {
        'model': os.path.join(MODEL_SAVE_DIR, 'temporal_schedule_model', 'temporal_model_best.pth'),
        'config': os.path.join(MODEL_SAVE_DIR, 'temporal_schedule_model', 'config.pkl')
    },
    'pxchange': {
        'model_dir': os.path.join(MODEL_SAVE_DIR, 'pxchange_models'), # Directory for saving fine-tuned PXChange models
        'sequence': os.path.join(MODEL_SAVE_DIR, 'pxchange_models', 'sequence_model_best.pth'),
        'duration': os.path.join(MODEL_SAVE_DIR, 'pxchange_models', 'duration_model_best.pth'),
        'metadata': os.path.join(PXCHANGE_DIR, 'data', 'preprocessed', 'metadata.pkl')
    },
    'seqofseq': {
        'model_dir': os.path.join(MODEL_SAVE_DIR, 'seqofseq_models'), # Directory for saving fine-tuned SeqofSeq models
        'sequence': os.path.join(MODEL_SAVE_DIR, 'seqofseq_models', 'sequence_model_best.pth'),
        'duration': os.path.join(MODEL_SAVE_DIR, 'seqofseq_models', 'duration_model_best.pth'),
        'metadata': os.path.join(SEQOFSEQ_DIR, 'data', 'preprocessed', 'metadata.pkl')
    }
}

# ============================================================================
# DATA SOURCES
# ============================================================================
DATA_SOURCES = {
    'pxchange_raw': os.path.join(PXCHANGE_DIR, 'data'),
    'pxchange_preprocessed': os.path.join(PXCHANGE_DIR, 'data', 'preprocessed'),
    'seqofseq_raw': os.path.join(SEQOFSEQ_DIR, 'data'),
    'seqofseq_preprocessed': os.path.join(SEQOFSEQ_DIR, 'data', 'preprocessed')
}

# ============================================================================
# RANDOM SEEDS
# ============================================================================
RANDOM_SEED = 42

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
USE_GPU = True  # Set to False to force CPU usage

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'log_file': os.path.join(BASE_DIR, 'pipeline.log'),
    'log_to_console': True,
    'log_to_file': True
}

# ============================================================================
# DEBUGGING/DEVELOPMENT FLAGS
# ============================================================================
DEBUG_CONFIG = {
    'save_intermediate_outputs': True,
    'validate_each_session': True,
    'print_generation_progress': True,
    'save_generation_stats': True
}
