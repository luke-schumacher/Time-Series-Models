"""
Generate Complete Daily MRI Schedule
Uses all 5 trained models to generate realistic MRI facility schedules.
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle

# Add paths
unified_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
px_dir = os.path.join(unified_dir, '..', 'PXChange_Refactored')
seq_dir = os.path.join(unified_dir, '..', 'SeqofSeq_Pipeline')
parent_dir = os.path.dirname(unified_dir)  # Time-Series-Models directory

# Add parent directory so PXChange_Refactored and SeqofSeq_Pipeline can be imported
sys.path.insert(0, parent_dir)
sys.path.insert(0, unified_dir)

# Import unified config explicitly
import importlib.util
spec = importlib.util.spec_from_file_location("unified_config", os.path.join(unified_dir, "config.py"))
unified_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unified_config)
MODEL_PATHS = unified_config.MODEL_PATHS
RANDOM_SEED = unified_config.RANDOM_SEED

# Import temporal model
from datetime_model.temporal_schedule_model import TemporalScheduleModel
from datetime_model.temporal_features import extract_temporal_features

# Import PXChange models
from PXChange_Refactored.models.conditional_sequence_generator import ConditionalSequenceGenerator as PXChangeSequenceModel
from PXChange_Refactored.models.conditional_counts_generator import ConditionalCountsGenerator as PXChangeDurationModel
from PXChange_Refactored.config import CONDITIONING_FEATURES as PX_FEATURES, SOURCEID_VOCAB

# Import SeqofSeq models
from SeqofSeq_Pipeline.models.conditional_sequence_generator import ConditionalSequenceGenerator as SeqofSeqSequenceModel
from SeqofSeq_Pipeline.models.conditional_duration_predictor import ConditionalDurationPredictor as SeqofSeqDurationModel

# Special token IDs for each vocabulary
# PXChange uses different IDs than SeqofSeq!
PX_PAD_ID = SOURCEID_VOCAB['PAD']          # 0
PX_START_ID = SOURCEID_VOCAB['START']      # 11
PX_END_ID = SOURCEID_VOCAB['END']          # 14
PX_EXU_ID = SOURCEID_VOCAB.get('MRI_EXU_95', 2)  # 2 - Measurement trigger

# SeqofSeq uses standard IDs
SEQ_PAD_ID = 0
SEQ_START_ID = 1
SEQ_END_ID = 2

# Body group mapping (ID to name)
BODYGROUP_NAMES = ['HEAD', 'NECK', 'CHEST', 'ABDOMEN', 'PELVIS',
                   'SPINE', 'ARM', 'LEG', 'HAND', 'FOOT', 'UNKNOWN']


def load_patient_data(csv_path):
    """
    Load patient data from CSV file.

    Expected columns:
        - patient_id: Unique patient identifier (required)
        - age: Patient age in years (optional, default 50)
        - weight: Patient weight in kg (optional, default 75)
        - height: Patient height in cm (optional, default 170)
        - bodygroup_from: Source body region 0-10 or name (optional, default 0=HEAD)
        - bodygroup_to: Target body region 0-10 or name (optional, default=bodygroup_from)
        - ptab: Patient table position (optional, default 0)

    Returns:
        DataFrame with normalized patient data
    """
    df = pd.read_csv(csv_path)

    # Ensure required column exists
    if 'patient_id' not in df.columns:
        raise ValueError("Patient CSV must have 'patient_id' column")

    # Set defaults for missing columns
    defaults = {
        'age': 50,
        'weight': 75,
        'height': 170,
        'bodygroup_from': 0,
        'bodygroup_to': None,  # Will copy from bodygroup_from
        'ptab': 0
    }

    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val

    # Handle bodygroup_to default (same as bodygroup_from)
    if df['bodygroup_to'].isna().all() or (df['bodygroup_to'] == None).all():
        df['bodygroup_to'] = df['bodygroup_from']

    # Convert string body groups to integers if needed
    bodygroup_map = {name.upper(): idx for idx, name in enumerate(BODYGROUP_NAMES)}

    for col in ['bodygroup_from', 'bodygroup_to']:
        if df[col].dtype == 'object':
            df[col] = df[col].str.upper().map(bodygroup_map).fillna(10).astype(int)
        df[col] = df[col].fillna(0).astype(int)

    return df


def load_temporal_model(device='cpu'):
    """Load trained temporal model."""
    model_path = MODEL_PATHS['temporal']['model']
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Use config from unified_config
    model_config = unified_config.TEMPORAL_MODEL_CONFIG
    model = TemporalScheduleModel(
        temporal_feature_dim=model_config['temporal_feature_dim'],
        d_model=model_config['d_model'],
        nhead=model_config['nhead'],
        num_layers=model_config['num_encoder_layers'],  # Note: different key name
        dim_feedforward=model_config['dim_feedforward']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def load_pxchange_models(device='cpu'):
    """Load trained PXChange models."""
    # Load sequence model
    seq_path = MODEL_PATHS['pxchange']['sequence']
    seq_checkpoint = torch.load(seq_path, map_location=device, weights_only=False)
    seq_model = PXChangeSequenceModel(config=seq_checkpoint['model_config'])
    seq_model.load_state_dict(seq_checkpoint['model_state_dict'])
    seq_model = seq_model.to(device)
    seq_model.eval()

    # Load duration model
    dur_path = MODEL_PATHS['pxchange']['duration']
    dur_checkpoint = torch.load(dur_path, map_location=device, weights_only=False)
    dur_model = PXChangeDurationModel(config=dur_checkpoint['model_config'])
    dur_model.load_state_dict(dur_checkpoint['model_state_dict'])
    dur_model = dur_model.to(device)
    dur_model.eval()

    # Load scaler
    scaler = seq_checkpoint.get('scaler', None)

    return seq_model, dur_model, scaler


def load_seqofseq_models(device='cpu'):
    """Load trained SeqofSeq models."""
    # Load sequence model
    seq_path = MODEL_PATHS['seqofseq']['sequence']
    seq_checkpoint = torch.load(seq_path, map_location=device, weights_only=False)
    seq_model = SeqofSeqSequenceModel(config=seq_checkpoint['model_config'])
    seq_model.load_state_dict(seq_checkpoint['model_state_dict'])
    seq_model = seq_model.to(device)
    seq_model.eval()

    # Load duration model
    dur_path = MODEL_PATHS['seqofseq']['duration']
    dur_checkpoint = torch.load(dur_path, map_location=device, weights_only=False)
    dur_model = SeqofSeqDurationModel(config=dur_checkpoint['model_config'])
    dur_model.load_state_dict(dur_checkpoint['model_state_dict'])
    dur_model = dur_model.to(device)
    dur_model.eval()

    # Load scaler and metadata
    scaler = seq_checkpoint.get('scaler', None)
    metadata = seq_checkpoint.get('metadata', {})

    return seq_model, dur_model, scaler, metadata


def generate_pxchange_session(seq_model, dur_model, conditioning, scaler, device='cpu'):
    """Generate one PXChange session (exchange events)."""
    # Prepare conditioning
    if scaler is not None:
        cond_scaled = scaler.transform(conditioning.reshape(1, -1))
        cond_tensor = torch.from_numpy(cond_scaled).float().to(device)
    else:
        cond_tensor = torch.from_numpy(conditioning).float().unsqueeze(0).to(device)

    # Generate sequence
    with torch.no_grad():
        tokens = seq_model.generate(
            cond_tensor,
            max_length=64,
            temperature=1.0,
            top_k=10,  # Smaller top_k to avoid index out of range
            top_p=0.95
        )[0]  # [seq_len]

        # Find actual length
        tokens_np = tokens.cpu().numpy()
        # Find END token (2) or use full length
        end_idx = np.where(tokens_np == 2)[0]
        if len(end_idx) > 0:
            actual_len = end_idx[0] + 1
        else:
            actual_len = len(tokens_np)

        tokens_trimmed = tokens[:actual_len].unsqueeze(0)  # [1, actual_len]

        # Create dummy sequence features
        seq_features = torch.zeros(1, actual_len, 2, device=device)
        mask = torch.ones(1, actual_len, dtype=torch.bool, device=device)

        # Predict durations
        mu, sigma = dur_model(cond_tensor, tokens_trimmed, seq_features, mask)

        # Sample durations
        durations = dur_model.sample_counts(mu, sigma, num_samples=1).squeeze()

    return tokens_trimmed[0].cpu().numpy(), durations.cpu().numpy()


def generate_seqofseq_sequence(seq_model, dur_model, conditioning, scaler, device='cpu'):
    """Generate one SeqofSeq sequence (scan events)."""
    # Prepare conditioning
    if scaler is not None:
        cond_scaled = scaler.transform(conditioning.reshape(1, -1))
        cond_tensor = torch.from_numpy(cond_scaled).float().to(device)
    else:
        cond_tensor = torch.from_numpy(conditioning).float().unsqueeze(0).to(device)

    # Generate sequence
    with torch.no_grad():
        tokens = seq_model.generate(
            cond_tensor,
            max_length=64,
            temperature=1.0,
            top_k=10,  # Smaller top_k to avoid index out of range
            top_p=0.95
        )[0]  # [seq_len]

        # Find actual length
        tokens_np = tokens.cpu().numpy()
        # Find END token (2) or use full length
        end_idx = np.where(tokens_np == 2)[0]
        if len(end_idx) > 0:
            actual_len = end_idx[0] + 1
        else:
            actual_len = len(tokens_np)

        tokens_trimmed = tokens[:actual_len].unsqueeze(0)  # [1, actual_len]

        # Create dummy sequence features
        seq_features = torch.zeros(1, actual_len, 2, device=device)

        # Predict durations
        mu, sigma = dur_model(cond_tensor, tokens_trimmed, seq_features)

        # Sample durations
        alpha = (mu / sigma) ** 2
        beta = mu / (sigma ** 2)
        dist = torch.distributions.Gamma(alpha, beta)
        durations = dist.sample().squeeze()

    return tokens_trimmed[0].cpu().numpy(), durations.cpu().numpy()


def seconds_from_midnight(dt):
    """Convert datetime to seconds from midnight"""
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return int((dt - midnight).total_seconds())


def generate_daily_schedule(
    date_str="2026-01-10",
    machine_id=141049,
    patient_csv=None,
    num_sessions=None,
    output_dir=None,
    seed=None
):
    """
    Generate complete daily MRI schedule using all 5 models.

    Args:
        date_str: Date in "YYYY-MM-DD" format
        machine_id: MRI machine ID
        patient_csv: Path to CSV file with patient data (patient_id, age, weight,
                     height, bodygroup_from, bodygroup_to, ptab columns).
                     If provided, num_sessions is determined by the CSV row count.
        num_sessions: Number of patient sessions to generate (only used if
                      patient_csv is not provided). If None, uses temporal model prediction.
        output_dir: Directory to save outputs
        seed: Random seed

    Returns:
        schedule_df: Complete daily schedule DataFrame
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load patient data if provided
    patient_data = None
    if patient_csv is not None:
        patient_data = load_patient_data(patient_csv)
        num_sessions = len(patient_data)
        print(f"Loaded {num_sessions} patients from {patient_csv}")
    elif num_sessions is None:
        # Default to a reasonable number if not specified
        num_sessions = 15

    print("=" * 70)
    print("GENERATING DAILY MRI SCHEDULE")
    print("=" * 70)
    print(f"Date: {date_str}")
    print(f"Machine ID: {machine_id}")
    print(f"Sessions: {num_sessions}")
    print(f"Patient CSV: {patient_csv if patient_csv else 'Not provided (using generated data)'}")
    print(f"Device: {device}\n")

    # Parse date
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

    # Load models
    print("[1/5] Loading temporal model...")
    temporal_model, temporal_checkpoint = load_temporal_model(device)

    print("[2/5] Loading PXChange models...")
    px_seq_model, px_dur_model, px_scaler = load_pxchange_models(device)

    print("[3/5] Loading SeqofSeq models...")
    seq_seq_model, seq_dur_model, seq_scaler, seq_metadata = load_seqofseq_models(device)

    # Extract temporal features and predict schedule
    print(f"\n[4/5] Predicting daily structure...")
    temporal_features_dict = extract_temporal_features(datetime_obj=date_obj, machine_id=machine_id)

    # Convert to array in the correct order
    from datetime_model.temporal_features import features_to_array
    temporal_features = features_to_array(temporal_features_dict)
    temporal_tensor = torch.from_numpy(temporal_features).float().unsqueeze(0).to(device)

    with torch.no_grad():
        session_lambda, timing_params = temporal_model(temporal_tensor)
        start_times_tensor = temporal_model.sample_start_times(timing_params, num_sessions)

    # Use sampled start times directly from the temporal model
    # The model should output realistic times based on its training data
    start_times = start_times_tensor.cpu().numpy()  # [num_sessions]
    start_times = np.sort(start_times)  # Sort chronologically

    # Convert to datetime
    start_datetimes = [date_obj + timedelta(seconds=float(t)) for t in start_times]

    print(f"  Predicted session count: lambda={session_lambda.item():.2f}")
    print(f"  Generated {num_sessions} sessions")
    print(f"  First session: {start_datetimes[0].strftime('%H:%M:%S')}")
    print(f"  Last session: {start_datetimes[-1].strftime('%H:%M:%S')}\n")

    # Generate complete schedule
    print(f"[5/5] Generating events for {num_sessions} sessions...\n")

    all_events = []
    event_id = 0

    # Create reverse vocab for decoding
    id_to_sourceid = {v: k for k, v in SOURCEID_VOCAB.items()}

    # Load SeqofSeq vocab
    try:
        vocab_path = os.path.join(seq_dir, 'data', 'preprocessed', 'metadata.pkl')
        with open(vocab_path, 'rb') as f:
            seq_vocab_data = pickle.load(f)
        seq_vocab = seq_vocab_data.get('sequence_vocab', seq_vocab_data.get('vocab', {}))
        id_to_sequence = {v: k for k, v in seq_vocab.items()}
    except:
        id_to_sequence = {i: f"SCAN_{i}" for i in range(35)}

    for session_idx, session_start in enumerate(start_datetimes):
        # Get patient data from CSV or generate
        if patient_data is not None:
            patient_row = patient_data.iloc[session_idx]
            patient_id = str(patient_row['patient_id'])
            patient_age = int(patient_row['age'])
            patient_weight = int(patient_row['weight'])
            patient_height = int(patient_row['height'])
            bodygroup_from = int(patient_row['bodygroup_from'])
            bodygroup_to = int(patient_row['bodygroup_to'])
            ptab = int(patient_row['ptab'])
        else:
            # Generate patient data
            patient_id = f"P{session_idx+1:03d}"
            patient_age = int(np.random.normal(50, 15))
            patient_weight = int(np.random.normal(75, 15))
            patient_height = int(np.random.normal(170, 10))
            bodygroup_from = np.random.randint(0, 10)
            bodygroup_to = bodygroup_from  # Usually same for single-region scans
            ptab = np.random.randint(0, 100)

        # Get body part name from body group ID
        body_part = BODYGROUP_NAMES[min(bodygroup_from, len(BODYGROUP_NAMES)-1)]

        # Build conditioning array for PXChange
        # PXChange conditioning features: ['Age', 'Weight', 'Height', 'BodyGroup_from', 'BodyGroup_to', 'PTAB', 'entity_type']
        px_conditioning = np.array([
            patient_age,
            patient_weight,
            patient_height,
            bodygroup_from,
            bodygroup_to,
            ptab,
            0  # entity_type (0 = real patient)
        ], dtype=np.float32)

        # Truncate to match expected feature count if needed
        px_conditioning = px_conditioning[:len(PX_FEATURES)]

        print(f"  Session {session_idx+1}/{num_sessions} - Start: {session_start.strftime('%H:%M:%S')} - Patient {patient_id}")
        print(f"    Age: {patient_age}, Weight: {patient_weight}kg, Height: {patient_height}cm, Body: {body_part}")

        # Generate PXChange events
        px_tokens, px_durations = generate_pxchange_session(
            px_seq_model, px_dur_model, px_conditioning, px_scaler, device
        )

        # Process PXChange events
        current_time = session_start
        measurement_started = False

        for step_idx, (token_id, duration) in enumerate(zip(px_tokens, px_durations)):
            # Skip special tokens using PXChange vocab IDs
            if token_id in [PX_PAD_ID, PX_START_ID, PX_END_ID]:
                continue

            token_name = id_to_sourceid.get(int(token_id), f"TOKEN_{token_id}")

            # Check if this is a PAUSE event
            if token_name == 'PAUSE':
                event_type = 'pause'
                sourceID = 'PAUSE'
                scan_sequence = 'PAUSE'
                body_part_value = ''
            else:
                event_type = 'pxchange'
                sourceID = token_name
                scan_sequence = ''
                body_part_value = ''

            # Add event with new format matching sample eventlog
            all_events.append({
                'event_id': event_id,
                'timestamp': seconds_from_midnight(current_time),
                'datetime': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'event_type': event_type,
                'session_id': session_idx,
                'patient_id': patient_id,
                'sourceID': sourceID,
                'scan_sequence': scan_sequence,
                'body_part': body_part_value,
                'bodygroup_from': bodygroup_from,
                'bodygroup_to': bodygroup_to,
                'duration': round(float(duration), 1),
                'cumulative_time': seconds_from_midnight(current_time)
            })
            event_id += 1
            current_time += timedelta(seconds=float(duration))

            # Check if measurement start (MRI_EXU_95 triggers scan sequence)
            # Use explicit token ID check for reliability
            if not measurement_started and token_id == PX_EXU_ID:
                measurement_started = True

                # Generate SeqofSeq scan sequence
                seq_conditioning = np.random.randn(5).astype(np.float32)  # 5 core features
                seq_tokens, seq_durations = generate_seqofseq_sequence(
                    seq_seq_model, seq_dur_model, seq_conditioning, seq_scaler, device
                )

                # Add scan events
                for scan_idx, (scan_token, scan_duration) in enumerate(zip(seq_tokens, seq_durations)):
                    # Skip special tokens using SeqofSeq vocab IDs
                    if scan_token in [SEQ_PAD_ID, SEQ_START_ID, SEQ_END_ID]:
                        continue

                    scan_name = id_to_sequence.get(int(scan_token), f"SCAN_{scan_token}")

                    # Add scan event with new format matching sample eventlog
                    all_events.append({
                        'event_id': event_id,
                        'timestamp': seconds_from_midnight(current_time),
                        'datetime': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'event_type': 'scan',
                        'session_id': session_idx,
                        'patient_id': patient_id,
                        'sourceID': '',
                        'scan_sequence': scan_name,
                        'body_part': body_part,
                        'bodygroup_from': bodygroup_from,
                        'bodygroup_to': bodygroup_to,
                        'duration': round(float(scan_duration), 1),
                        'cumulative_time': seconds_from_midnight(current_time)
                    })
                    event_id += 1
                    current_time += timedelta(seconds=float(scan_duration))

        print(f"    Generated {len([e for e in all_events if e['session_id'] == session_idx])} events")

    # Create DataFrame
    schedule_df = pd.DataFrame(all_events)

    # Sort by cumulative_time (chronological order)
    schedule_df = schedule_df.sort_values('cumulative_time').reset_index(drop=True)

    # Update event_id to be sequential after sorting
    schedule_df['event_id'] = range(len(schedule_df))

    # Reorder columns to match sample format with body group additions
    column_order = [
        'event_id', 'timestamp', 'datetime', 'event_type', 'session_id', 'patient_id',
        'sourceID', 'scan_sequence', 'body_part', 'bodygroup_from', 'bodygroup_to',
        'duration', 'cumulative_time'
    ]
    schedule_df = schedule_df[column_order]

    print(f"\n{'='*70}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Total events: {len(schedule_df)}")
    print(f"  Total sessions: {num_sessions}")
    print(f"  PXChange events: {len(schedule_df[schedule_df['event_type'] == 'pxchange'])}")
    print(f"  Scan events: {len(schedule_df[schedule_df['event_type'] == 'scan'])}")
    print(f"  Pause events: {len(schedule_df[schedule_df['event_type'] == 'pause'])}")
    print(f"  Total duration: {schedule_df['cumulative_time'].max() / 3600:.2f} hours")

    # Save results
    if output_dir is None:
        output_dir = os.path.join(unified_dir, 'outputs', 'generated_schedules')
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'daily_schedule_{date_str}.csv')
    schedule_df.to_csv(output_file, index=False)
    print(f"\n[OK] Schedule saved to: {output_file}")

    return schedule_df


def print_schedule_summary(schedule_df, num_examples=5):
    """Print summary of generated schedule."""
    print(f"\n{'='*70}")
    print(f"SCHEDULE SUMMARY")
    print(f"{'='*70}\n")

    # Show first few events
    print("First events:")
    for _, row in schedule_df.head(num_examples).iterrows():
        event_name = row['sourceID'] if row['sourceID'] else row['scan_sequence']
        print(f"  [{row['datetime']}] {row['event_type']:10s} - {event_name:20s} ({row['duration']:.1f}s)")

    print("\n  ...")

    # Show last few events
    print("\nLast events:")
    for _, row in schedule_df.tail(num_examples).iterrows():
        event_name = row['sourceID'] if row['sourceID'] else row['scan_sequence']
        print(f"  [{row['datetime']}] {row['event_type']:10s} - {event_name:20s} ({row['duration']:.1f}s)")

    # Statistics per session
    print(f"\n{'='*70}")
    print("PER-SESSION STATISTICS")
    print(f"{'='*70}\n")

    for session_id in sorted(schedule_df['session_id'].unique()):
        session_data = schedule_df[schedule_df['session_id'] == session_id]
        start_time_str = session_data['datetime'].min()
        end_time_str = session_data['datetime'].max()
        duration = session_data['cumulative_time'].max() - session_data['cumulative_time'].min()

        print(f"Session {session_id}:")
        print(f"  Start: {start_time_str.split()[1]}")
        print(f"  End: {end_time_str.split()[1]}")
        print(f"  Duration: {duration/60:.1f} minutes")
        print(f"  Events: {len(session_data)}")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate daily MRI schedule')
    parser.add_argument('--date', type=str, default="2026-01-10",
                        help='Date in YYYY-MM-DD format (default: 2026-01-10)')
    parser.add_argument('--machine-id', type=int, default=141049,
                        help='MRI machine ID (default: 141049)')
    parser.add_argument('--patient-csv', type=str, default=None,
                        help='Path to patient data CSV file')
    parser.add_argument('--num-sessions', type=int, default=None,
                        help='Number of sessions (default: from CSV or 15)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for generated schedule')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Generate daily schedule
    schedule_df = generate_daily_schedule(
        date_str=args.date,
        machine_id=args.machine_id,
        patient_csv=args.patient_csv,
        num_sessions=args.num_sessions,
        output_dir=args.output_dir,
        seed=args.seed
    )

    # Print summary
    print_schedule_summary(schedule_df)

    print("\n" + "=" * 70)
    print("SUCCESS - Daily schedule generated!")
    print("=" * 70)
