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
    num_sessions=5,
    output_dir=None,
    seed=None
):
    """
    Generate complete daily MRI schedule using all 5 models.

    Args:
        date_str: Date in "YYYY-MM-DD" format
        machine_id: MRI machine ID
        num_sessions: Number of patient sessions to generate
        output_dir: Directory to save outputs
        seed: Random seed

    Returns:
        schedule_df: Complete daily schedule DataFrame
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("GENERATING DAILY MRI SCHEDULE")
    print("=" * 70)
    print(f"Date: {date_str}")
    print(f"Machine ID: {machine_id}")
    print(f"Sessions: {num_sessions}")
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

    # Use sampled start times
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

    # Body part options for random selection
    body_parts = ['HEAD', 'CHEST', 'ABDOMEN', 'PELVIS', 'SPINE', 'EXTREMITY']

    for session_idx, session_start in enumerate(start_datetimes):
        # Generate patient ID (starting at P001, not P000)
        patient_id = f"P{session_idx+1:03d}"
        body_part = np.random.choice(body_parts)

        # Random conditioning for PXChange
        px_conditioning = np.random.randn(len(PX_FEATURES)).astype(np.float32)

        # Extract patient height/weight from conditioning
        # PXChange conditioning features: ['age', 'weight', 'height', 'BodyGroup_from', 'BodyGroup_to', 'ptab']
        if len(PX_FEATURES) >= 3:
            # Conditioning has: age, weight, height
            patient_age = int(px_conditioning[0] * 15 + 50)  # Denormalize: mean 50, std 15
            patient_weight = int(px_conditioning[1] * 15 + 75)  # Denormalize: mean 75kg, std 15
            patient_height = int(px_conditioning[2] * 10 + 170)  # Denormalize: mean 170cm, std 10
        else:
            # Defaults if not available
            patient_age = 50
            patient_weight = 75
            patient_height = 170

        print(f"  Session {session_idx+1}/{num_sessions} - Start: {session_start.strftime('%H:%M:%S')} - Patient {patient_id} (Age: {patient_age}, Weight: {patient_weight}kg, Height: {patient_height}cm)")

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

    # Reorder columns to match sample format EXACTLY
    column_order = [
        'event_id', 'timestamp', 'datetime', 'event_type', 'session_id', 'patient_id',
        'sourceID', 'scan_sequence', 'body_part', 'duration', 'cumulative_time'
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
    # Generate a sample daily schedule
    schedule_df = generate_daily_schedule(
        date_str="2026-01-10",
        machine_id=141049,
        num_sessions=5,
        seed=42
    )

    # Print summary
    print_schedule_summary(schedule_df)

    print("\n" + "=" * 70)
    print("SUCCESS - Prototype daily schedule generated!")
    print("=" * 70)
