"""
Simple End-to-End Schedule Generation

Loads all 5 trained models and generates complete daily MRI schedules.

Pipeline:
1. Temporal Model → predict session count + start times
2. For each session:
   - PXChange → generate patient exchange sequence + durations
   - When measurement detected → SeqofSeq → generate scan sequences + durations
3. Assemble timeline
4. Validate and save
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add paths
unified_dir = os.path.join(os.path.dirname(__file__), '..', '..')
pxchange_dir = os.path.join(unified_dir, '..', 'PXChange_Refactored')
seqofseq_dir = os.path.join(unified_dir, '..', 'SeqofSeq_Pipeline')
sys.path.insert(0, unified_dir)
sys.path.insert(0, pxchange_dir)
sys.path.insert(0, seqofseq_dir)

from config import MODEL_PATHS, VALIDATION_CONFIG, EVENT_TIMELINES_DIR, PATIENT_SESSIONS_DIR
from datetime_model.temporal_schedule_model import TemporalScheduleModel
from datetime_model.temporal_features import extract_temporal_features, features_to_array
from PXChange_Refactored.models.conditional_sequence_generator import ConditionalSequenceGenerator as PXChangeSeqGen
from PXChange_Refactored.models.conditional_counts_generator import ConditionalCountsGenerator as PXChangeDurGen
from SeqofSeq_Pipeline.models.conditional_sequence_generator import ConditionalSequenceGenerator as SeqofSeqSeqGen
from SeqofSeq_Pipeline.models.conditional_duration_predictor import ConditionalDurationPredictor as SeqofSeqDurGen
from PXChange_Refactored.config import SOURCEID_VOCAB, START_TOKEN_ID as PX_START, END_TOKEN_ID as PX_END
from SeqofSeq_Pipeline.config import START_TOKEN_ID as SEQ_START, END_TOKEN_ID as SEQ_END


class ScheduleGenerator:
    """End-to-end schedule generator."""

    def __init__(self, device=None):
        """
        Initialize generator and load all models.

        Args:
            device: Device to use (cuda/cpu)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create reverse vocabulary for decoding
        self.sourceid_vocab_reverse = {v: k for k, v in SOURCEID_VOCAB.items()}

        # Load models
        self.load_models()

    def load_models(self):
        """Load all 5 trained models."""
        print("\n[1/5] Loading models...")

        # Temporal model
        print("  Loading temporal model...")
        temporal_checkpoint = torch.load(MODEL_PATHS['temporal']['model'], map_location=self.device)
        self.temporal_model = TemporalScheduleModel()
        self.temporal_model.load_state_dict(temporal_checkpoint['model_state_dict'])
        self.temporal_model.to(self.device)
        self.temporal_model.eval()
        print("    ✓ Temporal model loaded")

        # PXChange sequence model
        print("  Loading PXChange sequence model...")
        pxchange_seq_checkpoint = torch.load(MODEL_PATHS['pxchange']['sequence'], map_location=self.device)
        pxchange_seq_config = pxchange_seq_checkpoint['model_config']
        self.pxchange_seq_model = PXChangeSeqGen(config=pxchange_seq_config)
        self.pxchange_seq_model.load_state_dict(pxchange_seq_checkpoint['model_state_dict'])
        self.pxchange_seq_model.to(self.device)
        self.pxchange_seq_model.eval()
        print("    ✓ PXChange sequence model loaded")

        # PXChange duration model
        print("  Loading PXChange duration model...")
        pxchange_dur_checkpoint = torch.load(MODEL_PATHS['pxchange']['duration'], map_location=self.device)
        pxchange_dur_config = pxchange_dur_checkpoint['model_config']
        self.pxchange_dur_model = PXChangeDurGen(config=pxchange_dur_config)
        self.pxchange_dur_model.load_state_dict(pxchange_dur_checkpoint['model_state_dict'])
        self.pxchange_dur_model.to(self.device)
        self.pxchange_dur_model.eval()
        print("    ✓ PXChange duration model loaded")

        # SeqofSeq sequence model
        print("  Loading SeqofSeq sequence model...")
        seqofseq_seq_checkpoint = torch.load(MODEL_PATHS['seqofseq']['sequence'], map_location=self.device)
        seqofseq_seq_config = seqofseq_seq_checkpoint['model_config']
        self.seqofseq_seq_model = SeqofSeqSeqGen(config=seqofseq_seq_config)
        self.seqofseq_seq_model.load_state_dict(seqofseq_seq_checkpoint['model_state_dict'])
        self.seqofseq_seq_model.to(self.device)
        self.seqofseq_seq_model.eval()
        print("    ✓ SeqofSeq sequence model loaded")

        # SeqofSeq duration model
        print("  Loading SeqofSeq duration model...")
        seqofseq_dur_checkpoint = torch.load(MODEL_PATHS['seqofseq']['duration'], map_location=self.device)
        seqofseq_dur_config = seqofseq_dur_checkpoint['model_config']
        self.seqofseq_dur_model = SeqofSeqDurGen(config=seqofseq_dur_config)
        self.seqofseq_dur_model.load_state_dict(seqofseq_dur_checkpoint['model_state_dict'])
        self.seqofseq_dur_model.to(self.device)
        self.seqofseq_dur_model.eval()
        print("    ✓ SeqofSeq duration model loaded")

        print("\n[SUCCESS] All models loaded!")

    def predict_daily_structure(self, date, machine_id):
        """
        Predict session count and start times for a given date.

        Args:
            date: datetime.date object
            machine_id: Machine identifier

        Returns:
            session_count: Number of sessions
            start_times: List of start times (seconds from midnight)
        """
        # Extract temporal features
        machine_id_encoded = hash(str(machine_id)) % 1000
        features_dict = extract_temporal_features(
            date=date,
            machine_id=machine_id_encoded,
            typical_load=12.0  # Default
        )
        features = features_to_array(features_dict)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            session_count = self.temporal_model.predict_session_count(features_tensor)
            _, timing_params = self.temporal_model(features_tensor)

            # Sample start times
            num_sessions = int(session_count[0].item())
            start_times = self.temporal_model.sample_start_times(timing_params, num_sessions)
            start_times = start_times.cpu().numpy()

        return num_sessions, start_times

    def generate_pxchange_sequence(self, conditioning):
        """
        Generate PXChange patient exchange sequence.

        Args:
            conditioning: [1, conditioning_dim] tensor

        Returns:
            tokens: List of sourceID tokens
            durations: List of durations
        """
        with torch.no_grad():
            # Generate sequence autoregressively
            generated_tokens = [PX_START]
            max_len = 50

            for _ in range(max_len):
                input_seq = torch.tensor([generated_tokens], dtype=torch.long).to(self.device)
                logits = self.pxchange_seq_model(conditioning, input_seq)
                next_token_logits = logits[0, -1, :]
                next_token = torch.argmax(next_token_logits).item()
                generated_tokens.append(next_token)

                if next_token == PX_END:
                    break

            # Generate durations
            sequence_tokens = torch.tensor([generated_tokens], dtype=torch.long).to(self.device)
            sequence_features = torch.zeros(1, len(generated_tokens), 2).to(self.device)  # Position, Direction

            mu, sigma = self.pxchange_dur_model(conditioning, sequence_tokens, sequence_features)
            durations = mu[0].cpu().numpy()

        return generated_tokens, durations

    def generate_seqofseq_sequence(self, conditioning):
        """
        Generate SeqofSeq scan sequence.

        Args:
            conditioning: [1, conditioning_dim] tensor

        Returns:
            tokens: List of scan sequence tokens
            durations: List of durations
        """
        with torch.no_grad():
            # Generate sequence
            generated_tokens = [SEQ_START]
            max_len = 30

            for _ in range(max_len):
                input_seq = torch.tensor([generated_tokens], dtype=torch.long).to(self.device)
                logits = self.seqofseq_seq_model(conditioning, input_seq)
                next_token_logits = logits[0, -1, :]
                next_token = torch.argmax(next_token_logits).item()
                generated_tokens.append(next_token)

                if next_token == SEQ_END:
                    break

            # Generate durations
            sequence_tokens = torch.tensor([generated_tokens], dtype=torch.long).to(self.device)
            mu, sigma = self.seqofseq_dur_model(conditioning, sequence_tokens)
            durations = mu[0].cpu().numpy()

        return generated_tokens, durations

    def generate_daily_schedule(self, date, machine_id, output_dir=None):
        """
        Generate complete daily schedule.

        Args:
            date: datetime.date object or string (YYYY-MM-DD)
            machine_id: Machine identifier
            output_dir: Directory to save output (default: outputs/)

        Returns:
            event_timeline_df: DataFrame with event timeline
            patient_sessions_df: DataFrame with patient sessions
        """
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d').date()

        if output_dir is None:
            output_dir = os.path.join(unified_dir, 'UnifiedSchedulePipeline', 'outputs')
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 70)
        print(f"GENERATING SCHEDULE: {date} (Machine: {machine_id})")
        print("=" * 70)

        # Step 1: Predict daily structure
        print("\n[2/5] Predicting daily structure...")
        num_sessions, start_times = self.predict_daily_structure(date, machine_id)
        print(f"  Predicted sessions: {num_sessions}")
        print(f"  Start times: {[int(t) for t in start_times]} seconds from midnight")

        # Step 2: Generate sessions
        print("\n[3/5] Generating sessions...")
        events = []
        current_time = 0
        event_id = 0

        for session_idx in range(num_sessions):
            session_start = int(start_times[session_idx])
            print(f"\n  Session {session_idx + 1}/{num_sessions} (start: {session_start}s = {session_start/3600:.1f}h)")

            # Dummy conditioning (would be real patient data)
            pxchange_cond = torch.randn(1, 7).to(self.device)  # Age, Weight, Height, BodyGroup_from/to, PTAB, entity_type

            # Generate PXChange sequence
            px_tokens, px_durations = self.generate_pxchange_sequence(pxchange_cond)
            print(f"    PXChange: {len(px_tokens)} events")

            # Add PXChange events
            for i, (token, duration) in enumerate(zip(px_tokens, px_durations)):
                sourceid_name = self.sourceid_vocab_reverse.get(token, f'UNK_{token}')

                events.append({
                    'event_id': event_id,
                    'timestamp': current_time,
                    'datetime': str(timedelta(seconds=int(current_time))),
                    'event_type': 'pxchange',
                    'session_id': session_idx + 1,
                    'patient_id': f'P{session_idx + 1:03d}',
                    'sourceID': sourceid_name,
                    'scan_sequence': '',
                    'body_part': '',
                    'duration': float(duration),
                    'cumulative_time': current_time
                })

                current_time += duration
                event_id += 1

                # If measurement start detected, generate scans
                if sourceid_name == 'MRI_MSR_104':
                    print(f"      Measurement detected, generating scans...")

                    # Dummy conditioning for scans
                    seqofseq_cond = torch.randn(1, 10).to(self.device)  # BodyPart, SystemType, Country, Group, entity_type + coils

                    # Generate scan sequence
                    scan_tokens, scan_durations = self.generate_seqofseq_sequence(seqofseq_cond)
                    print(f"      Generated {len(scan_tokens)} scans")

                    # Add scan events
                    for scan_token, scan_duration in zip(scan_tokens[1:-1], scan_durations[1:-1]):  # Skip START/END
                        events.append({
                            'event_id': event_id,
                            'timestamp': current_time,
                            'datetime': str(timedelta(seconds=int(current_time))),
                            'event_type': 'scan',
                            'session_id': session_idx + 1,
                            'patient_id': f'P{session_idx + 1:03d}',
                            'sourceID': '',
                            'scan_sequence': f'SCAN_{scan_token}',
                            'body_part': 'HEAD',
                            'duration': float(scan_duration),
                            'cumulative_time': current_time
                        })

                        current_time += scan_duration
                        event_id += 1

        # Step 3: Create DataFrames
        print("\n[4/5] Assembling timeline...")
        event_timeline_df = pd.DataFrame(events)
        print(f"  Total events: {len(event_timeline_df)}")
        print(f"  Total duration: {current_time/3600:.2f} hours")

        # Patient sessions summary
        sessions_summary = event_timeline_df.groupby('session_id').agg({
            'timestamp': ['min', 'max'],
            'event_id': 'count',
            'duration': 'sum'
        }).reset_index()

        patient_sessions_df = pd.DataFrame({
            'session_id': sessions_summary['session_id'],
            'patient_id': [f'P{i:03d}' for i in sessions_summary['session_id']],
            'session_start_time': sessions_summary[('timestamp', 'min')],
            'session_end_time': sessions_summary[('timestamp', 'max')],
            'session_duration': sessions_summary[('duration', 'sum')],
            'num_events': sessions_summary[('event_id', 'count')]
        })

        # Step 4: Validate
        print("\n[5/5] Validating schedule...")
        validation_results = self.validate_schedule(event_timeline_df, patient_sessions_df)

        # Save
        date_str = date.strftime('%Y%m%d')
        timeline_path = os.path.join(output_dir, f'event_timeline_{machine_id}_{date_str}.csv')
        sessions_path = os.path.join(output_dir, f'patient_sessions_{machine_id}_{date_str}.csv')

        event_timeline_df.to_csv(timeline_path, index=False)
        patient_sessions_df.to_csv(sessions_path, index=False)

        print(f"\n[SAVED] Event timeline: {timeline_path}")
        print(f"[SAVED] Patient sessions: {sessions_path}")

        print("\n" + "=" * 70)
        print("[SUCCESS] Schedule generation complete!")
        print("=" * 70 + "\n")

        return event_timeline_df, patient_sessions_df

    def validate_schedule(self, event_timeline_df, patient_sessions_df):
        """
        Validate generated schedule against constraints.

        Returns:
            validation_results: Dict with validation results
        """
        results = {}

        # Total duration
        total_duration_hours = event_timeline_df['duration'].sum() / 3600
        results['total_duration_hours'] = total_duration_hours
        results['duration_valid'] = (
            VALIDATION_CONFIG['min_total_duration_hours'] <= total_duration_hours <=
            VALIDATION_CONFIG['max_total_duration_hours']
        )

        # Session count
        num_sessions = len(patient_sessions_df)
        results['num_sessions'] = num_sessions
        results['session_count_valid'] = (
            VALIDATION_CONFIG['min_sessions'] <= num_sessions <=
            VALIDATION_CONFIG['max_sessions']
        )

        # Check for negative durations
        results['all_positive_durations'] = (event_timeline_df['duration'] > 0).all()

        # Print validation
        print(f"  Total duration: {total_duration_hours:.2f} hours {'✓' if results['duration_valid'] else '✗'}")
        print(f"  Session count: {num_sessions} {'✓' if results['session_count_valid'] else '✗'}")
        print(f"  Positive durations: {'✓' if results['all_positive_durations'] else '✗'}")

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate MRI daily schedules')
    parser.add_argument('--date', type=str, default=None, help='Date (YYYY-MM-DD)')
    parser.add_argument('--machine_id', type=str, default='141049', help='Machine ID')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')

    args = parser.parse_args()

    # Default to today if no date specified
    if args.date is None:
        date = datetime.now().date()
    else:
        date = datetime.strptime(args.date, '%Y-%m-%d').date()

    # Create generator
    generator = ScheduleGenerator()

    # Generate schedule
    event_timeline_df, patient_sessions_df = generator.generate_daily_schedule(
        date=date,
        machine_id=args.machine_id,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
