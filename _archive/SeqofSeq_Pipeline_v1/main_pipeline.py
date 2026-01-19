"""
Main Pipeline: Complete preprocessing, training, and generation workflow
for MRI Scan Sequence and Duration Prediction
"""
import os
import sys
import argparse
import torch
import pickle
import numpy as np

from preprocessing import preprocess_mri_data, load_preprocessed_data, create_dataloaders
from config import (
    DATA_DIR, MODEL_SAVE_DIR, OUTPUT_DIR, RANDOM_SEED,
    SEQUENCE_MODEL_CONFIG, SEQUENCE_TRAINING_CONFIG,
    DURATION_MODEL_CONFIG, DURATION_TRAINING_CONFIG,
    CONDITIONING_FEATURES, START_TOKEN_ID, END_TOKEN_ID,
    PAD_TOKEN_ID, MAX_SEQ_LEN
)
from models import ConditionalSequenceGenerator, ConditionalDurationPredictor
from training import train_sequence_model, train_duration_model

# Set random seeds
def set_random_seeds(seed=RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def preprocess_pipeline(args):
    """
    Preprocessing pipeline to convert raw CSV to training format.
    """
    print(f"\n{'='*70}")
    print(f"PREPROCESSING PIPELINE")
    print(f"{'='*70}\n")

    # Check if raw data exists
    data_file = args.data_file
    if not os.path.exists(os.path.join(DATA_DIR, data_file)):
        print(f"Error: Data file not found: {os.path.join(DATA_DIR, data_file)}")
        return

    # Run preprocessing
    preprocessed_df, metadata = preprocess_mri_data(data_file=data_file, save_preprocessed=True)

    print(f"\n{'='*70}")
    print("PREPROCESSING PIPELINE COMPLETE")
    print(f"{'='*70}\n")
    print(f"[OK] Preprocessed data saved to: {os.path.join(DATA_DIR, 'preprocessed')}")
    print(f"\nYou can now train the models with: python main_pipeline.py train")


def train_pipeline(args):
    """
    Complete training pipeline for both models.
    """
    print(f"\n{'='*70}")
    print(f"TRAINING PIPELINE")
    print(f"{'='*70}\n")

    # Set random seeds
    set_random_seeds()

    # Load data
    print("Step 1: Loading preprocessed data...")
    df, metadata = load_preprocessed_data()

    print("\n[OK] Data loaded successfully")
    print(f"Total sequences: {df['sequence_idx'].nunique()}")
    print(f"Vocabulary size: {metadata['vocab_size']}")
    print(f"Conditioning features: {metadata['num_conditioning_features']}")

    # Update model configs with metadata
    SEQUENCE_MODEL_CONFIG['vocab_size'] = metadata['vocab_size']
    SEQUENCE_MODEL_CONFIG['conditioning_dim'] = metadata['num_conditioning_features']
    DURATION_MODEL_CONFIG['conditioning_dim'] = metadata['num_conditioning_features']
    DURATION_MODEL_CONFIG['sequence_feature_dim'] = metadata['vocab_size']
    DURATION_MODEL_CONFIG['vocab_size'] = metadata['vocab_size']

    # Create dataloaders
    print("\nStep 2: Creating dataloaders...")
    train_loader, val_loader, conditioning_scaler = create_dataloaders(
        df, metadata,
        batch_size=args.batch_size,
        validation_split=args.val_split
    )

    # Save scaler and metadata
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    with open(os.path.join(MODEL_SAVE_DIR, 'conditioning_scaler.pkl'), 'wb') as f:
        pickle.dump(conditioning_scaler, f)
    with open(os.path.join(MODEL_SAVE_DIR, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    print("[OK] Conditioning scaler and metadata saved")

    # Update training configs with args
    SEQUENCE_TRAINING_CONFIG['epochs'] = args.epochs
    SEQUENCE_TRAINING_CONFIG['learning_rate'] = args.lr
    SEQUENCE_TRAINING_CONFIG['batch_size'] = args.batch_size
    DURATION_TRAINING_CONFIG['epochs'] = args.epochs
    DURATION_TRAINING_CONFIG['learning_rate'] = args.lr
    DURATION_TRAINING_CONFIG['batch_size'] = args.batch_size

    # Train sequence model
    print(f"\n{'='*70}")
    print("Step 3: Training Sequence Generator...")
    print(f"{'='*70}\n")

    sequence_model, seq_history = train_sequence_model(
        train_loader, val_loader,
        config=SEQUENCE_MODEL_CONFIG,
        training_config=SEQUENCE_TRAINING_CONFIG
    )

    # Train duration model
    print(f"\n{'='*70}")
    print("Step 4: Training Duration Predictor...")
    print(f"{'='*70}\n")

    duration_model, dur_history = train_duration_model(
        train_loader, val_loader,
        config=DURATION_MODEL_CONFIG,
        training_config=DURATION_TRAINING_CONFIG
    )

    print(f"\n{'='*70}")
    print("TRAINING PIPELINE COMPLETE")
    print(f"{'='*70}\n")
    print(f"[OK] Models saved to: {MODEL_SAVE_DIR}")
    print(f"    - sequence_model_best.pt")
    print(f"    - duration_model_best.pt")
    print(f"\nNext step: Generate sequences with:")
    print(f"  python main_pipeline.py generate --num-samples 10")


def generate_pipeline(args):
    """
    Generation pipeline using trained models.
    """
    print(f"\n{'='*70}")
    print(f"GENERATION PIPELINE")
    print(f"{'='*70}\n")

    # Set random seeds
    set_random_seeds()

    # Check if models exist
    seq_model_path = os.path.join(MODEL_SAVE_DIR, 'sequence_model_best.pt')
    dur_model_path = os.path.join(MODEL_SAVE_DIR, 'duration_model_best.pt')
    metadata_path = os.path.join(MODEL_SAVE_DIR, 'metadata.pkl')

    if not os.path.exists(seq_model_path):
        print(f"[ERROR] Sequence model not found at {seq_model_path}")
        print("Please train models first: python main_pipeline.py train")
        return

    if not os.path.exists(dur_model_path):
        print(f"[ERROR] Duration model not found at {dur_model_path}")
        print("Please train models first: python main_pipeline.py train")
        return

    # Load metadata
    print("Step 1: Loading metadata...")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    print(f"[OK] Loaded metadata (vocab_size={metadata['vocab_size']})")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load models
    print("Step 2: Loading trained models...")

    # Update configs
    SEQUENCE_MODEL_CONFIG['vocab_size'] = metadata['vocab_size']
    SEQUENCE_MODEL_CONFIG['conditioning_dim'] = metadata['num_conditioning_features']

    seq_checkpoint = torch.load(seq_model_path, map_location=device)
    sequence_model = ConditionalSequenceGenerator(seq_checkpoint['config'])
    sequence_model.load_state_dict(seq_checkpoint['model_state_dict'])
    sequence_model.to(device)
    sequence_model.eval()
    print("[OK] Sequence model loaded")

    DURATION_MODEL_CONFIG['conditioning_dim'] = metadata['num_conditioning_features']
    DURATION_MODEL_CONFIG['sequence_feature_dim'] = metadata['vocab_size']

    dur_checkpoint = torch.load(dur_model_path, map_location=device)
    duration_model = ConditionalDurationPredictor(dur_checkpoint['config'])
    duration_model.load_state_dict(dur_checkpoint['model_state_dict'])
    duration_model.to(device)
    duration_model.eval()
    print("[OK] Duration model loaded")

    # Load conditioning scaler
    scaler_path = os.path.join(MODEL_SAVE_DIR, 'conditioning_scaler.pkl')
    with open(scaler_path, 'rb') as f:
        conditioning_scaler = pickle.load(f)
    print("[OK] Conditioning scaler loaded")

    # Generate sequences
    print(f"\nStep 3: Generating {args.num_samples} sequences...")

    import pandas as pd

    # Load some example conditioning data
    df, _ = load_preprocessed_data()
    sample_patients = df.groupby('sequence_idx').first().sample(min(args.num_samples, len(df['sequence_idx'].unique())))

    results = []

    for idx, (_, patient_row) in enumerate(sample_patients.iterrows()):
        # Extract and scale conditioning
        coil_cols = metadata.get('coil_cols', [])
        all_cond_features = CONDITIONING_FEATURES + coil_cols

        conditioning = []
        for feat in all_cond_features:
            val = patient_row.get(feat, 0)
            conditioning.append(float(val) if val is not None else 0.0)
        conditioning = np.array(conditioning, dtype=np.float32)
        conditioning = np.nan_to_num(conditioning, nan=0.0)
        conditioning = conditioning_scaler.transform(conditioning.reshape(1, -1))[0]
        conditioning = np.nan_to_num(conditioning, nan=0.0)

        conditioning_tensor = torch.from_numpy(conditioning).unsqueeze(0).to(device)

        # Generate sequence
        with torch.no_grad():
            # Start with START token
            generated = torch.full((1, 1), START_TOKEN_ID, dtype=torch.long, device=device)

            for step in range(MAX_SEQ_LEN - 1):
                # Get predictions
                logits = sequence_model(conditioning_tensor, generated)

                # Get next token (greedy decoding)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

                # Stop if END token
                if next_token.item() == END_TOKEN_ID:
                    break

                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)

            # Remove START token
            sequence_tokens = generated[0, 1:].cpu().numpy()

            # Generate durations for the sequence
            seq_len = len(sequence_tokens)
            tokens_tensor = torch.from_numpy(sequence_tokens).unsqueeze(0).to(device)

            # Pad to match model input
            padded_tokens = torch.full((1, MAX_SEQ_LEN), PAD_TOKEN_ID, dtype=torch.long, device=device)
            padded_tokens[0, :seq_len] = tokens_tensor[0, :seq_len]

            # Create dummy sequence features
            sequence_features = torch.zeros(1, MAX_SEQ_LEN, 2, device=device)

            # Create mask
            mask = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.bool, device=device)
            mask[0, :seq_len] = True

            # Predict durations
            mu, sigma = duration_model(conditioning_tensor, padded_tokens, sequence_features, mask)
            durations = mu[0, :seq_len].cpu().numpy()

            # Decode tokens to sequence names
            id_to_token = metadata['sequence_id_to_token']

            # Save results
            for step_idx, (token_id, duration) in enumerate(zip(sequence_tokens, durations)):
                sequence_name = id_to_token.get(int(token_id), f"UNKNOWN_{token_id}")
                results.append({
                    'sample_id': idx,
                    'step': step_idx,
                    'sequence_id': int(token_id),
                    'sequence_name': sequence_name,
                    'predicted_duration': float(duration),
                    'patient_id': patient_row.get('PatientID', f'GENERATED_{idx}')
                })

        print(f"  Generated sequence {idx+1}/{args.num_samples}: {len(sequence_tokens)} steps, total time: {durations.sum():.1f}s")

    # Save to CSV
    results_df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, args.output_file)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\n{'='*70}")
    print("GENERATION COMPLETE")
    print(f"{'='*70}\n")
    print(f"[OK] Generated {args.num_samples} sequences")
    print(f"[OK] Results saved to: {output_path}")
    print(f"\nSample output:")
    print(results_df.head(10).to_string())


def main():
    parser = argparse.ArgumentParser(description="SeqofSeq Pipeline - MRI Scan Sequence Prediction")
    subparsers = parser.add_subparsers(dest='command', help='Pipeline command')

    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess raw CSV files')
    preprocess_parser.add_argument('--data-file', type=str, default='176625.csv',
                                  help='Name of data file in data/ directory')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train both models')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')

    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate sequences and durations')
    generate_parser.add_argument('--num-samples', type=int, default=10,
                                help='Number of sequences to generate')
    generate_parser.add_argument('--output-file', type=str, default='generated_sequences.csv',
                                help='Output filename')

    args = parser.parse_args()

    if args.command == 'preprocess':
        preprocess_pipeline(args)
    elif args.command == 'train':
        train_pipeline(args)
    elif args.command == 'generate':
        generate_pipeline(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
