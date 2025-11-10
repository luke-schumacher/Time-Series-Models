"""
Main Pipeline: Complete training and generation workflow
"""
import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd

from config import (
    SEQUENCE_MODEL_CONFIG, SEQUENCE_TRAINING_CONFIG,
    COUNTS_MODEL_CONFIG, COUNTS_TRAINING_CONFIG,
    MODEL_SAVE_DIR, OUTPUT_DIR, RANDOM_SEED, DATA_DIR
)
from preprocessing import load_preprocessed_data, create_dataloaders
from preprocessing.preprocess_raw_data import preprocess_all_datasets
from models import ConditionalSequenceGenerator, ConditionalCountsGenerator
from training import train_sequence_model, train_counts_model
from generation import generate_sequences_and_counts
from generation.generate_pipeline import save_generated_results, print_generation_examples


def set_random_seeds(seed=RANDOM_SEED):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def preprocess_pipeline(args):
    """
    Preprocessing pipeline to convert raw CSV files to training format.
    """
    print(f"\n{'='*70}")
    print(f"PREPROCESSING PIPELINE")
    print(f"{'='*70}\n")

    # Check if raw CSV files exist
    import glob
    raw_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    if not raw_files:
        print(f"Error: No CSV files found in {DATA_DIR}")
        print(f"Please place your raw MRI scan CSV files in the data directory.")
        return

    print(f"Found {len(raw_files)} raw CSV files")

    # Run preprocessing
    preprocess_all_datasets()

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
    print(f"CONDITIONAL GENERATION TRAINING PIPELINE")
    print(f"{'='*70}\n")

    # Set seeds
    set_random_seeds()

    # Load data
    print("Step 1: Loading preprocessed data...")
    df = load_preprocessed_data(dataset_ids=args.dataset_ids)

    # Create dataloaders
    print("\nStep 2: Creating dataloaders...")
    train_loader, val_loader, conditioning_scaler = create_dataloaders(
        df,
        batch_size=args.batch_size,
        validation_split=args.val_split
    )

    # Save scaler for later use
    import pickle
    with open(os.path.join(MODEL_SAVE_DIR, 'conditioning_scaler.pkl'), 'wb') as f:
        pickle.dump(conditioning_scaler, f)
    print("[OK] Conditioning scaler saved")

    # Train sequence model
    if not args.skip_sequence:
        print(f"\n{'='*70}")
        print("Step 3: Training Conditional Sequence Generator...")
        print(f"{'='*70}\n")

        sequence_model, seq_history = train_sequence_model(
            train_loader, val_loader,
            config=SEQUENCE_MODEL_CONFIG,
            training_config=SEQUENCE_TRAINING_CONFIG
        )
    else:
        print("\n[SKIP] Skipping sequence model training")

    # Train counts model
    if not args.skip_counts:
        print(f"\n{'='*70}")
        print("Step 4: Training Conditional Counts Generator...")
        print(f"{'='*70}\n")

        counts_model, counts_history = train_counts_model(
            train_loader, val_loader,
            config=COUNTS_MODEL_CONFIG,
            training_config=COUNTS_TRAINING_CONFIG
        )
    else:
        print("\n[SKIP] Skipping counts model training")

    print(f"\n{'='*70}")
    print("TRAINING PIPELINE COMPLETE")
    print(f"{'='*70}\n")
    print(f"[OK] Models saved to: {MODEL_SAVE_DIR}")


def generate_pipeline(args):
    """
    Generation pipeline using trained models.
    """
    print(f"\n{'='*70}")
    print(f"CONDITIONAL GENERATION PIPELINE")
    print(f"{'='*70}\n")

    # Set seeds
    set_random_seeds()

    # Check if models exist
    seq_model_path = os.path.join(MODEL_SAVE_DIR, 'sequence_model_best.pt')
    counts_model_path = os.path.join(MODEL_SAVE_DIR, 'counts_model_best.pt')
    scaler_path = os.path.join(MODEL_SAVE_DIR, 'conditioning_scaler.pkl')

    if not os.path.exists(seq_model_path):
        raise FileNotFoundError(f"Sequence model not found at {seq_model_path}. Train models first!")
    if not os.path.exists(counts_model_path):
        raise FileNotFoundError(f"Counts model not found at {counts_model_path}. Train models first!")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Using device: {device}\n")

    # Load models
    print("Step 1: Loading trained models...")

    # Load sequence model
    seq_checkpoint = torch.load(seq_model_path, map_location=device)
    sequence_model = ConditionalSequenceGenerator(seq_checkpoint['config'])
    sequence_model.load_state_dict(seq_checkpoint['model_state_dict'])
    sequence_model.to(device)
    sequence_model.eval()
    print("[OK] Sequence model loaded")

    # Load counts model
    counts_checkpoint = torch.load(counts_model_path, map_location=device)
    counts_model = ConditionalCountsGenerator(counts_checkpoint['config'])
    counts_model.load_state_dict(counts_checkpoint['model_state_dict'])
    counts_model.to(device)
    counts_model.eval()
    print("[OK] Counts model loaded")

    # Load scaler
    import pickle
    with open(scaler_path, 'rb') as f:
        conditioning_scaler = pickle.load(f)
    print("[OK] Conditioning scaler loaded")

    # Load conditioning data
    print("\nStep 2: Preparing conditioning data...")
    if args.conditioning_file:
        conditioning_df = pd.read_csv(args.conditioning_file)
        if 'dataset_id' not in conditioning_df.columns:
            raise ValueError("Conditioning file must contain 'dataset_id' column for per-customer generation")
        print(f"[OK] Loaded conditioning from {args.conditioning_file}")
        print(f"    Customers found: {conditioning_df['dataset_id'].nunique()}")
    else:
        # Use preprocessed data - get one representative row per customer
        df = load_preprocessed_data(dataset_ids=args.dataset_ids)
        # Get one row per customer (dataset_id)
        conditioning_df = df.groupby('dataset_id').first().reset_index()
        print(f"[OK] Loaded conditioning data for {len(conditioning_df)} customers")

    # Generate sequences and counts (per customer)
    print("\nStep 3: Generating sequences and counts (per customer)...")
    results_df = generate_sequences_and_counts(
        sequence_model,
        counts_model,
        conditioning_df,
        conditioning_scaler=conditioning_scaler,
        num_samples_per_customer=args.num_samples_per_customer,
        device=device,
        verbose=True
    )

    # Save results
    print("\nStep 4: Saving results...")
    output_file = save_generated_results(results_df, filename=args.output_file)

    # Print examples
    print_generation_examples(results_df, num_examples=min(5, len(conditioning_df)))

    print(f"\n{'='*70}")
    print("GENERATION PIPELINE COMPLETE")
    print(f"{'='*70}\n")
    print(f"[OK] Results saved to: {output_file}")


def evaluate_pipeline(args):
    """
    Evaluation pipeline: compare generated vs true sequences.
    """
    print(f"\n{'='*70}")
    print(f"EVALUATION PIPELINE")
    print(f"{'='*70}\n")

    # Load generated results
    generated_file = os.path.join(OUTPUT_DIR, args.generated_file)
    if not os.path.exists(generated_file):
        raise FileNotFoundError(f"Generated results not found at {generated_file}")

    generated_df = pd.read_csv(generated_file)
    print(f"[OK] Loaded generated results from {generated_file}")
    print(f"  Total customers: {generated_df['SN'].nunique()}")
    print(f"  Total sequences: {len(generated_df.groupby(['SN', 'sample_idx']))}")
    print(f"  Average length: {generated_df.groupby(['SN', 'sample_idx'])['step'].max().mean():.1f}")
    print(f"  Average total time: {generated_df.groupby(['SN', 'sample_idx'])['total_time'].first().mean():.1f}s")

    # Load true data for comparison
    print("\n[OK] Loading true data for comparison...")
    df = load_preprocessed_data(dataset_ids=args.dataset_ids)

    # Compute statistics
    true_lengths = df.groupby('SeqOrder').size()
    if 'true_total_time' in df.columns:
        true_total_times = df.groupby('SeqOrder')['true_total_time'].first()
    else:
        true_total_times = df.groupby('SeqOrder')['step_duration'].sum()

    generated_lengths = generated_df.groupby(['SN', 'sample_idx']).size()
    generated_total_times = generated_df.groupby(['SN', 'sample_idx'])['total_time'].first()

    print(f"\n{'='*70}")
    print("COMPARISON STATISTICS")
    print(f"{'='*70}\n")

    print("Sequence Length:")
    print(f"  True - Mean: {true_lengths.mean():.1f}, Std: {true_lengths.std():.1f}, Range: [{true_lengths.min()}, {true_lengths.max()}]")
    print(f"  Generated - Mean: {generated_lengths.mean():.1f}, Std: {generated_lengths.std():.1f}, Range: [{generated_lengths.min()}, {generated_lengths.max()}]")

    print("\nTotal Time (seconds):")
    print(f"  True - Mean: {true_total_times.mean():.1f}, Std: {true_total_times.std():.1f}, Range: [{true_total_times.min():.1f}, {true_total_times.max():.1f}]")
    print(f"  Generated - Mean: {generated_total_times.mean():.1f}, Std: {generated_total_times.std():.1f}, Range: [{generated_total_times.min():.1f}, {generated_total_times.max():.1f}]")

    # Token distribution
    print("\nToken Distribution (top 10):")
    true_tokens = df['sourceID'].value_counts().head(10)
    generated_tokens = generated_df['token_id'].value_counts().head(10)

    print("\n  True:")
    for token_id, count in true_tokens.items():
        print(f"    Token {token_id}: {count} ({count/len(df)*100:.1f}%)")

    print("\n  Generated:")
    for token_id, count in generated_tokens.items():
        print(f"    Token {token_id}: {count} ({count/len(generated_df)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Conditional Generation Pipeline")
    subparsers = parser.add_subparsers(dest='command', help='Pipeline command')

    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess raw CSV files')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train both models')
    train_parser.add_argument('--dataset-ids', nargs='+', default=None, help='Dataset IDs to use')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    train_parser.add_argument('--skip-sequence', action='store_true', help='Skip sequence model training')
    train_parser.add_argument('--skip-counts', action='store_true', help='Skip counts model training')

    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate sequences and counts per customer')
    generate_parser.add_argument('--dataset-ids', nargs='+', default=None, help='Dataset IDs (customers) for generation')
    generate_parser.add_argument('--conditioning-file', type=str, default=None, help='CSV file with conditioning data (must have dataset_id column)')
    generate_parser.add_argument('--num-samples-per-customer', type=int, default=15, help='Number of sequences to generate per customer (default: 15)')
    generate_parser.add_argument('--output-file', type=str, default='generated_sequences.csv', help='Output filename')
    generate_parser.add_argument('--use-gpu', action='store_true', help='Use GPU if available')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate generated sequences')
    eval_parser.add_argument('--generated-file', type=str, default='generated_sequences.csv', help='Generated results file')
    eval_parser.add_argument('--dataset-ids', nargs='+', default=None, help='Dataset IDs for comparison')

    args = parser.parse_args()

    if args.command == 'preprocess':
        preprocess_pipeline(args)
    elif args.command == 'train':
        train_pipeline(args)
    elif args.command == 'generate':
        generate_pipeline(args)
    elif args.command == 'evaluate':
        evaluate_pipeline(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
