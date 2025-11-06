"""
Complete generation pipeline combining both models
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SEQUENCE_SAMPLING_CONFIG, COUNTS_SAMPLING_CONFIG,
    OUTPUT_DIR, END_TOKEN_ID, PAD_TOKEN_ID
)
from preprocessing.sequence_encoder import decode_sequences, sequence_to_text


def generate_sequences_and_counts(
    sequence_model,
    counts_model,
    conditioning_data,
    conditioning_scaler=None,
    num_samples=10,
    device='cpu',
    verbose=True
):
    """
    Complete generation pipeline:
    1. Generate symbolic sequences using sequence model
    2. Predict counts with uncertainty using counts model
    3. Sample realistic counts from Gamma distributions

    Args:
        sequence_model: Trained ConditionalSequenceGenerator
        counts_model: Trained ConditionalCountsGenerator
        conditioning_data: DataFrame or array of conditioning features
        conditioning_scaler: Scaler for conditioning features
        num_samples: Number of sequences to generate per conditioning
        device: torch device
        verbose: Whether to print progress

    Returns:
        results_df: DataFrame with generated sequences and counts
    """
    sequence_model.eval()
    counts_model.eval()

    results = []

    # Prepare conditioning data
    if isinstance(conditioning_data, pd.DataFrame):
        from config import CONDITIONING_FEATURES
        conditioning_array = conditioning_data[CONDITIONING_FEATURES].values
    else:
        conditioning_array = conditioning_data

    if conditioning_scaler is not None:
        conditioning_array = conditioning_scaler.transform(conditioning_array)

    conditioning_tensor = torch.from_numpy(conditioning_array).float().to(device)

    batch_size = conditioning_tensor.shape[0]

    if verbose:
        print(f"\n{'='*70}")
        print(f"GENERATING SEQUENCES AND COUNTS")
        print(f"{'='*70}\n")
        print(f"Conditioning samples: {batch_size}")
        print(f"Samples per conditioning: {num_samples}")
        print(f"Total sequences to generate: {batch_size * num_samples}\n")

    with torch.no_grad():
        for cond_idx in tqdm(range(batch_size), desc="Generating", disable=not verbose):
            # Get single conditioning vector
            single_cond = conditioning_tensor[cond_idx:cond_idx+1]  # [1, cond_dim]

            # Repeat for num_samples
            batch_cond = single_cond.repeat(num_samples, 1)  # [num_samples, cond_dim]

            # Step 1: Generate symbolic sequences
            # Use model's max_seq_len - 1 to prevent exceeding positional encoding buffer
            max_gen_length = min(SEQUENCE_SAMPLING_CONFIG['max_length'], sequence_model.max_seq_len - 1)
            generated_tokens = sequence_model.generate(
                batch_cond,
                max_length=max_gen_length,
                temperature=SEQUENCE_SAMPLING_CONFIG['temperature'],
                top_k=SEQUENCE_SAMPLING_CONFIG['top_k'],
                top_p=SEQUENCE_SAMPLING_CONFIG['top_p']
            )  # [num_samples, seq_len]

            # Step 2: For each generated sequence, predict counts
            for sample_idx in range(num_samples):
                tokens = generated_tokens[sample_idx]
                cond = batch_cond[sample_idx:sample_idx+1]

                # Find actual sequence length (before PAD or after END)
                token_list = tokens.cpu().numpy()
                if END_TOKEN_ID in token_list:
                    end_idx = np.where(token_list == END_TOKEN_ID)[0][0] + 1
                else:
                    end_idx = len(token_list)

                # Trim to actual length
                tokens_trimmed = tokens[:end_idx].unsqueeze(0)  # [1, actual_len]

                # Create dummy sequence features (zeros for now)
                seq_features = torch.zeros(1, end_idx, 2, device=device)

                # Create mask
                mask = torch.ones(1, end_idx, dtype=torch.bool, device=device)

                # Predict counts with uncertainty
                mu, sigma = counts_model(
                    cond,
                    tokens_trimmed,
                    seq_features,
                    mask
                )  # [1, actual_len]

                # Sample counts from Gamma distribution
                counts_sampled = counts_model.sample_counts(
                    mu, sigma, num_samples=1
                ).squeeze(-1)  # [1, actual_len]

                # Decode tokens
                token_strings = decode_sequences(tokens_trimmed[0].cpu().numpy(), remove_special_tokens=False)

                # Store results
                for step_idx in range(end_idx):
                    results.append({
                        'conditioning_idx': cond_idx,
                        'sample_idx': sample_idx,
                        'step': step_idx,
                        'token_id': token_list[step_idx],
                        'token_name': token_strings[step_idx] if step_idx < len(token_strings) else 'PAD',
                        'predicted_mu': mu[0, step_idx].item(),
                        'predicted_sigma': sigma[0, step_idx].item(),
                        'sampled_duration': counts_sampled[0, step_idx].item()
                    })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate total time per sequence
    sequence_totals = results_df.groupby(['conditioning_idx', 'sample_idx'])['sampled_duration'].sum().reset_index()
    sequence_totals.columns = ['conditioning_idx', 'sample_idx', 'total_time']

    results_df = results_df.merge(sequence_totals, on=['conditioning_idx', 'sample_idx'])

    if verbose:
        print(f"\n✓ Generation complete!")
        print(f"  Total sequences generated: {len(results_df['sample_idx'].unique())}")
        print(f"  Average sequence length: {results_df.groupby('sample_idx')['step'].max().mean():.1f}")
        print(f"  Average total time: {results_df.groupby('sample_idx')['total_time'].first().mean():.1f}s")
        print(f"  Total time range: [{results_df['total_time'].min():.1f}s, {results_df['total_time'].max():.1f}s]")

    return results_df


def save_generated_results(results_df, filename='generated_sequences.csv'):
    """
    Save generated results to CSV.
    """
    output_path = os.path.join(OUTPUT_DIR, filename)
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")
    return output_path


def print_generation_examples(results_df, num_examples=3):
    """
    Print examples of generated sequences.
    """
    print(f"\n{'='*70}")
    print(f"GENERATION EXAMPLES")
    print(f"{'='*70}\n")

    unique_samples = results_df.groupby(['conditioning_idx', 'sample_idx']).groups.keys()
    examples = list(unique_samples)[:num_examples]

    for i, (cond_idx, sample_idx) in enumerate(examples, 1):
        sample_data = results_df[
            (results_df['conditioning_idx'] == cond_idx) &
            (results_df['sample_idx'] == sample_idx)
        ]

        token_sequence = sample_data['token_name'].tolist()
        durations = sample_data['sampled_duration'].values
        total_time = sample_data['total_time'].iloc[0]

        print(f"Example {i} (Conditioning={cond_idx}, Sample={sample_idx}):")
        print(f"  Length: {len(token_sequence)} steps")
        print(f"  Sequence: {' -> '.join(token_sequence)}")
        print(f"  Durations (first 10): {durations[:10].round(1)}")
        print(f"  Total time: {total_time:.1f}s\n")


if __name__ == "__main__":
    # Test generation pipeline
    print("Testing generation pipeline...")

    import torch
    from models import ConditionalSequenceGenerator, ConditionalCountsGenerator

    # Create dummy models
    seq_config = {
        'vocab_size': 18,
        'd_model': 64,
        'nhead': 4,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'dim_feedforward': 256,
        'dropout': 0.1,
        'max_seq_len': 64,
        'conditioning_dim': 6
    }

    counts_config = {
        'd_model': 64,
        'nhead': 4,
        'num_encoder_layers': 2,
        'num_cross_attention_layers': 2,
        'dim_feedforward': 256,
        'dropout': 0.1,
        'max_seq_len': 64,
        'conditioning_dim': 6,
        'sequence_feature_dim': 18,
        'min_sigma': 0.1
    }

    sequence_model = ConditionalSequenceGenerator(seq_config)
    counts_model = ConditionalCountsGenerator(counts_config)

    # Create dummy conditioning data
    conditioning_data = np.random.randn(3, 6)

    # Generate
    results_df = generate_sequences_and_counts(
        sequence_model,
        counts_model,
        conditioning_data,
        num_samples=5,
        verbose=True
    )

    # Print examples
    print_generation_examples(results_df)

    print("\n✓ Generation pipeline test complete!")
