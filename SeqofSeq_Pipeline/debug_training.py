"""
Debug script to check training data and model
"""
import torch
from preprocessing import load_preprocessed_data, create_dataloaders
from config import SEQUENCE_MODEL_CONFIG, DURATION_MODEL_CONFIG

# Load data
print("Loading data...")
df, metadata = load_preprocessed_data()

print(f"\nMetadata:")
print(f"  vocab_size: {metadata['vocab_size']}")
print(f"  num_conditioning_features: {metadata['num_conditioning_features']}")

# Update configs
SEQUENCE_MODEL_CONFIG['vocab_size'] = metadata['vocab_size']
SEQUENCE_MODEL_CONFIG['conditioning_dim'] = metadata['num_conditioning_features']

print(f"\nSequence Model Config:")
print(f"  vocab_size: {SEQUENCE_MODEL_CONFIG['vocab_size']}")
print(f"  conditioning_dim: {SEQUENCE_MODEL_CONFIG['conditioning_dim']}")
print(f"  d_model: {SEQUENCE_MODEL_CONFIG['d_model']}")

# Create dataloaders
train_loader, val_loader, scaler = create_dataloaders(df, metadata, batch_size=4)

# Get a batch
batch = next(iter(train_loader))

print(f"\nBatch shapes:")
for key, value in batch.items():
    print(f"  {key}: {value.shape}")

print(f"\nBatch values:")
print(f"  conditioning min/max: {batch['conditioning'].min():.3f} / {batch['conditioning'].max():.3f}")
print(f"  sequence_tokens unique values: {batch['sequence_tokens'].unique()}")
print(f"  sequence_tokens min/max: {batch['sequence_tokens'].min()} / {batch['sequence_tokens'].max()}")
print(f"  durations min/max: {batch['durations'].min():.1f} / {batch['durations'].max():.1f}")

# Test model creation
from models import ConditionalSequenceGenerator

print(f"\nCreating model...")
model = ConditionalSequenceGenerator(SEQUENCE_MODEL_CONFIG)
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
print(f"\nTesting forward pass...")
try:
    conditioning = batch['conditioning'][:2]
    sequence_tokens = batch['sequence_tokens'][:2]

    # Create decoder input
    start_token = 1  # START_TOKEN_ID
    batch_size, seq_len = sequence_tokens.shape
    start_tokens = torch.full((batch_size, 1), start_token, dtype=sequence_tokens.dtype)
    decoder_input = torch.cat([start_tokens, sequence_tokens[:, :-1]], dim=1)

    print(f"  Input shapes: conditioning={conditioning.shape}, decoder_input={decoder_input.shape}")

    with torch.no_grad():
        logits = model(conditioning, decoder_input)

    print(f"  Output shape: {logits.shape}")
    print(f"  Output min/max: {logits.min():.3f} / {logits.max():.3f}")
    print(f"  Output has NaN: {torch.isnan(logits).any()}")
    print(f"\n✓ Forward pass successful!")

except Exception as e:
    print(f"\n✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
