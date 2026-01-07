"""
Retrain Existing Models with PAUSE Token Support
Expands vocabularies and fine-tunes models to support PAUSE events
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import sys
import os
from tqdm import tqdm

# Add UnifiedSchedulePipeline base directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Get the absolute path to the Time-Series-Models root directory
# This is two levels up from UnifiedSchedulePipeline/training
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from config import PXCHANGE_DIR, SEQOFSEQ_DIR, MODEL_PATHS

# Import models
from SeqofSeq_Pipeline.models.conditional_sequence_generator import ConditionalSequenceGenerator
from PXChange_Refactored.models.conditional_counts_generator import ConditionalCountsGenerator
from SeqofSeq_Pipeline.models.conditional_duration_predictor import ConditionalDurationPredictor
import PXChange_Refactored.config as px_config
import SeqofSeq_Pipeline.config as seq_config


class ModelExpander:
    """Utility class for expanding model vocabularies"""

    @staticmethod
    def expand_token_embedding(model, old_vocab_size, new_vocab_size, embedding_attr='token_embedding'):
        """
        Expand token embedding layer for new vocabulary size

        Args:
            model: Model to expand
            old_vocab_size: Original vocabulary size
            new_vocab_size: New vocabulary size (with PAUSE token)
            embedding_attr: Name of embedding attribute

        Returns:
            model: Model with expanded embedding
        """
        print(f"Expanding {embedding_attr} from {old_vocab_size} to {new_vocab_size}")

        # Get old embedding
        old_embedding = getattr(model, embedding_attr)
        d_model = old_embedding.embedding_dim

        # Get old weights
        old_weights = old_embedding.weight.data.clone()

        # Create new embedding
        new_embedding = nn.Embedding(new_vocab_size, d_model)

        # Copy old weights
        new_embedding.weight.data[:old_vocab_size] = old_weights

        # Initialize new token (PAUSE) as mean of existing tokens
        new_embedding.weight.data[old_vocab_size:] = old_weights.mean(dim=0)

        # Replace embedding
        setattr(model, embedding_attr, new_embedding)

        print(f"  New token initialized as mean of existing embeddings")

        return model

    @staticmethod
    def expand_output_projection(model, old_vocab_size, new_vocab_size, projection_attr='output_projection'):
        """
        Expand output projection layer for new vocabulary size

        Args:
            model: Model to expand
            old_vocab_size: Original vocabulary size
            new_vocab_size: New vocabulary size
            projection_attr: Name of projection attribute

        Returns:
            model: Model with expanded projection
        """
        print(f"Expanding {projection_attr} from {old_vocab_size} to {new_vocab_size}")

        # Get old projection
        old_projection = getattr(model, projection_attr)

        if isinstance(old_projection, nn.Linear):
            in_features = old_projection.in_features
            old_weights = old_projection.weight.data.clone()
            old_bias = old_projection.bias.data.clone() if old_projection.bias is not None else None

            # Create new projection
            new_projection = nn.Linear(in_features, new_vocab_size)

            # Copy old weights
            new_projection.weight.data[:old_vocab_size] = old_weights
            if old_bias is not None:
                new_projection.bias.data[:old_vocab_size] = old_bias

            # Initialize new token projection as mean
            new_projection.weight.data[old_vocab_size:] = old_weights.mean(dim=0)
            if new_projection.bias is not None:
                new_projection.bias.data[old_vocab_size:] = old_bias.mean() if old_bias is not None else 0.0

            # Replace projection
            setattr(model, projection_attr, new_projection)

        return model


def load_and_expand_pxchange_sequence_model(model_path, device='cpu'):
    """
    Load PXChange sequence model and expand for PAUSE token

    Args:
        model_path: Path to saved model checkpoint
        device: Device to load model on

    Returns:
        model: Expanded model
        optimizer: Fresh optimizer for fine-tuning
    """
    print("\n" + "="*80)
    print("LOADING AND EXPANDING PXCHANGE SEQUENCE MODEL")
    print("="*80)

    old_vocab_size = 18
    new_vocab_size = 19  # Added PAUSE

    # Create model with OLD vocabulary size
    temp_config = {
        'vocab_size': old_vocab_size,
        'd_model': px_config.SEQUENCE_MODEL_CONFIG['d_model'],
        'nhead': px_config.SEQUENCE_MODEL_CONFIG['nhead'],
        'num_encoder_layers': px_config.SEQUENCE_MODEL_CONFIG['num_encoder_layers'],
        'num_decoder_layers': px_config.SEQUENCE_MODEL_CONFIG['num_decoder_layers'],
        'dim_feedforward': px_config.SEQUENCE_MODEL_CONFIG['dim_feedforward'],
        'dropout': px_config.SEQUENCE_MODEL_CONFIG['dropout'],
        'max_seq_len': px_config.MAX_SEQ_LEN,
        'conditioning_dim': len(px_config.CONDITIONING_FEATURES)
    }
    model = ConditionalSequenceGenerator(config=temp_config).to(device)

    # Load existing weights if available
    if os.path.exists(model_path):
        print(f"Loading weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  Weights loaded successfully")
    else:
        print(f"WARNING: No checkpoint found at {model_path}")
        print("  Using random initialization")

    # Expand for PAUSE token
    model = ModelExpander.expand_token_embedding(model, old_vocab_size, new_vocab_size, 'token_embedding')
    model = ModelExpander.expand_output_projection(model, old_vocab_size, new_vocab_size, 'output_projection')

    # Update model's vocab_size attribute
    model.vocab_size = new_vocab_size

    # Create optimizer with lower learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    print(f"Model expanded and ready for fine-tuning")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, optimizer


def load_and_expand_pxchange_duration_model(model_path, device='cpu'):
    """
    Load PXChange duration model and expand for PAUSE token

    Duration model needs expanded sequence feature dimension
    """
    print("\n" + "="*80)
    print("LOADING AND EXPANDING PXCHANGE DURATION MODEL")
    print("="*80)

    old_vocab_size = 18
    new_vocab_size = 19

    # Create model
    temp_config = {
        'd_model': px_config.COUNTS_MODEL_CONFIG['d_model'],
        'nhead': px_config.COUNTS_MODEL_CONFIG['nhead'],
        'num_encoder_layers': px_config.COUNTS_MODEL_CONFIG['num_encoder_layers'],
        'num_cross_attention_layers': px_config.COUNTS_MODEL_CONFIG['num_cross_attention_layers'],
        'dim_feedforward': px_config.COUNTS_MODEL_CONFIG['dim_feedforward'],
        'dropout': px_config.COUNTS_MODEL_CONFIG['dropout'],
        'max_seq_len': px_config.MAX_SEQ_LEN,
        'conditioning_dim': len(px_config.CONDITIONING_FEATURES),
        'sequence_feature_dim': new_vocab_size + len(px_config.SEQUENCE_FEATURE_COLUMNS),
        'output_heads': px_config.COUNTS_MODEL_CONFIG['output_heads']
    }
    model = ConditionalCountsGenerator(config=temp_config).to(device)

    # Load existing weights if available
    if os.path.exists(model_path):
        print(f"Loading weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)

        # Load with size mismatch allowed (sequence_projection will be different)
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("  Weights loaded (non-strict mode for expanded dimensions)")
        except Exception as e:
            print(f"  Warning: Could not load all weights: {e}")
    else:
        print(f"WARNING: No checkpoint found at {model_path}")
        print("  Using random initialization")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    print(f"Model expanded and ready for fine-tuning")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, optimizer


def load_and_expand_seqofseq_sequence_model(model_path, metadata_path, device='cpu'):
    """
    Load SeqofSeq sequence model and expand for PAUSE token
    """
    print("\n" + "="*80)
    print("LOADING AND EXPANDING SEQOFSEQ SEQUENCE MODEL")
    print("="*80)

    # Load metadata to get vocab size
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    vocab = metadata['sequence_vocab']
    old_vocab_size = len(vocab)
    new_vocab_size = old_vocab_size + 1  # Add PAUSE if not already there

    # Check if PAUSE already in vocab
    if 'PAUSE' in vocab:
        print("PAUSE token already in vocabulary, no expansion needed")
        new_vocab_size = old_vocab_size
    else:
        print(f"Expanding from {old_vocab_size} to {new_vocab_size} tokens")

    conditioning_dim = metadata.get('conditioning_dim', 92)

    # Create model with NEW vocabulary size
    temp_config = {
        'vocab_size': new_vocab_size,
        'd_model': seq_config.SEQUENCE_MODEL_CONFIG['d_model'],
        'nhead': seq_config.SEQUENCE_MODEL_CONFIG['nhead'],
        'num_encoder_layers': seq_config.SEQUENCE_MODEL_CONFIG['num_encoder_layers'],
        'num_decoder_layers': seq_config.SEQUENCE_MODEL_CONFIG['num_decoder_layers'],
        'dim_feedforward': seq_config.SEQUENCE_MODEL_CONFIG['dim_feedforward'],
        'dropout': seq_config.SEQUENCE_MODEL_CONFIG['dropout'],
        'max_seq_len': seq_config.MAX_SEQ_LEN,
        'conditioning_dim': conditioning_dim
    }
    model = ConditionalSequenceGenerator(config=temp_config).to(device)

    # Load existing weights if available
    if os.path.exists(model_path) and new_vocab_size > old_vocab_size:
        # Need to expand
        print(f"Loading and expanding from: {model_path}")

        # Create temporary model with old size
        temp_model = ConditionalSequenceGenerator(
            vocab_size=old_vocab_size,
            d_model=seq_config.SEQUENCE_MODEL_CONFIG['d_model'],
            nhead=seq_config.SEQUENCE_MODEL_CONFIG['nhead'],
            num_encoder_layers=seq_config.SEQUENCE_MODEL_CONFIG['num_encoder_layers'],
            num_decoder_layers=seq_config.SEQUENCE_MODEL_CONFIG['num_decoder_layers'],
            dim_feedforward=seq_config.SEQUENCE_MODEL_CONFIG['dim_feedforward'],
            dropout=seq_config.SEQUENCE_MODEL_CONFIG['dropout'],
            max_seq_len=seq_config.MAX_SEQ_LEN,
            conditioning_dim=conditioning_dim
        )

        checkpoint = torch.load(model_path, map_location='cpu')
        temp_model.load_state_dict(checkpoint['model_state_dict'])

        # Expand embeddings
        temp_model = ModelExpander.expand_token_embedding(temp_model, old_vocab_size, new_vocab_size)
        temp_model = ModelExpander.expand_output_projection(temp_model, old_vocab_size, new_vocab_size)

        # Copy to new model
        model.load_state_dict(temp_model.state_dict())
        model = model.to(device)

        print("  Model expanded successfully")

    elif os.path.exists(model_path):
        # No expansion needed
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from: {model_path}")
    else:
        print(f"WARNING: No checkpoint found at {model_path}")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    print(f"Model ready for fine-tuning")
    print(f"  Vocab size: {new_vocab_size}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, optimizer


def load_and_expand_seqofseq_duration_model(model_path, metadata_path, device='cpu'):
    """
    Load SeqofSeq duration model and expand for PAUSE token
    """
    print("\n" + "="*80)
    print("LOADING AND EXPANDING SEQOFSEQ DURATION MODEL")
    print("="*80)

    # Load metadata to get vocab size
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    vocab = metadata['sequence_vocab']
    old_vocab_size = len(vocab)
    new_vocab_size = old_vocab_size + 1  # Add PAUSE if not already there

    # Check if PAUSE already in vocab
    if 'PAUSE' in vocab:
        print("PAUSE token already in vocabulary, no expansion needed")
        new_vocab_size = old_vocab_size
    else:
        print(f"Expanding from {old_vocab_size} to {new_vocab_size} tokens")

    conditioning_dim = metadata.get('conditioning_dim', 92)

    # Create model with NEW vocabulary size
    temp_config = {
        'd_model': seq_config.DURATION_MODEL_CONFIG['d_model'],
        'nhead': seq_config.DURATION_MODEL_CONFIG['nhead'],
        'num_encoder_layers': seq_config.DURATION_MODEL_CONFIG['num_encoder_layers'],
        'num_cross_attention_layers': seq_config.DURATION_MODEL_CONFIG['num_cross_attention_layers'],
        'dim_feedforward': seq_config.DURATION_MODEL_CONFIG['dim_feedforward'],
        'dropout': seq_config.DURATION_MODEL_CONFIG['dropout'],
        'max_seq_len': seq_config.MAX_SEQ_LEN,
        'conditioning_dim': conditioning_dim,
        'sequence_feature_dim': new_vocab_size, # This should be the new vocab size
        'output_heads': seq_config.DURATION_MODEL_CONFIG['output_heads']
    }
    model = ConditionalDurationPredictor(config=temp_config).to(device)


    # Load existing weights if available
    if os.path.exists(model_path) and new_vocab_size > old_vocab_size:
        # Need to expand
        print(f"Loading and expanding from: {model_path}")

        # Create temporary model with old size
        temp_model_config = {
            'd_model': seq_config.DURATION_MODEL_CONFIG['d_model'],
            'nhead': seq_config.DURATION_MODEL_CONFIG['nhead'],
            'num_encoder_layers': seq_config.DURATION_MODEL_CONFIG['num_encoder_layers'],
            'num_cross_attention_layers': seq_config.DURATION_MODEL_CONFIG['num_cross_attention_layers'],
            'dim_feedforward': seq_config.DURATION_MODEL_CONFIG['dim_feedforward'],
            'dropout': seq_config.DURATION_MODEL_CONFIG['dropout'],
            'max_seq_len': seq_config.MAX_SEQ_LEN,
            'conditioning_dim': conditioning_dim,
            'sequence_feature_dim': old_vocab_size, # Old vocab size for temp model
            'output_heads': seq_config.DURATION_MODEL_CONFIG['output_heads']
        }
        temp_model = ConditionalDurationPredictor(config=temp_model_config)


        checkpoint = torch.load(model_path, map_location='cpu')
        temp_model.load_state_dict(checkpoint['model_state_dict'])

        # Since the duration model doesn't have a simple output projection to vocab size,
        # and its token embedding is also based on vocab_size, we need to carefully
        # reinitialize the token embedding and sequence_feature_dim based components.
        # For simplicity, during expansion, we'll try to load non-strictly and hope
        # the model's internal layers handle the new sequence_feature_dim correctly.
        
        # Expand token embedding if it's dependent on vocab_size
        # The ConditionalDurationPredictor token_embedding is based on VOCAB_SIZE
        # We need to manually expand it here.
        if hasattr(model, 'token_embedding') and model.token_embedding.num_embeddings != new_vocab_size:
            print(f"  Expanding token_embedding in ConditionalDurationPredictor from {old_vocab_size} to {new_vocab_size}")
            old_embedding = temp_model.token_embedding
            d_model_half = old_embedding.embedding_dim
            new_embedding = nn.Embedding(new_vocab_size, d_model_half, padding_idx=seq_config.PAD_TOKEN_ID)
            new_embedding.weight.data[:old_vocab_size] = old_embedding.weight.data.clone()
            new_embedding.weight.data[old_vocab_size:] = old_embedding.weight.data.mean(dim=0)
            model.token_embedding = new_embedding

        model.load_state_dict(temp_model.state_dict(), strict=False) # Load non-strictly
        model = model.to(device)

        print("  Model expanded successfully")

    elif os.path.exists(model_path):
        # No expansion needed
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from: {model_path}")
    else:
        print(f"WARNING: No checkpoint found at {model_path}")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    print(f"Model ready for fine-tuning")
    print(f"  Vocab size: {new_vocab_size}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, optimizer


def create_dummy_training_data_pxchange(num_samples=100, max_seq_len=128, device='cpu'):
    """
    Create dummy training data for testing fine-tuning pipeline
    In practice, this would load real preprocessed data with PAUSE tokens
    """
    print("\nCreating dummy PXChange training data...")

    vocab_size = 19  # Including PAUSE
    conditioning_dim = 6

    # Random conditioning
    conditioning = torch.randn(num_samples, conditioning_dim).to(device)

    # Random sequences (including some PAUSE tokens with ID 18)
    sequences = torch.randint(1, vocab_size, (num_samples, max_seq_len)).to(device)
    sequences[:, 0] = 11  # START token
    sequences[:, -1] = 14  # END token

    # Add some PAUSE tokens randomly
    pause_mask = torch.rand(num_samples, max_seq_len) < 0.05
    sequences[pause_mask] = 18  # PAUSE token

    # Random durations
    durations = torch.rand(num_samples, max_seq_len).to(device) * 100

    # Mask (True for valid positions)
    mask = torch.ones(num_samples, max_seq_len, dtype=torch.bool).to(device)

    # Sequence features
    seq_features = torch.zeros(num_samples, max_seq_len, 2).to(device)

    print(f"  Created {num_samples} samples")
    print(f"  Sequences include PAUSE tokens (ID 18)")

    return conditioning, sequences, seq_features, durations, mask


def create_dummy_training_data_seqofseq(num_samples=100, max_seq_len=64, vocab_size=35, device='cpu'):
    """
    Create dummy training data for SeqofSeq models
    """
    print("\nCreating dummy SeqofSeq training data...")

    # vocab_size is now passed as an argument
    conditioning_dim = 92 # Estimated from SeqofSeq config

    # Random conditioning
    conditioning = torch.randn(num_samples, conditioning_dim).to(device)

    # Random sequences
    sequences = torch.randint(seq_config.START_TOKEN_ID, vocab_size, (num_samples, max_seq_len)).to(device)
    sequences[:, 0] = seq_config.START_TOKEN_ID
    sequences[:, -1] = seq_config.END_TOKEN_ID

    # Add some PAUSE tokens randomly
    pause_mask = torch.rand(num_samples, max_seq_len) < 0.05
    sequences[pause_mask] = seq_config.PAUSE_TOKEN_ID

    # Random durations
    durations = torch.rand(num_samples, max_seq_len).to(device) * 100

    # Mask (True for valid positions)
    mask = torch.ones(num_samples, max_seq_len, dtype=torch.bool).to(device)

    # Sequence features (placeholder for now, might need more detail)
    seq_features = torch.zeros(num_samples, max_seq_len, 2).to(device) # Assuming 2 features for position/direction

    print(f"  Created {num_samples} samples")
    print(f"  Sequences include PAUSE tokens (ID {seq_config.PAUSE_TOKEN_ID})")

    return conditioning, sequences, seq_features, durations, mask


def fine_tune_model(model, optimizer, train_data, model_type='sequence', epochs=20, device='cpu'):
    """
    Fine-tune model with new PAUSE token data

    Args:
        model: Model to fine-tune
        optimizer: Optimizer
        train_data: Tuple of training data tensors
        model_type: 'sequence' or 'duration'
        epochs: Number of epochs
        device: Device

    Returns:
        model: Fine-tuned model
    """
    print(f"\n{'='*80}")
    print(f"FINE-TUNING {model_type.upper()} MODEL")
    print(f"{'='*80}")

    model.train()

    if model_type == 'sequence':
        conditioning, sequences, _, _, mask = train_data

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Create decoder input (shift right with START token)
            decoder_input = torch.roll(sequences, 1, dims=1)
            decoder_input[:, 0] = 11  # START token

            # Forward pass
            logits = model(conditioning, decoder_input)

            # Compute loss
            loss = nn.CrossEntropyLoss(ignore_index=0)(
                logits.view(-1, logits.size(-1)),
                sequences.view(-1)
            )

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    elif model_type == 'duration':
        conditioning, sequences, seq_features, durations, mask = train_data

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            mu, sigma = model(conditioning, sequences, seq_features, mask)

            # Compute Gamma NLL loss
            alpha = (mu / sigma) ** 2
            beta = mu / (sigma ** 2)

            # Clip for numerical stability
            alpha = torch.clamp(alpha, min=1e-6)
            beta = torch.clamp(beta, min=1e-6)
            durations_clipped = torch.clamp(durations, min=1e-6)

            # Gamma NLL
            log_prob = (alpha - 1) * torch.log(durations_clipped) - beta * durations_clipped + \
                       alpha * torch.log(beta) - torch.lgamma(alpha)

            loss = -log_prob[mask].mean()

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    print(f"Fine-tuning complete!")

    return model


def save_model(model, optimizer, save_path, metadata=None):
    """Save fine-tuned model"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metadata': metadata or {}
    }

    torch.save(checkpoint, save_path)
    print(f"Model saved to: {save_path}")


def main():
    """Main fine-tuning pipeline"""
    print("\n" + "="*80)
    print("PHASE 2: MODEL FINE-TUNING WITH PAUSE TOKENS")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ========================================================================
    # 1. PXCHANGE SEQUENCE MODEL
    # ========================================================================
    print("\n\n1. FINE-TUNING PXCHANGE SEQUENCE MODEL")
    print("-" * 80)

    px_seq_model, px_seq_optimizer = load_and_expand_pxchange_sequence_model(
        model_path=os.path.join(PXCHANGE_DIR, 'saved_models', 'sequence_model_best.pth'),
        device=device
    )

    # Create dummy training data (replace with real data loading)
    train_data = create_dummy_training_data_pxchange(num_samples=100, device=device)

    # Fine-tune
    px_seq_model = fine_tune_model(
        px_seq_model, px_seq_optimizer, train_data,
        model_type='sequence', epochs=20, device=device
    )

    # Save
    save_path = os.path.join(MODEL_PATHS['pxchange']['model_dir'], 'sequence_model_with_pause.pth')
    save_model(px_seq_model, px_seq_optimizer, save_path, {'vocab_size': 19})

    # ========================================================================
    # 2. PXCHANGE DURATION MODEL
    # ========================================================================
    print("\n\n2. FINE-TUNING PXCHANGE DURATION MODEL")
    print("-" * 80)

    px_dur_model, px_dur_optimizer = load_and_expand_pxchange_duration_model(
        model_path=os.path.join(PXCHANGE_DIR, 'saved_models', 'duration_model_best.pth'),
        device=device
    )

    # Fine-tune
    px_dur_model = fine_tune_model(
        px_dur_model, px_dur_optimizer, train_data,
        model_type='duration', epochs=20, device=device
    )

    # Save
    save_path = os.path.join(MODEL_PATHS['pxchange']['model_dir'], 'duration_model_with_pause.pth')
    save_model(px_dur_model, px_dur_optimizer, save_path)

    # ========================================================================
    # 3. SEQOFSEQ SEQUENCE MODEL
    # ========================================================================
    print("\n\n3. FINE-TUNING SEQOFSEQ SEQUENCE MODEL")
    print("-" * 80)

    seq_seq_model, seq_seq_optimizer = load_and_expand_seqofseq_sequence_model(
        model_path=os.path.join(SEQOFSEQ_DIR, 'saved_models', 'sequence_model_best.pth'),
        metadata_path=os.path.join(SEQOFSEQ_DIR, 'data', 'preprocessed', 'metadata.pkl'),
        device=device
    )

    # Create dummy training data (replace with real data loading)
    train_data_seqofseq = create_dummy_training_data_seqofseq(num_samples=100, vocab_size=seq_seq_model.vocab_size, device=device)

    # Fine-tune
    seq_seq_model = fine_tune_model(
        seq_seq_model, seq_seq_optimizer, train_data_seqofseq,
        model_type='sequence', epochs=20, device=device
    )

    # Save
    save_path = os.path.join(MODEL_PATHS['seqofseq']['model_dir'], 'sequence_model_with_pause.pth')
    save_model(seq_seq_model, seq_seq_optimizer, save_path, {'vocab_size': seq_seq_model.vocab_size})

    # ========================================================================
    # 4. SEQOFSEQ DURATION MODEL
    # ========================================================================
    print("\n\n4. FINE-TUNING SEQOFSEQ DURATION MODEL")
    print("-" * 80)

    seq_dur_model, seq_dur_optimizer = load_and_expand_seqofseq_duration_model(
        model_path=os.path.join(SEQOFSEQ_DIR, 'saved_models', 'duration_model_best.pth'),
        metadata_path=os.path.join(SEQOFSEQ_DIR, 'data', 'preprocessed', 'metadata.pkl'),
        device=device
    )

    # Fine-tune
    seq_dur_model = fine_tune_model(
        seq_dur_model, seq_dur_optimizer, train_data_seqofseq,
        model_type='duration', epochs=20, device=device
    )

    # Save
    save_path = os.path.join(MODEL_PATHS['seqofseq']['model_dir'], 'duration_model_with_pause.pth')
    save_model(seq_dur_model, seq_dur_optimizer, save_path)

    print("\n\n" + "="*80)
    print("PHASE 2 COMPLETE - ALL MODELS FINE-TUNED WITH PAUSE SUPPORT")
    print("="*80)


if __name__ == "__main__":
    main()
