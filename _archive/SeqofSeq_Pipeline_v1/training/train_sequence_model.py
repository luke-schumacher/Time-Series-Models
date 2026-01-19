"""
Training script for Conditional Sequence Generator
"""
import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SEQUENCE_MODEL_CONFIG, SEQUENCE_TRAINING_CONFIG,
    MODEL_SAVE_DIR, START_TOKEN_ID, PAD_TOKEN_ID
)
from models import ConditionalSequenceGenerator


def create_decoder_input(sequence_tokens, start_token_id):
    """
    Create decoder input by prepending START token and removing last token.

    Args:
        sequence_tokens: [batch_size, seq_len] - target tokens
        start_token_id: ID of the START token

    Returns:
        decoder_input: [batch_size, seq_len] - shifted input for decoder
    """
    batch_size, seq_len = sequence_tokens.shape
    start_tokens = torch.full((batch_size, 1), start_token_id, dtype=sequence_tokens.dtype, device=sequence_tokens.device)
    decoder_input = torch.cat([start_tokens, sequence_tokens[:, :-1]], dim=1)
    return decoder_input


def get_warmup_scheduler(optimizer, warmup_steps):
    """
    Learning rate scheduler with warmup.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def calculate_perplexity(logits, targets, ignore_index=PAD_TOKEN_ID):
    """
    Calculate perplexity metric.
    """
    with torch.no_grad():
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)

        loss = nn.functional.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=ignore_index,
            reduction='mean'
        )

        perplexity = torch.exp(loss)

    return perplexity.item()


def calculate_accuracy(logits, targets, ignore_index=PAD_TOKEN_ID):
    """
    Calculate token prediction accuracy.
    """
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        mask = targets != ignore_index
        correct = (predictions == targets) & mask
        accuracy = correct.sum().float() / mask.sum().float()

    return accuracy.item()


def train_epoch(model, dataloader, optimizer, scheduler, device, config):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0
    total_perplexity = 0
    total_accuracy = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:
        # Move data to device
        conditioning = batch['conditioning'].to(device)
        sequence_tokens = batch['sequence_tokens'].to(device)

        # Create decoder input (shift right with START token)
        decoder_input = create_decoder_input(sequence_tokens, start_token_id=START_TOKEN_ID)

        # Forward pass
        logits = model(conditioning, decoder_input)

        # Compute loss
        loss = model.compute_loss(
            logits,
            sequence_tokens,
            ignore_index=PAD_TOKEN_ID,
            label_smoothing=config.get('label_smoothing', 0.1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if config.get('gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])

        optimizer.step()
        scheduler.step()

        # Metrics
        with torch.no_grad():
            perplexity = calculate_perplexity(logits, sequence_tokens)
            accuracy = calculate_accuracy(logits, sequence_tokens)

        total_loss += loss.item()
        total_perplexity += perplexity
        total_accuracy += accuracy
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ppl': f'{perplexity:.2f}',
            'acc': f'{accuracy:.3f}'
        })

    avg_loss = total_loss / num_batches
    avg_perplexity = total_perplexity / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_perplexity, avg_accuracy


@torch.no_grad()
def validate_epoch(model, dataloader, device):
    """
    Validate for one epoch.
    """
    model.eval()
    total_loss = 0
    total_perplexity = 0
    total_accuracy = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validation"):
        # Move data to device
        conditioning = batch['conditioning'].to(device)
        sequence_tokens = batch['sequence_tokens'].to(device)

        # Create decoder input
        decoder_input = create_decoder_input(sequence_tokens, start_token_id=START_TOKEN_ID)

        # Forward pass
        logits = model(conditioning, decoder_input)

        # Compute loss
        loss = model.compute_loss(logits, sequence_tokens, ignore_index=PAD_TOKEN_ID)

        # Metrics
        perplexity = calculate_perplexity(logits, sequence_tokens)
        accuracy = calculate_accuracy(logits, sequence_tokens)

        total_loss += loss.item()
        total_perplexity += perplexity
        total_accuracy += accuracy
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_perplexity = total_perplexity / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_perplexity, avg_accuracy


def plot_training_curves(history, save_path):
    """
    Plot and save training curves.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Perplexity
    axes[1].plot(history['train_perplexity'], label='Train')
    axes[1].plot(history['val_perplexity'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('Perplexity')
    axes[1].legend()
    axes[1].grid(True)

    # Accuracy
    axes[2].plot(history['train_accuracy'], label='Train')
    axes[2].plot(history['val_accuracy'], label='Validation')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Token Accuracy')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Training curves saved to {save_path}")


def train_sequence_model(train_loader, val_loader, config=None, training_config=None):
    """
    Main training function for sequence model.

    Args:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Model configuration
        training_config: Training configuration

    Returns:
        model: Trained model
        history: Training history
    """
    if config is None:
        config = SEQUENCE_MODEL_CONFIG
    if training_config is None:
        training_config = SEQUENCE_TRAINING_CONFIG

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Initialize model
    model = ConditionalSequenceGenerator(config)
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = Adam(model.parameters(), lr=training_config['learning_rate'])

    # Learning rate scheduler
    total_steps = len(train_loader) * training_config['epochs']
    warmup_steps = training_config.get('warmup_steps', 4000)
    scheduler = get_warmup_scheduler(optimizer, warmup_steps)

    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_perplexity': [],
        'val_perplexity': [],
        'train_accuracy': [],
        'val_accuracy': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    patience = training_config.get('early_stopping_patience', 15)

    print(f"\n{'='*70}")
    print(f"TRAINING CONDITIONAL SEQUENCE GENERATOR")
    print(f"{'='*70}\n")

    for epoch in range(training_config['epochs']):
        print(f"\nEpoch {epoch + 1}/{training_config['epochs']}")

        # Train
        train_loss, train_ppl, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, training_config
        )

        # Validate
        val_loss, val_ppl, val_acc = validate_epoch(model, val_loader, device)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_perplexity'].append(train_ppl)
        history['val_perplexity'].append(val_ppl)
        history['train_accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, Perplexity: {train_ppl:.2f}, Accuracy: {train_acc:.3f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}, Accuracy: {val_acc:.3f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, os.path.join(MODEL_SAVE_DIR, 'sequence_model_best.pt'))
            print(f"  [OK] Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\n[OK] Early stopping triggered after {epoch + 1} epochs")
            break

    # Load best model if it exists
    best_model_path = os.path.join(MODEL_SAVE_DIR, 'sequence_model_best.pt')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n[OK] Loaded best model from epoch {checkpoint['epoch']}")
    else:
        print(f"\n[WARN] Best model not found, using final model state")
        # Save current model as best
        torch.save({
            'epoch': training_config['epochs'],
            'model_state_dict': model.state_dict(),
            'config': config
        }, best_model_path)

    # Plot training curves
    plot_training_curves(history, os.path.join(MODEL_SAVE_DIR, 'sequence_training_curves.png'))

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}\n")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return model, history


if __name__ == "__main__":
    # Test training with dummy data
    print("Testing sequence model training...")

    from preprocessing import load_preprocessed_data, create_dataloaders

    # Load data
    df = load_preprocessed_data()
    train_loader, val_loader, scaler = create_dataloaders(df, batch_size=8)

    # Test config
    test_config = {
        'vocab_size': 18,
        'd_model': 128,
        'nhead': 4,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'max_seq_len': 128,
        'conditioning_dim': 6
    }

    test_training_config = {
        'batch_size': 8,
        'epochs': 2,
        'learning_rate': 0.0001,
        'warmup_steps': 100,
        'label_smoothing': 0.1,
        'gradient_clip': 1.0,
        'early_stopping_patience': 5
    }

    # Train
    model, history = train_sequence_model(train_loader, val_loader, test_config, test_training_config)
    print("\n[OK] Training test complete!")
