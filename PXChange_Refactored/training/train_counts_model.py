"""
Training script for Conditional Counts Generator
"""
import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    COUNTS_MODEL_CONFIG, COUNTS_TRAINING_CONFIG,
    MODEL_SAVE_DIR
)
from models import ConditionalCountsGenerator


def calculate_metrics(mu, sigma, targets, mask=None):
    """
    Calculate evaluation metrics for count predictions.
    """
    with torch.no_grad():
        if mask is not None:
            mu_masked = mu[mask]
            targets_masked = targets[mask]
        else:
            mu_masked = mu.reshape(-1)
            targets_masked = targets.reshape(-1)

        # MAE
        mae = torch.abs(mu_masked - targets_masked).mean().item()

        # RMSE
        rmse = torch.sqrt(((mu_masked - targets_masked) ** 2).mean()).item()

        # MAPE
        mape = (torch.abs((targets_masked - mu_masked) / (targets_masked + 1e-8))).mean().item() * 100

    return mae, rmse, mape


def train_epoch(model, dataloader, optimizer, device):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0
    total_mae = 0
    total_rmse = 0
    total_mape = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:
        # Move data to device
        conditioning = batch['conditioning'].to(device)
        sequence_tokens = batch['sequence_tokens'].to(device)
        sequence_features = batch['sequence_features'].to(device)
        step_durations = batch['step_durations'].to(device)
        mask = batch['mask'].to(device)

        # Forward pass
        mu, sigma = model(conditioning, sequence_tokens, sequence_features, mask)

        # Compute loss
        loss = model.compute_loss(mu, sigma, step_durations, mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # Metrics
        mae, rmse, mape = calculate_metrics(mu, sigma, step_durations, mask)

        total_loss += loss.item()
        total_mae += mae
        total_rmse += rmse
        total_mape += mape
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mae': f'{mae:.2f}',
            'rmse': f'{rmse:.2f}'
        })

    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    avg_rmse = total_rmse / num_batches
    avg_mape = total_mape / num_batches

    return avg_loss, avg_mae, avg_rmse, avg_mape


@torch.no_grad()
def validate_epoch(model, dataloader, device):
    """
    Validate for one epoch.
    """
    model.eval()
    total_loss = 0
    total_mae = 0
    total_rmse = 0
    total_mape = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validation"):
        # Move data to device
        conditioning = batch['conditioning'].to(device)
        sequence_tokens = batch['sequence_tokens'].to(device)
        sequence_features = batch['sequence_features'].to(device)
        step_durations = batch['step_durations'].to(device)
        mask = batch['mask'].to(device)

        # Forward pass
        mu, sigma = model(conditioning, sequence_tokens, sequence_features, mask)

        # Compute loss
        loss = model.compute_loss(mu, sigma, step_durations, mask)

        # Metrics
        mae, rmse, mape = calculate_metrics(mu, sigma, step_durations, mask)

        total_loss += loss.item()
        total_mae += mae
        total_rmse += rmse
        total_mape += mape
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    avg_rmse = total_rmse / num_batches
    avg_mape = total_mape / num_batches

    return avg_loss, avg_mae, avg_rmse, avg_mape


def plot_training_curves(history, save_path):
    """
    Plot and save training curves.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (NLL)')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # MAE
    axes[0, 1].plot(history['train_mae'], label='Train')
    axes[0, 1].plot(history['val_mae'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE (seconds)')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # RMSE
    axes[1, 0].plot(history['train_rmse'], label='Train')
    axes[1, 0].plot(history['val_rmse'], label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE (seconds)')
    axes[1, 0].set_title('Root Mean Squared Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # MAPE
    axes[1, 1].plot(history['train_mape'], label='Train')
    axes[1, 1].plot(history['val_mape'], label='Validation')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAPE (%)')
    axes[1, 1].set_title('Mean Absolute Percentage Error')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Training curves saved to {save_path}")


def train_counts_model(train_loader, val_loader, config=None, training_config=None):
    """
    Main training function for counts model.

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
        config = COUNTS_MODEL_CONFIG
    if training_config is None:
        training_config = COUNTS_TRAINING_CONFIG

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Initialize model
    model = ConditionalCountsGenerator(config)
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = Adam(model.parameters(), lr=training_config['learning_rate'])

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
        'train_rmse': [],
        'val_rmse': [],
        'train_mape': [],
        'val_mape': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    patience = training_config.get('early_stopping_patience', 15)

    print(f"\n{'='*70}")
    print(f"TRAINING CONDITIONAL COUNTS GENERATOR")
    print(f"{'='*70}\n")

    for epoch in range(training_config['epochs']):
        print(f"\nEpoch {epoch + 1}/{training_config['epochs']}")

        # Train
        train_loss, train_mae, train_rmse, train_mape = train_epoch(
            model, train_loader, optimizer, device
        )

        # Validate
        val_loss, val_mae, val_rmse, val_mape = validate_epoch(
            model, val_loader, device
        )

        # Update scheduler
        scheduler.step(val_loss)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)
        history['train_mape'].append(train_mape)
        history['val_mape'].append(val_mape)

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, MAE: {train_mae:.2f}s, RMSE: {train_rmse:.2f}s, MAPE: {train_mape:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.2f}s, RMSE: {val_rmse:.2f}s, MAPE: {val_mape:.2f}%")

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
            }, os.path.join(MODEL_SAVE_DIR, 'counts_model_best.pt'))
            print(f"  [OK] Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\n[OK] Early stopping triggered after {epoch + 1} epochs")
            break

    # Load best model
    checkpoint = torch.load(os.path.join(MODEL_SAVE_DIR, 'counts_model_best.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Plot training curves
    plot_training_curves(history, os.path.join(MODEL_SAVE_DIR, 'counts_training_curves.png'))

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}\n")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation MAE: {history['val_mae'][-1]:.2f}s")
    print(f"Final validation RMSE: {history['val_rmse'][-1]:.2f}s")

    return model, history


if __name__ == "__main__":
    # Test training with dummy data
    print("Testing counts model training...")

    from preprocessing import load_preprocessed_data, create_dataloaders

    # Load data
    df = load_preprocessed_data()
    train_loader, val_loader, scaler = create_dataloaders(df, batch_size=8)

    # Test config
    test_config = {
        'd_model': 128,
        'nhead': 4,
        'num_encoder_layers': 2,
        'num_cross_attention_layers': 2,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'max_seq_len': 128,
        'conditioning_dim': 6,
        'sequence_feature_dim': 18,
        'min_sigma': 0.1
    }

    test_training_config = {
        'batch_size': 8,
        'epochs': 2,
        'learning_rate': 0.0001,
        'gradient_clip': 1.0,
        'early_stopping_patience': 5,
        'min_sigma': 0.1
    }

    # Train
    model, history = train_counts_model(train_loader, val_loader, test_config, test_training_config)
    print("\n[OK] Training test complete!")
