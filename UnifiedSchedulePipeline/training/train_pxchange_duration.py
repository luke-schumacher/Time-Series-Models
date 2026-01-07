"""
Train PXChange Duration Model from Scratch

Trains the ConditionalCountsGenerator to predict step durations
conditioned on sequences and patient/scan context.
Predicts Gamma distribution parameters (μ, σ) for each time step.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Add paths
unified_dir = os.path.join(os.path.dirname(__file__), '..', '..')
pxchange_dir = os.path.join(unified_dir, '..', 'PXChange_Refactored')
sys.path.insert(0, unified_dir)
sys.path.insert(0, pxchange_dir)

from config import MODEL_PATHS, RANDOM_SEED
from data_loaders.pxchange_data_loader import create_pxchange_dataloaders
from PXChange_Refactored.models.conditional_counts_generator import ConditionalCountsGenerator
from PXChange_Refactored.config import COUNTS_MODEL_CONFIG


def gamma_nll_loss(mu, sigma, target, mask, eps=1e-6):
    """
    Gamma Negative Log-Likelihood Loss.

    Parameterization: Gamma(α, β) where:
        α = (μ / σ)^2
        β = μ / σ^2

    Args:
        mu: Predicted mean [batch, seq_len]
        sigma: Predicted std [batch, seq_len]
        target: Ground truth durations [batch, seq_len]
        mask: Valid positions [batch, seq_len]
        eps: Small constant for numerical stability

    Returns:
        loss: Scalar loss
    """
    # Ensure positive values
    mu = torch.clamp(mu, min=eps)
    sigma = torch.clamp(sigma, min=eps)
    target = torch.clamp(target, min=eps)

    # Gamma parameters
    alpha = (mu / sigma) ** 2
    beta = mu / (sigma ** 2)

    # Gamma NLL: -log p(target | α, β)
    # log p(x) = (α-1)log(x) - βx - log(Γ(α)) + α*log(β)
    log_prob = (alpha - 1) * torch.log(target + eps) - beta * target - torch.lgamma(alpha) + alpha * torch.log(beta + eps)
    nll = -log_prob

    # Apply mask and average
    masked_nll = nll * mask
    loss = masked_nll.sum() / (mask.sum() + eps)

    return loss


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_tokens = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        # Move to device
        conditioning = batch['conditioning'].to(device)
        sequence_tokens = batch['sequence_tokens'].to(device)
        sequence_features = batch['sequence_features'].to(device)
        step_durations = batch['step_durations'].to(device)
        mask = batch['mask'].to(device)

        # Forward pass
        optimizer.zero_grad()
        mu, sigma = model(conditioning, sequence_tokens, sequence_features)  # [batch, seq_len]

        # Compute loss
        loss = gamma_nll_loss(mu, sigma, step_durations, mask)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Statistics
        num_valid = mask.sum().item()
        total_loss += loss.item() * num_valid
        total_tokens += num_valid

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / total_tokens
    return avg_loss


def validate(model, val_loader, device, epoch):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for batch in pbar:
            # Move to device
            conditioning = batch['conditioning'].to(device)
            sequence_tokens = batch['sequence_tokens'].to(device)
            sequence_features = batch['sequence_features'].to(device)
            step_durations = batch['step_durations'].to(device)
            mask = batch['mask'].to(device)

            # Forward pass
            mu, sigma = model(conditioning, sequence_tokens, sequence_features)

            # Compute loss
            loss = gamma_nll_loss(mu, sigma, step_durations, mask)

            num_valid = mask.sum().item()
            total_loss += loss.item() * num_valid
            total_tokens += num_valid

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / total_tokens
    return avg_loss


def train_pxchange_duration_model(
    batch_size=32,
    epochs=100,
    learning_rate=0.0001,
    validation_split=0.2,
    early_stopping_patience=15,
    save_dir=None,
    device=None
):
    """
    Train PXChange duration model from scratch.

    Args:
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        learning_rate: Learning rate for Adam optimizer
        validation_split: Fraction of data for validation
        early_stopping_patience: Number of epochs without improvement before stopping
        save_dir: Directory to save model checkpoints
        device: Device to train on (cuda/cpu)

    Returns:
        model: Trained model
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Save directory
    if save_dir is None:
        save_dir = MODEL_PATHS['pxchange']['model_dir']
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("TRAINING PXCHANGE DURATION MODEL")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading data...")
    train_loader, val_loader, scaler, metadata = create_pxchange_dataloaders(
        batch_size=batch_size,
        validation_split=validation_split
    )

    print(f"\n  Vocabulary size: {metadata['vocab_size']}")
    print(f"  Conditioning dimension: {metadata['conditioning_dim']}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")

    # Create model
    print("\n[2/5] Creating model...")
    model_config = COUNTS_MODEL_CONFIG.copy()
    model_config['conditioning_dim'] = metadata['conditioning_dim']

    model = ConditionalCountsGenerator(config=model_config)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")

    # Optimizer and scheduler
    print("\n[3/5] Setting up training...")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    print("\n[4/5] Training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, device, epoch)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print epoch summary
        print(f"\nEpoch {epoch}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            checkpoint_path = os.path.join(save_dir, 'duration_model_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model_config': model_config,
                'scaler': scaler
            }, checkpoint_path)
            print(f"  [SAVED] Best model: {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{early_stopping_patience}")

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n[EARLY STOPPING] No improvement for {early_stopping_patience} epochs")
            break

    elapsed_time = time.time() - start_time

    print("\n[5/5] Training complete!")
    print(f"  Total time: {elapsed_time/60:.2f} minutes")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    print(f"  Model saved to: {save_dir}")

    print("\n" + "=" * 70)

    return model, train_losses, val_losses


if __name__ == "__main__":
    # Train model
    model, train_losses, val_losses = train_pxchange_duration_model(
        batch_size=32,
        epochs=100,
        learning_rate=0.0001,
        early_stopping_patience=15
    )

    print("\n[SUCCESS] PXChange duration model training complete!")
