"""
Train Temporal Schedule Model from Scratch

Trains the TemporalScheduleModel to predict daily session structure:
1. Session count (Poisson distribution)
2. Session start times (Mixture of Gaussians)

This is the HIGHEST RISK component - no existing reference implementation.
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
sys.path.insert(0, unified_dir)

from config import MODEL_PATHS, RANDOM_SEED, TEMPORAL_TRAINING_CONFIG, TEMPORAL_MODEL_CONFIG
from data_loaders.temporal_data_extractor import prepare_temporal_training_data
from datetime_model.temporal_schedule_model import TemporalScheduleModel


def poisson_nll_loss(lambda_pred, target_counts, eps=1e-6):
    """
    Poisson Negative Log-Likelihood Loss for session count prediction.

    Args:
        lambda_pred: Predicted Poisson lambda [batch_size]
        target_counts: Ground truth session counts [batch_size]
        eps: Small constant for numerical stability

    Returns:
        loss: Scalar loss
    """
    lambda_pred = torch.clamp(lambda_pred, min=eps)
    target_counts = target_counts.float()

    # Poisson NLL: λ - k*log(λ) + log(k!)
    # We drop log(k!) as it doesn't depend on parameters
    nll = lambda_pred - target_counts * torch.log(lambda_pred + eps)
    loss = nll.mean()

    return loss


def mixture_gaussian_nll_loss(timing_params, target_start_times, num_valid_sessions, eps=1e-6):
    """
    Mixture of Gaussians Negative Log-Likelihood Loss for session start times.

    Args:
        timing_params: Dict with 'means' [B, K], 'stds' [B, K], 'weights' [B, K]
        target_start_times: Ground truth start times [B, max_sessions]
        num_valid_sessions: Number of valid sessions per sample [B]
        eps: Small constant for numerical stability

    Returns:
        loss: Scalar loss
    """
    means = timing_params['means']  # [B, K]
    stds = timing_params['stds']  # [B, K]
    weights = timing_params['weights']  # [B, K]

    batch_size = means.shape[0]
    num_components = means.shape[1]
    max_sessions = target_start_times.shape[1]

    total_loss = 0.0
    total_valid = 0

    # For each sample in batch
    for b in range(batch_size):
        num_valid = int(num_valid_sessions[b].item())
        if num_valid == 0:
            continue

        # Get valid start times for this sample
        valid_times = target_start_times[b, :num_valid]  # [num_valid]

        # Compute log probability for each start time under each component
        for t in valid_times:
            # Compute log probability under each Gaussian component
            component_log_probs = []
            for k in range(num_components):
                mean = means[b, k]
                std = stds[b, k]
                weight = weights[b, k]

                # Gaussian log probability
                log_prob = -0.5 * torch.log(2 * np.pi * std**2 + eps)
                log_prob -= 0.5 * ((t - mean)**2) / (std**2 + eps)

                # Weight by mixture component
                log_prob += torch.log(weight + eps)
                component_log_probs.append(log_prob)

            # Log-sum-exp for mixture
            component_log_probs = torch.stack(component_log_probs)
            log_prob_mixture = torch.logsumexp(component_log_probs, dim=0)

            # Negative log likelihood
            total_loss -= log_prob_mixture
            total_valid += 1

    # Average over all valid start times
    if total_valid > 0:
        loss = total_loss / total_valid
    else:
        loss = torch.tensor(0.0, device=means.device)

    return loss


def train_epoch(model, train_loader, optimizer, device, epoch, weight_count=1.0, weight_timing=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_count_loss = 0
    total_timing_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        # Move to device
        temporal_features = batch['temporal_features'].to(device)
        session_count = batch['session_count'].to(device)
        session_start_times = batch['session_start_times'].to(device)
        num_valid_sessions = batch['num_valid_sessions'].to(device)

        # Forward pass
        optimizer.zero_grad()
        session_count_lambda, timing_params = model(temporal_features)

        # Compute losses
        count_loss = poisson_nll_loss(session_count_lambda, session_count)
        timing_loss = mixture_gaussian_nll_loss(timing_params, session_start_times, num_valid_sessions)

        # Combined loss
        loss = weight_count * count_loss + weight_timing * timing_loss

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        total_count_loss += count_loss.item()
        total_timing_loss += timing_loss.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'count': f'{count_loss.item():.4f}',
            'timing': f'{timing_loss.item():.4f}'
        })

    avg_loss = total_loss / num_batches
    avg_count_loss = total_count_loss / num_batches
    avg_timing_loss = total_timing_loss / num_batches

    return avg_loss, avg_count_loss, avg_timing_loss


def validate(model, val_loader, device, epoch, weight_count=1.0, weight_timing=1.0):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_count_loss = 0
    total_timing_loss = 0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for batch in pbar:
            temporal_features = batch['temporal_features'].to(device)
            session_count = batch['session_count'].to(device)
            session_start_times = batch['session_start_times'].to(device)
            num_valid_sessions = batch['num_valid_sessions'].to(device)

            session_count_lambda, timing_params = model(temporal_features)

            count_loss = poisson_nll_loss(session_count_lambda, session_count)
            timing_loss = mixture_gaussian_nll_loss(timing_params, session_start_times, num_valid_sessions)

            loss = weight_count * count_loss + weight_timing * timing_loss

            total_loss += loss.item()
            total_count_loss += count_loss.item()
            total_timing_loss += timing_loss.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'count': f'{count_loss.item():.4f}',
                'timing': f'{timing_loss.item():.4f}'
            })

    avg_loss = total_loss / num_batches
    avg_count_loss = total_count_loss / num_batches
    avg_timing_loss = total_timing_loss / num_batches

    return avg_loss, avg_count_loss, avg_timing_loss


def train_temporal_model(
    batch_size=None,
    epochs=None,
    learning_rate=None,
    validation_split=0.2,
    early_stopping_patience=None,
    augmentation_factor=50,
    save_dir=None,
    device=None,
    weight_count=1.0,
    weight_timing=1.0
):
    """
    Train Temporal model from scratch.

    Args:
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        learning_rate: Learning rate for Adam optimizer
        validation_split: Fraction of data for validation
        early_stopping_patience: Number of epochs without improvement before stopping
        augmentation_factor: Data augmentation factor
        save_dir: Directory to save model checkpoints
        device: Device to train on (cuda/cpu)
        weight_count: Loss weight for session count
        weight_timing: Loss weight for timing prediction

    Returns:
        model: Trained model
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    # Use config defaults if not specified
    if batch_size is None:
        batch_size = TEMPORAL_TRAINING_CONFIG['batch_size']
    if epochs is None:
        epochs = TEMPORAL_TRAINING_CONFIG['epochs']
    if learning_rate is None:
        learning_rate = TEMPORAL_TRAINING_CONFIG['learning_rate']
    if early_stopping_patience is None:
        early_stopping_patience = TEMPORAL_TRAINING_CONFIG['early_stopping_patience']

    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Save directory
    if save_dir is None:
        save_dir = os.path.join(MODEL_PATHS['temporal']['model'].replace('temporal_model_best.pth', ''))
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("TRAINING TEMPORAL SCHEDULE MODEL")
    print("=" * 70)
    print("\n[WARNING] This is the highest-risk component - new implementation")
    print("          Complex dual loss: Poisson NLL + Mixture Gaussian NLL")

    # Load data
    print("\n[1/5] Preparing temporal training data...")
    print(f"  Augmentation factor: {augmentation_factor}x")

    train_loader, val_loader, scaler, metadata = prepare_temporal_training_data(
        augmentation_factor=augmentation_factor,
        batch_size=batch_size,
        validation_split=validation_split
    )

    print(f"\n  Real samples: {metadata['num_real_samples']}")
    print(f"  Augmented samples: {metadata['num_augmented_samples']}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")

    # Create model
    print("\n[2/5] Creating model...")
    model = TemporalScheduleModel(
        temporal_feature_dim=TEMPORAL_MODEL_CONFIG['temporal_feature_dim'],
        d_model=TEMPORAL_MODEL_CONFIG['d_model'],
        nhead=TEMPORAL_MODEL_CONFIG['nhead'],
        num_layers=TEMPORAL_MODEL_CONFIG['num_encoder_layers'],
        dim_feedforward=TEMPORAL_MODEL_CONFIG['dim_feedforward'],
        dropout=TEMPORAL_MODEL_CONFIG['dropout'],
        max_sessions=TEMPORAL_MODEL_CONFIG['max_sessions'],
        num_gaussian_components=TEMPORAL_MODEL_CONFIG['num_gaussian_components']
    )
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")
    print(f"  Gaussian components: {TEMPORAL_MODEL_CONFIG['num_gaussian_components']}")

    # Optimizer and scheduler
    print("\n[3/5] Setting up training...")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=TEMPORAL_TRAINING_CONFIG['scheduler_factor'],
        patience=TEMPORAL_TRAINING_CONFIG['scheduler_patience'],
        verbose=True
    )

    print(f"  Loss weights: count={weight_count:.1f}, timing={weight_timing:.1f}")

    # Training loop
    print("\n[4/5] Training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_count, train_timing = train_epoch(
            model, train_loader, optimizer, device, epoch, weight_count, weight_timing
        )
        train_losses.append(train_loss)

        # Validate
        val_loss, val_count, val_timing = validate(
            model, val_loader, device, epoch, weight_count, weight_timing
        )
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print epoch summary
        print(f"\nEpoch {epoch}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f} (count: {train_count:.4f}, timing: {train_timing:.4f})")
        print(f"  Val Loss:   {val_loss:.4f} (count: {val_count:.4f}, timing: {val_timing:.4f})")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            checkpoint_path = MODEL_PATHS['temporal']['model']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'scaler': scaler,
                'metadata': metadata
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
    print(f"  Model saved to: {save_dir}")

    print("\n" + "=" * 70)

    return model, train_losses, val_losses


if __name__ == "__main__":
    # Train model
    model, train_losses, val_losses = train_temporal_model(
        augmentation_factor=50,
        weight_count=1.0,
        weight_timing=1.0
    )

    print("\n[SUCCESS] Temporal model training complete!")
