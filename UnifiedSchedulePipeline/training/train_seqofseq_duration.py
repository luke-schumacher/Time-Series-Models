"""
Train SeqofSeq Duration Model from Scratch

Trains the ConditionalDurationPredictor to predict scan durations
conditioned on sequences and patient/scan context.
Predicts Gamma distribution parameters (μ, σ) for each scan.
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
seqofseq_dir = os.path.join(unified_dir, '..', 'SeqofSeq_Pipeline')
sys.path.insert(0, unified_dir)
sys.path.insert(0, seqofseq_dir)

from config import MODEL_PATHS, RANDOM_SEED
from data_loaders.seqofseq_data_loader import create_seqofseq_dataloaders
from SeqofSeq_Pipeline.models.conditional_duration_predictor import ConditionalDurationPredictor
from SeqofSeq_Pipeline.config import DURATION_MODEL_CONFIG


def gamma_nll_loss(mu, sigma, target, mask, eps=1e-6):
    """
    Gamma Negative Log-Likelihood Loss.
    """
    mu = torch.clamp(mu, min=eps)
    sigma = torch.clamp(sigma, min=eps)
    target = torch.clamp(target, min=eps)

    alpha = (mu / sigma) ** 2
    beta = mu / (sigma ** 2)

    log_prob = (alpha - 1) * torch.log(target + eps) - beta * target - torch.lgamma(alpha) + alpha * torch.log(beta + eps)
    nll = -log_prob

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
        conditioning = batch['conditioning'].to(device)
        sequence_tokens = batch['sequence_tokens'].to(device)
        durations = batch['durations'].to(device)
        mask = batch['mask'].to(device)

        optimizer.zero_grad()
        mu, sigma = model(conditioning, sequence_tokens)

        loss = gamma_nll_loss(mu, sigma, durations, mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

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
            conditioning = batch['conditioning'].to(device)
            sequence_tokens = batch['sequence_tokens'].to(device)
            durations = batch['durations'].to(device)
            mask = batch['mask'].to(device)

            mu, sigma = model(conditioning, sequence_tokens)
            loss = gamma_nll_loss(mu, sigma, durations, mask)

            num_valid = mask.sum().item()
            total_loss += loss.item() * num_valid
            total_tokens += num_valid

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / total_tokens
    return avg_loss


def train_seqofseq_duration_model(
    batch_size=32,
    epochs=100,
    learning_rate=0.0001,
    validation_split=0.2,
    early_stopping_patience=15,
    save_dir=None,
    device=None
):
    """
    Train SeqofSeq duration model from scratch.

    Returns:
        model: Trained model
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if save_dir is None:
        save_dir = MODEL_PATHS['seqofseq']['model_dir']
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("TRAINING SEQOFSEQ DURATION MODEL")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading data...")
    train_loader, val_loader, scaler, metadata = create_seqofseq_dataloaders(
        batch_size=batch_size,
        validation_split=validation_split
    )

    print(f"\n  Vocabulary size: {metadata['vocab_size']}")
    print(f"  Conditioning dimension: {metadata['num_conditioning_features']}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")

    # Create model
    print("\n[2/5] Creating model...")
    model_config = DURATION_MODEL_CONFIG.copy()
    model_config['conditioning_dim'] = metadata['num_conditioning_features']
    model_config['vocab_size'] = metadata['vocab_size']
    model_config['sequence_feature_dim'] = metadata['vocab_size']  # For SeqofSeq, no extra features

    model = ConditionalDurationPredictor(config=model_config)
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
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        train_losses.append(train_loss)

        val_loss = validate(model, val_loader, device, epoch)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f"\nEpoch {epoch}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

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
                'metadata': metadata,
                'scaler': scaler
            }, checkpoint_path)
            print(f"  [SAVED] Best model: {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{early_stopping_patience}")

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
    model, train_losses, val_losses = train_seqofseq_duration_model(
        batch_size=32,
        epochs=100,
        learning_rate=0.0001,
        early_stopping_patience=15
    )

    print("\n[SUCCESS] SeqofSeq duration model training complete!")
