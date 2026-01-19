"""
Train PXChange Sequence Model from Scratch

Trains the ConditionalSequenceGenerator to predict sourceID token sequences
conditioned on patient/scan context (Age, Weight, Height, BodyGroup, PTAB, entity_type).
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
from PXChange_Refactored.models.conditional_sequence_generator import ConditionalSequenceGenerator
from PXChange_Refactored.config import SEQUENCE_MODEL_CONFIG, PAD_TOKEN_ID


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_tokens = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        conditioning = batch['conditioning'].to(device)
        sequence_tokens = batch['sequence_tokens'].to(device)
        mask = batch['mask'].to(device)

        # Prepare input and target sequences
        # Input: all tokens except last
        # Target: all tokens except first (shifted by 1)
        input_seq = sequence_tokens[:, :-1]
        target_seq = sequence_tokens[:, 1:]
        input_mask = mask[:, :-1]
        target_mask = mask[:, 1:]

        # Forward pass
        optimizer.zero_grad()
        logits = model(conditioning, input_seq)  # [batch, seq_len-1, vocab_size]

        # Compute loss (only on non-padded positions)
        # Reshape for loss computation
        logits_flat = logits.reshape(-1, logits.size(-1))  # [batch * seq_len, vocab_size]
        target_flat = target_seq.reshape(-1)  # [batch * seq_len]
        mask_flat = target_mask.reshape(-1)  # [batch * seq_len]

        # Compute loss
        loss = criterion(logits_flat, target_flat)

        # Apply mask (only compute loss on valid positions)
        loss = (loss * mask_flat).sum() / mask_flat.sum()

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Statistics
        total_loss += loss.item() * mask_flat.sum().item()
        total_tokens += mask_flat.sum().item()

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / total_tokens
    return avg_loss


def validate(model, val_loader, criterion, device, epoch):
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
            mask = batch['mask'].to(device)

            # Prepare input and target
            input_seq = sequence_tokens[:, :-1]
            target_seq = sequence_tokens[:, 1:]
            input_mask = mask[:, :-1]
            target_mask = mask[:, 1:]

            # Forward pass
            logits = model(conditioning, input_seq)

            # Compute loss
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = target_seq.reshape(-1)
            mask_flat = target_mask.reshape(-1)

            loss = criterion(logits_flat, target_flat)
            loss = (loss * mask_flat).sum() / mask_flat.sum()

            total_loss += loss.item() * mask_flat.sum().item()
            total_tokens += mask_flat.sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / total_tokens
    return avg_loss


def train_pxchange_sequence_model(
    batch_size=32,
    epochs=100,
    learning_rate=0.0001,
    validation_split=0.2,
    early_stopping_patience=15,
    save_dir=None,
    device=None
):
    """
    Train PXChange sequence model from scratch.

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
    print("TRAINING PXCHANGE SEQUENCE MODEL")
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
    model_config = SEQUENCE_MODEL_CONFIG.copy()
    model_config['vocab_size'] = metadata['vocab_size']
    model_config['conditioning_dim'] = metadata['conditioning_dim']

    model = ConditionalSequenceGenerator(config=model_config)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")

    # Loss and optimizer
    print("\n[3/5] Setting up training...")
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=PAD_TOKEN_ID)
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
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, criterion, device, epoch)
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

            checkpoint_path = os.path.join(save_dir, 'sequence_model_best.pth')
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
    model, train_losses, val_losses = train_pxchange_sequence_model(
        batch_size=32,
        epochs=100,
        learning_rate=0.0001,
        early_stopping_patience=15
    )

    print("\n[SUCCESS] PXChange sequence model training complete!")
