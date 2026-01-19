"""
Training script for the Examination Model.

Trains the model to generate MRI event sequences for specific body regions.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EXAMINATION_MODEL_CONFIG, EXAMINATION_TRAINING_CONFIG,
    MODEL_SAVE_DIR, RANDOM_SEED, USE_GPU, MAX_SEQ_LEN,
    START_TOKEN_ID, END_TOKEN_ID, PAD_TOKEN_ID
)
from models.examination_model import create_examination_model
from data.preprocessing import load_preprocessed_data


class ExaminationDataset(Dataset):
    """Dataset for examination (scan sequence) training."""

    def __init__(self, examination_sequences, max_seq_len=None):
        """
        Args:
            examination_sequences: List of examination sequence dicts from preprocessing
            max_seq_len: Maximum sequence length (default: MAX_SEQ_LEN)
        """
        if max_seq_len is None:
            max_seq_len = MAX_SEQ_LEN

        self.max_seq_len = max_seq_len
        self.data = []

        for seq in examination_sequences:
            # Extract conditioning features
            cond = seq['conditioning']
            conditioning = torch.tensor([
                cond.get('Age', 0),
                cond.get('Weight', 0),
                cond.get('Height', 0),
                cond.get('PTAB', 0),
                cond.get('Direction_encoded', 0)
            ], dtype=torch.float32)

            # Body region
            body_region = seq['body_region']

            # Token sequence
            tokens = seq['sequence']

            # Prepare input and target sequences
            # Input: [START, tok1, tok2, ..., tokN]
            # Target: [tok1, tok2, ..., tokN, END]
            input_seq = [START_TOKEN_ID] + tokens[:max_seq_len - 1]
            target_seq = tokens[:max_seq_len - 1] + [END_TOKEN_ID]

            # Pad sequences
            pad_len = max_seq_len - len(input_seq)
            input_seq = input_seq + [PAD_TOKEN_ID] * pad_len
            target_seq = target_seq + [PAD_TOKEN_ID] * pad_len

            self.data.append({
                'conditioning': conditioning,
                'body_region': body_region,
                'input_seq': torch.tensor(input_seq, dtype=torch.long),
                'target_seq': torch.tensor(target_seq, dtype=torch.long)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            item['conditioning'],
            torch.tensor(item['body_region'], dtype=torch.long),
            item['input_seq'],
            item['target_seq']
        )


def train_examination_model(data_path=None, config=None, training_config=None,
                            save_dir=None, verbose=True):
    """
    Train the Examination Model.

    Args:
        data_path: Path to preprocessed data pickle file
        config: Model config dict (default: EXAMINATION_MODEL_CONFIG)
        training_config: Training config dict (default: EXAMINATION_TRAINING_CONFIG)
        save_dir: Directory to save model (default: MODEL_SAVE_DIR)
        verbose: Print progress

    Returns:
        Trained model, training history
    """
    if config is None:
        config = EXAMINATION_MODEL_CONFIG
    if training_config is None:
        training_config = EXAMINATION_TRAINING_CONFIG
    if save_dir is None:
        save_dir = os.path.join(MODEL_SAVE_DIR, 'examination')

    os.makedirs(save_dir, exist_ok=True)

    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Device
    device = torch.device('cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")

    # Load data
    if verbose:
        print("Loading data...")

    if data_path is None:
        preprocessed = load_preprocessed_data()
    else:
        with open(data_path, 'rb') as f:
            preprocessed = pickle.load(f)

    examination_sequences = preprocessed['examination']
    if verbose:
        print(f"Loaded {len(examination_sequences)} examination sequences")

    # Create dataset
    dataset = ExaminationDataset(examination_sequences)

    # Split into train/val
    val_size = int(len(dataset) * training_config['validation_split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if verbose:
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # Create model
    model = create_examination_model(config)
    model = model.to(device)

    if verbose:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer with warmup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate']
    )

    # Learning rate scheduler with warmup
    def lr_lambda(step):
        warmup_steps = training_config['warmup_steps']
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_perplexity': []}
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0

    for epoch in range(training_config['epochs']):
        # Training
        model.train()
        train_loss = 0.0

        for conditioning, body_region, input_seq, target_seq in tqdm(train_loader, disable=not verbose,
                                                                      desc=f"Epoch {epoch+1}"):
            conditioning = conditioning.to(device)
            body_region = body_region.to(device)
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(conditioning, body_region, input_seq)

            # Compute loss
            loss = model.compute_loss(
                logits, target_seq,
                label_smoothing=training_config['label_smoothing']
            )

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config['gradient_clip'])

            optimizer.step()
            scheduler.step()
            global_step += 1

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for conditioning, body_region, input_seq, target_seq in val_loader:
                conditioning = conditioning.to(device)
                body_region = body_region.to(device)
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)

                logits = model(conditioning, body_region, input_seq)
                loss = model.compute_loss(logits, target_seq)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_perplexity = np.exp(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_perplexity'].append(val_perplexity)

        if verbose:
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, perplexity={val_perplexity:.2f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(save_dir, 'examination_model_best.pt'))
        else:
            patience_counter += 1

        if patience_counter >= training_config['early_stopping_patience']:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'examination_model_final.pt'))

    # Save history
    with open(os.path.join(save_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    if verbose:
        print(f"\nTraining complete. Models saved to {save_dir}")

    return model, history


if __name__ == "__main__":
    print("Training Examination Model...")
    print("=" * 60)

    model, history = train_examination_model(verbose=True)

    print("\nFinal Results:")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Best validation perplexity: {min(history['val_perplexity']):.2f}")
