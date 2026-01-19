"""
Master Training Orchestrator

Trains all 5 models for the UnifiedSchedulePipeline:
1. PXChange Sequence Model
2. PXChange Duration Model
3. SeqofSeq Sequence Model
4. SeqofSeq Duration Model
5. Temporal Schedule Model

Can run sequentially or in parallel.
"""

import os
import sys
import argparse
import time
from datetime import datetime

# Add paths
unified_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, unified_dir)

import config
MODEL_PATHS = config.MODEL_PATHS


def check_training_status():
    """
    Check which models are already trained.

    Returns:
        status: Dict with model names and training status
    """
    status = {}

    # PXChange models
    pxchange_seq_path = MODEL_PATHS['pxchange']['sequence']
    pxchange_dur_path = MODEL_PATHS['pxchange']['duration']

    status['pxchange_sequence'] = os.path.exists(pxchange_seq_path)
    status['pxchange_duration'] = os.path.exists(pxchange_dur_path)

    # SeqofSeq models
    seqofseq_seq_path = MODEL_PATHS['seqofseq']['sequence']
    seqofseq_dur_path = MODEL_PATHS['seqofseq']['duration']

    status['seqofseq_sequence'] = os.path.exists(seqofseq_seq_path)
    status['seqofseq_duration'] = os.path.exists(seqofseq_dur_path)

    # Temporal model
    temporal_path = MODEL_PATHS['temporal']['model']

    status['temporal'] = os.path.exists(temporal_path)

    return status


def print_training_status(status):
    """Print the training status of all models."""
    print("\n" + "=" * 70)
    print("MODEL TRAINING STATUS")
    print("=" * 70)

    for model_name, is_trained in status.items():
        status_str = "[TRAINED]" if is_trained else "[NOT TRAINED]"
        print(f"  {model_name:25s}: {status_str}")

    num_trained = sum(status.values())
    total_models = len(status)

    print(f"\n  Overall Progress: {num_trained}/{total_models} models trained ({100*num_trained/total_models:.0f}%)")
    print("=" * 70 + "\n")


def train_model(model_name, force_retrain=False):
    """
    Train a specific model.

    Args:
        model_name: Name of the model to train
        force_retrain: If True, retrain even if model exists

    Returns:
        success: True if training succeeded
    """
    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name.upper()}")
    print("=" * 70)

    # Check if already trained
    status = check_training_status()
    if status[model_name] and not force_retrain:
        print(f"\n[SKIPPED] Model already trained. Use --force to retrain.")
        print("=" * 70)
        return True

    start_time = time.time()

    try:
        if model_name == 'pxchange_sequence':
            from train_pxchange_sequence import train_pxchange_sequence_model
            model, train_losses, val_losses = train_pxchange_sequence_model()

        elif model_name == 'pxchange_duration':
            from train_pxchange_duration import train_pxchange_duration_model
            model, train_losses, val_losses = train_pxchange_duration_model()

        elif model_name == 'seqofseq_sequence':
            from train_seqofseq_sequence import train_seqofseq_sequence_model
            model, train_losses, val_losses = train_seqofseq_sequence_model()

        elif model_name == 'seqofseq_duration':
            from train_seqofseq_duration import train_seqofseq_duration_model
            model, train_losses, val_losses = train_seqofseq_duration_model()

        elif model_name == 'temporal':
            from train_temporal_model import train_temporal_model
            model, train_losses, val_losses = train_temporal_model()

        else:
            print(f"[ERROR] Unknown model: {model_name}")
            return False

        elapsed = time.time() - start_time
        print(f"\n[SUCCESS] {model_name} training complete in {elapsed/60:.2f} minutes")
        print("=" * 70)
        return True

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[ERROR] Training failed after {elapsed/60:.2f} minutes: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70)
        return False


def train_all_models(models=None, force_retrain=False, parallel=False):
    """
    Train all models (or a subset).

    Args:
        models: List of model names to train. If None, train all.
        force_retrain: If True, retrain even if model exists
        parallel: If True, attempt to run in parallel (NOT IMPLEMENTED)

    Returns:
        results: Dict with training results for each model
    """
    if models is None:
        models = [
            'temporal',           # Train first - needed for temporal features
            'pxchange_sequence',
            'pxchange_duration',
            'seqofseq_sequence',
            'seqofseq_duration'
        ]

    # Check initial status
    initial_status = check_training_status()
    print_training_status(initial_status)

    # Sequential training
    if not parallel:
        print("\n[INFO] Running sequential training")
        print(f"[INFO] Models to train: {', '.join(models)}")

        overall_start = time.time()
        results = {}

        for model_name in models:
            success = train_model(model_name, force_retrain)
            results[model_name] = success

            if not success:
                print(f"\n[WARNING] Training failed for {model_name}")
                print("[WARNING] Continuing with next model...")

        overall_elapsed = time.time() - overall_start

        # Final status
        final_status = check_training_status()
        print_training_status(final_status)

        # Summary
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        print(f"  Total time: {overall_elapsed/60:.2f} minutes ({overall_elapsed/3600:.2f} hours)")
        print(f"\n  Results:")
        for model_name, success in results.items():
            status_str = "[SUCCESS]" if success else "[FAILED]"
            print(f"    {model_name:25s}: {status_str}")

        num_success = sum(results.values())
        total_attempted = len(results)
        print(f"\n  Success rate: {num_success}/{total_attempted} ({100*num_success/total_attempted:.0f}%)")
        print("=" * 70 + "\n")

        return results

    else:
        print("\n[ERROR] Parallel training not implemented yet")
        print("[INFO] Use sequential mode (default)")
        return {}


def main():
    """Main entry point for training orchestrator."""
    parser = argparse.ArgumentParser(description='Train UnifiedSchedulePipeline models')

    parser.add_argument(
        '--models',
        nargs='+',
        choices=['pxchange_sequence', 'pxchange_duration',
                 'seqofseq_sequence', 'seqofseq_duration', 'temporal', 'all'],
        default=['all'],
        help='Models to train (default: all)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force retraining even if model exists'
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run training in parallel (not implemented)'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Only show training status, do not train'
    )

    args = parser.parse_args()

    # Show status only
    if args.status:
        status = check_training_status()
        print_training_status(status)
        return

    # Determine models to train
    if 'all' in args.models:
        models = None  # Train all
    else:
        models = args.models

    # Train models
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*70}")
    print(f"UNIFIED SCHEDULE PIPELINE - TRAINING ORCHESTRATOR")
    print(f"{'='*70}")
    print(f"  Started: {timestamp}")
    print(f"  Force retrain: {args.force}")
    print(f"  Parallel: {args.parallel}")
    print(f"{'='*70}\n")

    results = train_all_models(models, force_retrain=args.force, parallel=args.parallel)

    # Exit code based on results
    if results:
        all_success = all(results.values())
        sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
