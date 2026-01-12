"""
Quick Training Status Check
Run this anytime to see current training progress
"""
import os
import re
from datetime import datetime

def quick_status_check():
    """Quick check of training status."""
    log_file = os.path.join(os.path.dirname(__file__), "training_log_new.txt")

    if not os.path.exists(log_file):
        print("❌ Training log not found. Training may not have started yet.")
        return

    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Get the last status section
    status_pattern = r'MODEL TRAINING STATUS\n=+\n(.*?)(?:Overall Progress:)'
    matches = re.findall(status_pattern, content, re.DOTALL)

    if not matches:
        print("⏳ Training initializing...")
        return

    # Get overall progress
    progress_matches = re.findall(r'Overall Progress: (\d+)/(\d+) models trained \((\d+)%\)', content)
    if progress_matches:
        trained, total, percent = progress_matches[-1]
        print("\n" + "=" * 70)
        print("TRAINING STATUS")
        print("=" * 70)
        print(f"Progress: {trained}/{total} models completed ({percent}%)")
        print("")

        # Check which model is currently training
        training_matches = re.findall(r'TRAINING: (\w+)\n=+\n', content)
        if training_matches:
            current_model = training_matches[-1]
            print(f"Currently training: {current_model.upper()}")

            # Check current epoch
            epoch_matches = re.findall(
                rf'TRAINING: {current_model}.*?Epoch (\d+)/(\d+):',
                content,
                re.DOTALL
            )
            if epoch_matches:
                current_epoch, total_epochs = epoch_matches[-1]
                progress_pct = int(current_epoch) / int(total_epochs) * 100
                print(f"  Epoch: {current_epoch}/{total_epochs} ({progress_pct:.1f}%)")

                # Get latest validation loss
                val_loss_pattern = rf'Epoch {current_epoch}/\d+:\n.*?Val Loss:\s+([\d.]+)'
                val_loss_matches = re.findall(val_loss_pattern, content, re.DOTALL)
                if val_loss_matches:
                    print(f"  Latest Val Loss: {val_loss_matches[-1]}")

        print("")
        print("Model Status:")
        print("-" * 70)

        # Parse model statuses
        last_status = matches[-1]
        for line in last_status.strip().split('\n'):
            if ':' in line:
                parts = line.split(':', 1)
                model_name = parts[0].strip()
                status = parts[1].strip()
                icon = "[OK]" if "[TRAINED]" in status else "[..]"
                print(f"  {icon} {model_name:25s} : {status}")

        print("=" * 70)

        if int(trained) == int(total):
            print("\n*** ALL MODELS TRAINED! Ready to generate results. ***")
        else:
            print(f"\n>> Training in progress... {int(total) - int(trained)} models remaining")

        print()

if __name__ == "__main__":
    try:
        quick_status_check()
    except Exception as e:
        print(f"Error checking status: {e}")
