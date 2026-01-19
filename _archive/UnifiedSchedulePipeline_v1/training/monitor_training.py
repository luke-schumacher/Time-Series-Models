"""
Training Progress Monitor
Monitors the training log and provides status updates
"""
import os
import time
import re
from datetime import datetime

LOG_FILE = "training_log_new.txt"
CHECK_INTERVAL = 300  # Check every 5 minutes

def parse_training_status(log_path):
    """Parse the training log to extract current status."""
    if not os.path.exists(log_path):
        return None

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Find the last occurrence of MODEL TRAINING STATUS
    status_sections = re.findall(
        r'MODEL TRAINING STATUS\n=+\n(.*?)(?:Overall Progress:|=====)',
        content,
        re.DOTALL
    )

    if not status_sections:
        return None

    last_status = status_sections[-1]

    # Parse model statuses
    models = {}
    for line in last_status.strip().split('\n'):
        if ':' in line:
            parts = line.split(':')
            model_name = parts[0].strip()
            status = parts[1].strip()
            models[model_name] = status

    # Find overall progress
    progress_match = re.findall(r'Overall Progress: (\d+)/(\d+) models trained \((\d+)%\)', content)
    if progress_match:
        trained, total, percent = progress_match[-1]
        progress = {
            'trained': int(trained),
            'total': int(total),
            'percent': int(percent)
        }
    else:
        progress = None

    # Find current epoch if training
    current_model = None
    current_epoch = None
    epoch_matches = re.findall(r'TRAINING: (\w+)\n=+.*?Epoch (\d+)/(\d+):', content, re.DOTALL)
    if epoch_matches:
        current_model, current_epoch, total_epochs = epoch_matches[-1]
        progress['current_model'] = current_model
        progress['current_epoch'] = int(current_epoch)
        progress['total_epochs'] = int(total_epochs)

    return {
        'models': models,
        'progress': progress,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def format_status_report(status):
    """Format a human-readable status report."""
    if not status:
        return "No training status found."

    report = []
    report.append("=" * 70)
    report.append("TRAINING PROGRESS MONITOR")
    report.append("=" * 70)
    report.append(f"Last updated: {status['timestamp']}")
    report.append("")

    if status['progress']:
        prog = status['progress']
        report.append(f"Overall Progress: {prog['trained']}/{prog['total']} models ({prog['percent']}%)")

        if 'current_model' in prog:
            report.append(f"Currently training: {prog['current_model'].upper()}")
            report.append(f"  Epoch: {prog['current_epoch']}/{prog['total_epochs']}")

        report.append("")

    report.append("Model Status:")
    report.append("-" * 70)
    for model, model_status in status['models'].items():
        status_icon = "[OK]" if "[TRAINED]" in model_status else "[..]" if "[NOT TRAINED]" in model_status else "[XX]"
        report.append(f"  {status_icon} {model:25s} : {model_status}")

    report.append("=" * 70)

    return "\n".join(report)

def is_training_complete(status):
    """Check if all models are trained."""
    if not status or not status['progress']:
        return False

    prog = status['progress']
    return prog['trained'] == prog['total']

def monitor_training(log_path, check_interval=CHECK_INTERVAL, max_checks=None):
    """
    Monitor training progress.

    Args:
        log_path: Path to training log file
        check_interval: Seconds between checks
        max_checks: Maximum number of checks (None for unlimited)
    """
    print(f"Starting training monitor...")
    print(f"Log file: {log_path}")
    print(f"Check interval: {check_interval} seconds")
    print()

    check_count = 0
    last_status_str = None

    while True:
        check_count += 1

        if max_checks and check_count > max_checks:
            print(f"\nReached maximum checks ({max_checks}). Stopping monitor.")
            break

        status = parse_training_status(log_path)

        if status:
            current_status_str = format_status_report(status)

            # Only print if status changed
            if current_status_str != last_status_str:
                print(current_status_str)
                last_status_str = current_status_str

            # Check if complete
            if is_training_complete(status):
                print("\n" + "=" * 70)
                print("*** TRAINING COMPLETE! All models successfully trained. ***")
                print("=" * 70)
                print("\nYou can now proceed to generate prototype results using:")
                print("  python generate_complete_schedule.py")
                print()
                break
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for training to start...")

        # Wait before next check
        time.sleep(check_interval)

if __name__ == "__main__":
    import sys

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, LOG_FILE)

    # Allow custom check interval from command line
    interval = CHECK_INTERVAL
    if len(sys.argv) > 1:
        try:
            interval = int(sys.argv[1])
        except ValueError:
            print(f"Invalid interval '{sys.argv[1]}', using default {CHECK_INTERVAL}s")

    try:
        monitor_training(log_path, check_interval=interval)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        print("Training will continue in the background.")
        sys.exit(0)
