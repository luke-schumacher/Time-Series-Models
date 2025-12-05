"""
Comprehensive Data Preprocessing with PAUSE Token Injection
Processes both PXChange and SeqofSeq data
"""
import os
import sys
import glob
import pandas as pd
from datetime import datetime

# Get absolute paths
unified_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, unified_dir)

# Import our config
import config as unified_config
PXCHANGE_DIR = unified_config.PXCHANGE_DIR
SEQOFSEQ_DIR = unified_config.SEQOFSEQ_DIR
PAUSE_DETECTION_THRESHOLD_MINUTES = unified_config.PAUSE_DETECTION_THRESHOLD_MINUTES

# Import pause injection directly from this directory
from pause_injection import inject_pauses_pxchange, inject_pauses_seqofseq


def preprocess_pxchange_data():
    """
    Preprocess all PXChange CSV files with PAUSE injection
    """
    print("\n" + "="*80)
    print("PREPROCESSING PXCHANGE DATA")
    print("="*80)

    # Input and output directories
    input_dir = os.path.join(PXCHANGE_DIR, 'data')
    output_dir = os.path.join(PXCHANGE_DIR, 'data', 'preprocessed_with_pauses')

    os.makedirs(output_dir, exist_ok=True)

    # Find all CSV files
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    csv_files = [f for f in csv_files if 'preprocessed' not in f]  # Exclude already processed

    print(f"Found {len(csv_files)} raw CSV files")
    print(f"Output directory: {output_dir}")
    print(f"Pause threshold: {PAUSE_DETECTION_THRESHOLD_MINUTES} minutes\n")

    stats = {
        'files_processed': 0,
        'total_events': 0,
        'total_pauses': 0,
        'errors': 0
    }

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        dataset_id = filename.replace('.csv', '')

        print(f"Processing: {filename}")

        try:
            # Load file to get initial count
            df_before = pd.read_csv(file_path)
            events_before = len(df_before)

            # Output path
            output_path = os.path.join(output_dir, f"{dataset_id}_with_pauses.csv")

            # Process with PAUSE injection
            df_after = inject_pauses_pxchange(file_path, output_path, dataset_id)

            events_after = len(df_after)
            pauses_added = events_after - events_before

            print(f"  Before: {events_before} events")
            print(f"  After: {events_after} events")
            print(f"  Pauses added: {pauses_added}\n")

            stats['files_processed'] += 1
            stats['total_events'] += events_after
            stats['total_pauses'] += pauses_added

        except Exception as e:
            print(f"  ERROR: {e}\n")
            stats['errors'] += 1

    print("\n" + "-"*80)
    print("PXCHANGE PREPROCESSING SUMMARY:")
    print(f"  Files processed: {stats['files_processed']}/{len(csv_files)}")
    print(f"  Total events: {stats['total_events']:,}")
    print(f"  Total pauses added: {stats['total_pauses']:,}")
    print(f"  Pause rate: {stats['total_pauses']/stats['total_events']*100:.2f}%" if stats['total_events'] > 0 else "  Pause rate: N/A")
    print(f"  Errors: {stats['errors']}")
    print("-"*80)

    return stats


def preprocess_seqofseq_data():
    """
    Preprocess all SeqofSeq CSV files with PAUSE injection
    """
    print("\n" + "="*80)
    print("PREPROCESSING SEQOFSEQ DATA")
    print("="*80)

    # Input and output directories
    input_dir = os.path.join(SEQOFSEQ_DIR, 'data')
    output_dir = os.path.join(SEQOFSEQ_DIR, 'data', 'preprocessed_with_pauses')

    os.makedirs(output_dir, exist_ok=True)

    # Find all CSV files
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    csv_files = [f for f in csv_files if 'preprocessed' not in f]

    print(f"Found {len(csv_files)} raw CSV files")
    print(f"Output directory: {output_dir}")
    print(f"Pause threshold: {PAUSE_DETECTION_THRESHOLD_MINUTES} minutes\n")

    stats = {
        'files_processed': 0,
        'total_scans': 0,
        'total_pauses': 0,
        'errors': 0
    }

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        dataset_id = filename.replace('.csv', '')

        print(f"Processing: {filename}")

        try:
            # Load file to get initial count
            df_before = pd.read_csv(file_path)
            scans_before = len(df_before)

            # Output path
            output_path = os.path.join(output_dir, f"{dataset_id}_with_pauses.csv")

            # Process with PAUSE injection
            df_after = inject_pauses_seqofseq(file_path, output_path, dataset_id)

            scans_after = len(df_after)
            pauses_added = scans_after - scans_before

            print(f"  Before: {scans_before} scans")
            print(f"  After: {scans_after} scans")
            print(f"  Pauses added: {pauses_added}\n")

            stats['files_processed'] += 1
            stats['total_scans'] += scans_after
            stats['total_pauses'] += pauses_added

        except Exception as e:
            print(f"  ERROR: {e}\n")
            stats['errors'] += 1

    print("\n" + "-"*80)
    print("SEQOFSEQ PREPROCESSING SUMMARY:")
    print(f"  Files processed: {stats['files_processed']}/{len(csv_files)}")
    print(f"  Total scans: {stats['total_scans']:,}")
    print(f"  Total pauses added: {stats['total_pauses']:,}")
    print(f"  Pause rate: {stats['total_pauses']/stats['total_scans']*100:.2f}%" if stats['total_scans'] > 0 else "  Pause rate: N/A")
    print(f"  Errors: {stats['errors']}")
    print("-"*80)

    return stats


def main():
    """
    Main preprocessing pipeline
    """
    print("\n" + "="*80)
    print("PHASE 2: DATA PREPROCESSING WITH PAUSE TOKENS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Preprocess PXChange data
    pxchange_stats = preprocess_pxchange_data()

    # Preprocess SeqofSeq data
    seqofseq_stats = preprocess_seqofseq_data()

    # Overall summary
    print("\n\n" + "="*80)
    print("OVERALL PREPROCESSING SUMMARY")
    print("="*80)
    print(f"\nPXChange:")
    print(f"  Files: {pxchange_stats['files_processed']}")
    print(f"  Events: {pxchange_stats['total_events']:,}")
    print(f"  Pauses: {pxchange_stats['total_pauses']:,}")

    print(f"\nSeqofSeq:")
    print(f"  Files: {seqofseq_stats['files_processed']}")
    print(f"  Scans: {seqofseq_stats['total_scans']:,}")
    print(f"  Pauses: {seqofseq_stats['total_pauses']:,}")

    total_data_points = pxchange_stats['total_events'] + seqofseq_stats['total_scans']
    total_pauses = pxchange_stats['total_pauses'] + seqofseq_stats['total_pauses']

    print(f"\nCombined:")
    print(f"  Total data points: {total_data_points:,}")
    print(f"  Total pauses: {total_pauses:,}")
    print(f"  Overall pause rate: {total_pauses/total_data_points*100:.2f}%" if total_data_points > 0 else "  Overall pause rate: N/A")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Save summary report
    report_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'preprocessing_summary.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, 'w') as f:
        f.write("DATA PREPROCESSING SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Pause threshold: {PAUSE_DETECTION_THRESHOLD_MINUTES} minutes\n\n")
        f.write(f"PXChange: {pxchange_stats['files_processed']} files, ")
        f.write(f"{pxchange_stats['total_events']:,} events, ")
        f.write(f"{pxchange_stats['total_pauses']:,} pauses\n")
        f.write(f"SeqofSeq: {seqofseq_stats['files_processed']} files, ")
        f.write(f"{seqofseq_stats['total_scans']:,} scans, ")
        f.write(f"{seqofseq_stats['total_pauses']:,} pauses\n\n")
        f.write(f"Total: {total_data_points:,} data points, {total_pauses:,} pauses ")
        f.write(f"({total_pauses/total_data_points*100:.2f}%)\n" if total_data_points > 0 else "\n")

    print(f"\nSummary saved to: {report_path}")

    return {
        'pxchange': pxchange_stats,
        'seqofseq': seqofseq_stats,
        'total_pauses': total_pauses,
        'total_data_points': total_data_points
    }


if __name__ == "__main__":
    results = main()

    print("\n✓ Data preprocessing complete!")
    print("  Next step: Fine-tune models with the preprocessed data")
