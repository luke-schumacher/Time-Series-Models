"""
Comprehensive Data Preprocessing with Pseudo-Patient Architecture
Processes both PXChange and SeqofSeq data with sequence segmentation

UPDATED: Replaces PAUSE token injection with pseudo-patient entity generation
"""
import os
import sys
import glob
import pandas as pd
from datetime import datetime

# Get absolute paths
unified_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, unified_dir)

# Get the absolute path to the Time-Series-Models root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import our config
import config as unified_config
PXCHANGE_DIR = unified_config.PXCHANGE_DIR
SEQOFSEQ_DIR = unified_config.SEQOFSEQ_DIR
PAUSE_DETECTION_THRESHOLD_MINUTES = unified_config.PAUSE_DETECTION_THRESHOLD_MINUTES
ENTITY_TYPES = unified_config.ENTITY_TYPES

# Import new segmentation and alignment modules
from sequence_segmentation import segment_pxchange_file, segment_seqofseq_file
from temporal_alignment import validate_temporal_consistency, generate_temporal_alignment_report
from SeqofSeq_Pipeline.preprocessing.preprocess_raw_data import preprocess_mri_data as seqofseq_preprocess_mri_data


def validate_temporal_alignment():
    """
    Validate temporal consistency between PXChange and SeqofSeq datasets
    """
    print("\n" + "="*80)
    print("TEMPORAL ALIGNMENT VALIDATION")
    print("="*80)

    # Load sample data from each pipeline to check alignment
    pxchange_files = glob.glob(os.path.join(PXCHANGE_DIR, 'data', '*.csv'))
    pxchange_files = [f for f in pxchange_files if 'preprocessed' not in f and 'segmented' not in f]

    seqofseq_files = glob.glob(os.path.join(SEQOFSEQ_DIR, 'data', '*.csv'))
    seqofseq_files = [f for f in seqofseq_files if 'preprocessed' not in f and 'segmented' not in f]

    if not pxchange_files or not seqofseq_files:
        print("⚠️  Warning: Insufficient data files for temporal alignment validation")
        print(f"   PXChange files found: {len(pxchange_files)}")
        print(f"   SeqofSeq files found: {len(seqofseq_files)}")
        return None

    # Load first file from each pipeline
    print(f"Validating alignment using:")
    print(f"  PXChange: {os.path.basename(pxchange_files[0])}")
    print(f"  SeqofSeq: {os.path.basename(seqofseq_files[0])}")
    print()

    try:
        pxchange_df = pd.read_csv(pxchange_files[0])
        seqofseq_df = pd.read_csv(seqofseq_files[0])

        validation_report, warnings = validate_temporal_consistency(
            pxchange_df, seqofseq_df,
            pxchange_datetime_col='datetime',
            seqofseq_datetime_col='startTime',
            verbose=True
        )

        # Save detailed report
        report_path = os.path.join(unified_dir, 'outputs', 'temporal_alignment_report.txt')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        generate_temporal_alignment_report(validation_report, report_path)

        return validation_report

    except Exception as e:
        print(f"ERROR during temporal validation: {e}")
        return None


def preprocess_pxchange_data():
    """
    Preprocess all PXChange CSV files with sequence segmentation
    """
    print("\n" + "="*80)
    print("PREPROCESSING PXCHANGE DATA (Pseudo-Patient Architecture)")
    print("="*80)

    # Input and output directories
    input_dir = os.path.join(PXCHANGE_DIR, 'data')
    output_dir = os.path.join(PXCHANGE_DIR, 'data', 'preprocessed_segmented')

    os.makedirs(output_dir, exist_ok=True)

    # Find all CSV files
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    csv_files = [f for f in csv_files if 'preprocessed' not in f and 'segmented' not in f]

    print(f"Found {len(csv_files)} raw CSV files")
    print(f"Output directory: {output_dir}")
    print(f"Pause threshold: {PAUSE_DETECTION_THRESHOLD_MINUTES} minutes")
    print(f"Entity types: {list(ENTITY_TYPES.keys())}\n")

    stats = {
        'files_processed': 0,
        'total_segments': 0,
        'real_patient_segments': 0,
        'pseudo_patient_segments': 0,
        'total_events': 0,
        'errors': 0
    }

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        dataset_id = filename.replace('.csv', '')

        print(f"Processing: {filename}")

        try:
            # Output path
            output_path = os.path.join(output_dir, f"{dataset_id}_segmented.csv")

            # Process with sequence segmentation
            df_segmented, file_stats = segment_pxchange_file(file_path, output_path, dataset_id)

            print(f"  Segments created: {file_stats['total_segments']}")
            print(f"    - Real patient: {file_stats['real_patient_segments']}")
            print(f"    - Pseudo-patient: {file_stats['pseudo_patient_segments']}")
            if file_stats['validation_errors']:
                print(f"  ⚠️  Validation warnings: {len(file_stats['validation_errors'])}")
            print()

            stats['files_processed'] += 1
            stats['total_segments'] += file_stats['total_segments']
            stats['real_patient_segments'] += file_stats['real_patient_segments']
            stats['pseudo_patient_segments'] += file_stats['pseudo_patient_segments']
            stats['total_events'] += len(df_segmented)

        except Exception as e:
            print(f"  ERROR: {e}\n")
            stats['errors'] += 1

    print("\n" + "-"*80)
    print("PXCHANGE PREPROCESSING SUMMARY:")
    print(f"  Files processed: {stats['files_processed']}/{len(csv_files)}")
    print(f"  Total segments: {stats['total_segments']:,}")
    print(f"    - Real patient segments: {stats['real_patient_segments']:,}")
    print(f"    - Pseudo-patient segments: {stats['pseudo_patient_segments']:,}")
    print(f"  Pseudo-patient ratio: {stats['pseudo_patient_segments']/stats['total_segments']*100:.2f}%" if stats['total_segments'] > 0 else "  Pseudo-patient ratio: N/A")
    print(f"  Total events: {stats['total_events']:,}")
    print(f"  Errors: {stats['errors']}")
    print("-"*80)

    return stats


def preprocess_seqofseq_data():
    """
    Preprocess all SeqofSeq CSV files with sequence segmentation AND generate metadata.
    """
    print("\n" + "="*80)
    print("PREPROCESSING SEQOFSEQ DATA (Pseudo-Patient Architecture)")
    print("="*80)

    # Input and output directories for segmentation
    input_dir_raw = os.path.join(SEQOFSEQ_DIR, 'data')
    output_dir_segmented = os.path.join(SEQOFSEQ_DIR, 'data', 'preprocessed_segmented')

    os.makedirs(output_dir_segmented, exist_ok=True)

    # Find all raw CSV files
    csv_files_raw = glob.glob(os.path.join(input_dir_raw, '*.csv'))
    csv_files_raw = [f for f in csv_files_raw if 'preprocessed' not in f and 'segmented' not in f]

    print(f"Found {len(csv_files_raw)} raw CSV files")
    print(f"Output directory for segmentation: {output_dir_segmented}")
    print(f"Pause threshold: {PAUSE_DETECTION_THRESHOLD_MINUTES} minutes")
    print(f"Entity types: {list(ENTITY_TYPES.keys())}\n")

    stats = {
        'files_processed': 0,
        'total_segments': 0,
        'real_patient_segments': 0,
        'pseudo_patient_segments': 0,
        'total_scans': 0,
        'errors': 0
    }

    processed_files_for_metadata = []

    for file_path_raw in csv_files_raw:
        filename = os.path.basename(file_path_raw)
        dataset_id = filename.replace('.csv', '')

        print(f"Processing (Segmentation): {filename}")

        try:
            # Output path for segmented file
            output_path_segmented = os.path.join(output_dir_segmented, f"{dataset_id}_segmented.csv")

            # Process with sequence segmentation
            df_segmented, file_stats = segment_seqofseq_file(file_path_raw, output_path_segmented, dataset_id)

            print(f"  Segments created: {file_stats['total_segments']}")
            print(f"    - Real patient: {file_stats['real_patient_segments']}")
            print(f"    - Pseudo-patient: {file_stats['pseudo_patient_segments']}")
            if file_stats['validation_errors']:
                print(f"  ⚠️  Validation warnings: {len(file_stats['validation_errors'])}")
            print()

            stats['files_processed'] += 1
            stats['total_segments'] += file_stats['total_segments']
            stats['real_patient_segments'] += file_stats['real_patient_segments']
            stats['pseudo_patient_segments'] += file_stats['pseudo_patient_segments']
            stats['total_scans'] += len(df_segmented)
            processed_files_for_metadata.append(output_path_segmented)

        except Exception as e:
            print(f"  ERROR (Segmentation): {e}\n")
            stats['errors'] += 1

    print("\n" + "-"*80)
    print("SEQOFSEQ SEGMENTATION SUMMARY:")
    print(f"  Files processed: {stats['files_processed']}/{len(csv_files_raw)}")
    print(f"  Total segments: {stats['total_segments']:,}")
    print(f"    - Real patient segments: {stats['real_patient_segments']:,}")
    print(f"    - Pseudo-patient segments: {stats['pseudo_patient_segments']:,}")
    print(f"  Pseudo-patient ratio: {stats['pseudo_patient_segments']/stats['total_segments']*100:.2f}%" if stats['total_segments'] > 0 else "  Pseudo-patient ratio: N/A")
    print(f"  Total scans: {stats['total_scans']:,}")
    print(f"  Errors: {stats['errors']}")
    print("-"*80)

    # Now, call seqofseq_preprocess_mri_data to generate metadata
    if len(processed_files_for_metadata) == 1:
        print("\n" + "="*80)
        print("GENERATING SEQOFSEQ METADATA FROM SEGMENTED DATA")
        print("="*80)
        try:
            # The seqofseq_preprocess_mri_data expects data_file and saves its own preprocessed_data.csv and metadata.pkl
            # We will use the already segmented file as its input.
            _, metadata_output = seqofseq_preprocess_mri_data(data_file_or_path=processed_files_for_metadata[0], save_preprocessed=True)
            print("[OK] SeqofSeq metadata generated successfully.")
        except Exception as e:
            print(f"ERROR: Failed to generate SeqofSeq metadata: {e}")
            stats['errors'] += 1
    elif len(processed_files_for_metadata) > 1:
        print("[WARN] Multiple SeqofSeq files after segmentation. Metadata generation currently only supports a single file.")
        print("Please combine files manually or adapt this script.")
        stats['errors'] += 1
    else:
        print("[WARN] No SeqofSeq files were processed for metadata generation.")

    print("\n" + "="*80)
    print("SEQOFSEQ PREPROCESSING COMPLETE")
    print("="*80)

    return stats


def main():
    """
    Main preprocessing pipeline with pseudo-patient architecture
    """
    print("\n" + "="*80)
    print("DATA PREPROCESSING WITH PSEUDO-PATIENT ARCHITECTURE")
    print("Sequence Segmentation + Temporal Validation")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Step 1: Validate temporal alignment
    validation_report = validate_temporal_alignment()

    # Step 2: Preprocess PXChange data
    pxchange_stats = preprocess_pxchange_data()

    # Step 3: Preprocess SeqofSeq data
    seqofseq_stats = preprocess_seqofseq_data()

    # Overall summary
    print("\n\n" + "="*80)
    print("OVERALL PREPROCESSING SUMMARY")
    print("="*80)

    if validation_report:
        print(f"\nTemporal Alignment:")
        print(f"  Has overlap: {validation_report['has_temporal_overlap']}")
        print(f"  Overlap days: {validation_report['overlap_days']}")
        print(f"  Warnings: {len(validation_report['warnings'])}")
        if validation_report['warnings']:
            for warning in validation_report['warnings']:
                print(f"    - {warning}")

    print(f"\nPXChange:")
    print(f"  Files: {pxchange_stats['files_processed']}")
    print(f"  Total segments: {pxchange_stats['total_segments']:,}")
    print(f"  Real patient segments: {pxchange_stats['real_patient_segments']:,}")
    print(f"  Pseudo-patient segments: {pxchange_stats['pseudo_patient_segments']:,}")
    print(f"  Events: {pxchange_stats['total_events']:,}")

    print(f"\nSeqofSeq:")
    print(f"  Files: {seqofseq_stats['files_processed']}")
    print(f"  Total segments: {seqofseq_stats['total_segments']:,}")
    print(f"  Real patient segments: {seqofseq_stats['real_patient_segments']:,}")
    print(f"  Pseudo-patient segments: {seqofseq_stats['pseudo_patient_segments']:,}")
    print(f"  Scans: {seqofseq_stats['total_scans']:,}")

    total_segments = pxchange_stats['total_segments'] + seqofseq_stats['total_segments']
    total_pseudo_patient = pxchange_stats['pseudo_patient_segments'] + seqofseq_stats['pseudo_patient_segments']
    total_data_points = pxchange_stats['total_events'] + seqofseq_stats['total_scans']

    print(f"\nCombined:")
    print(f"  Total segments: {total_segments:,}")
    print(f"  Pseudo-patient segments: {total_pseudo_patient:,}")
    print(f"  Pseudo-patient ratio: {total_pseudo_patient/total_segments*100:.2f}%" if total_segments > 0 else "  Pseudo-patient ratio: N/A")
    print(f"  Total data points: {total_data_points:,}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Save summary report
    report_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'preprocessing_summary.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, 'w') as f:
        f.write("DATA PREPROCESSING SUMMARY (Pseudo-Patient Architecture)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Pause threshold: {PAUSE_DETECTION_THRESHOLD_MINUTES} minutes\n")
        f.write(f"Architecture: Pseudo-Patient (Sequence Segmentation)\n\n")

        if validation_report:
            f.write("Temporal Alignment:\n")
            f.write(f"  Has overlap: {validation_report['has_temporal_overlap']}\n")
            f.write(f"  Overlap days: {validation_report['overlap_days']}\n")
            f.write(f"  Warnings: {len(validation_report['warnings'])}\n\n")

        f.write(f"PXChange: {pxchange_stats['files_processed']} files, ")
        f.write(f"{pxchange_stats['total_segments']} segments ")
        f.write(f"({pxchange_stats['real_patient_segments']} real, {pxchange_stats['pseudo_patient_segments']} pseudo)\n")

        f.write(f"SeqofSeq: {seqofseq_stats['files_processed']} files, ")
        f.write(f"{seqofseq_stats['total_segments']} segments ")
        f.write(f"({seqofseq_stats['real_patient_segments']} real, {seqofseq_stats['pseudo_patient_segments']} pseudo)\n\n")

        f.write(f"Total: {total_segments} segments, {total_pseudo_patient} pseudo-patient ")
        f.write(f"({total_pseudo_patient/total_segments*100:.2f}%)\n" if total_segments > 0 else "\n")

    print(f"\nSummary saved to: {report_path}")

    return {
        'pxchange': pxchange_stats,
        'seqofseq': seqofseq_stats,
        'total_segments': total_segments,
        'total_pseudo_patient': total_pseudo_patient,
        'total_data_points': total_data_points,
        'validation_report': validation_report
    }


if __name__ == "__main__":
    results = main()

    print("\n✓ Data preprocessing complete!")
    print("  Architecture: Pseudo-Patient (Sequence Segmentation)")
    print("  Next step: Train models with the segmented data")
