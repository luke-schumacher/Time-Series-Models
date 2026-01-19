"""
Temporal Alignment Validation
Ensures temporal consistency between PXChange and SeqofSeq datasets

Addresses the issue where exchange data from one day (e.g., Monday) is matched
with sequence data from another day (e.g., Tuesday), which creates temporal
mismatches in training data and affects model performance.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================================
# TEMPORAL INDEX EXTRACTION
# ============================================================================

def extract_temporal_index(df, datetime_column='datetime', id_column='PatientID'):
    """
    Extract temporal index from dataset

    Creates a summary of temporal patterns in the dataset including:
    - Date range
    - Day-of-week distribution
    - Time-of-day distribution
    - Temporal gaps

    Args:
        df: DataFrame with datetime column
        datetime_column: Name of datetime column
        id_column: Name of patient/sequence ID column

    Returns:
        temporal_index: Dictionary with temporal metadata
    """
    df = df.copy()

    # Ensure datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_column]):
        df[datetime_column] = pd.to_datetime(df[datetime_column])

    # Extract temporal features
    df['date'] = df[datetime_column].dt.date
    df['day_of_week'] = df[datetime_column].dt.dayofweek  # 0=Monday, 6=Sunday
    df['hour_of_day'] = df[datetime_column].dt.hour
    df['week_of_year'] = df[datetime_column].dt.isocalendar().week

    # Compute temporal statistics
    temporal_index = {
        'date_range': {
            'start': df[datetime_column].min(),
            'end': df[datetime_column].max(),
            'duration_days': (df[datetime_column].max() - df[datetime_column].min()).days
        },
        'unique_dates': df['date'].nunique(),
        'total_events': len(df),
        'unique_ids': df[id_column].nunique() if id_column in df.columns else None,

        # Day-of-week distribution
        'day_of_week_distribution': dict(Counter(df['day_of_week'])),

        # Hour-of-day distribution
        'hour_of_day_distribution': dict(Counter(df['hour_of_day'])),

        # Week-of-year coverage
        'weeks_covered': sorted(df['week_of_year'].unique().tolist()),

        # Working hours analysis
        'is_business_hours': ((df['hour_of_day'] >= 7) & (df['hour_of_day'] <= 19)).sum() / len(df),
        'is_weekend': (df['day_of_week'] >= 5).sum() / len(df),
    }

    # Add date-level summary
    date_summary = df.groupby('date').agg({
        datetime_column: 'count',
        id_column: 'nunique' if id_column in df.columns else 'count'
    }).reset_index()
    date_summary.columns = ['date', 'events_per_day', 'ids_per_day']

    temporal_index['date_summary'] = date_summary
    temporal_index['avg_events_per_day'] = date_summary['events_per_day'].mean()
    temporal_index['std_events_per_day'] = date_summary['events_per_day'].std()

    return temporal_index


# ============================================================================
# TEMPORAL CONSISTENCY VALIDATION
# ============================================================================

def validate_temporal_consistency(pxchange_df, seqofseq_df,
                                  pxchange_datetime_col='datetime',
                                  seqofseq_datetime_col='startTime',
                                  verbose=True):
    """
    Validate temporal consistency between PXChange and SeqofSeq datasets

    Checks for:
    1. Different date ranges
    2. Different day-of-week distributions
    3. Large temporal gaps
    4. Mismatched time-of-day patterns

    Args:
        pxchange_df: PXChange DataFrame
        seqofseq_df: SeqofSeq DataFrame
        pxchange_datetime_col: Datetime column name in PXChange
        seqofseq_datetime_col: Datetime column name in SeqofSeq
        verbose: Print detailed warnings

    Returns:
        validation_report: Dictionary with validation results
        warnings: List of warning messages
    """
    if verbose:
        print("Validating temporal consistency between PXChange and SeqofSeq datasets...")
        print("=" * 70)

    warnings = []

    # Extract temporal indices
    px_index = extract_temporal_index(pxchange_df, pxchange_datetime_col, 'PatientId')
    seq_index = extract_temporal_index(seqofseq_df, seqofseq_datetime_col, 'PatientID')

    # 1. Check date range overlap
    px_start = px_index['date_range']['start']
    px_end = px_index['date_range']['end']
    seq_start = seq_index['date_range']['start']
    seq_end = seq_index['date_range']['end']

    overlap_start = max(px_start, seq_start)
    overlap_end = min(px_end, seq_end)

    if overlap_start > overlap_end:
        warnings.append("CRITICAL: No temporal overlap between datasets!")
        has_overlap = False
        overlap_days = 0
    else:
        has_overlap = True
        overlap_days = (overlap_end - overlap_start).days

    if verbose:
        print(f"\n1. Date Range Analysis:")
        print(f"   PXChange: {px_start.date()} to {px_end.date()} ({px_index['date_range']['duration_days']} days)")
        print(f"   SeqofSeq: {seq_start.date()} to {seq_end.date()} ({seq_index['date_range']['duration_days']} days)")
        print(f"   Overlap:  {overlap_start.date()} to {overlap_end.date()} ({overlap_days} days)")

        if has_overlap:
            px_coverage = overlap_days / px_index['date_range']['duration_days'] * 100
            seq_coverage = overlap_days / seq_index['date_range']['duration_days'] * 100
            print(f"   Coverage: PXChange={px_coverage:.1f}%, SeqofSeq={seq_coverage:.1f}%")
        else:
            print(f"   ⚠️  WARNING: No overlap!")

    # 2. Check day-of-week distribution
    px_dow = px_index['day_of_week_distribution']
    seq_dow = seq_index['day_of_week_distribution']

    # Normalize distributions
    px_dow_total = sum(px_dow.values())
    seq_dow_total = sum(seq_dow.values())
    px_dow_norm = {k: v/px_dow_total for k, v in px_dow.items()}
    seq_dow_norm = {k: v/seq_dow_total for k, v in seq_dow.items()}

    # Compute KL divergence (approximate)
    dow_divergence = 0
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for day in range(7):
        p = px_dow_norm.get(day, 0.001)
        q = seq_dow_norm.get(day, 0.001)
        dow_divergence += p * np.log(p / q) if p > 0 and q > 0 else 0

    if dow_divergence > 0.5:
        warnings.append(f"Day-of-week distributions differ significantly (KL divergence: {dow_divergence:.3f})")

    if verbose:
        print(f"\n2. Day-of-Week Distribution:")
        print(f"   {'Day':<8} {'PXChange':<12} {'SeqofSeq':<12} {'Difference':<12}")
        print(f"   {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
        for day in range(7):
            px_pct = px_dow_norm.get(day, 0) * 100
            seq_pct = seq_dow_norm.get(day, 0) * 100
            diff = abs(px_pct - seq_pct)
            marker = "⚠️ " if diff > 10 else "  "
            print(f"   {day_names[day]:<8} {px_pct:>10.1f}% {seq_pct:>10.1f}% {marker}{diff:>9.1f}%")
        print(f"   KL Divergence: {dow_divergence:.3f}")

    # 3. Check time-of-day patterns
    px_business_hours = px_index['is_business_hours'] * 100
    seq_business_hours = seq_index['is_business_hours'] * 100
    px_weekend = px_index['is_weekend'] * 100
    seq_weekend = seq_index['is_weekend'] * 100

    if abs(px_business_hours - seq_business_hours) > 20:
        warnings.append(f"Business hours patterns differ: PXChange={px_business_hours:.1f}%, SeqofSeq={seq_business_hours:.1f}%")

    if abs(px_weekend - seq_weekend) > 20:
        warnings.append(f"Weekend patterns differ: PXChange={px_weekend:.1f}%, SeqofSeq={seq_weekend:.1f}%")

    if verbose:
        print(f"\n3. Time-of-Day Patterns:")
        print(f"   {'Pattern':<20} {'PXChange':<12} {'SeqofSeq':<12} {'Difference':<12}")
        print(f"   {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
        print(f"   {'Business Hours':<20} {px_business_hours:>10.1f}% {seq_business_hours:>10.1f}% {abs(px_business_hours - seq_business_hours):>10.1f}%")
        print(f"   {'Weekend Events':<20} {px_weekend:>10.1f}% {seq_weekend:>10.1f}% {abs(px_weekend - seq_weekend):>10.1f}%")

    # 4. Check data density
    px_density = px_index['avg_events_per_day']
    seq_density = seq_index['avg_events_per_day']

    if verbose:
        print(f"\n4. Data Density:")
        print(f"   PXChange: {px_density:.1f} ± {px_index['std_events_per_day']:.1f} events/day")
        print(f"   SeqofSeq: {seq_density:.1f} ± {seq_index['std_events_per_day']:.1f} events/day")

    # Create validation report
    validation_report = {
        'has_temporal_overlap': has_overlap,
        'overlap_days': overlap_days,
        'overlap_start': overlap_start,
        'overlap_end': overlap_end,
        'day_of_week_kl_divergence': dow_divergence,
        'business_hours_diff': abs(px_business_hours - seq_business_hours),
        'weekend_diff': abs(px_weekend - seq_weekend),
        'pxchange_index': px_index,
        'seqofseq_index': seq_index,
        'warnings': warnings,
        'is_aligned': len(warnings) == 0 and has_overlap and overlap_days > 7
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"Validation Summary:")
        print(f"  Temporal overlap: {'✓' if has_overlap else '✗'}")
        print(f"  Day-of-week alignment: {'✓' if dow_divergence < 0.5 else '⚠️ '}")
        print(f"  Time-of-day alignment: {'✓' if validation_report['business_hours_diff'] < 20 else '⚠️ '}")
        print(f"  Overall status: {'ALIGNED ✓' if validation_report['is_aligned'] else 'MISALIGNED ⚠️ '}")

        if warnings:
            print(f"\nWarnings ({len(warnings)}):")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")
        print()

    return validation_report, warnings


# ============================================================================
# TEMPORAL ALIGNMENT (Filtering)
# ============================================================================

def align_datasets_temporally(pxchange_df, seqofseq_df,
                              pxchange_datetime_col='datetime',
                              seqofseq_datetime_col='startTime',
                              strict=False,
                              verbose=True):
    """
    Filter datasets to overlapping temporal windows

    Args:
        pxchange_df: PXChange DataFrame
        seqofseq_df: SeqofSeq DataFrame
        pxchange_datetime_col: Datetime column name in PXChange
        seqofseq_datetime_col: Datetime column name in SeqofSeq
        strict: If True, filter to exact date overlap. If False, just return original data with warnings.
        verbose: Print alignment information

    Returns:
        pxchange_aligned: Filtered PXChange DataFrame
        seqofseq_aligned: Filtered SeqofSeq DataFrame
        alignment_info: Dictionary with alignment metadata
    """
    if verbose:
        print("Aligning datasets temporally...")

    # Validate consistency first
    validation_report, warnings = validate_temporal_consistency(
        pxchange_df, seqofseq_df,
        pxchange_datetime_col, seqofseq_datetime_col,
        verbose=False
    )

    if not strict:
        # Non-strict mode: just log warnings and return original data
        if verbose:
            print("Non-strict mode: Returning original data with validation warnings")
            if warnings:
                print("\nWarnings:")
                for warning in warnings:
                    print(f"  - {warning}")
        return pxchange_df, seqofseq_df, validation_report

    # Strict mode: filter to overlapping dates
    if not validation_report['has_temporal_overlap']:
        raise ValueError("No temporal overlap between datasets! Cannot align.")

    overlap_start = validation_report['overlap_start']
    overlap_end = validation_report['overlap_end']

    if verbose:
        print(f"Filtering to overlap period: {overlap_start.date()} to {overlap_end.date()}")

    # Ensure datetime types
    px_df = pxchange_df.copy()
    seq_df = seqofseq_df.copy()

    if not pd.api.types.is_datetime64_any_dtype(px_df[pxchange_datetime_col]):
        px_df[pxchange_datetime_col] = pd.to_datetime(px_df[pxchange_datetime_col])

    if not pd.api.types.is_datetime64_any_dtype(seq_df[seqofseq_datetime_col]):
        seq_df[seqofseq_datetime_col] = pd.to_datetime(seq_df[seqofseq_datetime_col])

    # Filter to overlap
    px_mask = (px_df[pxchange_datetime_col] >= overlap_start) & (px_df[pxchange_datetime_col] <= overlap_end)
    seq_mask = (seq_df[seqofseq_datetime_col] >= overlap_start) & (seq_df[seqofseq_datetime_col] <= overlap_end)

    pxchange_aligned = px_df[px_mask].reset_index(drop=True)
    seqofseq_aligned = seq_df[seq_mask].reset_index(drop=True)

    # Compute alignment statistics
    alignment_info = {
        'overlap_start': overlap_start,
        'overlap_end': overlap_end,
        'overlap_days': (overlap_end - overlap_start).days,
        'pxchange_before': len(pxchange_df),
        'pxchange_after': len(pxchange_aligned),
        'pxchange_retention': len(pxchange_aligned) / len(pxchange_df) * 100,
        'seqofseq_before': len(seqofseq_df),
        'seqofseq_after': len(seqofseq_aligned),
        'seqofseq_retention': len(seqofseq_aligned) / len(seqofseq_df) * 100,
        'validation_report': validation_report
    }

    if verbose:
        print(f"\nAlignment Results:")
        print(f"  PXChange: {alignment_info['pxchange_before']} → {alignment_info['pxchange_after']} rows ({alignment_info['pxchange_retention']:.1f}% retained)")
        print(f"  SeqofSeq: {alignment_info['seqofseq_before']} → {alignment_info['seqofseq_after']} rows ({alignment_info['seqofseq_retention']:.1f}% retained)")
        print()

    return pxchange_aligned, seqofseq_aligned, alignment_info


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_temporal_alignment_report(validation_report, output_path):
    """
    Generate a detailed temporal alignment report

    Args:
        validation_report: Output from validate_temporal_consistency()
        output_path: Path to save report (txt file)
    """
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TEMPORAL ALIGNMENT VALIDATION REPORT\n")
        f.write("="*70 + "\n\n")

        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Overall status
        f.write("Overall Status: ")
        if validation_report['is_aligned']:
            f.write("ALIGNED ✓\n\n")
        else:
            f.write("MISALIGNED ⚠️\n\n")

        # Date range overlap
        f.write("1. Date Range Overlap:\n")
        f.write(f"   Has Overlap: {validation_report['has_temporal_overlap']}\n")
        f.write(f"   Overlap Days: {validation_report['overlap_days']}\n")
        f.write(f"   Overlap Period: {validation_report['overlap_start'].date()} to {validation_report['overlap_end'].date()}\n\n")

        # Day-of-week alignment
        f.write("2. Day-of-Week Alignment:\n")
        f.write(f"   KL Divergence: {validation_report['day_of_week_kl_divergence']:.3f}\n")
        f.write(f"   Status: {'✓ Aligned' if validation_report['day_of_week_kl_divergence'] < 0.5 else '⚠️  Misaligned'}\n\n")

        # Time-of-day alignment
        f.write("3. Time-of-Day Alignment:\n")
        f.write(f"   Business Hours Difference: {validation_report['business_hours_diff']:.1f}%\n")
        f.write(f"   Weekend Difference: {validation_report['weekend_diff']:.1f}%\n\n")

        # Warnings
        if validation_report['warnings']:
            f.write(f"4. Warnings ({len(validation_report['warnings'])}):\n")
            for i, warning in enumerate(validation_report['warnings'], 1):
                f.write(f"   {i}. {warning}\n")
        else:
            f.write("4. Warnings: None\n")

        f.write("\n" + "="*70 + "\n")

    print(f"Temporal alignment report saved to: {output_path}")


# ============================================================================
# MAIN ENTRY POINT (For Testing)
# ============================================================================

if __name__ == "__main__":
    print("Temporal Alignment Validation Module")
    print("This module validates temporal consistency between datasets")
    print("\nUsage:")
    print("  from temporal_alignment import validate_temporal_consistency, align_datasets_temporally")
    print("\nSee preprocess_all_data.py for integration examples.")
