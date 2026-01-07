"""
Schedule Comparison Metrics
Compares synthetic schedules against ground truth

NOTE: This is a scaffold implementation. Will be fully implemented when ground truth data is available.
"""
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from collections import Counter
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================================
# QUALITY METRICS DEFINITIONS
# ============================================================================

QUALITY_METRICS = {
    # Temporal metrics
    'session_count_error': 'MAE(predicted_sessions, actual_sessions)',
    'session_timing_kl': 'KL divergence of start times',
    'daily_duration_error': 'MAE(total_time)',

    # Sequence metrics
    'sequence_diversity': 'Unique sequences / Total',
    'body_part_distribution': 'KL divergence',

    # Duration metrics
    'scan_duration_mae': 'MAE(durations)',
    'pause_duration_mae': 'MAE(pauses)',

    # Realism metrics
    'constraint_satisfaction': '% constraints satisfied'
}


# ============================================================================
# TEMPORAL METRICS
# ============================================================================

def compute_session_count_error(real_schedule, synthetic_schedule):
    """
    Compute Mean Absolute Error for session count

    Args:
        real_schedule: DataFrame with real schedule
        synthetic_schedule: DataFrame with synthetic schedule

    Returns:
        error: Absolute difference in session count
    """
    real_patients = real_schedule['patient_id'].nunique() if 'patient_id' in real_schedule.columns else 0
    synthetic_patients = synthetic_schedule['patient_id'].nunique() if 'patient_id' in synthetic_schedule.columns else 0

    error = abs(real_patients - synthetic_patients)
    return error


def compute_session_timing_divergence(real_schedule, synthetic_schedule):
    """
    Compute KL divergence for session start time distributions

    Args:
        real_schedule: DataFrame with real schedule
        synthetic_schedule: DataFrame with synthetic schedule

    Returns:
        kl_divergence: float
    """
    if 'start_time' not in real_schedule.columns or 'start_time' not in synthetic_schedule.columns:
        return None

    # Extract hour of day from start times
    real_hours = pd.to_datetime(real_schedule['start_time']).dt.hour
    synthetic_hours = pd.to_datetime(synthetic_schedule['start_time']).dt.hour

    # Create histograms (hour bins 0-23)
    real_hist, _ = np.histogram(real_hours, bins=24, range=(0, 24))
    synthetic_hist, _ = np.histogram(synthetic_hours, bins=24, range=(0, 24))

    # Normalize to probabilities
    real_prob = real_hist / real_hist.sum()
    synthetic_prob = synthetic_hist / synthetic_hist.sum()

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    real_prob = real_prob + epsilon
    synthetic_prob = synthetic_prob + epsilon
    real_prob = real_prob / real_prob.sum()
    synthetic_prob = synthetic_prob / synthetic_prob.sum()

    # Compute KL divergence
    kl_div = np.sum(real_prob * np.log(real_prob / synthetic_prob))

    return kl_div


def compute_daily_duration_error(real_schedule, synthetic_schedule):
    """
    Compute MAE for total daily duration

    Args:
        real_schedule: DataFrame with real schedule
        synthetic_schedule: DataFrame with synthetic schedule

    Returns:
        error_minutes: float
    """
    if 'duration' not in real_schedule.columns or 'duration' not in synthetic_schedule.columns:
        return None

    real_duration = real_schedule['duration'].sum() / 60  # Convert to minutes
    synthetic_duration = synthetic_schedule['duration'].sum() / 60

    error = abs(real_duration - synthetic_duration)
    return error


# ============================================================================
# SEQUENCE METRICS
# ============================================================================

def compute_sequence_diversity(schedule_df):
    """
    Compute sequence diversity score

    Args:
        schedule_df: DataFrame with schedule

    Returns:
        diversity: Unique sequences / Total sequences
    """
    if 'scan_sequence' not in schedule_df.columns:
        return None

    total_scans = len(schedule_df)
    unique_sequences = schedule_df['scan_sequence'].nunique()

    diversity = unique_sequences / total_scans if total_scans > 0 else 0
    return diversity


def compute_body_part_divergence(real_schedule, synthetic_schedule):
    """
    Compute KL divergence for body part distributions

    Args:
        real_schedule: DataFrame with real schedule
        synthetic_schedule: DataFrame with synthetic schedule

    Returns:
        kl_divergence: float
    """
    if 'body_part' not in real_schedule.columns or 'body_part' not in synthetic_schedule.columns:
        return None

    # Get body part counts
    real_counts = Counter(real_schedule['body_part'])
    synthetic_counts = Counter(synthetic_schedule['body_part'])

    # Get all unique body parts
    all_body_parts = set(real_counts.keys()) | set(synthetic_counts.keys())

    # Create probability distributions
    real_total = sum(real_counts.values())
    synthetic_total = sum(synthetic_counts.values())

    epsilon = 1e-10
    real_prob = {bp: (real_counts.get(bp, 0) + epsilon) / real_total for bp in all_body_parts}
    synthetic_prob = {bp: (synthetic_counts.get(bp, 0) + epsilon) / synthetic_total for bp in all_body_parts}

    # Normalize
    real_sum = sum(real_prob.values())
    synthetic_sum = sum(synthetic_prob.values())
    real_prob = {k: v/real_sum for k, v in real_prob.items()}
    synthetic_prob = {k: v/synthetic_sum for k, v in synthetic_prob.items()}

    # Compute KL divergence
    kl_div = sum(real_prob[bp] * np.log(real_prob[bp] / synthetic_prob[bp]) for bp in all_body_parts)

    return kl_div


# ============================================================================
# DURATION METRICS
# ============================================================================

def compute_scan_duration_mae(real_schedule, synthetic_schedule):
    """
    Compute MAE for scan durations

    Args:
        real_schedule: DataFrame with real schedule
        synthetic_schedule: DataFrame with synthetic schedule

    Returns:
        mae_seconds: float
    """
    if 'duration' not in real_schedule.columns or 'duration' not in synthetic_schedule.columns:
        return None

    # Compute mean durations
    real_mean = real_schedule['duration'].mean()
    synthetic_mean = synthetic_schedule['duration'].mean()

    mae = abs(real_mean - synthetic_mean)
    return mae


def compute_duration_distribution_similarity(real_schedule, synthetic_schedule):
    """
    Compute Kolmogorov-Smirnov statistic for duration distributions

    Args:
        real_schedule: DataFrame with real schedule
        synthetic_schedule: DataFrame with synthetic schedule

    Returns:
        ks_stat: float (0=identical, 1=completely different)
        p_value: float
    """
    if 'duration' not in real_schedule.columns or 'duration' not in synthetic_schedule.columns:
        return None, None

    real_durations = real_schedule['duration'].values
    synthetic_durations = synthetic_schedule['duration'].values

    ks_stat, p_value = ks_2samp(real_durations, synthetic_durations)
    return ks_stat, p_value


# ============================================================================
# CONSTRAINT METRICS
# ============================================================================

def validate_constraints(schedule_df):
    """
    Validate that schedule satisfies basic constraints

    Args:
        schedule_df: DataFrame with schedule

    Returns:
        violations: Dict with constraint violation counts
        satisfaction_rate: Overall constraint satisfaction percentage
    """
    violations = {}

    # Constraint 1: No overlapping scans
    if 'start_time' in schedule_df.columns and 'end_time' in schedule_df.columns:
        schedule_sorted = schedule_df.sort_values('start_time')
        overlaps = 0
        for i in range(len(schedule_sorted) - 1):
            if schedule_sorted.iloc[i]['end_time'] > schedule_sorted.iloc[i + 1]['start_time']:
                overlaps += 1
        violations['overlapping_scans'] = overlaps

    # Constraint 2: Reasonable scan durations (10-500 seconds)
    if 'duration' in schedule_df.columns:
        invalid_durations = ((schedule_df['duration'] < 10) | (schedule_df['duration'] > 500)).sum()
        violations['invalid_durations'] = invalid_durations

    # Constraint 3: Monotonic time progression
    if 'start_time' in schedule_df.columns:
        start_times = pd.to_datetime(schedule_df['start_time'])
        non_monotonic = (~start_times.is_monotonic_increasing)
        violations['non_monotonic_time'] = 1 if non_monotonic else 0

    # Constraint 4: All required fields present
    required_fields = ['patient_id', 'start_time', 'duration']
    missing_fields = sum(schedule_df[field].isnull().sum() for field in required_fields if field in schedule_df.columns)
    violations['missing_required_fields'] = missing_fields

    # Compute satisfaction rate
    total_checks = sum(violations.values())
    total_possible = len(schedule_df) * len(violations)
    satisfaction_rate = (1 - total_checks / total_possible) * 100 if total_possible > 0 else 100

    return violations, satisfaction_rate


# ============================================================================
# MAIN COMPARISON FUNCTION
# ============================================================================

def compare_schedules(real_schedule, synthetic_schedule):
    """
    Compute all comparison metrics between real and synthetic schedules

    Args:
        real_schedule: DataFrame with real schedule
        synthetic_schedule: DataFrame with synthetic schedule

    Returns:
        comparison_report: Dictionary with all metrics
    """
    report = {
        'timestamp': pd.Timestamp.now(),
        'real_schedule_events': len(real_schedule),
        'synthetic_schedule_events': len(synthetic_schedule),
        'metrics': {}
    }

    # Temporal metrics
    report['metrics']['session_count_error'] = compute_session_count_error(real_schedule, synthetic_schedule)
    report['metrics']['session_timing_kl'] = compute_session_timing_divergence(real_schedule, synthetic_schedule)
    report['metrics']['daily_duration_error'] = compute_daily_duration_error(real_schedule, synthetic_schedule)

    # Sequence metrics
    report['metrics']['real_sequence_diversity'] = compute_sequence_diversity(real_schedule)
    report['metrics']['synthetic_sequence_diversity'] = compute_sequence_diversity(synthetic_schedule)
    report['metrics']['body_part_kl'] = compute_body_part_divergence(real_schedule, synthetic_schedule)

    # Duration metrics
    report['metrics']['scan_duration_mae'] = compute_scan_duration_mae(real_schedule, synthetic_schedule)
    ks_stat, p_value = compute_duration_distribution_similarity(real_schedule, synthetic_schedule)
    report['metrics']['duration_ks_stat'] = ks_stat
    report['metrics']['duration_ks_pvalue'] = p_value

    # Constraint metrics
    real_violations, real_satisfaction = validate_constraints(real_schedule)
    synthetic_violations, synthetic_satisfaction = validate_constraints(synthetic_schedule)
    report['metrics']['real_constraint_satisfaction'] = real_satisfaction
    report['metrics']['synthetic_constraint_satisfaction'] = synthetic_satisfaction
    report['real_violations'] = real_violations
    report['synthetic_violations'] = synthetic_violations

    # Compute overall quality score (0-100)
    quality_score = compute_quality_score(report)
    report['quality_score'] = quality_score

    return report


def compute_quality_score(comparison_report):
    """
    Compute overall quality score from comparison metrics

    Args:
        comparison_report: Dictionary with comparison metrics

    Returns:
        quality_score: Float (0-100)
    """
    metrics = comparison_report['metrics']
    scores = []

    # Session count (100 if exact, decrease with error)
    if metrics.get('session_count_error') is not None:
        session_score = max(0, 100 - metrics['session_count_error'] * 10)
        scores.append(session_score)

    # Timing divergence (100 if KL=0, decrease with divergence)
    if metrics.get('session_timing_kl') is not None:
        timing_score = max(0, 100 - metrics['session_timing_kl'] * 50)
        scores.append(timing_score)

    # Duration error (100 if exact, decrease with error)
    if metrics.get('daily_duration_error') is not None:
        duration_score = max(0, 100 - metrics['daily_duration_error'] / 10)
        scores.append(duration_score)

    # Constraint satisfaction (direct percentage)
    if metrics.get('synthetic_constraint_satisfaction') is not None:
        scores.append(metrics['synthetic_constraint_satisfaction'])

    # Average all scores
    quality_score = np.mean(scores) if scores else 0
    return quality_score


def print_comparison_report(comparison_report):
    """
    Pretty print comparison report

    Args:
        comparison_report: Dictionary with comparison metrics
    """
    print("\n" + "="*70)
    print("SCHEDULE COMPARISON REPORT")
    print("="*70)

    print(f"\nTimestamp: {comparison_report['timestamp']}")
    print(f"Real schedule events: {comparison_report['real_schedule_events']}")
    print(f"Synthetic schedule events: {comparison_report['synthetic_schedule_events']}")

    print(f"\n{'Metric':<35} {'Value':<20} {'Status'}")
    print("-"*70)

    metrics = comparison_report['metrics']

    # Temporal
    print(f"{'Session Count Error':<35} {metrics.get('session_count_error', 'N/A'):<20} {'✓' if metrics.get('session_count_error', 999) < 3 else '⚠️'}")
    print(f"{'Session Timing KL Div':<35} {metrics.get('session_timing_kl', 'N/A'):<20.3f} {'✓' if metrics.get('session_timing_kl', 999) < 0.5 else '⚠️'}")
    print(f"{'Daily Duration Error (min)':<35} {metrics.get('daily_duration_error', 'N/A'):<20.1f} {'✓' if metrics.get('daily_duration_error', 999) < 30 else '⚠️'}")

    # Sequence
    print(f"{'Real Sequence Diversity':<35} {metrics.get('real_sequence_diversity', 'N/A'):<20.3f} -")
    print(f"{'Synthetic Sequence Diversity':<35} {metrics.get('synthetic_sequence_diversity', 'N/A'):<20.3f} -")
    print(f"{'Body Part KL Div':<35} {metrics.get('body_part_kl', 'N/A'):<20.3f} {'✓' if metrics.get('body_part_kl', 999) < 0.5 else '⚠️'}")

    # Duration
    print(f"{'Scan Duration MAE (sec)':<35} {metrics.get('scan_duration_mae', 'N/A'):<20.1f} {'✓' if metrics.get('scan_duration_mae', 999) < 60 else '⚠️'}")

    # Constraints
    print(f"{'Constraint Satisfaction (%)':<35} {metrics.get('synthetic_constraint_satisfaction', 'N/A'):<20.1f} {'✓' if metrics.get('synthetic_constraint_satisfaction', 0) > 95 else '⚠️'}")

    print(f"\n{'OVERALL QUALITY SCORE':<35} {comparison_report['quality_score']:<20.1f} {'✓' if comparison_report['quality_score'] > 75 else '⚠️'}")
    print("="*70)


# ============================================================================
# MAIN ENTRY POINT (For Testing)
# ============================================================================

if __name__ == "__main__":
    print("Schedule Comparator (Scaffold)")
    print("=" * 70)
    print("\nThis module compares synthetic schedules against ground truth.")
    print("\nMetrics implemented:")
    for metric, description in QUALITY_METRICS.items():
        print(f"  - {metric}: {description}")
    print("\nUse schedule_comparator.compare_schedules(real, synthetic) to compare.")
