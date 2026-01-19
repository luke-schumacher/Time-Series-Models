"""
Validation metrics for comparing real vs predicted schedules.

Used for dashboard integration and model evaluation.
"""
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats


def compare_real_vs_predicted(real_schedule, predicted_schedule):
    """
    Compare a real day schedule to a predicted one.

    Args:
        real_schedule: List of event dicts from real data
        predicted_schedule: List of event dicts from day simulator

    Returns:
        Dict of comparison metrics for dashboard
    """
    real_df = pd.DataFrame(real_schedule)
    pred_df = pd.DataFrame(predicted_schedule)

    metrics = {
        'summary': {},
        'per_body_region': {},
        'per_exchange_type': {},
        'timeline_comparison': []
    }

    # === Summary Metrics ===
    real_duration = real_df['duration'].sum() if 'duration' in real_df else 0
    pred_duration = pred_df['duration'].sum() if 'duration' in pred_df else 0

    metrics['summary'] = {
        'total_duration_real': real_duration,
        'total_duration_predicted': pred_duration,
        'duration_error_pct': abs(real_duration - pred_duration) / max(real_duration, 1) * 100,
        'num_events_real': len(real_df),
        'num_events_predicted': len(pred_df),
        'num_examinations_real': len(real_df[real_df['event_type'] == 'examination']) if 'event_type' in real_df else 0,
        'num_examinations_predicted': len(pred_df[pred_df['event_type'] == 'examination']) if 'event_type' in pred_df else 0,
        'num_exchanges_real': len(real_df[real_df['event_type'] == 'exchange']) if 'event_type' in real_df else 0,
        'num_exchanges_predicted': len(pred_df[pred_df['event_type'] == 'exchange']) if 'event_type' in pred_df else 0,
    }

    # === Per Body Region Metrics ===
    if 'body_region' in real_df.columns:
        for region in real_df['body_region'].dropna().unique():
            real_region = real_df[real_df['body_region'] == region]
            pred_region = pred_df[pred_df['body_region'] == region] if 'body_region' in pred_df else pd.DataFrame()

            metrics['per_body_region'][region] = {
                'count_real': len(real_region),
                'count_predicted': len(pred_region),
                'avg_duration_real': real_region['duration'].mean() if len(real_region) > 0 else 0,
                'avg_duration_predicted': pred_region['duration'].mean() if len(pred_region) > 0 else 0,
                'std_duration_real': real_region['duration'].std() if len(real_region) > 0 else 0,
                'std_duration_predicted': pred_region['duration'].std() if len(pred_region) > 0 else 0,
            }

    # === Per Exchange Type Metrics ===
    if 'body_from' in real_df.columns and 'body_to' in real_df.columns:
        real_exchanges = real_df[real_df['event_type'] == 'exchange'] if 'event_type' in real_df else real_df
        pred_exchanges = pred_df[pred_df['event_type'] == 'exchange'] if 'event_type' in pred_df else pred_df

        for _, row in real_exchanges.iterrows():
            key = f"{row.get('body_from', 'UNK')}_to_{row.get('body_to', 'UNK')}"
            if key not in metrics['per_exchange_type']:
                metrics['per_exchange_type'][key] = {
                    'count_real': 0,
                    'count_predicted': 0,
                    'durations_real': [],
                    'durations_predicted': []
                }
            metrics['per_exchange_type'][key]['count_real'] += 1
            metrics['per_exchange_type'][key]['durations_real'].append(row.get('duration', 0))

        for _, row in pred_exchanges.iterrows():
            key = f"{row.get('body_from', 'UNK')}_to_{row.get('body_to', 'UNK')}"
            if key not in metrics['per_exchange_type']:
                metrics['per_exchange_type'][key] = {
                    'count_real': 0,
                    'count_predicted': 0,
                    'durations_real': [],
                    'durations_predicted': []
                }
            metrics['per_exchange_type'][key]['count_predicted'] += 1
            metrics['per_exchange_type'][key]['durations_predicted'].append(row.get('duration', 0))

        # Compute averages
        for key in metrics['per_exchange_type']:
            data = metrics['per_exchange_type'][key]
            data['avg_duration_real'] = np.mean(data['durations_real']) if data['durations_real'] else 0
            data['avg_duration_predicted'] = np.mean(data['durations_predicted']) if data['durations_predicted'] else 0
            del data['durations_real']
            del data['durations_predicted']

    return metrics


def compute_bucket_metrics(bucket_samples, ground_truth_samples):
    """
    Compute validation metrics for a single bucket.

    Args:
        bucket_samples: List of generated samples from bucket
        ground_truth_samples: List of ground truth samples

    Returns:
        Dict of bucket-level metrics
    """
    metrics = {}

    # Extract durations
    bucket_durations = [s.get('total_duration', 0) for s in bucket_samples]
    gt_durations = [s.get('total_duration', 0) for s in ground_truth_samples]

    if len(bucket_durations) > 0 and len(gt_durations) > 0:
        # Duration distribution comparison
        metrics['duration_mean_generated'] = np.mean(bucket_durations)
        metrics['duration_mean_ground_truth'] = np.mean(gt_durations)
        metrics['duration_std_generated'] = np.std(bucket_durations)
        metrics['duration_std_ground_truth'] = np.std(gt_durations)

        # KS test for distribution similarity
        if len(bucket_durations) > 1 and len(gt_durations) > 1:
            ks_stat, ks_pvalue = stats.ks_2samp(bucket_durations, gt_durations)
            metrics['duration_ks_statistic'] = ks_stat
            metrics['duration_ks_pvalue'] = ks_pvalue

    # Sequence length comparison
    bucket_lengths = [len(s.get('sequence', [])) for s in bucket_samples]
    gt_lengths = [len(s.get('sequence', [])) for s in ground_truth_samples]

    if len(bucket_lengths) > 0 and len(gt_lengths) > 0:
        metrics['seq_length_mean_generated'] = np.mean(bucket_lengths)
        metrics['seq_length_mean_ground_truth'] = np.mean(gt_lengths)

    # Token frequency comparison
    bucket_tokens = []
    gt_tokens = []

    for s in bucket_samples:
        bucket_tokens.extend(s.get('sequence', []))
    for s in ground_truth_samples:
        gt_tokens.extend(s.get('sequence', []))

    if len(bucket_tokens) > 0 and len(gt_tokens) > 0:
        bucket_counter = Counter(bucket_tokens)
        gt_counter = Counter(gt_tokens)

        # Normalize to frequencies
        bucket_total = sum(bucket_counter.values())
        gt_total = sum(gt_counter.values())

        all_tokens = set(bucket_counter.keys()) | set(gt_counter.keys())

        bucket_freq = {t: bucket_counter.get(t, 0) / bucket_total for t in all_tokens}
        gt_freq = {t: gt_counter.get(t, 0) / gt_total for t in all_tokens}

        # Cosine similarity of token frequencies
        tokens = list(all_tokens)
        v1 = np.array([bucket_freq[t] for t in tokens])
        v2 = np.array([gt_freq[t] for t in tokens])

        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        metrics['token_frequency_cosine_similarity'] = cos_sim

    return metrics


def print_comparison_report(metrics):
    """
    Print a formatted comparison report.

    Args:
        metrics: Dict from compare_real_vs_predicted
    """
    print("\n" + "=" * 60)
    print("REAL VS PREDICTED COMPARISON REPORT")
    print("=" * 60)

    # Summary
    print("\n--- SUMMARY ---")
    summary = metrics['summary']
    print(f"Total Duration: Real={summary['total_duration_real']:.1f}s, "
          f"Predicted={summary['total_duration_predicted']:.1f}s "
          f"(Error: {summary['duration_error_pct']:.1f}%)")
    print(f"Total Events: Real={summary['num_events_real']}, "
          f"Predicted={summary['num_events_predicted']}")
    print(f"Examinations: Real={summary['num_examinations_real']}, "
          f"Predicted={summary['num_examinations_predicted']}")
    print(f"Exchanges: Real={summary['num_exchanges_real']}, "
          f"Predicted={summary['num_exchanges_predicted']}")

    # Per body region
    if metrics['per_body_region']:
        print("\n--- PER BODY REGION ---")
        for region, data in metrics['per_body_region'].items():
            print(f"  {region}: Real={data['count_real']} events "
                  f"(avg {data['avg_duration_real']:.1f}s), "
                  f"Predicted={data['count_predicted']} events "
                  f"(avg {data['avg_duration_predicted']:.1f}s)")

    # Per exchange type
    if metrics['per_exchange_type']:
        print("\n--- PER EXCHANGE TYPE ---")
        for exchange_type, data in list(metrics['per_exchange_type'].items())[:10]:  # Top 10
            print(f"  {exchange_type}: Real={data['count_real']}, "
                  f"Predicted={data['count_predicted']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("Testing validation metrics...")
    print("=" * 60)

    # Create sample data
    real_schedule = [
        {'event_type': 'exchange', 'body_from': 'START', 'body_to': 'HEAD', 'duration': 60},
        {'event_type': 'examination', 'body_region': 'HEAD', 'duration': 300},
        {'event_type': 'exchange', 'body_from': 'HEAD', 'body_to': 'CHEST', 'duration': 45},
        {'event_type': 'examination', 'body_region': 'CHEST', 'duration': 250},
    ]

    predicted_schedule = [
        {'event_type': 'exchange', 'body_from': 'START', 'body_to': 'HEAD', 'duration': 55},
        {'event_type': 'examination', 'body_region': 'HEAD', 'duration': 320},
        {'event_type': 'exchange', 'body_from': 'HEAD', 'body_to': 'CHEST', 'duration': 50},
        {'event_type': 'examination', 'body_region': 'CHEST', 'duration': 240},
    ]

    metrics = compare_real_vs_predicted(real_schedule, predicted_schedule)
    print_comparison_report(metrics)
