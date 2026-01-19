"""
Schedule Visualization

Creates Gantt charts of generated daily schedules with color-coded events.
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import timedelta

# Add paths
unified_dir = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, unified_dir)


def plot_daily_timeline(schedule_csv_path, save_path=None, show=True):
    """
    Plot a Gantt chart of the daily schedule.

    Args:
        schedule_csv_path: Path to event_timeline.csv
        save_path: Path to save plot (optional)
        show: Whether to display the plot

    Returns:
        fig: Matplotlib figure
    """
    # Load schedule
    df = pd.read_csv(schedule_csv_path)

    print(f"\n[Visualization] Loading schedule from: {schedule_csv_path}")
    print(f"  Total events: {len(df)}")
    print(f"  Sessions: {df['session_id'].nunique()}")

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))

    # Define colors for different event types
    colors = {
        'pxchange': '#3498db',  # Blue
        'scan': '#2ecc71',      # Green
        'pause': '#95a5a6',     # Gray
        'END': '#e74c3c'        # Red
    }

    # Plot each event as a horizontal bar
    for idx, row in df.iterrows():
        session_id = row['session_id']
        start_time = row['timestamp'] / 3600  # Convert to hours
        duration = row['duration'] / 3600     # Convert to hours
        event_type = row['event_type']

        # Determine color
        if row['sourceID'] == 'END':
            color = colors.get('END', '#95a5a6')
        else:
            color = colors.get(event_type, '#95a5a6')

        # Plot bar
        ax.barh(
            y=session_id,
            width=duration,
            left=start_time,
            height=0.8,
            color=color,
            edgecolor='white',
            linewidth=0.5,
            alpha=0.8
        )

        # Add label for longer events
        if duration > 0.1:  # Only label events longer than 6 minutes
            label = row['sourceID'] if row['sourceID'] else row['scan_sequence']
            if label and label != 'END':
                ax.text(
                    start_time + duration/2,
                    session_id,
                    label[:15],  # Truncate long labels
                    ha='center',
                    va='center',
                    fontsize=6,
                    color='white',
                    weight='bold'
                )

    # Formatting
    ax.set_xlabel('Time (hours from midnight)', fontsize=12, weight='bold')
    ax.set_ylabel('Session ID', fontsize=12, weight='bold')
    ax.set_title(f'Daily MRI Schedule\n{len(df)} events across {df["session_id"].nunique()} sessions',
                 fontsize=14, weight='bold', pad=20)

    # Set y-axis to show session IDs
    ax.set_yticks(range(1, df['session_id'].max() + 1))
    ax.set_ylim(0.5, df['session_id'].max() + 0.5)

    # Set x-axis to show full day (0-24 hours)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add legend
    legend_elements = [
        mpatches.Patch(color=colors['pxchange'], label='PXChange Events'),
        mpatches.Patch(color=colors['scan'], label='Scan Sequences'),
        mpatches.Patch(color=colors['pause'], label='Pauses'),
        mpatches.Patch(color=colors['END'], label='END Markers')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Add statistics box
    total_duration = df['duration'].sum() / 3600
    avg_session_duration = df.groupby('session_id')['duration'].sum().mean() / 60

    stats_text = f"Statistics:\n"
    stats_text += f"Total Duration: {total_duration:.2f} hours\n"
    stats_text += f"Avg Session: {avg_session_duration:.1f} minutes\n"
    stats_text += f"Events: {len(df)}"

    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[SAVED] Visualization: {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_session_comparison(schedule_csv_path, save_path=None, show=True):
    """
    Plot a comparison of session durations.

    Args:
        schedule_csv_path: Path to event_timeline.csv
        save_path: Path to save plot
        show: Whether to display the plot

    Returns:
        fig: Matplotlib figure
    """
    df = pd.read_csv(schedule_csv_path)

    # Calculate session durations
    session_durations = df.groupby('session_id').agg({
        'duration': 'sum',
        'event_id': 'count'
    }).reset_index()

    session_durations['duration_minutes'] = session_durations['duration'] / 60

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Session durations bar chart
    ax1.bar(
        session_durations['session_id'],
        session_durations['duration_minutes'],
        color='#3498db',
        edgecolor='white',
        linewidth=0.5
    )
    ax1.set_xlabel('Session ID', fontsize=11, weight='bold')
    ax1.set_ylabel('Duration (minutes)', fontsize=11, weight='bold')
    ax1.set_title('Session Durations', fontsize=12, weight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Event counts bar chart
    ax2.bar(
        session_durations['session_id'],
        session_durations['event_id'],
        color='#2ecc71',
        edgecolor='white',
        linewidth=0.5
    )
    ax2.set_xlabel('Session ID', fontsize=11, weight='bold')
    ax2.set_ylabel('Number of Events', fontsize=11, weight='bold')
    ax2.set_title('Events per Session', fontsize=12, weight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[SAVED] Session comparison: {save_path}")

    if show:
        plt.show()

    return fig


def print_schedule_summary(schedule_csv_path):
    """
    Print a text summary of the schedule.

    Args:
        schedule_csv_path: Path to event_timeline.csv
    """
    df = pd.read_csv(schedule_csv_path)

    print("\n" + "=" * 70)
    print("SCHEDULE SUMMARY")
    print("=" * 70)

    # Overall statistics
    total_events = len(df)
    total_sessions = df['session_id'].nunique()
    total_duration_hours = df['duration'].sum() / 3600

    print(f"\nOverall Statistics:")
    print(f"  Total Events: {total_events}")
    print(f"  Total Sessions: {total_sessions}")
    print(f"  Total Duration: {total_duration_hours:.2f} hours")

    # Event type breakdown
    event_type_counts = df['event_type'].value_counts()
    print(f"\nEvent Types:")
    for event_type, count in event_type_counts.items():
        print(f"  {event_type:15s}: {count:4d} ({100*count/total_events:.1f}%)")

    # Session breakdown
    print(f"\nSession Details:")
    for session_id in sorted(df['session_id'].unique()):
        session_df = df[df['session_id'] == session_id]
        session_duration = session_df['duration'].sum() / 60  # minutes
        num_events = len(session_df)
        start_time = session_df['timestamp'].min() / 3600  # hours

        print(f"  Session {session_id}: {num_events:3d} events, {session_duration:5.1f} min (start: {start_time:.1f}h)")

    # Time range
    start_time_hours = df['timestamp'].min() / 3600
    end_time_hours = df['timestamp'].max() / 3600
    print(f"\nTime Range:")
    print(f"  Start: {start_time_hours:.2f} hours ({int(start_time_hours)}:{int((start_time_hours % 1) * 60):02d})")
    print(f"  End:   {end_time_hours:.2f} hours ({int(end_time_hours)}:{int((end_time_hours % 1) * 60):02d})")

    print("\n" + "=" * 70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Visualize MRI schedules')
    parser.add_argument('schedule', type=str, help='Path to event_timeline.csv')
    parser.add_argument('--output', type=str, default=None, help='Output path for plot')
    parser.add_argument('--no-show', action='store_true', help='Do not display plot')
    parser.add_argument('--comparison', action='store_true', help='Also plot session comparison')
    parser.add_argument('--summary', action='store_true', help='Print text summary')

    args = parser.parse_args()

    # Print summary if requested
    if args.summary:
        print_schedule_summary(args.schedule)

    # Determine output path
    if args.output:
        timeline_output = args.output
    else:
        base = args.schedule.replace('.csv', '')
        timeline_output = f"{base}_timeline.png"

    # Plot timeline
    plot_daily_timeline(
        args.schedule,
        save_path=timeline_output,
        show=not args.no_show
    )

    # Plot comparison if requested
    if args.comparison:
        comparison_output = timeline_output.replace('_timeline.png', '_comparison.png')
        plot_session_comparison(
            args.schedule,
            save_path=comparison_output,
            show=not args.no_show
        )

    print("\n[SUCCESS] Visualization complete!")


if __name__ == "__main__":
    main()
