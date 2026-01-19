"""
Visualization utilities for creating Plotly charts
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Dict, Optional


# Modern color palette
COLORS = {
    'primary': '#2563eb',
    'secondary': '#7c3aed',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'info': '#06b6d4',
    'purple': '#a855f7',
    'pink': '#ec4899',
}

COLOR_SCALE = px.colors.sequential.Blues

# Dark theme template
DARK_THEME = {
    'paper_bgcolor': '#1e293b',
    'plot_bgcolor': '#0f172a',
    'font': {'color': '#f1f5f9', 'family': 'Inter, system-ui, sans-serif'},
    'xaxis': {
        'gridcolor': '#334155',
        'zerolinecolor': '#475569',
    },
    'yaxis': {
        'gridcolor': '#334155',
        'zerolinecolor': '#475569',
    },
}


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Apply dark theme to a Plotly figure"""
    fig.update_layout(
        paper_bgcolor=DARK_THEME['paper_bgcolor'],
        plot_bgcolor=DARK_THEME['plot_bgcolor'],
        font=DARK_THEME['font'],
        xaxis=DARK_THEME['xaxis'],
        yaxis=DARK_THEME['yaxis'],
        hovermode='closest',
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def create_sequence_timeline(df: pd.DataFrame, sample_id: int) -> go.Figure:
    """
    Create a Gantt-style timeline for a sequence

    Args:
        df: DataFrame with sequence data
        sample_id: Sample ID to visualize

    Returns:
        Plotly figure
    """
    sample_df = df[df['sample_id'] == sample_id].copy()
    sample_df = sample_df.sort_values('step')

    # Calculate start and end times
    sample_df['start'] = sample_df['predicted_duration'].cumsum() - sample_df['predicted_duration']
    sample_df['end'] = sample_df['predicted_duration'].cumsum()

    # Create figure
    fig = go.Figure()

    # Create bars for each sequence step
    for idx, row in sample_df.iterrows():
        fig.add_trace(go.Bar(
            name=row['sequence_name'],
            x=[row['predicted_duration']],
            y=[f"Step {row['step']}"],
            orientation='h',
            marker=dict(
                color=COLORS['primary'],
                line=dict(color=COLORS['secondary'], width=1)
            ),
            text=f"{row['sequence_name']}<br>{row['predicted_duration']:.1f}s",
            textposition='inside',
            textfont=dict(color='white'),
            hovertemplate=(
                f"<b>{row['sequence_name']}</b><br>"
                f"Step: {row['step']}<br>"
                f"Duration: {row['predicted_duration']:.2f}s<br>"
                f"Start: {row['start']:.2f}s<br>"
                f"End: {row['end']:.2f}s<br>"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        title=f'Sequence Timeline - Sample {sample_id}',
        xaxis_title='Time (seconds)',
        yaxis_title='Sequence Step',
        barmode='stack',
        height=max(400, len(sample_df) * 40),
    )

    return apply_dark_theme(fig)


def create_duration_distribution(df: pd.DataFrame, sequence_name: Optional[str] = None) -> go.Figure:
    """
    Create histogram of duration distribution

    Args:
        df: DataFrame with sequence data
        sequence_name: Optional sequence name to filter

    Returns:
        Plotly figure
    """
    if sequence_name:
        plot_df = df[df['sequence_name'] == sequence_name]
        title = f'Duration Distribution - {sequence_name}'
    else:
        plot_df = df
        title = 'Overall Duration Distribution'

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=plot_df['predicted_duration'],
        nbinsx=50,
        marker=dict(
            color=COLORS['primary'],
            line=dict(color=COLORS['secondary'], width=1)
        ),
        opacity=0.8,
        hovertemplate='Duration: %{x:.2f}s<br>Count: %{y}<extra></extra>',
    ))

    # Add mean line
    mean_val = plot_df['predicted_duration'].mean()
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color=COLORS['warning'],
        annotation_text=f"Mean: {mean_val:.2f}s",
        annotation_position="top"
    )

    fig.update_layout(
        title=title,
        xaxis_title='Duration (seconds)',
        yaxis_title='Frequency',
        showlegend=False,
    )

    return apply_dark_theme(fig)


def create_sequence_type_bar_chart(stats_df: pd.DataFrame) -> go.Figure:
    """
    Create bar chart of sequence type frequencies

    Args:
        stats_df: DataFrame with sequence type statistics

    Returns:
        Plotly figure
    """
    # Sort by count
    plot_df = stats_df.sort_values('count', ascending=True).tail(20)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=plot_df['count'],
        y=plot_df['sequence_name'],
        orientation='h',
        marker=dict(
            color=plot_df['count'],
            colorscale=COLOR_SCALE,
            line=dict(color=COLORS['secondary'], width=1)
        ),
        text=plot_df['count'],
        textposition='auto',
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Count: %{x}<br>'
            '<extra></extra>'
        ),
    ))

    fig.update_layout(
        title='Top 20 Sequence Types by Frequency',
        xaxis_title='Count',
        yaxis_title='Sequence Type',
        showlegend=False,
        height=max(400, len(plot_df) * 25),
    )

    return apply_dark_theme(fig)


def create_duration_by_sequence_type(stats_df: pd.DataFrame) -> go.Figure:
    """
    Create box plot of durations by sequence type

    Args:
        stats_df: DataFrame with sequence type statistics

    Returns:
        Plotly figure
    """
    # Get top 15 by count
    top_sequences = stats_df.nlargest(15, 'count')['sequence_name'].tolist()

    fig = go.Figure()

    for i, seq_name in enumerate(top_sequences):
        fig.add_trace(go.Box(
            y=[top_sequences.index(seq_name)],
            x=[stats_df[stats_df['sequence_name'] == seq_name]['mean_duration'].values[0]],
            name=seq_name,
            marker=dict(color=COLORS['primary']),
            orientation='h',
            showlegend=False,
        ))

    fig.update_layout(
        title='Average Duration by Sequence Type (Top 15)',
        xaxis_title='Average Duration (seconds)',
        yaxis_title='Sequence Type',
        height=500,
    )

    return apply_dark_theme(fig)


def create_sequence_length_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Create bar chart of sequence length distribution

    Args:
        df: DataFrame with sequence length distribution

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['sequence_length'],
        y=df['count'],
        marker=dict(
            color=COLORS['secondary'],
            line=dict(color=COLORS['primary'], width=1)
        ),
        hovertemplate=(
            'Sequence Length: %{x}<br>'
            'Count: %{y}<br>'
            '<extra></extra>'
        ),
    ))

    fig.update_layout(
        title='Distribution of Sequence Lengths',
        xaxis_title='Sequence Length (number of steps)',
        yaxis_title='Count',
        showlegend=False,
    )

    return apply_dark_theme(fig)


def create_sankey_diagram(transition_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Create Sankey diagram of sequence transitions

    Args:
        transition_df: DataFrame with sequence transitions
        top_n: Number of top transitions to show

    Returns:
        Plotly figure
    """
    # Get top transitions
    top_transitions = transition_df.nlargest(top_n, 'count')

    # Get unique sequences
    all_sequences = list(set(
        top_transitions['from_sequence'].tolist() +
        top_transitions['to_sequence'].tolist()
    ))

    # Create mapping
    sequence_to_idx = {seq: idx for idx, seq in enumerate(all_sequences)}

    # Create Sankey data
    source = [sequence_to_idx[seq] for seq in top_transitions['from_sequence']]
    target = [sequence_to_idx[seq] for seq in top_transitions['to_sequence']]
    value = top_transitions['count'].tolist()

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='#334155', width=0.5),
            label=all_sequences,
            color=COLORS['primary'],
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color='rgba(37, 99, 235, 0.3)',
        )
    )])

    fig.update_layout(
        title=f'Top {top_n} Sequence Transitions',
        height=600,
    )

    return apply_dark_theme(fig)


def create_cumulative_duration_plot(df: pd.DataFrame, sample_ids: List[int]) -> go.Figure:
    """
    Create line plot of cumulative duration for multiple samples

    Args:
        df: DataFrame with sequence data
        sample_ids: List of sample IDs to plot

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    colors_list = [COLORS['primary'], COLORS['secondary'], COLORS['success'],
                   COLORS['warning'], COLORS['info'], COLORS['purple']]

    for idx, sample_id in enumerate(sample_ids[:6]):  # Limit to 6 samples
        sample_df = df[df['sample_id'] == sample_id].sort_values('step')
        sample_df['cumulative_duration'] = sample_df['predicted_duration'].cumsum()

        fig.add_trace(go.Scatter(
            x=sample_df['step'],
            y=sample_df['cumulative_duration'],
            mode='lines+markers',
            name=f'Sample {sample_id}',
            line=dict(color=colors_list[idx % len(colors_list)], width=2),
            marker=dict(size=6),
            hovertemplate=(
                f'<b>Sample {sample_id}</b><br>'
                'Step: %{x}<br>'
                'Cumulative Duration: %{y:.2f}s<br>'
                '<extra></extra>'
            ),
        ))

    fig.update_layout(
        title='Cumulative Duration Over Steps',
        xaxis_title='Step',
        yaxis_title='Cumulative Duration (seconds)',
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
    )

    return apply_dark_theme(fig)


def create_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create heatmap of sequence types by step position

    Args:
        df: DataFrame with sequence data

    Returns:
        Plotly figure
    """
    # Create pivot table
    pivot_df = df.groupby(['step', 'sequence_name']).size().unstack(fill_value=0)

    # Limit to top sequences and first 20 steps
    top_sequences = df['sequence_name'].value_counts().head(15).index
    pivot_df = pivot_df[top_sequences].head(20)

    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=[f'Step {i}' for i in pivot_df.index],
        colorscale=COLOR_SCALE,
        hovertemplate=(
            'Step: %{y}<br>'
            'Sequence: %{x}<br>'
            'Count: %{z}<br>'
            '<extra></extra>'
        ),
    ))

    fig.update_layout(
        title='Sequence Type Frequency by Step Position',
        xaxis_title='Sequence Type',
        yaxis_title='Step Position',
        height=600,
    )

    return apply_dark_theme(fig)
