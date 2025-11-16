"""
Duration analysis component for analyzing predicted durations
"""

from dash import html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from app.utils.visualizations import (
    create_duration_distribution,
    apply_dark_theme,
    COLORS,
)


def create_duration_analysis_layout(df: pd.DataFrame, seq_stats: pd.DataFrame):
    """
    Create the duration analysis layout

    Args:
        df: Full DataFrame
        seq_stats: Sequence type statistics

    Returns:
        Dash layout component
    """
    sequence_names = sorted(df['sequence_name'].unique())

    return html.Div([
        # Filters
        html.Div([
            html.H3("⏱️ Duration Analysis Filters", className="filter-title"),
            dbc.Row([
                dbc.Col([
                    html.Label("Select Sequence Type:", className="text-muted mb-2"),
                    dcc.Dropdown(
                        id='sequence-type-selector',
                        options=[{'label': 'All Sequences', 'value': 'all'}] +
                                [{'label': seq, 'value': seq} for seq in sequence_names],
                        value='all',
                        clearable=False,
                        className="mb-3"
                    ),
                ], xs=12, md=6),
            ]),
        ], className="filter-container"),

        # Duration Statistics Cards
        html.Div([
            html.H2("📊 Duration Statistics", className="mb-3"),
            html.Div(id='duration-stats-cards', className="mb-4"),
        ]),

        # Distribution Visualizations
        html.Div([
            html.H2("📈 Duration Distributions", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H3("Histogram", className="chart-title"),
                        dcc.Graph(
                            id='duration-histogram',
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="chart-card")
                ], xs=12, lg=6, className="mb-4"),

                dbc.Col([
                    html.Div([
                        html.H3("Box Plot by Sequence Type", className="chart-title"),
                        dcc.Graph(
                            id='duration-boxplot',
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="chart-card")
                ], xs=12, lg=6, className="mb-4"),
            ]),

            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H3("Duration Over Steps", className="chart-title"),
                        dcc.Graph(
                            id='duration-over-steps',
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="chart-card")
                ], xs=12, lg=6, className="mb-4"),

                dbc.Col([
                    html.Div([
                        html.H3("Violin Plot", className="chart-title"),
                        dcc.Graph(
                            id='duration-violin',
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="chart-card")
                ], xs=12, lg=6, className="mb-4"),
            ]),
        ]),

        # Top Sequences by Duration
        html.Div([
            html.H2("🏆 Top Sequences", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H3("Longest Average Durations", className="chart-title"),
                        html.Div(
                            dbc.Table.from_dataframe(
                                seq_stats.nlargest(10, 'mean_duration')[
                                    ['sequence_name', 'mean_duration', 'count']
                                ].round(2),
                                striped=True,
                                bordered=True,
                                hover=True,
                                responsive=True,
                                className="table-dark"
                            ),
                        )
                    ], className="chart-card")
                ], xs=12, lg=6, className="mb-4"),

                dbc.Col([
                    html.Div([
                        html.H3("Shortest Average Durations", className="chart-title"),
                        html.Div(
                            dbc.Table.from_dataframe(
                                seq_stats.nsmallest(10, 'mean_duration')[
                                    ['sequence_name', 'mean_duration', 'count']
                                ].round(2),
                                striped=True,
                                bordered=True,
                                hover=True,
                                responsive=True,
                                className="table-dark"
                            ),
                        )
                    ], className="chart-card")
                ], xs=12, lg=6, className="mb-4"),
            ]),
        ]),
    ])


def create_duration_stats_cards(df: pd.DataFrame) -> html.Div:
    """Create duration statistics cards"""
    stats = {
        'mean': df['predicted_duration'].mean(),
        'median': df['predicted_duration'].median(),
        'std': df['predicted_duration'].std(),
        'min': df['predicted_duration'].min(),
        'max': df['predicted_duration'].max(),
        'total': df['predicted_duration'].sum(),
    }

    return dbc.Row([
        dbc.Col([
            html.Div([
                html.Div("📊", className="metric-icon"),
                html.Div(f"{stats['mean']:.2f}s", className="metric-value text-primary"),
                html.Div("Mean Duration", className="metric-label"),
            ], className="metric-card")
        ], xs=6, md=4, lg=2),

        dbc.Col([
            html.Div([
                html.Div("📊", className="metric-icon"),
                html.Div(f"{stats['median']:.2f}s", className="metric-value text-success"),
                html.Div("Median Duration", className="metric-label"),
            ], className="metric-card")
        ], xs=6, md=4, lg=2),

        dbc.Col([
            html.Div([
                html.Div("📊", className="metric-icon"),
                html.Div(f"{stats['std']:.2f}s", className="metric-value text-warning"),
                html.Div("Std Deviation", className="metric-label"),
            ], className="metric-card")
        ], xs=6, md=4, lg=2),

        dbc.Col([
            html.Div([
                html.Div("⬇️", className="metric-icon"),
                html.Div(f"{stats['min']:.2f}s", className="metric-value text-info"),
                html.Div("Min Duration", className="metric-label"),
            ], className="metric-card")
        ], xs=6, md=4, lg=2),

        dbc.Col([
            html.Div([
                html.Div("⬆️", className="metric-icon"),
                html.Div(f"{stats['max']:.2f}s", className="metric-value text-danger"),
                html.Div("Max Duration", className="metric-label"),
            ], className="metric-card")
        ], xs=6, md=4, lg=2),

        dbc.Col([
            html.Div([
                html.Div("⏱️", className="metric-icon"),
                html.Div(f"{stats['total']:.0f}s", className="metric-value"),
                html.Div("Total Duration", className="metric-label"),
            ], className="metric-card")
        ], xs=6, md=4, lg=2),
    ], className="metrics-grid")
