"""
Overview dashboard component showing key metrics and summary visualizations
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from typing import Dict
import pandas as pd
from app.utils.visualizations import (
    create_sequence_type_bar_chart,
    create_duration_distribution,
    create_sequence_length_distribution,
)


def create_metric_card(title: str, value: str, icon: str = "📊") -> dbc.Col:
    """Create a metric card component"""
    return dbc.Col(
        html.Div(
            [
                html.Div(icon, className="metric-icon"),
                html.Div(value, className="metric-value"),
                html.Div(title, className="metric-label"),
            ],
            className="metric-card"
        ),
        xs=12, sm=6, md=4, lg=3, xl=2,
    )


def create_overview_layout(stats: Dict, df: pd.DataFrame, seq_stats: pd.DataFrame, length_dist: pd.DataFrame):
    """
    Create the overview dashboard layout

    Args:
        stats: Dictionary of summary statistics
        df: Full DataFrame
        seq_stats: Sequence type statistics
        length_dist: Sequence length distribution

    Returns:
        Dash layout component
    """
    return html.Div([
        # Key Metrics Row
        html.Div([
            html.H2("📈 Key Metrics", className="mb-3"),
            dbc.Row([
                create_metric_card(
                    "Total Samples",
                    f"{stats['total_samples']:,}",
                    "🔢"
                ),
                create_metric_card(
                    "Total Sequences",
                    f"{stats['total_sequences']:,}",
                    "📊"
                ),
                create_metric_card(
                    "Unique Patients",
                    f"{stats['unique_patients']:,}",
                    "👥"
                ),
                create_metric_card(
                    "Sequence Types",
                    f"{stats['unique_sequence_types']:,}",
                    "🏷️"
                ),
                create_metric_card(
                    "Avg Sequence Length",
                    f"{stats['avg_sequence_length']:.1f}",
                    "📏"
                ),
                create_metric_card(
                    "Avg Duration",
                    f"{stats['avg_duration']:.1f}s",
                    "⏱️"
                ),
            ], className="metrics-grid mb-4"),
        ], className="mb-4"),

        # Charts Row
        html.Div([
            html.H2("📊 Overview Visualizations", className="mb-3"),
            dbc.Row([
                # Sequence Type Distribution
                dbc.Col([
                    html.Div([
                        html.H3("Sequence Type Distribution", className="chart-title"),
                        dcc.Graph(
                            id='sequence-type-chart',
                            figure=create_sequence_type_bar_chart(seq_stats),
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="chart-card")
                ], xs=12, lg=6, className="mb-4"),

                # Duration Distribution
                dbc.Col([
                    html.Div([
                        html.H3("Duration Distribution", className="chart-title"),
                        dcc.Graph(
                            id='duration-dist-chart',
                            figure=create_duration_distribution(df),
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="chart-card")
                ], xs=12, lg=6, className="mb-4"),
            ]),

            dbc.Row([
                # Sequence Length Distribution
                dbc.Col([
                    html.Div([
                        html.H3("Sequence Length Distribution", className="chart-title"),
                        dcc.Graph(
                            id='length-dist-chart',
                            figure=create_sequence_length_distribution(length_dist),
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="chart-card")
                ], xs=12, lg=6, className="mb-4"),

                # Sequence Type Statistics Table
                dbc.Col([
                    html.Div([
                        html.H3("Sequence Type Statistics", className="chart-title"),
                        html.Div(
                            dbc.Table.from_dataframe(
                                seq_stats.head(10).round(2),
                                striped=True,
                                bordered=True,
                                hover=True,
                                responsive=True,
                                className="table-dark"
                            ),
                            style={'maxHeight': '400px', 'overflowY': 'auto'}
                        )
                    ], className="chart-card")
                ], xs=12, lg=6, className="mb-4"),
            ]),
        ]),
    ])
