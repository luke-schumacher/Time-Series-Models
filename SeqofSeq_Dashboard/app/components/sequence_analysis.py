"""
Sequence analysis component for detailed sequence exploration
"""

from dash import html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import pandas as pd
from app.utils.visualizations import (
    create_sequence_timeline,
    create_cumulative_duration_plot,
    create_sankey_diagram,
    create_heatmap,
)


def create_sequence_analysis_layout(df: pd.DataFrame, transition_df: pd.DataFrame):
    """
    Create the sequence analysis layout

    Args:
        df: Full DataFrame
        transition_df: Transition DataFrame

    Returns:
        Dash layout component
    """
    sample_ids = sorted(df['sample_id'].unique())

    return html.Div([
        # Filters
        html.Div([
            html.H3("🔍 Sequence Explorer", className="filter-title"),
            dbc.Row([
                dbc.Col([
                    html.Label("Select Sample ID:", className="text-muted mb-2"),
                    dcc.Dropdown(
                        id='sample-selector',
                        options=[{'label': f'Sample {sid}', 'value': sid} for sid in sample_ids[:100]],
                        value=sample_ids[0] if sample_ids else None,
                        clearable=False,
                        className="mb-3"
                    ),
                ], xs=12, md=6),

                dbc.Col([
                    html.Label("Compare Samples (Multi-select):", className="text-muted mb-2"),
                    dcc.Dropdown(
                        id='multi-sample-selector',
                        options=[{'label': f'Sample {sid}', 'value': sid} for sid in sample_ids[:50]],
                        value=sample_ids[:3] if len(sample_ids) >= 3 else sample_ids,
                        multi=True,
                        clearable=False,
                        className="mb-3"
                    ),
                ], xs=12, md=6),
            ]),
        ], className="filter-container"),

        # Timeline Visualization
        html.Div([
            html.H2("📅 Sequence Timeline", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H3("Gantt Chart - Individual Sample", className="chart-title"),
                        dcc.Graph(
                            id='sequence-timeline',
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="chart-card")
                ], xs=12, className="mb-4"),
            ]),
        ]),

        # Comparative Analysis
        html.Div([
            html.H2("📊 Comparative Analysis", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H3("Cumulative Duration Comparison", className="chart-title"),
                        dcc.Graph(
                            id='cumulative-duration-chart',
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="chart-card")
                ], xs=12, className="mb-4"),
            ]),
        ]),

        # Transition Analysis
        html.Div([
            html.H2("🔄 Sequence Transitions", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H3("Sequence Flow Diagram", className="chart-title"),
                        dcc.Graph(
                            id='sankey-diagram',
                            figure=create_sankey_diagram(transition_df),
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="chart-card")
                ], xs=12, lg=6, className="mb-4"),

                dbc.Col([
                    html.Div([
                        html.H3("Sequence Position Heatmap", className="chart-title"),
                        dcc.Graph(
                            id='position-heatmap',
                            figure=create_heatmap(df),
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="chart-card")
                ], xs=12, lg=6, className="mb-4"),
            ]),
        ]),

        # Sample Details Table
        html.Div([
            html.H2("📋 Sample Details", className="mb-3"),
            html.Div([
                html.H3("Sequence Steps", className="chart-title"),
                html.Div(id='sample-details-table')
            ], className="chart-card"),
        ]),
    ])
