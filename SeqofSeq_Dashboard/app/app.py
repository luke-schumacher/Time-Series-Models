"""
SeqofSeq Dashboard - Main Application
A sophisticated dashboard for visualizing MRI sequence predictions
"""

import os
from pathlib import Path
from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

from app.utils.data_loader import DataLoader
from app.utils.visualizations import (
    create_sequence_timeline,
    create_cumulative_duration_plot,
    create_duration_distribution,
    apply_dark_theme,
    COLORS,
)
from app.components.overview import create_overview_layout
from app.components.sequence_analysis import create_sequence_analysis_layout
from app.components.duration_analysis import (
    create_duration_analysis_layout,
    create_duration_stats_cards,
)

# Initialize Dash app
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="SeqofSeq Dashboard",
    update_title="Loading..."
)

# Data path configuration
DATA_PATH = os.getenv(
    'DATA_PATH',
    '/app/data/generated_sequences.csv'
)

# Initialize data loader
data_loader = DataLoader(DATA_PATH)

# Load data
try:
    df = data_loader.load_data()
    stats = data_loader.get_summary_stats(df)
    seq_stats = data_loader.get_sequence_type_stats(df)
    length_dist = data_loader.get_sequence_length_distribution(df)
    transition_df = data_loader.get_sequence_transitions(df)
    DATA_LOADED = True
except Exception as e:
    print(f"Error loading data: {e}")
    DATA_LOADED = False
    df = pd.DataFrame()
    stats = {}
    seq_stats = pd.DataFrame()
    length_dist = pd.DataFrame()
    transition_df = pd.DataFrame()


def create_header():
    """Create the dashboard header"""
    return html.Div([
        html.Div([
            html.Div([
                html.H1("🧬 SeqofSeq Dashboard", className="header-title"),
                html.P(
                    "MRI Sequence & Duration Prediction Visualization",
                    className="header-subtitle"
                ),
            ]),
            html.Div([
                dbc.Button(
                    "🔄 Refresh Data",
                    id="refresh-button",
                    color="light",
                    outline=True,
                    size="sm",
                    className="me-2"
                ),
            ]),
        ], className="header-content"),
    ], className="dashboard-header")


def create_error_message():
    """Create error message for when data fails to load"""
    return html.Div([
        html.Div([
            html.Div("⚠️", className="empty-state-icon"),
            html.H2("Data Not Available", className="text-warning"),
            html.P(
                f"Unable to load data from: {DATA_PATH}",
                className="text-muted"
            ),
            html.P(
                "Please ensure the data file exists and the path is correct.",
                className="text-muted"
            ),
            html.Hr(),
            html.P(
                "To generate data, run the SeqofSeq pipeline first:",
                className="text-muted"
            ),
            html.Pre(
                "docker exec -it seqofseq_pipeline bash\n"
                "python main_pipeline.py generate",
                style={
                    'backgroundColor': '#0f172a',
                    'padding': '1rem',
                    'borderRadius': '8px',
                    'color': '#f1f5f9'
                }
            ),
        ], className="empty-state"),
    ], className="main-container")


# Create app layout
if DATA_LOADED:
    app.layout = html.Div([
        create_header(),

        html.Div([
            # Tabs for different sections
            dcc.Tabs(
                id='main-tabs',
                value='overview',
                className='custom-tabs',
                children=[
                    dcc.Tab(
                        label='📊 Overview',
                        value='overview',
                        className='custom-tab',
                        selected_className='custom-tab--selected'
                    ),
                    dcc.Tab(
                        label='🔍 Sequence Analysis',
                        value='sequence',
                        className='custom-tab',
                        selected_className='custom-tab--selected'
                    ),
                    dcc.Tab(
                        label='⏱️ Duration Analysis',
                        value='duration',
                        className='custom-tab',
                        selected_className='custom-tab--selected'
                    ),
                ]
            ),

            # Tab content
            html.Div(id='tab-content', className='mt-4'),

        ], className='main-container'),

        # Store for data refresh
        dcc.Store(id='data-store', data={'refresh': 0}),
    ])
else:
    app.layout = html.Div([
        create_header(),
        create_error_message(),
    ])


# Callbacks
@callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value'),
    Input('data-store', 'data')
)
def render_tab_content(active_tab, data_store):
    """Render content based on selected tab"""
    if not DATA_LOADED:
        return create_error_message()

    if active_tab == 'overview':
        return create_overview_layout(stats, df, seq_stats, length_dist)
    elif active_tab == 'sequence':
        return create_sequence_analysis_layout(df, transition_df)
    elif active_tab == 'duration':
        return create_duration_analysis_layout(df, seq_stats)


@callback(
    Output('sequence-timeline', 'figure'),
    Input('sample-selector', 'value')
)
def update_timeline(sample_id):
    """Update sequence timeline based on selected sample"""
    if sample_id is None:
        return go.Figure()
    return create_sequence_timeline(df, sample_id)


@callback(
    Output('cumulative-duration-chart', 'figure'),
    Input('multi-sample-selector', 'value')
)
def update_cumulative_chart(sample_ids):
    """Update cumulative duration chart"""
    if not sample_ids:
        return go.Figure()
    return create_cumulative_duration_plot(df, sample_ids)


@callback(
    Output('sample-details-table', 'children'),
    Input('sample-selector', 'value')
)
def update_sample_table(sample_id):
    """Update sample details table"""
    if sample_id is None:
        return html.P("No sample selected", className="text-muted")

    sample_df = df[df['sample_id'] == sample_id].sort_values('step')
    sample_df = sample_df[['step', 'sequence_name', 'predicted_duration', 'sequence_id']]
    sample_df = sample_df.round(2)

    return dbc.Table.from_dataframe(
        sample_df,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        className="table-dark"
    )


@callback(
    Output('duration-stats-cards', 'children'),
    Output('duration-histogram', 'figure'),
    Output('duration-boxplot', 'figure'),
    Output('duration-over-steps', 'figure'),
    Output('duration-violin', 'figure'),
    Input('sequence-type-selector', 'value')
)
def update_duration_analysis(sequence_type):
    """Update duration analysis visualizations"""
    # Filter data
    if sequence_type == 'all':
        filtered_df = df
    else:
        filtered_df = df[df['sequence_name'] == sequence_type]

    # Stats cards
    stats_cards = create_duration_stats_cards(filtered_df)

    # Histogram
    histogram = create_duration_distribution(
        df,
        sequence_name=None if sequence_type == 'all' else sequence_type
    )

    # Box plot by sequence type
    top_sequences = seq_stats.nlargest(15, 'count')['sequence_name'].tolist()
    box_df = df[df['sequence_name'].isin(top_sequences)]

    boxplot = go.Figure()
    for seq_name in top_sequences:
        seq_data = box_df[box_df['sequence_name'] == seq_name]
        boxplot.add_trace(go.Box(
            y=seq_data['predicted_duration'],
            name=seq_name,
            marker=dict(color=COLORS['primary']),
        ))

    boxplot.update_layout(
        title='Duration Distribution by Sequence Type',
        yaxis_title='Duration (seconds)',
        showlegend=False,
        height=500,
    )
    boxplot = apply_dark_theme(boxplot)

    # Duration over steps
    step_stats = filtered_df.groupby('step')['predicted_duration'].agg(['mean', 'std']).reset_index()

    duration_steps = go.Figure()
    duration_steps.add_trace(go.Scatter(
        x=step_stats['step'],
        y=step_stats['mean'],
        mode='lines+markers',
        name='Mean',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=6),
    ))

    duration_steps.add_trace(go.Scatter(
        x=step_stats['step'],
        y=step_stats['mean'] + step_stats['std'],
        mode='lines',
        name='Mean + Std',
        line=dict(color=COLORS['secondary'], width=1, dash='dash'),
        showlegend=True,
    ))

    duration_steps.add_trace(go.Scatter(
        x=step_stats['step'],
        y=step_stats['mean'] - step_stats['std'],
        mode='lines',
        name='Mean - Std',
        line=dict(color=COLORS['secondary'], width=1, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(124, 58, 237, 0.2)',
        showlegend=True,
    ))

    duration_steps.update_layout(
        title='Average Duration by Step Position',
        xaxis_title='Step',
        yaxis_title='Duration (seconds)',
        height=400,
    )
    duration_steps = apply_dark_theme(duration_steps)

    # Violin plot
    violin = go.Figure()
    violin.add_trace(go.Violin(
        y=filtered_df['predicted_duration'],
        box_visible=True,
        meanline_visible=True,
        fillcolor=COLORS['secondary'],
        line_color=COLORS['primary'],
        opacity=0.8,
    ))

    violin.update_layout(
        title='Duration Distribution (Violin Plot)',
        yaxis_title='Duration (seconds)',
        showlegend=False,
        height=400,
    )
    violin = apply_dark_theme(violin)

    return stats_cards, histogram, boxplot, duration_steps, violin


@callback(
    Output('data-store', 'data'),
    Input('refresh-button', 'n_clicks'),
    State('data-store', 'data')
)
def refresh_data(n_clicks, current_data):
    """Refresh data when button is clicked"""
    if n_clicks is None:
        return current_data

    # In a production app, you would reload the data here
    # For now, just increment the counter to trigger a re-render
    return {'refresh': current_data.get('refresh', 0) + 1}


# Server for deployment
server = app.server

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8050))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'

    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║         🧬 SeqofSeq Dashboard Starting...                   ║
    ║                                                              ║
    ║         Dashboard URL: http://localhost:{port}              ║
    ║         Data Path: {DATA_PATH:<40} ║
    ║         Data Loaded: {str(DATA_LOADED):<39} ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    app.run(host='0.0.0.0', port=port, debug=debug)
