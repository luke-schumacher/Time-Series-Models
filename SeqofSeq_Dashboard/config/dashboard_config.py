"""
Dashboard Configuration
Centralized configuration for the SeqofSeq Dashboard
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "app" / "assets"

# Data configuration
DATA_PATH = os.getenv(
    'DATA_PATH',
    str(DATA_DIR / 'generated_sequences.csv')
)

# Server configuration
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 8050))
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# Dashboard configuration
DASHBOARD_TITLE = "SeqofSeq Dashboard"
DASHBOARD_SUBTITLE = "MRI Sequence & Duration Prediction Visualization"

# Visualization defaults
DEFAULT_CHART_HEIGHT = 400
DEFAULT_BINS = 50
MAX_SAMPLES_TO_COMPARE = 6
TOP_N_SEQUENCES = 20

# Chart colors (matching CSS design system)
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

# Theme colors
THEME = {
    'dark_bg': '#0f172a',
    'card_bg': '#1e293b',
    'text_primary': '#f1f5f9',
    'text_secondary': '#94a3b8',
    'border_color': '#334155',
}

# Data refresh configuration
AUTO_REFRESH_INTERVAL = 60000  # milliseconds (60 seconds)
ENABLE_AUTO_REFRESH = False

# Cache configuration
ENABLE_CACHING = True
CACHE_TIMEOUT = 300  # seconds (5 minutes)

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Performance configuration
MAX_ROWS_TO_DISPLAY = 1000
PAGINATION_PAGE_SIZE = 50

# Feature flags
FEATURES = {
    'show_debug_info': DEBUG,
    'enable_export': True,
    'enable_filtering': True,
    'show_advanced_stats': True,
}
