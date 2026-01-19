#!/usr/bin/env python
"""
Local development runner for SeqofSeq Dashboard
"""

import os
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set development environment variables
os.environ.setdefault('DEBUG', 'true')
os.environ.setdefault('DATA_PATH', str(Path(__file__).parent.parent / 'SeqofSeq_Pipeline' / 'outputs' / 'generated_sequences.csv'))
os.environ.setdefault('PORT', '8050')

# Import and run app
from app.app import app

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║         🧬 SeqofSeq Dashboard - Development Mode            ║
    ║                                                              ║
    ║         Dashboard URL: http://localhost:8050                ║
    ║         Debug Mode: ENABLED                                 ║
    ║         Auto-reload: ENABLED                                ║
    ║                                                              ║
    ║         Press Ctrl+C to stop                                ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    app.run(
        host='0.0.0.0',
        port=8050,
        debug=True
    )
