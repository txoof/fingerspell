#!/usr/bin/env python3
"""
Fingerspell - NGT Fingerspelling Recognition

Main entry point for the application.
"""

from pathlib import Path
from src.fingerspell.ui.window import run_app


if __name__ == '__main__':
    # Get absolute paths to models
    project_root = Path(__file__).parent
    static_model = project_root / 'models' / 'ngt_static_classifier.pkl'
    dynamic_model = project_root / 'models' / 'ngt_dynamic_classifier.pkl'
    
    # Run the application
    run_app(str(static_model), str(dynamic_model))