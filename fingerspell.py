#!/usr/bin/env python3
"""
Fingerspell - NGT Fingerspelling Recognition

Main entry point for the application.
"""

import multiprocessing
from src.fingerspell.ui.window import run_app
from src.fingerspell.utils import get_resource_path, verify_resource_exists


if __name__ == '__main__':
    # Required for PyInstaller to prevent infinite process spawning
    multiprocessing.freeze_support()
    
    # Get paths to models using resource utility
    static_model = get_resource_path('models/ngt_static_classifier.pkl')
    dynamic_model = get_resource_path('models/ngt_dynamic_classifier.pkl')
    
    # Verify models exist before launching
    if not verify_resource_exists('models/ngt_static_classifier.pkl'):
        print(f"ERROR: Static model not found at {static_model}")
        exit(1)
    
    if not verify_resource_exists('models/ngt_dynamic_classifier.pkl'):
        print(f"ERROR: Dynamic model not found at {dynamic_model}")
        exit(1)
    
    # Run the application
    run_app(static_model, dynamic_model)