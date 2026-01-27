#!/usr/bin/env python3
"""
Fingerspell - NGT Fingerspelling Recognition

Main entry point for the application.
"""

import multiprocessing
from pathlib import Path
from src.fingerspell.utils import get_resource_path
from src.fingerspell.app import main


if __name__ == '__main__':
    # Required for PyInstaller to prevent infinite process spawning
    multiprocessing.freeze_support()
    
    # Use resource utility to get correct paths in both dev and frozen environments
    project_root_str = get_resource_path('')
    project_root = Path(project_root_str)
    
    # Run the application
    main(project_root)
