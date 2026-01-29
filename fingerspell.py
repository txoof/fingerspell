#!/usr/bin/env python3
"""
Fingerspell - NGT Fingerspelling Recognition

Main entry point for the application.
"""

from pathlib import Path
from src.fingerspell.app import main


if __name__ == '__main__':
    # Get project root for resource paths
    project_root = Path(__file__).parent
    
    # Run the application
    main(project_root)