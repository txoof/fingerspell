"""
Resource path resolution for development and packaged environments.

Handles path resolution whether running from source or as a PyInstaller bundle.
"""
import sys
from pathlib import Path


def get_base_path():
    """
    Get the base path for the application.
    
    Returns the correct base directory whether running:
    - From source (development)
    - As PyInstaller one-folder bundle
    - As PyInstaller one-file bundle
    
    Returns:
        Path: Absolute path to the application base directory
    """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running as PyInstaller bundle
        # sys._MEIPASS is the temp folder where PyInstaller extracts files
        return Path(sys._MEIPASS)
    else:
        # Running from source
        # Navigate up from utils/ to project root
        return Path(__file__).parent.parent.parent.parent


def get_resource_path(relative_path):
    """
    Get absolute path to a resource file.
    
    Args:
        relative_path: Path relative to project root (e.g., 'models/static.pkl')
    
    Returns:
        str: Absolute path as string (for library compatibility)
    """
    base = get_base_path()
    resource_path = base / relative_path
    
    # Convert to string for compatibility with libraries that don't accept Path objects
    return str(resource_path)


def get_models_dir():
    """
    Get path to models directory.
    
    Returns:
        str: Absolute path to models directory
    """
    return get_resource_path('models')


def verify_resource_exists(relative_path):
    """
    Check if a resource file exists.
    
    Args:
        relative_path: Path relative to project root
    
    Returns:
        bool: True if resource exists, False otherwise
    """
    path = Path(get_resource_path(relative_path))
    return path.exists()