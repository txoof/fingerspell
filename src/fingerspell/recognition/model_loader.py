"""
Model discovery and loading for recognition.

Handles scanning for models, formatting for display, and loading workflow.
"""

from pathlib import Path
from datetime import datetime
import cv2
import numpy as np


def scan_path_for_models(working_path='~/Desktop', model_files=None):
    """
    Scan path for directories containing trained models.
    
    A valid model directory must contain at minimum:
    - For static: static_model.pkl + keypoint_classifier_label_static.csv
    - For dynamic: dynamic_model.pkl + keypoint_classifier_label_dynamic.csv
    
    Args:
        working_path: Path to scan for model directories
        model_files: Dict with keys 'static_model', 'static_labels', 'dynamic_model', 'dynamic_labels'
                    Default values provided if None
    
    Returns:
        list: List of dicts with directory info, sorted by modification time (newest first)
              Each dict contains:
              - 'path': Path object to directory
              - 'name': Directory name
              - 'has_static': bool
              - 'has_dynamic': bool
              - 'modified': datetime of last modification
              - 'static_model_path': Path to static model or None
              - 'static_labels_path': Path to static labels or None
              - 'dynamic_model_path': Path to dynamic model or None
              - 'dynamic_labels_path': Path to dynamic labels or None
    """
    # Set defaults if not provided
    if model_files is None:
        model_files = {
            'static_model': 'static_model.pkl',
            'static_labels': 'keypoint_classifier_label_static.csv',
            'dynamic_model': 'dynamic_model.pkl',
            'dynamic_labels': 'keypoint_classifier_label_dynamic.csv'
        }
    
    working_path = Path(working_path).expanduser().absolute()
    
    if not working_path.exists():
        return []
    
    valid_dirs = []
    
    # Scan all directories in working path
    for item in working_path.iterdir():
        if not item.is_dir():
            continue
        
        # Check for static files
        static_model = item / model_files['static_model']
        static_labels = item / model_files['static_labels']
        has_static = static_model.exists() and static_labels.exists()
        
        # Check for dynamic files
        dynamic_model = item / model_files['dynamic_model']
        dynamic_labels = item / model_files['dynamic_labels']
        has_dynamic = dynamic_model.exists() and dynamic_labels.exists()
        
        # Skip if no valid models
        if not has_static and not has_dynamic:
            continue
        
        # Get modification time
        modified = datetime.fromtimestamp(item.stat().st_mtime)
        
        valid_dirs.append({
            'path': item,
            'name': item.name,
            'has_static': has_static,
            'has_dynamic': has_dynamic,
            'modified': modified,
            'static_model_path': static_model if has_static else None,
            'static_labels_path': static_labels if has_static else None,
            'dynamic_model_path': dynamic_model if has_dynamic else None,
            'dynamic_labels_path': dynamic_labels if has_dynamic else None
        })
    
    # Sort by modification time, newest first
    valid_dirs.sort(key=lambda x: x['modified'], reverse=True)
    
    return valid_dirs


def format_model_dir(item):
    """
    Format model directory item for PaginatedMenu display.
    
    Args:
        item: Dict with model directory info
        
    Returns:
        tuple: (main_text, detail_text)
    """
    main_text = item['name']
    
    # Build detail text
    types = []
    if item['has_static']:
        types.append("Static")
    if item['has_dynamic']:
        types.append("Dynamic")
    detail_text = " + ".join(types) + " models"
    
    return (main_text, detail_text)


def get_default_models(project_root):
    """
    Get default model paths from project.
    
    Args:
        project_root: Path to project root
        
    Returns:
        dict: {
            'static_model_path': Path or None,
            'static_labels_path': Path or None,
            'dynamic_model_path': Path or None,
            'dynamic_labels_path': Path or None
        }
    """
    models_dir = Path(project_root) / 'models'
    
    # Check for static
    static_model = models_dir / 'static_model.pkl'
    static_labels = models_dir / 'keypoint_classifier_label_static.csv'
    
    # Check for dynamic
    dynamic_model = models_dir / 'dynamic_model.pkl'
    dynamic_labels = models_dir / 'keypoint_classifier_label_dynamic.csv'
    
    return {
        'static_model_path': static_model if static_model.exists() else None,
        'static_labels_path': static_labels if static_labels.exists() else None,
        'dynamic_model_path': dynamic_model if dynamic_model.exists() else None,
        'dynamic_labels_path': dynamic_labels if dynamic_labels.exists() else None
    }


def load_custom_models():
    """
    Show model selection menu and return selected model paths.
    
    Scans Desktop for model directories and lets user select.
    
    Returns:
        dict: Selected model paths (same format as get_default_models()) or None if cancelled
    """
    from src.fingerspell.ui.menu import PaginatedMenu
    from src.fingerspell.ui.common import draw_modal_overlay
    
    # Scan Desktop for models
    model_dirs = scan_path_for_models('~/Desktop')
    
    if not model_dirs:
        # Show "no models found" message
        screen = np.zeros((720, 1280, 3), dtype=np.uint8)
        screen[:] = (40, 40, 40)
        
        message = "No custom models found on Desktop.\n\nTrain models first.\n\nPress any key to continue"
        screen = draw_modal_overlay(screen, message, position='center')
        
        cv2.imshow("Load Models", screen)
        while cv2.waitKey(1) == -1:
            pass
        cv2.destroyAllWindows()
        return None
    
    # Show selection menu
    menu = PaginatedMenu(
        title="Select Custom Models",
        items=model_dirs,
        format_fn=format_model_dir,
        items_per_page=10
    )
    
    selected = menu.run()
    
    if not selected:
        return None  # User cancelled
    
    # Return model paths in standard format
    return {
        'static_model_path': selected['static_model_path'],
        'static_labels_path': selected['static_labels_path'],
        'dynamic_model_path': selected['dynamic_model_path'],
        'dynamic_labels_path': selected['dynamic_labels_path']
    }
