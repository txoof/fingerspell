"""
Main application logic for fingerspell.

Handles menu system and mode dispatching.
"""

from pathlib import Path
from src.fingerspell.ui.menu import Menu
from src.fingerspell.ui.window import run_app
from src.fingerspell.collection.collector import run_collection
from src.fingerspell.training.trainer import run_training_workflow
from src.fingerspell.recognition.model_loader import load_custom_models, get_default_models



def run_collection_mode(project_root):
    """Run data collection workflow."""
    run_collection(project_root)


def run_training_mode(project_root):
    """Run model training workflow."""
    run_training_workflow(project_root)

def run_recognition_mode(project_root, custom_models=None):
    """
    Run real-time recognition.
    
    Args:
        project_root: Path to project root
        custom_models: Optional dict with custom model paths (from load_custom_models())
    """
    # Use custom models if provided, otherwise use defaults
    if custom_models:
        static_model = custom_models.get('static_model_path')
        dynamic_model = custom_models.get('dynamic_model_path')
    else:
        # Get default models
        defaults = get_default_models(project_root)
        static_model = defaults.get('static_model_path')
        dynamic_model = defaults.get('dynamic_model_path')
    
    # Convert to strings for run_app
    static_model_str = str(static_model) if static_model else None
    dynamic_model_str = str(dynamic_model) if dynamic_model else None
    
    run_app(static_model_str, dynamic_model_str)


def main(project_root):
    """Main application entry point."""
    # Track custom models across menu loops
    custom_models = None
    
    while True:
        # Show main menu
        menu = Menu(
            title="Fingerspelling System",
            options=[
                ('1', 'Collect Training Data'),
                ('2', 'Train Models'),
                ('3', 'Run Recognition'),
                ('4', 'Load Custom Models'),
                ('', ''),
                ('', 'ESC - Quit')
            ],
            window_name="Fingerspelling"
        )
        
        choice = menu.run()
        
        if choice is None:
            print("\nGoodbye!")
            break
        
        elif choice == '1':
            run_collection_mode(project_root)
        
        elif choice == '2':
            run_training_mode(project_root)
        
        elif choice == '3':
            run_recognition_mode(project_root, custom_models)
        
        elif choice == '4':
            # Load custom models
            selected = load_custom_models()
            if selected:
                custom_models = selected
                print(f"\nCustom models loaded successfully")
            else:
                print(f"\nNo models selected, using defaults")