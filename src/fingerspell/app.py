"""
Main application logic for fingerspell.

Handles menu system and mode dispatching.
"""

from pathlib import Path
from src.fingerspell.ui.menu import Menu
from src.fingerspell.ui.window import run_app
from src.fingerspell.collection.collector import run_collection



def run_collection_mode(project_root):
    """Run data collection workflow."""
    run_collection(project_root)


def run_training_mode(project_root):
    """Run model training workflow."""
    # print("\nStarting training mode...")
    # # TODO: Implement training workflow
    # print("Training mode - not yet implemented")
    # input("Press ENTER to return to menu...")
    pass

def run_recognition_mode(project_root):
    """Run real-time recognition."""
    # print("\nStarting recognition mode...")
    
    # Get model paths
    static_model = project_root / 'models' / 'ngt_static_classifier.pkl'
    dynamic_model = project_root / 'models' / 'ngt_dynamic_classifier.pkl'
    
    run_app(str(static_model), str(dynamic_model))


def main(project_root):
    """Main application entry point."""
    # print("NGT Fingerspelling System")
    
    while True:
        # Show main menu
        menu = Menu(
            title="Fingerspelling System",
            options=[
                ('1', 'Collect Training Data'),
                ('2', 'Train Models'),
                ('3', 'Run Recognition'),
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
            run_recognition_mode(project_root)