"""
Training workflow for fingerspelling models.

Handles scanning for training data, user selection, model training,
and saving trained models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import csv
import shutil
import cv2


def scan_path_for_training_data(working_path='~/Desktop',
                                data_files=None):
    """
    Scan path for directories containing valid training data.
    
    A valid training data directory must contain at minimum:
    - For static: keypoint_data_static.csv + keypoint_classifier_label_static.csv
    - For dynamic: keypoint_data_dynamic.csv + keypoint_classifier_label_dynamic.csv
    
    Args:
        working_path: Path to scan for training data directories
        data_files: Dict with keys 'static_data', 'static_labels', 'dynamic_data', 'dynamic_labels'
                   Default values provided if None
    
    Returns:
        list: List of dicts with directory info, sorted by modification time (newest first)
              Each dict contains:
              - 'path': Path object to directory
              - 'name': Directory name
              - 'has_static': bool
              - 'has_dynamic': bool
              - 'modified': datetime of last modification
              - 'static_samples': int (0 if no static data)
              - 'dynamic_samples': int (0 if no dynamic data)
    """
    # Set defaults if not provided
    if data_files is None:
        data_files = {
            'static_data': 'keypoint_data_static.csv',
            'static_labels': 'keypoint_classifier_label_static.csv',
            'dynamic_data': 'keypoint_data_dynamic.csv',
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
        static_data = item / data_files['static_data']
        static_labels = item / data_files['static_labels']
        has_static = static_data.exists() and static_labels.exists()
        
        # Check for dynamic files
        dynamic_data = item / data_files['dynamic_data']
        dynamic_labels = item / data_files['dynamic_labels']
        has_dynamic = dynamic_data.exists() and dynamic_labels.exists()
        
        # Skip if no valid data
        if not has_static and not has_dynamic:
            continue
        
        # Count samples
        static_samples = 0
        dynamic_samples = 0
        
        if has_static:
            try:
                with open(static_data) as f:
                    static_samples = sum(1 for line in f)
            except:
                static_samples = 0
        
        if has_dynamic:
            try:
                with open(dynamic_data) as f:
                    dynamic_samples = sum(1 for line in f)
            except:
                dynamic_samples = 0
        
        # Get modification time
        modified = datetime.fromtimestamp(item.stat().st_mtime)
        
        valid_dirs.append({
            'path': item,
            'name': item.name,
            'has_static': has_static,
            'has_dynamic': has_dynamic,
            'modified': modified,
            'static_samples': static_samples,
            'dynamic_samples': dynamic_samples
        })
    
    # Sort by modification time, newest first
    valid_dirs.sort(key=lambda x: x['modified'], reverse=True)
    
    return valid_dirs


def format_training_dir(item):
    """
    Format training directory item for PaginatedMenu display.
    
    Args:
        item: Dict with 'name', 'has_static', 'has_dynamic', 'static_samples', 'dynamic_samples'
        
    Returns:
        tuple: (main_text, detail_text)
    """
    main_text = item['name']
    
    # Build detail text
    types = []
    if item['has_static']:
        types.append(f"Static ({item['static_samples']} samples)")
    if item['has_dynamic']:
        types.append(f"Dynamic ({item['dynamic_samples']} samples)")
    detail_text = " + ".join(types)
    
    return (main_text, detail_text)


def train_static_model(data_path, labels_path):
    """
    Train static letter recognition model.
    
    Args:
        data_path: Path to keypoint_data_static.csv
        labels_path: Path to keypoint_classifier_label_static.csv
    
    Returns:
        dict: {
            'model': trained RandomForestClassifier,
            'accuracy': overall accuracy score,
            'per_class_accuracy': dict of {letter: accuracy},
            'report': classification report string,
            'low_accuracy_letters': list of (letter, accuracy) tuples with accuracy < 0.80
        }
    """
    # Load data
    df = pd.read_csv(data_path, header=None)
    
    # Separate features and labels
    y = df[0]  # Label indices
    X = df.iloc[:, 1:]  # 42 landmark features
    
    # Load letter names
    with open(labels_path, 'r', encoding='utf-8-sig') as f:
        label_names = [row[0] for row in csv.reader(f)]
    
    # Train/test split - stratified to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get detailed report
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=label_names)
    
    # Extract per-class accuracy
    per_class_accuracy = {}
    low_accuracy_letters = []
    
    for letter in label_names:
        if letter in report:
            recall = report[letter]['recall']
            per_class_accuracy[letter] = recall
            if recall < 0.80:
                low_accuracy_letters.append((letter, recall))
    
    return {
        'model': clf,
        'accuracy': accuracy,
        'per_class_accuracy': per_class_accuracy,
        'report': report_str,
        'low_accuracy_letters': low_accuracy_letters
    }


def train_dynamic_model(data_path, labels_path):
    """
    Train dynamic letter recognition model.
    
    Args:
        data_path: Path to keypoint_data_dynamic.csv
        labels_path: Path to keypoint_classifier_label_dynamic.csv
    
    Returns:
        dict: {
            'model': trained RandomForestClassifier,
            'accuracy': overall accuracy score,
            'per_class_accuracy': dict of {letter: accuracy},
            'report': classification report string,
            'low_accuracy_letters': list of (letter, accuracy) tuples with accuracy < 0.80
        }
    """
    # Load data
    df = pd.read_csv(data_path, header=None)
    
    # Separate features and labels
    y = df[0]  # Label indices
    X = df.iloc[:, 1:]  # 84 features: 42 current + 42 delta
    
    # Load letter names
    with open(labels_path, 'r', encoding='utf-8-sig') as f:
        label_names = [row[0] for row in csv.reader(f)]
    
    # Train/test split - stratified to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get detailed report
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=label_names)
    
    # Extract per-class accuracy
    per_class_accuracy = {}
    low_accuracy_letters = []
    
    for letter in label_names:
        if letter in report:
            recall = report[letter]['recall']
            per_class_accuracy[letter] = recall
            if recall < 0.80:
                low_accuracy_letters.append((letter, recall))
    
    return {
        'model': clf,
        'accuracy': accuracy,
        'per_class_accuracy': per_class_accuracy,
        'report': report_str,
        'low_accuracy_letters': low_accuracy_letters
    }


def save_models(static_result, dynamic_result, static_labels_path, dynamic_labels_path, output_dir):
    """
    Save trained models and label files to output directory.
    
    Args:
        static_result: Result dict from train_static_model() or None
        dynamic_result: Result dict from train_dynamic_model() or None
        static_labels_path: Path to static labels CSV or None
        dynamic_labels_path: Path to dynamic labels CSV or None
        output_dir: Path to output directory
    
    Returns:
        None
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save static model and labels
    if static_result:
        model_path = output_dir / 'static_model.pkl'
        joblib.dump(static_result['model'], model_path)
        
        if static_labels_path:
            shutil.copy(static_labels_path, output_dir / 'keypoint_classifier_label_static.csv')
    
    # Save dynamic model and labels
    if dynamic_result:
        model_path = output_dir / 'dynamic_model.pkl'
        joblib.dump(dynamic_result['model'], model_path)
        
        if dynamic_labels_path:
            shutil.copy(dynamic_labels_path, output_dir / 'keypoint_classifier_label_dynamic.csv')


def show_training_message(model_type):
    """
    Show training message.
    
    Returns window_name for later cleanup.
    
    Args:
        model_type: String like "Static" or "Dynamic"
        
    Returns:
        str: Window name
    """
    window_name = "Training"
    
    # Create blank screen
    screen = np.zeros((720, 1280, 3), dtype=np.uint8)
    screen[:] = (40, 40, 40)
    
    message = f"Training {model_type} Model..."
    
    # Center text
    from src.fingerspell.ui.common import draw_text
    screen = draw_text(screen, message, (640 - 200, 360),
                      font_size=32, color=(255, 255, 255))
    
    cv2.imshow(window_name, screen)
    cv2.waitKey(1)  # Force display update
    
    return window_name


def show_training_results(static_result, dynamic_result, output_dir):
    """
    Show training results with accuracy and warnings.
    
    Displays success message and warnings for low-accuracy letters.
    Waits for keypress to continue.
    
    Args:
        static_result: Result dict from train_static_model() or None
        dynamic_result: Result dict from train_dynamic_model() or None
        output_dir: Path where models were saved
    """
    window_name = "Training Complete"
    
    # Build message
    lines = ["Training Complete!"]
    lines.append("")
    
    if static_result:
        lines.append(f"Static Model: {static_result['accuracy']:.1%} accuracy")
        if static_result['low_accuracy_letters']:
            lines.append("  WARNING: Low accuracy letters (<80%):")
            for letter, acc in static_result['low_accuracy_letters']:
                lines.append(f"    {letter}: {acc:.1%}")
            lines.append("  Consider collecting more data for these letters")
        lines.append("")
    
    if dynamic_result:
        lines.append(f"Dynamic Model: {dynamic_result['accuracy']:.1%} accuracy")
        if dynamic_result['low_accuracy_letters']:
            lines.append("  WARNING: Low accuracy letters (<80%):")
            for letter, acc in dynamic_result['low_accuracy_letters']:
                lines.append(f"    {letter}: {acc:.1%}")
            lines.append("  Consider collecting more data for these letters")
        lines.append("")
    
    lines.append(f"Models saved to:")
    lines.append(f"{output_dir}")
    lines.append("")
    lines.append("Press any key to continue")
    
    message = "\n".join(lines)
    
    # Show message
    while True:
        screen = np.zeros((720, 1280, 3), dtype=np.uint8)
        screen[:] = (40, 40, 40)
        
        from src.fingerspell.ui.common import draw_modal_overlay
        screen = draw_modal_overlay(screen, message, position='center', width_percent=80)
        
        cv2.imshow(window_name, screen)
        
        key = cv2.waitKey(1)
        if key != -1:  # Any key pressed
            break
    
    cv2.destroyWindow(window_name)


def run_training_workflow(project_root=None):
    """
    Run complete training workflow.
    
    1. Scan Desktop for training data
    2. Show selection menu
    3. Train models with visual feedback
    4. Save models
    5. Show results
    
    Args:
        project_root: Project root path (unused, for consistency with other modes)
    """
    from src.fingerspell.ui.menu import PaginatedMenu
    
    # Step 1: Scan for training data
    training_dirs = scan_path_for_training_data('~/Desktop')
    
    if not training_dirs:
        # Show "no data found" message
        screen = np.zeros((720, 1280, 3), dtype=np.uint8)
        screen[:] = (40, 40, 40)
        
        from src.fingerspell.ui.common import draw_modal_overlay
        message = "No training data found on Desktop.\n\nCollect training data first.\n\nPress any key to continue"
        screen = draw_modal_overlay(screen, message, position='center')
        
        cv2.imshow("Training", screen)
        while cv2.waitKey(1) == -1:
            pass
        cv2.destroyAllWindows()
        return
    
    # Step 2: Show selection menu
    menu = PaginatedMenu(
        title="Select Training Data",
        items=training_dirs,
        format_fn=format_training_dir,
        items_per_page=10
    )
    
    selected = menu.run()
    
    if not selected:
        return  # User cancelled
    
    # Step 3: Train models
    static_result = None
    dynamic_result = None
    
    # Train static if available
    if selected['has_static']:
        static_data = selected['path'] / 'keypoint_data_static.csv'
        static_labels = selected['path'] / 'keypoint_classifier_label_static.csv'
        
        # Show training message
        window_name = show_training_message("Static")
        
        # Train model
        static_result = train_static_model(static_data, static_labels)
        
        # Close message
        cv2.destroyWindow(window_name)
    
    # Train dynamic if available
    if selected['has_dynamic']:
        dynamic_data = selected['path'] / 'keypoint_data_dynamic.csv'
        dynamic_labels = selected['path'] / 'keypoint_classifier_label_dynamic.csv'
        
        # Show training message
        window_name = show_training_message("Dynamic")
        
        # Train model
        dynamic_result = train_dynamic_model(dynamic_data, dynamic_labels)
        
        # Close message
        cv2.destroyWindow(window_name)
    
    # Step 4: Save models
    desktop = Path.home() / 'Desktop'
    output_dir = desktop / f'fingerspell_models_{datetime.now().strftime("%Y%m%d_%H%M")}'
    
    save_models(
        static_result,
        dynamic_result,
        static_labels if selected['has_static'] else None,
        dynamic_labels if selected['has_dynamic'] else None,
        output_dir
    )
    
    # Step 5: Show results
    show_training_results(static_result, dynamic_result, output_dir)
