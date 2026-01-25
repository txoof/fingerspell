"""
Data management utilities for sign language collection.

Handles saving collected data and discarding samples during collection.
"""

import csv
from datetime import datetime
from pathlib import Path
import pandas as pd
from src.fingerspell.ui.common import draw_modal_input, draw_modal_overlay
import cv2


def save_final_data(temp_filename, alphabet, label_map):
    """
    Save final training data to Desktop.
    
    Strips sample_id column from temporary collection file and saves
    to a timestamped folder on the Desktop with label mapping.
    
    Args:
        temp_filename: Path to temporary CSV file with sample_id column
        alphabet: List of characters in the alphabet
        label_map: Dict mapping characters to label indices
    
    Returns:
        bool: True if save successful, False otherwise
    """
    print("Reading temp file...")
    
    # Read temp file manually to handle variable column counts
    # (static rows have 44 columns, dynamic rows have 86 columns)
    rows = []
    with open(temp_filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # Strip sample_id (first column), keep rest
            rows.append(row[1:])
    
    print(f"Read {len(rows)} rows")
    
    # Save to Desktop in timestamped folder
    desktop = Path.home() / "Desktop"
    output_dir = desktop / f"fingerspell_data_{datetime.now().strftime('%Y%m%d_%H%M')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save keypoints
    filepath = output_dir / "keypoint_data.csv"
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"Saved keypoints: {filepath}")
    
    # Save label mapping (sorted by label index)
    label_filepath = output_dir / "keypoint_classifier_label.csv"
    
    sorted_labels = sorted(label_map.items(), key=lambda x: x[1])
    with open(label_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        for letter, idx in sorted_labels:
            writer.writerow([letter])
    
    print(f"Saved labels: {label_filepath}")
    print(f"\nAll files saved to: {output_dir}")
    
    return True


def discard_samples(temp_filename, alphabet, label_map, collected_per_letter):
    """
    Interactive workflow to discard samples for a specific letter.
    
    Shows camera-based modals to select letter, specify count, and confirm.
    Removes samples from the temporary CSV file.
    
    Args:
        temp_filename: Path to temporary CSV file
        alphabet: List of valid characters
        label_map: Dict mapping characters to label indices
        collected_per_letter: Dict tracking samples per letter
    
    Returns:
        dict: Updated collected_per_letter dict
    """
    # Step 1: Get letter to discard
    cap = cv2.VideoCapture(0)
    letter_input = ""
    
    while True:
        ret, image = cap.read()
        if not ret:
            break
        
        image = cv2.flip(image, 1)
        image = draw_modal_input(image, "Which letter to discard?", letter_input)
        
        cv2.imshow('Discard', image)
        key = cv2.waitKey(1)
        
        if key == 27:  # ESC - cancel
            cap.release()
            cv2.destroyAllWindows()
            return collected_per_letter
        elif key == 13:  # ENTER
            if letter_input in alphabet:
                break
            else:
                letter_input = ""  # Invalid, clear
        elif key == 8:  # BACKSPACE
            letter_input = letter_input[:-1]
        elif 32 <= key <= 126:
            letter_input += chr(key).upper()
    
    target_letter = letter_input
    current_count = collected_per_letter[target_letter]
    
    # Step 2: Get count to discard
    count_input = ""
    
    while True:
        ret, image = cap.read()
        if not ret:
            break
        
        image = cv2.flip(image, 1)
        prompt = f"Discard how many samples of '{target_letter}'? (1-{current_count} or 'all')"
        image = draw_modal_input(image, prompt, count_input)
        
        cv2.imshow('Discard', image)
        key = cv2.waitKey(1)
        
        if key == 27:  # ESC - cancel
            cap.release()
            cv2.destroyAllWindows()
            return collected_per_letter
        elif key == 13:  # ENTER
            if count_input.lower() == 'all':
                discard_count = current_count
                break
            elif count_input.isdigit():
                discard_count = int(count_input)
                if 1 <= discard_count <= current_count:
                    break
            count_input = ""  # Invalid
        elif key == 8:  # BACKSPACE
            count_input = count_input[:-1]
        elif 32 <= key <= 126:
            count_input += chr(key)
    
    # Step 3: Confirm
    confirm_input = ""
    
    while True:
        ret, image = cap.read()
        if not ret:
            break
        
        image = cv2.flip(image, 1)
        message = f"Discard {discard_count} samples of '{target_letter}'?\n\nType 'yes' to confirm, ESC to cancel"
        image = draw_modal_overlay(image, message, position='center')
        
        # Show what they're typing
        cv2.putText(image, confirm_input, (image.shape[1]//2 - 50, image.shape[0]//2 + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        cv2.imshow('Discard', image)
        key = cv2.waitKey(1)
        
        if key == 27:  # ESC - cancel
            cap.release()
            cv2.destroyAllWindows()
            return collected_per_letter
        elif key == 13:  # ENTER
            if confirm_input.lower() == 'yes':
                break
            confirm_input = ""
        elif key == 8:  # BACKSPACE
            confirm_input = confirm_input[:-1]
        elif 32 <= key <= 126:
            confirm_input += chr(key)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Actually discard from CSV
    print(f"Discarding {discard_count} samples of '{target_letter}'...")
    
    # Read CSV
    df = pd.read_csv(temp_filename, header=None)
    
    # Get label index for this letter
    target_label = label_map[target_letter]
    
    # Find all rows with this label
    mask = df[1] == target_label  # Column 1 is label_index
    matching_indices = df[mask].index.tolist()
    
    # Remove last N samples
    if discard_count >= len(matching_indices):
        # Remove all
        df = df[~mask]
    else:
        # Remove last N
        indices_to_remove = matching_indices[-discard_count:]
        df = df.drop(indices_to_remove)
    
    # Rewrite CSV
    df.to_csv(temp_filename, header=False, index=False)
    
    # Update tracking
    collected_per_letter[target_letter] -= discard_count
    
    print(f"Discarded {discard_count} samples. Remaining for '{target_letter}': {collected_per_letter[target_letter]}")
    
    return collected_per_letter

def draw_letter_status(image, alphabet, collected_per_letter, targets, dynamic_letters):
    """
    Draw letter status with color coding and checkboxes.
    
    Args:
        image: OpenCV image to draw on
        alphabet: List of letters in order
        collected_per_letter: Dict mapping letter -> count
        targets: Dict mapping letter -> target count
        dynamic_letters: Set of dynamic letters
    
    Returns:
        image: Image with letter status drawn
    """
    h = image.shape[0]
    y_start = h - 80
    
    # Semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (10, y_start), (image.shape[1] - 10, h - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # Build status text
    text = ""
    for letter in alphabet:
        count = collected_per_letter.get(letter, 0)
        target = targets.get(letter, 0)
        is_complete = count >= target
        
        # Checkbox: ☑ for complete, ☐ for incomplete
        checkbox = chr(0x2611) if is_complete else chr(0x2610)
        text += f"{checkbox}{letter}({count}) "
    
    # Draw text (may need to wrap if too long)
    cv2.putText(image, text, (20, y_start + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image