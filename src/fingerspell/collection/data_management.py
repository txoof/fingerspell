"""
Data management utilities for sign language collection.

Handles saving collected data, discarding samples, and displaying collection status.
"""

import csv
from datetime import datetime
from pathlib import Path
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
from src.fingerspell.ui.common import draw_modal_input, draw_modal_overlay, draw_text, draw_text_window


def save_final_data(temp_filename, alphabet, label_map, dynamic_letters):
    """
    Save final training data to Desktop, split into static and dynamic files.
    
    Reads combined collection data and separates into:
    - Static letters: 43 columns (label + 42 landmarks)
    - Dynamic letters: 85 columns (label + 42 landmarks + 42 deltas)
    
    Each file gets continuous label indices (0, 1, 2...) with corresponding label files.
    
    Args:
        temp_filename: Path to temporary CSV file with sample_id column
        alphabet: List of characters in the alphabet
        label_map: Dict mapping characters to original label indices
        dynamic_letters: Set of dynamic letter characters
    
    Returns:
        str: Path to output directory if save successful, None if no data to save
    """
    # Read temp file manually to handle variable column counts
    rows = []
    try:
        with open(temp_filename, 'r', newline='', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            for row in reader:
                # Skip empty rows or rows with NUL bytes
                if not row or any('\x00' in cell for cell in row):
                    continue
                # Strip sample_id (first column), keep rest
                rows.append(row[1:])
    except Exception as e:
        print(f"Error reading temp file: {e}")
        return None
    
    # Check if we have any data
    if len(rows) == 0:
        return None
    
    # Separate static and dynamic rows
    static_rows = []
    dynamic_rows = []
    
    for row in rows:
        col_count = len(row)
        label_idx = int(row[0])
        
        # Find which letter this label corresponds to
        letter = None
        for char, idx in label_map.items():
            if idx == label_idx:
                letter = char
                break
        
        if letter is None:
            raise ValueError(f"Unknown label index: {label_idx}")
        
        # Validate column count and categorize
        if letter in dynamic_letters:
            # Dynamic: label + 42 landmarks + 42 deltas = 85 columns
            if col_count != 85:
                raise ValueError(f"Dynamic letter '{letter}' has {col_count} columns, expected 85")
            dynamic_rows.append((letter, row))
        else:
            # Static: label + 42 landmarks = 43 columns
            if col_count != 43:
                raise ValueError(f"Static letter '{letter}' has {col_count} columns, expected 43")
            static_rows.append((letter, row))
    
    # Create output directory
    desktop = Path.home() / "Desktop"
    output_dir = desktop / f"fingerspell_data_{datetime.now().strftime('%Y%m%d_%H%M')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process static data
    if static_rows:
        # Get unique static letters in sorted order
        static_letters = sorted(set(letter for letter, _ in static_rows), key=lambda x: ord(x))
        static_label_map = {char: idx for idx, char in enumerate(static_letters)}
        
        # Relabel rows
        relabeled_static = []
        for letter, row in static_rows:
            new_label = static_label_map[letter]
            relabeled_static.append([new_label] + row[1:])  # Replace old label with new
        
        # Save static keypoints
        static_path = output_dir / "keypoint_data_static.csv"
        with open(static_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(relabeled_static)
        
        # Save static labels
        static_label_path = output_dir / "keypoint_classifier_label_static.csv"
        with open(static_label_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for letter in static_letters:
                writer.writerow([letter])
    
    # Process dynamic data
    if dynamic_rows:
        # Get unique dynamic letters in sorted order
        dynamic_letters_list = sorted(set(letter for letter, _ in dynamic_rows), key=lambda x: ord(x))
        dynamic_label_map = {char: idx for idx, char in enumerate(dynamic_letters_list)}
        
        # Relabel rows
        relabeled_dynamic = []
        for letter, row in dynamic_rows:
            new_label = dynamic_label_map[letter]
            relabeled_dynamic.append([new_label] + row[1:])  # Replace old label with new
        
        # Save dynamic keypoints
        dynamic_path = output_dir / "keypoint_data_dynamic.csv"
        with open(dynamic_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(relabeled_dynamic)
        
        # Save dynamic labels
        dynamic_label_path = output_dir / "keypoint_classifier_label_dynamic.csv"
        with open(dynamic_label_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for letter in dynamic_letters_list:
                writer.writerow([letter])
    
    return str(output_dir)


def discard_samples(cap, temp_filename, alphabet, label_map, collected_per_letter, window_name='Data Collection'):
    """
    Interactive workflow to discard samples for a specific letter.
    
    Shows camera-based modals to select letter, specify count, and confirm.
    Removes samples from the temporary CSV file.
    
    Args:
        cap: Existing cv2.VideoCapture object to reuse
        temp_filename: Path to temporary CSV file
        alphabet: List of valid characters
        label_map: Dict mapping characters to label indices
        collected_per_letter: Dict tracking samples per letter
        window_name: Name of the window to display in
    
    Returns:
        dict: Updated collected_per_letter dict
    """
    # Step 1: Get letter to discard
    letter_input = ""
    
    while True:
        ret, image = cap.read()
        if not ret:
            break
        
        image = cv2.flip(image, 1)
        image = draw_modal_input(image, "Which letter to discard?", letter_input)
        
        cv2.imshow(window_name, image)
        key = cv2.waitKey(1)
        
        if key == 27:  # ESC - cancel
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
        
        cv2.imshow(window_name, image)
        key = cv2.waitKey(1)
        
        if key == 27:  # ESC - cancel
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
        
        cv2.imshow(window_name, image)
        key = cv2.waitKey(1)
        
        if key == 27:  # ESC - cancel
            return collected_per_letter
        elif key == 13:  # ENTER
            if confirm_input.lower() == 'yes':
                break
            confirm_input = ""
        elif key == 8:  # BACKSPACE
            confirm_input = confirm_input[:-1]
        elif 32 <= key <= 126:
            confirm_input += chr(key)
    
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
    Draw letter status as a full width, wrapped text window at the bottom.
    """
    parts = []
    for letter in alphabet:
        count = collected_per_letter.get(letter, 0)
        target = targets.get(letter, 0)
        is_complete = count >= target
        checkbox = '☑' if is_complete else '☐'
        parts.append(f'{checkbox}{letter}({count})')

    status = ' '.join(parts)

    return draw_text_window(
        image=image,
        text=status,
        font_size=24,
        first_line_color=(255, 255, 255),
        color=(255, 255, 255),
        position='bottomleft',
        margin=10,
        padding=20,
        bg_color=(0, 0, 0),
        bg_alpha=0.85,
        border_color=(100, 100, 100),
        wrap=True,
        fill_width=True
    )


def show_save_confirmation(cap, window_name='Data Collection'):
    """
    Show modal asking user if they want to save collected data.
    
    Args:
        cap: Existing cv2.VideoCapture object to reuse
        window_name: Name of the window to display in
    
    Returns:
        bool: True if user wants to save (pressed Y), False if not (pressed N or ESC)
    """
    message = "Save collected data?\n\nPress Y to save\nPress N or ESC to exit without saving"
    
    while True:
        ret, image = cap.read()
        if not ret:
            return False
        
        image = cv2.flip(image, 1)
        image = draw_modal_overlay(image, message, position='center')
        
        cv2.imshow(window_name, image)
        
        key = cv2.waitKey(1)
        
        if key == ord('Y') or key == ord('y'):
            return True
        elif key == ord('N') or key == ord('n') or key == 27:  # N or ESC
            return False


def show_save_success(cap, save_path, window_name='Data Collection'):
    """
    Show modal with successful save path and wait for keypress.
    
    Args:
        cap: Existing cv2.VideoCapture object to reuse
        save_path: Path where data was saved
        window_name: Name of the window to display in
    """
    message = f"Data saved successfully!\n\nLocation:\n{save_path}\n\nPress any key to close"
    
    while True:
        ret, image = cap.read()
        if not ret:
            return
        
        image = cv2.flip(image, 1)
        image = draw_modal_overlay(image, message, position='center', width_percent=80)
        
        cv2.imshow(window_name, image)
        
        key = cv2.waitKey(1)
        
        if key != -1:  # Any key pressed
            return

# def draw_letter_status(image, alphabet, collected_per_letter, targets, dynamic_letters):
#     """
#     Draw letter status with auto-wrapping for long alphabets.
#     """
#     h, w = image.shape[:2]
    
#     # Build status text with Unicode checkboxes
#     text = ""
#     for letter in alphabet:
#         count = collected_per_letter.get(letter, 0)
#         target = targets.get(letter, 0)
#         is_complete = count >= target
        
#         # Unicode ballot box: ☑ for complete, ☐ for incomplete
#         checkbox = "☑" if is_complete else "☐"
#         text += f"{checkbox}{letter}({count}) "
    
#     # Measure text to calculate wrapping
#     font_size = 24
#     bundled_font = Path("../assets/fonts/DejaVuSans.ttf")
#     try:
#         font = ImageFont.truetype(str(bundled_font), font_size)
#     except:
#         font = ImageFont.load_default()
    
#     pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     draw_obj = ImageDraw.Draw(pil_image)
    
#     # Calculate available width
#     available_width = w - 40  # 20px padding on each side
    
#     # Wrap text manually by measuring
#     words = text.split()
#     lines = []
#     current_line = ""
    
#     for word in words:
#         test_line = current_line + word + " "
#         bbox = draw_obj.textbbox((0, 0), test_line, font=font)
#         line_width = bbox[2] - bbox[0]
        
#         if line_width <= available_width:
#             current_line = test_line
#         else:
#             if current_line:
#                 lines.append(current_line.strip())
#             current_line = word + " "
    
#     if current_line:
#         lines.append(current_line.strip())
    
#     # Calculate box height based on number of lines
#     line_height = 35
#     padding = 20
#     box_height = len(lines) * line_height + padding * 2
#     y_start = h - box_height - 10
    
#     # Semi-transparent background
#     overlay = image.copy()
#     cv2.rectangle(overlay, (10, y_start), (w - 10, h - 10), (0, 0, 0), -1)
#     cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)
    
#     # Draw each line
#     y_text = y_start + padding
#     for line in lines:
#         image = draw_text(image, line, (20, y_text),
#                          font_size=font_size, color=(255, 255, 255))
#         y_text += line_height
    
#     return image