"""
Collection workflow orchestration.

Handles the complete data collection process from alphabet configuration
through sample collection to final data export.
"""

import csv
import tempfile
import numpy as np
import mediapipe as mp
from collections import deque
import cv2
from src.fingerspell.ui.common import (
    draw_modal_input, 
    draw_modal_overlay,
    draw_landmarks,
    draw_progress_bar,
    draw_text_window
)
from src.fingerspell.ui.user_input import clean_alphabet, get_text_input, create_label_mapping
from src.fingerspell.collection.data_management import (
    save_final_data, 
    discard_samples, 
    draw_letter_status,
    show_save_confirmation,
    show_save_success
)
from src.fingerspell.core.landmarks import calc_landmark_list, pre_process_landmark

def draw_instructions(image, is_paused, position='topright', project_root='./'):
    """
    Draw on-screen instructions using draw_text_window.
    """

    if is_paused:
        instructions = [
            'PAUSED',
            'SPACE - Resume',
            'Letter - Switch',
            'SHIFT+D - Discard',
            'SHIFT+S - Save',
            'ESC - Quit & Save'
        ]
    else:
        instructions = [
            'COLLECTING',
            'SPACE - Pause',
            'SHIFT+D - Discard',
            'ESC - Quit'
        ]

    image = draw_text_window(
        image=image,
        text=instructions,
        font_size=20,
        first_line_color=(0, 255, 255),
        color=(255, 255, 255),
        position=position,
        project_root=project_root,
        wrap=False
    )

    return image

def get_alphabet_configuration(cap=None, window_name="Alphabet Configuration"):
    """
    Get alphabet from user.
    
    Args:
        cap: Optional existing cv2.VideoCapture to reuse
        window_name: OpenCV window name
    
    Returns:
        list: Cleaned alphabet or None if cancelled
    """
    alphabet_input = get_text_input(
        prompt="Enter alphabet (default: A-Z):",
        default_value="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        window_name=window_name,
        cap=cap
    )
    
    if alphabet_input is None:
        return None
    
    alphabet = clean_alphabet(alphabet_input)
    
    return alphabet

def get_dynamic_letters_configuration(alphabet, cap=None, window_name="Dynamic Letters"):
    """
    Get dynamic letters from user via camera interface.
    
    Args:
        alphabet: List of valid alphabet characters
        cap: Optional existing cv2.VideoCapture to reuse
        window_name: OpenCV window name
    
    Returns:
        set: Set of dynamic letters, or empty set if none/cancelled
    """
    # Validation function - only allow letters in alphabet
    def validate_in_alphabet(char, current_input):
        return char in alphabet
    
    dynamic_input = get_text_input(
        prompt="Enter dynamic letters (ESC for none):",
        validation_fn=validate_in_alphabet,
        default_value="",
        window_name=window_name,
        cap=cap
    )
    
    if dynamic_input is None:
        dynamic_input = ""
    
    dynamic_letters = set(clean_alphabet(dynamic_input))
    
    return dynamic_letters

def show_configuration_summary(alphabet, dynamic_letters, cap=None, window_name='Configuration Summary'):
    """
    Show configuration summary and wait for user to start.
    
    Args:
        alphabet: List of alphabet characters
        dynamic_letters: Set of dynamic letters
        cap: Optional existing cv2.VideoCapture to reuse
        window_name: OpenCV window name
    
    Returns:
        bool: True if user wants to proceed, False if cancelled
    """
    # Use provided camera or create new one
    owns_camera = (cap is None)
    if owns_camera:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Build summary text
    summary = "Configuration Complete\n\n"
    summary += f"Alphabet: {''.join(alphabet)} ({len(alphabet)} letters)\n\n"
    summary += f"Dynamic: {', '.join(sorted(dynamic_letters)) if dynamic_letters else 'None'}\n\n"
    summary += "Targets:\n"
    summary += "  Static letters: 2500 samples\n"
    summary += "  Dynamic letters: 3500 samples\n\n"
    summary += "Press SPACE to start collecting\n"
    summary += "Press ESC to cancel"
    
    while True:
        ret, image = cap.read()
        if not ret:
            if owns_camera:
                cap.release()
                cv2.destroyAllWindows()
            return False
        
        image = cv2.flip(image, 1)
        image = draw_modal_overlay(image, summary, position='center')
        
        cv2.imshow(window_name, image)
        
        key = cv2.waitKey(1)
        
        if key == 27:  # ESC - cancel
            if owns_camera:
                cap.release()
                cv2.destroyAllWindows()
            return False
        elif key == 32:  # SPACE - proceed
            if owns_camera:
                cap.release()
                cv2.destroyAllWindows()
            return True
        

def initialize_collection(alphabet, dynamic_letters):
    """
    Initialize collection state.
    
    Sets up temp files (separate for static/dynamic), MediaPipe, tracking dictionaries, and targets.
    
    Args:
        alphabet: List of alphabet characters
        dynamic_letters: Set of dynamic letters
    
    Returns:
        dict: Collection state containing all necessary variables, or None if setup fails
    """
    # Create label mapping
    label_map = create_label_mapping(alphabet)
    
    # Set targets
    STATIC_TARGET = 2500
    DYNAMIC_TARGET = 3500
    targets = {}
    for char in alphabet:
        targets[char] = DYNAMIC_TARGET if char in dynamic_letters else STATIC_TARGET
    
    # Create temp files - separate for static and dynamic
    temp_file_static = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_static.csv', newline='')
    csv_writer_static = csv.writer(temp_file_static)
    
    temp_file_dynamic = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_dynamic.csv', newline='')
    csv_writer_dynamic = csv.writer(temp_file_dynamic)
    
    # Setup MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # Initialize tracking
    collected_per_letter = {letter: 0 for letter in alphabet}
    
    # Return all state
    return {
        'alphabet': alphabet,
        'dynamic_letters': dynamic_letters,
        'label_map': label_map,
        'targets': targets,
        'temp_file_static': temp_file_static,
        'csv_writer_static': csv_writer_static,
        'temp_file_dynamic': temp_file_dynamic,
        'csv_writer_dynamic': csv_writer_dynamic,
        'hands': hands,
        'collected_per_letter': collected_per_letter,
        'sample_id': 0,
        'current_letter': alphabet[0],
        'is_paused': True,
        'landmark_buffer': deque(maxlen=5)
    }

def run_collection_loop(state):
    """
    Main collection loop.
    
    Args:
        state: Dict from initialize_collection() with all necessary variables
    """
    
    # Unpack state
    alphabet = state['alphabet']
    dynamic_letters = state['dynamic_letters']
    label_map = state['label_map']
    targets = state['targets']
    temp_file_static = state['temp_file_static']
    csv_writer_static = state['csv_writer_static']
    temp_file_dynamic = state['temp_file_dynamic']
    csv_writer_dynamic = state['csv_writer_dynamic']
    hands = state['hands']
    collected_per_letter = state['collected_per_letter']
    sample_id = state['sample_id']
    current_letter = state['current_letter']
    is_paused = state['is_paused']
    landmark_buffer = state['landmark_buffer']
    project_root = state['project_root']
    cap = state['cap']  # Use camera from state
    
    BUFFER_SIZE = 5
    


    while True:
        ret, image = cap.read()
        if not ret:
            break
        
        image = cv2.flip(image, 1)

        
        if is_paused:
            # Show modal when paused
            target = targets[current_letter]
            current_count = collected_per_letter[current_letter]
            instructions_text = f"Ready to collect: {current_letter}\n\nPress SPACE to start\nTarget: {current_count}/{target}"
            image = draw_modal_overlay(image, instructions_text, position='center')
        else:
            # Process hand detection
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmark_list = calc_landmark_list(image, hand_landmarks)
                image = draw_landmarks(image, landmark_list)
                normalized = pre_process_landmark(landmark_list)
                
                is_dynamic = current_letter in dynamic_letters
                
                if is_dynamic:
                    landmark_buffer.append(normalized)
                    if len(landmark_buffer) >= BUFFER_SIZE:
                        current = landmark_buffer[-1]
                        old = landmark_buffer[0]
                        delta = [c - o for c, o in zip(current, old)]
                        
                        label_index = label_map[current_letter]
                        csv_writer_dynamic.writerow([sample_id, label_index] + current + delta)
                        temp_file_dynamic.flush()
                        sample_id += 1
                        collected_per_letter[current_letter] += 1
                else:
                    label_index = label_map[current_letter]
                    csv_writer_static.writerow([sample_id, label_index] + normalized)
                    temp_file_static.flush()
                    sample_id += 1
                    collected_per_letter[current_letter] += 1
            
            # Draw UI
            target = targets[current_letter]
            current_count = collected_per_letter[current_letter]
            image = draw_progress_bar(image, current_count, target, current_letter, y_position=120, project_root=project_root)
            image = draw_instructions(image, is_paused=False, position='topright', project_root=project_root)
            image = draw_letter_status(image, alphabet, collected_per_letter, targets, dynamic_letters)
        
        cv2.imshow('Data Collection', image)
        
        key = cv2.waitKey(1)
        
        if key == 27:  # ESC - check if we should save
            # Check if any samples were collected
            total_samples = sum(collected_per_letter.values())
            
            if total_samples > 0:
                # Ask user if they want to save (reuse camera)
                if show_save_confirmation(cap, 'Data Collection'):
                    # Close both temp files before reading
                    temp_file_static.flush()
                    temp_file_static.close()
                    temp_file_dynamic.flush()
                    temp_file_dynamic.close()
                    
                    # Save data
                    save_path = save_final_data(
                        temp_file_static.name,
                        temp_file_dynamic.name,
                        alphabet,
                        label_map,
                        dynamic_letters
                    )
                    
                    if save_path:
                        # Show success message (reuse camera)
                        show_save_success(cap, save_path, 'Data Collection')
            
            # Exit regardless of save decision
            break
        elif key == 32:  # SPACE - pause/resume
            is_paused = not is_paused
            if is_paused:
                landmark_buffer.clear()
        elif 97 <= key <= 122:  # lowercase a-z - switch letter
            letter = chr(key).upper()
            if letter in alphabet:
                current_letter = letter
                is_paused = True
                landmark_buffer.clear()
        elif key == ord('D'):  # SHIFT+D - discard
            if is_paused:
                temp_file_static.flush()
                temp_file_dynamic.flush()
                collected_per_letter = discard_samples(
                    cap,
                    temp_file_static.name,
                    temp_file_dynamic.name,
                    alphabet,
                    label_map,
                    collected_per_letter,
                    dynamic_letters,
                    'Data Collection'
                )
                # Update total
                sample_id = sum(collected_per_letter.values())
        elif key == ord('S'):  # SHIFT+S - save and continue
            # Close both temp files before reading
            temp_file_static.flush()
            temp_file_static.close()
            temp_file_dynamic.flush()
            temp_file_dynamic.close()
            
            # Save data
            save_path = save_final_data(
                temp_file_static.name,
                temp_file_dynamic.name,
                alphabet,
                label_map,
                dynamic_letters
            )
            
            if save_path:
                show_save_success(cap, save_path, 'Data Collection')
            
            # Reopen both temp files for continued collection
            temp_file_static = open(temp_file_static.name, 'a', newline='')
            csv_writer_static = csv.writer(temp_file_static)
            temp_file_dynamic = open(temp_file_dynamic.name, 'a', newline='')
            csv_writer_dynamic = csv.writer(temp_file_dynamic)
    
    # Cleanup
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    
    # Close both temp files
    temp_file_static.close()
    temp_file_dynamic.close()
    
    return temp_file_static.name, temp_file_dynamic.name, alphabet, label_map

def run_collection(project_root):
    """Run the data collection workflow."""
    
    # Create single camera for entire configuration process
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    window_name = "Data Collection"
    
    # Step 1: Get alphabet
    alphabet = get_alphabet_configuration(cap, window_name)
    if alphabet is None:
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # Step 2: Get dynamic letters
    dynamic_letters = get_dynamic_letters_configuration(alphabet, cap, window_name)
    
    # Step 3: Show summary and confirm
    if not show_configuration_summary(alphabet, dynamic_letters, cap, window_name):
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # Camera stays open - don't release it!
    # Step 4: Initialize collection
    state = initialize_collection(alphabet, dynamic_letters)
    if state is None:
        cap.release()
        cv2.destroyAllWindows()
        return
    state['project_root'] = project_root
    state['cap'] = cap  # Pass camera to collection loop
    
    # Step 5: Run collection loop (will handle camera cleanup)
    temp_file_static, temp_file_dynamic, alphabet, label_map = run_collection_loop(state)


