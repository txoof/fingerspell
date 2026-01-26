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
    draw_instructions
)
from src.fingerspell.ui.user_input import clean_alphabet, get_text_input, create_label_mapping
from src.fingerspell.collection.data_management import save_final_data, discard_samples, draw_letter_status
from src.fingerspell.core.landmarks import calc_landmark_list, pre_process_landmark

def get_alphabet_configuration():
    """Get alphabet from user."""
    
    alphabet_input = get_text_input(
        prompt="Enter alphabet (default: A-Z):",
        default_value="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        window_name="Alphabet Configuration"
    )
    
    if alphabet_input is None:
        return None
    
    alphabet = clean_alphabet(alphabet_input)
    
    return alphabet

def get_dynamic_letters_configuration(alphabet):
    """
    Get dynamic letters from user via camera interface.
    
    Args:
        alphabet: List of valid alphabet characters
    
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
        window_name="Dynamic Letters"
    )
    
    if dynamic_input is None:
        dynamic_input = ""
    
    dynamic_letters = set(clean_alphabet(dynamic_input))
    
    return dynamic_letters

def show_configuration_summary(alphabet, dynamic_letters):
    """
    Show configuration summary and wait for user to start.
    
    Args:
        alphabet: List of alphabet characters
        dynamic_letters: Set of dynamic letters
    
    Returns:
        bool: True if user wants to proceed, False if cancelled
    """
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
            cap.release()
            cv2.destroyAllWindows()
            return False
        
        image = cv2.flip(image, 1)
        image = draw_modal_overlay(image, summary, position='center')
        
        cv2.imshow('Configuration Summary', image)
        
        key = cv2.waitKey(1)
        
        if key == 27:  # ESC - cancel
            cap.release()
            cv2.destroyAllWindows()
            return False
        elif key == 32:  # SPACE - proceed
            cap.release()
            cv2.destroyAllWindows()
            return True
        

def initialize_collection(alphabet, dynamic_letters):
    """
    Initialize collection state.
    
    Sets up temp file, MediaPipe, tracking dictionaries, and targets.
    
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
    
    # Create temp file
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='')
    csv_writer = csv.writer(temp_file)
    
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
        'temp_file': temp_file,
        'csv_writer': csv_writer,
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
    temp_file = state['temp_file']
    csv_writer = state['csv_writer']
    hands = state['hands']
    collected_per_letter = state['collected_per_letter']
    sample_id = state['sample_id']
    current_letter = state['current_letter']
    is_paused = state['is_paused']
    landmark_buffer = state['landmark_buffer']
    
    BUFFER_SIZE = 5
    
    # Camera setup
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    


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
                        csv_writer.writerow([sample_id, label_index] + current + delta)
                        temp_file.flush()
                        sample_id += 1
                        collected_per_letter[current_letter] += 1
                else:
                    label_index = label_map[current_letter]
                    csv_writer.writerow([sample_id, label_index] + normalized)
                    temp_file.flush()
                    sample_id += 1
                    collected_per_letter[current_letter] += 1
            
            # Draw UI
            target = targets[current_letter]
            current_count = collected_per_letter[current_letter]
            image = draw_progress_bar(image, current_count, target, current_letter, y_position=120)
            image = draw_instructions(image, is_paused=False, position='topright')
            image = draw_letter_status(image, alphabet, collected_per_letter, targets, dynamic_letters)
        
        cv2.imshow('Data Collection', image)
        
        key = cv2.waitKey(1)
        
        if key == 27:  # ESC - save and quit
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
                temp_file.flush()
                collected_per_letter = discard_samples(temp_file.name, alphabet, label_map, collected_per_letter)
                # Update total
                sample_id = sum(collected_per_letter.values())
        elif key == ord('S'):  # SHIFT+S - save and continue
            temp_file.flush()
            save_final_data(temp_file.name, alphabet, label_map)
    
    # Cleanup
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    
    # Final save
    temp_file.close()
    save_final_data(temp_file.name, alphabet, label_map)

def run_collection(project_root):
    """Run the data collection workflow."""
    
    # Step 1: Get alphabet
    alphabet = get_alphabet_configuration()
    if alphabet is None:
        return
    
    # Step 2: Get dynamic letters
    dynamic_letters = get_dynamic_letters_configuration(alphabet)
    
    # Step 3: Show summary and confirm
    if not show_configuration_summary(alphabet, dynamic_letters):
        return
    
    # Step 4: Initialize collection
    state = initialize_collection(alphabet, dynamic_letters)
    if state is None:
        return
    
    # Step 5: Run collection loop
    run_collection_loop(state)