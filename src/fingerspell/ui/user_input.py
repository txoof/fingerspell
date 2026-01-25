"""
User input utilities for alphabet configuration.

Handles alphabet input processing, validation, and label mapping.
"""

import cv2
# Note: In production, use: from src.fingerspell.ui.common import draw_modal_overlay
from src.fingerspell.ui.common import draw_modal_overlay
from src.fingerspell.ui.common import draw_modal_overlay, draw_modal_input



def clean_alphabet(raw_input):
    """
    Clean and normalize alphabet input.
    
    - Remove spaces
    - Convert to uppercase
    - Remove duplicates while preserving order
    
    Args:
        raw_input: String from user
    
    Returns:
        list: Cleaned list of unique characters in order entered
    
    Examples:
        >>> clean_alphabet("abc def")
        ['A', 'B', 'C', 'D', 'E', 'F']
        >>> clean_alphabet("aabbcc")
        ['A', 'B', 'C']
    """
    cleaned = []
    seen = set()
    
    for char in raw_input.upper().replace(' ', ''):
        if char not in seen:
            cleaned.append(char)
            seen.add(char)
    
    return cleaned


def create_label_mapping(alphabet_list):
    """
    Create label mapping sorted by unicode codepoint.
    
    Args:
        alphabet_list: List of characters (in any order)
    
    Returns:
        dict: Mapping from character to label index (sorted by unicode)
    
    Examples:
        >>> create_label_mapping(['Z', 'A', 'B'])
        {'A': 0, 'B': 1, 'Z': 2}
        >>> create_label_mapping(['Ø', 'A', 'Æ'])
        {'A': 0, 'Æ': 1, 'Ø': 2}
    """
    sorted_alphabet = sorted(alphabet_list, key=lambda x: ord(x))
    return {char: idx for idx, char in enumerate(sorted_alphabet)}


def show_validation_warnings(alphabet):
    """
    Show warnings for non-standard characters in alphabet.
    
    Opens camera window and displays warning if non-letter characters found.
    User presses any key to continue.
    
    Args:
        alphabet: List of characters to validate
    
    Returns:
        None (displays warning via camera if needed)
    """
    non_standard = [char for char in alphabet if not char.isalpha()]
    
    if not non_standard:
        return  # No warnings needed
    
    cap = cv2.VideoCapture(0)
    
    warning_text = f"Found non-standard characters: {', '.join(non_standard)}. Models work best with letters only."
    instructions = "Press any key to continue."
    full_text = f"{warning_text}\n\n{instructions}"
    
    print(f"WARNING: {warning_text}")
    
    while True:
        ret, image = cap.read()
        if not ret:
            break
        
        image = cv2.flip(image, 1)
        image = draw_modal_overlay(image, full_text, position='center')
        
        cv2.imshow('Validation Warning', image)
        
        if cv2.waitKey(1) != -1:  # Any key pressed
            break
    
    cap.release()
    cv2.destroyAllWindows()

def get_text_input(prompt, validation_fn=None, default_value="", window_name="Input"):
    """
    Get text input from user via camera interface.
    
    Displays modal overlay with prompt and accumulates keyboard input.
    Handles ENTER to confirm, BACKSPACE to delete (cross-platform), ESC for default/cancel.
    
    Args:
        prompt: Prompt text to show user
        validation_fn: Optional function(char, current_input) -> bool to validate each character
        default_value: Default value if ESC pressed or empty ENTER
        window_name: OpenCV window name
    
    Returns:
        str: User input string, or None if validation_fn rejects ENTER
    
    Example:
        # Simple input
        alphabet = get_text_input("Enter alphabet:", default_value="ABC")
        
        # With validation
        def only_letters(char, current):
            return char.isalpha()
        
        result = get_text_input("Letters only:", validation_fn=only_letters)
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    current_input = ""
    error_msg = None
    
    print(f"\n{prompt}")
    if default_value:
        print(f"(Default: {default_value})")
    
    while True:
        ret, image = cap.read()
        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            return None
        
        image = cv2.flip(image, 1)
        
        # Show current input or default
        display = current_input if current_input else f"[{default_value}]" if default_value else ""
        image = draw_modal_input(image, prompt, display, error_msg)
        
        cv2.imshow(window_name, image)
        
        key = cv2.waitKey(1)
        
        if key == 27:  # ESC - use default
            cap.release()
            cv2.destroyAllWindows()
            return default_value
        
        elif key == 13:  # ENTER - confirm
            result = current_input if current_input else default_value
            cap.release()
            cv2.destroyAllWindows()
            return result
        
        elif key == 8 or key == 127:  # BACKSPACE (cross-platform)
            current_input = current_input[:-1]
            error_msg = None
        
        elif 32 <= key <= 126:  # Printable character
            char = chr(key).upper()
            
            # Validate if function provided
            if validation_fn is None or validation_fn(char, current_input):
                current_input += char
                error_msg = None
            else:
                error_msg = f"'{char}' not allowed"
    
    cap.release()
    cv2.destroyAllWindows()
    return current_input