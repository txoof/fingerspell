"""
Dataset naming utilities for collection.

Handles user input for dataset names and slugification.
"""

import unicodedata
import re
from datetime import datetime


def slugify(text):
    """
    Convert text to safe filesystem name.
    
    Rules:
    - Keep uppercase and lowercase
    - Replace spaces with underscores
    - Convert accented characters to nearest ASCII equivalent (ø→o, ä→a)
    - Remove all non-alphanumeric except underscores
    
    Args:
        text: User input text
        
    Returns:
        str: Slugified text safe for filesystem
        
    Examples:
        >>> slugify("Norwegian Alphabet")
        'Norwegian_Alphabet'
        >>> slugify("Español ÑÜ")
        'Espanol_NU'
        >>> slugify("Test@123#Data")
        'Test123Data'
    """
    # Replace spaces with underscores first
    text = text.replace(' ', '_')
    
    # Normalize unicode to NFKD (compatibility decomposition)
    # This handles more cases than NFD
    nfkd = unicodedata.normalize('NFKD', text)
    
    # Filter out combining characters (accent marks)
    ascii_text = ''.join(
        char for char in nfkd
        if unicodedata.category(char) != 'Mn'
    )
    
    # Manual replacements for characters that don't decompose well
    replacements = {
        'ł': 'l', 'Ł': 'L',
        'ø': 'o', 'Ø': 'O',
        'æ': 'ae', 'Æ': 'AE',
        'ß': 'ss',
        'đ': 'd', 'Đ': 'D',
        'ð': 'd', 'Ð': 'D',
        'þ': 'th', 'Þ': 'TH'
    }
    
    for old, new in replacements.items():
        ascii_text = ascii_text.replace(old, new)
    
    # Remove all characters except alphanumeric and underscore
    cleaned = re.sub(r'[^\w]', '', ascii_text)
    
    # Collapse multiple underscores to single
    cleaned = re.sub(r'_+', '_', cleaned)
    
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    
    return cleaned


def get_dataset_name(cap, window_name='Data Collection'):
    """
    Get dataset name from user via camera interface.
    
    Shows input modal, displays slugified preview, and confirms.
    If user enters nothing or cancels, uses default name.
    
    Args:
        cap: Existing cv2.VideoCapture object to reuse
        window_name: OpenCV window name
        
    Returns:
        str: Final directory name in format 'fingerspell_NAME_TIMESTAMP'
    """
    from src.fingerspell.ui.user_input import get_text_input
    
    # Get user input
    user_input = get_text_input(
        prompt="Name this dataset (ESC for default):",
        default_value="",
        window_name=window_name,
        cap=cap
    )
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Build final name
    if user_input and user_input.strip():
        slugified = slugify(user_input.strip())
        final_name = f"fingerspell_{slugified}_{timestamp}"
    else:
        final_name = f"fingerspell_data_{timestamp}"
    
    return final_name
