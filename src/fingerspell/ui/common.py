"""
Common UI utilities shared across recognition and collection.

Provides modal overlay functions for displaying text and getting user input,
and hand skeleton drawing.
"""

import cv2
import textwrap


def draw_modal_overlay(image, text, position='center', width_percent=70):
    """
    Draw modal overlay with text.
    
    Displays semi-transparent dark overlay covering entire frame with
    text box at specified position. Text wraps automatically.
    
    Args:
        image: Frame to draw on (modified in place)
        text: Text to display (will wrap automatically)
        position: 'top', 'center', or 'bottom'
        width_percent: Width of text box as percentage of frame width (10-90)
    
    Returns:
        Modified image
    """
    h, w = image.shape[:2]
    
    # Clamp width_percent
    width_percent = max(10, min(90, width_percent))
    box_width = int(w * width_percent / 100)
    
    # Full screen semi-transparent overlay
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # Calculate text wrapping
    # Estimate characters per line based on box width
    font_scale = 0.7
    chars_per_line = int(box_width / 12)  # Rough estimate for font size
    wrapped_lines = textwrap.wrap(text, width=chars_per_line)
    
    # Calculate box height based on number of lines
    line_height = 30
    padding = 40
    box_height = len(wrapped_lines) * line_height + padding * 2
    
    # Calculate box position
    box_x = (w - box_width) // 2
    if position == 'top':
        box_y = 50
    elif position == 'bottom':
        box_y = h - box_height - 50
    else:  # center
        box_y = (h - box_height) // 2
    
    # Draw text box
    cv2.rectangle(image, (box_x, box_y), 
                 (box_x + box_width, box_y + box_height),
                 (40, 40, 40), -1)
    cv2.rectangle(image, (box_x, box_y),
                 (box_x + box_width, box_y + box_height),
                 (200, 200, 200), 2)
    
    # Draw wrapped text lines
    y_text = box_y + padding + 20
    for line in wrapped_lines:
        cv2.putText(image, line, (box_x + padding, y_text),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        y_text += line_height
    
    return image


def draw_modal_input(image, prompt, current_input, error_msg=None, 
                     position='center', width_percent=70):
    """
    Draw modal overlay for user input.
    
    Displays prompt, input box showing current typed text, optional error,
    and instructions for ENTER/BACKSPACE/ESC.
    
    Args:
        image: Frame to draw on (modified in place)
        prompt: Prompt text to show user
        current_input: Current text user has typed
        error_msg: Optional error message to display in red
        position: 'top', 'center', or 'bottom'
        width_percent: Width of box as percentage of frame width (10-90)
    
    Returns:
        Modified image
    """
    h, w = image.shape[:2]
    
    # Clamp width_percent
    width_percent = max(10, min(90, width_percent))
    box_width = int(w * width_percent / 100)
    
    # Full screen semi-transparent overlay
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # Calculate box height (fixed for input)
    box_height = 200 if error_msg else 160
    
    # Calculate box position
    box_x = (w - box_width) // 2
    if position == 'top':
        box_y = 50
    elif position == 'bottom':
        box_y = h - box_height - 50
    else:  # center
        box_y = (h - box_height) // 2
    
    # Draw main box
    cv2.rectangle(image, (box_x, box_y),
                 (box_x + box_width, box_y + box_height),
                 (40, 40, 40), -1)
    cv2.rectangle(image, (box_x, box_y),
                 (box_x + box_width, box_y + box_height),
                 (200, 200, 200), 2)
    
    # Draw prompt
    cv2.putText(image, prompt, (box_x + 20, box_y + 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw input box
    input_y = box_y + 60
    cv2.rectangle(image, (box_x + 20, input_y),
                 (box_x + box_width - 20, input_y + 40),
                 (255, 255, 255), 2)
    
    # Draw current input text
    display_text = current_input if current_input else ""
    cv2.putText(image, display_text, (box_x + 30, input_y + 28),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw error if present
    y_offset = 110
    if error_msg:
        cv2.putText(image, error_msg, (box_x + 20, box_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += 30
    
    # Draw instructions
    instructions = "ENTER=Confirm | BACKSPACE=Delete | ESC=Cancel"
    cv2.putText(image, instructions, (box_x + 20, box_y + y_offset + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return image


def draw_landmarks(image, landmark_point):
    """
    Draw hand skeleton connections on image.
    
    Draws lines connecting hand landmarks to create a skeleton visualization.
    Uses black outline with white lines for visibility.
    
    Args:
        image: OpenCV image to draw on (modified in place)
        landmark_point: List of [x, y] coordinate pairs (21 landmarks)
    
    Returns:
        image: Image with hand skeleton drawn
    """
    if len(landmark_point) == 0:
        return image
    
    def draw_line(p1, p2):
        """Draw a line between two landmark points with outline."""
        cv2.line(image, tuple(landmark_point[p1]), tuple(landmark_point[p2]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[p1]), tuple(landmark_point[p2]), (255, 255, 255), 2)
    
    # Thumb
    draw_line(2, 3)
    draw_line(3, 4)
    
    # Index finger
    draw_line(5, 6)
    draw_line(6, 7)
    draw_line(7, 8)
    
    # Middle finger
    draw_line(9, 10)
    draw_line(10, 11)
    draw_line(11, 12)
    
    # Ring finger
    draw_line(13, 14)
    draw_line(14, 15)
    draw_line(15, 16)
    
    # Pinky
    draw_line(17, 18)
    draw_line(18, 19)
    draw_line(19, 20)
    
    # Palm
    draw_line(0, 1)
    draw_line(1, 2)
    draw_line(2, 5)
    draw_line(5, 9)
    draw_line(9, 13)
    draw_line(13, 17)
    draw_line(17, 0)
    
    return image


# Smoke test
if __name__ == '__main__':
    import numpy as np
    
    print("common.py smoke test")
    print("=" * 50)
    
    # Create test frame
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    test_frame[:] = (50, 100, 150)  # Some color so we can see overlay
    
    # Test draw_modal_overlay
    print("\nTesting draw_modal_overlay...")
    
    # Short text, center
    frame1 = test_frame.copy()
    result1 = draw_modal_overlay(frame1, "Collection complete!", position='center')
    print(f"  Center position: OK (shape: {result1.shape})")
    
    # Long text that wraps, top
    long_text = "This is a very long message that should wrap across multiple lines automatically when displayed to the user. It will test the text wrapping functionality."
    frame2 = test_frame.copy()
    result2 = draw_modal_overlay(frame2, long_text, position='top', width_percent=60)
    print(f"  Top position with wrapping: OK")
    
    # Bottom position
    frame3 = test_frame.copy()
    result3 = draw_modal_overlay(frame3, "Bottom message", position='bottom')
    print(f"  Bottom position: OK")
    
    # Test draw_modal_input
    print("\nTesting draw_modal_input...")
    
    # Basic input
    frame4 = test_frame.copy()
    result4 = draw_modal_input(frame4, "Enter alphabet:", "ABCDEFGH")
    print(f"  Basic input: OK")
    
    # With error
    frame5 = test_frame.copy()
    result5 = draw_modal_input(frame5, "Enter dynamic letters:", "123", 
                               error_msg="Numbers not allowed")
    print(f"  Input with error: OK")
    
    # Empty input
    frame6 = test_frame.copy()
    result6 = draw_modal_input(frame6, "Type something:", "")
    print(f"  Empty input: OK")
    
    print("\nAll tests passed!")
    print("\nVisual test (press any key to advance):")
    print("  Displaying center overlay...")
    cv2.imshow('Test', result1)
    cv2.waitKey(0)
    
    print("  Displaying wrapped text...")
    cv2.imshow('Test', result2)
    cv2.waitKey(0)
    
    print("  Displaying input box...")
    cv2.imshow('Test', result4)
    cv2.waitKey(0)
    
    print("  Displaying input with error...")
    cv2.imshow('Test', result5)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    print("\nSmoke test complete!")


def draw_progress_bar(image, current_count, target, letter, y_position=60):
    """
    Draw progress bar for current letter.
    
    Args:
        image: OpenCV image to draw on
        current_count: Current number of samples
        target: Target number of samples
        letter: Current letter being collected
        y_position: Y coordinate for bar
    
    Returns:
        image: Image with progress bar drawn
    """
    bar_x = 20
    bar_width = 400
    bar_height = 30
    
    # Background
    cv2.rectangle(image, (bar_x, y_position), 
                 (bar_x + bar_width, y_position + bar_height),
                 (80, 80, 80), -1)
    
    # Fill based on progress
    progress = min(1.0, current_count / target) if target > 0 else 0
    fill_width = int(bar_width * progress)
    
    # Color: green if complete, yellow if in progress
    if progress >= 1.0:
        fill_color = (0, 255, 0)
    else:
        fill_color = (0, 255, 255)
    
    if fill_width > 0:
        cv2.rectangle(image, (bar_x, y_position),
                     (bar_x + fill_width, y_position + bar_height),
                     fill_color, -1)
    
    # Border
    cv2.rectangle(image, (bar_x, y_position),
                 (bar_x + bar_width, y_position + bar_height),
                 (200, 200, 200), 2)
    
    # Text: Letter and percentage
    text = f"{letter}: {current_count}/{target} ({progress*100:.1f}%)"
    cv2.putText(image, text, (bar_x + 10, y_position - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return image


def draw_instructions(image, is_paused, position='topright'):
    """
    Draw on-screen instructions.
    
    Args:
        image: OpenCV image to draw on
        is_paused: Whether collection is paused (shows more detail)
        position: 'topright' or 'topleft'
    
    Returns:
        image: Image with instructions drawn
    """
    h, w = image.shape[:2]
    
    if position == 'topright':
        x = w - 250
        y = 20
    else:
        x = 20
        y = 20
    
    # Semi-transparent background
    box_height = 180 if is_paused else 100
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + 230, y + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # Instructions
    font_scale = 0.5
    line_height = 25
    y_text = y + 20
    
    if is_paused:
        instructions = [
            "PAUSED",
            "SPACE - Resume",
            "Letter - Switch letter",
            "SHIFT+D - Discard",
            "SHIFT+S - Save",
            "ESC - Save & Quit"
        ]
    else:
        instructions = [
            "COLLECTING",
            "SPACE - Pause",
            "SHIFT+D - Discard",
            "ESC - Quit"
        ]
    
    for i, text in enumerate(instructions):
        color = (0, 255, 255) if i == 0 else (255, 255, 255)
        cv2.putText(image, text, (x + 10, y_text + i * line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
    
    return image

def draw_text(image, text, position, font_size=20, color=(255, 255, 255), 
              font_path=None):
    """
    Draw text with Unicode support using PIL/Pillow.
    
    Args:
        image: OpenCV image (BGR format)
        text: Text to draw (supports full Unicode)
        position: (x, y) tuple for text position
        font_size: Font size in points
        color: BGR tuple (OpenCV format)
        font_path: Optional path to TTF file (uses bundled DejaVu Sans if None)
    
    Returns:
        image: Modified image with text drawn
    """
    from PIL import Image, ImageDraw, ImageFont
    from pathlib import Path
    
    # Convert BGR to RGB for PIL
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # Load font
    if font_path is None:
        # Try bundled font
        bundled_font = Path(__file__).parent.parent.parent / "assets" / "fonts" / "DejaVuSans.ttf"
        font_path = str(bundled_font) if bundled_font.exists() else None
    
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    
    # Convert BGR to RGB for PIL
    color_rgb = (color[2], color[1], color[0])
    
    draw.text(position, text, font=font, fill=color_rgb)
    
    # Convert back to BGR for OpenCV
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)