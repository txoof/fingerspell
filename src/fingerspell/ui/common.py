"""
Common UI utilities shared across recognition and collection.

Provides modal overlay functions for displaying text and getting user input,
hand skeleton drawing, progress bars, and instructions.
"""

import cv2
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

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
    # Convert BGR to RGB for PIL
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # Load font
    if font_path is None:
        # Try to find bundled font
        # Try multiple paths for flexibility
        possible_paths = [
            Path("../assets/fonts/DejaVuSans.ttf"),  # From notebooks
            Path("assets/fonts/DejaVuSans.ttf"),     # From project root
            Path(__file__).parent.parent.parent / "assets" / "fonts" / "DejaVuSans.ttf",  # From module
        ]
        
        font_path = None
        for p in possible_paths:
            if p.exists():
                font_path = str(p)
                break
    
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except Exception as e:
        print(f"Font loading error: {e}, using default")
        font = ImageFont.load_default()
    
    # Convert BGR to RGB for PIL
    color_rgb = (color[2], color[1], color[0])
    
    draw.text(position, text, font=font, fill=color_rgb)
    
    # Convert back to BGR for OpenCV
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def draw_modal_overlay(image, text, position='center', width_percent=70):
    """
    Draw modal overlay with text using Unicode-capable rendering.
    
    Displays semi-transparent dark overlay covering entire frame with
    text box at specified position. Text wraps automatically.
    
    Args:
        image: Frame to draw on (modified in place)
        text: Text to display (will wrap automatically, use \n for line breaks)
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
    
    # Split text by newlines first
    paragraphs = text.split('\n')
    
    # Wrap each paragraph and measure
    font_size = 24
    bundled_font = Path(__file__).parent.parent.parent / "assets" / "fonts" / "DejaVuSans.ttf"
    try:
        font = ImageFont.truetype(str(bundled_font), font_size)
    except:
        font = ImageFont.load_default()
    
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw_obj = ImageDraw.Draw(pil_image)
    
    # Wrap text to fit box width
    available_width = box_width - 80  # Account for padding
    wrapped_lines = []
    
    for paragraph in paragraphs:
        if not paragraph:  # Empty line
            wrapped_lines.append("")
            continue
        
        words = paragraph.split()
        current_line = ""
        
        for word in words:
            test_line = current_line + word + " "
            bbox = draw_obj.textbbox((0, 0), test_line, font=font)
            line_width = bbox[2] - bbox[0]
            
            if line_width <= available_width:
                current_line = test_line
            else:
                if current_line:
                    wrapped_lines.append(current_line.strip())
                current_line = word + " "
        
        if current_line:
            wrapped_lines.append(current_line.strip())
    
    # Calculate box height based on number of lines
    line_height = 35
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
    
    # Draw wrapped text lines using draw_text
    y_text = box_y + padding
    for line in wrapped_lines:
        if line:  # Skip empty lines
            image = draw_text(image, line, (box_x + padding, y_text),
                             font_size=font_size, color=(255, 255, 255))
        y_text += line_height
    
    return image


def draw_modal_input(image, prompt, current_input, error_msg=None, 
                     position='center', width_percent=70):
    """
    Draw modal overlay for user input using Unicode-capable rendering.
    
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
    
    # Calculate box height
    box_height = 240 if error_msg else 200
    
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
    image = draw_text(image, prompt, (box_x + 20, box_y + 30),
                     font_size=24, color=(255, 255, 255))
    
    # Draw input box
    input_y = box_y + 80
    cv2.rectangle(image, (box_x + 20, input_y),
                 (box_x + box_width - 20, input_y + 50),
                 (255, 255, 255), 2)
    
    # Draw current input text
    display_text = current_input if current_input else ""
    image = draw_text(image, display_text, (box_x + 30, input_y + 15),
                     font_size=28, color=(255, 255, 255))
    
    # Draw error if present
    if error_msg:
        image = draw_text(image, error_msg, (box_x + 20, box_y + 150),
                         font_size=20, color=(0, 0, 255))
    
    # Draw instructions
    instructions = "ENTER=Confirm | BACKSPACE=Delete | ESC=Cancel"
    image = draw_text(image, instructions, (box_x + 20, box_y + box_height - 40),
                     font_size=18, color=(200, 200, 200))
    
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


def draw_progress_bar(image, current_count, target, letter, y_position=60):
    """
    Draw progress bar for current letter with auto-sized text box.
    """
    bar_x = 20
    bar_width = 400
    bar_height = 30
    
    # Calculate progress
    progress = min(1.0, current_count / target) if target > 0 else 0
    
    # Build text
    text = f"{letter}: {current_count}/{target} ({progress*100:.1f}%)"
    
    # Measure text to size the box
    font_size = 24
    bundled_font = Path("../assets/fonts/DejaVuSans.ttf")
    try:
        font = ImageFont.truetype(str(bundled_font), font_size)
    except:
        font = ImageFont.load_default()
    
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw_obj = ImageDraw.Draw(pil_image)
    bbox = draw_obj.textbbox((0, 0), text, font=font)
    text_height = bbox[3] - bbox[1]
    
    # Calculate text box dimensions
    padding_vertical = 10
    text_box_height = text_height + padding_vertical * 2
    
    # Draw semi-transparent background box for text
    overlay = image.copy()
    cv2.rectangle(overlay, (bar_x, y_position - text_box_height - 5), 
                 (bar_x + bar_width, y_position - 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
    
    # Text ABOVE the bar, centered vertically in box
    text_y = y_position - text_box_height - 5 + padding_vertical
    image = draw_text(image, text, (bar_x + 10, text_y),
                     font_size=font_size, color=(255, 255, 255))
    
    # Progress bar background
    cv2.rectangle(image, (bar_x, y_position), 
                 (bar_x + bar_width, y_position + bar_height),
                 (80, 80, 80), -1)
    
    # Fill based on progress
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
    
    return image

def draw_text_window(
    image,
    text,
    font_size=20,
    font_path=None,
    first_line_color=(0, 255, 255),   # cv2 BGR
    color=(255, 255, 255),            # cv2 BGR
    position='topright',              # str anchor OR (x, y)
    margin=20,
    padding=16,
    bg_color=(0, 0, 0),
    bg_alpha=0.85,
    border_color=(100, 100, 100),
    project_root='./',
    wrap=True,
    fill_width=False
):

    project_root = Path(project_root)
    h, w = image.shape[:2]

    if font_path is None:
        font_path = project_root / 'assets/fonts/DejaVuSans.ttf'

    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except Exception:
        font = ImageFont.load_default()

    def _bgr_to_rgb(bgr):
        return (bgr[2], bgr[1], bgr[0])

    def _measure_line(draw, line):
        bbox = draw.textbbox((0, 0), line, font=font)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])

    def _wrap_text(draw, s, max_width):
        words = str(s).split()
        if not words:
            return ['']

        lines = []
        current = ''

        for word in words:
            test = f'{current}{word} '
            tw, _ = _measure_line(draw, test)
            if tw <= max_width:
                current = test
            else:
                if current:
                    lines.append(current.strip())
                    current = f'{word} '
                else:
                    # single word longer than max_width, hard break
                    lines.append(word)
                    current = ''
        if current:
            lines.append(current.strip())

        return lines

    # Convert BGR -> RGB for PIL measurement
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Normalize incoming text into lines
    if isinstance(text, list):
        raw_lines = [str(x) for x in text]
    else:
        raw_lines = str(text).split('\n')

    # Wrapping requires a target width, so do it after we know the box width.
    # If fill_width, we wrap to the full available width inside margins and padding.
    if fill_width:
        max_text_width = max(1, (w - 2 * margin) - 2 * padding)
        lines = []
        if wrap:
            for raw in raw_lines:
                lines.extend(_wrap_text(draw, raw, max_text_width))
        else:
            lines = raw_lines

        box_w = (w - 2 * margin)
    else:
        # measure unwrapped lines, then optionally wrap to that width later if needed
        lines = raw_lines

        max_w = 0
        total_h = 0
        line_sizes = []

        for line in lines:
            tw, th = _measure_line(draw, line)
            line_sizes.append((tw, th))
            max_w = max(max_w, tw)
            total_h += th

        line_spacing = max(int(font_size * 0.25), 4)
        total_h += line_spacing * (len(lines) - 1)

        box_w = max_w + padding * 2

        if wrap:
            max_text_width = max(1, box_w - 2 * padding)
            wrapped = []
            for raw in raw_lines:
                wrapped.extend(_wrap_text(draw, raw, max_text_width))
            lines = wrapped

    # Re measure final lines now that wrapping may have changed them
    line_sizes = []
    max_w = 0
    total_h = 0

    for line in lines:
        tw, th = _measure_line(draw, line)
        line_sizes.append((tw, th))
        max_w = max(max_w, tw)
        total_h += th

    line_spacing = max(int(font_size * 0.25), 4)
    total_h += line_spacing * (len(lines) - 1)

    if not fill_width:
        box_w = max_w + padding * 2

    box_h = total_h + padding * 2

    # Position resolution
    if isinstance(position, (tuple, list)) and len(position) == 2:
        x, y = int(position[0]), int(position[1])

    elif isinstance(position, str):
        pos = position.lower()

        if pos == 'topleft':
            x = margin
            y = margin
        elif pos == 'topright':
            x = w - box_w - margin
            y = margin
        elif pos == 'bottomleft':
            x = margin
            y = h - box_h - margin
        elif pos == 'bottomright':
            x = w - box_w - margin
            y = h - box_h - margin
        else:
            raise ValueError(f'Unknown position: {position}')
    else:
        raise TypeError('position must be a string anchor or (x, y) tuple')

    # Clamp to image bounds
    x = max(0, min(x, w - box_w))
    y = max(0, min(y, h - box_h))

    # Draw background using cv2
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (x, y),
        (x + box_w, y + box_h),
        bg_color,
        -1
    )
    cv2.addWeighted(overlay, bg_alpha, image, 1 - bg_alpha, 0, image)

    cv2.rectangle(
        image,
        (x, y),
        (x + box_w, y + box_h),
        border_color,
        1
    )

    # Draw text using PIL
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    cur_y = y + padding
    for i, line in enumerate(lines):
        bgr = first_line_color if i == 0 else color
        rgb = _bgr_to_rgb(bgr)

        draw.text(
            (x + padding, cur_y),
            line,
            font=font,
            fill=rgb
        )
        cur_y += line_sizes[i][1] + line_spacing

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


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
    print("common.py smoke test")
    print("=" * 50)
    
    # Create test frame
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    test_frame[:] = (50, 100, 150)  # Some color so we can see overlay
    
    # Test draw_text
    print("\nTesting draw_text...")
    frame_text = test_frame.copy()
    result_text = draw_text(frame_text, "Test ÆØÅ ☑☐", (100, 100), font_size=30)
    print(f"  Unicode text: OK")
    
    # Test draw_modal_overlay
    print("\nTesting draw_modal_overlay...")
    
    frame1 = test_frame.copy()
    result1 = draw_modal_overlay(frame1, "Collection complete!", position='center')
    print(f"  Center position: OK")
    
    long_text = "This is a very long message that should wrap across multiple lines automatically."
    frame2 = test_frame.copy()
    result2 = draw_modal_overlay(frame2, long_text, position='top', width_percent=60)
    print(f"  Top position with wrapping: OK")
    
    # Test draw_modal_input
    print("\nTesting draw_modal_input...")
    
    frame4 = test_frame.copy()
    result4 = draw_modal_input(frame4, "Enter alphabet:", "ABCDEFGH")
    print(f"  Basic input: OK")
    
    frame5 = test_frame.copy()
    result5 = draw_modal_input(frame5, "Enter dynamic letters:", "123", 
                               error_msg="Numbers not allowed")
    print(f"  Input with error: OK")
    
    # Test draw_landmarks
    print("\nTesting draw_landmarks...")
    frame6 = test_frame.copy()
    fake_landmarks = [[100 + i*20, 100 + i*10] for i in range(21)]
    result6 = draw_landmarks(frame6, fake_landmarks)
    print(f"  Hand skeleton: OK")
    
    # Test draw_progress_bar
    print("\nTesting draw_progress_bar...")
    frame7 = test_frame.copy()
    result7 = draw_progress_bar(frame7, 1250, 2500, 'A', y_position=100)
    print(f"  Progress bar: OK")
    
    # Test draw_instructions
    print("\nTesting draw_instructions...")
    frame8 = test_frame.copy()
    result8 = draw_instructions(frame8, is_paused=True, position='topright')
    print(f"  Instructions (paused): OK")
    
    frame9 = test_frame.copy()
    result9 = draw_instructions(frame9, is_paused=False, position='topright')
    print(f"  Instructions (running): OK")
    
    print("\nAll tests passed!")
    print("\nVisual test available - check individual frames")