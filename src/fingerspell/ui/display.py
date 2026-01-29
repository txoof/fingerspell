"""
Refactored UI display functions for fingerspelling recognition.

Optimized for speed with minimal rendering overhead.
Uses draw_text_window for consistency.
"""

import cv2


def get_confidence_color(confidence):
    """
    Get BGR color based on confidence level.
    
    High (>80): Green
    Medium (55-80): Yellow
    Low (<55): Red
    
    Args:
        confidence: Confidence percentage (0-100)
        
    Returns:
        Tuple of (B, G, R) color values
    """
    if confidence > 80:
        return (0, 255, 0)  # Green
    elif confidence >= 55:
        return (0, 255, 255)  # Yellow
    else:
        return (0, 0, 255)  # Red


def draw_prediction_display(frame, letter, confidence):
    """
    Draw main prediction display in upper-left corner.
    
    Shows just the letter with color-coded confidence using draw_text_window.
    Optimized for speed - minimal rendering.
    
    Args:
        frame: Frame to draw on (modified in place)
        letter: Letter to display
        confidence: Confidence percentage
        
    Returns:
        Modified frame
    """
    from src.fingerspell.ui.common import draw_text_window
    
    # Get color based on confidence
    color = get_confidence_color(confidence)
    
    # Draw letter in top-left using draw_text_window
    frame = draw_text_window(
        image=frame,
        text=letter,
        font_size=80,
        first_line_color=color,
        color=color,
        position='topleft',
        margin=20,
        padding=30,
        bg_color=(0, 0, 0),
        bg_alpha=0.7,
        border_color=(100, 100, 100)
    )
    
    return frame


def draw_no_hand_display(frame):
    """
    Draw display when no hand is detected.
    
    Args:
        frame: Frame to draw on (modified in place)
        
    Returns:
        Modified frame
    """
    from src.fingerspell.ui.common import draw_text_window
    
    frame = draw_text_window(
        image=frame,
        text="?",
        font_size=80,
        first_line_color=(100, 100, 100),
        color=(100, 100, 100),
        position='topleft',
        margin=20,
        padding=30,
        bg_color=(0, 0, 0),
        bg_alpha=0.7,
        border_color=(100, 100, 100)
    )
    
    return frame


def draw_debug_display(frame, result, supervisor, static_pred, dynamic_pred, model_manager):
    """
    Draw debug information in separate window using draw_text_window.
    
    Shows:
    - Current prediction details
    - Motion info
    - Both model predictions
    - Confidence thresholds
    - Loaded models
    
    Args:
        frame: Frame to draw on (modified in place)
        result: PredictionResult from supervisor
        supervisor: Supervisor instance
        static_pred: Tuple of (letter, confidence) from static model
        dynamic_pred: Tuple of (letter, confidence) from dynamic model
        model_manager: ModelManager instance
        
    Returns:
        Modified frame
    """
    from src.fingerspell.ui.common import draw_text_window
    
    # Build debug text
    lines = []
    
    # Current prediction
    if result:
        lines.append(f"PREDICTION: {result.letter} ({result.confidence:.0f}%)")
        lines.append(f"Source: {result.source}")
        lines.append(f"Motion: {result.motion:.3f} (threshold: {supervisor.motion_threshold:.3f})")
    else:
        lines.append("PREDICTION: None")
    
    lines.append("")
    
    # Model predictions
    static_letter, static_conf = static_pred
    if static_letter:
        lines.append(f"Static: {static_letter} ({static_conf:.0f}%)")
    else:
        lines.append("Static: No model")
    
    dynamic_letter, dynamic_conf = dynamic_pred
    if dynamic_letter:
        lines.append(f"Dynamic: {dynamic_letter} ({dynamic_conf:.0f}%)")
    else:
        lines.append("Dynamic: Buffer not ready" if model_manager.has_dynamic_model() else "Dynamic: No model")
    
    lines.append("")
    
    # Thresholds
    lines.append(f"Conf Low: {supervisor.confidence_threshold_low:.0f}")
    lines.append(f"Conf High: {supervisor.confidence_threshold_high:.0f}")
    
    lines.append("")
    
    # Loaded models
    lines.append("LOADED MODELS:")
    if model_manager.has_static_model():
        lines.append(f"Static: {model_manager.static_model_path.name}")
    else:
        lines.append("Static: None")
    
    if model_manager.has_dynamic_model():
        lines.append(f"Dynamic: {model_manager.dynamic_model_path.name}")
    else:
        lines.append("Dynamic: None")
    
    lines.append("")
    lines.append("CONTROLS:")
    lines.append("k/j: motion +/- 0.01")
    lines.append("K/J: motion +/- 0.05")
    lines.append("w/s: conf low +/- 5")
    lines.append("W/S: conf high +/- 5")
    
    # Draw in top-right corner
    frame = draw_text_window(
        image=frame,
        text=lines,
        font_size=18,
        first_line_color=(0, 255, 255),
        color=(255, 255, 255),
        position='topright',
        margin=20,
        padding=15,
        bg_color=(0, 0, 0),
        bg_alpha=0.85,
        border_color=(100, 100, 100)
    )
    
    return frame
