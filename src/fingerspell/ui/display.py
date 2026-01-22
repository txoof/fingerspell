"""
UI display functions for fingerspelling recognition.

All OpenCV drawing operations for predictions, debug overlays, and visual feedback.
"""

import cv2


def get_confidence_color(confidence, low_threshold, high_threshold):
    """
    Get BGR color based on confidence level.
    
    Args:
        confidence: Confidence percentage (0-100)
        low_threshold: Threshold for red/yellow boundary
        high_threshold: Threshold for yellow/green boundary
        
    Returns:
        Tuple of (B, G, R) color values
    """
    if confidence >= high_threshold:
        return (0, 255, 0)  # Green
    elif confidence >= low_threshold:
        return (0, 255, 255)  # Yellow
    else:
        return (0, 0, 255)  # Red


def draw_semitransparent_box(frame, x, y, width, height, alpha=0.7):
    """
    Draw a semi-transparent black box.
    
    Args:
        frame: Frame to draw on (modified in place)
        x, y: Top-left corner coordinates
        width, height: Box dimensions
        alpha: Opacity (0=transparent, 1=opaque)
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_prediction_display(frame, letter, confidence, low_threshold, high_threshold, debug=False):
    """
    Draw main prediction display in upper-left corner.
    
    Shows large letter with color-coded confidence.
    
    Args:
        frame: Frame to draw on (modified in place)
        letter: Letter to display (or '?' if no hand)
        confidence: Confidence percentage
        low_threshold: Low confidence threshold
        high_threshold: High confidence threshold
    """
    # Semi-transparent background box
    box_x, box_y = 20, 20
    box_width, box_height = 200, 150
    draw_semitransparent_box(frame, box_x, box_y, box_width, box_height, alpha=0.6)
    
    # Get color based on confidence
    color = get_confidence_color(confidence, low_threshold, high_threshold)
    
    # Draw large letter
    cv2.putText(frame, letter,
               (box_x + 50, box_y + 100),
               cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 6, cv2.LINE_AA)
    
    # Draw confidence percentage below
    if debug:
        conf_text = f"{confidence:.0f}%"
        cv2.putText(frame, conf_text,
                (box_x + 50, box_y + 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def draw_no_hand_display(frame):
    """
    Draw display when no hand is detected.
    
    Args:
        frame: Frame to draw on (modified in place)
    """
    draw_prediction_display(frame, "?", 0, 70, 85)


def draw_motion_bar(frame, motion, threshold, max_display=0.3):
    """
    Draw motion indicator bar at bottom of frame.
    
    Args:
        frame: Frame to draw on (modified in place)
        motion: Current motion value
        threshold: Motion threshold (drawn as red line)
        max_display: Maximum motion value for bar scaling
    """
    h = frame.shape[0]
    
    bar_x = 20
    bar_y = h - 60
    bar_width = 300
    bar_height = 30
    
    # Background
    cv2.rectangle(frame, (bar_x, bar_y),
                 (bar_x + bar_width, bar_y + bar_height),
                 (80, 80, 80), -1)
    
    # Fill based on motion
    fill_ratio = min(1.0, motion / max_display)
    fill_width = int(bar_width * fill_ratio)
    
    # Color: green if above threshold, gray if below
    if motion >= threshold:
        fill_color = (0, 255, 0)
    else:
        fill_color = (100, 100, 100)
    
    if fill_width > 0:
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + fill_width, bar_y + bar_height),
                     fill_color, -1)
    
    # Threshold line (red)
    threshold_x = bar_x + int(bar_width * (threshold / max_display))
    cv2.line(frame, (threshold_x, bar_y),
            (threshold_x, bar_y + bar_height),
            (0, 0, 255), 2)
    
    # Border
    cv2.rectangle(frame, (bar_x, bar_y),
                 (bar_x + bar_width, bar_y + bar_height),
                 (200, 200, 200), 2)
    
    # Label
    cv2.putText(frame, "Motion", (bar_x, bar_y - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def draw_debug_overlay(frame, supervisor, static_pred, dynamic_pred):
    """
    Draw debug information overlay.
    
    Args:
        frame: Frame to draw on (modified in place)
        supervisor: Supervisor instance (for thresholds and motion)
        static_pred: Tuple of (letter, confidence) from static model
        dynamic_pred: Tuple of (letter, confidence) from dynamic model or (None, 0)
    """
    panel_x = 20
    panel_y = 230
    panel_width = 500
    panel_height = 150
    
    # Semi-transparent background
    draw_semitransparent_box(frame, panel_x, panel_y, panel_width, panel_height, alpha=0.7)
    
    # Get motion from supervisor
    motion = supervisor._calculate_wrist_motion()
    
    y_pos = panel_y + 25
    
    # Motion and threshold
    cv2.putText(frame,
               f"Motion: {motion:.3f} | Threshold: {supervisor.motion_threshold:.3f}",
               (panel_x + 10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    y_pos += 30
    
    # Static prediction
    static_letter, static_conf = static_pred
    cv2.putText(frame,
               f"Static:  {static_letter} ({static_conf:.0f}%)",
               (panel_x + 10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    y_pos += 30
    
    # Dynamic prediction
    dynamic_letter, dynamic_conf = dynamic_pred
    if dynamic_letter is not None:
        cv2.putText(frame,
                   f"Dynamic: {dynamic_letter} ({dynamic_conf:.0f}%)",
                   (panel_x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame,
                   "Dynamic: waiting for buffer...",
                   (panel_x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1, cv2.LINE_AA)
    y_pos += 30
    
    # Confidence thresholds
    cv2.putText(frame,
               f"Confidence: Low={supervisor.confidence_threshold_low:.0f} High={supervisor.confidence_threshold_high:.0f}",
               (panel_x + 10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    y_pos += 25
    
    # Keybinding hints
    cv2.putText(frame,
               "k/j: motion +/- 0.01 | K/J: +/- 0.05 | w/s: conf low +/- 5 | W/S: conf high +/- 5",
               (panel_x + 10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)