"""
Unit tests for display module.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock
from src.fingerspell.ui.display import (
    get_confidence_color,
    draw_semitransparent_box,
    draw_prediction_display,
    draw_no_hand_display,
    draw_motion_bar,
    draw_debug_overlay
)


class TestGetConfidenceColor:
    """Tests for get_confidence_color function."""
    
    def test_returns_green_for_high_confidence(self):
        """Should return green for confidence above high threshold."""
        color = get_confidence_color(90, 70, 85)
        assert color == (0, 255, 0)  # Green in BGR
    
    def test_returns_yellow_for_medium_confidence(self):
        """Should return yellow for confidence between thresholds."""
        color = get_confidence_color(75, 70, 85)
        assert color == (0, 255, 255)  # Yellow in BGR
    
    def test_returns_red_for_low_confidence(self):
        """Should return red for confidence below low threshold."""
        color = get_confidence_color(60, 70, 85)
        assert color == (0, 0, 255)  # Red in BGR
    
    def test_boundary_at_high_threshold(self):
        """Should return green at exactly high threshold."""
        color = get_confidence_color(85, 70, 85)
        assert color == (0, 255, 0)
    
    def test_boundary_at_low_threshold(self):
        """Should return yellow at exactly low threshold."""
        color = get_confidence_color(70, 70, 85)
        assert color == (0, 255, 255)


class TestDrawSemitransparentBox:
    """Tests for draw_semitransparent_box function."""
    
    def test_modifies_frame_in_place(self):
        """Should modify the frame passed to it."""
        # Use white frame so black box creates visible change
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        original_frame = frame.copy()
        
        draw_semitransparent_box(frame, 10, 10, 100, 100)
        
        # Frame should be modified
        assert not np.array_equal(frame, original_frame)
    
    def test_draws_within_bounds(self):
        """Should only modify pixels within specified bounds."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White frame
        
        draw_semitransparent_box(frame, 10, 10, 50, 50, alpha=1.0)
        
        # Pixels outside box should remain white
        assert frame[0, 0].tolist() == [255, 255, 255]
        assert frame[100, 100].tolist() == [255, 255, 255]


class TestDrawPredictionDisplay:
    """Tests for draw_prediction_display function."""
    
    def test_draws_without_error(self):
        """Should draw prediction without raising errors."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Should not raise
        draw_prediction_display(frame, 'A', 95.0, 70, 85)
    
    def test_modifies_frame(self):
        """Should modify the frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original = frame.copy()
        
        draw_prediction_display(frame, 'B', 80.0, 70, 85)
        
        assert not np.array_equal(frame, original)


class TestDrawNoHandDisplay:
    """Tests for draw_no_hand_display function."""
    
    def test_draws_without_error(self):
        """Should draw no-hand indicator without errors."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Should not raise
        draw_no_hand_display(frame)
    
    def test_modifies_frame(self):
        """Should modify the frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original = frame.copy()
        
        draw_no_hand_display(frame)
        
        assert not np.array_equal(frame, original)


class TestDrawMotionBar:
    """Tests for draw_motion_bar function."""
    
    def test_draws_without_error(self):
        """Should draw motion bar without errors."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Should not raise
        draw_motion_bar(frame, 0.15, 0.1)
    
    def test_draws_at_bottom_of_frame(self):
        """Should draw bar near bottom of frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        draw_motion_bar(frame, 0.1, 0.1)
        
        # Check that bottom area is modified
        bottom_area = frame[420:, :, :]
        assert not np.all(bottom_area == 0)
    
    def test_modifies_frame(self):
        """Should modify the frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original = frame.copy()
        
        draw_motion_bar(frame, 0.05, 0.1, max_display=0.3)
        
        assert not np.array_equal(frame, original)


class TestDrawDebugOverlay:
    """Tests for draw_debug_overlay function."""
    
    def test_draws_without_error(self):
        """Should draw debug overlay without errors."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock supervisor
        mock_supervisor = Mock()
        mock_supervisor.motion_threshold = 0.1
        mock_supervisor.confidence_threshold_low = 70.0
        mock_supervisor.confidence_threshold_high = 85.0
        mock_supervisor._calculate_wrist_motion.return_value = 0.05
        
        static_pred = ('A', 90.0)
        dynamic_pred = ('H', 95.0)
        
        # Should not raise
        draw_debug_overlay(frame, mock_supervisor, static_pred, dynamic_pred)
    
    def test_handles_no_dynamic_prediction(self):
        """Should handle case where dynamic prediction is not ready."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        mock_supervisor = Mock()
        mock_supervisor.motion_threshold = 0.1
        mock_supervisor.confidence_threshold_low = 70.0
        mock_supervisor.confidence_threshold_high = 85.0
        mock_supervisor._calculate_wrist_motion.return_value = 0.05
        
        static_pred = ('B', 85.0)
        dynamic_pred = (None, 0.0)
        
        # Should not raise
        draw_debug_overlay(frame, mock_supervisor, static_pred, dynamic_pred)
    
    def test_modifies_frame(self):
        """Should modify the frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original = frame.copy()
        
        mock_supervisor = Mock()
        mock_supervisor.motion_threshold = 0.1
        mock_supervisor.confidence_threshold_low = 70.0
        mock_supervisor.confidence_threshold_high = 85.0
        mock_supervisor._calculate_wrist_motion.return_value = 0.15
        
        draw_debug_overlay(frame, mock_supervisor, ('C', 80.0), ('Z', 90.0))
        
        assert not np.array_equal(frame, original)