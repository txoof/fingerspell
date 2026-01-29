"""
Unit tests for Supervisor class.
"""

import pytest
from unittest.mock import Mock
from src.fingerspell.core.supervisor import Supervisor, PredictionResult


class TestSupervisor:
    """Tests for Supervisor class."""
    
    def test_initializes_with_correct_defaults(self):
        """Should initialize with default buffer sizes and thresholds."""
        mock_model = Mock()
        supervisor = Supervisor(mock_model)
        
        assert supervisor.rolling_window_size == 5
        assert supervisor.wrist_motion_window == 10
        assert supervisor.motion_threshold == 0.1
        assert supervisor.confidence_threshold_low == 70.0
        assert supervisor.confidence_threshold_high == 85.0
    
    def test_initializes_with_custom_sizes(self):
        """Should accept custom buffer sizes."""
        mock_model = Mock()
        supervisor = Supervisor(mock_model, rolling_window_size=10, wrist_motion_window=20)
        
        assert supervisor.rolling_window_size == 10
        assert supervisor.wrist_motion_window == 20
    
    def test_buffers_start_empty(self):
        """Buffers should be empty on initialization."""
        mock_model = Mock()
        supervisor = Supervisor(mock_model)
        
        assert len(supervisor.landmark_buffer) == 0
        assert len(supervisor.wrist_buffer) == 0
    
    def test_process_frame_adds_to_buffers(self):
        """Should add landmarks and wrist position to buffers."""
        mock_model = Mock()
        mock_model.predict_static.return_value = ('A', 95.0)
        
        supervisor = Supervisor(mock_model)
        
        landmarks = [0.5] * 42
        wrist = [0.1, 0.2, 0.3]
        
        supervisor.process_frame(landmarks, wrist)
        
        assert len(supervisor.landmark_buffer) == 1
        assert len(supervisor.wrist_buffer) == 1
    
    def test_process_frame_returns_static_prediction_initially(self):
        """Should return static prediction before dynamic buffer fills."""
        mock_model = Mock()
        mock_model.predict_static.return_value = ('B', 90.0)
        
        supervisor = Supervisor(mock_model, rolling_window_size=5)
        
        landmarks = [0.5] * 42
        wrist = [0.1, 0.2, 0.3]
        
        # Add first frame (buffer not full yet)
        result = supervisor.process_frame(landmarks, wrist)
        
        assert result.letter == 'B'
        assert result.confidence == 90.0
        assert result.source == 'static'
    
    def test_process_frame_routes_to_dynamic_with_motion(self):
        """Should route to dynamic model when motion exceeds threshold."""
        mock_model = Mock()
        mock_model.predict_static.return_value = ('A', 85.0)
        mock_model.predict_dynamic.return_value = ('H', 95.0)
        
        supervisor = Supervisor(mock_model, rolling_window_size=5)
        supervisor.motion_threshold = 0.1
        
        # Fill buffer with moving wrist positions
        for i in range(5):
            landmarks = [0.5] * 42
            wrist = [0.1 * i, 0.2 * i, 0.0]  # Moving wrist
            result = supervisor.process_frame(landmarks, wrist)
        
        # Last result should be dynamic (motion should exceed threshold)
        assert result.source == 'dynamic'
        assert result.letter == 'H'
    
    def test_calculate_wrist_motion_returns_zero_for_empty_buffer(self):
        """Should return 0 motion for empty buffer."""
        mock_model = Mock()
        supervisor = Supervisor(mock_model)
        
        motion = supervisor._calculate_wrist_motion()
        assert motion == 0.0
    
    def test_calculate_wrist_motion_returns_zero_for_single_position(self):
        """Should return 0 motion for single wrist position."""
        mock_model = Mock()
        supervisor = Supervisor(mock_model)
        supervisor.wrist_buffer.append([0.5, 0.5, 0.0])
        
        motion = supervisor._calculate_wrist_motion()
        assert motion == 0.0
    
    def test_calculate_wrist_motion_computes_distance(self):
        """Should compute cumulative distance for moving wrist."""
        mock_model = Mock()
        supervisor = Supervisor(mock_model)
        
        # Add positions that move in a line
        supervisor.wrist_buffer.append([0.0, 0.0, 0.0])
        supervisor.wrist_buffer.append([0.1, 0.0, 0.0])
        supervisor.wrist_buffer.append([0.2, 0.0, 0.0])
        
        motion = supervisor._calculate_wrist_motion()
        
        # Should be ~0.2 (0.1 + 0.1)
        assert abs(motion - 0.2) < 1e-10
    
    def test_clear_buffers_empties_both_buffers(self):
        """Should clear both landmark and wrist buffers."""
        mock_model = Mock()
        mock_model.predict_static.return_value = ('A', 90.0)
        
        supervisor = Supervisor(mock_model)
        
        # Fill buffers
        for i in range(3):
            landmarks = [0.5] * 42
            wrist = [0.1, 0.2, 0.3]
            supervisor.process_frame(landmarks, wrist)
        
        assert len(supervisor.landmark_buffer) > 0
        assert len(supervisor.wrist_buffer) > 0
        
        # Clear
        supervisor.clear_buffers()
        
        assert len(supervisor.landmark_buffer) == 0
        assert len(supervisor.wrist_buffer) == 0
    
    def test_adjust_motion_threshold_increases(self):
        """Should increase motion threshold by delta."""
        mock_model = Mock()
        supervisor = Supervisor(mock_model)
        
        initial = supervisor.motion_threshold
        supervisor.adjust_motion_threshold(0.05)
        
        assert supervisor.motion_threshold == initial + 0.05
    
    def test_adjust_motion_threshold_decreases(self):
        """Should decrease motion threshold by negative delta."""
        mock_model = Mock()
        supervisor = Supervisor(mock_model)
        
        initial = supervisor.motion_threshold
        supervisor.adjust_motion_threshold(-0.05)
        
        assert supervisor.motion_threshold == initial - 0.05
    
    def test_adjust_motion_threshold_cannot_go_negative(self):
        """Should not allow negative motion threshold."""
        mock_model = Mock()
        supervisor = Supervisor(mock_model)
        supervisor.motion_threshold = 0.01
        
        supervisor.adjust_motion_threshold(-0.1)
        
        assert supervisor.motion_threshold == 0.0
    
    def test_adjust_confidence_thresholds_low(self):
        """Should adjust low confidence threshold."""
        mock_model = Mock()
        supervisor = Supervisor(mock_model)
        
        initial = supervisor.confidence_threshold_low
        supervisor.adjust_confidence_thresholds(low_delta=5)
        
        assert supervisor.confidence_threshold_low == initial + 5
    
    def test_adjust_confidence_thresholds_high(self):
        """Should adjust high confidence threshold."""
        mock_model = Mock()
        supervisor = Supervisor(mock_model)
        
        initial = supervisor.confidence_threshold_high
        supervisor.adjust_confidence_thresholds(high_delta=5)
        
        assert supervisor.confidence_threshold_high == initial + 5
    
    def test_adjust_confidence_thresholds_maintains_order(self):
        """Should ensure high threshold stays >= low threshold."""
        mock_model = Mock()
        supervisor = Supervisor(mock_model)
        
        supervisor.confidence_threshold_low = 70
        supervisor.confidence_threshold_high = 85
        
        # Try to make high lower than low
        supervisor.adjust_confidence_thresholds(high_delta=-20)
        
        # High should be clamped to low
        assert supervisor.confidence_threshold_high >= supervisor.confidence_threshold_low
    
    def test_adjust_confidence_thresholds_clamps_to_range(self):
        """Should clamp confidence thresholds to 0-100 range."""
        mock_model = Mock()
        supervisor = Supervisor(mock_model)
        
        # Try to go above 100
        supervisor.confidence_threshold_high = 95
        supervisor.adjust_confidence_thresholds(high_delta=10)
        assert supervisor.confidence_threshold_high == 100.0
        
        # Try to go below 0
        supervisor.confidence_threshold_low = 5
        supervisor.adjust_confidence_thresholds(low_delta=-10)
        assert supervisor.confidence_threshold_low == 0.0