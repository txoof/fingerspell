"""
Unit tests for landmarks processing functions.
"""

import pytest
from src.fingerspell.core.landmarks import calc_landmark_list, pre_process_landmark


class TestCalcLandmarkList:
    """Tests for calc_landmark_list function."""
    
    def test_extracts_21_landmarks(self):
        """Should extract 21 coordinate pairs from MediaPipe landmarks."""
        # Create mock image
        import numpy as np
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create mock MediaPipe landmarks
        class MockLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.z = 0.0
        
        class MockLandmarks:
            def __init__(self):
                # 21 landmarks in normalized coordinates (0-1)
                self.landmark = [MockLandmark(0.5, 0.5) for _ in range(21)]
        
        mock_landmarks = MockLandmarks()
        result = calc_landmark_list(mock_image, mock_landmarks)
        
        assert len(result) == 21
        assert all(len(point) == 2 for point in result)
    
    def test_converts_normalized_to_pixel_coords(self):
        """Should convert MediaPipe normalized coords to pixel coordinates."""
        import numpy as np
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        class MockLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.z = 0.0
        
        class MockLandmarks:
            def __init__(self):
                # Center of image: (0.5, 0.5) should become (320, 240)
                self.landmark = [MockLandmark(0.5, 0.5) for _ in range(21)]
        
        mock_landmarks = MockLandmarks()
        result = calc_landmark_list(mock_image, mock_landmarks)
        
        # Check first landmark
        assert result[0][0] == 320  # 0.5 * 640
        assert result[0][1] == 240  # 0.5 * 480
    
    def test_clamps_coordinates_to_image_bounds(self):
        """Should clamp coordinates to image dimensions."""
        import numpy as np
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        class MockLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.z = 0.0
        
        class MockLandmarks:
            def __init__(self):
                # Values > 1.0 should be clamped
                self.landmark = [MockLandmark(1.5, 1.5) for _ in range(21)]
        
        mock_landmarks = MockLandmarks()
        result = calc_landmark_list(mock_image, mock_landmarks)
        
        # Should clamp to max values
        assert result[0][0] == 639  # width - 1
        assert result[0][1] == 479  # height - 1


class TestPreProcessLandmark:
    """Tests for pre_process_landmark function."""
    
    def test_returns_42_values(self):
        """Should return 42 normalized values (21 landmarks * 2 coords)."""
        # Simple landmark list: wrist at origin, one other point
        landmark_list = [[100, 200]] + [[110 + i, 210 + i] for i in range(20)]
        
        result = pre_process_landmark(landmark_list)
        
        assert len(result) == 42
    
    def test_wrist_becomes_origin(self):
        """First landmark (wrist) should be at (0, 0) after normalization."""
        landmark_list = [[100, 200]] + [[110 + i, 210 + i] for i in range(20)]
        
        result = pre_process_landmark(landmark_list)
        
        # First two values should be 0 (wrist x, wrist y)
        assert result[0] == 0.0
        assert result[1] == 0.0
    
    def test_values_in_range(self):
        """All normalized values should be in range [-1, 1]."""
        landmark_list = [[100, 200]] + [[100 + i*10, 200 + i*10] for i in range(20)]
        
        result = pre_process_landmark(landmark_list)
        
        assert all(-1.0 <= val <= 1.0 for val in result)
    
    def test_max_absolute_value_is_one(self):
        """At least one value should have absolute value of 1.0."""
        landmark_list = [[100, 200]] + [[100 + i*10, 200 + i*10] for i in range(20)]
        
        result = pre_process_landmark(landmark_list)
        
        max_abs = max(abs(val) for val in result)
        assert abs(max_abs - 1.0) < 1e-10  # Close to 1.0
    
    def test_preserves_relative_positions(self):
        """Scaling should preserve relative relationships between points."""
        # Create two landmark sets with same shape but different scales
        landmark_list_1 = [[0, 0], [10, 0], [0, 10]]
        landmark_list_2 = [[0, 0], [20, 0], [0, 20]]
        
        # Pad to 21 landmarks
        landmark_list_1.extend([[0, 0] for _ in range(18)])
        landmark_list_2.extend([[0, 0] for _ in range(18)])
        
        result_1 = pre_process_landmark(landmark_list_1)
        result_2 = pre_process_landmark(landmark_list_2)
        
        # Should produce same normalized result (scale-invariant)
        assert all(abs(a - b) < 1e-10 for a, b in zip(result_1, result_2))
    
    def test_handles_zero_max_value(self):
        """Should handle edge case where all points are at wrist."""
        # All landmarks at same position
        landmark_list = [[100, 200] for _ in range(21)]
        
        result = pre_process_landmark(landmark_list)
        
        # Should return all zeros without crashing
        assert len(result) == 42
        assert all(val == 0.0 for val in result)