"""
Unit tests for common.py draw_landmarks function.

Tests hand skeleton drawing on images.
"""

import pytest
import numpy as np
from src.fingerspell.ui.common import draw_landmarks


class TestDrawLandmarks:
    """Tests for draw_landmarks function."""
    
    def test_empty_landmark_list(self):
        """Should handle empty landmark list without error."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = draw_landmarks(image, [])
        
        # Should return image unchanged
        assert result.shape == (100, 100, 3)
        assert np.array_equal(result, image)
    
    def test_returns_image(self):
        """Should return the modified image."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = [[10, 10] for _ in range(21)]  # 21 dummy landmarks
        
        result = draw_landmarks(image, landmarks)
        
        assert result is image  # Same object
        assert result.shape == (100, 100, 3)
    
    def test_draws_on_image(self):
        """Should modify the image by drawing lines."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Create landmarks in a recognizable pattern
        landmarks = [
            [100, 100],  # 0: wrist
            [100, 90],   # 1
            [90, 80],    # 2: thumb base
            [80, 70],    # 3
            [70, 60],    # 4: thumb tip
            [110, 80],   # 5: index base
            [120, 70],   # 6
            [130, 60],   # 7
            [140, 50],   # 8: index tip
            [110, 85],   # 9: middle base
            [120, 75],   # 10
            [130, 65],   # 11
            [140, 55],   # 12: middle tip
            [110, 90],   # 13: ring base
            [120, 80],   # 14
            [130, 70],   # 15
            [140, 60],   # 16: ring tip
            [105, 95],   # 17: pinky base
            [115, 85],   # 18
            [125, 75],   # 19
            [135, 65],   # 20: pinky tip
        ]
        
        result = draw_landmarks(image, landmarks)
        
        # Image should no longer be all zeros (lines were drawn)
        assert not np.array_equal(result, np.zeros((200, 200, 3), dtype=np.uint8))
        
        # Check that some pixels are non-zero (lines exist)
        assert np.any(result > 0)
    
    def test_correct_number_of_landmarks(self):
        """Should work with exactly 21 landmarks."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        landmarks = [[50 + i*5, 50 + i*5] for i in range(21)]
        
        # Should not raise any errors
        result = draw_landmarks(image, landmarks)
        assert result.shape == (200, 200, 3)
    
    def test_landmark_coordinates_within_bounds(self):
        """Should handle landmarks at image edges."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Landmarks at various positions including edges
        landmarks = [
            [0, 0],      # Top-left corner
            [99, 0],     # Top-right corner
            [0, 99],     # Bottom-left corner
            [99, 99],    # Bottom-right corner
            [50, 50],    # Center
        ] + [[25, 25] for _ in range(16)]  # Fill remaining
        
        # Should not raise any errors
        result = draw_landmarks(image, landmarks)
        assert result.shape == (100, 100, 3)
    
    def test_modifies_image_in_place(self):
        """Should modify the original image, not create a copy."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = [[20 + i*3, 20 + i*3] for i in range(21)]
        
        result = draw_landmarks(image, landmarks)
        
        # Should be the same object
        assert result is image
        
        # Both should show the modifications
        assert np.any(image > 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
