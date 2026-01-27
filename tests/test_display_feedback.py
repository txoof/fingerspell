"""
Unit tests for display feedback functions.

Tests progress bar, instructions, letter status, and Unicode text rendering.
"""

import pytest
import numpy as np
from src.fingerspell.ui.common import draw_progress_bar, draw_instructions, draw_text
from src.fingerspell.collection.data_management import draw_letter_status


class TestDrawText:
    """Tests for draw_text function with Unicode support."""
    
    def test_basic_ascii(self):
        """Should render basic ASCII text."""
        image = np.zeros((100, 300, 3), dtype=np.uint8)
        result = draw_text(image, "Hello World", (10, 50), font_size=20)
        
        # Image should be modified (not all zeros)
        assert not np.array_equal(result, np.zeros((100, 300, 3), dtype=np.uint8))
        assert result.shape == (100, 300, 3)
    
    def test_unicode_characters(self):
        """Should render Unicode characters (Norwegian, German, Polish)."""
        image = np.zeros((100, 300, 3), dtype=np.uint8)
        
        # Test various European characters
        texts = ["ÆØÅ", "ÄÖÜß", "ĄĆĘŁŃÓŚŹŻ", "ÇĞİÖŞÜ"]
        
        for text in texts:
            result = draw_text(image, text, (10, 50), font_size=20)
            # Should not raise error and should modify image
            assert result.shape == (100, 300, 3)
            assert not np.array_equal(result, image)
    
    def test_ballot_box_characters(self):
        """Should render ballot box Unicode characters."""
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = draw_text(image, "☐☑", (10, 50), font_size=20)
        
        # Should render without error
        assert result.shape == (100, 200, 3)
        assert not np.array_equal(result, np.zeros((100, 200, 3), dtype=np.uint8))
    
    def test_color_parameter(self):
        """Should accept BGR color tuple."""
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        
        # Should not raise error with different colors
        result = draw_text(image, "Test", (10, 50), font_size=20, color=(0, 255, 0))
        assert result.shape == (100, 200, 3)
    
    def test_font_size_parameter(self):
        """Should accept different font sizes."""
        image = np.zeros((200, 400, 3), dtype=np.uint8)
        
        # Different font sizes should work
        for size in [12, 20, 30]:
            result = draw_text(image, "Test", (10, 50), font_size=size)
            assert result.shape == (200, 400, 3)
    
    def test_returns_modified_image(self):
        """Should return the modified image."""
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = draw_text(image, "Test", (10, 50), font_size=20)
        
        # Should return an image
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8


class TestDrawProgressBar:
    """Tests for draw_progress_bar function."""
    
    def test_draws_progress_bar(self):
        """Should draw a progress bar on image."""
        image = np.zeros((200, 500, 3), dtype=np.uint8)
        result = draw_progress_bar(image, 1250, 2500, 'A', y_position=60)
        
        # Image should be modified
        assert not np.array_equal(result, np.zeros((200, 500, 3), dtype=np.uint8))
        assert result.shape == (200, 500, 3)
    
    def test_zero_progress(self):
        """Should handle zero progress."""
        image = np.zeros((200, 500, 3), dtype=np.uint8)
        result = draw_progress_bar(image, 0, 2500, 'B', y_position=60)
        
        assert result.shape == (200, 500, 3)
    
    def test_complete_progress(self):
        """Should handle 100% completion."""
        image = np.zeros((200, 500, 3), dtype=np.uint8)
        result = draw_progress_bar(image, 2500, 2500, 'C', y_position=60)
        
        assert result.shape == (200, 500, 3)
    
    def test_over_target(self):
        """Should handle collection exceeding target."""
        image = np.zeros((200, 500, 3), dtype=np.uint8)
        result = draw_progress_bar(image, 3000, 2500, 'D', y_position=60)
        
        assert result.shape == (200, 500, 3)
    
    def test_unicode_letter(self):
        """Should handle Unicode letters in progress bar."""
        image = np.zeros((200, 500, 3), dtype=np.uint8)
        result = draw_progress_bar(image, 1000, 2500, 'Ø', y_position=60)
        
        assert result.shape == (200, 500, 3)


class TestDrawInstructions:
    """Tests for draw_instructions function."""
    
    def test_paused_instructions(self):
        """Should draw paused instructions."""
        image = np.zeros((400, 800, 3), dtype=np.uint8)
        result = draw_instructions(image, is_paused=True, position='topright')
        
        # Image should be modified
        assert not np.array_equal(result, np.zeros((400, 800, 3), dtype=np.uint8))
        assert result.shape == (400, 800, 3)
    
    def test_running_instructions(self):
        """Should draw running (collecting) instructions."""
        image = np.zeros((400, 800, 3), dtype=np.uint8)
        result = draw_instructions(image, is_paused=False, position='topright')
        
        assert result.shape == (400, 800, 3)
    
    def test_position_topleft(self):
        """Should handle topleft position."""
        image = np.zeros((400, 800, 3), dtype=np.uint8)
        result = draw_instructions(image, is_paused=True, position='topleft')
        
        assert result.shape == (400, 800, 3)
    
    def test_position_topright(self):
        """Should handle topright position."""
        image = np.zeros((400, 800, 3), dtype=np.uint8)
        result = draw_instructions(image, is_paused=False, position='topright')
        
        assert result.shape == (400, 800, 3)


class TestDrawLetterStatus:
    """Tests for draw_letter_status function."""
    
    def test_draws_letter_status(self):
        """Should draw letter status with checkboxes."""
        image = np.zeros((300, 800, 3), dtype=np.uint8)
        
        alphabet = ['A', 'B', 'C']
        collected_per_letter = {'A': 2500, 'B': 1200, 'C': 0}
        targets = {'A': 2500, 'B': 2500, 'C': 2500}
        dynamic_letters = set()
        
        result = draw_letter_status(image, alphabet, collected_per_letter, 
                                    targets, dynamic_letters)
        
        # Image should be modified
        assert not np.array_equal(result, np.zeros((300, 800, 3), dtype=np.uint8))
        assert result.shape == (300, 800, 3)
    
    def test_unicode_letters(self):
        """Should handle Unicode letters in status."""
        image = np.zeros((300, 800, 3), dtype=np.uint8)
        
        alphabet = ['A', 'Æ', 'Ø', 'Å']
        collected_per_letter = {'A': 2500, 'Æ': 2500, 'Ø': 1000, 'Å': 0}
        targets = {'A': 2500, 'Æ': 2500, 'Ø': 2500, 'Å': 2500}
        dynamic_letters = set()
        
        result = draw_letter_status(image, alphabet, collected_per_letter, 
                                    targets, dynamic_letters)
        
        assert result.shape == (300, 800, 3)
    
    def test_all_complete(self):
        """Should handle all letters complete."""
        image = np.zeros((300, 800, 3), dtype=np.uint8)
        
        alphabet = ['A', 'B']
        collected_per_letter = {'A': 2500, 'B': 2500}
        targets = {'A': 2500, 'B': 2500}
        dynamic_letters = set()
        
        result = draw_letter_status(image, alphabet, collected_per_letter, 
                                    targets, dynamic_letters)
        
        assert result.shape == (300, 800, 3)
    
    def test_none_complete(self):
        """Should handle no letters complete."""
        image = np.zeros((300, 800, 3), dtype=np.uint8)
        
        alphabet = ['A', 'B', 'C']
        collected_per_letter = {'A': 0, 'B': 0, 'C': 0}
        targets = {'A': 2500, 'B': 2500, 'C': 2500}
        dynamic_letters = set()
        
        result = draw_letter_status(image, alphabet, collected_per_letter, 
                                    targets, dynamic_letters)
        
        assert result.shape == (300, 800, 3)
    
    def test_mixed_static_dynamic(self):
        """Should handle mix of static and dynamic letters."""
        image = np.zeros((300, 800, 3), dtype=np.uint8)
        
        alphabet = ['A', 'B', 'H', 'J']
        collected_per_letter = {'A': 2500, 'B': 1200, 'H': 3500, 'J': 1000}
        targets = {'A': 2500, 'B': 2500, 'H': 3500, 'J': 3500}
        dynamic_letters = {'H', 'J'}
        
        result = draw_letter_status(image, alphabet, collected_per_letter, 
                                    targets, dynamic_letters)
        
        assert result.shape == (300, 800, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])