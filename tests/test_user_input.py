"""
Unit tests for user_input module.

Tests alphabet cleaning, label mapping, and validation.
"""

import pytest
# Note: In production, use: from src.fingerspell.ui.user_input import ...
from src.fingerspell.ui.user_input import clean_alphabet, create_label_mapping


class TestCleanAlphabet:
    """Tests for clean_alphabet function."""
    
    def test_removes_spaces(self):
        """Should remove all spaces."""
        result = clean_alphabet("abc def")
        assert result == ['A', 'B', 'C', 'D', 'E', 'F']
    
    def test_converts_to_uppercase(self):
        """Should convert all characters to uppercase."""
        result = clean_alphabet("abc")
        assert result == ['A', 'B', 'C']
        
        result = clean_alphabet("Hello World")
        assert result == ['H', 'E', 'L', 'O', 'W', 'R', 'D']
    
    def test_removes_duplicates(self):
        """Should remove duplicate characters while preserving order."""
        result = clean_alphabet("aabbcc")
        assert result == ['A', 'B', 'C']
        
        result = clean_alphabet("zzz aaa")
        assert result == ['Z', 'A']
    
    def test_preserves_order(self):
        """Should preserve the order characters were entered."""
        result = clean_alphabet("zyx")
        assert result == ['Z', 'Y', 'X']
        
        result = clean_alphabet("bac")
        assert result == ['B', 'A', 'C']
    
    def test_handles_non_ascii(self):
        """Should handle non-ASCII characters like Norwegian letters."""
        result = clean_alphabet("ABCÆØÅ")
        assert result == ['A', 'B', 'C', 'Æ', 'Ø', 'Å']
        
        result = clean_alphabet("æøå")
        assert result == ['Æ', 'Ø', 'Å']
    
    def test_handles_empty_string(self):
        """Should handle empty input."""
        result = clean_alphabet("")
        assert result == []
    
    def test_handles_only_spaces(self):
        """Should handle input with only spaces."""
        result = clean_alphabet("   ")
        assert result == []
    
    def test_complex_input(self):
        """Should handle complex mixed input."""
        result = clean_alphabet("a B c   A b C")
        assert result == ['A', 'B', 'C']


class TestCreateLabelMapping:
    """Tests for create_label_mapping function."""
    
    def test_sorts_by_unicode(self):
        """Should sort characters by unicode codepoint."""
        result = create_label_mapping(['Z', 'Y', 'X'])
        assert result == {'X': 0, 'Y': 1, 'Z': 2}
        
        result = create_label_mapping(['C', 'A', 'B'])
        assert result == {'A': 0, 'B': 1, 'C': 2}
    
    def test_sequential_indices(self):
        """Should assign sequential indices starting from 0."""
        result = create_label_mapping(['A', 'B', 'C'])
        assert result == {'A': 0, 'B': 1, 'C': 2}
    
    def test_handles_non_ascii(self):
        """Should correctly sort non-ASCII characters by unicode."""
        # Unicode values: A=65, B=66, Å=197, Æ=198, Ø=216
        result = create_label_mapping(['Æ', 'A', 'Ø', 'B', 'Å'])
        assert result == {'A': 0, 'B': 1, 'Å': 2, 'Æ': 3, 'Ø': 4}
    
    def test_single_character(self):
        """Should handle single character input."""
        result = create_label_mapping(['A'])
        assert result == {'A': 0}
    
    def test_empty_list(self):
        """Should handle empty list."""
        result = create_label_mapping([])
        assert result == {}
    
    def test_full_alphabet(self):
        """Should handle full English alphabet."""
        alphabet = list("ZYXWVUTSRQPONMLKJIHGFEDCBA")
        result = create_label_mapping(alphabet)
        
        # Check it has 26 entries
        assert len(result) == 26
        
        # Check A is 0 and Z is 25
        assert result['A'] == 0
        assert result['Z'] == 25
        
        # Check sequential
        for i, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            assert result[char] == i


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_full_workflow(self):
        """Test complete workflow: clean then map."""
        raw = "zzz abc AAA"
        cleaned = clean_alphabet(raw)
        mapping = create_label_mapping(cleaned)
        
        assert cleaned == ['Z', 'A', 'B', 'C']
        assert mapping == {'A': 0, 'B': 1, 'C': 2, 'Z': 3}
    
    def test_norwegian_workflow(self):
        """Test workflow with Norwegian alphabet."""
        raw = "ÆØÅABC"
        cleaned = clean_alphabet(raw)
        mapping = create_label_mapping(cleaned)
        
        assert cleaned == ['Æ', 'Ø', 'Å', 'A', 'B', 'C']
        # Sorted by unicode: A, B, C, Å, Æ, Ø
        assert mapping == {'A': 0, 'B': 1, 'C': 2, 'Å': 3, 'Æ': 4, 'Ø': 5}
    
    def test_with_duplicates_and_spaces(self):
        """Test workflow with messy input."""
        raw = "a b c   a a a   b b"
        cleaned = clean_alphabet(raw)
        mapping = create_label_mapping(cleaned)
        
        assert cleaned == ['A', 'B', 'C']
        assert mapping == {'A': 0, 'B': 1, 'C': 2}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
