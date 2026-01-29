"""
Unit tests for models module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.fingerspell.core.models import ModelManager


class TestModelManager:
    """Tests for ModelManager class."""
    
    @patch('src.fingerspell.core.models.joblib.load')
    def test_loads_both_models_on_init(self, mock_load):
        """Should load both static and dynamic models during initialization."""
        mock_static = Mock()
        mock_dynamic = Mock()
        mock_load.side_effect = [mock_static, mock_dynamic]
        
        manager = ModelManager('static.pkl', 'dynamic.pkl')
        
        assert mock_load.call_count == 2
        assert manager.static_model == mock_static
        assert manager.dynamic_model == mock_dynamic
    
    @patch('src.fingerspell.core.models.joblib.load')
    def test_predict_static_returns_letter_and_confidence(self, mock_load):
        """Should return letter and confidence for static prediction."""
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = [0]  # Index 0 = 'A'
        mock_model.predict_proba.return_value = [[0.95, 0.03, 0.02]]
        mock_load.return_value = mock_model
        
        manager = ModelManager('static.pkl', 'dynamic.pkl')
        
        # Dummy 42 features
        landmarks = [0.5] * 42
        letter, confidence = manager.predict_static(landmarks)
        
        assert letter == 'A'
        assert confidence == 95.0
    
    @patch('src.fingerspell.core.models.joblib.load')
    def test_predict_static_calls_model_with_correct_features(self, mock_load):
        """Should pass landmarks to model as list in list."""
        mock_model = Mock()
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.95, 0.03, 0.02]]
        mock_load.return_value = mock_model
        
        manager = ModelManager('static.pkl', 'dynamic.pkl')
        landmarks = [0.1, 0.2, 0.3] + [0.0] * 39
        
        manager.predict_static(landmarks)
        
        # Check that predict was called with landmarks wrapped in list
        mock_model.predict.assert_called_once()
        # Get the first argument (should be [landmarks])
        call_args = mock_model.predict.call_args[0][0]
        # The argument should be a list containing the landmarks
        assert call_args == [landmarks] or call_args == landmarks
    
    @patch('src.fingerspell.core.models.joblib.load')
    def test_predict_static_converts_indices_correctly(self, mock_load):
        """Should convert prediction indices to correct letters."""
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        manager = ModelManager('static.pkl', 'dynamic.pkl')
        
        # Test several indices
        test_cases = [
            (0, 'A'),
            (1, 'B'),
            (25, 'Z'),
            (7, 'H'),
        ]
        
        for index, expected_letter in test_cases:
            mock_model.predict.return_value = [index]
            mock_model.predict_proba.return_value = [[0.9] + [0.01] * 25]
            
            letter, _ = manager.predict_static([0.5] * 42)
            assert letter == expected_letter
    
    @patch('src.fingerspell.core.models.joblib.load')
    def test_predict_dynamic_computes_delta_features(self, mock_load):
        """Should compute delta features correctly."""
        mock_model = Mock()
        mock_model.predict.return_value = [7]  # 'H'
        mock_model.predict_proba.return_value = [[0.98] + [0.01] * 4]
        mock_load.return_value = mock_model
        
        manager = ModelManager('static.pkl', 'dynamic.pkl')
        
        current = [1.0] * 42
        old = [0.5] * 42
        
        letter, confidence = manager.predict_dynamic(current, old)
        
        # Check that predict was called
        mock_model.predict.assert_called_once()
        
        # Get the features: call_args[0] is tuple of positional args
        # First positional arg is [features], so we need [0][0] to get features
        call_args_outer = mock_model.predict.call_args[0][0]  # Gets [features]
        call_args_list = call_args_outer[0] if isinstance(call_args_outer[0], list) else call_args_outer
        
        # Verify features: should be 84 values (42 current + 42 delta)
        assert len(call_args_list) == 84
        
        # First 42 should be current landmarks
        assert call_args_list[:42] == current
        
        # Next 42 should be delta (1.0 - 0.5 = 0.5 for all)
        expected_delta = [0.5] * 42
        assert call_args_list[42:] == expected_delta
    
    @patch('src.fingerspell.core.models.joblib.load')
    def test_predict_dynamic_returns_letter_and_confidence(self, mock_load):
        """Should return letter and confidence for dynamic prediction."""
        mock_model = Mock()
        mock_model.predict.return_value = [25]  # Index 25 = 'Z'
        mock_model.predict_proba.return_value = [[0.99, 0.01]]
        mock_load.return_value = mock_model
        
        manager = ModelManager('static.pkl', 'dynamic.pkl')
        
        current = [0.5] * 42
        old = [0.3] * 42
        
        letter, confidence = manager.predict_dynamic(current, old)
        
        assert letter == 'Z'
        assert confidence == 99.0
    
    @patch('src.fingerspell.core.models.joblib.load')
    def test_predict_dynamic_handles_negative_deltas(self, mock_load):
        """Should correctly compute negative delta values."""
        mock_model = Mock()
        mock_model.predict.return_value = [9]  # 'J'
        mock_model.predict_proba.return_value = [[0.85] + [0.01] * 4]
        mock_load.return_value = mock_model
        
        manager = ModelManager('static.pkl', 'dynamic.pkl')
        
        current = [0.2] * 42
        old = [0.8] * 42
        
        letter, _ = manager.predict_dynamic(current, old)
        
        # Get the features
        call_args_outer = mock_model.predict.call_args[0][0]
        call_args_list = call_args_outer[0] if isinstance(call_args_outer[0], list) else call_args_outer
        
        # Verify delta computation (0.2 - 0.8 = -0.6)
        expected_delta = [-0.6] * 42
        
        for i in range(42):
            assert abs(call_args_list[42 + i] - expected_delta[i]) < 1e-10
    
    @patch('src.fingerspell.core.models.joblib.load')
    def test_confidence_converts_probability_to_percentage(self, mock_load):
        """Should convert model probability to percentage."""
        mock_model = Mock()
        mock_model.predict.return_value = [0]
        mock_load.return_value = mock_model
        
        manager = ModelManager('static.pkl', 'dynamic.pkl')
        
        # Test different confidence values
        test_cases = [
            ([0.5, 0.3, 0.2], 50.0),
            ([0.95, 0.03, 0.02], 95.0),
            ([0.333, 0.333, 0.334], 33.4),
        ]
        
        for proba, expected_conf in test_cases:
            mock_model.predict_proba.return_value = [proba]
            _, confidence = manager.predict_static([0.5] * 42)
            
            assert abs(confidence - expected_conf) < 0.1