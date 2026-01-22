"""
Model loading and prediction functions.

Handles both static and dynamic letter classification.
"""

import joblib
from pathlib import Path


class ModelManager:
    """Manages loading and prediction for static and dynamic models."""
    
    def __init__(self, static_model_path, dynamic_model_path):
        """
        Initialize model manager.
        
        Args:
            static_model_path: Path to static classifier pickle file
            dynamic_model_path: Path to dynamic classifier pickle file
        """
        self.static_model_path = Path(static_model_path)
        self.dynamic_model_path = Path(dynamic_model_path)
        
        self.static_model = None
        self.dynamic_model = None
        
        self._load_models()
    
    def _load_models(self):
        """Load both models from disk."""
        print(f"Loading static model from {self.static_model_path}")
        self.static_model = joblib.load(self.static_model_path)
        
        print(f"Loading dynamic model from {self.dynamic_model_path}")
        self.dynamic_model = joblib.load(self.dynamic_model_path)
        
        print("Models loaded successfully")
    
    def predict_static(self, normalized_landmarks):
        """
        Predict static letter from normalized landmarks.
        
        Args:
            normalized_landmarks: List of 42 normalized coordinate values
            
        Returns:
            Tuple of (letter, confidence) where confidence is 0-100
        """
        prediction_idx = self.static_model.predict([normalized_landmarks])[0]
        probabilities = self.static_model.predict_proba([normalized_landmarks])[0]
        confidence = max(probabilities) * 100
        
        # Convert index to letter
        letter = chr(prediction_idx + ord('A'))
        
        return letter, confidence
    
    def predict_dynamic(self, current_landmarks, old_landmarks):
        """
        Predict dynamic letter from current and historical landmarks.
        
        Args:
            current_landmarks: List of 42 normalized values (current frame)
            old_landmarks: List of 42 normalized values (5 frames ago)
            
        Returns:
            Tuple of (letter, confidence) where confidence is 0-100
        """
        # Compute delta features
        delta_features = [curr - old for curr, old in zip(current_landmarks, old_landmarks)]
        
        # Concatenate: [current_42, delta_42] = 84 features
        features = current_landmarks + delta_features
        
        prediction_idx = self.dynamic_model.predict([features])[0]
        probabilities = self.dynamic_model.predict_proba([features])[0]
        confidence = max(probabilities) * 100
        
        # Convert index to letter
        letter = chr(prediction_idx + ord('A'))
        
        return letter, confidence