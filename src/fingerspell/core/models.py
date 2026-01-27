"""
Model loading and prediction functions.

Handles both static and dynamic letter classification.
"""

import joblib
from pathlib import Path


class ModelManager:
    """Manages loading and prediction for static and dynamic models."""
    
    def __init__(self, static_model_path=None, dynamic_model_path=None, 
                 static_labels_path=None, dynamic_labels_path=None):
        """
        Initialize model manager.
        
        Args:
            static_model_path: Path to static classifier pickle file (optional)
            dynamic_model_path: Path to dynamic classifier pickle file (optional)
            static_labels_path: Path to static labels CSV (optional)
            dynamic_labels_path: Path to dynamic labels CSV (optional)
        """
        self.static_model_path = Path(static_model_path) if static_model_path else None
        self.dynamic_model_path = Path(dynamic_model_path) if dynamic_model_path else None
        self.static_labels_path = Path(static_labels_path) if static_labels_path else None
        self.dynamic_labels_path = Path(dynamic_labels_path) if dynamic_labels_path else None
        
        self.static_model = None
        self.dynamic_model = None
        self.static_labels = []
        self.dynamic_labels = []
        
        self._load_models()
        self._load_labels()
    
    def _load_models(self):
        """Load models from disk if paths provided."""
        if self.static_model_path and self.static_model_path.exists():
            self.static_model = joblib.load(self.static_model_path)
     
        
        if self.dynamic_model_path and self.dynamic_model_path.exists():
            self.dynamic_model = joblib.load(self.dynamic_model_path)
  
        
    def _load_labels(self):
        """Load label mappings from CSV files if provided."""
        import csv
        
        if self.static_labels_path and self.static_labels_path.exists():
            with open(self.static_labels_path, 'r', encoding='utf-8-sig') as f:
                self.static_labels = [row[0] for row in csv.reader(f)]
        
        if self.dynamic_labels_path and self.dynamic_labels_path.exists():
            with open(self.dynamic_labels_path, 'r', encoding='utf-8-sig') as f:
                self.dynamic_labels = [row[0] for row in csv.reader(f)]
    
    def predict_static(self, normalized_landmarks):
        """
        Predict static letter from normalized landmarks.
        
        Args:
            normalized_landmarks: List of 42 normalized coordinate values
            
        Returns:
            Tuple of (letter, confidence) where confidence is 0-100
            Returns (None, 0.0) if no static model loaded
        """
        if self.static_model is None:
            return None, 0.0
        
        prediction_idx = self.static_model.predict([normalized_landmarks])[0]
        probabilities = self.static_model.predict_proba([normalized_landmarks])[0]
        confidence = max(probabilities) * 100
        
        # Convert index to letter using loaded labels
        if self.static_labels and prediction_idx < len(self.static_labels):
            letter = self.static_labels[prediction_idx]
        else:
            # Fallback to A-Z mapping
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
            Returns (None, 0.0) if no dynamic model loaded
        """
        if self.dynamic_model is None:
            return None, 0.0
        
        # Compute delta features
        delta_features = [curr - old for curr, old in zip(current_landmarks, old_landmarks)]
        
        # Concatenate: [current_42, delta_42] = 84 features
        features = current_landmarks + delta_features
        
        prediction_idx = self.dynamic_model.predict([features])[0]
        probabilities = self.dynamic_model.predict_proba([features])[0]
        confidence = max(probabilities) * 100
        
        # Convert index to letter using loaded labels
        if self.dynamic_labels and prediction_idx < len(self.dynamic_labels):
            letter = self.dynamic_labels[prediction_idx]
        else:
            # Fallback to A-Z mapping
            letter = chr(prediction_idx + ord('A'))
        
        return letter, confidence
    
    def has_static_model(self):
        """Check if static model is loaded."""
        return self.static_model is not None
    
    def has_dynamic_model(self):
        """Check if dynamic model is loaded."""
        return self.dynamic_model is not None