"""
Supervisor module for routing between static and dynamic models.

Manages rolling buffers, motion detection, and prediction routing.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class PredictionResult:
    """Result from supervisor prediction."""
    letter: str
    confidence: float
    source: str  # 'static' or 'dynamic'
    motion: float


class Supervisor:
    """
    Orchestrates hand gesture recognition by routing to appropriate models.
    
    Maintains rolling buffers for landmarks and wrist positions, calculates
    motion, and decides whether to use static or dynamic model.
    """
    
    def __init__(self, model_manager, rolling_window_size=5, wrist_motion_window=10):
        """
        Initialize supervisor.
        
        Args:
            model_manager: ModelManager instance for predictions
            rolling_window_size: Number of frames for dynamic prediction buffer
            wrist_motion_window: Number of frames for motion calculation
        """
        self.model_manager = model_manager
        self.rolling_window_size = rolling_window_size
        self.wrist_motion_window = wrist_motion_window
        
        # Buffers
        self.landmark_buffer = deque(maxlen=rolling_window_size)
        self.wrist_buffer = deque(maxlen=wrist_motion_window)
        
        # Thresholds
        self.motion_threshold = 0.1
        self.confidence_threshold_low = 50.0
        self.confidence_threshold_high = 80.0
    
    def process_frame(self, normalized_landmarks, wrist_position) -> Optional[PredictionResult]:
        """
        Process a single frame and return prediction.
        
        Args:
            normalized_landmarks: List of 42 normalized coordinate values
            wrist_position: List of [x, y, z] in normalized MediaPipe coords
            
        Returns:
            PredictionResult or None if buffers not ready
        """
        # Add to buffers
        self.landmark_buffer.append(normalized_landmarks)
        self.wrist_buffer.append(wrist_position)
        
        # Calculate wrist motion
        motion = self._calculate_wrist_motion()
        
        # Get predictions
        static_letter, static_conf = self.model_manager.predict_static(normalized_landmarks)
        
        # Decide routing
        is_dynamic_motion = motion > self.motion_threshold
        
        # Try dynamic if motion detected and buffer is full
        if is_dynamic_motion and len(self.landmark_buffer) >= self.rolling_window_size:
            current = self.landmark_buffer[-1]
            old = self.landmark_buffer[0]
            dynamic_letter, dynamic_conf = self.model_manager.predict_dynamic(current, old)
            
            # Use dynamic prediction
            return PredictionResult(
                letter=dynamic_letter,
                confidence=dynamic_conf,
                source='dynamic',
                motion=motion
            )
        else:
            # Use static prediction
            return PredictionResult(
                letter=static_letter,
                confidence=static_conf,
                source='static',
                motion=motion
            )
    
    def _calculate_wrist_motion(self) -> float:
        """
        Calculate total wrist movement over the window.
        
        Returns:
            Total distance traveled by wrist
        """
        if len(self.wrist_buffer) < 2:
            return 0.0
        
        total_distance = 0.0
        positions = list(self.wrist_buffer)
        
        for i in range(len(positions) - 1):
            diff = np.array(positions[i + 1]) - np.array(positions[i])
            distance = np.linalg.norm(diff)
            total_distance += distance
        
        return total_distance
    
    def clear_buffers(self):
        """Clear all buffers (e.g., when hand disappears)."""
        self.landmark_buffer.clear()
        self.wrist_buffer.clear()
    
    def adjust_motion_threshold(self, delta):
        """
        Adjust motion threshold.
        
        Args:
            delta: Amount to adjust (can be negative)
        """
        self.motion_threshold = max(0.0, self.motion_threshold + delta)
    
    def adjust_confidence_thresholds(self, low_delta=0, high_delta=0):
        """
        Adjust confidence thresholds.
        
        Args:
            low_delta: Amount to adjust low threshold
            high_delta: Amount to adjust high threshold
        """
        if low_delta != 0:
            self.confidence_threshold_low = max(0.0, min(100.0, 
                self.confidence_threshold_low + low_delta))
        
        if high_delta != 0:
            self.confidence_threshold_high = max(0.0, min(100.0, 
                self.confidence_threshold_high + high_delta))
        
        # Ensure high >= low
        if self.confidence_threshold_high < self.confidence_threshold_low:
            self.confidence_threshold_high = self.confidence_threshold_low