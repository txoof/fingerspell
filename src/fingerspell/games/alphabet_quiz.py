"""
Main application window and event loop.

Coordinates camera capture, hand detection, supervisor processing, and display.
"""

import random
import cv2
import mediapipe as mp
from src.fingerspell.core.landmarks import calc_landmark_list, pre_process_landmark
from src.fingerspell.core.models import ModelManager
from src.fingerspell.core.supervisor import Supervisor
from src.fingerspell.ui.display import (
    draw_prediction_display,
    draw_no_hand_display,
    draw_debug_display
)
from src.fingerspell.ui.common import draw_landmarks


class AlphabetQuizWindow:
    """Main application window for alphabet quiz."""
    
    def __init__(self, static_model_path=None, dynamic_model_path=None,
                 static_labels_path=None, dynamic_labels_path=None):
        """
        Initialize quiz.
        
        Args:
            static_model_path: Path to static classifier (optional)
            dynamic_model_path: Path to dynamic classifier (optional)
            static_labels_path: Path to static labels CSV (optional)
            dynamic_labels_path: Path to dynamic labels CSV (optional)
        """
        # Initialize models and supervisor
        self.model_manager = ModelManager(
            static_model_path,
            dynamic_model_path,
            static_labels_path,
            dynamic_labels_path
        )
        self.supervisor = Supervisor(self.model_manager)
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = None
        
        # Camera
        self.cap = None
        
        # UI state
        self.show_debug = False
        
        # Ctrl key state tracking
        self.ctrl_pressed = False

        # Alphabet
        self.alphabet = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
    
    def setup_camera(self, width=960, height=540):
        """
        Setup camera capture.
        
        Args:
            width: Frame width
            height: Frame height
        """
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def setup_mediapipe(self):
        """Setup MediaPipe hands."""
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    def handle_keyboard(self, key):
        """
        Handle keyboard input.
        
        Args:
            key: Key code from cv2.waitKey
            
        Returns:
            True to continue running, False to quit
        """
        # Global controls (always available)
        if key == 27:  # ESC to quit
            return 'quit'
        
        if key == ord('f'):
            return 'next'
        
        # elif key == 9:  # Tab to toggle debug
        #     self.show_debug = not self.show_debug
        #     print(f"Debug overlay: {'ON' if self.show_debug else 'OFF'}")
        
        # Debug-only controls (threshold adjustments)
        elif self.show_debug:
            # Motion threshold adjustment
            if key == ord('k'):
                self.supervisor.adjust_motion_threshold(0.01)
                print(f"Motion threshold: {self.supervisor.motion_threshold:.3f}")
            
            elif key == ord('j'):
                self.supervisor.adjust_motion_threshold(-0.01)
                print(f"Motion threshold: {self.supervisor.motion_threshold:.3f}")
            
            elif key == ord('K'):
                self.supervisor.adjust_motion_threshold(0.05)
                print(f"Motion threshold: {self.supervisor.motion_threshold:.3f}")
            
            elif key == ord('J'):
                self.supervisor.adjust_motion_threshold(-0.05)
                print(f"Motion threshold: {self.supervisor.motion_threshold:.3f}")
            
            # Confidence threshold adjustment
            elif key == ord('w'):
                self.supervisor.adjust_confidence_thresholds(low_delta=5)
                print(f"Confidence thresholds: Low={self.supervisor.confidence_threshold_low:.0f} "
                      f"High={self.supervisor.confidence_threshold_high:.0f}")
            
            elif key == ord('s'):
                self.supervisor.adjust_confidence_thresholds(low_delta=-5)
                print(f"Confidence thresholds: Low={self.supervisor.confidence_threshold_low:.0f} "
                      f"High={self.supervisor.confidence_threshold_high:.0f}")
            
            elif key == ord('W'):
                self.supervisor.adjust_confidence_thresholds(high_delta=5)
                print(f"Confidence thresholds: Low={self.supervisor.confidence_threshold_low:.0f} "
                      f"High={self.supervisor.confidence_threshold_high:.0f}")
            
            elif key == ord('S'):
                self.supervisor.adjust_confidence_thresholds(high_delta=-5)
                print(f"Confidence thresholds: Low={self.supervisor.confidence_threshold_low:.0f} "
                      f"High={self.supervisor.confidence_threshold_high:.0f}")
        
        return 'continue'
    
    def process_frame(self, frame):
        """
        Process a single frame.
        
        Args:
            frame: Camera frame
            
        Returns:
            Processed frame with overlays
        """
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Initialize result and predictions for debug
        result = None
        static_pred = (None, 0.0)
        dynamic_pred = (None, 0.0)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract and normalize landmarks
            landmark_list = calc_landmark_list(frame, hand_landmarks)
            normalized_landmarks = pre_process_landmark(landmark_list)
            
            # Draw hand skeleton using our function
            frame = draw_landmarks(frame, landmark_list)
            
            # Get wrist position from MediaPipe (normalized coords)
            wrist_landmark = hand_landmarks.landmark[0]
            wrist_pos = [wrist_landmark.x, wrist_landmark.y, wrist_landmark.z]
            
            # Get predictions (for both display and debug)
            static_pred = self.model_manager.predict_static(normalized_landmarks)
            
            # Get dynamic prediction if buffer ready
            if len(self.supervisor.landmark_buffer) >= self.supervisor.rolling_window_size:
                current = self.supervisor.landmark_buffer[-1] if self.supervisor.landmark_buffer else normalized_landmarks
                old = self.supervisor.landmark_buffer[0]
                dynamic_pred = self.model_manager.predict_dynamic(current, old)
            
            # Process through supervisor
            result = self.supervisor.process_frame(normalized_landmarks, wrist_pos)
            
            if result:
                # Draw main prediction
                frame = draw_prediction_display(frame, result.letter, result.confidence)
        else:
            # No hand detected
            self.supervisor.clear_buffers()
            frame = draw_no_hand_display(frame)
        
        # Draw debug overlay if enabled (always show when debug mode active)
        if self.show_debug:
            frame = draw_debug_display(
                frame,
                result,
                self.supervisor,
                static_pred,
                dynamic_pred,
                self.model_manager
            )
        
        return frame, result
    
    def run(self):
        """Run the main application loop."""
        self.setup_camera()
        self.setup_mediapipe()
        celebration_words = ['Excellent!', 'Nice!', 'Wonderful!', 'Amazing!', 'Great!', 'Perfect!']

        try:
            for l in self.alphabet:
                should_exit = False
                while self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    frame, result = self.process_frame(frame)
                    
                    # Draw outline (thicker, black) and main text on top (thinner, green)
                    cv2.putText(frame, l, (1100, 120), cv2.FONT_HERSHEY_DUPLEX, 4, (0, 0, 0), 10)
                    cv2.putText(frame, l, (1100, 120), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 100, 100), 6)

                    if result is not None:
                        if result.letter == l and result.confidence > self.supervisor.confidence_threshold_low:
                            word = random.choice(celebration_words)
                            for _ in range(45):
                                ret, frame = self.cap.read()
                                if not ret:
                                    break
                                frame = cv2.flip(frame, 1)

                                # Draw outline (thicker, black) and main text on top (thinner, green)
                                cv2.putText(frame, l, (1100, 120), cv2.FONT_HERSHEY_DUPLEX, 4, (0, 0, 0), 10)
                                cv2.putText(frame, l, (1100, 120), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 100, 100), 6)

                                # Center the text at bottom
                                text_size = cv2.getTextSize(word, cv2.FONT_HERSHEY_DUPLEX, 3, 4)[0]
                                text_x = (1280 - text_size[0]) // 2
                                text_y = 680

                                # Draw outline (thicker, black) and main text on top (thinner, green)
                                cv2.putText(frame, word, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 0), 10)
                                cv2.putText(frame, word, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 255, 0), 4)
                                
                                cv2.imshow('Alphabet Quiz', frame)
                                key = cv2.waitKey(1) & 0xFF
                                if key == 27:  # Allow ESC during celebration
                                    should_exit = True
                                    break
                            break
                        print(result.letter, result.confidence)
                    # Display
                    cv2.imshow('Alphabet Quiz', frame)
                    
                    # Handle keyboard
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255:  # Key was pressed
                        result = self.handle_keyboard(key)
                        if result == 'quit':
                            should_exit = True
                            break
                        elif result == 'next':
                            break
                
                if should_exit:
                    break
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        if self.hands:
            self.hands.close()
        cv2.destroyAllWindows()


def run_quiz(static_model_path=None, dynamic_model_path=None,
            static_labels_path=None, dynamic_labels_path=None):
    """
    Run the alphabet quiz.
    
    Args:
        static_model_path: Path to static classifier (optional)
        dynamic_model_path: Path to dynamic classifier (optional)
        static_labels_path: Path to static labels CSV (optional)
        dynamic_labels_path: Path to dynamic labels CSV (optional)
    """
    window = AlphabetQuizWindow(
        static_model_path,
        dynamic_model_path,
        static_labels_path,
        dynamic_labels_path
    )
    window.run()