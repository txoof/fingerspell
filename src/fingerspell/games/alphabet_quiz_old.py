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


class AlphabetQuizWindow:
    """Main application window for fingerspelling alphabet quiz."""
    
    def __init__(self, static_model_path, dynamic_model_path):
        """
        Initialize window.
        
        Args:
            static_model_path: Path to static classifier
            dynamic_model_path: Path to dynamic classifier
        """
        # Initialize models and supervisor
        self.model_manager = ModelManager(static_model_path, dynamic_model_path)
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
            return False
        
        elif key == 9:  # Tab to toggle debug
            self.show_debug = not self.show_debug
            print(f"Debug overlay: {'ON' if self.show_debug else 'OFF'}")
        
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
        
        return True
    
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
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw hand skeleton
            self.mp_drawing.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
            )
            
            # Extract and normalize landmarks
            landmark_list = calc_landmark_list(frame, hand_landmarks)
            normalized_landmarks = pre_process_landmark(landmark_list)
            
            # Get wrist position from MediaPipe (normalized coords)
            wrist_landmark = hand_landmarks.landmark[0]
            wrist_pos = [wrist_landmark.x, wrist_landmark.y, wrist_landmark.z]
            
            # Get predictions for debug display
            static_letter, static_conf = self.model_manager.predict_static(normalized_landmarks)
            print(f"Log: {static_letter, static_conf}")
            
            # Check if dynamic buffer is ready
            if len(self.supervisor.landmark_buffer) >= self.supervisor.rolling_window_size:
                current = self.supervisor.landmark_buffer[-1] if self.supervisor.landmark_buffer else normalized_landmarks
                old = self.supervisor.landmark_buffer[0] if len(self.supervisor.landmark_buffer) >= self.supervisor.rolling_window_size else normalized_landmarks
                dynamic_letter, dynamic_conf = self.model_manager.predict_dynamic(current, old)
                dynamic_pred = (dynamic_letter, dynamic_conf)
            else:
                dynamic_pred = (None, 0.0)
            print(f"Log: {dynamic_pred}")
            
            # Process through supervisor
            result = self.supervisor.process_frame(normalized_landmarks, wrist_pos)
            
            if result:
                # Draw main prediction
                draw_prediction_display(
                    frame,
                    result.letter,
                    result.confidence,
                    self.supervisor.confidence_threshold_low,
                    self.supervisor.confidence_threshold_high,
                    debug=self.show_debug
                )
                
                
                # Draw debug overlay if enabled
                if self.show_debug:
                    draw_debug_display(
                        frame,
                        result,
                        self.supervisor,
                        (static_letter, static_conf),
                        dynamic_pred
                    )

        else:
            # No hand detected
            self.supervisor.clear_buffers()
            draw_no_hand_display(frame)
        
        return frame
    
    def run(self):
        """Run the main application loop."""
        self.setup_camera()
        self.setup_mediapipe()
        
        print("\nNGT Fingerspelling Recognizer")
        print("Press 'ESC' to quit")
        print("Press 'Tab' to toggle debug overlay")
        print("\nDebug mode controls:")
        print("  k/j: Adjust motion threshold (+/- 0.01)")
        print("  K/J: Adjust motion threshold (+/- 0.05)")
        print("  w/s: Adjust low confidence threshold (+/- 5)")
        print("  W/S: Adjust high confidence threshold (+/- 5)")
        print()
        
        try:
            for l in self.alphabet:
                while self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    frame = self.process_frame(frame)
                    cv2.putText(frame, l, (1100, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 6)
                    
                    # Display
                    cv2.imshow('NGT Recognizer', frame)
                    
                    # Handle keyboard
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255:  # Key was pressed
                        if not self.handle_keyboard(key):
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
        print("\nShutdown complete")


def run_quiz(static_model_path='../models/ngt_static_classifier.pkl',
            dynamic_model_path='../models/ngt_dynamic_classifier.pkl'):
    """
    Run the fingerspelling alphabet quiz application.
    
    Args:
        static_model_path: Path to static classifier
        dynamic_model_path: Path to dynamic classifier
    """
    window = AlphabetQuizWindow(static_model_path, dynamic_model_path)
    window.run()