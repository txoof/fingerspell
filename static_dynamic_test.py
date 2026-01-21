import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

print("Loading models...")
static_model = joblib.load('./models/static_model_clf.pkl')
dynamic_model = joblib.load('./models/dynamic_model_clf.pkl')
dynamic_metadata = joblib.load('./models/dynamic_metadata.pkl')

TARGET_FRAMES = dynamic_metadata['target_frames']
STATIC_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'K', 'L', 'M', 
                  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
DYNAMIC_LETTERS = ['H', 'J', 'U', 'X', 'Z']

# TUNABLE THRESHOLDS
STATIC_CONFIDENCE_THRESHOLD = 60  # If static confidence > this, trust it
DYNAMIC_CONFIDENCE_THRESHOLD = 60  # Only use dynamic if > this
MIN_FRAMES_FOR_DYNAMIC = 30        # Need at least this many frames for dynamic

print("✓ Models loaded")
print(f"Strategy: Static-first (threshold: {STATIC_CONFIDENCE_THRESHOLD}%)")
print(f"          Dynamic fallback (threshold: {DYNAMIC_CONFIDENCE_THRESHOLD}%)")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def normalize_frame(landmarks_63):
    """Normalize single frame"""
    coords = np.array(landmarks_63).reshape(21, 3)
    wrist = coords[0]
    coords = coords - wrist
    hand_size = np.linalg.norm(coords[12] - coords[0])
    if hand_size > 0:
        coords = coords / hand_size
    return coords.flatten()

def predict_static(normalized_frame):
    """Predict static letter"""
    prediction = static_model.predict([normalized_frame])[0]
    probabilities = static_model.predict_proba([normalized_frame])[0]
    confidence = max(probabilities) * 100
    return prediction, confidence

def predict_dynamic(frame_buffer):
    """Predict dynamic letter from sequence"""
    frames = list(frame_buffer)
    
    if len(frames) < TARGET_FRAMES:
        padding = [frames[-1]] * (TARGET_FRAMES - len(frames))
        frames.extend(padding)
    else:
        frames = frames[:TARGET_FRAMES]
    
    sequence = np.array(frames).flatten()
    prediction = dynamic_model.predict([sequence])[0]
    probabilities = dynamic_model.predict_proba([sequence])[0]
    confidence = max(probabilities) * 100
    
    return prediction, confidence

# ============================================================
# MAIN LOOP
# ============================================================

BUFFER_SIZE = 50
frame_buffer = deque(maxlen=BUFFER_SIZE)

current_prediction = None
current_confidence = 0
prediction_source = None
static_conf_display = 0
dynamic_conf_display = 0

# Stability tracking
STABILITY_THRESHOLD = 3  # Reduced for faster response
prediction_history = deque(maxlen=STABILITY_THRESHOLD)

cap = cv2.VideoCapture(0)

print("\n" + "="*60)
print("NGT RECOGNIZER - STATIC FIRST STRATEGY")
print("="*60)
print("Logic:")
print("  1. Try static model")
print(f"  2. If confidence < {STATIC_CONFIDENCE_THRESHOLD}% → Try dynamic")
print("  3. Use best prediction")
print("\nPress 'q' to quit")
print("="*60 + "\n")

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract and normalize
                coords = [coord for lm in hand_landmarks.landmark 
                         for coord in [lm.x, lm.y, lm.z]]
                normalized = normalize_frame(coords)
                frame_buffer.append(normalized)
                
                # ============================================================
                # DECISION LOGIC: STATIC FIRST!
                # ============================================================
                
                # Step 1: ALWAYS try static model first
                static_pred, static_conf = predict_static(normalized)
                static_conf_display = static_conf
                
                # Step 2: Check if static prediction is confident enough
                if static_conf >= STATIC_CONFIDENCE_THRESHOLD:
                    # Static model is confident → Use it!
                    best_pred = static_pred
                    best_conf = static_conf
                    source = "static ✓"
                    dynamic_conf_display = 0  # Didn't even check dynamic
                
                else:
                    # Static model NOT confident → Try dynamic as fallback
                    
                    if len(frame_buffer) >= MIN_FRAMES_FOR_DYNAMIC:
                        # Have enough frames for dynamic prediction
                        dynamic_pred, dynamic_conf = predict_dynamic(frame_buffer)
                        dynamic_conf_display = dynamic_conf
                        
                        # Compare static vs dynamic
                        if (dynamic_conf >= DYNAMIC_CONFIDENCE_THRESHOLD and 
                            dynamic_pred in DYNAMIC_LETTERS):
                            # Dynamic model is confident AND predicts a dynamic letter
                            best_pred = dynamic_pred
                            best_conf = dynamic_conf
                            source = "dynamic (fallback)"
                        else:
                            # Dynamic also not confident → Use static anyway
                            best_pred = static_pred
                            best_conf = static_conf
                            source = "static (low conf)"
                    else:
                        # Not enough frames yet for dynamic
                        best_pred = static_pred
                        best_conf = static_conf
                        source = "static (buffer filling)"
                        dynamic_conf_display = 0
                
                # Stability filtering
                prediction_history.append(best_pred)
                
                if len(prediction_history) >= STABILITY_THRESHOLD:
                    from collections import Counter
                    most_common = Counter(prediction_history).most_common(1)[0][0]
                    current_prediction = most_common
                    current_confidence = best_conf
                    prediction_source = source
                else:
                    current_prediction = best_pred
                    current_confidence = best_conf
                    prediction_source = source
        
        else:
            # No hand detected
            frame_buffer.clear()
            prediction_history.clear()
            current_prediction = None
            static_conf_display = 0
            dynamic_conf_display = 0
        
        # ============================================================
        # DISPLAY UI
        # ============================================================
        
        # Dark background for info
        cv2.rectangle(frame, (0, 0), (w, 200), (40, 40, 40), -1)
        
        if current_prediction:
            # Color based on confidence
            if current_confidence > 85:
                color = (0, 255, 0)  # Green
            elif current_confidence > 70:
                color = (0, 200, 255)  # Orange
            else:
                color = (0, 165, 255)  # Yellow
            
            # Main prediction
            cv2.putText(frame, f"Letter: {current_prediction}", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
            
            # Confidence
            cv2.putText(frame, f"Confidence: {current_confidence:.0f}%", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Source (which model was used)
            cv2.putText(frame, f"Source: {prediction_source}", 
                       (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Model confidences (debug info)
            debug_text = f"Static: {static_conf_display:.0f}%"
            if dynamic_conf_display > 0:
                debug_text += f" | Dynamic: {dynamic_conf_display:.0f}%"
            cv2.putText(frame, debug_text, 
                       (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        else:
            cv2.putText(frame, "Show your hand", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # Buffer status (top right)
        buffer_text = f"Buffer: {len(frame_buffer)}/{BUFFER_SIZE}"
        cv2.putText(frame, buffer_text, 
                   (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Threshold indicator
        threshold_text = f"Threshold: {STATIC_CONFIDENCE_THRESHOLD}%"
        cv2.putText(frame, threshold_text, 
                   (w - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit | '+'/'-' to adjust threshold", 
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow('NGT Recognizer - Static First', frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            STATIC_CONFIDENCE_THRESHOLD = min(95, STATIC_CONFIDENCE_THRESHOLD + 5)
            print(f"Threshold increased to {STATIC_CONFIDENCE_THRESHOLD}%")
        elif key == ord('-') or key == ord('_'):
            STATIC_CONFIDENCE_THRESHOLD = max(50, STATIC_CONFIDENCE_THRESHOLD - 5)
            print(f"Threshold decreased to {STATIC_CONFIDENCE_THRESHOLD}%")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("Session Summary:")
print(f"Final threshold: {STATIC_CONFIDENCE_THRESHOLD}%")
print("="*60)