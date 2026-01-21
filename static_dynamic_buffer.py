import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

print("Loading models...")
# Load both models
static_model = joblib.load('./models/static_model_clf.pkl')
dynamic_model = joblib.load('./models/dynamic_model_clf.pkl')
dynamic_metadata = joblib.load('./models/dynamic_metadata.pkl')

TARGET_FRAMES = dynamic_metadata['target_frames']
STATIC_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'K', 'L', 'M', 
                  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
DYNAMIC_LETTERS = ['H', 'J', 'U', 'X', 'Z']

print("✓ Models loaded")

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

def calculate_motion(frame_buffer, window=10):
    """
    Calculate how much motion is in the buffer
    Returns: motion score (0 = no motion, 1 = lots of motion)
    """
    if len(frame_buffer) < window:
        return 0.0
    
    # Compare recent frames to detect movement
    recent_frames = list(frame_buffer)[-window:]
    
    # Calculate average distance between consecutive frames
    total_distance = 0
    for i in range(len(recent_frames) - 1):
        # Calculate difference between frames (focus on wrist and key landmarks)
        # Using first 30 coordinates (wrist + first few landmarks)
        diff = np.array(recent_frames[i+1][:30]) - np.array(recent_frames[i][:30])
        distance = np.linalg.norm(diff)
        total_distance += distance
    
    avg_distance = total_distance / (len(recent_frames) - 1)
    
    # Normalize motion score (empirically tuned threshold)
    # Typical static hand: ~0.01-0.05
    # Dynamic gesture: ~0.1-0.4
    motion_score = min(1.0, avg_distance / 0.15)
    
    return motion_score

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
prediction_type = None
current_motion_score = 0

STABILITY_THRESHOLD = 5
prediction_history = deque(maxlen=STABILITY_THRESHOLD)

cap = cv2.VideoCapture(0)

print("\n" + "="*60)
print("NGT RECOGNIZER WITH MOTION DETECTION")
print("="*60)
print("Static letters: Hold steady")
print("Dynamic letters: Perform gesture")
print("Press 'q' to quit")
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
                
                # Calculate motion score
                motion_score = calculate_motion(frame_buffer)
                current_motion_score = motion_score
                
                # Always predict with static model
                static_pred, static_conf = predict_static(normalized)
                
                # Only predict with dynamic if motion detected
                if len(frame_buffer) >= 25 and motion_score > 0.15:
                    # Significant motion detected
                    dynamic_pred, dynamic_conf = predict_dynamic(frame_buffer)
                    use_dynamic = True
                else:
                    dynamic_pred, dynamic_conf = None, 0
                    use_dynamic = False
                
                # DECISION LOGIC (IMPROVED)
                
                if not use_dynamic:
                    # No significant motion → Must be static
                    best_pred = static_pred
                    best_conf = static_conf
                    pred_type = "static"
                
                elif motion_score > 0.3:
                    # HIGH motion → Likely dynamic gesture
                    if dynamic_pred and dynamic_conf > 70:
                        best_pred = dynamic_pred
                        best_conf = dynamic_conf
                        pred_type = "dynamic"
                    else:
                        # Motion but uncertain → prefer static
                        best_pred = static_pred
                        best_conf = static_conf
                        pred_type = "static (motion)"
                
                elif static_conf > 80:
                    # Very confident static → use it even if some motion
                    best_pred = static_pred
                    best_conf = static_conf
                    pred_type = "static"
                
                elif dynamic_pred and dynamic_conf > 85 and static_conf < 70:
                    # High confidence dynamic, low confidence static → use dynamic
                    best_pred = dynamic_pred
                    best_conf = dynamic_conf
                    pred_type = "dynamic"
                
                else:
                    # Default to static (safer choice)
                    best_pred = static_pred
                    best_conf = static_conf
                    pred_type = "static (default)"
                
                # Stability filtering
                prediction_history.append(best_pred)
                
                if len(prediction_history) >= STABILITY_THRESHOLD:
                    from collections import Counter
                    most_common = Counter(prediction_history).most_common(1)[0][0]
                    current_prediction = most_common
                    current_confidence = best_conf
                    prediction_type = pred_type
                else:
                    current_prediction = best_pred
                    current_confidence = best_conf
                    prediction_type = pred_type
        
        else:
            frame_buffer.clear()
            prediction_history.clear()
            current_prediction = None
            current_motion_score = 0
        
        # ============================================================
        # DISPLAY
        # ============================================================
        
        cv2.rectangle(frame, (0, 0), (w, 180), (40, 40, 40), -1)
        
        if current_prediction:
            # Color based on confidence
            if current_confidence > 85:
                color = (0, 255, 0)
            elif current_confidence > 70:
                color = (0, 200, 255)
            else:
                color = (0, 165, 255)
            
            cv2.putText(frame, f"Letter: {current_prediction}", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
            cv2.putText(frame, f"Confidence: {current_confidence:.0f}%", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Motion indicator
            motion_text = f"Motion: {current_motion_score:.2f}"
            motion_color = (0, 255, 0) if current_motion_score > 0.15 else (200, 200, 200)
            cv2.putText(frame, motion_text, 
                       (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, motion_color, 2)
            
            # Type indicator
            cv2.putText(frame, f"({prediction_type})", 
                       (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(frame, "Show your hand", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # Motion visualization bar
        bar_width = 300
        bar_x, bar_y = w - bar_width - 20, 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (100, 100, 100), -1)
        motion_bar_width = int(bar_width * min(1.0, current_motion_score))
        bar_color = (0, 255, 0) if current_motion_score > 0.15 else (100, 100, 100)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + motion_bar_width, bar_y + 20), bar_color, -1)
        cv2.putText(frame, "Motion", (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Buffer status
        buffer_text = f"Buffer: {len(frame_buffer)}/{BUFFER_SIZE}"
        cv2.putText(frame, buffer_text, 
                   (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.putText(frame, "Press 'q' to quit", 
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow('NGT Recognizer - Motion-Aware', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()