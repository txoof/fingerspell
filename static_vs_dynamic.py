import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import time

# ============================================================
# CONFIGURATION
# ============================================================

static_model = joblib.load('./models/static_model_clf.pkl')
dynamic_model = joblib.load('./models/dynamic_model_clf.pkl')
dynamic_metadata = joblib.load('./models/dynamic_metadata.pkl')

TARGET_FRAMES = dynamic_metadata['target_frames']
STATIC_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'K', 'L', 'M', 
                  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
DYNAMIC_LETTERS = ['H', 'J', 'U', 'X', 'Z']

STATIC_CONFIDENCE_THRESHOLD = 70
WRIST_MOTION_THRESHOLD = 0.1
WRIST_MOTION_WINDOW = 10
MIN_FRAMES_FOR_DYNAMIC = 30

MIN_DISPLAY_CONFIDENCE = 40
PREDICTION_HOLD_TIME = 2.0

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_frame(landmarks_63):
    coords = np.array(landmarks_63).reshape(21, 3)
    wrist = coords[0]
    coords = coords - wrist
    hand_size = np.linalg.norm(coords[12])
    if hand_size > 0:
        coords = coords / hand_size
    return coords.flatten()

def calculate_wrist_motion(wrist_positions):
    if len(wrist_positions) < 2:
        return 0.0
    
    total_distance = 0.0
    positions = list(wrist_positions)
    
    for i in range(len(positions) - 1):
        diff = np.array(positions[i+1]) - np.array(positions[i])
        distance = np.linalg.norm(diff)
        total_distance += distance
    
    return total_distance

def predict_static(normalized_frame):
    prediction = static_model.predict([normalized_frame])[0]
    confidence = max(static_model.predict_proba([normalized_frame])[0]) * 100
    return prediction, confidence

def predict_dynamic(frame_buffer):
    frames = list(frame_buffer)
    
    if len(frames) < TARGET_FRAMES:
        frames.extend([frames[-1]] * (TARGET_FRAMES - len(frames)))
    else:
        frames = frames[:TARGET_FRAMES]
    
    sequence = np.array(frames).flatten()
    prediction = dynamic_model.predict([sequence])[0]
    confidence = max(dynamic_model.predict_proba([sequence])[0]) * 100
    
    return prediction, confidence

# ============================================================
# MAIN LOOP
# ============================================================

frame_buffer = deque(maxlen=50)
wrist_buffer = deque(maxlen=WRIST_MOTION_WINDOW)
prediction_history = deque(maxlen=3)

current_prediction = None
current_confidence = 0
prediction_source = None
wrist_motion = 0.0

last_valid_prediction = None
last_valid_confidence = 0
last_valid_source = None
last_prediction_time = None

cap = cv2.VideoCapture(0)

print("NGT Recognizer")
print(f"Min confidence to display: {MIN_DISPLAY_CONFIDENCE}%")
print(f"Prediction hold time: {PREDICTION_HOLD_TIME}s")
print("Press 'q' to quit\n")

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
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            coords = [coord for lm in hand_landmarks.landmark 
                     for coord in [lm.x, lm.y, lm.z]]
            
            wrist_pos = coords[0:3]
            wrist_buffer.append(wrist_pos)
            
            normalized = normalize_frame(coords)
            frame_buffer.append(normalized)
            
            wrist_motion = calculate_wrist_motion(wrist_buffer)
            
            # Decision logic
            static_pred, static_conf = predict_static(normalized)
            is_dynamic_motion = wrist_motion > WRIST_MOTION_THRESHOLD
            
            if static_conf >= STATIC_CONFIDENCE_THRESHOLD and not is_dynamic_motion:
                best_pred = static_pred
                best_conf = static_conf
                source = "static"
            
            elif is_dynamic_motion and len(frame_buffer) >= MIN_FRAMES_FOR_DYNAMIC:
                dynamic_pred, dynamic_conf = predict_dynamic(frame_buffer)
                
                if dynamic_pred in DYNAMIC_LETTERS:
                    best_pred = dynamic_pred
                    best_conf = dynamic_conf
                    source = "dynamic"
                else:
                    best_pred = static_pred
                    best_conf = static_conf
                    source = "static"
            else:
                best_pred = static_pred
                best_conf = static_conf
                source = "static"
            
            prediction_history.append(best_pred)
            from collections import Counter
            current_prediction = Counter(prediction_history).most_common(1)[0][0]
            current_confidence = best_conf
            prediction_source = source
            
            if current_confidence >= MIN_DISPLAY_CONFIDENCE:
                last_valid_prediction = current_prediction
                last_valid_confidence = current_confidence
                last_valid_source = prediction_source
                last_prediction_time = time.time()
        
        else:
            frame_buffer.clear()
            wrist_buffer.clear()
            prediction_history.clear()
            current_prediction = None
            wrist_motion = 0.0
        
        # Determine what to display
        display_prediction = None
        display_confidence = 0
        display_source = None
        is_held_prediction = False
        
        if last_valid_prediction is not None:
            time_since_prediction = time.time() - last_prediction_time if last_prediction_time else 0
            
            if time_since_prediction <= PREDICTION_HOLD_TIME:
                display_prediction = last_valid_prediction
                display_confidence = last_valid_confidence
                display_source = last_valid_source
                
                if current_prediction is None:
                    is_held_prediction = True
            else:
                last_valid_prediction = None
                last_valid_confidence = 0
                last_valid_source = None
                last_prediction_time = None
        
        # ============================================================
        # UI - LEFT SIDE: PREDICTION
        # ============================================================
        
        if display_prediction:
            if display_confidence > 85:
                color = (0, 255, 0)
            elif display_confidence > 70:
                color = (0, 200, 255)
            else:
                color = (0, 165, 255)
            
            if is_held_prediction:
                color = tuple(int(c * 0.7) for c in color)
            
            # Letter
            cv2.putText(frame, display_prediction, 
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                       4, color, 8, cv2.LINE_AA)
            
            # Confidence
            cv2.putText(frame, f"{display_confidence:.0f}%", 
                       (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.2, color, 3, cv2.LINE_AA)
            
            # Source
            if display_source == "dynamic":
                source_color = (255, 0, 255)
                source_text = "DYNAMIC"
            else:
                source_color = (255, 255, 255)
                source_text = "STATIC"
            
            if is_held_prediction:
                source_color = tuple(int(c * 0.7) for c in source_color)
            
            cv2.putText(frame, source_text, 
                       (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, source_color, 2, cv2.LINE_AA)
            
            # Held indicator
            if is_held_prediction:
                time_remaining = PREDICTION_HOLD_TIME - (time.time() - last_prediction_time)
                cv2.putText(frame, f"HELD ({time_remaining:.1f}s)", 
                           (50, 235), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (150, 150, 150), 2, cv2.LINE_AA)
        
        else:
            if current_prediction and current_confidence < MIN_DISPLAY_CONFIDENCE:
                cv2.putText(frame, "Uncertain...", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.5, (100, 100, 100), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Show hand", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                           2, (0, 0, 255), 4, cv2.LINE_AA)
        
        # ============================================================
        # UI - TOP RIGHT: MOTION BAR
        # ============================================================
        
        # Bar settings (top-right corner)
        bar_width = 250
        bar_height = 20
        bar_x = w - bar_width - 20  # 20px from right edge
        bar_y = 20                   # 20px from top
        
        # Label above bar
        cv2.putText(frame, "Motion", 
                   (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (80, 80, 80), -1)
        
        # Fill bar
        max_display_motion = 0.3
        fill_ratio = min(1.0, wrist_motion / max_display_motion)
        fill_width = int(bar_width * fill_ratio)
        
        if wrist_motion >= WRIST_MOTION_THRESHOLD:
            fill_color = (0, 255, 0)
        else:
            fill_color = (100, 100, 100)
        
        if fill_width > 0:
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + fill_width, bar_y + bar_height), 
                         fill_color, -1)
        
        # Threshold line (red)
        threshold_x = bar_x + int(bar_width * (WRIST_MOTION_THRESHOLD / max_display_motion))
        cv2.line(frame, (threshold_x, bar_y), 
                (threshold_x, bar_y + bar_height), 
                (0, 0, 255), 2)
        
        # Bar border
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (200, 200, 200), 1)
        
        # Motion value below bar
        cv2.putText(frame, f"{wrist_motion:.3f}", 
                   (bar_x, bar_y + bar_height + 15), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.45, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Quit instruction
        cv2.putText(frame, "Press 'q' to quit", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        cv2.imshow('NGT Recognizer', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()