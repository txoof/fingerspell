import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import Counter, deque
import copy
import itertools

# ============================================================
# LOAD MODEL
# ============================================================

static_model = joblib.load('./static_collection/models/static_letters_rf.pkl')

STATIC_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'K', 'L', 'M', 
                  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ============================================================
# PREPROCESSING (Match training exactly)
# ============================================================

def calc_landmark_list(image, landmarks):
    """Extract landmark coordinates from MediaPipe"""
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    """Normalize landmarks exactly as done in training"""
    temp_landmark_list = copy.deepcopy(landmark_list)
    
    # Convert to relative coordinates
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    
    # Flatten
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    
    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    if max_value > 0:
        temp_landmark_list = list(map(lambda n: n / max_value, temp_landmark_list))
    
    return temp_landmark_list

# ============================================================
# MAIN LOOP
# ============================================================

prediction_history = deque(maxlen=5)
cap = cv2.VideoCapture(0)

print("Static Letter Tester")
print(f"Testing letters: {', '.join(STATIC_LETTERS)}")
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
        
        current_prediction = None
        current_confidence = 0
        top_3_predictions = []
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Process landmarks exactly as in training
            landmark_list = calc_landmark_list(frame, hand_landmarks)
            processed_landmarks = pre_process_landmark(landmark_list)
            
            # Predict
            prediction = static_model.predict([processed_landmarks])[0]
            probabilities = static_model.predict_proba([processed_landmarks])[0]
            confidence = max(probabilities) * 100
            
            # Get top 3 predictions
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            top_3_predictions = [
                (static_model.classes_[idx], probabilities[idx] * 100) 
                for idx in top_3_indices
            ]
            
            prediction_history.append(prediction)
            
            # Smooth prediction with voting
            if len(prediction_history) >= 3:
                current_prediction = Counter(prediction_history).most_common(1)[0][0]
            else:
                current_prediction = prediction
            
            current_confidence = confidence
        
        else:
            prediction_history.clear()
        
        # ============================================================
        # UI
        # ============================================================
        
        # Draw black background for info
        cv2.rectangle(frame, (10, 10), (400, 250), (0, 0, 0), -1)
        
        if current_prediction:
            # Confidence-based color
            if current_confidence > 85:
                color = (0, 255, 0)  # Green
            elif current_confidence > 70:
                color = (0, 200, 255)  # Orange
            else:
                color = (0, 165, 255)  # Light orange
            
            # Main prediction
            cv2.putText(frame, f"Letter: {current_prediction}", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.2, color, 3, cv2.LINE_AA)
            
            # Confidence
            cv2.putText(frame, f"Confidence: {current_confidence:.1f}%", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, color, 2, cv2.LINE_AA)
            
            # Top 3 predictions
            cv2.putText(frame, "Top 3:", 
                       (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (200, 200, 200), 1, cv2.LINE_AA)
            
            for i, (letter, prob) in enumerate(top_3_predictions):
                y_pos = 160 + i * 30
                cv2.putText(frame, f"{i+1}. {letter}: {prob:.1f}%", 
                           (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (200, 200, 200), 1, cv2.LINE_AA)
        
        else:
            cv2.putText(frame, "Show hand gesture", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        cv2.imshow('Static Letter Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()