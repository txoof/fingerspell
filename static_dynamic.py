# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: fingerspell-venv-af5b43d44a
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib



# %%
ngt_static_classifier = Path('./data/models/ngt_static_classifier_normalized.pkl')
ngt_dynamic_landmarks = Path('./data/dataset/ngt_dynamic_landmarks.csv')
ngt_dynamic_landmarks_clean = Path('./data/dataset/ngt_dynamic_landmarks_clean.csv')
ngt_dynamic_classifier = Path('./data/models/ngt_dynamic_classifier.pkl')
ngt_dynamic_scaler = Path('./data/models/ngt_dynamic_scaler.pkl')
ngt_static_scaler = Path('./data/models/ngt_static_scaler.pkl')
ngt_dynamic_classifier.parent.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# Cleanup the dynamic data

# %%

# Read line by line to find bad rows
bad_lines = []
good_lines = []

with open(ngt_dynamic_landmarks, 'r') as f:
    header = f.readline()
    expected_cols = len(header.strip().split(','))
    print(f"Header has {expected_cols} columns")
    
    good_lines.append(header)
    
    for i, line in enumerate(f, start=2):
        cols = len(line.strip().split(','))
        if cols != expected_cols:
            bad_lines.append((i, cols, line[:100]))  # Store line number, column count, preview
        else:
            good_lines.append(line)

print(f"\nFound {len(bad_lines)} bad lines out of {i} total")

if bad_lines:
    print("\nFirst few bad lines:")
    for line_num, cols, preview in bad_lines[:5]:
        print(f"Line {line_num}: {cols} columns - {preview}...")

# Write cleaned file
with open(ngt_dynamic_landmarks_clean, 'w') as f:
    f.writelines(good_lines)

print(f"\nCleaned file saved: {ngt_dynamic_landmarks_clean}")
print(f"Removed {len(bad_lines)} corrupted rows")
print(f"Kept {len(good_lines)-1} good rows")

# %%
# Load ONLY dynamic features
dynamic_df = pd.read_csv(ngt_dynamic_landmarks_clean)
dynamic_df = dynamic_df[dynamic_df['letter'].isin(['H', 'J', 'U', 'X', 'Z'])]

X = dynamic_df.drop('letter', axis=1)
y = dynamic_df['letter']

# Normalize
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.2, random_state=42, stratify=y
)

# Train
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(f"Dynamic-only accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred))

# Save
joblib.dump(clf, ngt_dynamic_classifier)
joblib.dump(scaler, ngt_dynamic_scaler)
print("\nSaved dynamic model")

# %%
import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque, Counter

# Load STATIC model 
static_clf = joblib.load(ngt_static_classifier)
static_scaler = None  # We'll load this if it exists

try:
    dynamic_clf = joblib.load(ngt_dynamic_classifier)
    dynamic_scaler = joblib.load(ngt_dynamic_scaler)
except:
    print("Dynamic model not found - train it first!")
    dynamic_clf = None
    dynamic_scaler = None

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

DYNAMIC_LETTERS = ['H', 'J', 'U', 'X', 'Z']
BUFFER_SIZE = 30
CONFIDENCE_THRESHOLD = 0.45


# %%
def normalize_static_landmarks(landmarks):
    """Normalize for static prediction"""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = coords[0]
    coords = coords - wrist
    hand_size = np.linalg.norm(coords[12] - coords[0])
    if hand_size > 0:
        coords = coords / hand_size
    return coords.flatten()

def extract_sequence_features(coords_array):
    """Extract features for dynamic prediction"""
    coords = coords_array.reshape(len(coords_array), 21, 3)
    features = []
    features.extend(coords.mean(axis=0).flatten())
    if len(coords) > 1:
        features.extend(coords.std(axis=0).flatten())
    else:
        features.extend(np.zeros(63))
    if len(coords) > 1:
        features.extend((coords[-1] - coords[0]).flatten())
    else:
        features.extend(np.zeros(63))
    for lm_idx in range(21):
        if len(coords) > 1:
            path = coords[:, lm_idx, :]
            dists = np.linalg.norm(np.diff(path, axis=0), axis=1)
            features.append(dists.sum())
        else:
            features.append(0.0)
    return np.array(features)

# State
frame_buffer = deque(maxlen=BUFFER_SIZE)
static_predictions = deque(maxlen=BUFFER_SIZE)
final_predictions = deque(maxlen=5)

print("NGT Recognition - Hybrid Static/Dynamic")
print("Press 'q' to quit\n")

cap = cv2.VideoCapture(0)

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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                
                # Step 1: Always predict with static model
                normalized_coords = normalize_static_landmarks(hand_landmarks.landmark)
                static_pred = static_clf.predict([normalized_coords])[0]
                static_proba = static_clf.predict_proba([normalized_coords])[0]
                static_conf = static_proba.max()
                
                # Buffer the frame and prediction
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                frame_buffer.append(coords)
                static_predictions.append(static_pred)
                
                # Step 3: Check if static predictions are changing
                if len(static_predictions) >= 10:
                    recent_static = list(static_predictions)[-10:]
                    unique_predictions = len(set(recent_static))
                    
                    # If predictions are unstable AND we have full buffer, try dynamic
                    if unique_predictions > 3 and len(frame_buffer) == BUFFER_SIZE and dynamic_clf:
                        buffer_array = np.array(frame_buffer)
                        dyn_features = extract_sequence_features(buffer_array)
                        dyn_features_norm = dynamic_scaler.transform(dyn_features.reshape(1, -1))
                        
                        dyn_pred = dynamic_clf.predict(dyn_features_norm)[0]
                        dyn_proba = dynamic_clf.predict_proba(dyn_features_norm)[0]
                        dyn_conf = dyn_proba.max()
                        
                        # Use dynamic prediction if confident
                        if dyn_conf >= CONFIDENCE_THRESHOLD:
                            final_predictions.append((dyn_pred, dyn_conf))
                            frame_buffer.clear()
                            static_predictions.clear()
                    
                    # Use static prediction if stable and confident
                    elif unique_predictions <= 2 and static_conf >= CONFIDENCE_THRESHOLD:
                        most_common = Counter(recent_static).most_common(1)[0][0]
                        final_predictions.append((most_common, static_conf))
        else:
            frame_buffer.clear()
            static_predictions.clear()
        
        # Step 4: Display last 5 confident predictions
        if final_predictions:
            # Get most recent prediction
            last_pred, last_conf = final_predictions[-1]
            
            letter_type = "DYNAMIC" if last_pred in DYNAMIC_LETTERS else "STATIC"
            cv2.putText(frame, f"{last_pred}", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
            cv2.putText(frame, f"{letter_type} - {last_conf:.0%}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show history
            history_text = "History: " + " ".join([p[0] for p in final_predictions])
            cv2.putText(frame, history_text, 
                       (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Debug info
        cv2.putText(frame, f"Buffer: {len(frame_buffer)}", 
                   (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('NGT Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# %%
# ! jupytext --to py './static_dynamic.ipynb'
