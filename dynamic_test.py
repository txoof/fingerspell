import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

# Load model
model = joblib.load('./models/dynamic_model_clf.pkl')
metadata = joblib.load('./models/dynamic_metadata.pkl')
TARGET_FRAMES = metadata['target_frames']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# State
recording = False
frame_buffer = []
recording_start = None

cap = cv2.VideoCapture(0)

print("="*60)
print("REAL-WORLD DYNAMIC LETTER TEST")
print("="*60)
print("Dynamic letters: H, J, U, X, Z")
print("\nControls:")
print("  SPACE - Start/stop recording")
print("  Q - Quit")
print("\nTest each letter 5 times and track accuracy!")
print("="*60)

# Track results
test_results = []

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                if recording:
                    coords = [coord for lm in hand_landmarks.landmark 
                             for coord in [lm.x, lm.y, lm.z]]
                    frame_buffer.append(coords)
        
        # UI
        if recording:
            cv2.putText(frame, f"🔴 RECORDING: {len(frame_buffer)} frames", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Press SPACE to record gesture", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow('Dynamic Test', frame)
        
        # Auto-stop after 2 seconds
        if recording and len(frame_buffer) > 0:
            elapsed = time.time() - recording_start
            if elapsed > 2.5 or len(frame_buffer) >= 70:
                # Process and predict
                if len(frame_buffer) >= 20:
                    # Normalize each frame
                    normalized = []
                    for coords in frame_buffer:
                        c = np.array(coords).reshape(21, 3)
                        wrist = c[0]
                        c = c - wrist
                        hand_size = np.linalg.norm(c[12])
                        if hand_size > 0:
                            c = c / hand_size
                        normalized.append(c.flatten())
                    
                    normalized = np.array(normalized)
                    
                    # Pad/truncate
                    if len(normalized) < TARGET_FRAMES:
                        padding = np.repeat(normalized[-1:], 
                                          TARGET_FRAMES - len(normalized), axis=0)
                        normalized = np.vstack([normalized, padding])
                    else:
                        normalized = normalized[:TARGET_FRAMES]
                    
                    # Predict
                    features = normalized.flatten()
                    prediction = model.predict([features])[0]
                    probs = model.predict_proba([features])[0]
                    confidence = max(probs) * 100
                    
                    print(f"Predicted: {prediction} (confidence: {confidence:.1f}%)")
                    print(f"Frames captured: {len(frame_buffer)}")
                    
                    # Ask for actual letter
                    actual = input("What letter did you sign? (or press Enter to skip): ").strip().upper()
                    if actual:
                        correct = (actual == prediction)
                        test_results.append({
                            'actual': actual,
                            'predicted': prediction,
                            'correct': correct,
                            'confidence': confidence
                        })
                        print(f"{'✓ CORRECT!' if correct else '✗ WRONG'}\n")
                
                recording = False
                frame_buffer = []
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            if not recording:
                recording = True
                frame_buffer = []
                recording_start = time.time()
                print("\n🔴 Recording gesture...")
            else:
                recording = False
                frame_buffer = []
                print("⏹ Stopped")
        
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Show results
if test_results:
    print("\n" + "="*60)
    print("REAL-WORLD TEST RESULTS")
    print("="*60)
    
    correct = sum(1 for r in test_results if r['correct'])
    total = len(test_results)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.1f}%")
    
    print("\nPer-letter breakdown:")
    from collections import defaultdict
    letter_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for r in test_results:
        letter = r['actual']
        letter_stats[letter]['total'] += 1
        if r['correct']:
            letter_stats[letter]['correct'] += 1
    
    for letter in sorted(letter_stats.keys()):
        stats = letter_stats[letter]
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {letter}: {stats['correct']}/{stats['total']} = {acc:.1f}%")
    
    print("\nConfusion cases:")
    for r in test_results:
        if not r['correct']:
            print(f"  Signed {r['actual']}, predicted {r['predicted']}")