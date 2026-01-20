import cv2
import mediapipe as mp
import csv
from pathlib import Path

# Configuration
STATIC_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'K', 'L', 'M', 
                  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
SAMPLES_PER_LETTER = 50

# Setup
csv_path = Path('./data/dataset/ngt_static_landmarks.csv')
csv_path.parent.mkdir(parents=True, exist_ok=True)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create CSV with header
if not csv_path.exists():
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['letter'] + [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']]
        writer.writerow(header)

# State
current_letter_idx = 0
samples_count = 0
recording = False

print("STATIC LETTERS COLLECTION")
print("Controls: SPACE=toggle recording | N=next letter | Q=quit\n")

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
) as hands:
    
    while cap.isOpened() and current_letter_idx < len(STATIC_LETTERS):
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
                
                # Auto-capture when recording
                if recording:
                    coords = [coord for lm in hand_landmarks.landmark for coord in [lm.x, lm.y, lm.z]]
                    
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([STATIC_LETTERS[current_letter_idx]] + coords)
                    
                    samples_count += 1
                    
                    # Auto-advance when done
                    if samples_count >= SAMPLES_PER_LETTER:
                        print(f"✓ Letter '{STATIC_LETTERS[current_letter_idx]}' complete!")
                        current_letter_idx += 1
                        samples_count = 0
                        recording = False
        
        # Display info
        if current_letter_idx < len(STATIC_LETTERS):
            letter = STATIC_LETTERS[current_letter_idx]
            status = "RECORDING" if recording else "PAUSED"
            cv2.putText(frame, f"Letter: {letter} | {samples_count}/{SAMPLES_PER_LETTER}", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Status: {status}", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if recording else (200, 200, 200), 2)
        
        cv2.imshow('Static Collection', frame)
        
        key = cv2.waitKey(10) & 0xFF
        
        if key == ord(' '):
            recording = not recording
            print(f"{'Started' if recording else 'Paused'} recording '{STATIC_LETTERS[current_letter_idx]}'")
        elif key == ord('n'):
            current_letter_idx += 1
            samples_count = 0
            recording = False
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"\n✓ Data saved to: {csv_path}")