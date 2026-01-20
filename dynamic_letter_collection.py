import cv2
import mediapipe as mp
import csv
from pathlib import Path
import time

# Configuration
DYNAMIC_LETTERS = ['H', 'J', 'U', 'X', 'Z']
SAMPLES_PER_LETTER = 30
MAX_FRAMES = 60  # Max 2 seconds at 30fps
MIN_FRAMES = 20  # Minimum valid gesture

# Setup
csv_path = Path('./data/dataset/ngt_dynamic_landmarks.csv')
csv_path.parent.mkdir(exist_ok=True, parents=True)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create CSV with header
if not csv_path.exists():
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['letter', 'sample_id', 'frame'] + [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']]
        writer.writerow(header)

# State
current_letter_idx = 0
samples_count = 0
recording = False
frame_buffer = []
sample_id = 0
recording_start = None

print("DYNAMIC LETTERS COLLECTION")
print("Controls: SPACE=start/stop recording | N=next letter | Q=quit")
print("Auto-stops after 2 seconds or 60 frames\n")

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
) as hands:
    
    while cap.isOpened() and current_letter_idx < len(DYNAMIC_LETTERS):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw landmarks and capture if recording
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                if recording:
                    coords = [coord for lm in hand_landmarks.landmark for coord in [lm.x, lm.y, lm.z]]
                    frame_buffer.append(coords)
        
        # Display
        letter = DYNAMIC_LETTERS[current_letter_idx]
        if recording:
            cv2.putText(frame, f"RECORDING: {len(frame_buffer)} frames", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"Letter: {letter} | {samples_count}/{SAMPLES_PER_LETTER}", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Dynamic Collection', frame)
        
        # Auto-stop after timeout or max frames
        if recording:
            elapsed = time.time() - recording_start
            if elapsed > 2.0 or len(frame_buffer) >= MAX_FRAMES:
                # Save if valid
                if len(frame_buffer) >= MIN_FRAMES:
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        for i, coords in enumerate(frame_buffer):
                            writer.writerow([letter, sample_id, i] + coords)
                    
                    sample_id += 1
                    samples_count += 1
                    print(f"✓ Saved sample {samples_count}/{SAMPLES_PER_LETTER} for '{letter}' ({len(frame_buffer)} frames)")
                    
                    # Auto-advance when done
                    if samples_count >= SAMPLES_PER_LETTER:
                        print(f"✓ Letter '{letter}' complete!")
                        current_letter_idx += 1
                        samples_count = 0
                else:
                    print(f"⚠ Too short ({len(frame_buffer)} frames), need {MIN_FRAMES}+")
                
                recording = False
                frame_buffer = []
        
        key = cv2.waitKey(10) & 0xFF
        
        if key == ord(' '):
            if not recording:
                recording = True
                frame_buffer = []
                recording_start = time.time()
                print(f"🔴 Recording '{letter}'...")
            else:
                # Manual stop
                if len(frame_buffer) >= MIN_FRAMES:
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        for i, coords in enumerate(frame_buffer):
                            writer.writerow([letter, sample_id, i] + coords)
                    
                    sample_id += 1
                    samples_count += 1
                    print(f"✓ Saved {samples_count}/{SAMPLES_PER_LETTER} for '{letter}' ({len(frame_buffer)} frames)")
                    
                    if samples_count >= SAMPLES_PER_LETTER:
                        current_letter_idx += 1
                        samples_count = 0
                else:
                    print(f"⚠ Cancelled (only {len(frame_buffer)} frames)")
                
                recording = False
                frame_buffer = []
        
        elif key == ord('n'):
            current_letter_idx += 1
            samples_count = 0
            recording = False
            frame_buffer = []
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"\n✓ Data saved to: {csv_path}")