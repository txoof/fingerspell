# Quick test script
import cv2
import mediapipe as mp
import joblib
import numpy as np

model = joblib.load('./models/static_model_clf.pkl')
mp_hands = mp.solutions.hands.Hands()

cap = cv2.VideoCapture(0)

print("Show different letters and see if model is correct!")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    results = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        # Extract and normalize
        coords = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
        wrist = coords[0]
        coords = coords - wrist
        hand_size = np.linalg.norm(coords[12])
        if hand_size > 0:
            coords = coords / hand_size
        
        prediction = model.predict([coords.flatten()])[0]
        cv2.putText(frame, f"Letter: {prediction}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    cv2.imshow('Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()