import cv2
import numpy as np
from tensorflow import keras
import time

# Load trained model
model = keras.models.load_model('mask_detector.h5')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Configuration
IMG_SIZE = 128
ALERT_COOLDOWN = 3  # seconds between alerts

# Labels
labels = ['with_mask', 'without_mask']
colors = [(0, 255, 0), (0, 0, 255)]  # Green for mask, Red for no mask

last_alert_time = 0

def detect_and_predict_mask(frame):
    """Detect faces and predict mask status"""
    global last_alert_time
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
        face_roi = face_roi / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)
        
        # Predict
        prediction = model.predict(face_roi, verbose=0)
        label_idx = np.argmax(prediction)
        confidence = prediction[0][label_idx] * 100
        
        label = labels[label_idx]
        color = colors[label_idx]
        
        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        text = f'{label}: {confidence:.1f}%'
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Alert if no mask detected
        if label == 'without_mask':
            current_time = time.time()
            if current_time - last_alert_time > ALERT_COOLDOWN:
                print("⚠️  ALERT: No mask detected!")
                last_alert_time = current_time
                cv2.putText(frame, "ALERT: NO MASK!", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    return frame

def main():
    """Run real-time mask detection"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return
    
    print("Starting face mask detection...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = detect_and_predict_mask(frame)
        
        cv2.imshow('Face Mask Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
