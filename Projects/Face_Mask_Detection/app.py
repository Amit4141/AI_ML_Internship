from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow import keras

app = Flask(__name__)

# Load model and cascade
model = keras.models.load_model('mask_detector.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

IMG_SIZE = 128
labels = ['with_mask', 'without_mask']
colors = [(0, 255, 0), (0, 0, 255)]

def generate_frames():
    """Generate video frames with mask detection"""
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
            face_roi = face_roi / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            
            prediction = model.predict(face_roi, verbose=0)
            label_idx = np.argmax(prediction)
            confidence = prediction[0][label_idx] * 100
            
            label = labels[label_idx]
            color = colors[label_idx]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            text = f'{label}: {confidence:.1f}%'
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
