# 😷 Face Mask Detection System

## 📌 Project Title
Face Mask Detection Using Deep Learning

---

## 📖 Project Description

The Face Mask Detection System is a deep learning-based project that detects whether a person is wearing a face mask or not.

This system uses:
- Image Processing
- Convolutional Neural Network (CNN)
- OpenCV for real-time detection
- Deep Learning model for classification

The model classifies images into two categories:
- With Mask
- Without Mask

---

## 🎯 Objectives

- Detect faces in images or video.
- Classify whether the detected face is wearing a mask or not.
- Provide real-time mask detection using webcam.
- Help in safety monitoring during pandemic situations.

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

---

## 📂 Project Structure

Face_Mask_Detection/
│
├── dataset/
│   ├── with_mask/
│   └── without_mask/
│
├── model/
│   └── mask_detector.model
│
├── train_mask_detector.py
├── detect_mask_video.py
├── detect_mask_image.py
└── README.md

---

## ⚙️ Installation Steps

1. Install Required Libraries

pip install tensorflow opencv-python numpy matplotlib

2. Train the Model

python train_mask_detector.py

This will:
- Load dataset
- Train CNN model
- Save the trained model file

3. Run Real-Time Detection

python detect_mask_video.py

This will:
- Start webcam
- Detect faces
- Show Mask / No Mask label

---

## 🧠 Model Details

- CNN (Convolutional Neural Network)
- Image resizing and normalization
- Binary classification (Mask / No Mask)
- Sigmoid activation function

---

## 📊 Output

- Green box → With Mask
- Red box → Without Mask
- Confidence percentage displayed on screen

---

## 🚀 Future Enhancements

- Add alarm system for no mask detection
- Deploy as web application
- Improve accuracy using larger dataset
- Add social distancing detection

---

## 📌 Conclusion

The Face Mask Detection System successfully detects whether a person is wearing a mask or not using deep learning techniques. It can be used in public places, offices, hospitals, and schools for safety monitoring.
