# 🎭 Emotion Detector (Real-Time Face Emotion Recognition)

A real-time **face emotion detection** project using a webcam.  
The application detects a face from live video and classifies the **dominant facial emotion** such as:

- Happy
- Sad
- Angry
- Fear
- Surprise
- Neutral
- Disgust

This project demonstrates **computer vision**, **deep learning inference**, and **real-time video processing** using Python.

---

## 🚀 Features

- 📷 Real-time webcam emotion detection
- 🧠 Pre-trained deep learning model (mini_XCEPTION)
- 🟩 Face detection + emotion classification
- 🎯 Emotion smoothing across frames to reduce flickering
- 💻 Runs locally (no cloud or API calls)

---

## 🛠️ Tech Stack

- **Python 3**
- **OpenCV (cv2)** – webcam & image processing
- **FER** – facial emotion recognition library
- **mini_XCEPTION** – CNN architecture for emotion classification
- **TensorFlow / Keras** – model backend
- **MTCNN** – face detection
- **NumPy**

---

1. Clone the repository

git clone https://github.com/NomadicJazz/emotion-detector.git
cd emotion-detector

2. Create & activate a virtual environment
python3 -m venv venv
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Real-time Webcam Emotion Detection

python3 src/detect_webcam_miniX.py


