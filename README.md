## 👩‍💻 Author

**Reem Ali Salem**  
Coventry University – BEng Electrical and Electronic Engineering  
Student ID: 202200411  
Graduation Year: 2025  
Project Title: AI-Powered Smart Mirror for Health & Safety Monitoring  
Supervisor: Dr. Mostafa Rabie






# 🪞 AI-Powered Smart Mirror – Health & Safety Assistant

This is a **real-time, AI-enhanced Smart Mirror** project developed for my graduation thesis at Coventry University. The system leverages **computer vision, natural language processing, physiological monitoring, and emergency response** capabilities to offer an intelligent and interactive user experience – all running on edge devices like the **Raspberry Pi 4B**.

---

##  Features

###  Health Monitoring
- **Heart Rate & SpO2** using MAX30102
- Adaptive thresholding for accurate readings
- Real-time UI display with alerts

### 🎙️ Voice Assistant
- **Speech-to-Text** with Whisper (Tiny)
- **NLP** using HuggingFace Zephyr-7b
- **Text-to-Speech** via pyttsx3
- Add/show grocery list, start modules, system queries

### 🧠 Emotion & Face Recognition
- Facial encoding via MediaPipe FaceMesh
- Cosine similarity + PIN for 2FA login
- Emotion classification based on calibrated facial metrics

### 🏋️ Workout Tracker
- Pose Estimation (MediaPipe)
- Rep counting for squats, pushups, curls
- Real-time posture feedback and time tracking

### 🆘 Emergency SOS System
- PIN-protected SOS button
- Twilio API sends alert SMS with a single tap

### 🧴 Skin Analysis
- HSV-based facial skin detection
- Flags dryness, oiliness, and redness levels

---

## 📦 Tech Stack

| Type                | Stack                                 |
|---------------------|----------------------------------------|
| Programming Language | Python 3.x                             |
| UI Framework        | [Kivy](https://kivy.org/)              |
| Computer Vision     | OpenCV, MediaPipe                      |
| NLP & STT/TTS       | Whisper, HuggingFace Zephyr-7b, pyttsx3|
| Hardware            | Raspberry Pi 4B, MAX30102, USB camera  |
| APIs                | Twilio, HuggingFace                    |

---

## 📊 Performance

| Feature           | Accuracy |
|-------------------|----------|
| Heart Rate Alerts | ~98%     |
| Emotion Detection | ~89%     |
| Face Auth         | ~92%     |
| Voice Assistant   | ~95%     |
| Workout Tracking  | ~90%     |

## 🧠 Future Work

- 🧴 **CNN-Based Skin Analysis**: Replace HSV with deep learning for precision skincare detection.
- 🌐 **Multilingual NLP**: Add Arabic + French support for Whisper and Zephyr.
- 🥽 **AR Overlays**: Feedback layers for workout correction and accessibility.
- ☁️ **Cloud Backup + Mobile App**: Log vitals remotely and sync with iOS/Android.
- 📡 **Offline SOS**: Integrate SIM800L for emergency SMS without internet.
- 🔒 **Multi-User Roles & AES Encryption**: Admin/Guest login and encrypted credentials.


## 🛠 Setup Instructions

### ⚙️ Prerequisites

- Raspberry Pi 4B (8GB recommended)
- Python 3.9+
- Raspberry Pi OS or Ubuntu Lite
- A camera and microphone connected via USB

### 🔧 Install Dependencies

```bash
pip install -r requirements.txt


