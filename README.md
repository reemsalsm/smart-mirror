# 🪞 AI-Powered Smart Mirror – Health & Safety Assistant

This is a **real-time, AI-enhanced Smart Mirror** project developed for my graduation thesis at Coventry University. The system leverages **computer vision, natural language processing, physiological monitoring, and emergency response** capabilities to offer an intelligent and interactive user experience – all running on edge devices like the **Raspberry Pi 4B**.

---

## 🚀 Features

### ✅ Health Monitoring
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

## 🛠 Setup Instructions

### ⚙️ Prerequisites

- Raspberry Pi 4B (8GB recommended)
- Python 3.9+
- Raspberry Pi OS or Ubuntu Lite
- A camera and microphone connected via USB

### 🔧 Install Dependencies

```bash
pip install -r requirements.txt
