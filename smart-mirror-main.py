
import os
import json
import threading
import numpy as np
import sounddevice as sd
import requests
import pyttsx3
import cv2
import mediapipe as mp
import pygame
import time
from collections import deque
from datetime import datetime
import pytz
from twilio.rest import Client
import smbus
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.uix.modalview import ModalView
from kivy.uix.gridlayout import GridLayout
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from kivy.properties import StringProperty, BooleanProperty, ObjectProperty
from dotenv import load_dotenv
from faster_whisper import WhisperModel

# Initialize mediapipe with updated settings
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# Initialize pygame mixer with error handling
try:
    pygame.mixer.init()
    pygame.mixer.pre_init(44100, -16, 2, 2048)
except Exception as e:
    print(f"Audio initialization error: {e}")

# Configuration (remove sensitive data before sharing)
TWILIO_SID = "*******************"
TWILIO_AUTH_TOKEN = "******************"
TWILIO_PHONE = "+18312735198"
EMERGENCY_CONTACT = "+201013577939"
SOS_PASSWORD = "1234"

class HeartRateMonitor:
    def __init__(self, bus_number=1):
        # MAX30102 Registers
        self.MAX30102_ADDR = 0x57
        self.REG_INTR_STATUS_1 = 0x00
        self.REG_INTR_ENABLE_1 = 0x02
        self.REG_FIFO_WR_PTR = 0x04
        self.REG_FIFO_RD_PTR = 0x06
        self.REG_FIFO_DATA = 0x07
        self.REG_MODE_CONFIG = 0x09
        self.REG_SPO2_CONFIG = 0x0A
        self.REG_LED1_PA = 0x0C
        self.REG_LED2_PA = 0x0D
        self.REG_PILOT_PA = 0x10
        
        self.bus = smbus.SMBus(bus_number)
        self.ir_history = deque(maxlen=25)  # Increased buffer size for better baseline
        self.red_history = deque(maxlen=25)
        self.beat_times = deque(maxlen=5)   # Smaller buffer for faster response
        self.last_beat_time = 0
        self.bpm = 0
        self.initialized = False
        
        self.setup_sensor()
    
    def setup_sensor(self):
        try:
            # Reset sensor
            self.bus.write_byte_data(self.MAX30102_ADDR, self.REG_MODE_CONFIG, 0x40)
            time.sleep(0.1)
            
            # Configuration for fast response
            self.bus.write_byte_data(self.MAX30102_ADDR, self.REG_SPO2_CONFIG, 0x27)  # 400Hz, 16-bit
            self.bus.write_byte_data(self.MAX30102_ADDR, self.REG_LED1_PA, 0x3F)      # Max LED current
            self.bus.write_byte_data(self.MAX30102_ADDR, self.REG_LED2_PA, 0x3F)
            self.bus.write_byte_data(self.MAX30102_ADDR, self.REG_MODE_CONFIG, 0x03)  # HR mode
            
            self.initialized = True
            print("Sensor initialized for fast response")
        except Exception as e:
            print(f"Initialization failed: {e}")
            self.initialized = False
    
    def read_fifo(self):
        try:
            # Read 6 bytes of data (3 bytes each for red and IR)
            data = self.bus.read_i2c_block_data(self.MAX30102_ADDR, self.REG_FIFO_DATA, 6)
            red = (data[0] << 16) | (data[1] << 8) | data[2]
            ir = (data[3] << 16) | (data[4] << 8) | data[5]
            return red, ir
        except Exception as e:
            print(f"Read error: {e}")
            return 0, 0
    
    def update(self):
        if not self.initialized:
            return None
        
        red, ir = self.read_fifo()
        self.ir_history.append(ir)
        
        # Dynamic threshold with fast adaptation
        threshold = np.percentile(list(self.ir_history)[-10:], 80) * 1.1 if len(self.ir_history) > 10 else 30000
        
        # Detect pulse peak
        if ir > threshold and (time.time() - self.last_beat_time) > 0.25:  # 240 BPM max
            current_time = time.time()
            
            if self.last_beat_time > 0:
                beat_period = current_time - self.last_beat_time
                current_bpm = int(60 / beat_period)
                
                # Validate physiologically plausible range
                if 40 <= current_bpm <= 200:
                    self.beat_times.append(current_bpm)
                    self.bpm = int(np.mean(self.beat_times)) if len(self.beat_times) > 0 else current_bpm
            
            self.last_beat_time = current_time
            return self.bpm
        
        return None

# Voice Assistant Config
load_dotenv()

class VoiceConfig:
    HF_API_KEY = os.getenv("HF_API_KEY", "")
    HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    WHISPER_MODEL = "tiny"
    SAMPLE_RATE = 16000
    TTS_VOICE = "english"
    GROCERY_FILE = "grocery_list.txt"

def calculate_angle(a, b, c):
    """Calculate the angle between three points with improved accuracy."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    return angle if angle <= 180 else 360-angle

class GroceryManager:
    def __init__(self):
        self.file_path = VoiceConfig.GROCERY_FILE
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                json.dump([], f)
        
    def get_items(self):
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except:
            return []
    
    def add_item(self, item):
        items = self.get_items()
        if item.lower() not in [i.lower() for i in items]:
            items.append(item)
            self._save(items)
            return True
        return False
    
    def _save(self, items):
        with open(self.file_path, 'w') as f:
            json.dump(items, f)

class VoiceAssistant:
    def __init__(self):
        self.stt_model = WhisperModel(VoiceConfig.WHISPER_MODEL, device="cpu", compute_type="int8")
        self.grocery = GroceryManager()

        try:
            self.tts_engine = pyttsx3.init()
            voices = self.tts_engine.getProperty('voices')
            if len(voices) > 0:
                self.tts_engine.setProperty('voice', voices[0].id)
            self.tts_engine.setProperty('rate', 150)
        except Exception as e:
            print(f"TTS Error: {e}")
            self.tts_engine = None

        self.is_listening = False
        self.audio_buffer = np.array([])
        
    def record_callback(self, indata, frames, time, status):
        if self.is_listening:
            self.audio_buffer = np.append(self.audio_buffer, indata.copy())
    
    def listen(self, timeout=5):
        self.is_listening = True
        self.audio_buffer = np.array([], dtype=np.float32)

        with sd.InputStream(
            callback=self.record_callback,
            channels=1,
            samplerate=VoiceConfig.SAMPLE_RATE
        ):
            start_time = time.time()
            while (time.time() - start_time < timeout) and self.is_listening:
                time.sleep(0.1)

        self.is_listening = False
        if len(self.audio_buffer) > 0:
            try:
                segments, _ = self.stt_model.transcribe(self.audio_buffer)
                return " ".join(segment.text for segment in segments).strip()
            except Exception as e:
                print(f"STT Error: {e}")
                return ""
        return ""
    
    def speak(self, text):
        if self.tts_engine:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"Speech Error: {e}")
    
    def query_huggingface(self, payload):
        headers = {"Authorization": f"Bearer {VoiceConfig.HF_API_KEY}"} if VoiceConfig.HF_API_KEY else {}
        try:
            response = requests.post(VoiceConfig.HF_API_URL, headers=headers, json=payload)
            if response.status_code == 401:
                return {"error": "Invalid Hugging Face API token"}
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def process_command(self, query):
        query = query.lower()

        # Grocery List Management
        if "add to grocery list" in query:
            item = query.replace("add to grocery list", "").strip()
            if self.grocery.add_item(item):
                return f"Added {item} to your grocery list."
            return f"{item} is already on the list."

        elif "show grocery list" in query:
            items = self.grocery.get_items()
            return "Grocery List: " + ", ".join(items) if items else "Your list is empty"

        # Hugging Face API
        try:
            output = self.query_huggingface({
                "inputs": f"<s>[INST] {query} [/INST]",
                "parameters": {"max_new_tokens": 100}
            })

            if isinstance(output, dict) and 'error' in output:
                return output['error']

            if isinstance(output, list) and len(output) > 0 and 'generated_text' in output[0]:
                response = output[0]['generated_text'].split('[/INST]')[-1].strip()
                return response if response else "I didn't get a response from the AI"

            return "I couldn't understand the AI service response."
        except Exception as e:
            return f"Error processing your request: {str(e)}"

Builder.load_string('''
<MainScreen>:
    orientation: 'vertical'
    padding: 20
    spacing: 15
    canvas.before:
        Color:
            rgba: 0.95, 0.95, 0.97, 1
        Rectangle:
            pos: self.pos
            size: self.size
    
    BoxLayout:
        size_hint_y: 0.2
        Label:
            id: clock
            text: "Loading..."
            font_size: '30sp'
            halign: 'center'
            color: 0.2, 0.2, 0.2, 1
    
    BoxLayout:
        size_hint_y: 0.3
        Label:
            id: sensor_status
            text: "Sensor: Initializing..."
            font_size: '24sp'
            color: 0.2, 0.2, 0.2, 1

        Label:
            id: status
            text: "Status: Normal"
            font_size: '24sp'
            color: 0.2, 0.2, 0.2, 1

    
    BoxLayout:
        size_hint_y: 0.2
        Label:
            id: heart_rate
            text: "Heart Rate: -- BPM"
            font_size: '24sp'
            color: 0.2, 0.2, 0.2, 1

        Button:
            id: sos_btn
            text: "🚨 SOS"
            font_size: '24sp'
            background_color: 0.9, 0.3, 0.3, 1
            on_release: root.show_keypad()
    
    GridLayout:
        id: keypad
        cols: 3
        rows: 4
        size_hint_y: 0.3
        opacity: 0
        spacing: 5
        padding: 5
        canvas.before:
            Color:
                rgba: 0.85, 0.85, 0.9, 1
            Rectangle:
                pos: self.pos
                size: self.size

        Button:
            text: "1"
            background_color: 1, 1, 1, 1
            color: 0.2, 0.2, 0.2, 1
            on_release: root.append_password("1")
        Button:
            text: "2"
            background_color: 1, 1, 1, 1
            color: 0.2, 0.2, 0.2, 1
            on_release: root.append_password("2")
        Button:
            text: "3"
            background_color: 1, 1, 1, 1
            color: 0.2, 0.2, 0.2, 1
            on_release: root.append_password("3")
        Button:
            text: "4"
            background_color: 1, 1, 1, 1
            color: 0.2, 0.2, 0.2, 1
            on_release: root.append_password("4")
        Button:
            text: "5"
            background_color: 1, 1, 1, 1
            color: 0.2, 0.2, 0.2, 1
            on_release: root.append_password("5")
        Button:
            text: "6"
            background_color: 1, 1, 1, 1
            color: 0.2, 0.2, 0.2, 1
            on_release: root.append_password("6")
        Button:
            text: "7"
            background_color: 1, 1, 1, 1
            color: 0.2, 0.2, 0.2, 1
            on_release: root.append_password("7")
        Button:
            text: "8"
            background_color: 1, 1, 1, 1
            color: 0.2, 0.2, 0.2, 1
            on_release: root.append_password("8")
        Button:
            text: "9"
            background_color: 1, 1, 1, 1
            color: 0.2, 0.2, 0.2, 1
            on_release: root.append_password("9")
        Button:
            text: "0"
            background_color: 1, 1, 1, 1
            color: 0.2, 0.2, 0.2, 1
            on_release: root.append_password("0")
        Button:
            text: "Clear"
            background_color: 0.8, 0.8, 0.8, 1
            color: 0.2, 0.2, 0.2, 1
            on_release: root.clear_password()
        Button:
            text: "Submit"
            background_color: 0.5, 0.8, 0.5, 1
            color: 0.2, 0.2, 0.2, 1
            on_release: root.verify_sos()
    
    BoxLayout:
        size_hint_y: 0.2
        spacing: 5
        Button:
            text: "Start Workout"
            font_size: '20sp'
            background_color: 0.4, 0.6, 0.8, 1
            color: 1, 1, 1, 1
            on_release: root.start_workout_mode()
        Button:
            text: "Skin Analysis"
            font_size: '20sp'
            background_color: 0.8, 0.6, 0.4, 1
            color: 1, 1, 1, 1
            on_release: root.start_skin_analysis_mode()
        Button:
            text: "Emotion Detection"
            font_size: '20sp'
            background_color: 0.6, 0.4, 0.8, 1
            color: 1, 1, 1, 1
            on_release: root.start_emotion_detection_mode()

        Button:
            text: "Face Auth"
            font_size: '20sp'
            background_color: 0.6, 0.4, 0.6, 1
            color: 1, 1, 1, 1
            on_release: root.start_face_auth_mode()
    
    BoxLayout:
        size_hint_y: 0.2
        Button:
            id: voice_btn
            text: "Voice Assistant"
            font_size: '20sp'
            background_color: 0.4, 0.8, 0.6, 1
            color: 1, 1, 1, 1
            on_release: root.toggle_voice_assistant()
        Button:
            text: "Exit"
            font_size: '20sp'
            background_color: 0.8, 0.4, 0.4, 1
            color: 1, 1, 1, 1
            on_release: App.get_running_app().stop()
    
    Label:
        id: voice_status
        text: ""
        font_size: '18sp'
        color: 0.2, 0.2, 0.2, 1
        size_hint_y: 0.1
    
    Label:
        id: voice_response
        text: ""
        font_size: '16sp'
        color: 0.2, 0.2, 0.2, 1
        size_hint_y: 0.2

<FaceAuthScreen>:
    orientation: 'vertical'
    padding: 10
    spacing: 10
    canvas.before:
        Color:
            rgba: 0.95, 0.95, 0.97, 1
        Rectangle:
            pos: self.pos
            size: self.size
    
    BoxLayout:
        size_hint_y: 0.6
        Image:
            id: camera_feed
    
    BoxLayout:
        size_hint_y: 0.1
        Label:
            id: status_label
            text: "Status: Ready"
            font_size: '20sp'
            halign: 'center'
            color: 0.2, 0.2, 0.2, 1
    
    BoxLayout:
        size_hint_y: 0.1
        TextInput:
            id: name_input
            hint_text: "Enter name"
            font_size: '20sp'
            size_hint_x: 0.6
            background_color: 1, 1, 1, 1
    
    BoxLayout:
        size_hint_y: 0.1
        TextInput:
            id: password_input
            hint_text: "Enter PIN"
            font_size: '20sp'
            password: True
            input_filter: 'int'
            size_hint_x: 0.6
            background_color: 1, 1, 1, 1
        Button:
            text: "⌫"
            font_size: '20sp'
            size_hint_x: 0.2
            background_color: 0.8, 0.8, 0.8, 1
            on_press: root.clear_password()
    
    GridLayout:
        id: keypad
        cols: 3
        rows: 4
        size_hint_y: 0.2
        spacing: 5
        canvas.before:
            Color:
                rgba: 0.85, 0.85, 0.9, 1
            Rectangle:
                pos: self.pos
                size: self.size

        Button:
            text: "1"
            font_size: '20sp'
            background_color: 1, 1, 1, 1
            on_press: root.append_password('1')
        Button:
            text: "2"
            font_size: '20sp'
            background_color: 1, 1, 1, 1
            on_press: root.append_password('2')
        Button:
            text: "3"
            font_size: '20sp'
            background_color: 1, 1, 1, 1
            on_press: root.append_password('3')
        Button:
            text: "4"
            font_size: '20sp'
            background_color: 1, 1, 1, 1
            on_press: root.append_password('4')
        Button:
            text: "5"
            font_size: '20sp'
            background_color: 1, 1, 1, 1
            on_press: root.append_password('5')
        Button:
            text: "6"
            font_size: '20sp'
            background_color: 1, 1, 1, 1
            on_press: root.append_password('6')
        Button:
            text: "7"
            font_size: '20sp'
            background_color: 1, 1, 1, 1
            on_press: root.append_password('7')
        Button:
            text: "8"
            font_size: '20sp'
            background_color: 1, 1, 1, 1
            on_press: root.append_password('8')
        Button:
            text: "9"
            font_size: '20sp'
            background_color: 1, 1, 1, 1
            on_press: root.append_password('9')
        Button:
            text: "Clear"
            font_size: '20sp'
            background_color: 0.8, 0.8, 0.8, 1
            on_press: root.clear_password()
        Button:
            text: "0"
            font_size: '20sp'
            background_color: 1, 1, 1, 1
            on_press: root.append_password('0')
        Button:
            text: "Auth"
            font_size: '20sp'
            background_color: 0.5, 0.8, 0.5, 1
            on_press: root.authenticate()
    
    BoxLayout:
        size_hint_y: 0.1
        Button:
            text: "Register"
            font_size: '20sp'
            background_color: 0.4, 0.6, 0.8, 1
            on_press: root.register_face()
        Button:
            text: "Exit"
            font_size: '20sp'
            background_color: 0.8, 0.4, 0.4, 1
            on_press: root.exit_face_auth()

<ExerciseSelectionDialog@Popup>:
    title: "Select Workout"
    size_hint: 0.8, 0.8
    BoxLayout:
        orientation: 'vertical'
        padding: 20
        spacing: 15
        Label:
            text: "Choose your workout:"
            font_size: '24sp'
            size_hint_y: 0.2
        GridLayout:
            cols: 1
            spacing: 10
            size_hint_y: 0.8
            Button:
                text: "Bicep Curls"
                font_size: '20sp'
                background_color: 0.4, 0.6, 0.8, 1
                on_press: root.dismiss(); app.start_exercise("Bicep Curls")
            Button:
                text: "Squats"
                font_size: '20sp'
                background_color: 0.8, 0.6, 0.4, 1
                on_press: root.dismiss(); app.start_exercise("Squats")
            Button:
                text: "Pushups"
                font_size: '20sp'
                background_color: 0.6, 0.4, 0.8, 1
                on_press: root.dismiss(); app.start_exercise("Pushups")
            Button:
                text: "Cancel"
                font_size: '20sp'
                background_color: 0.8, 0.4, 0.4, 1
                on_press: root.dismiss()
''')

class MainScreen(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.password_input = ""
        self.current_mode = None
        self.cap = None
        self.voice_assistant = VoiceAssistant()
        self.voice_listening = False
        self.clock_event = None
        self.heart_rate_event = None

        # Initialize heart rate monitor with improved logic
        self.heart_rate_monitor = HeartRateMonitor()
        if self.heart_rate_monitor.initialized:
            self.ids.sensor_status.text = "Sensor: Ready"
            self.heart_rate_event = Clock.schedule_interval(self.update_heart_rate, 0.1)  # 100ms updates
        else:
            self.ids.sensor_status.text = "Sensor: Not Available"

        self.start_clock()
    
    def start_clock(self):
        if self.clock_event is not None:
            Clock.unschedule(self.clock_event)
        self.clock_event = Clock.schedule_interval(self.update_clock, 1)
    
    def update_clock(self, dt):
        try:
            cairo_tz = pytz.timezone('Africa/Cairo')
            now = datetime.now(cairo_tz)
            self.ids.clock.text = now.strftime("%H:%M:%S\n%A, %d %B %Y")
        except Exception as e:
            print(f"Clock update error: {e}")
    

    def update_heart_rate(self, dt):
        bpm = self.heart_rate_monitor.update()

        if bpm is not None:
            # Only update if we don't already have a valid reading
            if not hasattr(self, 'last_hr_update') or time.time() - self.last_hr_update > 5:  # 5 seconds between updates
                self.ids.heart_rate.text = f"Heart Rate: {bpm} BPM"
                self.last_hr_update = time.time()
                
                # Schedule clearing the display after 3 seconds
                Clock.schedule_once(lambda dt: self.clear_hr_display(), 3)

                # Update status based on heart rate
                if bpm < 60:
                    self.ids.status.text = "Status: Low Heart Rate"
                    self.ids.status.color = (0, 0, 1, 1)  # Blue
                elif bpm > 100:
                    self.ids.status.text = "Status: High Heart Rate"
                    self.ids.status.color = (1, 0, 0, 1)  # Red
                else:
                    self.ids.status.text = "Status: Normal"
                    self.ids.status.color = (0, 0.7, 0, 1)  # Green
        else:
            # Only show this if we don't have a recent reading
            if not hasattr(self, 'last_hr_update') or time.time() - self.last_hr_update > 5:
                self.ids.heart_rate.text = "Heart Rate: -- BPM"
                self.ids.status.text = "Status: Place finger on sensor"
                self.ids.status.color = (0.2, 0.2, 0.2, 1)

    def clear_hr_display(self):
        """Clear the heart rate display after the timeout period"""
        self.ids.heart_rate.text = "Heart Rate: -- BPM"
        self.ids.status.text = "Status: Waiting for reading..."
        self.ids.status.color = (0.2, 0.2, 0.2, 1)


    def clear_hr_display(self):
        """Clear the heart rate display after the timeout period"""
        self.ids.heart_rate.text = "Heart Rate: -- BPM"
        self.ids.status.text = "Status: Waiting for reading..."
        self.ids.status.color = (0.2, 0.2, 0.2, 1)
    
    def show_keypad(self):
        self.ids.keypad.opacity = 1
    
    def append_password(self, digit):
        self.password_input += digit
    
    def clear_password(self):
        self.password_input = ""
    
    def verify_sos(self):
        if self.password_input == SOS_PASSWORD:
            self.send_sos()
        else:
            self.ids.status.text = "Wrong Password!"
            self.ids.status.color = (1, 0, 0, 1)
            Clock.schedule_once(lambda dt: setattr(self.ids.status, 'color', (0.2, 0.2, 0.2, 1)), 2)

        self.password_input = ""
        self.ids.keypad.opacity = 0
    
    def send_sos(self):
        try:
            client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
            client.messages.create(
                body="🚨 EMERGENCY: Smart Mirror User Needs Help!",
                from_=TWILIO_PHONE,
                to=EMERGENCY_CONTACT
            )
            self.ids.status.text = "SOS Sent!"
            self.ids.status.color = (0, 0.7, 0, 1)
        except Exception as e:
            self.ids.status.text = f"Error: {str(e)}"
            self.ids.status.color = (1, 0, 0, 1)
    
    def toggle_voice_assistant(self):
        if not self.voice_listening:
            self.voice_listening = True
            self.ids.voice_btn.text = "Listening..."
            self.ids.voice_btn.background_color = (0.8, 0.2, 0.2, 1)
            self.ids.voice_status.text = "Listening..."
            threading.Thread(target=self.process_voice_command, daemon=True).start()
        else:
            self.voice_listening = False
            self.ids.voice_btn.text = "Voice Assistant"
            self.ids.voice_btn.background_color = (0.4, 0.8, 0.6, 1)
            self.ids.voice_status.text = "Voice assistant ready"
    
    def process_voice_command(self):
        while self.voice_listening:
            query = self.voice_assistant.listen()

            if not query:
                continue

            Clock.schedule_once(lambda dt: setattr(self.ids.voice_status, 'text', f"You said: {query}"))
            response = self.voice_assistant.process_command(query)
            Clock.schedule_once(lambda dt: setattr(self.ids.voice_response, 'text', response))
            self.voice_assistant.speak(response)

            if "workout" in query.lower():
                Clock.schedule_once(lambda dt: self.start_workout_mode())
            elif "skin" in query.lower():
                Clock.schedule_once(lambda dt: self.start_skin_analysis_mode())
            elif "emotion" in query.lower():
                Clock.schedule_once(lambda dt: self.start_emotion_detection_mode())
            elif "face auth" in query.lower():
                Clock.schedule_once(lambda dt: self.start_face_auth_mode())
            elif "exit" in query.lower():
                Clock.schedule_once(lambda dt: self.toggle_voice_assistant())

        Clock.schedule_once(lambda dt: setattr(self.ids.voice_status, 'text', ""))
    
    def start_workout_mode(self):
        self.cleanup_camera()
        self.current_mode = "workout"
        app = App.get_running_app()
        app.show_exercise_selection()
    
    def start_skin_analysis_mode(self):
        self.cleanup_camera()
        self.current_mode = "skin"
        app = App.get_running_app()
        app.show_skin_analysis_screen()
    
    def start_emotion_detection_mode(self):
        self.cleanup_camera()
        self.current_mode = "emotion"
        app = App.get_running_app()
        app.show_emotion_detection_screen()
    
    def start_face_auth_mode(self):
        self.cleanup_camera()
        self.current_mode = "face_auth"
        app = App.get_running_app()
        app.show_face_auth_screen()
    
    def cleanup_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def cleanup(self):
        if self.clock_event is not None:
            Clock.unschedule(self.clock_event)
        if self.heart_rate_event is not None:
            Clock.unschedule(self.heart_rate_event)
        self.cleanup_camera()
        self.voice_listening = False

class FaceAuthScreen(BoxLayout):
    def __init__(self, main_app, **kwargs):
        super().__init__(**kwargs)
        self.main_app = main_app
        self.capture = None
        self.initialize_camera()
        # Improved face mesh configuration
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )

        self.known_faces = {}
        self.current_encoding = None
        self.data_file = "face_data.json"
        self.DEFAULT_PASSWORD = "1234"
        self.RECOGNITION_THRESHOLD = 0.85
        self.key_landmarks = [10, 33, 152, 133, 362, 168, 397, 4, 164, 61, 291]
        self.calibration_samples = 10
        self.calibration_delay = 0.3

        self.load_face_data()
        if self.capture is not None:
            Clock.schedule_interval(self.update_camera, 1.0/30.0)
    
    def initialize_camera(self):
        """Initialize camera with multiple attempts and error handling"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.capture = cv2.VideoCapture(0)
                if self.capture.isOpened():
                    self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.ids.status_label.text = "Camera: Ready"
                    return
                else:
                    self.capture.release()
            except Exception as e:
                print(f"Camera initialization attempt {attempt + 1} failed: {e}")
                if hasattr(self, 'capture') and self.capture is not None:
                    self.capture.release()

        self.ids.status_label.text = "Camera: Not Available"

    def play_sound(self, sound_type):
        try:
            if sound_type == "success":
                sound = pygame.mixer.Sound("success.wav")
            elif sound_type == "error":
                sound = pygame.mixer.Sound("error.wav")
            elif sound_type == "welcome":
                sound = pygame.mixer.Sound("welcome.wav")
            sound.play()
        except Exception as e:
            print(f"Audio error: {e}")

    def load_face_data(self):
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.known_faces = {}
                    for name, face_data in data.items():
                        encoding = np.array(face_data['encoding'])
                        if len(encoding) > 0:
                            self.known_faces[name] = {
                                'encoding': encoding,
                                'password': face_data['password']
                            }
        except Exception as e:
            print(f"Error loading data: {e}")
            self.known_faces = {}

    def save_face_data(self):
        try:
            save_data = {}
            for name in self.known_faces:
                save_data[name] = {
                    'encoding': self.known_faces[name]['encoding'].tolist(),
                    'password': self.known_faces[name]['password']
                }
            with open(self.data_file, 'w') as f:
                json.dump(save_data, f, indent=4)
        except Exception as e:
            print(f"Error saving data: {e}")

    def get_face_encoding(self, frame):
        # Flip frame horizontally for mirror view before processing
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            encoding = []

            for idx in self.key_landmarks:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    encoding.extend([lm.x, lm.y, lm.z])

            if len(landmarks) > 300:
                left_eye = landmarks[33]
                right_eye = landmarks[263]
                eye_dist = np.sqrt((right_eye.x-left_eye.x)**2 + (right_eye.y-left_eye.y)**2)
                encoding.append(eye_dist)

                left_mouth = landmarks[61]
                right_mouth = landmarks[291]
                mouth_width = np.sqrt((right_mouth.x-left_mouth.x)**2 + (right_mouth.y-left_mouth.y)**2)
                encoding.append(mouth_width)

            if len(encoding) > 0:
                return np.array(encoding)
        return None

    def compare_faces(self, encoding1, encoding2):
        if encoding1 is None or encoding2 is None:
            return 0.0

        # Normalize encodings
        encoding1 = encoding1 / np.linalg.norm(encoding1)
        encoding2 = encoding2 / np.linalg.norm(encoding2)

        # Use cosine similarity
        similarity = np.dot(encoding1, encoding2)

        # Apply sigmoid to get a score between 0 and 1
        return 1 / (1 + np.exp(-15*(similarity-0.88)))

    def update_camera(self, dt):
        if self.capture is None:
            return

        try:
            ret, frame = self.capture.read()
            if not ret:
                print("Failed to capture frame")
                self.initialize_camera()  # Try to reinitialize camera
                return

            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)

            # Get face encoding for current frame
            self.current_encoding = self.get_face_encoding(frame)

            if self.current_encoding is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        results.multi_face_landmarks[0],
                        mp.solutions.face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles
                        .get_default_face_mesh_contours_style()
                    )

            # Convert to texture for display
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.ids.camera_feed.texture = texture
        except Exception as e:
            print(f"Camera error: {e}")
            self.ids.status_label.text = "Camera Error"

    def append_password(self, digit):
        current = self.ids.password_input.text
        if len(current) < 4:
            self.ids.password_input.text += digit
            self.play_sound("success")

    def clear_password(self):
        self.ids.password_input.text = ""

    def register_face(self):
        if self.current_encoding is None:
            self.show_popup("Error", "No face detected!")
            self.play_sound("error")
            return

        name = self.ids.name_input.text.strip()
        password = self.ids.password_input.text or self.DEFAULT_PASSWORD

        if not name:
            self.show_popup("Error", "Name is required!")
            self.play_sound("error")
            return

        if len(password) != 4 or not password.isdigit():
            self.show_popup("Error", "PIN must be 4 digits!")
            self.play_sound("error")
            return

        if name in self.known_faces:
            self.show_popup("Error", "Name already registered!")
            self.play_sound("error")
            return

        encodings = []
        self.ids.status_label.text = "Calibrating... Look straight ahead"

        for i in range(self.calibration_samples):
            ret, frame = self.capture.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Mirror view for processing
                encoding = self.get_face_encoding(frame)
                if encoding is not None and len(encoding) > 0:
                    encodings.append(encoding)
                    self.ids.status_label.text = f"Calibrating... {i+1}/{self.calibration_samples}"
                    time.sleep(self.calibration_delay)

        if len(encodings) < self.calibration_samples:
            self.show_popup("Error", "Could not capture enough face samples!")
            self.play_sound("error")
            return

        # Weight later samples more heavily
        weights = np.linspace(0.5, 1.5, len(encodings))
        avg_encoding = np.average(encodings, axis=0, weights=weights)

        self.known_faces[name] = {
            'encoding': avg_encoding,
            'password': password
        }
        self.save_face_data()
        self.ids.status_label.text = f"Registered: {name}"
        self.show_popup("Success", f"Face registered!\nPIN: {password}")
        self.play_sound("success")
        self.clear_password()
        self.ids.name_input.text = ""

    def authenticate(self):
        if self.current_encoding is None or len(self.current_encoding) < 10:
            self.show_popup("Error", "No face detected or bad detection!")
            self.play_sound("error")
            return

        password = self.ids.password_input.text or self.DEFAULT_PASSWORD
        best_match = None
        best_similarity = 0

        # Take multiple verification samples
        verification_samples = []
        for _ in range(3):
            ret, frame = self.capture.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Mirror view for processing
                encoding = self.get_face_encoding(frame)
                if encoding is not None:
                    verification_samples.append(encoding)
            time.sleep(0.1)

        if not verification_samples:
            self.show_popup("Error", "Could not verify face!")
            return

        # Use average of verification samples
        avg_verification = np.mean(verification_samples, axis=0)

        # Compare with known faces
        for name, data in self.known_faces.items():
            if len(data['encoding']) != len(avg_verification):
                continue

            similarity = self.compare_faces(data['encoding'], avg_verification)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name

        # Check if authentication succeeds
        if (best_similarity > self.RECOGNITION_THRESHOLD and 
            best_match is not None and 
            self.known_faces[best_match]['password'] == password):
            self.ids.status_label.text = f"Welcome {best_match}!"
            self.show_popup("Success", f"Authentication successful!\nSimilarity: {best_similarity:.2f}")
            self.play_sound("welcome")
            # Add delay before exiting
            Clock.schedule_once(lambda dt: self.exit_face_auth(), 2)
        else:
            self.ids.status_label.text = "Access denied"
            feedback = []
            if best_similarity <= self.RECOGNITION_THRESHOLD:
                feedback.append("Face not recognized")
            if (best_match is not None and 
                self.known_faces[best_match]['password'] != password):
                feedback.append("Incorrect PIN")
            self.show_popup("Error", "\n".join(feedback) if feedback else "Authentication failed")
            self.play_sound("error")

        self.clear_password()

    def show_popup(self, title, message):
        content = BoxLayout(orientation='vertical', spacing=10)
        content.add_widget(Label(text=message, font_size='20sp'))
        btn = Button(text="OK", size_hint=(1, 0.2))
        popup = Popup(title=title, content=content, size_hint=(0.8, 0.4))
        btn.bind(on_press=popup.dismiss)
        content.add_widget(btn)
        popup.open()

    def exit_face_auth(self):
        # Show exit message
        self.ids.status_label.text = "Exiting face authentication..."

        # Schedule cleanup and return to main screen after a brief delay
        Clock.schedule_once(lambda dt: self.cleanup_and_exit(), 0.5)

    def cleanup_and_exit(self):
        # Perform cleanup
        self.cleanup()

        # Return to main screen
        self.main_app.show_main_screen()

    def cleanup(self):
        if hasattr(self, 'capture') and self.capture is not None:
            self.capture.release()
            self.capture = None
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        try:
            pygame.mixer.quit()
        except:
            pass

class ExerciseScreen(BoxLayout):
    def __init__(self, exercise, main_app, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.exercise = exercise
        self.main_app = main_app
        self.counter = 0
        self.stage = None
        self.start_time = time.time()

        # Improved pose estimation configuration
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )

        self.angles = {
            "Bicep Curls": {"up": 160, "down": 60},
            "Squats": {"up": 160, "down": 90},
            "Pushups": {"up": 160, "down": 90}
        }

        with self.canvas.before:
            Color(0.95, 0.95, 0.97, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)

        self.camera_display = Image(size_hint=(1, 0.6))
        self.add_widget(self.camera_display)

        self.feedback_label = Label(text=f"Starting {exercise}...", font_size='24sp')
        self.feedback_label.color = (0.2, 0.2, 0.2, 1)
        self.add_widget(self.feedback_label)

        self.info_label = Label(text=f"Reps: 0/10\nTime: 00:00", size_hint=(1, 0.1))
        self.info_label.color = (0.2, 0.2, 0.2, 1)
        self.add_widget(self.info_label)

        buttons_layout = BoxLayout(size_hint=(1, 0.1))
        self.exit_button = Button(text="Exit", background_color=(0.8, 0.4, 0.4, 1))
        self.exit_button.bind(on_press=self.exit_workout)
        buttons_layout.add_widget(self.exit_button)
        self.add_widget(buttons_layout)

        # Initialize camera with proper orientation
        self.cap = None
        self.initialize_camera()
        if self.cap is not None:
            self.update_event = Clock.schedule_interval(self.update, 1.0/30.0)
    
    def initialize_camera(self):
        """Initialize camera with multiple attempts and error handling"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    return
                else:
                    self.cap.release()
            except Exception as e:
                print(f"Camera initialization attempt {attempt + 1} failed: {e}")
                if hasattr(self, 'cap') and self.cap is not None:
                    self.cap.release()

        self.feedback_label.text = "Camera: Not Available"

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def update(self, dt):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame")
            self.initialize_camera()  # Try to reinitialize camera
            return

        # Flip frame horizontally for mirror view before processing
        frame = cv2.flip(frame, 1)

        # Process the frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True

        try:
            landmarks = results.pose_landmarks.landmark

            if self.exercise == "Bicep Curls":
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                angle = calculate_angle(shoulder, elbow, wrist)

                feedback = "Keep elbows close to your body"
                if angle > 100:
                    feedback = "Good form!"
                self.feedback_label.text = feedback

            elif self.exercise == "Squats":
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                angle = calculate_angle(hip, knee, ankle)

                feedback = "Keep your back straight"
                if angle > 100:
                    feedback = "Good depth!"
                self.feedback_label.text = feedback

            if angle > self.angles[self.exercise]["up"]:
                self.stage = "up"
            if angle < self.angles[self.exercise]["down"] and self.stage == "up":
                self.stage = "down"
                self.counter += 1
                if self.counter >= 10:
                    self.workout_complete()

        except Exception as e:
            print(f"Tracking error: {str(e)}")

        elapsed_time = int(time.time() - self.start_time)
        self.info_label.text = f"Reps: {self.counter}/10\nTime: {elapsed_time//60:02d}:{elapsed_time%60:02d}"

        # Draw landmarks on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        # Convert back to BGR for display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Flip vertically for correct display in Kivy
        buf = cv2.flip(image, 0).tobytes()
        texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.camera_display.texture = texture
    
    def workout_complete(self):
        self.cleanup()
        self.main_app.show_completion_screen(self.exercise)
    
    def exit_workout(self, instance):
        self.cleanup()
        self.main_app.show_main_screen()
    
    def cleanup(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
        if hasattr(self, 'pose'):
            self.pose.close()
        if hasattr(self, 'update_event'):
            Clock.unschedule(self.update_event)

class EmotionDetectionScreen(BoxLayout):
    def __init__(self, main_app, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.main_app = main_app
        # Improved Face Mesh setup
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # Calibration variables
        self.calibrated = False
        self.calibrating = False
        self.neutral_baseline = {
            'mouth_open': 0,
            'eyebrow_mean': 0,
            'mouth_corners': 0
        }
        self.calibration_samples = []
        self.calibration_frames = 30

        # Emotion smoothing
        self.emotion_history = deque(maxlen=5)

        with self.canvas.before:
            Color(0.95, 0.95, 0.97, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)

        # UI Components
        self.camera_display = Image(size_hint=(1, 0.6))
        self.add_widget(self.camera_display)

        self.emotion_label = Label(
            text="Press 'Calibrate' to start", 
            font_size='30sp',
            size_hint=(1, 0.1),
            color=(0, 0, 0, 1)
        )
        self.add_widget(self.emotion_label)

        self.metrics_label = Label(
            text="Waiting for calibration...", 
            font_size='20sp',
            size_hint=(1, 0.1))
        self.add_widget(self.metrics_label)

        buttons_layout = BoxLayout(size_hint=(1, 0.1))
        self.calibration_button = Button(
            text="Calibrate",
            background_color=(0.4, 0.6, 0.8, 1))
        self.calibration_button.bind(on_press=self.start_calibration)
        buttons_layout.add_widget(self.calibration_button)

        self.exit_button = Button(
            text="Exit",
            background_color=(0.8, 0.4, 0.4, 1))
        self.exit_button.bind(on_press=self.exit_emotion_detection)
        buttons_layout.add_widget(self.exit_button)
        self.add_widget(buttons_layout)

        # Initialize camera with proper orientation
        self.cap = None
        self.initialize_camera()
        if self.cap is not None:
            self.update_event = Clock.schedule_interval(self.update, 1.0/30.0)
    
    def initialize_camera(self):
        """Initialize camera with multiple attempts and error handling"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    return
                else:
                    self.cap.release()
            except Exception as e:
                print(f"Camera initialization attempt {attempt + 1} failed: {e}")
                if hasattr(self, 'cap') and self.cap is not None:
                    self.cap.release()

        self.emotion_label.text = "Camera: Not Available"

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def start_calibration(self, instance):
        if self.cap is None:
            self.emotion_label.text = "Camera not available!"
            return

        self.calibrating = True
        self.calibrated = False
        self.calibration_samples = []
        self.emotion_label.text = "Maintain neutral expression..."
        self.metrics_label.text = f"Calibrating: 0/{self.calibration_frames}"
        self.calibration_button.disabled = True

    def calculate_facial_metrics(self, landmarks, frame_shape):
        h, w = frame_shape

        # Mouth openness (using multiple points for robustness)
        mouth_upper = np.mean([(landmarks[13].x, landmarks[13].y),
                              (landmarks[14].x, landmarks[14].y)], axis=0)
        mouth_lower = np.mean([(landmarks[17].x, landmarks[17].y),
                             (landmarks[18].x, landmarks[18].y)], axis=0)
        mouth_open = abs(mouth_lower[1] - mouth_upper[1]) * h

        # Eyebrow positions (average of multiple points)
        left_eyebrow = np.mean([(landmarks[65].x, landmarks[65].y),
                               (landmarks[158].x, landmarks[158].y)], axis=0)
        right_eyebrow = np.mean([(landmarks[295].x, landmarks[295].y),
                                (landmarks[385].x, landmarks[385].y)], axis=0)
        eyebrow_mean = (left_eyebrow[1] + right_eyebrow[1]) / 2

        # Mouth corner curvature
        left_corner = (landmarks[61].x, landmarks[61].y)
        right_corner = (landmarks[291].x, landmarks[291].y)
        mouth_center = (mouth_upper + mouth_lower) / 2
        mouth_corners = ((left_corner[1] + right_corner[1]) / 2) - mouth_center[1]

        return {
            'mouth_open': mouth_open,
            'eyebrow_mean': eyebrow_mean,
            'mouth_corners': mouth_corners
        }

    def detect_emotion(self, metrics):
        # Calculate differences from neutral baseline
        mouth_diff = metrics['mouth_open'] - self.neutral_baseline['mouth_open']
        eyebrow_diff = metrics['eyebrow_mean'] - self.neutral_baseline['eyebrow_mean']
        mouth_curve_diff = metrics['mouth_corners'] - self.neutral_baseline['mouth_corners']

        # More sensitive thresholds for emotion detection
        if mouth_diff > 20 and eyebrow_diff < -0.02:
           return "SURPRISED 😲"
        elif mouth_curve_diff < -0.02:
            return "HAPPY 😊"
        elif mouth_curve_diff > 0.02:
            return "SAD 😢"
        elif eyebrow_diff < -0.03:
            return "ANGRY 😠"
        elif mouth_diff > 15:
            return "CONFUSED 🤔"
        else:
            return "NEUTRAL 😐"

    def update(self, dt):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame")
            self.initialize_camera()  # Try to reinitialize camera
            return

        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            metrics = self.calculate_facial_metrics(landmarks, frame.shape[:2])

            if self.calibrating:
                # Calibration in progress
                self.calibration_samples.append(metrics)
                progress = len(self.calibration_samples)
                self.metrics_label.text = f"Calibrating: {progress}/{self.calibration_frames}"

                if progress >= self.calibration_frames:
                    self.finish_calibration()
            elif self.calibrated:
                # Normal detection mode
                emotion = self.detect_emotion(metrics)
                self.emotion_history.append(emotion)

                # Get most frequent emotion from history
                final_emotion = max(set(self.emotion_history), 
                                   key=self.emotion_history.count)

                self.emotion_label.text = final_emotion
                self.metrics_label.text = (
                    f"Mouth: {metrics['mouth_open']:.1f} | "
                    f"Eyebrow: {metrics['eyebrow_mean']:.3f} | "
                    f"Curve: {metrics['mouth_corners']:.3f}"
                )

            # Draw landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.multi_face_landmarks[0],
                mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
        else:
            if self.calibrating:
                self.emotion_label.text = "Face not detected! Maintain neutral expression"
            elif not self.calibrated:
                self.emotion_label.text = "Face not detected"

        # Convert to texture for Kivy display
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), 
            colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.camera_display.texture = texture

    def finish_calibration(self):
        # Calculate average neutral values
        self.neutral_baseline = {
            'mouth_open': np.mean([s['mouth_open'] for s in self.calibration_samples]),
            'eyebrow_mean': np.mean([s['eyebrow_mean'] for s in self.calibration_samples]),
            'mouth_corners': np.mean([s['mouth_corners'] for s in self.calibration_samples])
        }

        self.calibrating = False
        self.calibrated = True
        self.emotion_label.text = "Calibration complete! Show your emotions"
        self.metrics_label.text = (
            f"Baseline - Mouth: {self.neutral_baseline['mouth_open']:.1f} | "
            f"Eyebrow: {self.neutral_baseline['eyebrow_mean']:.3f} | "
            f"Curve: {self.neutral_baseline['mouth_corners']:.3f}"
        )
        self.calibration_button.text = "Recalibrate"
        self.calibration_button.disabled = False

    def exit_emotion_detection(self, instance):
        self.cleanup()
        self.main_app.show_main_screen()

    def cleanup(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        if hasattr(self, 'update_event'):
            Clock.unschedule(self.update_event)


class SkinAnalysisScreen(BoxLayout):
    def __init__(self, main_app, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.main_app = main_app

        # Improved Face Mesh configuration
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        with self.canvas.before:
            Color(0.95, 0.95, 0.97, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)

        self.camera_display = Image(size_hint=(1, 0.6))
        self.add_widget(self.camera_display)

        self.info_label = Label(text="Face the camera for skin analysis",
                              font_size='24sp', size_hint=(1, 0.1))
        self.info_label.color = (0.2, 0.2, 0.2, 1)
        self.add_widget(self.info_label)

        self.analysis_label = Label(text="", font_size='24sp', size_hint=(1, 0.1))
        self.analysis_label.color = (0.2, 0.2, 0.2, 1)
        self.add_widget(self.analysis_label)

        buttons_layout = BoxLayout(size_hint=(1, 0.1))
        self.analyze_button = Button(text="Analyze", background_color=(0.4, 0.8, 0.6, 1))
        self.analyze_button.bind(on_press=self.analyze_skin)
        buttons_layout.add_widget(self.analyze_button)


        self.exit_button = Button(text="Exit", background_color=(0.8, 0.4, 0.4, 1))
        self.exit_button.bind(on_press=self.exit_analysis)
        buttons_layout.add_widget(self.exit_button)
        self.add_widget(buttons_layout)

        # Initialize camera with proper orientation
        self.cap = None
        self.initialize_camera()
        if self.cap is not None:
            self.update_event = Clock.schedule_interval(self.update, 1.0/30.0)
    
    def initialize_camera(self):
        """Initialize camera with multiple attempts and error handling"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    return
                else:
                    self.cap.release()
            except Exception as e:
                print(f"Camera initialization attempt {attempt + 1} failed: {e}")
                if hasattr(self, 'cap') and self.cap is not None:
                    self.cap.release()

        self.info_label.text = "Camera: Not Available"

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def update(self, dt):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame")
            self.initialize_camera()  # Try to reinitialize camera
            return

        # Flip frame horizontally for mirror view before processing
        frame = cv2.flip(frame, 1)

        # Process the frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.face_mesh.process(image)
        image.flags.writeable = True

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=1, circle_radius=1))

        # Convert back to BGR for display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Flip vertically for correct display in Kivy
        buf = cv2.flip(image, 0).tobytes()
        texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.camera_display.texture = texture

    def analyze_skin(self, instance):
        if self.cap is None:
            self.analysis_label.text = "Camera not available"
            return

        ret, frame = self.cap.read()
        if not ret:
            self.analysis_label.text = "Failed to capture image"
            return

        # Flip frame horizontally for mirror view before processing
        frame = cv2.flip(frame, 1)

        # Process the frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)

        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Skin color range
        lower_skin = np.array([0, 48, 80], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

        # Get face region
        face_mask = np.zeros_like(skin_mask)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            convexhull = cv2.convexHull(np.array(points))
            cv2.fillConvexPoly(face_mask, convexhull, 255)

        # Combine with skin mask
        final_mask = cv2.bitwise_and(skin_mask, face_mask)

        # Analyze skin properties
        mean_val = cv2.mean(frame, mask=final_mask)
        feedback = []

        # Skin type detection
        saturation = mean_val[1]
        brightness = mean_val[2]

        if saturation > 180:
            feedback.append("Very oily skin (consider oil-control products)")
        elif saturation > 150:
            feedback.append("Oily skin (use gentle cleanser)")
        elif brightness < 90:
            feedback.append("Very dry skin (need rich moisturizer)")
        elif brightness < 120:
            feedback.append("Normal to dry skin (use light moisturizer)")
        else:
            feedback.append("Normal skin (maintain current routine)")

        # Redness detection
        if mean_val[0] > 130:
            feedback.append("Significant redness (possible irritation)")
        elif mean_val[0] > 100:
            feedback.append("Some redness (consider soothing products)")

        self.analysis_label.text = "Skin Analysis:\n" + "\n".join(feedback)

    def exit_analysis(self, instance):
        self.cleanup()
        self.main_app.show_main_screen()


    def cleanup(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        if hasattr(self, 'update_event'):
            Clock.unschedule(self.update_event)

class SmartWorkoutMirrorApp(App):
    def build(self):
        Window.fullscreen = 'auto'
        self.main_screen = MainScreen()
        return self.main_screen

    def show_emotion_detection_screen(self):
        self.root.clear_widgets()
        self.emotion_detection_screen = EmotionDetectionScreen(self)
        self.root.add_widget(self.emotion_detection_screen)

    def show_exercise_selection(self):
        content = BoxLayout(orientation='vertical', padding=20, spacing=15)
        content.add_widget(Label(text="Select Your Workout:", font_size='24sp', size_hint_y=0.2))

        grid = GridLayout(cols=1, spacing=10, size_hint_y=0.8)

        exercises = ["Bicep Curls", "Squats", "Pushups"]
        colors = [(0.4, 0.6, 0.8, 1), (0.8, 0.6, 0.4, 1), (0.6, 0.4, 0.8, 1)]

        for exercise, color in zip(exercises, colors):
            btn = Button(text=exercise, font_size='20sp', background_color=color)
            btn.bind(on_press=lambda instance, ex=exercise: self.on_exercise_selected(ex))
            grid.add_widget(btn)

        cancel_btn = Button(text="Cancel", font_size='20sp', background_color=(0.8, 0.4, 0.4, 1))
        cancel_btn.bind(on_press=lambda x: self.exercise_dialog.dismiss())
        grid.add_widget(cancel_btn)

        content.add_widget(grid)

        self.exercise_dialog = Popup(title="Workout Selection",
                                   content=content,
                                   size_hint=(0.8, 0.8))
        self.exercise_dialog.open()

    def on_exercise_selected(self, exercise):
        self.exercise_dialog.dismiss()
        self.start_exercise(exercise)

    def start_exercise(self, exercise):
        self.root.clear_widgets()
        self.exercise_screen = ExerciseScreen(exercise, self)
        self.root.add_widget(self.exercise_screen)

    def show_skin_analysis_screen(self):
        self.root.clear_widgets()
        self.skin_analysis_screen = SkinAnalysisScreen(self)
        self.root.add_widget(self.skin_analysis_screen)

    def show_face_auth_screen(self):
        self.root.clear_widgets()
        self.face_auth_screen = FaceAuthScreen(self)
        self.root.add_widget(self.face_auth_screen)

    def show_main_screen(self):
        self.root.clear_widgets()
        self.main_screen = MainScreen()
        self.root.add_widget(self.main_screen)

    def show_completion_screen(self, exercise):
        content = BoxLayout(orientation='vertical', spacing=10, padding=10)
        content.add_widget(Label(text=f"You completed 10 reps of {exercise}!", 
                               font_size='24sp', size_hint_y=0.6))

        buttons = GridLayout(cols=2, spacing=10, size_hint_y=0.4)
        another_btn = Button(text="Another Workout", background_color=(0.4, 0.6, 0.8, 1))
        another_btn.bind(on_press=lambda x: (self.completion_dialog.dismiss(), 
                                           self.show_exercise_selection()))
        buttons.add_widget(another_btn)

        finish_btn = Button(text="Finish", background_color=(0.8, 0.4, 0.4, 1))
        finish_btn.bind(on_press=lambda x: (self.completion_dialog.dismiss(), 
                                          self.show_main_screen()))
        buttons.add_widget(finish_btn)

        content.add_widget(buttons)

        self.completion_dialog = Popup(title="Workout Complete!",
                                     content=content,
                                     size_hint=(0.8, 0.6))
        self.completion_dialog.open()

if __name__ == '__main__':
    # Install required packages if needed
    try:
        import pygame
        import pyttsx3
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.run(["pip", "install", "pygame", "pyttsx3", "requests", "sounddevice", "opencv-python", "mediapipe", "numpy", "python-dotenv", "faster-whisper", "twilio"])
    
    SmartWorkoutMirrorApp().run()
