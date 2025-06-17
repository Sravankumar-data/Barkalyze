import cv2
import streamlit as st
import threading
from pathlib import Path
import requests
import base64
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import Fconfig
from streamlit_autorefresh import st_autorefresh
import numpy as np
import json
import os
import random

import threading
import requests
import time

METRICS_API_URL = "http://localhost:8000/record_metrics/"  # FastAPI metrics endpoint

def send_metrics_async(emotion: str, latency: float, confidence: float = None, face_found: bool = None, upload_status: str = None, error_occurred: bool = False):
    def _send():
        try:
            payload = {
                "emotion": emotion,
                "latency": latency,
                "confidence": confidence,
                "face_found": face_found,
                "upload_status": upload_status,
                "error_occurred": error_occurred
            }
            response = requests.post(METRICS_API_URL, json=payload, timeout=1)
            if response.status_code == 200:
                print("[✓] Metrics sent.")
            else:
                print(f"[✗] Failed to send metrics: {response.status_code}")
        except Exception as e:
            print(f"[!] Metrics error: {e}")
    threading.Thread(target=_send, daemon=True).start()




UPLOAD_API_URL = "http://localhost:8000/upload/"  # Update if deployed or hosted elsewhere

def send_to_api(image: np.ndarray, prediction: str):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, img_encoded = cv2.imencode(".jpg", gray_image)
    files = {"file": ("image.jpg", img_encoded.tobytes(), "image/jpeg")}
    data = {"prediction": prediction}
    
    try:
        response = requests.post(UPLOAD_API_URL, files=files, data=data)
        if response.status_code == 200:
            print("[✓] Image uploaded successfully.")
        else:
            print(f"[✗] Failed to upload: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"[!] Upload error: {e}")

EMOTION_FILE = "Frontend/emotion_state.json"
class_names = ["Angry", "Happy", "Neutral"]  # Customize for your model

# Load the TFLite model
current_dir = Path(__file__).resolve().parent
model_path = current_dir.parent / "model" / "model.tflite"
interpreter = tf.lite.Interpreter(model_path=str(model_path))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.set_page_config(layout="wide")

def write_emotion_to_file(emotion):
    data = read_emotion_from_file()
    data["emotion"] = emotion
    with open(EMOTION_FILE, "w") as f:
        json.dump(data, f)

def write_previous_emotion_to_file(emotion,r):
    data = read_emotion_from_file()
    data["previous_emotion"] = emotion
    data["random"] = r
    with open(EMOTION_FILE, "w") as f:
        json.dump(data, f)

def write_emotion_timestamp(now):
    data = read_emotion_from_file()
    data["timestamp"] = now
    with open(EMOTION_FILE, "w") as f:
        json.dump(data, f)

def read_emotion_from_file():
    if not os.path.exists(EMOTION_FILE):
        return {"emotion": "neutral", "timestamp": 0, "previous_emotion" : "neutral", "random":1}
    try:
        with open(EMOTION_FILE, "r") as f:
            return json.load(f)
    except:
        return {"emotion": "neutral", "timestamp": 0, "previous_emotion" : "neutral", "random":1}

        
emotion_emojis = {
    "happy": "😄",
    "angry": "😠",
    "neutral": "😐",
    "noface": "🚫"
}

st.markdown("""
    <style>
        .block-container { 
            padding: 2rem 4rem !important; 
            font-family: 'Segoe UI', sans-serif;
        }
        h1, h2, h3 {
            color: #333333;
        }
        .emotion-header {
            font-size: 1rem;       /* Smaller, readable text */
            font-weight: 500;
            color: #ffffff;
            margin: 0;
        }
        .webrtc-container {
            margin-top: 1rem;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
            border: 2px solid #ddd;
        }
        .video-card {
            display: inline-block;
            background-color: #4a90e2;
            padding: 8px 12px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin-top: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)





class FaceEmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.emotion = "neutral"
        self.emotion_lock = threading.Lock()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.frame_count = 0
        self.process_every_n = 5
        self.thread_lock = threading.Lock()
        self.thread_result = None

    def recv(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")

        # Downsample for faster face detection
        gray_small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), fx=0.5, fy=0.5)
        faces = self.face_cascade.detectMultiScale(gray_small, scaleFactor=1.3, minNeighbors=5)

        if self.frame_count % self.process_every_n != 0:
            return frame.from_ndarray(img, format="bgr24")

        if len(faces) > 0:
            x, y, w, h = [int(coord * 2) for coord in faces[0]]  # scale back to original size
            face_crop = img[y:y + h, x:x + w]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            def infer_async(crop):
                emotion = self.update_emotion(crop)
                with self.thread_lock:
                    self.thread_result = emotion
                    write_emotion_to_file(emotion)

            threading.Thread(target=infer_async, args=(face_crop,), daemon=True).start()

        else:
            with self.emotion_lock:
                self.emotion = "noFace"
                write_emotion_to_file("noFace")

        current_emotion = self.emotion
        with self.thread_lock:
            if self.thread_result:
                current_emotion = self.thread_result
                self.emotion = current_emotion
                self.thread_result = None

        # cv2.putText(img, f"Detected: {current_emotion}", (20, 40),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame.from_ndarray(img, format="bgr24")

    def update_emotion(self, face_crop):
        try:
            if face_crop.size == 0:
                send_metrics_async("noFace", 0.0, face_found=False)
                return "noFace"

            face_resized = cv2.resize(face_crop, (224, 224))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_normalized = face_rgb / 255.0
            input_tensor = np.expand_dims(face_normalized, axis=0).astype(np.float32)

            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])
            latency = time.time() - start_time

            predicted_index = np.argmax(predictions)
            confidence = float(predictions[0][predicted_index])
            emotion = class_names[predicted_index].lower()

            upload_status = None
            try:
                send_to_api(face_crop, emotion)
                upload_status = "success"
            except Exception:
                upload_status = "failure"

            send_metrics_async(
                emotion=emotion,
                latency=latency,
                confidence=confidence,
                face_found=True,
                upload_status=upload_status,
            )

            return emotion

        except Exception as e:
            print(f"[Error] Emotion prediction failed: {e}")
            send_metrics_async("noFace", 0.0, face_found=False, error_occurred=True)
            return "noFace"



def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")

def get_current_video(emotion):
    video_path = Fconfig.get_video_path(emotion)  # dynamically toggled
    return encode_video(video_path)

def main():
    st.title("🐶 BARKALYZE")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("📸 Webcam")
        st.markdown('<div class="webrtc-container">', unsafe_allow_html=True)
        webrtc_streamer(
            key="emotion-detection",
            video_processor_factory=FaceEmotionProcessor,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        header_col1, header_col2 = st.columns([2, 1])
        
        with header_col1:
            st.subheader("😄 Feel Better with Barkalyze")
        
        with header_col2:
            d = read_emotion_from_file()
            current_emotion = d["emotion"]
            time_stamp = d["timestamp"]
            previous_emotion = d.get("previous_emotion", "neutral")

            now = time.time()
            emoji = emotion_emojis.get(current_emotion.lower(), "❓")

            st.markdown(
                f'<div class="video-card"><p class="emotion-header">{current_emotion.capitalize()}{emoji}</p></div>',
                unsafe_allow_html=True
            )

        mandatory_emotion = ["angry", "happy"]

        # Conditions
        emotion_changed = current_emotion != previous_emotion
        in_mandatory = current_emotion in mandatory_emotion
        prev_not_mandatory = previous_emotion not in mandatory_emotion
        time_elapsed = now - time_stamp >= 4

        # Trigger update if:
        # (1) normal delay passed, or
        # (2) emotion changed to mandatory from non-mandatory
        if emotion_changed and (time_elapsed or (in_mandatory and prev_not_mandatory)):
            r = int(random.choice([1, 2]))
            write_previous_emotion_to_file(current_emotion, r)
            write_emotion_timestamp(now)

        n = read_emotion_from_file()
        now_pre_emo = n["previous_emotion"]
        video_emotion = now_pre_emo or "neutral"
        video_source = get_current_video(video_emotion)

        with open("Frontend/templates/index.html", "r") as file:
            html_template = file.read()

        html_template = html_template.replace("{{ VIDEO_DATA }}", f'"{video_source}"')
        st.components.v1.html(html_template, height=800)
        st_autorefresh(interval=1000, key="refresh")


if __name__ == "__main__":
    main()