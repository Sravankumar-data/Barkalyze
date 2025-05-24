import cv2
import streamlit as st
import threading
import base64
import time
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import Fconfig
from streamlit_autorefresh import st_autorefresh
import numpy as np
import json
import os

EMOTION_FILE = "emotion_state.json"
class_names = ["Angry", "Happy", "Neutral"]  # Customize for your model

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.set_page_config(layout="wide")

def write_emotion_to_file(emotion):
    with open(EMOTION_FILE, "w") as f:
        json.dump({"emotion": emotion}, f)

def read_emotion_from_file():
    if not os.path.exists(EMOTION_FILE):
        return "neutral"
    with open(EMOTION_FILE, "r") as f:
        try:
            data = json.load(f)
            return data.get("emotion", "neutral")
        except json.JSONDecodeError:
            return "neutral"

st.markdown("""
    <style>
        .block-container { padding: 30px !important; margin: 0px !important; }
        .webrtc-container {
            position: absolute; top: 30px; left: 20px;
            width: 200px !important; height: 200px !important;
            z-index: 1000; padding: 10px; border-radius: 10px;
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

        cv2.putText(img, f"Detected: {current_emotion}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame.from_ndarray(img, format="bgr24")

    def update_emotion(self, face_crop):
        try:
            if face_crop.size == 0:
                return "noFace"

            face_resized = cv2.resize(face_crop, (224, 224))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_normalized = face_rgb / 255.0
            input_tensor = np.expand_dims(face_normalized, axis=0).astype(np.float32)

            # ⏱ Profile model prediction
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])
            inference_time = time.time() - start_time
            print(f"Inference time: {inference_time:.4f} seconds")

            predicted_index = np.argmax(predictions)
            emotion = class_names[predicted_index].lower()
            return emotion if emotion in Fconfig.EMOTION_VIDEOS else "neutral"

        except Exception as e:
            print(f"[Error] Emotion prediction failed: {e}")
            return "noFace"


def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")

def get_current_video(emotion):
    video_path = Fconfig.EMOTION_VIDEOS.get(emotion, Fconfig.EMOTION_VIDEOS["neutral"])
    return encode_video(video_path)

def main():
    st.title("Emotion-Based Reaction System")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Live Emotion Detection")
        with st.container():
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
        st.subheader("Video Reactions")
        current_emotion = read_emotion_from_file()
        video_source = get_current_video(current_emotion)
        with open("templates/index.html", "r") as file:
            html_template = file.read()

        html_template = html_template.replace("{{ VIDEO_DATA }}", f'"{video_source}"')
        html_template = html_template.replace("{{ EMOTION }}", current_emotion)
        st.components.v1.html(html_template, height=1000)
        st_autorefresh(interval=1000, key="refresh")

if __name__ == "__main__":
    main()
