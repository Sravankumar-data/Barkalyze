# Instead of static EMOTION_VIDEOS dict, use this function
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
VIDEOS_DIR = os.path.join(BASE_DIR, "static")

EMOTION_FILE = "Frontend/emotion_state.json"
def read_emotion_from_file():
    if not os.path.exists(EMOTION_FILE):
        return {"emotion": "neutral", "timestamp": 0, "previous_emotion" : "neutral", "random":1}
    try:
        with open(EMOTION_FILE, "r") as f:
            return json.load(f)
    except:
        return {"emotion": "neutral", "timestamp": 0, "previous_emotion" : "neutral", "random":1}


def get_video_path(emotion):
    data = read_emotion_from_file()
    r = data["random"]
    return os.path.join(VIDEOS_DIR, f"{emotion}/{r}.mp4")
