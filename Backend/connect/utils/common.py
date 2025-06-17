import os
import cv2
from pymongo import MongoClient
import gridfs
from datetime import datetime
import numpy as np
from dotenv import load_dotenv


load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
print("DEBUG MONGO_URI:", os.getenv("MONGO_URI"))

DB_NAME = "emotion_dataset"
THRESHOLD = 9000  # Max allowed files per emotion

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
fs = gridfs.GridFS(db)


def is_blurry(image, threshold=100.0):
    lap_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return lap_var < threshold

def count_files_by_prediction(prediction: str) -> int:
    return db.fs.files.count_documents({"metadata.prediction": prediction})

def process_and_store(image_bytes: bytes, prediction: str):
    # Load image from bytes
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if prediction not in ["happy", "angry", "neutral"]:
        print("[Skip] Invalid prediction.")
        return

    if is_blurry(img):
        print("[Skip] Image is blurry.")
        return

    if count_files_by_prediction(prediction) >= THRESHOLD:
        print(f"[Skip] Threshold reached for '{prediction}'")
        return

    # Encode to bytes again for GridFS
    success, encoded_img = cv2.imencode(".jpg", img)
    if not success:
        print("[Error] Failed to encode image.")
        return

    image_bytes = encoded_img.tobytes()

    filename = f"{prediction}_{datetime.utcnow().isoformat()}.jpg"
    fs.put(
        image_bytes,
        filename=filename,
        metadata={
            "prediction": prediction,
            "timestamp": datetime.utcnow()
        }
    )

    db["emotion_counts"].update_one(
        {"emotion": prediction},
        {"$inc": {"count": 1}},
        upsert=True
    )


    print(f"[Saved to GridFS] {filename}")
