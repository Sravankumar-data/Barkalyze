import os
import cv2
import numpy as np
from datetime import datetime
from pymongo import MongoClient
import gridfs
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = "emotion_dataset"
THRESHOLD = 9000  # Maximum allowed files per emotion category

# Initialize MongoDB client and GridFS
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
fs = gridfs.GridFS(db)


def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Determines whether an image is blurry using the Laplacian variance method.
    
    Args:
        image (np.ndarray): Image array.
        threshold (float): Threshold below which the image is considered blurry.

    Returns:
        bool: True if the image is blurry, False otherwise.
    """
    lap_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return lap_var < threshold


def count_files_by_prediction(prediction: str) -> int:
    """
    Counts the number of files in GridFS for a given emotion prediction.

    Args:
        prediction (str): Emotion prediction label.

    Returns:
        int: Number of files stored with that prediction.
    """
    return db.fs.files.count_documents({"metadata.prediction": prediction})


def process_and_store(image_bytes: bytes, prediction: str) -> None:
    """
    Processes an image and stores it in MongoDB GridFS if it passes validation checks.

    Args:
        image_bytes (bytes): The raw image bytes.
        prediction (str): Predicted emotion label (happy, angry, or neutral).

    Returns:
        None
    """
    # Validate emotion label
    if prediction not in ["happy", "angry", "neutral"]:
        print("[Skip] Invalid prediction.")
        return

    # Decode image bytes into OpenCV format
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Blur check
    if is_blurry(img):
        print("[Skip] Image is blurry.")
        return

    # Threshold check for the number of images per emotion
    if count_files_by_prediction(prediction) >= THRESHOLD:
        print(f"[Skip] Threshold reached for '{prediction}'.")
        return

    # Re-encode image to JPEG format
    success, encoded_img = cv2.imencode(".jpg", img)
    if not success:
        print("[Error] Failed to encode image.")
        return

    image_bytes = encoded_img.tobytes()
    filename = f"{prediction}_{datetime.utcnow().isoformat()}.jpg"

    # Store image in GridFS with metadata
    fs.put(
        image_bytes,
        filename=filename,
        metadata={
            "prediction": prediction,
            "timestamp": datetime.utcnow()
        }
    )

    # Update emotion count in a separate collection
    db["emotion_counts"].update_one(
        {"emotion": prediction},
        {"$inc": {"count": 1}},
        upsert=True
    )

    print(f"[Saved to GridFS] {filename}")
