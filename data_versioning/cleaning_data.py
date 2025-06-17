from pymongo import MongoClient
import gridfs
import os
import re


# Replace with your MongoDB Atlas connection string
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_URI)
db = client["emotion_dataset"]
fs = gridfs.GridFS(db)
COLLECTION_NAME = "emotion_counts"
collection = db[COLLECTION_NAME]

# Root folder to save files
root_output_folder = os.path.join(os.path.dirname(__file__), "emotion_dataset")
os.makedirs(root_output_folder, exist_ok=True)

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def extract_emotion_from_filename(filename):
    # Assumes emotion is before the first "_" in the filename
    return filename.split("_")[0]

# Fetch and save files
for file in fs.find():
    raw_filename = file.filename
    safe_filename = sanitize_filename(raw_filename)

    # Extract emotion from filename
    emotion_label = extract_emotion_from_filename(safe_filename)

    # Folder: downloaded_files/angry/, downloaded_files/happy/, etc.
    emotion_folder = os.path.join(root_output_folder, emotion_label)
    os.makedirs(emotion_folder, exist_ok=True)

    # Final file path
    filepath = os.path.join(emotion_folder, safe_filename)

    # Read and save file
    data = file.read()
    if not data:
        print(f"⚠️ File empty: {raw_filename}")
        continue

    try:
        with open(filepath, "wb") as f:
            f.write(data)
        print(f"✅ Saved: {filepath}")
    except Exception as e:
        print(f"❌ Error saving {filepath}: {e}")

# Clear GridFS (fs.files and fs.chunks)
fs_files = db.fs.files
fs_chunks = db.fs.chunks

deleted_files = fs_files.delete_many({})
deleted_chunks = fs_chunks.delete_many({})
print(f"🧹 Cleared GridFS: {deleted_files.deleted_count} files and {deleted_chunks.deleted_count} chunks.")


counts = {"angry":0,"happy":0,"neutral":0}
# Step 2: Save to MongoDB collection (overwrite existing)
collection.delete_many({})  # Optional: clear old counts
bulk_docs = [{"emotion": emotion, "count": count} for emotion, count in counts.items()]
if bulk_docs:
    collection.insert_many(bulk_docs)
    print("✅ Emotion counts saved to MongoDB collection.")
