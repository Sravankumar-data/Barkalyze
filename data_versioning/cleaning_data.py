import os
import re
from pymongo import MongoClient
import gridfs
from dotenv import load_dotenv

# ---------------------------- Environment Setup ---------------------------- #

load_dotenv()
MONGO_URI = os.getenv("MONGODB_URI")

# ---------------------------- MongoDB & GridFS Setup ---------------------------- #

client = MongoClient(MONGO_URI)
db = client["emotion_dataset"]
fs = gridfs.GridFS(db)

COLLECTION_NAME = "emotion_counts"
collection = db[COLLECTION_NAME]

# ---------------------------- Local File System Setup ---------------------------- #

# Directory where files will be saved
root_output_folder = os.path.join(os.path.dirname(__file__), "emotion_dataset")
os.makedirs(root_output_folder, exist_ok=True)

# ---------------------------- Utility Functions ---------------------------- #

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to avoid OS-restricted characters."""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def extract_emotion_from_filename(filename: str) -> str:
    """Extract emotion label from the filename (assumes format: emotion_timestamp.jpg)."""
    return filename.split("_")[0]

# ---------------------------- Fetch and Save Files ---------------------------- #

for file in fs.find():
    raw_filename = file.filename
    safe_filename = sanitize_filename(raw_filename)

    # Determine emotion label from filename
    emotion_label = extract_emotion_from_filename(safe_filename)

    # Create emotion-specific subfolder
    emotion_folder = os.path.join(root_output_folder, emotion_label)
    os.makedirs(emotion_folder, exist_ok=True)

    # Full path for output file
    filepath = os.path.join(emotion_folder, safe_filename)

    # Read data and write to disk
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

# ---------------------------- Clear GridFS ---------------------------- #

fs_files = db.fs.files
fs_chunks = db.fs.chunks

deleted_files = fs_files.delete_many({})
deleted_chunks = fs_chunks.delete_many({})
print(f"🧹 Cleared GridFS: {deleted_files.deleted_count} files and {deleted_chunks.deleted_count} chunks.")

# ---------------------------- Reset Emotion Counts ---------------------------- #

counts = {"angry": 0, "happy": 0, "neutral": 0}

# Clear existing documents
collection.delete_many({})

# Insert new reset counts
bulk_docs = [{"emotion": emotion, "count": count} for emotion, count in counts.items()]
if bulk_docs:
    collection.insert_many(bulk_docs)
    print("✅ Emotion counts saved to MongoDB collection.")
