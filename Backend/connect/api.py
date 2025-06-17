from fastapi import FastAPI, File, Form, UploadFile, BackgroundTasks, APIRouter, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
from connect.utils.common import process_and_store
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from pymongo import MongoClient
import gridfs
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
print("DEBUG MONGO_URI:", os.getenv("MONGO_URI"))

DB_NAME = "emotion_dataset"
THRESHOLD = 9000  # Max allowed files per emotion

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
fs = gridfs.GridFS(db)


# Custom registry to avoid duplicate metric registration on reload
REGISTRY = CollectorRegistry()

# Emotion thresholds
THRESHOLDS = {"happy": 8989,"angry": 4953,"neutral": 6198}

# Inference metrics with custom registry
emotion_counter = Counter("emotion_inference_count","Count of each emotion",["emotion"],registry=REGISTRY)

inference_latency = Histogram("inference_latency_seconds","Latency of emotion inference in seconds",registry=REGISTRY)

frame_processing_counter = Counter("frame_processing_total","Total number of frames processed",registry=REGISTRY)

face_detected_counter = Counter("face_detected_total","Number of frames with a detected face",registry=REGISTRY)

no_face_counter = Counter("no_face_total","Number of frames with no detected face",registry=REGISTRY)

invalid_frame_counter = Counter("invalid_frame_total","Frames with invalid or empty face crops",registry=REGISTRY)

upload_success_counter = Counter("upload_success_total","Successful uploads to API",registry=REGISTRY)

upload_failure_counter = Counter("upload_failure_total","Failed uploads to API",registry=REGISTRY)

inference_error_counter = Counter("inference_error_total","Total errors during inference",registry=REGISTRY)

emotion_confidence = Histogram("emotion_confidence","Confidence scores of emotion predictions",["emotion"],registry=REGISTRY)

# Storage count metrics
emotion_image_count = Gauge("emotion_image_count","Current number of images per emotion",["emotion"],registry=REGISTRY)

emotion_threshold_reached = Gauge("emotion_threshold_reached","Whether the emotion has reached threshold (1=yes)",["emotion"],registry=REGISTRY)

emotion_progress = Gauge("emotion_threshold_percentage","Percentage progress toward the emotion threshold",["emotion"],registry=REGISTRY)

app = FastAPI()

# Optional: remove Instrumentator if not using default metrics
# from prometheus_fastapi_instrumentator import Instrumentator
# Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    prediction: Annotated[str, Form()]
):
    image_bytes = await file.read()
    background_tasks.add_task(process_and_store, image_bytes, prediction.lower())
    return {"status": "processing"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

router = APIRouter()

@router.post("/record_metrics/")
async def receive_metrics(request: Request):
    data = await request.json()
    emotion = data.get("emotion", "unknown")
    latency = data.get("latency", 0.0)
    confidence = data.get("confidence", None)
    face_found = data.get("face_found", None)
    upload_status = data.get("upload_status", None)
    error_occurred = data.get("error_occurred", False)

    frame_processing_counter.inc()
    emotion_counter.labels(emotion=emotion).inc()
    inference_latency.observe(latency)

    if confidence is not None:
        emotion_confidence.labels(emotion=emotion).observe(confidence)

    if face_found is True:
        face_detected_counter.inc()
    elif face_found is False:
        no_face_counter.inc()

    if upload_status == "success":
        upload_success_counter.inc()
    elif upload_status == "failure":
        upload_failure_counter.inc()

    if error_occurred:
        inference_error_counter.inc()

         # Update Prometheus metrics
    doc = db["emotion_counts"].find_one({"emotion": emotion})
    count = doc["count"] if doc else 0
    emotion_image_count.labels(emotion=emotion).set(count)

    threshold = THRESHOLDS.get(emotion, 999999)
    if count >= threshold:
        emotion_threshold_reached.labels(emotion=emotion).set(1)
    else:
        emotion_threshold_reached.labels(emotion=emotion).set(0)


    # Assuming 'emotion' is the label, and 'count' is the current image count:
    threshold = THRESHOLDS.get(emotion, 100)  # fallback to 100 if undefined
    percentage = (count / threshold) * 100
    percentage = min(percentage, 100)  # Cap at 100%

    emotion_progress.labels(emotion=emotion).set(percentage)

    return {"status": "metrics recorded"}

app.include_router(router)
