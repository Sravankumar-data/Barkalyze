# metrics.py


# Custom registry to avoid duplicate metric registration on reload
REGISTRY = CollectorRegistry()

# Emotion thresholds
THRESHOLDS = {
    "happy": 8989,
    "angry": 4953,
    "neutral": 6201
}

# Inference metrics with custom registry
emotion_counter = Counter(
    "emotion_inference_count",
    "Count of each emotion",
    ["emotion"],
    registry=REGISTRY
)

inference_latency = Histogram(
    "inference_latency_seconds",
    "Latency of emotion inference in seconds",
    registry=REGISTRY
)

frame_processing_counter = Counter(
    "frame_processing_total",
    "Total number of frames processed",
    registry=REGISTRY
)

face_detected_counter = Counter(
    "face_detected_total",
    "Number of frames with a detected face",
    registry=REGISTRY
)

no_face_counter = Counter(
    "no_face_total",
    "Number of frames with no detected face",
    registry=REGISTRY
)

invalid_frame_counter = Counter(
    "invalid_frame_total",
    "Frames with invalid or empty face crops",
    registry=REGISTRY
)

upload_success_counter = Counter(
    "upload_success_total",
    "Successful uploads to API",
    registry=REGISTRY
)

upload_failure_counter = Counter(
    "upload_failure_total",
    "Failed uploads to API",
    registry=REGISTRY
)

inference_error_counter = Counter(
    "inference_error_total",
    "Total errors during inference",
    registry=REGISTRY
)

emotion_confidence = Histogram(
    "emotion_confidence",
    "Confidence scores of emotion predictions",
    ["emotion"],
    registry=REGISTRY
)

# Storage count metrics
emotion_image_count = Gauge(
    "emotion_image_count",
    "Current number of images per emotion",
    ["emotion"],
    registry=REGISTRY
)

emotion_threshold_reached = Gauge(
    "emotion_threshold_reached",
    "Whether the emotion has reached threshold (1=yes)",
    ["emotion"],
    registry=REGISTRY
)
