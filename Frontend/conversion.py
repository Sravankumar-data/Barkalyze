import tensorflow as tf
import dagshub
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

dagshub.init(repo_owner=os.getenv("REPO_OWNER"), repo_name=os.getenv("REPO_NAME"), mlflow=True)

import mlflow.keras
from tensorflow.keras.models import load_model


# Load Keras model
model = load_model('model/model.h5')
# model = mlflow.keras.load_model(model_uri="models:/MobileNet@production")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optional: reduce size further
tflite_model = converter.convert()

# Save the TFLite model
with open("model/model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Converted to TensorFlow Lite: model/model.tflite")