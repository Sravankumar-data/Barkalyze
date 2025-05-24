import tensorflow as tf

# Load Keras model
model = tf.keras.models.load_model("model/model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optional: reduce size further
tflite_model = converter.convert()

# Save the TFLite model
with open("model/model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Converted to TensorFlow Lite: model/model.tflite")





# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# import uvicorn
# import numpy as np
# from tensorflow.keras.models import load_model
# from PIL import Image
# import io

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # allow Streamlit access
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load the model only once
# model = load_model("artifacts/training/model.h5")
# class_names = ["Angry", "Happy", "Neutral"]  # Customize for your model

# def preprocess_image(image_bytes):
#     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     image = image.resize((224, 224))  # adjust to your model input size
#     image_array = np.array(image) / 255.0  # normalize
#     image_array = np.expand_dims(image_array, axis=0)  # batch dimension
#     return image_array

# @app.post("/predict_image")
# async def predict_image(file: UploadFile = File(...)):
#     try:
#         image_bytes = await file.read()
#         processed_image = preprocess_image(image_bytes)

#         prediction = model.predict(processed_image)
#         predicted_class = class_names[np.argmax(prediction)]

#         return JSONResponse({
#             "predicted_class": predicted_class,
#             "confidence": float(np.max(prediction))
#         })
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
