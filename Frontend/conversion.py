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