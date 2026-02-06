import tensorflow as tf
import os

MODEL_PATH = os.path.join("models", "oralguard_full_model.keras")

print("MODEL PATH:", os.path.abspath(MODEL_PATH))
print("MODEL EXISTS:", os.path.exists(MODEL_PATH))

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("✅ MODEL LOAD OK")
