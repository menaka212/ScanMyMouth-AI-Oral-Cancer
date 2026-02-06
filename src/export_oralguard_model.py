import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# ---------------- CONFIG ----------------
IMG_SIZE = 224

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

WEIGHTS_PATH = os.path.join(MODEL_DIR, "oralguard_final.weights.h5")
EXPORT_PATH = os.path.join(MODEL_DIR, "oralguard_full_model.keras")

# ---------------- MODEL ----------------
base_model = EfficientNetB0(
    weights=None,
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# ---------------- LOAD WEIGHTS ----------------
model.load_weights(WEIGHTS_PATH)

# ---------------- SAVE FULL MODEL ----------------
model.save(EXPORT_PATH)

print("✅ Full OralGuard model exported successfully!")
print("📦 Saved at:", EXPORT_PATH)
