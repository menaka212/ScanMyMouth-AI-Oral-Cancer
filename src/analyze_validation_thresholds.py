import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------- CONFIG ----------------
IMG_SIZE = 224
BATCH_SIZE = 16

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "histo_split", "val")
WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "oralguard_final.weights.h5")

# ---------------- DATA ----------------
val_gen = ImageDataGenerator(rescale=1./255)

val_data = val_gen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

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
model.load_weights(WEIGHTS_PATH)

# ---------------- PREDICT ----------------
y_true = val_data.classes
y_probs = model.predict(val_data).ravel()

# ---------------- THRESHOLD SEARCH ----------------
print("\nThreshold tuning results:\n")

for t in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65]:
    y_pred = (y_probs >= t).astype(int)
    acc = np.mean(y_pred == y_true)
    print(f"Threshold {t:.2f} → Accuracy: {acc:.4f}")
