import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# ---------------- CONFIG ----------------
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "histo_split")
MODEL_DIR = os.path.join(BASE_DIR, "models")

BEST_WEIGHTS = os.path.join(MODEL_DIR, "oralguard_best.weights.h5")
FINETUNED_WEIGHTS = os.path.join(MODEL_DIR, "oralguard_finetuned.weights.h5")
FULL_MODEL_PATH = os.path.join(MODEL_DIR, "oralguard_full_model.keras")

# ---------------- DATA ----------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# ---------------- MODEL ----------------
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# 🔓 Unfreeze last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# Load previous best weights
model.load_weights(BEST_WEIGHTS)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ---------------- MANUAL FINETUNING LOOP ----------------
print("\n🚀 Starting fine-tuning...\n")

for epoch in range(EPOCHS):
    print(f"===== Epoch {epoch+1}/{EPOCHS} =====")

    # ---- TRAIN ----
    for step in range(len(train_data)):
        x_batch, y_batch = train_data[step]
        model.train_on_batch(x_batch, y_batch)

    # ---- VALIDATE ----
    val_losses, val_accs = [], []
    for step in range(len(val_data)):
        x_val, y_val = val_data[step]
        loss, acc = model.test_on_batch(x_val, y_val)
        val_losses.append(float(loss))
        val_accs.append(float(acc))

    print(
        f"Validation loss: {sum(val_losses)/len(val_losses):.4f} | "
        f"accuracy: {sum(val_accs)/len(val_accs):.4f}"
    )

# ---------------- SAVE ----------------
model.save_weights(FINETUNED_WEIGHTS)
model.save_weights(
    os.path.join(MODEL_DIR, "oralguard_finetuned.weights.h5")
)
print("✅ Fine-tuned weights saved")


print("\n✅ Fine-tuning completed successfully")
print(f"📦 Weights saved : {FINETUNED_WEIGHTS}")
print(f"📦 Full model   : {FULL_MODEL_PATH}")
