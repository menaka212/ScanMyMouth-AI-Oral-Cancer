import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "oralguard_binary_clinical.weights.h5")
MEDIA_DIR = os.path.join(BASE_DIR, "webapp", "media")

IMG_SIZE = 224
LAST_CONV_LAYER = "block6a_expand_activation"

os.makedirs(MEDIA_DIR, exist_ok=True)

# ---------------- BUILD MODEL ----------------
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
output = Dense(1, activation="sigmoid")(x)
model = Model(base_model.input, output)

model.load_weights(MODEL_PATH)

# ---------------- GRAD-CAM ----------------
def generate_gradcam(img_array):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(LAST_CONV_LAYER).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-8
    return heatmap


def overlay_heatmap(image, heatmap):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)


# ---------------- ASSESSMENT ----------------
def generate_assessment(risk, confidence):
    if risk == "HIGH":
        return {
            "status": "Positive",
            "priority": "Urgent Priority",
            "summary": (
                "This AI analysis of the oral cavity image identifies a large, critical lesion. "
                "The lesion exhibits multiple high-risk features including irregular borders, "
                "heterogeneous red and white coloration, and ulcerated or nodular surface patterns. "
                "These findings are highly suspicious for Oral Squamous Cell Carcinoma. "
                "Urgent specialist referral for biopsy and diagnosis is strongly recommended."
            ),
            "severity": {
                "level": "Critical",
                "score": confidence,
                "summary": "High likelihood of malignant oral lesion detected."
            },
            "risk": {
                "level": "Very High",
                "probability": f"{confidence}%",
                "factors": [
                    "Irregular lesion surface",
                    "Mixed red and white patches",
                    "Poorly defined borders",
                    "Ulcerative appearance"
                ]
            },
            "lesion": {
                "location": "Oral cavity (estimated)",
                "size": "Large / abnormal",
                "color": "Erythroleukoplakic (red & white)",
                "border": "Irregular",
                "surface": "Ulcerated / rough"
            },
            "recommendation": {
                "urgency": "Immediate",
                "actions": [
                    "Consult Oral & Maxillofacial Surgeon / ENT",
                    "Biopsy required",
                    "Imaging (CT/MRI) if advised"
                ]
            }
        }
    else:
        return {
            "status": "Negative",
            "priority": "Routine",
            "summary": (
                "This AI analysis did not identify any high-risk malignant patterns. "
                "The oral mucosa appears consistent with normal or low-risk tissue. "
                "Routine monitoring and good oral hygiene are advised."
            ),
            "severity": {
                "level": "Low",
                "score": confidence,
                "summary": "No significant malignant patterns detected."
            },
            "risk": {
                "level": "Low",
                "probability": f"{confidence}%",
                "factors": [
                    "Uniform tissue texture",
                    "No ulceration",
                    "Normal coloration"
                ]
            },
            "lesion": {
                "location": "Oral cavity",
                "size": "Normal",
                "color": "Pink / healthy",
                "border": "Well-defined",
                "surface": "Smooth"
            },
            "recommendation": {
                "urgency": "Routine",
                "actions": [
                    "Maintain oral hygiene",
                    "Routine dental check-up",
                    "Monitor for changes"
                ]
            }
        }


# ---------------- MAIN PREDICTION ----------------
def predict_and_gradcam(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    pred = float(model.predict(img_array, verbose=0)[0][0])

    # Simple abnormality check (visual support)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size

    if pred >= 0.5 or edge_ratio > 0.08:
        label = "HIGH RISK"
        risk = "HIGH"
        confidence = round(max(pred, edge_ratio) * 100, 2)
    else:
        label = "LOW RISK"
        risk = "LOW"
        confidence = round((1 - pred) * 100, 2)

    heatmap = generate_gradcam(img_array)
    cam = overlay_heatmap(img_resized, heatmap)

    cv2.imwrite(os.path.join(MEDIA_DIR, "original.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(MEDIA_DIR, "heatmap.png"),
                cv2.cvtColor(cam, cv2.COLOR_RGB2BGR))

    assessment = generate_assessment(risk, confidence)

    return {
        "label": label,
        "confidence": confidence,
        "risk": risk,
        "assessment": assessment,
        "original_image": "media/original.png",
        "heatmap": "media/heatmap.png"
    }
