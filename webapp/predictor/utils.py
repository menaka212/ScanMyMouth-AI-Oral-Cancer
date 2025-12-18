import os
import cv2
import numpy as np
import tensorflow as tf

# suppress TF warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# BASE DIR = project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load model ONCE
base_model = tf.keras.applications.EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=x)

# load trained weights
model.load_weights(os.path.join(BASE_DIR, 'models', 'efficientnet_histo_weights'))

last_conv_layer = base_model.get_layer("top_conv")


def predict_and_gradcam(image_path):
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_norm = img / 255.0
    img_array = np.expand_dims(img_norm, axis=0)

    # Prediction
    pred = model.predict(img_array)[0][0]
    confidence = round(float(pred) * 100, 2)

    result = "CANCER" if pred >= 0.5 else "NON-CANCER"

    # ---------- GRAD-CAM ----------
    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # ---------- SAVE HEATMAP ----------
    media_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'media'
)

    os.makedirs(media_dir, exist_ok=True)

    heatmap_path = os.path.join(media_dir, 'heatmap.png')
    cv2.imwrite(heatmap_path, superimposed)
    print("Heatmap saved at:", heatmap_path)
    return result, confidence
