"""
predict.py — Model loader + prediction
=======================================
Auto-downloads model from HuggingFace on Render if not found locally.
Set MODEL_URL environment variable to your HuggingFace model URL.
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "xray_model.h5")

# ── HuggingFace URL (set as env var on Render dashboard) ──────────────────────
MODEL_URL = os.environ.get("MODEL_URL", "")

CLASS_NAMES = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

_model = None


def _download_model():
    if not MODEL_URL:
        print("❌ MODEL_URL env var not set. Cannot download model.")
        return False

    import urllib.request
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"⬇  Downloading model from HuggingFace...")

    try:
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(downloaded / total_size * 100, 100)
                print(f"\r   {pct:.1f}%", end="", flush=True)

        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=_progress)
        print(f"\n✅ Model downloaded successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False


def get_model():
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        if not _download_model():
            return None

    try:
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully!")
        return _model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def predict_image(image: Image.Image):
    model = get_model()
    if model is None:
        return None, None, None

    try:
        image      = image.convert("RGB").resize((224, 224))
        img_array  = np.array(image, dtype="float32") / 255.0
        img_array  = np.expand_dims(img_array, axis=0)
        preds      = model.predict(img_array, verbose=0)[0]
        idx        = int(np.argmax(preds))
        return CLASS_NAMES[idx], float(preds[idx] * 100), preds
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None, None, None


if __name__ == "__main__":
    m = get_model()
    if m:
        print("Input shape:", m.input_shape)
