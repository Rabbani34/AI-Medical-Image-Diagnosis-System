import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ======================
# MODEL PATH
# ======================
# CURRENT (probably wrong path):
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "xray_model.h5")

# ======================
# CLASS NAMES (MUST MATCH TRAINING ORDER)
# ======================
CLASS_NAMES = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

_model = None

# ======================
# LOAD MODEL
# ======================
def get_model():
    global _model

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            print("❌ Model file not found.")
            return None

        try:
            # 🔥 FIX: ignore optimizer (prevents weight_decay error)
            _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("✅ Model loaded successfully!")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None

    return _model


# ======================
# PREDICT FUNCTION
# ======================
def predict_image(image: Image.Image):

    model = get_model()
    if model is None:
        return None, None, None

    try:
        # Preprocess image
        image = image.convert("RGB")
        image = image.resize((224, 224))

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        preds = model.predict(img_array, verbose=0)[0]

        idx = np.argmax(preds)
        confidence = float(preds[idx] * 100)
        disease = CLASS_NAMES[idx]

        return disease, confidence, preds

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None, None, None


# ======================
# TEST MODEL (OPTIONAL)
# ======================
if __name__ == "__main__":
    model = get_model()
    if model is not None:
        print("Model Input Shape:", model.input_shape)