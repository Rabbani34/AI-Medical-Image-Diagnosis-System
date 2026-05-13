from tensorflow.keras.models import load_model
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_models():
    return {
        "xray": load_model(
            os.path.join(BASE_DIR, "saved_models", "xray_disease_model")
        )
    }
