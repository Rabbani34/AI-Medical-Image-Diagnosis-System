import os
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from models.image_type_classifier import build_image_type_model
from data_loader.data_loader import load_dataset

# Ensure output directory exists
os.makedirs("saved_models", exist_ok=True)

paths = {
    "xray": "dataset/xray",
    "mri": "dataset/mri",
    "skin": "dataset/skin",
    "retina": "dataset/retina"
}

X, y = [], []

print("Loading datasets...")

for label, path in paths.items():
    images, _ = load_dataset(path)
    if len(images) == 0:
        raise ValueError(f"No images found in {path}")
    X.extend(images)
    y.extend([label] * len(images))


X = np.array(X)
y = np.array(y)

print("Total images:", len(X))

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = build_image_type_model(len(encoder.classes_))

print("Training image type model...")
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# 🔥 THIS IS THE IMPORTANT LINE
model.save("saved_models/image_type_model")

print("✅ Image type model saved successfully")


