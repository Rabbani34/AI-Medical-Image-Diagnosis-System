import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

BASE_PATH = r"E:\medical_image_ai\dataset\xray"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
VAL_PATH = os.path.join(BASE_PATH, "val")

MODEL_PATH = r"E:\medical_image_ai\model\xray_model.h5"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_WARMUP = 5
EPOCHS_FINE = 10

# ======================
# DATA GENERATORS
# ======================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    VAL_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)
import json

# Save history
with open("model/history.json", "w") as f:
    json.dump(history.history, f)

print("Training history saved.")

print("Classes:", train_data.class_indices)

# ======================
# BASE MODEL
# ======================
base_model = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # Phase 1 freeze

# ======================
# HEAD
# ======================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
output = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# ======================
# PHASE 1 - WARMUP
# ======================
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n--- Phase 1 Training ---")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_WARMUP
)

# ======================
# PHASE 2 - FINE TUNE
# ======================
base_model.trainable = True

# Freeze early layers only
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n--- Phase 2 Fine Tuning ---")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_FINE
)

os.makedirs(r"E:\medical_image_ai\model", exist_ok=True)
model.save(MODEL_PATH)

print("✅ Model saved:", MODEL_PATH)