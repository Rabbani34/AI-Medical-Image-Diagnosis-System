import numpy as np
import tensorflow as tf
import cv2
from PIL import Image


# ------------------------------------------------
# Find last convolution layer automatically
# ------------------------------------------------
def get_last_conv_layer(model):

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    raise ValueError("No Conv2D layer found in model.")


# ------------------------------------------------
# Prepare image for model
# ------------------------------------------------
def preprocess_image(image, model):

    input_shape = model.input_shape
    expected_channels = input_shape[-1]

    image = image.resize((224, 224))

    img_array = np.array(image)

    # If model expects grayscale
    if expected_channels == 1:

        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        img_array = np.expand_dims(img_array, axis=-1)

    # If model expects RGB
    else:

        image = image.convert("RGB")
        img_array = np.array(image)

    img_array = img_array.astype("float32") / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ------------------------------------------------
# GradCAM generation
# ------------------------------------------------
def make_gradcam(image, model):

    img_array = preprocess_image(image, model)

    last_conv_layer_name = get_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        class_index = tf.argmax(predictions[0])

        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


# ------------------------------------------------
# Overlay heatmap on image
# ------------------------------------------------
def overlay_heatmap(image, heatmap, alpha=0.4):

    image = image.convert("RGB")
    img = np.array(image)

    h, w = img.shape[:2]

    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)

    return overlay


def detect_infection_area(heatmap, threshold=0.6):

    heatmap = cv2.resize(heatmap, (224,224))

    mask = heatmap > threshold

    mask = mask.astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)

        if w*h > 200:  # ignore tiny noise
            boxes.append((x,y,w,h))

    infection_percent = (np.sum(mask)/255) / (224*224) * 100

    return boxes, infection_percent