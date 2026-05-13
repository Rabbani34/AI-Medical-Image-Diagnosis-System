import numpy as np
import tensorflow as tf
import cv2

# ---------------------------------------
# Auto-detect last convolution layer
# ---------------------------------------
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")

# ---------------------------------------
# GradCAM Generator
# ---------------------------------------
def make_gradcam(image, model):

    input_shape = model.input_shape

    # Get expected channels
    expected_channels = input_shape[-1]

    image = image.resize((224, 224))

    img_array = np.array(image)

    # If grayscale expected
    if expected_channels == 1:
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        img_array = np.expand_dims(img_array, axis=-1)

    # If RGB expected
    else:
        image = image.convert("RGB")
        img_array = np.array(image)

    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

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

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def overlay_heatmap(image, heatmap):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    img = np.array(image)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img

    return np.uint8(superimposed_img)