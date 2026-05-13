import cv2
import numpy as np

IMG_SIZE = 224

def preprocess_image(image):

    # If PIL image convert to numpy
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    return image
