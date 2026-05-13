import os
import numpy as np
from preprocessing.preprocessing import preprocess_image

def load_dataset(base_path):
    images, labels = [], []

    for label in os.listdir(base_path):
        class_path = os.path.join(base_path, label)
        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):
            try:
                img_path = os.path.join(class_path, file)
                images.append(preprocess_image(img_path))
                labels.append(label)
            except:
                pass

    return np.array(images), np.array(labels)
