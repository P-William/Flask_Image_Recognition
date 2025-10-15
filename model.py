"""Model utilities for preprocessing images and making predictions."""

from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image

# Load pre-trained model
model = load_model("digit_model.h5")


def preprocess_img(img_path):
    """
    Preprocess the input image for model prediction.

    Args:
        img_path (str or file-like): Path or stream of the image to preprocess.

    Returns:
        np.ndarray: Preprocessed image ready for prediction.
    """
    op_img = Image.open(img_path)
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, 224, 224, 3)
    return img_reshape


def predict_result(image_array):
    """
    Predict the digit class of a preprocessed image.

    Args:
        image_array (np.ndarray): Preprocessed image array.

    Returns:
        int: Predicted digit (0-9).
    """
    pred = model.predict(image_array)
    return int(np.argmax(pred[0], axis=-1))
