"""Unit tests for the digit classification model."""

# pylint: disable=redefined-outer-name,unused-argument

from io import BytesIO
import numpy as np
from keras.models import load_model
from PIL import UnidentifiedImageError
import pytest
from model import preprocess_img, predict_result

# Load the model before tests run
@pytest.fixture(scope="module")
def test_model():
    """Load the model before running tests."""
    return load_model("digit_model.h5")  # Adjust path if needed


def test_preprocess_img():
    """Test the preprocess_img function."""
    img_path = "test_images/2/Sign 2 (97).jpeg"
    processed_img = preprocess_img(img_path)

    # Check that the output shape is as expected
    assert processed_img.shape == (1, 224, 224, 3), (
        "Processed image shape should be (1, 224, 224, 3)"
    )

    # Check that values are normalized (between 0 and 1)
    assert np.min(processed_img) >= 0 and np.max(processed_img) <= 1, (
        "Image pixel values should be normalized between 0 and 1"
    )


def test_predict_result(test_model):
    """Test the predict_result function."""
    img_path = "test_images/4/Sign 4 (92).jpeg"  # Ensure the path is correct
    processed_img = preprocess_img(img_path)

    # Make a prediction
    prediction = predict_result(processed_img)

    # Print the prediction for debugging
    print(f"Prediction: {prediction} (Type: {type(prediction)})")

    # Check that the prediction is an integer (convert if necessary)
    assert isinstance(prediction, (int, np.integer)), "Prediction should be an integer class index"



def test_invalid_image_path():
    """Test preprocess_img with an invalid image path."""
    with pytest.raises(FileNotFoundError):
        preprocess_img("invalid/path/to/image.jpeg")

def test_image_shape_on_prediction(test_model):
    """Test the prediction output shape."""
    img_path = "test_images/5/Sign 5 (86).jpeg"  # Ensure the path is correct
    processed_img = preprocess_img(img_path)

    # Ensure that the prediction output is an integer
    prediction = predict_result(processed_img)
    assert isinstance(prediction, (int, np.integer)), "The prediction should be an integer"

def test_model_predictions_consistency(test_model):
    """Test that predictions for the same input are consistent."""
    img_path = "test_images/7/Sign 7 (54).jpeg"
    processed_img = preprocess_img(img_path)

    # Make multiple predictions
    predictions = [predict_result(processed_img) for _ in range(5)]

    # Check that all predictions are the same
    assert all(p == predictions[0] for p in predictions), (
        "Predictions for the same input should be consistent"
    )

def test_unsupported_file_type():
    """Test preprocess_img with an unsupported file type."""
    with pytest.raises(UnidentifiedImageError):
        preprocess_img("invalid_file.txt")


def test_empty_file_input():
    """Test preprocess_img with an empty file input."""
    empty_file = BytesIO()  # Simulate an empty file
    with pytest.raises(UnidentifiedImageError):
        preprocess_img(empty_file)

def test_prediction_on_random_noise(test_model):
    """Test predict_result with random noise as input."""
    random_noise = np.random.rand(1, 224, 224, 3).astype(np.float32)
    prediction = predict_result(random_noise)

    # Check that the prediction is an integer
    assert isinstance(prediction, (int, np.integer)), "Prediction should be an integer class index"
