"""Acceptance tests for the /prediction endpoint of the app."""

from io import BytesIO

def upload_test_image(client):
    """Helper to upload a mock image for prediction tests."""
    img_data = BytesIO(b"fake_image_data")
    img_data.name = "test.jpg"
    return client.post(
        "/prediction",
        data={"file": (img_data, img_data.name)},
        content_type="multipart/form-data"
    )

def test_successful_prediction(client):
    """Test the successful image upload and prediction."""
    response = upload_test_image(client)
    assert response.status_code == 200
    assert b"Prediction" in response.data
