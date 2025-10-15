"""Integration tests for the /prediction endpoint of the app."""

# pylint: disable=redefined-outer-name,unused-argument

import pytest

from app import app

@pytest.fixture
def test_client():
    """Fixture for the Flask test client."""
    with app.test_client() as client:
        yield client

def test_missing_file(test_client):
    """Test the prediction route with a missing file."""
    response = test_client.post("/prediction", data={}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"File cannot be processed." in response.data  # Check if the error message is displayed
