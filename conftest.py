"""Pytest configuration for testing the Flask app."""

import pytest
from app import app as flask_app  # Avoids redefinition warning


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    with flask_app.test_client() as client_instance:
        yield client_instance
