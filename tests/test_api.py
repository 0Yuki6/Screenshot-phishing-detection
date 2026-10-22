"""
Unit and integration tests for FastAPI application in api/main.py.
"""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from api.main import app, MAX_FILE_SIZE


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def create_test_image_bytes(format: str = "PNG", size=(200, 100), color=(255, 255, 255)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["offline_mode"] is True
    assert "version" in data


def test_analyze_unsupported_mime_type(client):
    file_bytes = b"Hello, this is plain text, not an image."
    response = client.post(
        "/analyze",
        files={"file": ("notes.txt", file_bytes, "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_analyze_empty_file(client):
    response = client.post(
        "/analyze",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_analyze_file_too_large(client):
    # Simulated oversize payload
    oversize_bytes = b"0" * (MAX_FILE_SIZE + 1024)
    response = client.post(
        "/analyze",
        files={"file": ("huge.png", oversize_bytes, "image/png")},
    )
    assert response.status_code == 413
    assert "exceeds maximum allowable limit" in response.json()["detail"]


def test_analyze_valid_image(client):
    img_bytes = create_test_image_bytes(format="PNG")
    response = client.post(
        "/analyze",
        files={"file": ("test_screenshot.png", img_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()

    # Validate required response schema
    assert "prediction_id" in data
    assert data["verdict"] in {"phishing", "legitimate", "uncertain"}
    assert 0.0 <= data["phishing_score"] <= 1.0
    assert data["channel"] in {"email", "sms", "whatsapp", "unknown"}
    assert isinstance(data["alerts"], list)
    assert isinstance(data["extracted_text"], str)
    assert isinstance(data["clean_text"], str)
    assert isinstance(data["rules_triggered"], list)
    assert "ocr_confidence" in data
    assert "winning_variant" in data
    assert "raw_ml_score" in data
    # Frontend compatibility fields
    assert "label" in data
    assert "confidence" in data
    assert "flagged_features" in data


def test_predict_alias_endpoint(client):
    img_bytes = create_test_image_bytes(format="JPEG")
    response = client.post(
        "/predict",
        files={"file": ("test_screenshot.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] in {"phishing", "legitimate", "uncertain"}
    assert 0.0 <= data["phishing_score"] <= 1.0
