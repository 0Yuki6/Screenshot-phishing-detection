"""
FastAPI application for the Phishing Screenshot Detector.

Endpoints:
- POST /analyze: Multipart image upload (max 10MB), returns phishing prediction.
- POST /predict: Alias endpoint for React frontend compatibility.
- GET /health: Healthcheck endpoint.
"""

import io
import os
import uuid
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.pipeline import PhishingPipeline

logger = logging.getLogger("phishing_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Phishing Screenshot Detector API",
    description="High-precision screenshot-based phishing detection across Email, SMS, and WhatsApp.",
    version="1.0.0",
)

# CORS Middleware (permitting local frontend development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
}
STORE_UPLOADS = os.environ.get("STORE_UPLOADS", "false").lower() == "true"
UPLOADS_DIR = Path("data/uploads")

if STORE_UPLOADS:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Lazy/Global Pipeline Instance
pipeline: Optional[PhishingPipeline] = None


def get_pipeline_instance() -> PhishingPipeline:
    global pipeline
    if pipeline is None:
        pipeline = PhishingPipeline()
    return pipeline


class AnalysisResponse(BaseModel):
    prediction_id: str = Field(..., description="Unique analysis identifier")
    verdict: str = Field(..., description="One of: 'phishing', 'legitimate', 'uncertain'")
    phishing_score: float = Field(..., description="Calibrated risk score between 0.0 and 1.0")
    channel: str = Field(..., description="Detected channel: 'email', 'sms', 'whatsapp'")
    alerts: List[str] = Field(default_factory=list, description="Plain-English explanation of risk factors")
    extracted_text: str = Field(..., description="Raw text extracted via OCR")
    clean_text: str = Field(..., description="Text normalized after template chrome removal")
    ocr_confidence: float = Field(..., description="Mean OCR word confidence (0-100)")
    winning_variant: str = Field(..., description="Winning image preprocessing variant")
    rules_triggered: List[str] = Field(default_factory=list, description="Names of triggered deterministic rules")
    raw_ml_score: float = Field(..., description="Unscaled probability from random forest")
    suspicion_reasons: List[str] = Field(default_factory=list, description="Top SHAP features raising suspicion")
    safe_reasons: List[str] = Field(default_factory=list, description="Top SHAP features pointing to legitimate")
    # Compatibility fields for legacy frontend
    label: str = Field(..., description="Legacy alias for verdict")
    confidence: float = Field(..., description="Legacy decision confidence")
    flagged_features: List[str] = Field(default_factory=list, description="List of alert summaries")


@app.get("/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "offline_mode": True,
        "store_uploads": STORE_UPLOADS,
    }


def strip_exif_and_decode(file_bytes: bytes) -> np.ndarray:
    """
    Decodes an image in-memory, stripping all EXIF metadata.
    Returns a BGR numpy array suitable for OpenCV processing.
    """
    try:
        with Image.open(io.BytesIO(file_bytes)) as pil_img:
            # Force conversion to RGB (strips EXIF, palette, orientation tags)
            rgb_img = pil_img.convert("RGB")
            # Convert to numpy array (RGB) then BGR for OpenCV
            rgb_arr = np.array(rgb_img)
            bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
            return bgr_arr
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted image format: {exc}",
        )


async def process_image_upload(file: UploadFile) -> AnalysisResponse:
    """Core image processing logic shared by /analyze and /predict."""
    # 1. MIME Validation
    content_type = file.content_type or ""
    if content_type.lower() not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{content_type}'. Supported formats: PNG, JPEG, WebP.",
        )

    # 2. Size Validation (Max 10 MB)
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowable limit of 10 MB (received {len(file_bytes)/(1024*1024):.2f} MB).",
        )
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # 3. Optional persistence
    prediction_id = str(uuid.uuid4())
    if STORE_UPLOADS:
        ext = Path(file.filename or "image.png").suffix or ".png"
        saved_path = UPLOADS_DIR / f"{prediction_id}{ext}"
        try:
            with open(saved_path, "wb") as f_out:
                f_out.write(file_bytes)
        except Exception as e:
            logger.warning("Failed to store upload %s: %s", saved_path, e)

    # 4. In-Memory EXIF stripping & image decoding
    img_bgr = strip_exif_and_decode(file_bytes)

    # 5. Pipeline Inference
    pipe = get_pipeline_instance()
    result = pipe.predict_image(img_bgr)

    # Legacy confidence calculation for backwards compatibility
    if result["verdict"] == "phishing":
        legacy_confidence = result["phishing_score"]
    elif result["verdict"] == "legitimate":
        legacy_confidence = round(1.0 - result["phishing_score"], 4)
    else:
        legacy_confidence = 0.50

    return AnalysisResponse(
        prediction_id=prediction_id,
        verdict=result["verdict"],
        phishing_score=result["phishing_score"],
        channel=result["channel"],
        alerts=result["alerts"],
        extracted_text=result["extracted_text"],
        clean_text=result["clean_text"],
        ocr_confidence=result["ocr_confidence"],
        winning_variant=result["winning_variant"],
        rules_triggered=result["rules_triggered"],
        raw_ml_score=result["raw_ml_score"],
        suspicion_reasons=result.get("suspicion_reasons", []),
        safe_reasons=result.get("safe_reasons", []),
        label=result["verdict"],
        confidence=legacy_confidence,
        flagged_features=result.get("suspicion_reasons", []) + result["alerts"],
    )


@app.post("/analyze", response_model=AnalysisResponse, tags=["Inference"])
async def analyze_screenshot(file: UploadFile = File(...)) -> AnalysisResponse:
    """
    Primary analysis endpoint for screenshot phishing detection.
    Accepts multipart/form-data with max 10MB image upload.
    """
    return await process_image_upload(file)


@app.post("/predict", response_model=AnalysisResponse, tags=["Inference"])
async def predict_screenshot(file: UploadFile = File(...)) -> AnalysisResponse:
    """
    Frontend alias endpoint matching the React client API route.
    """
    return await process_image_upload(file)
