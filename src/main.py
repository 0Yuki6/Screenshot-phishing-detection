"""
FastAPI service for the phishing detection pipeline.

Endpoints:
    POST /predict    — Upload a screenshot → get phishing/legitimate verdict
    GET  /health     — Service health check
    POST /feedback   — Submit label correction for future retraining

Startup:
    Loads the trained model and TF-IDF vectorizer from data/models/.
    If no model exists, returns a clear error on /predict.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from dotenv import load_dotenv
load_dotenv()  # Load .env file (contains GOOGLE_SAFE_BROWSING_API_KEY)

import numpy as np
import joblib
from scipy.sparse import csr_matrix, hstack
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.preprocessing import preprocess_image
from src.ocr import extract_text
from src.channel_detector import detect_channel
from src.feature_extraction import extract_features, FEATURE_NAMES

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── App setup ────────────────────────────────────────────────────────
app = FastAPI(
    title="Phishing Detection API",
    description=(
        "Upload a screenshot of an email, SMS, or WhatsApp message "
        "to determine whether it's phishing or legitimate."
    ),
    version="0.1.0",
)

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model loading ────────────────────────────────────────────────────
MODEL_PATH = "data/models/model.pkl"
TFIDF_PATH = "data/models/tfidf_vectorizer.pkl"
FEATURE_NAMES_PATH = "data/models/feature_names.json"
FEEDBACK_PATH = "data/feedback.jsonl"

_model = None
_tfidf = None
_feature_names = None


@app.on_event("startup")
def load_artifacts():
    """Load model and TF-IDF vectorizer at startup."""
    global _model, _tfidf, _feature_names

    if os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
        logger.info("✓ Model loaded from %s", MODEL_PATH)
    else:
        logger.warning("⚠ No model found at %s. /predict will return an error.", MODEL_PATH)

    if os.path.exists(TFIDF_PATH):
        _tfidf = joblib.load(TFIDF_PATH)
        logger.info("✓ TF-IDF vectorizer loaded from %s", TFIDF_PATH)
    else:
        logger.warning("⚠ No TF-IDF vectorizer found at %s.", TFIDF_PATH)

    if os.path.exists(FEATURE_NAMES_PATH):
        with open(FEATURE_NAMES_PATH, "r") as f:
            _feature_names = json.load(f)
        logger.info("✓ Feature names loaded (%d features)", len(_feature_names))


def _build_feature_vector(ocr_text: str, channel: str) -> np.ndarray:
    """
    Build the full feature vector (scalar features + TF-IDF) for a single sample.
    """
    # Scalar features
    features = extract_features(ocr_text, channel, skip_whois=True)
    scalar_vector = np.array(
        [features[name] for name in FEATURE_NAMES],
        dtype=np.float64,
    ).reshape(1, -1)

    # TF-IDF features
    if _tfidf is not None:
        tfidf_vector = _tfidf.transform([ocr_text])
        combined = hstack([csr_matrix(scalar_vector), tfidf_vector]).toarray()
    else:
        combined = scalar_vector

    return combined


def _get_flagged_features(
    feature_vector: np.ndarray,
    prediction_proba: float,
    ocr_text: str,
    channel: str,
    label: str = "phishing",
) -> list:
    """
    Identify the top contributing features for explainability.
    Instead of global model importances, we look at the specific features
    triggered by this message and return human-readable alerts.
    """
    if _feature_names is None:
        return []

    flagged = []
    
    # Map feature values to dictionary for easy access
    val_map = {name: feature_vector[0, idx] for idx, name in enumerate(_feature_names) if idx < feature_vector.shape[1]}
    
    # ── High-priority explicit indicators ──
    if val_map.get("blacklist_status", 0) > 0:
        flagged.append("Domain blacklisted by Google Safe Browsing")
    if val_map.get("typosquatting_score", 0) > 0:
        flagged.append(f"High typosquatting risk (score: {val_map['typosquatting_score']:.1f})")
    if val_map.get("is_shortened_url", 0) > 0:
        flagged.append("Suspicious URL shortener detected")
    if val_map.get("raw_ip_address", 0) > 0:
        flagged.append("URL uses a raw IP address instead of a domain")
    if val_map.get("number_of_subdomains", 0) > 2:
        flagged.append(f"Excessive subdomains ({int(val_map['number_of_subdomains'])})")
    if val_map.get("special_character_count", 0) > 5 and val_map.get("whitelist_match", 0) == 0:
        flagged.append(f"Excessive special characters in URL ({int(val_map['special_character_count'])})")
        
    # ── New Text Group Indicators ──
    if val_map.get("threat_plus_urgency", 0) > 0:
        flagged.append("Combined threat and urgency language detected")
    elif val_map.get("urgency_count", 0) > 0:
        flagged.append(f"Urgent or pressure language detected ({int(val_map['urgency_count'])} keywords)")
    elif val_map.get("threat_count", 0) > 0:
        flagged.append(f"Threatening or consequence language detected ({int(val_map['threat_count'])} keywords)")
        
    if val_map.get("sensitive_request", 0) > 0:
        flagged.append("Direct request for sensitive credentials or personal info")
    elif val_map.get("credential_request", 0) > 0:
        flagged.append("Requests login/payment credentials")
    
    if val_map.get("financial_context_count", 0) > 0:
        flagged.append("Financial or banking context detected")
    if val_map.get("reward_bait_count", 0) > 0:
        flagged.append("Prize, reward, or bait language detected")
    if val_map.get("generic_greeting_flag", 0) > 0:
        flagged.append("Impersonal or generic greeting detected")

    # ── New URL/Domain Group Indicators ──
    if val_map.get("brand_domain_mismatch", 0) > 0:
        flagged.append("Domain does not match the brand mentioned in text")
    if val_map.get("has_punycode", 0) > 0:
        flagged.append("URL uses obfuscated (Punycode) characters")
    if val_map.get("url_was_unrolled", 0) > 0 and val_map.get("redirect_domain_change", 0) > 0:
        flagged.append("URL redirects to a completely different domain")
    if val_map.get("suspicious_url_keyword_count", 0) > 0:
        flagged.append("URL contains deceptive security/login keywords")
    if val_map.get("domain_age", 0) != -1 and val_map.get("domain_age", 0) < 30:
        flagged.append("Newly registered domain (less than 30 days old)")
        
    # Check TF-IDF features for top keywords
    tfidf_matches = []
    # Common benign tokens that appear in TF-IDF features but are not
    # meaningful phishing indicators on their own.
    tfidf_stopwords = {
        "com", "www", "http", "https", "org", "net", "html", "php",
        "today", "yesterday", "tomorrow", "monday", "tuesday",
        "wednesday", "thursday", "friday", "saturday", "sunday",
        "morning", "evening", "night", "hello", "thanks", "thank",
        "please", "sorry", "yes", "the", "and", "for", "you", "are",
        "this", "that", "with", "from", "have", "been", "will",
    }
    for name, value in val_map.items():
        if name.startswith("tfidf_") and value > 0.15:
            word = name.split("_", 1)[1]
            # Skip purely numeric tokens, single characters, or very short
            # tokens that look meaningless to the user (e.g. "0 2", "ww")
            if len(word) <= 2 or word.replace(" ", "").isdigit():
                continue
            # Skip common benign words
            if word.lower() in tfidf_stopwords:
                continue
            tfidf_matches.append(word)
    
    if tfidf_matches:
        flagged.append(f"Suspicious vocabulary matched: {', '.join(tfidf_matches[:4])}")

    # Fallback if nothing was caught but model still flagged it as phishing
    if not flagged and label == "phishing" and prediction_proba > 0.5:
        flagged.append("Overall linguistic structure matches known phishing templates")

    return flagged



# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Basic service health check."""
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "tfidf_loaded": _tfidf is not None,
    }


@app.post("/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    bypass_ocr: bool = Query(
        False,
        description=(
            "If true, use the 'text' form field directly instead of running OCR "
            "on an uploaded image. Useful for testing the pipeline without Tesseract."
        ),
    ),
):
    """
    Analyze a screenshot for phishing content.

    Either upload an image file (PNG/JPEG/WebP) or set `bypass_ocr=true`
    and provide raw text in the `text` form field.

    Returns:
        - channel: detected message source (email/sms/whatsapp/unknown)
        - label: "phishing" or "legitimate"
        - confidence: float (0-1)
        - flagged_features: top contributing features
        - prediction_id: unique ID for feedback submission
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained model available. Run 'python -m src.generate_synthetic_data' "
                "to train a placeholder model."
            ),
        )

    # ── Get OCR text ─────────────────────────────────────────────────
    if bypass_ocr:
        if not text:
            raise HTTPException(
                status_code=400,
                detail="When bypass_ocr=true, a 'text' form field is required.",
            )
        ocr_text = text
    else:
        if file is None:
            raise HTTPException(
                status_code=400,
                detail="An image file is required (or set bypass_ocr=true with a text field).",
            )
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            preprocessed = preprocess_image(image_bytes)
            ocr_text = extract_text(preprocessed)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Image processing error: {e}")
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

    if not ocr_text.strip():
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted from the image. Try a clearer screenshot.",
        )

    # ── Pipeline ─────────────────────────────────────────────────────
    channel_result = detect_channel(ocr_text)
    channel = channel_result["channel"]

    feature_vector = _build_feature_vector(ocr_text, channel)

    prediction = _model.predict(feature_vector)[0]
    proba = _model.predict_proba(feature_vector)[0]

    label = "phishing" if prediction == 1 else "legitimate"
    confidence = float(proba[prediction])

    # ── Hardened Scoring & Decision Override ─────────────────────────
    # Identify high-risk flags from the feature vector using val_map
    val_map = {name: feature_vector[0, idx] for idx, name in enumerate(_feature_names) if idx < feature_vector.shape[1]}
    
    is_mismatch = val_map.get("brand_domain_mismatch", 0) > 0
    is_high_risk_link = (val_map.get("has_high_risk_tld", 0) > 0) or (val_map.get("blacklist_status", 0) > 0)

    if (is_mismatch or is_high_risk_link) and label == "legitimate":
        # Force verdict to phishing, override confidence to a high value
        label = "phishing"
        confidence = 0.95
        logger.info("Hardened Scoring: Overriding NLP prediction to phishing due to high-risk heuristic.")

    # Whitelist safeguard: If the URL is on a verified legitimate domain (e.g. YouTube, Google)
    # and has zero threat / credential / phishing signals, prevent false positives from query tracking parameters
    is_whitelisted = val_map.get("whitelist_match", 0) > 0
    has_zero_threat_signals = (
        val_map.get("blacklist_status", 0) == 0
        and val_map.get("brand_domain_mismatch", 0) == 0
        and val_map.get("typosquatting_score", 0) == 0
        and val_map.get("has_high_risk_tld", 0) == 0
        and val_map.get("raw_ip_address", 0) == 0
        and val_map.get("credential_request", 0) == 0
        and val_map.get("personal_info_request", 0) == 0
        and val_map.get("sensitive_request", 0) == 0
        and val_map.get("threat_plus_urgency", 0) == 0
        and val_map.get("threat_count", 0) == 0
        and val_map.get("urgency_count", 0) == 0
        and val_map.get("reward_bait_count", 0) == 0
        and val_map.get("redirect_domain_change", 0) == 0
    )

    if is_whitelisted and has_zero_threat_signals and label == "phishing":
        label = "legitimate"
        confidence = 0.95
        logger.info("Whitelist Safeguard: Overriding false positive to legitimate for verified whitelisted domain with zero threats.")

    flagged = _get_flagged_features(feature_vector, confidence, ocr_text, channel, label=label)

    # ── Append Override Explanations to Key Indicators ───────────────
    if (is_mismatch or is_high_risk_link) and "High-risk URL heuristics triggered an automatic phishing verdict" not in flagged:
         # Put the heuristic alert at the top
         flagged.insert(0, "High-risk URL heuristics triggered an automatic phishing verdict")

    prediction_id = str(uuid4())

    logger.info(
        "Prediction: %s (%.2f confidence) | Channel: %s | ID: %s",
        label, confidence, channel, prediction_id,
    )

    return {
        "prediction_id": prediction_id,
        "channel": channel,
        "label": label,
        "confidence": round(confidence, 4),
        "flagged_features": flagged,
        "channel_scores": channel_result["scores"],
    }


@app.post("/feedback")
async def submit_feedback(
    prediction_id: str = Form(..., description="The prediction_id from a /predict response"),
    correct_label: str = Form(..., description="The correct label: 'phishing' or 'legitimate'"),
):
    """
    Submit a label correction for a previous prediction.

    Stores the feedback in a JSONL file for future retraining.
    """
    if correct_label not in ("phishing", "legitimate"):
        raise HTTPException(
            status_code=400,
            detail="correct_label must be 'phishing' or 'legitimate'.",
        )

    feedback_entry = {
        "prediction_id": prediction_id,
        "correct_label": correct_label,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(FEEDBACK_PATH) or ".", exist_ok=True)
    with open(FEEDBACK_PATH, "a") as f:
        f.write(json.dumps(feedback_entry) + "\n")

    logger.info("Feedback received for prediction %s: %s", prediction_id, correct_label)

    return {
        "status": "accepted",
        "prediction_id": prediction_id,
        "correct_label": correct_label,
    }
