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
from src.masking import strip_template_chrome, mask_text
from src.feature_extraction import extract_features, FEATURE_NAMES
from src.display_indicators import compute_display_indicators, format_indicator_signals

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
MODEL_PATH = "data/models/calibrated_model.pkl" if os.path.exists("data/models/calibrated_model.pkl") else "data/models/model.pkl"
TFIDF_PATH = "data/models/tfidf_vectorizer.pkl"
FEATURE_NAMES_PATH = "data/models/feature_names.json"
THRESHOLDS_PATH = "data/models/thresholds.json"
FEEDBACK_PATH = "data/feedback.jsonl"

_model = None
_tfidf = None
_feature_names = None
_explainer = None
_t_legit = 0.35
_t_phish = 0.65


@app.on_event("startup")
def load_artifacts():
    """Load model, TF-IDF vectorizer, thresholds, and SHAP explainer at startup."""
    global _model, _tfidf, _feature_names, _explainer, _t_legit, _t_phish

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

    if os.path.exists(THRESHOLDS_PATH):
        try:
            with open(THRESHOLDS_PATH, "r") as f:
                th_data = json.load(f)
                _t_legit = float(th_data.get("t_legit", 0.35))
                _t_phish = float(th_data.get("t_phish", 0.65))
            logger.info("✓ Thresholds loaded: t_legit=%.3f, t_phish=%.3f", _t_legit, _t_phish)
        except Exception as e:
            logger.warning("Failed to load thresholds from %s: %s", THRESHOLDS_PATH, e)

    expected_names = list(FEATURE_NAMES)
    if _tfidf is not None:
        expected_names += [f"tfidf_{n}" for n in _tfidf.get_feature_names_out()]
    _feature_names = expected_names

    if os.path.exists(FEATURE_NAMES_PATH):
        try:
            with open(FEATURE_NAMES_PATH, "r") as f:
                stored_names = json.load(f)
            if stored_names != expected_names:
                logger.error(
                    "Feature names mismatch: stored has %d features, expected %d. Updating feature names.",
                    len(stored_names), len(expected_names),
                )
            else:
                logger.info("✓ Feature names validated against feature_names.json (%d features)", len(_feature_names))
        except Exception as e:
            logger.warning("Error reading feature_names.json: %s", e)

    # Initialize SHAP explainer
    if _model is not None:
        try:
            import shap
            from sklearn.calibration import CalibratedClassifierCV
            base = _model
            if isinstance(base, CalibratedClassifierCV):
                if hasattr(base, 'calibrated_classifiers_') and len(base.calibrated_classifiers_) > 0:
                    cc = base.calibrated_classifiers_[0]
                    base = getattr(cc, 'estimator', getattr(cc, 'base_estimator', base))
            _explainer = shap.TreeExplainer(base)
            logger.info("✓ SHAP TreeExplainer initialized")
        except Exception as e:
            logger.warning("SHAP explainer unavailable: %s", e)
            _explainer = None


def _build_feature_vector(ocr_text: str, channel: str) -> np.ndarray:
    """
    Build the full feature vector (scalar features + TF-IDF) for a single sample.
    Ensures identical preprocessing (chrome stripping + entity masking) as training.
    """
    clean_text = strip_template_chrome(ocr_text, channel=channel)
    features = extract_features(clean_text, channel, skip_whois=True)
    scalar_vector = np.array(
        [features[name] for name in FEATURE_NAMES],
        dtype=np.float64,
    ).reshape(1, -1)

    # TF-IDF features (must match training: masked entities)
    if _tfidf is not None:
        masked_text = mask_text(clean_text)
        tfidf_vector = _tfidf.transform([masked_text])
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



# ── SHAP Explanation Helpers ─────────────────────────────────────────
_FEATURE_LABELS = {
    "urgency_count": "Uses urgent or pressure language",
    "threat_count": "Contains threatening or consequence language",
    "credential_request": "Asks for login or payment credentials",
    "personal_info_request": "Requests personal identity information",
    "financial_context_count": "Mentions banking or financial context",
    "reward_bait_count": "Contains prize, reward, or bait language",
    "sensitive_request": "Directly requests sensitive credentials",
    "generic_greeting_flag": "Uses an impersonal or generic greeting",
    "threat_plus_urgency": "Combines threat and urgency language",
    "brand_domain_mismatch": "Domain doesn't match the claimed brand",
    "has_high_risk_tld": "Link uses a high-risk domain extension",
    "blacklist_status": "Domain is flagged by threat databases",
    "typosquatting_score": "Domain resembles a known brand (typosquatting)",
    "is_shortened_url": "Uses a URL shortener to hide destination",
    "raw_ip_address": "Link goes to a raw IP address",
    "has_punycode": "URL uses obfuscated characters (Punycode)",
    "redirect_domain_change": "URL redirects to a different domain",
    "suspicious_url_keyword_count": "URL contains deceptive keywords",
    "number_of_subdomains": "Excessive subdomains in URL",
    "special_character_count": "Many special characters in URL",
    "whitelist_match": "Domain matches a verified official domain",
    "uses_https": "Link uses HTTPS encryption",
    "total_word_count": "Overall message length",
    "action_word_count": "Contains call-to-action words",
    "account_security_count": "Mentions account security topics",
    "url_length": "URL is unusually long",
    "url_entropy": "URL has high character randomness",
    "domain_entropy": "Domain name appears random",
    "has_dns_record": "Domain has valid DNS records",
    "domain_length": "Domain name length",
    "brand_in_domain": "Brand name appears in the domain",
    "channel": "Message channel type",
}


def _translate_feature(name: str) -> str:
    """Translate a raw feature name into a plain-English sentence."""
    if name in _FEATURE_LABELS:
        return _FEATURE_LABELS[name]
    if name.startswith("tfidf_"):
        token = name[6:]
        return f"Text contains the pattern '{token}'"
    return name.replace("_", " ").capitalize()


def _get_shap_explanations(feature_vector: np.ndarray, top_n: int = 3):
    """Return (suspicion_reasons, safe_reasons) based on SHAP values."""
    suspicion = []
    safe = []
    if _explainer is None or not _feature_names:
        return suspicion, safe

    try:
        shap_values = _explainer.shap_values(feature_vector)
        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif getattr(shap_values, "ndim", 0) == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        pairs = list(zip(_feature_names, sv))
        positive = sorted([(n, v) for n, v in pairs if v > 0.001], key=lambda x: -x[1])
        for name, _ in positive[:top_n]:
            suspicion.append(_translate_feature(name))

        negative = sorted([(n, v) for n, v in pairs if v < -0.001], key=lambda x: x[1])
        for name, _ in negative[:top_n]:
            safe.append(_translate_feature(name))
    except Exception as e:
        logger.warning("SHAP explanation failed: %s", e)

    return suspicion, safe


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

    # Class 1 predict_proba
    proba = _model.predict_proba(feature_vector)[0]
    classes = list(_model.classes_)
    phish_idx = classes.index(1) if 1 in classes else 1
    p_phish = float(proba[phish_idx])
    phishing_score = p_phish
    confidence = max(p_phish, 1.0 - p_phish)

    # Threshold-based verdict
    if p_phish >= _t_phish:
        verdict = "phishing"
    elif p_phish <= _t_legit:
        verdict = "legitimate"
    else:
        verdict = "uncertain"

    val_map = {name: feature_vector[0, idx] for idx, name in enumerate(_feature_names) if idx < feature_vector.shape[1]}

    # Hard overrides allowed ONLY for:
    # 1. Known malicious URL/domain (blacklist hit) -> force phishing
    # 2. Confirmed brand impersonation (brand detected, domain present and does not match brand's official domain) -> force phishing
    # 3. High-risk TLD + credential-harvesting keywords together -> force phishing
    # 4. Whitelist match may DOWNGRADE phishing -> uncertain, but NEVER directly to legitimate if the model flagged it
    is_blacklist = val_map.get("blacklist_status", 0) > 0
    is_mismatch = val_map.get("brand_domain_mismatch", 0) > 0
    is_high_risk_combo = (val_map.get("has_high_risk_tld", 0) > 0) and (
        val_map.get("credential_request", 0) > 0 or val_map.get("sensitive_request", 0) > 0
    )

    override_reason = None
    if is_blacklist:
        verdict = "phishing"
        phishing_score = max(phishing_score, 0.95)
        confidence = max(confidence, 0.95)
        override_reason = "Known malicious URL/domain detected by threat database"
    elif is_mismatch:
        verdict = "phishing"
        phishing_score = max(phishing_score, 0.95)
        confidence = max(confidence, 0.95)
        override_reason = "Domain does not match the claimed brand (brand impersonation)"
    elif is_high_risk_combo:
        verdict = "phishing"
        phishing_score = max(phishing_score, 0.95)
        confidence = max(confidence, 0.95)
        override_reason = "High-risk domain extension combined with credential harvesting keywords"
    elif val_map.get("whitelist_match", 0) > 0:
        if verdict == "phishing":
            verdict = "uncertain"
            phishing_score = (_t_phish + _t_legit) / 2.0
            confidence = 0.50
            override_reason = "Verified whitelist match downgraded phishing to uncertain"

    suspicion_reasons, safe_reasons = _get_shap_explanations(feature_vector, top_n=3)
    if override_reason and override_reason not in suspicion_reasons:
        suspicion_reasons.insert(0, override_reason)

    flagged = _get_flagged_features(feature_vector, confidence, ocr_text, channel, label=verdict)

    # Compute display-only indicators from clean text
    clean_text = strip_template_chrome(ocr_text, channel=channel)
    rules_triggered = ["BRAND_IMPERSONATION_MISMATCH"] if is_mismatch else []
    alerts = [override_reason] if override_reason else []
    display_indicators = compute_display_indicators(
        clean_text, rules_triggered=rules_triggered, alerts=alerts
    )
    display_signals = format_indicator_signals(display_indicators)
    combined_suspicion = display_signals + [s for s in suspicion_reasons if s not in display_signals]
    combined_flagged = display_signals + [s for s in (flagged if flagged else suspicion_reasons) if s not in display_signals]

    prediction_id = str(uuid4())

    logger.info(
        "Prediction: %s (score: %.4f, confidence: %.4f) | Channel: %s | ID: %s",
        verdict, phishing_score, confidence, channel, prediction_id,
    )

    return {
        "prediction_id": prediction_id,
        "channel": channel,
        "verdict": verdict,
        "label": verdict,
        "phishing_score": round(phishing_score, 4),
        "confidence": round(confidence, 4),
        "suspicion_reasons": combined_suspicion,
        "safe_reasons": safe_reasons,
        "flagged_features": combined_flagged,
        "alerts": alerts,
        "rules_triggered": rules_triggered,
        "indicators": display_indicators,
        "display_indicators": display_indicators,
        "channel_scores": channel_result.get("scores", {}),
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
