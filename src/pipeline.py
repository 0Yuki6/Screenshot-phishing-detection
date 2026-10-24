"""
End-to-end inference pipeline for the Phishing Screenshot Detector.

Coordinates:
1. Image preprocessing and clamped resizing (src/preprocess.py)
2. Dual-variant OCR with quality gating (src/ocr.py)
3. Channel detection fallback (src/channel_detector.py)
4. Chrome stripping and text normalization (src/masking.py)
5. Offline feature extraction and TF-IDF transform (src/features.py)
6. Calibrated ML inference (src/train.py)
7. Deterministic rules engine and safety overrides (src/rules.py)
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

import cv2
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix

from src.ocr import run_dual_variant_ocr, check_unreadable_gate
from src.channel_detector import detect_channel
from src.masking import strip_template_chrome, mask_text
from src.features import extract_features, FEATURE_NAMES
from src.rules import evaluate_rules
from src.display_indicators import compute_display_indicators, format_indicator_signals

logger = logging.getLogger(__name__)


class PhishingPipeline:
    """
    End-to-end inference engine for screenshot analysis.
    """

    def __init__(
        self,
        model_path: str = "data/models/calibrated_model.pkl",
        tfidf_path: str = "data/models/tfidf_vectorizer.pkl",
        thresholds_path: str = "data/models/thresholds.json",
        feature_names_path: str = "data/models/feature_names.json",
    ):
        self.model_path = model_path
        self.tfidf_path = tfidf_path
        self.thresholds_path = thresholds_path
        self.feature_names_path = feature_names_path

        # Lazy loaded attributes
        self._model = None
        self._tfidf = None
        self._explainer = None
        self._feature_names: List[str] = []
        self._t_legit = 0.40
        self._t_phish = 0.71

        self._load_artifacts()

    def _load_artifacts(self):
        if os.path.exists(self.model_path):
            self._model = joblib.load(self.model_path)
            logger.info("Loaded calibrated model from %s", self.model_path)
        else:
            logger.warning("Model file not found at %s. Please run src.train first.", self.model_path)

        if os.path.exists(self.tfidf_path):
            self._tfidf = joblib.load(self.tfidf_path)
            logger.info("Loaded TF-IDF vectorizer from %s", self.tfidf_path)

        if os.path.exists(self.thresholds_path):
            with open(self.thresholds_path, "r") as f:
                data = json.load(f)
                self._t_legit = float(data.get("t_legit", 0.40))
                self._t_phish = float(data.get("t_phish", 0.71))
            logger.info("Loaded decision thresholds: T_legit=%.3f, T_phish=%.3f", self._t_legit, self._t_phish)

        # Build canonical feature name list (scalar + tfidf)
        expected_names = list(FEATURE_NAMES)
        if self._tfidf is not None:
            expected_names += [f"tfidf_{n}" for n in self._tfidf.get_feature_names_out()]
        self._feature_names = expected_names

        # Startup assertion: validate feature_names.json if it exists
        if os.path.exists(self.feature_names_path):
            with open(self.feature_names_path, "r") as f:
                stored_names = json.load(f)
            if stored_names != expected_names:
                logger.warning(
                    "feature_names.json (%d names) does not match expected order "
                    "(FEATURE_NAMES + tfidf = %d). Overwriting with correct names.",
                    len(stored_names), len(expected_names),
                )
                with open(self.feature_names_path, "w") as f:
                    json.dump(expected_names, f)
                logger.info("Corrected feature_names.json (%d features)", len(expected_names))

        # Validate model input dimensions
        if self._model is not None:
            n_expected = getattr(self._model, "n_features_in_", None)
            if n_expected is not None and n_expected != len(expected_names):
                logger.error(
                    "FATAL: Model expects %d features but pipeline produces %d. "
                    "Retrain or fix feature alignment.",
                    n_expected, len(expected_names),
                )

        # Initialize SHAP explainer (lazy, may fail gracefully)
        if self._model is not None:
            try:
                import shap
                from sklearn.calibration import CalibratedClassifierCV
                base = self._model
                if isinstance(base, CalibratedClassifierCV):
                    if hasattr(base, 'calibrated_classifiers_') and len(base.calibrated_classifiers_) > 0:
                        cc = base.calibrated_classifiers_[0]
                        base = getattr(cc, 'estimator', getattr(cc, 'base_estimator', base))
                self._explainer = shap.TreeExplainer(base)
                logger.info("SHAP TreeExplainer initialized")
            except Exception as e:
                logger.warning("SHAP explainer unavailable: %s", e)
                self._explainer = None

    def predict_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        channel_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a screenshot and produce a calibrated phishing verdict.

        Args:
            image_input: File path, Path object, or BGR numpy array.
            channel_hint: Optional override for message channel ('email', 'sms', 'whatsapp').

        Returns:
            Dict conforming to the JSON specification:
            - verdict: 'legitimate', 'phishing', or 'uncertain'
            - phishing_score: calibrated float (0.0 to 1.0)
            - channel: detected or provided channel
            - alerts: list of human-readable explanation strings
            - extracted_text: raw OCR output
            - clean_text: chrome-stripped message body
            - ocr_confidence: float percentage
            - rules_triggered: list of rule IDs fired
            - raw_ml_score: uncalibrated or pre-rule ML probability
        """
        # 1. Load image
        if isinstance(image_input, (str, Path)):
            img_path = str(image_input)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found at {img_path}")
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to decode image from {img_path}")
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # 2. Run dual-variant OCR
        ocr_result = run_dual_variant_ocr(img)
        raw_text = ocr_result["text"]
        ocr_conf = float(ocr_result["confidence"])
        winning_variant = ocr_result["winning_variant"]

        # 3. Channel Detection
        if channel_hint and channel_hint.lower() in {"email", "sms", "whatsapp"}:
            channel = channel_hint.lower()
        else:
            ch_info = detect_channel(raw_text)
            detected = ch_info.get("channel", "email")
            channel = detected if detected in {"email", "sms", "whatsapp"} else "email"

        # 4. Chrome Stripping & Masking
        clean_text = strip_template_chrome(raw_text, channel=channel)

        # 5. Check Unreadable Gate
        is_unreadable, unreadable_reason = check_unreadable_gate(
            text=clean_text,
            confidence=ocr_conf,
            min_text_len=15,
            min_confidence=35.0,
        )
        if is_unreadable:
            return {
                "verdict": "uncertain",
                "phishing_score": 0.5,
                "channel": channel,
                "alerts": [f"Image text quality too low to decide reliably: {unreadable_reason}"],
                "extracted_text": raw_text,
                "clean_text": clean_text,
                "ocr_confidence": ocr_conf,
                "winning_variant": winning_variant,
                "rules_triggered": ["UNREADABLE_OCR"],
                "raw_ml_score": 0.5,
            }

        # 6. Feature Extraction (100% offline)
        scalar_dict = extract_features(clean_text, channel, allow_network=False, skip_whois=True)
        scalar_vector = np.array([[scalar_dict[name] for name in FEATURE_NAMES]], dtype=np.float64)

        # Apply the same masking used during training before TF-IDF vectorization
        masked_text = mask_text(clean_text)

        if self._tfidf is not None:
            tfidf_vec = self._tfidf.transform([masked_text])
            X = hstack([csr_matrix(scalar_vector), tfidf_vec]).toarray()
        else:
            X = scalar_vector

        # 7. Model Inference
        if self._model is not None:
            proba = self._model.predict_proba(X)[0]
            classes = list(self._model.classes_)
            phish_idx = classes.index(1) if 1 in classes else -1
            raw_ml_score = float(proba[phish_idx]) if phish_idx >= 0 else 0.5
        else:
            raw_ml_score = 0.5

        # 8. SHAP-based explanations
        suspicion_reasons, safe_reasons = self._explain(X)

        # 9. Rules Engine Evaluation & Safety Overrides
        rule_eval = evaluate_rules(
            raw_ocr_text=raw_text,
            clean_text=clean_text,
            ocr_confidence=ocr_conf,
            ml_score=raw_ml_score,
            channel=channel,
            t_legit=self._t_legit,
            t_phish=self._t_phish,
        )

        # 8. Compute display-only indicators (does NOT feed the model)
        display_indicators = compute_display_indicators(
            clean_text,
            rules_triggered=rule_eval["rules_triggered"],
            alerts=rule_eval["alerts"],
        )
        display_signals = format_indicator_signals(display_indicators)
        combined_suspicion = display_signals + [s for s in suspicion_reasons if s not in display_signals]

        return {
            "verdict": rule_eval["verdict"],
            "phishing_score": round(float(rule_eval["phishing_score"]), 4),
            "channel": channel,
            "alerts": rule_eval["alerts"],
            "extracted_text": raw_text,
            "clean_text": clean_text,
            "ocr_confidence": round(ocr_conf, 2),
            "winning_variant": winning_variant,
            "rules_triggered": rule_eval["rules_triggered"],
            "raw_ml_score": round(raw_ml_score, 4),
            "suspicion_reasons": combined_suspicion,
            "safe_reasons": safe_reasons,
        }

    # ── SHAP Explanation Helper ───────────────────────────────────────

    # Human-readable translations for feature names
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

    def _translate_feature(self, name: str) -> str:
        """Translate a raw feature name into a plain-English sentence."""
        if name in self._FEATURE_LABELS:
            return self._FEATURE_LABELS[name]
        if name.startswith("tfidf_"):
            token = name[6:]
            return f"Text contains the pattern '{token}'"
        return name.replace("_", " ").capitalize()

    def _explain(self, X: np.ndarray, top_n: int = 5):
        """Return (suspicion_reasons, safe_reasons) based on SHAP values."""
        suspicion: List[str] = []
        safe: List[str] = []

        if self._explainer is None or not self._feature_names:
            return suspicion, safe

        try:
            shap_values = self._explainer.shap_values(X)

            # Handle both list output (binary classification) and 3D array
            if isinstance(shap_values, list):
                # shap_values[1] = contributions toward class 1 (phishing)
                sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            elif shap_values.ndim == 3:
                sv = shap_values[0, :, 1]
            else:
                sv = shap_values[0]

            # Pair (feature_name, shap_value)
            pairs = list(zip(self._feature_names, sv))

            # Top features raising suspicion (positive SHAP toward phishing)
            positive = sorted([(n, v) for n, v in pairs if v > 0.001], key=lambda x: -x[1])
            for name, val in positive[:top_n]:
                suspicion.append(self._translate_feature(name))

            # Top features pointing to legitimate (negative SHAP)
            negative = sorted([(n, v) for n, v in pairs if v < -0.001], key=lambda x: x[1])
            for name, val in negative[:top_n]:
                safe.append(self._translate_feature(name))

        except Exception as e:
            logger.warning("SHAP explanation failed: %s", e)

        return suspicion, safe
