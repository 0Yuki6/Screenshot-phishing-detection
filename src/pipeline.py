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
from typing import Dict, Any, Optional, Union
from pathlib import Path

import cv2
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix

from src.ocr import run_dual_variant_ocr, check_unreadable_gate
from src.channel_detector import detect_channel
from src.masking import strip_template_chrome
from src.features import extract_features, FEATURE_NAMES
from src.rules import evaluate_rules

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
    ):
        self.model_path = model_path
        self.tfidf_path = tfidf_path
        self.thresholds_path = thresholds_path

        # Lazy loaded attributes
        self._model = None
        self._tfidf = None
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

        if self._tfidf is not None:
            tfidf_vec = self._tfidf.transform([clean_text])
            X = hstack([csr_matrix(scalar_vector), tfidf_vec]).toarray()
        else:
            X = scalar_vector

        # 7. Model Inference
        if self._model is not None:
            raw_ml_score = float(self._model.predict_proba(X)[:, 1][0])
        else:
            raw_ml_score = 0.5

        # 8. Rules Engine Evaluation & Safety Overrides
        rule_eval = evaluate_rules(
            raw_ocr_text=raw_text,
            clean_text=clean_text,
            ocr_confidence=ocr_conf,
            ml_score=raw_ml_score,
            channel=channel,
            t_legit=self._t_legit,
            t_phish=self._t_phish,
        )

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
        }
