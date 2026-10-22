"""
Unit tests for the rules engine (src/rules.py).
Tests unreadable gating, whitelist bypass, brand mismatch detection,
and dual-threshold abstention logic.
"""

import pytest
from src.rules import evaluate_rules


def test_unreadable_gate_triggers_uncertain():
    res = evaluate_rules(
        raw_ocr_text="short",
        clean_text="short",
        ocr_confidence=80.0,
        ml_score=0.9,
        channel="sms",
    )
    assert res["verdict"] == "uncertain"
    assert "UNREADABLE_OCR" in res["rules_triggered"]

    # Low confidence
    res2 = evaluate_rules(
        raw_ocr_text="This is a reasonable length text for testing purposes.",
        clean_text="This is a reasonable length text for testing purposes.",
        ocr_confidence=25.0,
        ml_score=0.9,
        channel="sms",
    )
    assert res2["verdict"] == "uncertain"
    assert "UNREADABLE_OCR" in res2["rules_triggered"]


def test_exact_whitelist_bypass_protects_legit():
    clean_text = "Your HDFC Bank statement is ready. Visit https://netbanking.hdfcbank.com/entry for details."
    res = evaluate_rules(
        raw_ocr_text=clean_text,
        clean_text=clean_text,
        ocr_confidence=88.0,
        ml_score=0.65,  # Even if ML model was uncertain/high
        channel="email",
    )
    assert res["verdict"] == "legitimate"
    assert "EXACT_WHITELIST_BYPASS" in res["rules_triggered"]
    assert res["phishing_score"] <= 0.05


def test_brand_domain_mismatch_escalates_phish():
    clean_text = "PayPal: Your account is locked. Verify now at http://paypal-resolution-center.com/login"
    res = evaluate_rules(
        raw_ocr_text=clean_text,
        clean_text=clean_text,
        ocr_confidence=85.0,
        ml_score=0.20,  # Even if ML model score was low
        channel="sms",
    )
    assert res["verdict"] == "phishing"
    assert "BRAND_IMPERSONATION_MISMATCH" in res["rules_triggered"]
    assert res["phishing_score"] >= 0.95


def test_high_risk_host_with_sensitive_request():
    clean_text = "Urgent: Enter your OTP to confirm transaction at http://192.168.1.100/verify"
    res = evaluate_rules(
        raw_ocr_text=clean_text,
        clean_text=clean_text,
        ocr_confidence=90.0,
        ml_score=0.40,
        channel="sms",
    )
    assert res["verdict"] == "phishing"
    assert "HIGH_RISK_HOST_WITH_SENSITIVE_REQUEST" in res["rules_triggered"]


def test_dual_threshold_abstention():
    clean_text = "Hello there, just checking in to see if you received the document earlier."
    res = evaluate_rules(
        raw_ocr_text=clean_text,
        clean_text=clean_text,
        ocr_confidence=80.0,
        ml_score=0.45,  # Between 0.20 and 0.70
        channel="whatsapp",
        t_legit=0.20,
        t_phish=0.70,
    )
    assert res["verdict"] == "uncertain"
    assert "Uncertain evidence" in res["alerts"][0]

    # Clearly legitimate
    res_legit = evaluate_rules(
        raw_ocr_text=clean_text,
        clean_text=clean_text,
        ocr_confidence=80.0,
        ml_score=0.10,
        channel="whatsapp",
        t_legit=0.20,
        t_phish=0.70,
    )
    assert res_legit["verdict"] == "legitimate"

    # Clearly phishing
    res_phish = evaluate_rules(
        raw_ocr_text=clean_text,
        clean_text=clean_text,
        ocr_confidence=80.0,
        ml_score=0.85,
        channel="whatsapp",
        t_legit=0.20,
        t_phish=0.70,
    )
    assert res_phish["verdict"] == "phishing"
