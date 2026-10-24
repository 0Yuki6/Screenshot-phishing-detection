"""
Display-only indicators computation module.

Computes six display-only boolean indicators directly from OCR text using
whole-word and exact-phrase matching:
1. Urgent Language
2. Financial Keywords
3. Credential Request
4. Personal Info Request
5. Suspicious URL
6. Suspicious Vocabulary

These indicators are strictly for UI explainability and display.
They do NOT feed or modify the machine learning feature vector or classifier.
"""

import re
from typing import Dict, List, Optional, Any

from src.feature_extraction.url_utils import extract_urls, get_domain
from src.whitelist import is_exact_whitelisted_domain, extract_registered_domain, HIGH_RISK_TLDS


# ── Whole-word regex helpers ──────────────────────────────────────────

def _matches_any_phrase(text: str, phrases: List[str]) -> bool:
    """Match any phrase or whole word, avoiding substring false positives."""
    text_lower = text.lower()
    for phrase in phrases:
        pattern = r"\b" + re.escape(phrase.lower()) + r"\b"
        if re.search(pattern, text_lower):
            return True
    return False


# ── Indicator matching definitions ───────────────────────────────────

URGENT_PHRASES = [
    "immediately",
    "within 24 hours",
    "within 48 hours",
    "within 2 hours",
    "suspended",
    "suspension",
    "expires",
    "expiration",
    "act now",
    "avoid permanent",
    "final notice",
    "urgent",
    "account closed",
    "account terminated",
    "action required",
]

FINANCIAL_PHRASES = [
    "billing",
    "payment",
    "refund",
    "card",
    "bank",
    "banking",
    "account on hold",
    "prize",
    "lottery",
    "wire transfer",
    "transaction",
    "invoice",
]

PERSONAL_INFO_PATTERNS = [
    r"\b(?:review|confirm|update|verify|submit|provide)\s+(?:your\s+)?(?:information|details|identity|kyc|profile|address)\b",
    r"\bkyc\s+(?:verification|update|expired|pending|documents?)\b",
    r"\b(?:aadhaar|pan\s+card|passport\s+number|driving\s+licen[sc]e|national\s+id)\b",
]

CREDENTIAL_REQUEST_PATTERNS = [
    r"\b(?:enter|provide|confirm|submit|type|input|share|send|insert)\s+(?:your\s+)?(?:password|pin|otp|cvv|credentials|passcode|secret\s+code)\b",
    r"\breply\s+(?:with|by\s+sending)\s+(?:your\s+)?(?:password|pin|otp|cvv|credentials|passcode|secret\s+code|code)\b",
    r"\breply\s+stop\s+followed\s+by\s+(?:your\s+)?(?:password|pin|otp|credentials)\b",
    r"\bverify\s+here\b",
    r"\b(?:login|sign\s+in)\s+at\s+https?://\S+",
]

SUSPICIOUS_VOCAB_PHRASES = [
    "unrecognized activity",
    "unauthorized access",
    "compromised",
    "temporary hold",
    "security alert",
    "unrecognized device",
    "unauthorized transaction",
]


def compute_display_indicators(
    clean_text: str,
    rules_triggered: Optional[List[str]] = None,
    alerts: Optional[List[str]] = None,
    urls: Optional[List[str]] = None,
) -> Dict[str, bool]:
    """
    Compute six display-only indicators from OCR text and existing rule signals.

    Returns:
        dict: {
            "urgent_language": bool,
            "financial_keywords": bool,
            "credential_request": bool,
            "personal_info_request": bool,
            "suspicious_url": bool,
            "suspicious_vocabulary": bool,
        }
    """
    if not clean_text:
        return {
            "urgent_language": False,
            "financial_keywords": False,
            "credential_request": False,
            "personal_info_request": False,
            "suspicious_url": False,
            "suspicious_vocabulary": False,
        }

    text_lower = clean_text.lower()
    rules_set = set(rules_triggered or [])
    alerts_text = " ".join(alerts or []).lower()

    # 1. Urgent Language (whole-word/phrase match)
    urgent_hit = _matches_any_phrase(text_lower, URGENT_PHRASES)

    # 2. Financial Keywords (whole-word/phrase match)
    financial_hit = _matches_any_phrase(text_lower, FINANCIAL_PHRASES)

    # 3. Credential Request (imperative/solicitation request wording only)
    credential_hit = False
    for pat in CREDENTIAL_REQUEST_PATTERNS:
        # Negative context safeguard ("do not share", "never share")
        safe_pat = r"(?<!do not\s)(?<!don't\s)(?<!never\s)(?<!should not\s)" + pat
        if re.search(safe_pat, text_lower):
            credential_hit = True
            break

    # 4. Personal Info Request
    personal_hit = False
    for pat in PERSONAL_INFO_PATTERNS:
        if re.search(pat, text_lower):
            personal_hit = True
            break

    # 5. Suspicious URL
    url_list = urls if urls is not None else extract_urls(clean_text)
    suspicious_url_hit = False
    if "BRAND_IMPERSONATION_MISMATCH" in rules_set or "HIGH_RISK_HOST_WITH_SENSITIVE_REQUEST" in rules_set:
        suspicious_url_hit = True
    elif "unauthorized domain" in alerts_text or "near-miss" in alerts_text:
        suspicious_url_hit = True
    else:
        for u in url_list:
            reg_d = extract_registered_domain(u)
            if not reg_d:
                continue
            if is_exact_whitelisted_domain(reg_d):
                continue
            # Check for high-risk TLDs, raw IPs, or keywords in domain
            d_lower = reg_d.lower()
            if any(d_lower.endswith(tld) for tld in HIGH_RISK_TLDS):
                suspicious_url_hit = True
                break
            if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", d_lower):
                suspicious_url_hit = True
                break
            if any(k in d_lower for k in ["verify", "secure", "update", "login", "account", "billing"]):
                suspicious_url_hit = True
                break

    # 6. Suspicious Vocabulary
    vocab_hit = _matches_any_phrase(text_lower, SUSPICIOUS_VOCAB_PHRASES)
    if "suspicious vocabulary" in alerts_text:
        vocab_hit = True

    return {
        "urgent_language": urgent_hit,
        "financial_keywords": financial_hit,
        "credential_request": credential_hit,
        "personal_info_request": personal_hit,
        "suspicious_url": suspicious_url_hit,
        "suspicious_vocabulary": vocab_hit,
    }


def format_indicator_signals(indicators: Dict[str, bool]) -> List[str]:
    """
    Format boolean indicators into descriptive signal strings for UI explainability.
    These populate suspicion_reasons / flagged_features without altering response shape.
    """
    signals = []
    if indicators.get("urgent_language"):
        signals.append("Urgent language: Pressure keywords or threatening deadlines detected in message")
    if indicators.get("financial_keywords"):
        signals.append("Financial keywords: Billing, payment, refund, or banking context identified")
    if indicators.get("credential_request"):
        signals.append("Credential request: Explicit request for password, PIN, OTP, or login credentials")
    if indicators.get("personal_info_request"):
        signals.append("Personal info request: Solicitations to verify, confirm, or update identity/KYC details")
    if indicators.get("suspicious_url"):
        signals.append("Suspicious URL: High-risk URL heuristics, deceptive link, or suspicious domain")
    if indicators.get("suspicious_vocabulary"):
        signals.append("Suspicious vocabulary: Patterns and keyword clusters matching documented phishing campaigns")
    return signals
