"""
Heuristic rules engine for phishing screenshot detection.

Provides deterministic safety overrides:
- UNREADABLE_OCR: Abstains (uncertain) if OCR text is degraded/unreadable.
- EXACT_WHITELIST_BYPASS: Protects genuine messages linking exclusively to verified domains.
- BRAND_IMPERSONATION_MISMATCH: Escalates messages impersonating known brands with unrelated domains.
- HIGH_RISK_TLD_OR_RAW_IP: Flags raw IP addresses and dangerous TLDs targeting credentials.

Non-negotiable principle: A false alarm on a real bank or OTP message is worse than a missed phish.
"""

from typing import Dict, Any, List, Optional, Tuple
import re
from urllib.parse import urlparse

from src.whitelist import (
    is_exact_whitelisted_domain,
    check_brand_domain_mismatch,
    extract_registered_domain,
    PUBLIC_BRAND_DOMAINS,
)
from src.ocr import check_unreadable_gate
from src.feature_extraction.url_utils import extract_urls, normalize_url, get_domain
from src.feature_extraction.keyword_lists import HIGH_RISK_TLDS


def evaluate_rules(
    raw_ocr_text: str,
    clean_text: str,
    ocr_confidence: float,
    ml_score: float,
    channel: str,
    t_legit: float = 0.20,
    t_phish: float = 0.70,
) -> Dict[str, Any]:
    """
    Apply rule hierarchy combining OCR quality gating, heuristic overrides,
    and dual-threshold ML calibration.

    Args:
        raw_ocr_text: Raw OCR extraction.
        clean_text: Template chrome-stripped text.
        ocr_confidence: Mean OCR word confidence percentage (0-100).
        ml_score: Calibrated ML phishing probability (0.0 to 1.0).
        channel: Message channel ('email', 'sms', 'whatsapp').
        t_legit: Low threshold below which verdict is 'legitimate'.
        t_phish: High threshold above which verdict is 'phishing'.

    Returns:
        Dict with:
        - verdict: 'legitimate', 'phishing', or 'uncertain'
        - phishing_score: calibrated float (0.0 to 1.0)
        - alerts: List of plain-English explanation strings
        - rules_triggered: List of rule IDs that fired
    """
    alerts: List[str] = []
    rules_triggered: List[str] = []
    final_score = float(ml_score)

    # ── Rule 1: UNREADABLE_OCR (Quality Gate) ─────────────────────────
    is_unreadable, unreadable_reason = check_unreadable_gate(
        text=clean_text,
        confidence=ocr_confidence,
        min_text_len=15,
        min_confidence=35.0,
    )
    if is_unreadable:
        return {
            "verdict": "uncertain",
            "phishing_score": 0.5,
            "alerts": [f"Image text quality too low to decide reliably: {unreadable_reason}"],
            "rules_triggered": ["UNREADABLE_OCR"],
        }

    # Extract all URLs and domains from clean text
    urls = extract_urls(clean_text)
    domains = [get_domain(u) for u in urls if get_domain(u)]
    registered_domains = [extract_registered_domain(d) for d in domains if extract_registered_domain(d)]

    # ── Rule 2: BRAND_IMPERSONATION_MISMATCH (Phishing Escalation) ────
    # Check if a known brand is claimed in the message body with action/account context,
    # but the destination URL points to an unverified third-party domain
    from src.feature_extraction.url_utils import _EMAIL_PATTERN

    # Stop firing on routine OAuth and third-party app permission notifications
    is_oauth_permission_notice = bool(
        re.search(
            r'\b(?:you shared\b.*?\baccount data with|granted\b.*?\bpermission|requesting updated permissions|third-party app permission|connected app permissions|review or revoke third-party|access requested by)\b',
            clean_text,
            re.IGNORECASE,
        )
    )

    text_no_addresses = _EMAIL_PATTERN.sub(" ", clean_text)
    text_no_addresses = re.sub(r'https?://[^\s<>"\'\)\]]+|www\.[^\s<>"\'\)\]]+', '', text_no_addresses)
    text_no_addresses = re.sub(r'\b(yahoo/?mail|gmail|outlook|icloud)\b', '', text_no_addresses, flags=re.IGNORECASE)

    has_account_action_context = bool(
        re.search(
            r'\b(account|verify|verification|security|login|sign in|password|suspend|locked|alert|update|billing|invoice|payment|otp|pin)\b',
            text_no_addresses,
            re.IGNORECASE,
        )
    )

    mismatch_detected = False
    mismatched_brand = None
    mismatched_domain = None
    mismatch_reason = None

    if not is_oauth_permission_notice:
        for url in urls:
            reg_domain = extract_registered_domain(url)
            if not reg_domain:
                continue
            # If the destination domain itself is whitelisted (e.g. google.com, youtube.com, linkedin.com),
            # it is not an attacker's phishing harvest site
            if is_exact_whitelisted_domain(reg_domain):
                continue

            for brand, legit_domains in PUBLIC_BRAND_DOMAINS.items():
                if len(brand) < 3 and brand not in {"sbi", "pnb", "bof", "dhl", "ups"}:
                    continue
                # Must be an explicit claim of brand identity (e.g. 'PayPal:', 'Your PayPal Account', 'Dear SBI Customer', 'Netflix Support')
                # rather than an incidental social login mention ('sign in with Google', 'available on Google Play')
                brand_claim_pattern = (
                    rf"(?:^{re.escape(brand)}\s*:|\b{re.escape(brand)}\s*:|"
                    rf"\b(?:your\s+{re.escape(brand)}\s+(?:account|card|id|service|statement)|"
                    rf"dear\s+{re.escape(brand)}\s+(?:user|customer|member|client)|"
                    rf"{re.escape(brand)}\s+(?:security|team|support|alerts?|bank|notice|billing|official|admin))\b)"
                )
                is_incidental_mention = bool(
                    re.search(rf"\b(?:via|with|using|through|on)\s+{re.escape(brand)}\b|\b{re.escape(brand)}\s+play\b", text_no_addresses, re.IGNORECASE)
                )
                has_brand_claim = bool(re.search(brand_claim_pattern, text_no_addresses, re.IGNORECASE)) and not is_incidental_mention
                if has_brand_claim:
                    if reg_domain not in legit_domains and has_account_action_context:
                        mismatch_detected = True
                        mismatched_brand = brand
                        mismatched_domain = reg_domain
                        mismatch_reason = f"claims to be from '{brand.title()}' but directs to unauthorized domain '{reg_domain}'"
                        break
            if mismatch_detected:
                break

        # Check sender domain near-miss (e.g. microsoft.co vs microsoft.com)
        if not mismatch_detected:
            from_match = re.search(r'\bfrom:\s*(?:[^<@\n\r]*<)?([a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,}))>?', clean_text, re.IGNORECASE)
            if from_match:
                sender_domain = from_match.group(2).lower()
                sender_reg = extract_registered_domain(sender_domain)
                if sender_reg:
                    sender_sld = sender_reg.split('.')[0]
                    for brand, legit_domains in PUBLIC_BRAND_DOMAINS.items():
                        if len(brand) < 3 and brand not in {"sbi", "pnb", "bof", "dhl", "ups"}:
                            continue
                        brand_in_text = bool(re.search(rf"\b{re.escape(brand)}\b", text_no_addresses, re.IGNORECASE))
                        if brand_in_text and sender_reg not in legit_domains:
                            is_near_miss = (sender_sld == brand) or (
                                abs(len(sender_sld) - len(brand)) <= 1 and
                                sum(c1 != c2 for c1, c2 in zip(sender_sld, brand)) <= 1
                            )
                            if is_near_miss:
                                mismatch_detected = True
                                mismatched_brand = brand
                                mismatched_domain = sender_reg
                                mismatch_reason = f"email sender domain '{sender_domain}' is an unauthorized near-miss of '{brand.title()}'"
                                break

    if mismatch_detected:
        rules_triggered.append("BRAND_IMPERSONATION_MISMATCH")
        alerts.append(f"Brand impersonation detected: Message {mismatch_reason}.")
        final_score = max(final_score, 0.95)
        return {
            "verdict": "phishing",
            "phishing_score": final_score,
            "alerts": alerts,
            "rules_triggered": rules_triggered,
        }

    # ── Rule 3: RAW IP OR HIGH-RISK TLD WITH SENSITIVE REQUEST ────────
    raw_ip_pattern = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
    has_raw_ip = any(raw_ip_pattern.match(d) for d in domains)
    has_high_risk_tld = any(
        any(d.lower().endswith(tld) for tld in HIGH_RISK_TLDS)
        for d in domains
    )
    has_sensitive = bool(re.search(r"\b(password|otp|pin|cvv|bank account|verify your account)\b", clean_text, re.IGNORECASE))

    if (has_raw_ip or has_high_risk_tld) and has_sensitive:
        rules_triggered.append("HIGH_RISK_HOST_WITH_SENSITIVE_REQUEST")
        alerts.append(
            "High-risk destination: Message requests credentials/banking verification using an unverified raw IP or suspicious TLD."
        )
        final_score = max(final_score, 0.92)
        return {
            "verdict": "phishing",
            "phishing_score": final_score,
            "alerts": alerts,
            "rules_triggered": rules_triggered,
        }

    # ── Rule 4: EXACT_WHITELIST_BYPASS (Legitimate Protection) ────────
    # If links are present and ALL registered domains are verified whitelisted
    if registered_domains and all(is_exact_whitelisted_domain(rd) for rd in registered_domains):
        rules_triggered.append("EXACT_WHITELIST_BYPASS")
        alerts.append(
            f"Verified sender: All embedded links point exclusively to verified official domains ({', '.join(set(registered_domains))})."
        )
        # Suppress score below legit threshold to protect genuine bank/OTP/notification
        final_score = min(final_score, 0.05)
        return {
            "verdict": "legitimate",
            "phishing_score": final_score,
            "alerts": alerts,
            "rules_triggered": rules_triggered,
        }

    # ── Rule 5: Dual-Threshold Calibrated ML Decision ──────────────────
    if final_score <= t_legit:
        verdict = "legitimate"
        alerts.append(f"Classified legitimate with low risk score ({final_score:.3f} <= {t_legit:.2f}).")
    elif final_score >= t_phish:
        verdict = "phishing"
        alerts.append(f"Classified phishing with elevated risk score ({final_score:.3f} >= {t_phish:.2f}).")
    else:
        verdict = "uncertain"
        alerts.append(
            f"Uncertain evidence ({final_score:.3f} falls in abstention band [{t_legit:.2f}, {t_phish:.2f}])."
        )

    return {
        "verdict": verdict,
        "phishing_score": final_score,
        "alerts": alerts,
        "rules_triggered": rules_triggered,
    }
