"""
Main orchestration module for phishing feature extraction.

Combines text, URL, and domain features into a single fixed-size feature vector
for machine learning. The order of FEATURE_NAMES defines the final vector layout.
"""

from typing import Dict, Any, Union

from src.feature_extraction import text_utils as tu
from src.feature_extraction import url_utils as uu
from src.feature_extraction import domain_utils as du

# ── Feature Vector Definition ─────────────────────────────────────────
# Order of these names matters! It determines the input shape for the Random Forest.
FEATURE_NAMES = [
    # TEXT FEATURES (13)
    "total_word_count",
    "urgency_count",
    "threat_count",
    "credential_request",
    "personal_info_request",
    "financial_context_count",
    "account_security_count",
    "reward_bait_count",
    "action_word_count",
    "threat_plus_urgency",
    "sensitive_request",
    "generic_greeting_flag",
    "channel",

    # URL FEATURES (15)
    "url_length",
    "raw_ip_address",
    "special_character_count",
    "url_entropy",
    "digit_count_url",
    "url_token_count",
    "suspicious_url_keyword_count",
    "has_at_symbol",
    "is_shortened_url",
    "has_punycode",
    "uses_https",
    "action_plus_url",
    "redirect_count",
    "url_was_unrolled",
    "redirect_domain_change",

    # DOMAIN FEATURES (17)
    "blacklist_status",
    "domain_age",
    "typosquatting_score",
    "number_of_subdomains",
    "whitelist_match",
    "domain_length",
    "digit_count_domain",
    "digit_ratio_domain",
    "hyphen_count_domain",
    "domain_entropy",
    "subdomain_depth",
    "has_dns_record",
    "domain_registration_length",
    "days_until_expiration",
    "brand_in_domain",
    "brand_domain_mismatch",
    "has_high_risk_tld",
]


def _map_channel(channel_str: str) -> int:
    """Map the string channel label to an integer for the model."""
    channel_map = {
        "email": 0,
        "sms": 1,
        "whatsapp": 2,
        "social": 3
    }
    return channel_map.get(channel_str.lower(), -1)


def extract_features(
    text: str,
    channel: str,
    allow_network: bool = False,
    skip_whois: bool = True,
) -> Dict[str, Union[int, float]]:
    """
    Extract a comprehensive set of features from OCR text.

    Args:
        text: The raw text extracted from the screenshot.
        channel: The origin channel (e.g., 'email', 'sms').
        allow_network: If False (default), all external network calls (WHOIS, DNS,
                       Safe Browsing, URL unrolling) are skipped for deterministic offline speed.
        skip_whois: If True (default), skips slow WHOIS lookups.

    Returns:
        Dictionary mapping feature names to numerical values.
    """
    # ── Channel-Specific Preprocessing ──
    cleaned_text = tu.clean_text_for_channel(text, channel)

    # ── Text Features ──
    features = {
        "total_word_count": tu.get_total_word_count(cleaned_text),
        "urgency_count": tu.count_urgency_keywords(cleaned_text),
        "threat_count": tu.count_threat_keywords(cleaned_text),
        "credential_request": tu.count_credential_requests(cleaned_text),
        "personal_info_request": tu.count_personal_info_requests(cleaned_text),
        "financial_context_count": tu.count_financial_context(cleaned_text),
        "account_security_count": tu.count_account_security(cleaned_text),
        "reward_bait_count": tu.count_reward_bait(cleaned_text),
        "action_word_count": tu.count_action_words(cleaned_text),
        "threat_plus_urgency": tu.has_threat_plus_urgency(cleaned_text),
        "sensitive_request": tu.has_sensitive_request(cleaned_text),
        "generic_greeting_flag": tu.has_generic_greeting(cleaned_text),
        "channel": _map_channel(channel),
    }

    # ── Initial URL Extraction ──
    # We grab the first/primary URL found in the text for extraction
    initial_url = uu.get_primary_url(cleaned_text)

    # ── URL Unrolling (Offline-safe) ──
    unroll_data = uu.unroll_url(initial_url, timeout=3, allow_network=allow_network)
    final_url = unroll_data["final_url"]
    
    # ── URL Features ──
    features.update({
        "url_length": uu.get_url_length(final_url),
        "raw_ip_address": uu.has_raw_ip(final_url),
        "special_character_count": uu.count_special_chars(final_url),
        "url_entropy": uu.get_url_entropy(final_url),
        "digit_count_url": uu.count_url_digits(final_url),
        "url_token_count": uu.count_url_tokens(final_url),
        "suspicious_url_keyword_count": uu.count_suspicious_url_keywords(final_url),
        "has_at_symbol": uu.has_at_symbol(final_url),
        "is_shortened_url": uu.is_url_shortener(initial_url),  # check original URL for this
        "has_punycode": uu.has_punycode(final_url),
        "uses_https": uu.has_https(final_url),
        "action_plus_url": uu.has_action_plus_url(cleaned_text, final_url),
        "redirect_count": unroll_data["redirect_count"],
        "url_was_unrolled": unroll_data["url_was_unrolled"],
        "redirect_domain_change": unroll_data["domain_changed"],
    })

    # ── Domain Extraction ──
    domain = uu.get_domain(final_url)

    # ── Domain Features ──
    features.update({
        "blacklist_status": du.check_blacklist(domain, allow_network=allow_network),
        "typosquatting_score": du.get_typosquatting_score(domain),
        "number_of_subdomains": uu.count_subdomains(final_url),
        "whitelist_match": du.check_whitelist(domain),
        "domain_length": du.get_domain_length(domain),
        "digit_count_domain": du.count_domain_digits(domain),
        "digit_ratio_domain": du.get_domain_digit_ratio(domain),
        "hyphen_count_domain": du.count_domain_hyphens(domain),
        "domain_entropy": du.get_domain_entropy(domain),
        "subdomain_depth": du.get_subdomain_depth(domain),
        "has_dns_record": du.has_dns_record(domain, allow_network=allow_network),
        "brand_in_domain": du.has_brand_in_domain(cleaned_text, domain),
        "brand_domain_mismatch": du.has_brand_domain_mismatch(cleaned_text, domain),
        "has_high_risk_tld": du.has_high_risk_tld(domain),
    })
    
    # Slow features (WHOIS)
    if skip_whois or not allow_network:
        features.update({
            "domain_age": -1,
            "domain_registration_length": -1,
            "days_until_expiration": -1,
        })
    else:
        features.update({
            "domain_age": du.get_domain_age_days(domain, allow_network=allow_network),
            "domain_registration_length": du.get_domain_registration_length(domain, allow_network=allow_network),
            "days_until_expiration": du.get_days_until_expiration(domain, allow_network=allow_network),
        })

    # Validate output vector completeness
    for name in FEATURE_NAMES:
        if name not in features:
            raise ValueError(f"Feature vector mismatch: missing {name}")

    # Ensure correct order
    ordered_features = {name: features[name] for name in FEATURE_NAMES}
    return ordered_features
