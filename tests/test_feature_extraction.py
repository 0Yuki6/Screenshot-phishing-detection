"""
Tests for the feature extraction pipeline.

Covers URL utilities, text utilities, domain utilities,
and the full feature extractor.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extraction.url_utils import (
    extract_urls,
    get_primary_url,
    get_url_length,
    count_subdomains,
    count_special_chars,
    has_raw_ip,
    has_https,
    is_url_shortener,
    get_domain,
    get_url_entropy,
    count_url_digits,
    count_url_tokens,
    count_suspicious_url_keywords,
    has_at_symbol,
    has_punycode,
    has_action_plus_url,
    unroll_url
)
from src.feature_extraction.text_utils import (
    get_total_word_count,
    count_urgency_keywords,
    count_threat_keywords,
    count_credential_requests,
    count_personal_info_requests,
    count_financial_context,
    count_account_security,
    count_reward_bait,
    count_action_words,
    has_threat_plus_urgency,
    has_sensitive_request,
    has_generic_greeting,
)
from src.feature_extraction.domain_utils import (
    get_typosquatting_score,
    check_whitelist,
    check_blacklist,
    get_domain_length,
    count_domain_digits,
    get_domain_digit_ratio,
    count_domain_hyphens,
    get_domain_entropy,
    get_subdomain_depth,
    has_brand_in_domain,
    has_brand_domain_mismatch
)
from src.feature_extraction import extract_features, FEATURE_NAMES


# ── URL Utils Tests ──────────────────────────────────────────────────

class TestURLExtraction:
    def test_extract_http_url(self):
        urls = extract_urls("Click here: http://example.com/path")
        assert len(urls) >= 1
        assert any("example.com" in u for u in urls)

    def test_extract_https_url(self):
        urls = extract_urls("Visit https://secure.example.com/login")
        assert len(urls) >= 1

    def test_no_urls(self):
        primary = get_primary_url("This text has no URLs at all.")
        assert primary is None or "." not in primary.split("/")[0]

    def test_url_length(self):
        assert get_url_length("http://example.com/very/long/path") > 20
        assert get_url_length(None) == 0

    def test_subdomain_count(self):
        assert count_subdomains("http://mail.google.com") == 1
        assert count_subdomains("http://example.com") == 0

    def test_special_chars(self):
        assert count_special_chars("http://example.com/path?q=1&r=2") > 0

    def test_raw_ip(self):
        assert has_raw_ip("http://192.168.1.1/login") == 1
        assert has_raw_ip("http://example.com") == 0

    def test_https(self):
        assert has_https("https://example.com") == 1

    def test_url_shortener(self):
        assert is_url_shortener("http://bit.ly/abc123") == 1
        assert is_url_shortener("http://example.com") == 0

    def test_get_domain(self):
        assert get_domain("http://mail.google.com/inbox") == "mail.google.com"

    def test_url_entropy(self):
        assert get_url_entropy("http://example.com") > 0

    def test_count_url_digits(self):
        assert count_url_digits("http://example123.com/page456") == 6

    def test_count_url_tokens(self):
        assert count_url_tokens("http://example.com/a/b?c=d") >= 4

    def test_count_suspicious_url_keywords(self):
        assert count_suspicious_url_keywords("http://secure-login.com") >= 2

    def test_has_at_symbol(self):
        assert has_at_symbol("http://user@example.com") == 1

    def test_has_punycode(self):
        assert has_punycode("http://xn--example-h2b.com") == 1

    def test_action_plus_url(self):
        assert has_action_plus_url("Please verify your account", "http://example.com") == 1


# ── Text Utils Tests ─────────────────────────────────────────────────

class TestTextUtils:
    def test_word_count(self):
        assert get_total_word_count("hello world foo bar") == 4

    def test_urgency_keywords(self):
        text = "URGENT: Verify immediately."
        assert count_urgency_keywords(text) >= 2

    def test_threat_keywords(self):
        text = "Your account has been suspended!"
        assert count_threat_keywords(text) >= 1

    def test_credential_requests(self):
        text = "Please enter your password and OTP."
        assert count_credential_requests(text) >= 2

    def test_personal_info_requests(self):
        text = "Please provide your PAN and Aadhaar."
        assert count_personal_info_requests(text) >= 2

    def test_financial_context(self):
        text = "Your bank transaction is complete."
        assert count_financial_context(text) >= 2

    def test_threat_plus_urgency(self):
        assert has_threat_plus_urgency("Urgent: your account is suspended") == 1
        assert has_threat_plus_urgency("Your account is suspended") == 0

    def test_sensitive_request(self):
        assert has_sensitive_request("Enter your password") == 1
        assert has_sensitive_request("Enter your PAN") == 1
        assert has_sensitive_request("Hello world") == 0

    def test_generic_greeting(self):
        assert has_generic_greeting("Dear Customer, your account needs attention.") == 1


# ── Domain Utils Tests ───────────────────────────────────────────────

class TestDomainUtils:
    def test_typosquatting(self):
        assert get_typosquatting_score("google.com") == 0.0
        assert get_typosquatting_score("g00gle.com") > 0.5

    def test_whitelist(self):
        assert check_whitelist("google.com") == 1
        assert check_whitelist("unknown.xyz") == 0
        
    def test_domain_length(self):
        assert get_domain_length("example.com") == 11
        
    def test_domain_digits(self):
        assert count_domain_digits("example123.com") == 3
        
    def test_domain_digit_ratio(self):
        assert get_domain_digit_ratio("ex1.com") == 1 / 7
        
    def test_domain_hyphens(self):
        assert count_domain_hyphens("my-test-domain.com") == 2
        
    def test_subdomain_depth(self):
        assert get_subdomain_depth("a.b.c.example.com") == 3
        
    def test_brand_in_domain(self):
        assert has_brand_in_domain("Welcome to Amazon", "amazon-support.com") == 1
        
    def test_brand_domain_mismatch(self):
        # Text mentions Amazon, but domain is totally different
        assert has_brand_domain_mismatch("Welcome to Amazon", "xyz.com") == 1


# ── Full Feature Extractor Tests ─────────────────────────────────────

class TestFeatureExtractor:
    def test_phishing_email_features(self):
        text = (
            "Subject: Urgent Account Alert\n"
            "From: security@paypa1-verify.com\n"
            "To: user@gmail.com\n\n"
            "Dear Customer,\n\n"
            "Your account has been suspended due to unauthorized activity. "
            "Verify your identity immediately at http://bit.ly/verify-now\n\n"
            "Enter your password and OTP to restore access. "
            "Failure to respond within 24 hours will result in account deactivation."
        )
        features = extract_features(text, "email", skip_whois=True)

        assert isinstance(features, dict)
        assert features["channel"] == 0  # email = 0
        assert features["is_shortened_url"] == 1
        assert features["urgency_count"] >= 3
        assert features["threat_count"] >= 1
        assert features["credential_request"] >= 2
        assert features["generic_greeting_flag"] == 1
        assert features["threat_plus_urgency"] == 1
        assert features["sensitive_request"] == 1

    def test_legitimate_email_features(self):
        text = (
            "Subject: Your order has shipped!\n"
            "From: orders@amazon.com\n"
            "To: customer@gmail.com\n\n"
            "Hi John,\n\n"
            "Great news! Your order #12345 has shipped.\n"
            "Track your package: https://www.amazon.com/track/12345"
        )
        features = extract_features(text, "email", skip_whois=True)

        assert features["channel"] == 0
        assert features["uses_https"] == 1
        assert features["generic_greeting_flag"] == 0
        assert features["whitelist_match"] == 1  # amazon.com is whitelisted

    def test_all_feature_names_present(self):
        features = extract_features("Hello world http://example.com", "unknown", skip_whois=True)
        for name in FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"
        assert len(features) == len(FEATURE_NAMES)

    def test_empty_text(self):
        features = extract_features("", "unknown", skip_whois=True)
        assert features["total_word_count"] == 0
        assert features["url_length"] == 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
