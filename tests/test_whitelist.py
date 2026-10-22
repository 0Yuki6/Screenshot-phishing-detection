"""
Unit tests for public brand domain whitelist and spoof protection.
"""

import pytest
from src.whitelist import (
    extract_registered_domain,
    is_exact_whitelisted_domain,
    check_brand_domain_mismatch,
    HIGH_RISK_TLDS,
    KNOWN_SHORTENERS,
)


def test_extract_registered_domain():
    assert extract_registered_domain("https://login.paypal.com/signin") == "paypal.com"
    assert extract_registered_domain("https://sub.sub2.google.co.uk/path") == "google.co.uk"
    assert extract_registered_domain("http://paypal.com.evil.xyz/login") == "evil.xyz"
    assert extract_registered_domain("https://amazon-support.top") == "amazon-support.top"


def test_is_exact_whitelisted_domain():
    # Legitimate subdomains of whitelisted brands
    assert is_exact_whitelisted_domain("https://paypal.com") is True
    assert is_exact_whitelisted_domain("https://login.paypal.com") is True
    assert is_exact_whitelisted_domain("https://mail.google.com/mail") is True
    assert is_exact_whitelisted_domain("https://support.microsoft.com") is True

    # Malicious subdomain and look-alike spoofing
    assert is_exact_whitelisted_domain("https://paypal.com.evil.xyz") is False
    assert is_exact_whitelisted_domain("https://paypal-update.com") is False
    assert is_exact_whitelisted_domain("https://chase.com.security-verify.net") is False
    assert is_exact_whitelisted_domain("https://google.com.phish.org") is False


def test_check_brand_domain_mismatch():
    # Brand mentioned + malicious mismatch
    text_phish = "Urgent: Your PayPal account has been suspended. Verify at http://account-update.xyz"
    assert check_brand_domain_mismatch(text_phish, "http://account-update.xyz") is True

    # Brand mentioned + legit matching domain (MUST NOT fire mismatch)
    text_legit = "Your PayPal receipt is ready. View at https://www.paypal.com/activity"
    assert check_brand_domain_mismatch(text_legit, "https://www.paypal.com/activity") is False

    # Subdomain spoof attempt: brand mentioned, registered domain is evil.xyz
    text_spoof = "Amazon notice: verify your order at https://amazon.com.order-tracking.xyz"
    assert check_brand_domain_mismatch(text_spoof, "https://amazon.com.order-tracking.xyz") is True

    # No brand mentioned
    text_nobrand = "Hey, are you free for lunch tomorrow? https://my-blog.com"
    assert check_brand_domain_mismatch(text_nobrand, "https://my-blog.com") is False
