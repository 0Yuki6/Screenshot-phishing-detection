"""
URL parsing and analysis utilities for phishing feature extraction.

Extracts URLs from text and computes URL-based features such as length,
subdomain count, special character count, IP presence, HTTPS usage,
and URL shortener detection. Includes URL unrolling for shortened URLs.
"""

import re
import math
import requests
from urllib.parse import urlparse
from typing import List, Optional, Tuple, Dict, Any

from src.feature_extraction.keyword_lists import URL_SHORTENERS, SUSPICIOUS_URL_KEYWORDS, ACTION_URL_TRIGGERS, DOMAIN_ALIASES


# ── Regex for extracting URLs from raw text ──────────────────────────
# Matches http(s)://, www., and bare domain patterns
_URL_PATTERN = re.compile(
    r'https?://[^\s<>\"\')\]]+|'       # http:// or https://
    r'www\.[^\s<>\"\')\]]+|'           # www.
    r'\b[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:/[^\s<>\"\')\]]*)?',  # bare domain
    re.IGNORECASE,
)

# Regex for detecting email addresses (used to filter false-positive URLs)
_EMAIL_PATTERN = re.compile(
    r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}'
)

# Regex for detecting raw IP addresses in URLs
_IP_PATTERN = re.compile(
    r'(?:https?://)?(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
)


def extract_urls(text: str) -> List[str]:
    """
    Extract all URLs from the given text.

    Filters out bare-domain matches that are substrings of email addresses
    (e.g. won't return "gmail.com" from "user@gmail.com").

    Args:
        text: Raw OCR text to scan for URLs.

    Returns:
        List of URL strings found in the text.
    """
    # 1. Strip full email addresses so local parts (e.g. sarah.mitchell@...) are never mistaken for URLs
    text_clean = _EMAIL_PATTERN.sub(" ", text)

    # 2. Fix URLs broken across lines (e.g., https://\ndomain.com)
    text_clean = re.sub(r'(https?://|www\.)\s+', r'\1', text_clean, flags=re.IGNORECASE)

    candidates = _URL_PATTERN.findall(text_clean)
    filtered = []
    for url in candidates:
        # Keep anything with an explicit scheme (http/https) or www.
        if url.lower().startswith(("http://", "https://", "www.")):
            filtered.append(url)
        else:
            # Bare domain — strictly verify it has a valid internet TLD suffix
            from src.whitelist import extract_registered_domain
            bare = url.split("/")[0]
            if extract_registered_domain(bare):
                filtered.append(url)
    return filtered


def get_primary_url(text: str) -> Optional[str]:
    """
    Return the first URL found in the text, or None.

    The "primary" URL is used as the representative URL for feature
    extraction when a message contains multiple links.
    """
    urls = extract_urls(text)
    return urls[0] if urls else None


def normalize_url(url: str) -> str:
    """
    Ensure a URL has a scheme so urlparse works correctly.
    """
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def get_domain(url: Optional[str]) -> Optional[str]:
    """
    Extract the domain (hostname) from a URL.

    Returns:
        The hostname string, or None if parsing fails.
    """
    if not url:
        return None
    parsed = urlparse(normalize_url(url))
    return parsed.hostname


def get_url_length(url: Optional[str]) -> int:
    """
    Return the total character length of the URL.
    Returns 0 if no URL provided.
    """
    if not url:
        return 0
    return len(url)


def count_subdomains(url: Optional[str]) -> int:
    """
    Count the number of subdomain segments in a URL's hostname.
    """
    if not url:
        return 0
    parsed = urlparse(normalize_url(url))
    hostname = parsed.hostname
    if not hostname:
        return 0
    parts = hostname.split(".")
    # domain + TLD = 2 parts; everything before that is a subdomain
    return max(0, len(parts) - 2)


def count_special_chars(url: Optional[str]) -> int:
    """
    Count non-alphanumeric characters in the URL (excluding scheme delimiters).
    """
    if not url:
        return 0
    cleaned = re.sub(r'^https?://', '', url)
    return sum(1 for c in cleaned if not c.isalnum())


def has_raw_ip(url: Optional[str]) -> int:
    """
    Check if the URL contains a raw IP address instead of a domain name.
    """
    if not url:
        return 0
    return 1 if _IP_PATTERN.search(url) else 0


def has_https(url: Optional[str]) -> int:
    """
    Check if the URL uses HTTPS scheme.
    """
    if not url:
        return 0
    return 1 if url.lower().startswith("https://") else 0


def is_url_shortener(url: Optional[str]) -> int:
    """
    Check if the URL's domain matches a known URL shortener service.
    """
    if not url:
        return 0
    hostname = get_domain(url)
    if not hostname:
        return 0
    hostname_lower = hostname.lower()
    for shortener in URL_SHORTENERS:
        if hostname_lower == shortener or hostname_lower.endswith("." + shortener):
            return 1
    return 0


# ── New URL features ────────────────────────────────────────────────

def get_url_entropy(url: Optional[str]) -> float:
    """
    Calculate Shannon entropy of the URL.
    Purpose: identify unusually random/obfuscated URLs.
    """
    if not url:
        return 0.0
    cleaned = re.sub(r'^https?://', '', url)
    if not cleaned:
        return 0.0
    
    prob = [float(cleaned.count(c)) / len(cleaned) for c in dict.fromkeys(list(cleaned))]
    entropy = -sum(p * math.log(p) / math.log(2.0) for p in prob)
    return entropy


def count_url_digits(url: Optional[str]) -> int:
    """Number of digits in the complete URL."""
    if not url:
        return 0
    return sum(1 for c in url if c.isdigit())


def count_url_tokens(url: Optional[str]) -> int:
    """
    Count URL components/tokens separated by: /, ., -, _, ?, =, &
    """
    if not url:
        return 0
    tokens = re.split(r'[/.\-_?=&]', url)
    return len([t for t in tokens if t])


def count_suspicious_url_keywords(url: Optional[str]) -> int:
    """
    Detect security/credential/action-related terms inside the URL itself.
    """
    if not url:
        return 0
    url_lower = url.lower()
    return sum(1 for kw in SUSPICIOUS_URL_KEYWORDS if kw in url_lower)


def has_at_symbol(url: Optional[str]) -> int:
    """Binary feature indicating whether @ appears in the URL."""
    if not url:
        return 0
    return 1 if "@" in url else 0


def has_punycode(url: Optional[str]) -> int:
    """
    Detect domains containing xn--
    This can indicate possible IDN/punycode usage.
    """
    if not url:
        return 0
    hostname = get_domain(url)
    if not hostname:
        return 0
    return 1 if "xn--" in hostname.lower() else 0


def has_action_plus_url(text: str, url: Optional[str]) -> int:
    """
    Binary feature indicating whether an action/request term occurs
    together with a URL in the text.
    """
    if not url:
        return 0
    text_lower = text.lower()
    for trigger in ACTION_URL_TRIGGERS:
        if trigger in text_lower:
            return 1
    return 0


def is_same_domain_or_alias(orig_domain: Optional[str], final_domain: Optional[str]) -> bool:
    """
    Check if two domains belong to the same entity, taking into account
    subdomains, www prefixes, and official platform domain aliases (e.g., youtu.be -> youtube.com).
    """
    if not orig_domain or not final_domain:
        return False
    d1 = orig_domain.lower().strip()
    d2 = final_domain.lower().strip()
    if d1 == d2 or d1.endswith("." + d2) or d2.endswith("." + d1):
        return True

    d1_clean = d1[4:] if d1.startswith("www.") else d1
    d2_clean = d2[4:] if d2.startswith("www.") else d2
    if d1_clean == d2_clean or d1_clean.endswith("." + d2_clean) or d2_clean.endswith("." + d1_clean):
        return True

    alias_1 = DOMAIN_ALIASES.get(d1_clean, d1_clean)
    alias_2 = DOMAIN_ALIASES.get(d2_clean, d2_clean)
    if alias_1 == alias_2 or d2_clean.endswith("." + alias_1) or d1_clean.endswith("." + alias_2):
        return True

    return False


# ── URL Unrolling ──────────────────────────────────────────────────

def unroll_url(url: Optional[str], timeout: int = 3, allow_network: bool = False) -> Dict[str, Any]:
    """
    Attempt to unroll a shortened or redirecting URL.
    Uses a strict timeout and handles failures safely by returning
    the original URL and fallback metrics.
    If allow_network=False (default), skips network requests entirely.
    
    Returns:
        Dict with:
        - 'final_url': The ultimate resolved URL (or original if failed)
        - 'redirect_count': Number of redirects
        - 'domain_changed': 1 if final domain != original domain, else 0
        - 'url_was_unrolled': 1 if expansion succeeded with redirects, else 0
    """
    fallback = {
        "final_url": url,
        "redirect_count": 0,
        "domain_changed": 0,
        "url_was_unrolled": 0,
    }
    
    if not url or not allow_network:
        return fallback

    original_domain = get_domain(url)
    if not original_domain:
        return fallback

    normalized = normalize_url(url)
    try:
        # head request allows us to follow redirects without downloading the body
        response = requests.head(normalized, allow_redirects=True, timeout=timeout)
        final_url = response.url
        redirect_count = len(response.history)
        
        # If no redirects, we didn't unroll anything interesting
        if redirect_count == 0:
            return fallback
            
        final_domain = get_domain(final_url)
        domain_changed = 0 if is_same_domain_or_alias(original_domain, final_domain) else 1
        
        return {
            "final_url": final_url,
            "redirect_count": redirect_count,
            "domain_changed": domain_changed,
            "url_was_unrolled": 1,
        }
    except Exception:
        # If timeout, dead link, connection fail, just fallback silently
        return fallback
