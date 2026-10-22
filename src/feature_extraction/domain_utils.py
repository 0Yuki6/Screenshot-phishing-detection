"""
Domain analysis utilities for phishing feature extraction.

Provides WHOIS-based domain age lookup, typosquatting score computation
(Levenshtein distance against known brands), whitelist/blacklist checks,
and various domain structure/entropy features.

Note: WHOIS lookups are slow (2-5s) and may be rate-limited. All external
calls are wrapped in try/except with sensible fallbacks for robustness.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Optional
import dns.resolver

import Levenshtein

from src.feature_extraction.keyword_lists import KNOWN_BRANDS, LEGITIMATE_DOMAINS, BRAND_NAME_MAP, HIGH_RISK_TLDS, DOMAIN_ALIASES
from src.whitelist import is_exact_whitelisted_domain, ALL_WHITELISTED_DOMAINS, PUBLIC_BRAND_DOMAINS

logger = logging.getLogger(__name__)


# ── Existing Domain features ───────────────────────────────────────

def has_high_risk_tld(domain: Optional[str]) -> int:
    """
    Check if the domain ends in a high-risk TLD commonly used for phishing.
    """
    if not domain:
        return 0
    domain_lower = domain.lower()
    for tld in HIGH_RISK_TLDS:
        if domain_lower.endswith(tld):
            return 1
    return 0


def get_domain_age_days(domain: Optional[str], allow_network: bool = False) -> int:
    """
    Look up the WHOIS registration date for a domain and compute its age in days.
    If allow_network=False (default), skips network lookup and returns -1.
    """
    if not domain or not allow_network:
        return -1
    try:
        import whois
        w = whois.whois(domain)
        creation_date = w.creation_date
        if creation_date is None:
            return -1
        # Some registrars return a list of dates
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if not isinstance(creation_date, datetime):
            return -1
        # Make timezone-aware if naive
        if creation_date.tzinfo is None:
            creation_date = creation_date.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age = (now - creation_date).days
        return max(0, age)
    except Exception as e:
        logger.debug("WHOIS lookup failed for %s: %s", domain, e)
        return -1


def get_typosquatting_score(domain: Optional[str]) -> float:
    """
    Compute a typosquatting similarity score for a domain against known brands.
    """
    if not domain:
        return 0.0
    domain_lower = domain.lower().strip()
    max_score = 0.0
    for brand in KNOWN_BRANDS:
        if domain_lower == brand:
            # Exact match → it's the real brand, not a typosquat
            return 0.0
        # Compute normalized similarity
        distance = Levenshtein.distance(domain_lower, brand)
        max_len = max(len(domain_lower), len(brand))
        if max_len == 0:
            continue
        similarity = 1.0 - (distance / max_len)
        # Only consider high-similarity matches as potential typosquats
        if similarity > max_score:
            max_score = similarity
    return max_score if max_score >= 0.5 else 0.0


def check_blacklist(domain: Optional[str], allow_network: bool = False) -> int:
    """
    Check if a domain appears in threat lists or exhibits high-risk zero-day heuristics.
    Live Google Safe Browsing API is only queried if allow_network=True.
    By default (offline), uses static indicators: raw IP address or high-risk TLD.
    """
    if not domain:
        return 0

    import os
    import requests
    import re

    # Static offline heuristics (always evaluated, 0 network calls)
    ip_pattern = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
    if ip_pattern.match(domain):
        return 1

    if has_high_risk_tld(domain):
        return 1

    if not allow_network:
        return 0

    api_key = os.environ.get("GOOGLE_SAFE_BROWSING_API_KEY", "")
    if api_key:
        url = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={api_key}"
        payload = {
            "client": {
                "clientId": "phishing-detector",
                "clientVersion": "0.1.0",
            },
            "threatInfo": {
                "threatTypes": [
                    "MALWARE",
                    "SOCIAL_ENGINEERING",
                    "UNWANTED_SOFTWARE",
                    "POTENTIALLY_HARMFUL_APPLICATION",
                ],
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": [
                    {"url": f"http://{domain}/"},
                    {"url": f"https://{domain}/"},
                ],
            },
        }

        try:
            response = requests.post(url, json=payload, timeout=3)
            response.raise_for_status()
            data = response.json()
            if data.get("matches"):
                return 1
        except Exception:
            pass

    return 0


def check_whitelist(domain: Optional[str]) -> int:
    """
    Check if a domain matches any known-legitimate domain in the authoritative public whitelist.
    Uses offline tldextract with bundled PSL snapshot.
    """
    if not domain:
        return 0
    return 1 if is_exact_whitelisted_domain(domain) else 0


# ── New Domain features ────────────────────────────────────────────

def get_domain_length(domain: Optional[str]) -> int:
    """Length of the registered domain/hostname."""
    return len(domain) if domain else 0


def count_domain_digits(domain: Optional[str]) -> int:
    """Number of digits in the domain."""
    if not domain:
        return 0
    return sum(1 for c in domain if c.isdigit())


def get_domain_digit_ratio(domain: Optional[str]) -> float:
    """Ratio of digits to domain characters."""
    if not domain or len(domain) == 0:
        return 0.0
    return count_domain_digits(domain) / len(domain)


def count_domain_hyphens(domain: Optional[str]) -> int:
    """Number of hyphens in the domain."""
    if not domain:
        return 0
    return domain.count('-')


def get_domain_entropy(domain: Optional[str]) -> float:
    """Entropy of the domain string."""
    if not domain:
        return 0.0
    prob = [float(domain.count(c)) / len(domain) for c in dict.fromkeys(list(domain))]
    entropy = -sum(p * math.log(p) / math.log(2.0) for p in prob)
    return entropy


def get_subdomain_depth(domain: Optional[str]) -> int:
    """
    Number of hierarchical subdomain levels.
    Complementary to number_of_subdomains.
    """
    if not domain:
        return 0
    parts = domain.split(".")
    # A standard domain has 2 parts (e.g. example.com). Anything extra is depth.
    return max(0, len(parts) - 2)


def has_dns_record(domain: Optional[str], allow_network: bool = False) -> int:
    """
    Binary indicating whether the domain successfully resolves through DNS.
    If allow_network=False (default), returns -1 to indicate offline status.
    """
    if not domain or not allow_network:
        return -1
    try:
        resolver = dns.resolver.Resolver()
        resolver.timeout = 1
        resolver.lifetime = 1
        resolver.resolve(domain, 'A')
        return 1
    except Exception:
        return 0


def get_domain_registration_length(domain: Optional[str], allow_network: bool = False) -> int:
    """
    If WHOIS data is available: expiration_date - registration_date
    If allow_network=False (default), returns -1.
    """
    if not domain or not allow_network:
        return -1
    try:
        import whois
        w = whois.whois(domain)
        creation = w.creation_date
        expiration = w.expiration_date
        
        if creation is None or expiration is None:
            return -1
            
        if isinstance(creation, list):
            creation = creation[0]
        if isinstance(expiration, list):
            expiration = expiration[0]
            
        if not isinstance(creation, datetime) or not isinstance(expiration, datetime):
            return -1
            
        return max(0, (expiration - creation).days)
    except Exception:
        return -1


def get_days_until_expiration(domain: Optional[str], allow_network: bool = False) -> int:
    """
    If WHOIS data is available, calculate remaining registration time.
    If allow_network=False (default), returns -1.
    """
    if not domain or not allow_network:
        return -1
    try:
        import whois
        w = whois.whois(domain)
        expiration = w.expiration_date
        
        if expiration is None:
            return -1
            
        if isinstance(expiration, list):
            expiration = expiration[0]
            
        if not isinstance(expiration, datetime):
            return -1
            
        if expiration.tzinfo is None:
            expiration = expiration.replace(tzinfo=timezone.utc)
            
        now = datetime.now(timezone.utc)
        return max(0, (expiration - now).days)
    except Exception:
        return -1
    except Exception:
        return -1


def has_brand_in_domain(text: str, domain: Optional[str]) -> int:
    """
    Detect whether a brand/entity mentioned in the message appears in the domain.
    Do not invent brands - only use known brands extracted from the text.
    """
    if not domain:
        return 0
    text_lower = text.lower()
    domain_lower = domain.lower()
    domain_canonical = DOMAIN_ALIASES.get(domain_lower, domain_lower)
    
    for brand_name in BRAND_NAME_MAP.keys():
        if brand_name in text_lower:
            # The brand is mentioned in the text. Is it in the domain?
            if brand_name.replace(" ", "") in domain_lower or brand_name.replace(" ", "") in domain_canonical:
                return 1
    return 0


def has_brand_domain_mismatch(text: str, domain: Optional[str]) -> int:
    """
    Detect cases where the message appears to claim a brand/entity but the
    domain does not correspond to that claimed entity's official domain.

    Brand names that appear only inside their own legitimate URLs
    (e.g. "instagram" inside "https://www.instagram.com/reel/...")
    are excluded to prevent false positives in multi-link messages.
    """
    if not domain:
        return 0

    import re
    text_lower = text.lower()
    domain_lower = domain.lower()
    domain_canonical = DOMAIN_ALIASES.get(domain_lower, domain_lower)

    # Strip all URLs from the text so brand names appearing only inside
    # their own legitimate URLs are not treated as brand impersonation.
    text_no_urls = re.sub(
        r'https?://[^\s<>"\')\]]+|www\.[^\s<>"\')\]]+',
        '',
        text_lower,
    )
    # Also strip bare domain-like tokens (e.g. "youtu.be" on its own line)
    text_no_urls = re.sub(
        r'\b[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:/[^\s<>"\')\]]*)?',
        '',
        text_no_urls,
    )

    for brand_name, official_domain in BRAND_NAME_MAP.items():
        if brand_name in text_no_urls:
            # Brand is mentioned in the message body (not just inside a URL).
            # If the current primary domain is not the official one,
            # and it's not a legitimate subdomain of the official one, it's a mismatch.
            if (domain_lower != official_domain and not domain_lower.endswith("." + official_domain)) and \
               (domain_canonical != official_domain and not domain_canonical.endswith("." + official_domain)):
                return 1
    return 0
