"""
Public Brand & Domain Whitelist.

Constructed exclusively from top global financial, technology, telecom,
and commerce services (Tranco / Alexa Top Domains). Never derived from holdout data.
"""

from typing import Dict, Set, Optional
import tldextract


# Global verified brand-to-registered-domains mapping
PUBLIC_BRAND_DOMAINS: Dict[str, Set[str]] = {
    # Tech & Infrastructure
    "google": {"google.com", "google.co.uk", "google.co.in", "youtube.com", "gmail.com"},
    "microsoft": {"microsoft.com", "office.com", "live.com", "outlook.com", "azure.com", "microsoftonline.com"},
    "apple": {"apple.com", "icloud.com"},
    "amazon": {"amazon.com", "amazon.co.uk", "amazon.in", "aws.amazon.com"},
    "meta": {"facebook.com", "instagram.com", "whatsapp.com", "meta.com"},
    "x": {"x.com", "twitter.com"},
    "linkedin": {"linkedin.com"},
    "netflix": {"netflix.com"},
    "spotify": {"spotify.com"},
    "github": {"github.com"},
    "adobe": {"adobe.com"},
    "dropbox": {"dropbox.com"},
    "zoom": {"zoom.us"},
    "slack": {"slack.com"},
    "yahoo": {"yahoo.com"},
    "twitch": {"twitch.tv"},
    "steam": {"steampowered.com", "steamcommunity.com"},

    # Financial & Payments (Global & Major Indian Institutions)
    "paypal": {"paypal.com"},
    "stripe": {"stripe.com"},
    "chase": {"chase.com"},
    "bank of america": {"bankofamerica.com"},
    "wells fargo": {"wellsfargo.com"},
    "citi": {"citi.com", "citibank.com"},
    "american express": {"americanexpress.com", "amex.com"},
    "capital one": {"capitalone.com"},
    "discover": {"discover.com"},
    "fidelity": {"fidelity.com"},
    "schwab": {"schwab.com"},
    "vanguard": {"vanguard.com"},
    "venmo": {"venmo.com"},
    "cash app": {"cash.app"},
    "intuit": {"intuit.com", "turbotax.com", "quickbooks.com"},
    "barclays": {"barclays.co.uk", "barclays.com"},
    "hsbc": {"hsbc.com", "hsbc.co.uk"},

    # Major Indian Banks & Financial Institutions
    "sbi": {"sbi.co.in", "onlinesbi.sbi", "statebankofindia.com"},
    "state bank of india": {"sbi.co.in", "onlinesbi.sbi", "statebankofindia.com"},
    "hdfc": {"hdfcbank.com", "hdfc.com"},
    "hdfc bank": {"hdfcbank.com"},
    "icici": {"icicibank.com"},
    "icici bank": {"icicibank.com"},
    "axis bank": {"axisbank.com"},
    "kotak": {"kotak.com"},
    "kotak mahindra": {"kotak.com"},
    "punjab national bank": {"pnbindia.in"},
    "pnb": {"pnbindia.in"},
    "bank of baroda": {"bankofbaroda.in"},

    # Major Indian UPI & Payment Services
    "phonepe": {"phonepe.com"},
    "paytm": {"paytm.com"},
    "cred": {"cred.club"},
    "bharatpe": {"bharatpe.com"},
    "npci": {"npci.org.in"},
    "bhim": {"npci.org.in", "bhimupi.org.in"},
    "razorpay": {"razorpay.com"},
    "mobikwik": {"mobikwik.com"},

    # Indian Telecom Providers
    "jio": {"jio.com", "reliancedigital.in"},
    "airtel": {"airtel.in", "airtel.com", "i.airtel"},
    "vi": {"myvi.in"},
    "vodafone idea": {"myvi.in"},
    "bsnl": {"bsnl.co.in"},

    # Logistics & Retail (Global & Indian)
    "fedex": {"fedex.com"},
    "ups": {"ups.com"},
    "usps": {"usps.com"},
    "dhl": {"dhl.com"},
    "india post": {"indiapost.gov.in"},
    "delhivery": {"delhivery.com"},
    "blue dart": {"bluedart.com"},
    "flipkart": {"flipkart.com"},
    "swiggy": {"swiggy.com"},
    "zomato": {"zomato.com"},
    "myntra": {"myntra.com"},
    "zepto": {"zeptonow.com"},
    "blinkit": {"blinkit.com"},
    "walmart": {"walmart.com"},
    "target": {"target.com"},
    "ebay": {"ebay.com"},
    "best buy": {"bestbuy.com"},
    "uber": {"uber.com"},
    "lyft": {"lyft.com"},
    "airbnb": {"airbnb.com"},
    "booking": {"booking.com"},

    # Major Indian Government & Public Portals
    "income tax": {"incometax.gov.in"},
    "incometax": {"incometax.gov.in"},
    "uidai": {"uidai.gov.in"},
    "aadhaar": {"uidai.gov.in"},
    "digilocker": {"digilocker.gov.in"},
    "irctc": {"irctc.co.in"},
    "parivahan": {"parivahan.gov.in"},
    "passport seva": {"passportindia.gov.in"},
    "epfo": {"epfindia.gov.in"},
    "india.gov.in": {"india.gov.in"},
}

# Flat set of all exact registered domains
ALL_WHITELISTED_DOMAINS: Set[str] = {
    domain
    for domains in PUBLIC_BRAND_DOMAINS.values()
    for domain in domains
}


# High-risk TLD list commonly observed in phishing
HIGH_RISK_TLDS: Set[str] = {
    ".xyz", ".top", ".work", ".buzz", ".club", ".icu", ".cam", ".wang",
    ".gq", ".ml", ".cf", ".ga", ".tk", ".fit", ".rest", ".monster",
    ".quest", ".surf", ".sbs", ".click", ".link", ".online", ".site"
}

# Known free URL shorteners
KNOWN_SHORTENERS: Set[str] = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "buff.ly",
    "ow.ly", "cutt.ly", "rb.gy", "shorturl.at"
}


# Offline tldextract extractor (using bundled PSL snapshot, zero network calls)
_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=None, fallback_to_snapshot=True)


def extract_registered_domain(url_or_host: str) -> str:
    """
    Extract the registered domain (domain + suffix) using offline tldextract.
    
    Example:
        'https://sub.paypal.com/path' -> 'paypal.com'
        'paypal.com.evil.xyz' -> 'evil.xyz'
    """
    if not url_or_host:
        return ""
    extracted = _EXTRACTOR(url_or_host)
    if not extracted.domain:
        return ""
    if extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}".lower()
    return ""


def is_exact_whitelisted_domain(url_or_host: str) -> bool:
    """
    Check if the URL's registered domain is strictly in the public whitelist.
    
    Prevents subdomain spoofing:
        is_exact_whitelisted_domain('https://login.paypal.com') -> True
        is_exact_whitelisted_domain('https://paypal.com.evil.xyz') -> False (registered domain is evil.xyz)
        is_exact_whitelisted_domain('https://paypal-security.com') -> False
    """
    reg_domain = extract_registered_domain(url_or_host)
    return reg_domain in ALL_WHITELISTED_DOMAINS


def check_brand_domain_mismatch(text: str, url_or_host: str) -> bool:
    """
    Check if a known brand is named in the message text but the URL's registered domain
    belongs to a completely different entity.
    
    Returns:
        True if brand is mentioned AND registered domain is NOT in brand's approved domains.
        False if no brand mentioned, or if registered domain matches the mentioned brand.
    """
    if not text or not url_or_host:
        return False

    reg_domain = extract_registered_domain(url_or_host)
    if not reg_domain:
        return False

    text_lower = text.lower()
    for brand, legit_domains in PUBLIC_BRAND_DOMAINS.items():
        # Match brand name as a whole word
        import re
        if re.search(rf"\b{re.escape(brand)}\b", text_lower):
            if reg_domain not in legit_domains:
                return True

    return False
