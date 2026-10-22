"""
Centralized keyword and domain configuration lists for phishing detection.

All keyword lists are stored in lowercase. Matching functions in text_utils.py
perform case-insensitive comparison.

Lists are organised into non-overlapping groups so that a single word does
not inflate multiple feature counts simultaneously.
"""

# ──────────────────────────────────────────────
# 1. Urgency / pressure language
# ──────────────────────────────────────────────
URGENCY_KEYWORDS = [
    "urgent",
    "immediately",
    "act now",
    "act immediately",
    "within 24 hours",
    "within 48 hours",
    "expires today",
    "last warning",
    "final notice",
    "as soon as possible",
    "limited time",
    "expiring",
    "last chance",
    "time is running out",
    "hurry",
    "right away",
]

# ──────────────────────────────────────────────
# 2. Threat / consequence language
# ──────────────────────────────────────────────
THREAT_KEYWORDS = [
    "suspended",
    "blocked",
    "terminated",
    "deactivated",
    "restricted",
    "account locked",
    "access revoked",
    "penalty",
    "legal action",
    "closed",
    "compromised",
    "unauthorized",
    "failure to",
    "will be closed",
    "permanently disabled",
]

# ──────────────────────────────────────────────
# 3. Credential request keywords
#    Actual sensitive credential terms (NOT generic words like "account")
# ──────────────────────────────────────────────
CREDENTIAL_KEYWORDS = [
    "password",
    "otp",
    "pin",
    "cvv",
    "card number",
    "username",
    "security code",
    "verification code",
    "login credentials",
    "ssn",
    "social security",
    "routing number",
]

# Patterns that indicate an actual *request* for credentials
CREDENTIAL_REQUEST_PATTERNS = [
    "enter your otp",
    "provide your pin",
    "confirm your password",
    "submit your card details",
    "enter your password",
    "enter your credentials",
    "provide your password",
    "share your otp",
    "type your pin",
    "input your cvv",
    "update your password",
    "reset your password",
]

# ──────────────────────────────────────────────
# 4. Personal information request keywords
# ──────────────────────────────────────────────
PERSONAL_INFO_KEYWORDS = [
    "aadhaar",
    "pan",
    "date of birth",
    "phone number",
    "address",
    "identity number",
    "personal details",
    "passport number",
    "driving license",
    "national id",
]

# ──────────────────────────────────────────────
# 5. Financial / banking context keywords
# ──────────────────────────────────────────────
FINANCIAL_CONTEXT_KEYWORDS = [
    "bank",
    "payment",
    "transaction",
    "refund",
    "invoice",
    "credit",
    "debit",
    "wallet",
    "kyc",
    "upi",
    "card",
    "billing",
    "wire transfer",
    "bitcoin",
    "cryptocurrency",
]

# ──────────────────────────────────────────────
# 6. Account / security context keywords
# ──────────────────────────────────────────────
ACCOUNT_SECURITY_KEYWORDS = [
    "account",
    "login",
    "security",
    "verification",
    "authentication",
    "access",
    "sign in",
    "log in",
]

# ──────────────────────────────────────────────
# 7. Reward / bait keywords
# ──────────────────────────────────────────────
REWARD_BAIT_KEYWORDS = [
    "you won",
    "claim your prize",
    "free gift",
    "cashback",
    "reward",
    "bonus",
    "exclusive offer",
    "congratulations",
    "winner",
    "free upgrade",
    "prize",
    "lucky",
    "selected for",
]

# ──────────────────────────────────────────────
# 8. Action-oriented words
# ──────────────────────────────────────────────
ACTION_KEYWORDS = [
    "verify",
    "confirm",
    "update",
    "activate",
    "submit",
    "claim",
    "click",
    "login",
    "download",
    "open",
    "click here",
    "tap here",
]

# ──────────────────────────────────────────────
# 9. Suspicious keywords inside URLs
# ──────────────────────────────────────────────
SUSPICIOUS_URL_KEYWORDS = [
    "login",
    "signin",
    "verify",
    "verification",
    "secure",
    "account",
    "update",
    "password",
    "banking",
    "confirm",
    "wallet",
]

# ──────────────────────────────────────────────
# Action words used for action_plus_url detection
# ──────────────────────────────────────────────
ACTION_URL_TRIGGERS = [
    "verify",
    "confirm",
    "login",
    "update",
    "activate",
    "claim",
]

# ──────────────────────────────────────────────
# Known-legitimate domains (whitelist)
# ──────────────────────────────────────────────
LEGITIMATE_DOMAINS = [
    "google.com",
    "drive.google.com",
    "mail.google.com",
    "youtube.com",
    "youtu.be",
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "microsoft.com",
    "outlook.com",
    "live.com",
    "apple.com",
    "icloud.com",
    "amazon.com",
    "aws.amazon.com",
    "netflix.com",
    "github.com",
    "dropbox.com",
    "zoom.us",
    "slack.com",
    "whatsapp.com",
    "telegram.org",
    "reddit.com",
    "wikipedia.org",
    "stackoverflow.com",
    "adobe.com",
    "salesforce.com",
    "stripe.com",
    "paypal.com",
]

# ──────────────────────────────────────────────
# URL shortener domains
# ──────────────────────────────────────────────
URL_SHORTENERS = [
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "rebrand.ly",
    "bl.ink",
    "short.io",
    "cutt.ly",
    "rb.gy",
    "v.gd",
    "qr.ae",
    "lnkd.in",
    "db.tt",
    "adf.ly",
    "shorte.st",
    "bc.vc",
    "shorturl.at",
]

# ──────────────────────────────────────────────
# Known brand domains (for typosquatting detection)
# ──────────────────────────────────────────────
KNOWN_BRANDS = [
    "google.com",
    "facebook.com",
    "apple.com",
    "amazon.com",
    "microsoft.com",
    "netflix.com",
    "paypal.com",
    "instagram.com",
    "twitter.com",
    "linkedin.com",
    "dropbox.com",
    "chase.com",
    "wellsfargo.com",
    "bankofamerica.com",
    "citibank.com",
    "americanexpress.com",
    "usps.com",
    "fedex.com",
    "ups.com",
    "dhl.com",
    "ebay.com",
    "walmart.com",
    "target.com",
    "bestbuy.com",
    "costco.com",
    "spotify.com",
    "zoom.us",
    "slack.com",
    "github.com",
    "adobe.com",
]

# ──────────────────────────────────────────────
# Brand name → official domain mapping
# Used for brand_in_domain and brand_domain_mismatch
# ──────────────────────────────────────────────
BRAND_NAME_MAP = {
    "google": "google.com",
    "facebook": "facebook.com",
    "apple": "apple.com",
    "amazon": "amazon.com",
    "microsoft": "microsoft.com",
    "netflix": "netflix.com",
    "paypal": "paypal.com",
    "instagram": "instagram.com",
    "twitter": "twitter.com",
    "linkedin": "linkedin.com",
    "dropbox": "dropbox.com",
    "chase": "chase.com",
    "wells fargo": "wellsfargo.com",
    "wellsfargo": "wellsfargo.com",
    "bank of america": "bankofamerica.com",
    "citibank": "citibank.com",
    "american express": "americanexpress.com",
    "amex": "americanexpress.com",
    "usps": "usps.com",
    "fedex": "fedex.com",
    "ups": "ups.com",
    "dhl": "dhl.com",
    "ebay": "ebay.com",
    "walmart": "walmart.com",
    "spotify": "spotify.com",
    "zoom": "zoom.us",
    "slack": "slack.com",
    "github": "github.com",
    "adobe": "adobe.com",
    "sbi": "sbi.co.in",
    "hdfc": "hdfcbank.com",
    "icici": "icicibank.com",
    "axis": "axisbank.com",
    "kotak": "kotak.com",
    "whatsapp": "whatsapp.com",
    "telegram": "telegram.org",
    "youtube": "youtube.com",
}

# ──────────────────────────────────────────────
# Known domain aliases (official shorteners & brand redirects)
# ──────────────────────────────────────────────
DOMAIN_ALIASES = {
    "youtu.be": "youtube.com",
    "g.co": "google.com",
    "goo.gl": "google.com",
    "wa.me": "whatsapp.com",
    "amzn.to": "amazon.com",
    "fb.me": "facebook.com",
    "t.co": "twitter.com",
    "x.com": "twitter.com",
    "twitter.com": "x.com",
    "instagr.am": "instagram.com",
    "msft.it": "microsoft.com",
    "git.io": "github.com",
    "spoti.fi": "spotify.com",
    "apple.co": "apple.com",
    "lnkd.in": "linkedin.com",
}

# ──────────────────────────────────────────────
# High-risk TLDs commonly used for phishing
# ──────────────────────────────────────────────
HIGH_RISK_TLDS = [
    ".xyz",
    ".top",
    ".ru",
    ".tk",
    ".ml",
    ".ga",
    ".cf",
    ".gq",
    ".club",
    ".online",
    ".site",
    ".vip",
    ".work",
    ".biz",
    ".info",
    ".live",
    ".icu",
    ".pw",
    ".cc",
    ".ws",
    ".su",
]
