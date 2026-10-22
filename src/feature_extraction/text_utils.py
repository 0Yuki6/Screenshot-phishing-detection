"""
Text analysis utilities for phishing feature extraction.

Provides keyword counting, ratio computation, generic greeting detection,
and composite text features using the centralized keyword lists.
Also provides channel-aware text cleaning (e.g. WhatsApp noise removal).
"""

import re
from typing import List

from src.feature_extraction.keyword_lists import (
    URGENCY_KEYWORDS,
    THREAT_KEYWORDS,
    CREDENTIAL_KEYWORDS,
    CREDENTIAL_REQUEST_PATTERNS,
    PERSONAL_INFO_KEYWORDS,
    FINANCIAL_CONTEXT_KEYWORDS,
    ACCOUNT_SECURITY_KEYWORDS,
    REWARD_BAIT_KEYWORDS,
    ACTION_KEYWORDS,
)


# ── WhatsApp system message patterns to strip ────────────────────────
# These are UI chrome / platform disclaimers that pollute NLP features
# with words like 'encrypted', 'calls', 'chat', 'end-to-end', etc.
_WHATSAPP_NOISE_PATTERNS = [
    # Encryption disclaimers (various OCR variants)
    re.compile(
        r'messages?\s+and\s+calls?\s+are\s+end[- ]to[- ]end\s+encrypted[^.]*\.?',
        re.IGNORECASE,
    ),
    re.compile(
        r'messages?\s+to\s+this\s+chat\s+and\s+calls?\s+are\s+now\s+secured\s+with\s+end[- ]to[- ]end\s+encryption[^.]*\.?',
        re.IGNORECASE,
    ),
    re.compile(
        r'no\s+one\s+outside\s+of\s+this\s+chat.*?can\s+read\s+or\s+listen\s+to\s+them[^.]*\.?',
        re.IGNORECASE,
    ),
    re.compile(
        r'tap\s+to\s+learn\s+more\s*\.?',
        re.IGNORECASE,
    ),
    # "online" / "last seen" / typing indicators
    re.compile(r'\bonline\b', re.IGNORECASE),
    re.compile(r'\blast\s+seen\s+\w+\b', re.IGNORECASE),
    re.compile(r'\btyping\.\.\.\b', re.IGNORECASE),
    # Timestamp lines like "12:34 PM", "Today, 3:45 PM", "Yesterday, ..."
    re.compile(r'\b(?:today|yesterday)\s*,?\s*\d{1,2}:\d{2}\s*(?:am|pm)?\b', re.IGNORECASE),
    # WhatsApp footer / read receipts
    re.compile(r'✓✓?', re.IGNORECASE),
    re.compile(r'\bdelivered\b', re.IGNORECASE),
    re.compile(r'\bread\b(?:\s+\d{1,2}:\d{2})?\b', re.IGNORECASE),
]


def clean_whatsapp_noise(text: str) -> str:
    """
    Strip standard WhatsApp UI/system text that pollutes NLP features.

    Removes encryption disclaimers, read receipts, timestamps, and
    other platform chrome so that words like 'encrypted', 'calls',
    and 'chat' from the WhatsApp UI don't appear as phishing indicators.

    Args:
        text: The raw OCR text from a WhatsApp screenshot.

    Returns:
        Cleaned text with WhatsApp noise removed.
    """
    cleaned = text
    for pattern in _WHATSAPP_NOISE_PATTERNS:
        cleaned = pattern.sub('', cleaned)
    # Collapse multiple whitespace / blank lines left behind
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r'  +', ' ', cleaned)
    return cleaned.strip()


def clean_text_for_channel(text: str, channel: str) -> str:
    """
    Apply channel-specific text cleaning before feature extraction.

    Args:
        text: Raw OCR text.
        channel: Detected channel ('email', 'sms', 'whatsapp', etc.).

    Returns:
        Cleaned text ready for feature extraction.
    """
    if channel.lower() == 'whatsapp':
        return clean_whatsapp_noise(text)
    return text


def _count_keyword_matches(text: str, keywords: List[str]) -> int:
    """
    Count how many keywords from the given list appear in the text.

    Uses case-insensitive whole-word matching (with regex word boundaries)
    so that substrings like 'pan' inside 'japan' or 'won' inside 'wonderful'
    do NOT trigger false positive matches. Each keyword is counted at most once
    (presence-based, not occurrence-based).

    Args:
        text: The text to search within.
        keywords: List of keyword strings to look for.

    Returns:
        Number of distinct keywords found in the text.
    """
    text_lower = text.lower()
    count = 0
    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
        if re.search(pattern, text_lower):
            count += 1
    return count


def get_total_word_count(text: str) -> int:
    """
    Return the total number of whitespace-delimited words in the text.
    """
    words = text.split()
    return len(words)


def get_keyword_ratio(keyword_count: int, total_words: int) -> float:
    """
    Compute the ratio of keyword matches to total words.

    Returns 0.0 if total_words is 0 to avoid division by zero.
    """
    if total_words == 0:
        return 0.0
    return keyword_count / total_words


# ── New text feature functions ───────────────────────────────────────

def count_urgency_keywords(text: str) -> int:
    """Count urgency/pressure language keywords."""
    return _count_keyword_matches(text, URGENCY_KEYWORDS)


def count_threat_keywords(text: str) -> int:
    """Count consequence/threat keywords."""
    return _count_keyword_matches(text, THREAT_KEYWORDS)


def count_credential_requests(text: str) -> int:
    """
    Count actual requests for sensitive credentials.

    Checks both individual credential terms (password, OTP, PIN, CVV…)
    and request patterns (enter your OTP, confirm your password…).

    Does NOT treat generic words like 'account', 'details', 'verify',
    or 'information' as credential requests.
    """
    count = _count_keyword_matches(text, CREDENTIAL_KEYWORDS)
    count += _count_keyword_matches(text, CREDENTIAL_REQUEST_PATTERNS)
    return count


def count_personal_info_requests(text: str) -> int:
    """
    Count requests for personal identifying information.

    Detects Aadhaar, PAN, date of birth, phone number, address,
    identity number, personal details, etc.
    """
    return _count_keyword_matches(text, PERSONAL_INFO_KEYWORDS)


def count_financial_context(text: str) -> int:
    """
    Count financial/banking context keywords.

    This is contextual evidence only — these words alone do not
    indicate phishing.
    """
    return _count_keyword_matches(text, FINANCIAL_CONTEXT_KEYWORDS)


def count_account_security(text: str) -> int:
    """
    Count account/login/security context keywords.

    These are contextual features, not automatic phishing indicators.
    """
    return _count_keyword_matches(text, ACCOUNT_SECURITY_KEYWORDS)


def count_reward_bait(text: str) -> int:
    """Count reward/prize/bait keywords."""
    return _count_keyword_matches(text, REWARD_BAIT_KEYWORDS)


def count_action_words(text: str) -> int:
    """Count action-oriented words (verify, confirm, click, etc.)."""
    return _count_keyword_matches(text, ACTION_KEYWORDS)


def has_threat_plus_urgency(text: str) -> int:
    """
    Binary feature: 1 if BOTH threat and urgency signals are present.

    Captures the combination pattern rather than relying on individual words.
    """
    has_threat = count_threat_keywords(text) > 0
    has_urgency = count_urgency_keywords(text) > 0
    return 1 if (has_threat and has_urgency) else 0


def has_sensitive_request(text: str) -> int:
    """
    Binary feature indicating whether the message contains an actual
    request for sensitive information (credentials OR personal info).

    Does NOT trigger from generic words like 'details' alone.
    """
    has_credential = count_credential_requests(text) > 0
    has_personal = count_personal_info_requests(text) > 0
    return 1 if (has_credential or has_personal) else 0


def has_generic_greeting(text: str) -> int:
    """
    Detect generic / impersonal greetings commonly used in phishing messages.

    Checks for patterns like "Dear Customer", "Dear User", "Dear Account Holder",
    "Dear Sir/Madam", "Dear Valued", etc.

    Returns:
        1 if a generic greeting is found, 0 otherwise.
    """
    generic_patterns = [
        r'\bdear\s+customer\b',
        r'\bdear\s+user\b',
        r'\bdear\s+account\s*holder\b',
        r'\bdear\s+sir\b',
        r'\bdear\s+madam\b',
        r'\bdear\s+sir\s*/\s*madam\b',
        r'\bdear\s+valued\s+\w+\b',
        r'\bdear\s+member\b',
        r'\bdear\s+client\b',
        r'\bdear\s+recipient\b',
        r'\bdear\s+subscriber\b',
        r'\bhello\s+user\b',
        r'\bhello\s+customer\b',
        r'\bgreetings\b',
    ]
    text_lower = text.lower()
    for pattern in generic_patterns:
        if re.search(pattern, text_lower):
            return 1
    return 0
