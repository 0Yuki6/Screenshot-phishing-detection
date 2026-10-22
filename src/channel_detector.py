"""
Channel detection module — rule-based classification of message source.

Determines whether an OCR-extracted text originated from an email, SMS,
or WhatsApp message using keyword/pattern scoring. Each channel has a
set of weighted signals; the highest-scoring channel wins.

If no signals match, the channel is reported as "unknown".
"""

import re
from typing import Dict


# ── Signal definitions ───────────────────────────────────────────────
# Each signal is (pattern, weight). Patterns are compiled regex or plain strings.

_EMAIL_SIGNALS = [
    (re.compile(r'\bsubject\s*:', re.IGNORECASE), 3),
    (re.compile(r'\bfrom\s*:.*\bto\s*:', re.IGNORECASE | re.DOTALL), 3),
    (re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}'), 2),
    (re.compile(r'\b(reply|forward|fwd|re:)\b', re.IGNORECASE), 2),
    (re.compile(r'sent\s+from\s+my', re.IGNORECASE), 2),
    (re.compile(r'\b(inbox|unsubscribe|mailing\s*list)\b', re.IGNORECASE), 1),
    (re.compile(r'\b(cc|bcc)\s*:', re.IGNORECASE), 2),
    (re.compile(r'\battachment\b', re.IGNORECASE), 1),
]

_SMS_SIGNALS = [
    (re.compile(r'text\s*message', re.IGNORECASE), 3),
    (re.compile(r'\b\d{1,2}:\d{2}\s*(AM|PM)\b', re.IGNORECASE), 2),
    (re.compile(r'^\+?\d{10,15}$', re.MULTILINE), 3),
    (re.compile(r'\b(imessage|messages?)\b', re.IGNORECASE), 2),
    (re.compile(r'\b(sms|mms)\b', re.IGNORECASE), 3),
    (re.compile(r'today\s+\d{1,2}:\d{2}', re.IGNORECASE), 2),
    (re.compile(r'\b(carrier|signal|bars?)\b', re.IGNORECASE), 1),
]

_WHATSAPP_SIGNALS = [
    (re.compile(r'\bwhatsapp\b', re.IGNORECASE), 4),
    (re.compile(r'\b(delivered|read|online|last\s+seen)\b', re.IGNORECASE), 2),
    (re.compile(r'\bforwarded\b', re.IGNORECASE), 2),
    (re.compile(r'✓✓|✓|double.?check', re.IGNORECASE), 2),
    (re.compile(r'\btyping\.{3}\b|\btyping\b', re.IGNORECASE), 2),
    (re.compile(r'\bend-to-end\s+encrypt', re.IGNORECASE), 3),
    (re.compile(r'\b(voice\s+call|video\s+call)\b', re.IGNORECASE), 1),
    (re.compile(r'\b(group|broadcast)\s+(message|list)\b', re.IGNORECASE), 1),
]


def _score_channel(text: str, signals: list) -> int:
    """Compute total score for a channel by summing weights of matched signals."""
    total = 0
    for pattern, weight in signals:
        if pattern.search(text):
            total += weight
    return total


def detect_channel(ocr_text: str) -> Dict:
    """
    Detect the communication channel from OCR-extracted text.

    Uses rule-based pattern matching with weighted signals per channel.
    The channel with the highest score wins. Ties are broken by priority
    order: email > whatsapp > sms. If all scores are 0, returns "unknown".

    Args:
        ocr_text: Raw text extracted from the screenshot via OCR.

    Returns:
        Dictionary with:
            - "channel": str — "email", "sms", "whatsapp", or "unknown"
            - "scores": dict — per-channel raw scores for debugging
            - "confidence": int — winning channel's score (0 if unknown)
    """
    scores = {
        "email": _score_channel(ocr_text, _EMAIL_SIGNALS),
        "sms": _score_channel(ocr_text, _SMS_SIGNALS),
        "whatsapp": _score_channel(ocr_text, _WHATSAPP_SIGNALS),
    }

    max_score = max(scores.values())

    if max_score == 0:
        return {
            "channel": "unknown",
            "scores": scores,
            "confidence": 0,
        }

    # Priority order for tie-breaking
    priority = ["email", "whatsapp", "sms"]
    winning_channel = max(
        priority,
        key=lambda ch: (scores.get(ch, 0), -priority.index(ch)),
    )

    return {
        "channel": winning_channel,
        "scores": scores,
        "confidence": scores[winning_channel],
    }
