"""
Channel detection module — rule-based classification of message source.

Determines whether OCR-extracted text came from an email, SMS or WhatsApp screenshot
using weighted keyword/pattern signals. The highest-scoring channel wins.
If no signals match, the channel is reported as "unknown".

Signals cover both iPhone-style screens and Android apps (Gmail, Google Messages, WhatsApp).
"""

import re
from typing import Dict

_MONTHS = (
    r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|'
    r'aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
)

_EMAIL_SIGNALS = [
    (re.compile(r'\bsubject\s*:', re.IGNORECASE), 3),
    (re.compile(r'\bfrom\s*:.*\bto\s*:', re.IGNORECASE | re.DOTALL), 3),
    (re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}'), 2),
    (re.compile(r'\b(reply|forward|fwd|re:)\b', re.IGNORECASE), 2),
    (re.compile(r'sent\s+from\s+my', re.IGNORECASE), 2),
    (re.compile(r'\b(inbox|mailing\s*list)\b', re.IGNORECASE), 1),
    (re.compile(r'\bunsubscribe\b', re.IGNORECASE), 2),
    (re.compile(r'\b(cc|bcc)\s*:', re.IGNORECASE), 2),
    (re.compile(r'\battachment\b', re.IGNORECASE), 1),
    # Gmail / Outlook app header line: "to me", "to <name>" next to the sender
    (re.compile(r'\bto\s*me\b', re.IGNORECASE), 3),
    (re.compile(r'\be-?mail\b', re.IGNORECASE), 1),
]

_SMS_SIGNALS = [
    (re.compile(r'text\s*message', re.IGNORECASE), 3),
    (re.compile(r'\b\d{1,2}:\d{2}\s*(AM|PM)\b', re.IGNORECASE), 1),
    (re.compile(r'^\+?\d{10,15}$', re.MULTILINE), 3),
    (re.compile(r'\b(imessage|messages?)\b', re.IGNORECASE), 1),
    (re.compile(r'\b(sms|mms|rcs)\b', re.IGNORECASE), 3),
    (re.compile(r'today\s+\d{1,2}:\d{2}', re.IGNORECASE), 2),
    (re.compile(r'\b(carrier|signal|bars?)\b', re.IGNORECASE), 1),
    # Google Messages / Android SMS screens
    (re.compile(r"(can'?t|cannot)\s+accept\s+replies", re.IGNORECASE), 5),
    (re.compile(r'\btap\s+to\s+load\s+preview\b', re.IGNORECASE), 3),
    (re.compile(r'\btexting\s+with\b', re.IGNORECASE), 4),
    (re.compile(r'\bunread\b', re.IGNORECASE), 2),
    # Alphanumeric sender header used on Indian (TRAI) and many other SMS: AX-AIRTEL-P, JD-SBIBNK-S
    (re.compile(r'\b[A-Z]{2}-[A-Z0-9]{4,9}(?:-[A-Z])?\b'), 3),
]

_WHATSAPP_SIGNALS = [
    (re.compile(r'\bwhatsapp\b', re.IGNORECASE), 4),
    (re.compile(r'\b(delivered|read|online|last\s+seen)\b', re.IGNORECASE), 1),
    (re.compile(r'\bforwarded\b', re.IGNORECASE), 2),
    (re.compile(r'✓✓|✓|double.?check', re.IGNORECASE), 2),
    (re.compile(r'\btyping\b', re.IGNORECASE), 2),
    (re.compile(r'\bend-to-end\s+encrypt', re.IGNORECASE), 4),
    (re.compile(r'\b(voice\s+call|video\s+call)\b', re.IGNORECASE), 1),
    (re.compile(r'\b(group|broadcast)\s+(message|list)\b', re.IGNORECASE), 1),
    # WhatsApp date separators: "October 9, 2026" / "Today" / "Yesterday" chips
    # a date followed by a clock time is an SMS thread header ("Friday, Jul 4, 2025 - 10:15 AM"), not a chat date chip
    (re.compile(r'\b' + _MONTHS + r'\s+\d{1,2},?\s+20\d{2}\b(?!\s*\W?\s*\d{1,2}:\d{2})', re.IGNORECASE), 3),
    (re.compile(r'^\s*(today|yesterday)\s*$', re.IGNORECASE | re.MULTILINE), 1),
    # Message-input placeholder at the bottom of the chat screen
    (re.compile(r'^\W*message\W*$', re.IGNORECASE | re.MULTILINE), 2),
]


_TIME_RE = re.compile(r'\b\d{1,2}:\d{2}\s?(?:AM|PM)\b', re.IGNORECASE)
_DATE_WORD_RE = re.compile(
    r'(?:mon|tue|wed|thu|fri|sat|sun|' + _MONTHS + r'|20\d\d)', re.IGNORECASE
)


def _bare_times(text: str) -> int:
    """Count clock times that are NOT part of a dated header ("Friday, Jul 4, 2025 - 10:15 AM").
    Chat bubbles (WhatsApp) show a bare time under every message; SMS thread headers show a date."""
    n = 0
    for m in _TIME_RE.finditer(text):
        before = text[max(0, m.start() - 22):m.start()]
        if not _DATE_WORD_RE.search(before):
            n += 1
    return n


def _score_channel(text: str, signals: list) -> int:
    total = 0
    for pattern, weight in signals:
        if pattern.search(text):
            total += weight
    return total


def detect_channel(ocr_text: str) -> Dict:
    """
    Returns {"channel": "email"|"sms"|"whatsapp"|"unknown", "scores": {...}, "confidence": int}.
    Ties are broken by priority email > whatsapp > sms; all-zero scores give "unknown".
    """
    scores = {
        "email": _score_channel(ocr_text, _EMAIL_SIGNALS),
        "sms": _score_channel(ocr_text, _SMS_SIGNALS),
        "whatsapp": _score_channel(ocr_text, _WHATSAPP_SIGNALS),
    }
    # Many bare bubble times and no strong SMS/email evidence -> a WhatsApp-style chat.
    if _bare_times(ocr_text) >= 3 and scores["email"] < 3 and scores["sms"] < 3:
        scores["whatsapp"] += 3

    max_score = max(scores.values())
    if max_score == 0:
        return {"channel": "unknown", "scores": scores, "confidence": 0}

    priority = ["email", "whatsapp", "sms"]
    winning_channel = max(
        priority, key=lambda ch: (scores.get(ch, 0), -priority.index(ch))
    )
    return {"channel": winning_channel, "scores": scores, "confidence": scores[winning_channel]}