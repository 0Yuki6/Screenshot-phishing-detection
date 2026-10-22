"""
Text normalization, template chrome stripping, and word-boundary masking.

Removes UI chrome (status bars, client search/navigation buttons, WhatsApp encryption notices)
and masks variable tokens (years, numbers, emails, URLs) to prevent models from learning
dataset/template artifacts.
"""

import re
from typing import Optional


# ── 1. General Chrome Stripping Regexes ──────────────────────────────────────

# Generic end-to-end encryption / security notices (non-greedy, strictly matching the notice phrase)
_RE_SECURITY_NOTICE = re.compile(
    r"@?\s*(?:messages|calls|chats|conversations)\s+[^.]*?end-to-end\s+encrypted[^.]*\.?\s*"
    r"(?:only\s+people\s+in\s+this\s+chat\s+can\s+read[^.]*\.?)?\s*"
    r"(?:read,\s*listen\s+to,\s*or\s+share\s+them[^.]*\.?)?",
    re.IGNORECASE,
)
# WhatsApp / SMS status bar and contact headers at start of text
_RE_CHAT_HEADER = re.compile(
    r"^\s*\d{1,2}:\d{2}\s*(?:am|pm)?[^\n:]*?:\s*(?:today|yesterday|[a-z]{3},\s+[a-z]{3}\s+\d{1,2})\b\s*",
    re.IGNORECASE,
)
# Trailing chat input bar footer
_RE_CHAT_FOOTER = re.compile(
    r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\s*[\W_]*\s*message\b.*$",
    re.IGNORECASE,
)
_RE_CHAT_STATUS = re.compile(
    r"^(?:online|typing\.\.\.|last\s+seen\s+.*|business\s+account|forwarded(?:\s+many\s+times)?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_RE_READ_RECEIPTS = re.compile(r"[✓✔]{1,2}\s*(?:read|delivered)?", re.IGNORECASE)

# Generic corporate disclosure & routing boilerplate
_RE_CORP_BOILERPLATE = re.compile(
    r"\b(?:confidentiality\s+notice|all\s+rights\s+reserved|internal\s+use\s+only|"
    r"privileged/confidential|this\s+e-?mail\s+is\s+intended\s+only\s+for|forwarded\s+by\s+[a-z\s]+)\b",
    re.IGNORECASE,
)

# Generic email client navigation & folders pattern
_RE_CLIENT_NAVIGATION = re.compile(
    r"^[\W_]*\s*(?:search\s+mail|inbox|starred|stared|snoozed|sent|drafts|trash|archive|"
    r"deleted\s+items|junk\s+mail|spam)\b.*$",
    re.IGNORECASE,
)

# Footers / Subscriptions
_RE_FOOTERS = re.compile(
    r"(?:to\s+unsubscribe|click\s+here\s+to\s+unsubscribe|view\s+(?:this\s+email\s+)?in\s+your\s+browser|"
    r"privacy\s+policy|terms\s+(?:of\s+service|and\s+conditions)).*$",
    re.IGNORECASE | re.MULTILINE,
)


def strip_template_chrome(text: str, channel: Optional[str] = None) -> str:
    """
    Remove template chrome, headers, status bars, and boilerplate using general patterns.
    
    Args:
        text: Raw OCR text.
        channel: Optional channel ('email', 'sms', 'whatsapp').
        
    Returns:
        Cleaned text content stripped of UI chrome.
    """
    if not text:
        return ""

    cleaned = text

    # Strip chat status bar header and trailing input bar footer
    cleaned = _RE_CHAT_HEADER.sub("", cleaned)
    cleaned = _RE_CHAT_FOOTER.sub("", cleaned)

    # Strip security notices & read receipts
    cleaned = _RE_SECURITY_NOTICE.sub(" ", cleaned)
    cleaned = _RE_READ_RECEIPTS.sub(" ", cleaned)

    # Strip corporate boilerplate
    cleaned = _RE_CORP_BOILERPLATE.sub(" ", cleaned)

    # Line-by-line filtering using general patterns
    preserved_lines = []
    lines = cleaned.split("\n")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        lower = stripped.lower()

        # Chat status words
        if lower in {"online", "typing...", "business account", "forwarded", "forwarded many times"}:
            continue
        if lower.startswith("last seen"):
            continue

        # Mobile Status bar line: time + battery or signal or standalone time/date
        # Only discard short lines (< 80 chars) that look like status bars
        has_time = bool(re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b", lower))
        has_battery = bool(re.search(r"\b\d{1,3}%\b", lower))
        has_signal = bool(re.search(r"\b(?:all|ull)\s*(?:4[gc]|5[gc]|lte|tte)\b", lower))
        has_date = bool(re.search(r"\b(?:mon|tue|wed|thu|fri|sat|sun),?\s+[a-z]{3,9}\s+\d{1,2}\b", lower))
        is_today_yesterday = lower in {"today", "yesterday"}

        if len(stripped) < 80:
            if (has_time and (has_battery or has_signal)) or (has_battery and has_signal) or is_today_yesterday:
                continue
        if (has_time or has_date) and len(stripped.split()) <= 4 and not re.search(r"[a-z]{5,}", lower):
            # standalone time/date header
            continue

        # SMS contact header: e.g. "< Contact Name" or phone number header
        if re.match(r"^<\s*(?:\([a-z]\)\s*)?[a-z][a-z]+\s+[a-z]\.?.*$", lower):
            continue
        if re.match(r"^(?:[^\w\s])?\s*(?:\([a-z]\)\s*)?\+?1?\s*\(?\d{3}\)?[\s.-]*\d{3}[\s.-]*\d{4}.*$", lower):
            continue

        # Email Client Navigation & Folders
        # Only match short standalone nav/sidebar lines (< 60 chars) to avoid
        # gutting content-bearing lines that contain sidebar words in OCR output
        if len(stripped) < 60:
            if any(term in lower for term in [
                "search mail", "search in mail", "deleted items", "deleted messages",
                "junk mail", "sent tems", "deleted tems"
            ]):
                continue
            if lower in {"sent", "& sent", "| drafts", "e# drafts", "drafts",
                         "trash", "tome", "inbox", "starred", "stared", "snoozed",
                         "compose", "spam", "archive"}:
                continue

        # Email Headers (RFC standard headers)
        if re.match(r"^(?:from|to|subject|date|cc|bcc|reply-to):\s*.*$", lower):
            continue
        if re.match(r"^[^\w\s]*\s*(?:mon|tue|wed|thu|fri|sat|sun),?\s+[a-z]{3,9}\s+\d{1,2},?\s+\d{4}.*$", lower):
            continue

        # Footers & Unsubscribe
        if any(term in lower for term in ["to unsubscribe", "click here to unsubscribe", "view in your browser", "view this email in your browser"]):
            continue

        preserved_lines.append(stripped)

    result = "\n".join(preserved_lines)
    return result.strip()


# ── 2. Variable Token Masking ────────────────────────────────────────────────

_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_RE_YEAR = re.compile(r"\b(19\d\d|20\d\d)\b")
_RE_DATE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b|"
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b|"
    r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)(?:day)?\b",
    re.IGNORECASE,
)
_RE_TIME = re.compile(
    r"\b(?:today|yesterday|tomorrow)\b|"
    r"\b\d+\s*(?:mins?|minutes?|hours?|hrs?|days?|secs?|seconds?)\s+ago\b|"
    r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b|\b(?:am|pm)\b",
    re.IGNORECASE,
)
_RE_UI_WORDS = re.compile(
    r"\b(?:unread|dismiss|tap\s+to\s+load(?:\s+preview)?|unsubscribe|"
    r"forwarded(?:\s+many\s+times)?|enter\s+message|sent\s+from\s+my|"
    r"search(?:\s+in)?\s+mail|starred|snoozed|view\s+in\s+browser|reply)\b",
    re.IGNORECASE,
)
_RE_NUM = re.compile(r"\b\d+\b")
_RE_PUNCT_SPAM = re.compile(r"[!?.]{2,}")


def mask_text(text: str, mask_urls: bool = True) -> str:
    """
    Mask variable tokens with standard semantic placeholders.
    Masks: URLs, emails, dates, years, relative times, UI words, and numbers.
    Prevents spurious correlations with timestamps, years, and mobile chrome.

    Args:
        text: Input text (usually already chrome-stripped).
        mask_urls: Whether to replace URLs with '<URL>' token.

    Returns:
        Masked text with <URL>, <EMAIL>, <DATE>, <YEAR>, <TIME>, <UI>, <NUM>.
    """
    if not text:
        return ""

    masked = text

    if mask_urls:
        masked = _RE_URL.sub(" <URL> ", masked)

    masked = _RE_EMAIL.sub(" <EMAIL> ", masked)
    masked = _RE_DATE.sub(" <DATE> ", masked)
    masked = _RE_YEAR.sub(" <YEAR> ", masked)
    masked = _RE_TIME.sub(" <TIME> ", masked)
    masked = _RE_UI_WORDS.sub(" <UI> ", masked)
    masked = _RE_NUM.sub(" <NUM> ", masked)

    # Clean excessive whitespace
    masked = re.sub(r"\s+", " ", masked).strip()
    return masked
