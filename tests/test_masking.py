"""
Unit tests for template chrome stripping and word-boundary token masking.
"""

import pytest
from src.masking import strip_template_chrome, mask_text


def test_strip_whatsapp_chrome():
    raw_whatsapp = (
        "10:24 AM\n"
        "@ Messages and calls are end-to-end encrypted. Only people in this chat can read, listen to, or share them.\n"
        "Hey, can you send the account details?\n"
        "✓✓ Read\n"
    )
    cleaned = strip_template_chrome(raw_whatsapp, channel="whatsapp")
    assert "end-to-end encrypted" not in cleaned
    assert "Only people in this chat" not in cleaned
    assert "10:24" not in cleaned
    assert "Read" not in cleaned
    assert "account details" in cleaned


def test_strip_email_chrome():
    raw_email = (
        "= M ail Q Search mail\n"
        "i Stared\n"
        " Snoozed\n"
        "Sent\n"
        "From: service@paypal.com\n"
        "To: user@example.com\n"
        "Subject: Notice of Account Restriction\n"
        "Dear Customer, your account is restricted. Click here to verify.\n"
        "To unsubscribe from this list click here.\n"
    )
    cleaned = strip_template_chrome(raw_email, channel="email")
    assert "Search mail" not in cleaned
    assert "Snoozed" not in cleaned
    assert "Subject:" not in cleaned
    assert "From:" not in cleaned
    assert "unsubscribe" not in cleaned.lower()
    assert "account is restricted" in cleaned


def test_strip_sms_chrome():
    raw_sms = (
        "8:52 PM all tte = ) 32%\n"
        "< (w) William H. mw OG :\n"
        "Mon, Sep 14\n"
        "Call me when free\n"
    )
    cleaned = strip_template_chrome(raw_sms, channel="sms")
    assert "all tte" not in cleaned
    assert "32%" not in cleaned
    assert "Sep 14" not in cleaned
    assert "Call me when free" in cleaned


def test_strip_corporate_boilerplate():
    raw = (
        "Project status update.\n"
        "Confidentiality Notice: This e-mail is intended only for the designated recipient.\n"
        "Forwarded by System Administrator for review.\n"
        "Please review the timeline attached."
    )
    cleaned = strip_template_chrome(raw)
    assert "confidentiality notice" not in cleaned.lower()
    assert "forwarded by" not in cleaned.lower()
    assert "Project status update" in cleaned
    assert "timeline attached" in cleaned


def test_mask_tokens():
    text = (
        "In 2002, John sent 500 dollars to john.doe@enron.com via https://secure-bank.com/login?id=48291. "
        "Meeting is on Oct 15th, 2024."
    )
    masked = mask_text(text, mask_urls=True)
    assert "2002" not in masked
    assert "2024" not in masked
    assert "500" not in masked
    assert "john.doe@enron.com" not in masked
    assert "https://secure-bank.com" not in masked
    assert "<URL>" in masked
    assert "<EMAIL>" in masked
    assert "<NUM>" in masked
    assert "<YEAR>" in masked or "<DATE>" in masked
