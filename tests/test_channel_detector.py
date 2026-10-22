"""
Tests for the channel detection module.

Verifies that the rule-based channel detector correctly identifies
email, SMS, WhatsApp, and unknown message types from OCR text.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.channel_detector import detect_channel


class TestEmailDetection:
    def test_email_with_subject_and_from(self):
        text = "Subject: Important Update\nFrom: admin@example.com\nTo: user@example.com\n\nPlease review."
        result = detect_channel(text)
        assert result["channel"] == "email"
        assert result["confidence"] > 0

    def test_email_with_reply_and_forward(self):
        text = "Re: Your application\nFrom: hr@company.com\nTo: applicant@gmail.com\n\nForward this to your manager."
        result = detect_channel(text)
        assert result["channel"] == "email"

    def test_email_with_sent_from(self):
        text = "Hey, can you send me the file?\n\nSent from my iPhone"
        result = detect_channel(text)
        assert result["channel"] == "email"

    def test_email_with_unsubscribe(self):
        text = "Subject: Newsletter\nFrom: news@example.com\nTo: sub@test.com\n\nClick to unsubscribe from this mailing list."
        result = detect_channel(text)
        assert result["channel"] == "email"


class TestSMSDetection:
    def test_sms_with_text_message_header(self):
        text = "Text Message\n+15551234567\nToday 3:45 PM\n\nYour verification code is 123456."
        result = detect_channel(text)
        assert result["channel"] == "sms"
        assert result["confidence"] > 0

    def test_sms_with_sms_keyword(self):
        text = "SMS\n+447700900000\n2:30 PM\n\nYour package is on its way."
        result = detect_channel(text)
        assert result["channel"] == "sms"

    def test_sms_with_phone_and_time(self):
        text = "Text Message\n5551234567\nToday 10:15 AM\n\nReminder: Appointment tomorrow."
        result = detect_channel(text)
        assert result["channel"] == "sms"


class TestWhatsAppDetection:
    def test_whatsapp_explicit(self):
        text = "WhatsApp\n+919876543210\nonline\n\nHey, are you coming tonight?\n✓✓ Read"
        result = detect_channel(text)
        assert result["channel"] == "whatsapp"
        assert result["confidence"] > 0

    def test_whatsapp_with_encryption(self):
        text = "Messages are end-to-end encrypted. No one outside of this chat can read them.\n\nHello!\n✓✓ Delivered"
        result = detect_channel(text)
        assert result["channel"] == "whatsapp"

    def test_whatsapp_with_last_seen(self):
        text = "WhatsApp\n+1234567890\nlast seen today at 5:30 PM\n\nForwarded\nCheck this out!"
        result = detect_channel(text)
        assert result["channel"] == "whatsapp"


class TestUnknownDetection:
    def test_empty_text(self):
        result = detect_channel("")
        assert result["channel"] == "unknown"
        assert result["confidence"] == 0

    def test_generic_text(self):
        text = "Hello, how are you doing today? Nice weather we're having."
        result = detect_channel(text)
        assert result["channel"] == "unknown"

    def test_scores_returned(self):
        text = "Subject: Test\nFrom: test@test.com\nTo: user@test.com"
        result = detect_channel(text)
        assert "scores" in result
        assert "email" in result["scores"]
        assert "sms" in result["scores"]
        assert "whatsapp" in result["scores"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
