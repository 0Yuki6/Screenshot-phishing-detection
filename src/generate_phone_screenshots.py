"""
Render realistic phone-style (1080x2400) screenshots for training augmentation.

Generates both light-mode and dark-mode screenshots for both legitimate and phishing classes
across email, SMS, and WhatsApp channels with simulated phone UI chrome (status bar, app bar, navigation bar).
OCRs each image with the production pipeline (including geometric cropping).
Strictly kept separate from real holdout data.
"""

import os
import random
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.ocr import run_dual_variant_ocr

# Set deterministic random seed
random.seed(42)

PHONE_SAMPLES = [
    # ── EMAIL LEGITIMATE ──
    {"channel": "email", "label": "legitimate", "sender": "Google Security <no-reply@accounts.google.com>", "subject": "Security alert: New sign-in on Windows", "text": "Security alert for user. New sign-in detected on Chrome on Windows. If this was you, you can safely ignore this email. Review your devices at https://myaccount.google.com/notifications"},
    {"channel": "email", "label": "legitimate", "sender": "GitHub <notifications@github.com>", "subject": "[GitHub] Personal access token generated", "text": "A new personal access token (classic) was recently generated on your GitHub account. If you did not generate this token, revoke it immediately at https://github.com/settings/tokens"},
    {"channel": "email", "label": "legitimate", "sender": "Amazon.in <auto-confirm@amazon.in>", "subject": "Shipped: Order #402-91823-192", "text": "Your package has shipped and is on its way. Estimated delivery: tomorrow by 8 PM. Track your package at https://www.amazon.in/your-orders"},
    {"channel": "email", "label": "legitimate", "sender": "HDFC Bank Alerts <alerts@hdfcbank.net>", "subject": "E-Statement for Account ending 4108", "text": "Dear Customer, Your monthly bank account statement for September is now available for download. Log in to NetBanking at https://netbanking.hdfcbank.com to view transactions."},

    # ── EMAIL PHISHING ──
    {"channel": "email", "label": "phishing", "sender": "PayPal Support <security@paypal-notice-alert.com>", "subject": "Urgent: Your account has been limited", "text": "Dear Customer, We detected unauthorized access to your PayPal wallet. Your account has been temporarily restricted. Verify your identity immediately at http://paypal.com.account-update.xyz to avoid permanent suspension."},
    {"channel": "email", "label": "phishing", "sender": "Microsoft 365 Admin <support@office365-upgrade.top>", "subject": "Mailbox storage quota exceeded (99%)", "text": "Your mailbox has reached 99.4% capacity. Incoming emails will be rejected. Upgrade your email quota for free within 24 hours at http://outlook.office365.storage-upgrade.top"},
    {"channel": "email", "label": "phishing", "sender": "Chase Online <alerts@chase-banking-support.icu>", "subject": "Action Required: Suspicious debit attempt", "text": "A debit transaction of $849.00 at Apple Store was declined. If you did not authorize this transaction, restore and secure your online banking access at http://chase-online.verify-login.icu"},
    {"channel": "email", "label": "phishing", "sender": "Netflix Billing <info@netflix-renewal-pay.click>", "subject": "Notice: Your membership has been suspended", "text": "We were unable to validate your billing details for the next subscription cycle. Please update your payment card information immediately at http://netflix.billing-verify.click to continue watching."},

    # ── SMS LEGITIMATE ──
    {"channel": "sms", "label": "legitimate", "sender": "AD-SBIUPI-S", "subject": "SBI UPI Debit", "text": "Dear UPI user A/C X7838 debited by Rs 250.00 on 21Oct26 transfer to SWIGGY Ref No 605030463660. If not done by you, call 1800111109 or SMS BLOCK to 567676."},
    {"channel": "sms", "label": "legitimate", "sender": "AX-AIRTEL-P", "subject": "Airtel Recharge", "text": "Recharge of INR 299.00 successful for Airtel mobile. Unlimited calls + 1.5GB/day valid for 28 days. Check data balance and OTT coupons on Airtel Thanks App."},
    {"channel": "sms", "label": "legitimate", "sender": "VM-HDFCBK-S", "subject": "HDFC OTP", "text": "Your NetBanking login OTP is 849201. Valid for 5 minutes. Never share your OTP, UPI PIN, or password with bank executives or callers."},
    {"channel": "sms", "label": "legitimate", "sender": "JM-BLDRT-S", "subject": "BlueDart Delivery", "text": "Shipment 94810291 from Amazon is out for delivery today. Delivery associate will arrive by 5 PM. Track live at https://www.bluedart.com"},

    # ── SMS PHISHING ──
    {"channel": "sms", "label": "phishing", "sender": "AD-SBIBNK-P", "subject": "YONO KYC Suspended", "text": "Dear Customer, Your SBI YONO account will be blocked today due to pending KYC. Update your PAN card and Aadhaar immediately at http://sbi-yono.kyc-update.xyz to prevent account deactivation."},
    {"channel": "sms", "label": "phishing", "sender": "BIJLI-ALERT", "subject": "Power Cut Notice", "text": "Electricity Dept Alert: Your power supply will be disconnected tonight at 9:30 PM due to unpaid bill. Pay immediately or update details at http://bijli-bill.online"},
    {"channel": "sms", "label": "phishing", "sender": "KB-LOTTRY", "subject": "Prize Winner", "text": "Congratulations! Your mobile number won cash prize of Rs 25,00,000 in KBC Lucky Draw 2026. Claim your prize money before midnight at http://kbc-lottery-winner.buzz"},
    {"channel": "sms", "label": "phishing", "sender": "IN-POST-S", "subject": "Delivery Failed", "text": "IndiaPost Notice: Package delivery failed due to incorrect house number. Confirm address and pay Rs 15 re-delivery fee at http://indiapost.redelivery-fee.top"},

    # ── WHATSAPP LEGITIMATE ──
    {"channel": "whatsapp", "label": "legitimate", "sender": "Alex Project Lead", "subject": "Project Presentation", "text": "Hey team, sharing the revised slide deck for tomorrow's client review: https://docs.google.com/presentation/d/1XyZ987/edit. Let me know if any numbers need updating."},
    {"channel": "whatsapp", "label": "legitimate", "sender": "Priya", "subject": "Dinner Plan", "text": "Hey! Are we still on for dinner this Friday? Thinking of trying that Italian place near downtown. Let me know what time works best for you."},
    {"channel": "whatsapp", "label": "legitimate", "sender": "College Study Group", "subject": "Lecture Notes", "text": "Does anyone have the notes from today's distributed systems lecture? The audio was cutting out during the second half."},
    {"channel": "whatsapp", "label": "legitimate", "sender": "Mom", "subject": "Safe Arrival", "text": "Just landed and picked up the luggage. Traffic looks light so should be home in about 45 minutes. See you soon!"},

    # ── WHATSAPP PHISHING ──
    {"channel": "whatsapp", "label": "phishing", "sender": "Amazon Recruitment HR", "subject": "Part-Time Job Offer", "text": "Hello! I am Global Talent Recruiter from Amazon Online. We have remote part-time positions reviewing products. Work 30 mins/day, earn Rs 3,000 - 8,000 daily. Contact our agent at http://amazon-jobs-hr.club"},
    {"channel": "whatsapp", "label": "phishing", "sender": "Jio Festive Promo", "subject": "Free 5G Recharge", "text": "Festive Special Offer: Reliance Jio is gifting free 3-month unlimited 5G recharge to celebrate 5G rollout! Claim your voucher before midnight at http://jio-free-recharge.site"},
    {"channel": "whatsapp", "label": "phishing", "sender": "Crypto VIP Signals", "subject": "Investment Return", "text": "Join our VIP crypto trading channel with 98% guaranteed monthly profit. New members receive 150 USDT welcome deposit. Register your wallet at http://crypto-wealth-fund.xyz"},
    {"channel": "whatsapp", "label": "phishing", "sender": "WhatsApp Security", "subject": "Account Termination Warning", "text": "WhatsApp Official Notice: Your account will be deleted within 12 hours due to unauthorized logins. Verify your phone number immediately at http://whatsapp-helpdesk.link/verify"},
]


def render_phone_screenshot(
    sample: Dict[str, Any],
    output_path: str,
    dark_mode: bool = False,
    width: int = 1080,
    height: int = 2400,
) -> None:
    """
    Render a realistic full-screen phone screenshot (1080x2400) with simulated
    OS status bar, app container, and bottom navigation bar.
    """
    bg_color = (18, 18, 20) if dark_mode else (245, 246, 248)
    card_bg = (30, 32, 38) if dark_mode else (255, 255, 255)
    text_color = (235, 238, 245) if dark_mode else (25, 30, 40)
    subtext_color = (150, 160, 175) if dark_mode else (100, 110, 125)
    header_bg = (24, 26, 32) if dark_mode else (255, 255, 255)
    border_color = (50, 55, 65) if dark_mode else (225, 230, 240)

    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # 1. Status Bar (top 108px - 4.5% of height)
    draw.rectangle([0, 0, width, 108], fill=header_bg)
    font_status = None
    for fn in ["arial.ttf", "segoeui.ttf"]:
        try:
            font_status = ImageFont.truetype(fn, 32)
            break
        except Exception:
            continue
    if font_status is None:
        font_status = ImageFont.load_default()

    draw.text((60, 36), "9:45", fill=text_color, font=font_status)
    draw.text((width - 240, 36), "5G  98%", fill=text_color, font=font_status)

    # 2. App Header (y: 108 -> 260)
    draw.rectangle([0, 108, width, 260], fill=header_bg)
    draw.line([0, 260, width, 260], fill=border_color, width=2)

    font_header = None
    for fn in ["arialbd.ttf", "segoeuib.ttf", "arial.ttf"]:
        try:
            font_header = ImageFont.truetype(fn, 38)
            break
        except Exception:
            continue
    if font_header is None:
        font_header = font_status

    channel = sample["channel"]
    if channel == "email":
        draw.text((60, 140), "<  Inbox", fill=text_color, font=font_header)
        draw.text((60, 195), sample.get("sender", "")[:50], fill=subtext_color, font=font_status)
    elif channel == "sms":
        draw.text((60, 140), f"<  {sample.get('sender', 'Messages')}", fill=text_color, font=font_header)
        draw.text((60, 195), "SMS · Today", fill=subtext_color, font=font_status)
    else:
        draw.text((60, 140), f"<  {sample.get('sender', 'Chat')}", fill=text_color, font=font_header)
        draw.text((60, 195), "online", fill=(74, 222, 128) if dark_mode else (22, 163, 74), font=font_status)

    # 3. Message Body Card (y: 300 -> 2100)
    card_x1, card_y1 = 60, 300
    card_x2, card_y2 = width - 60, 2150
    draw.rounded_rectangle([card_x1, card_y1, card_x2, card_y2], radius=24, fill=card_bg, outline=border_color, width=2)

    # Subject line (for email)
    curr_y = card_y1 + 50
    if channel == "email" and "subject" in sample:
        draw.text((card_x1 + 50, curr_y), sample["subject"], fill=text_color, font=font_header)
        curr_y += 70
        draw.line([card_x1 + 50, curr_y, card_x2 - 50, curr_y], fill=border_color, width=1)
        curr_y += 40

    # Body text wrapping
    font_body = None
    for fn in ["arial.ttf", "segoeui.ttf"]:
        try:
            font_body = ImageFont.truetype(fn, 34)
            break
        except Exception:
            continue
    if font_body is None:
        font_body = font_status

    words = sample["text"].split()
    lines = []
    curr_line = []
    max_line_width = (card_x2 - card_x1) - 100

    for w in words:
        test_line = " ".join(curr_line + [w])
        try:
            line_w = draw.textlength(test_line, font=font_body)
        except Exception:
            line_w = len(test_line) * 20
        if line_w <= max_line_width:
            curr_line.append(w)
        else:
            if curr_line:
                lines.append(" ".join(curr_line))
            curr_line = [w]
    if curr_line:
        lines.append(" ".join(curr_line))

    for line in lines:
        draw.text((card_x1 + 50, curr_y), line, fill=text_color, font=font_body)
        curr_y += 54

    # 4. Navigation Bar (bottom 108px - y: 2292 -> 2400)
    draw.rectangle([0, height - 108, width, height], fill=header_bg)
    # Draw home gesture bar
    pill_w = 280
    pill_h = 10
    pill_x1 = (width - pill_w) // 2
    pill_y1 = height - 54
    draw.rounded_rectangle([pill_x1, pill_y1, pill_x1 + pill_w, pill_y1 + pill_h], radius=5, fill=(120, 130, 145))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, quality=95)


def _ocr_single_task(task_data):
    import cv2
    img_bgr = cv2.imread(task_data["img_path"])
    ocr_res = run_dual_variant_ocr(img_bgr)
    return {
        "message_id": task_data["filename"],
        "channel": task_data["channel"],
        "label": task_data["label"],
        "ocr_text": ocr_res["text"],
        "confidence": ocr_res["confidence"],
        "winning_variant": ocr_res["winning_variant"],
        "phone_cropped": ocr_res.get("phone_cropped", True),
        "dark_mode": task_data["dark_mode"],
        "split": "train",
        "source": "phone_screenshot",
    }


def generate_and_ocr_phone_samples(
    output_img_dir: str = "data/phone_screenshots",
    output_csv_path: str = "data/processed/phone_training_samples.csv",
    variations_per_sample: int = 5,
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    Render phone-style screenshots in both dark and light modes, OCR them with
    the production pipeline (geometric crop included), and save metadata.
    """
    from concurrent.futures import ThreadPoolExecutor

    render_tasks = []
    sample_idx = 0

    print(f"1. Rendering phone-style screenshots ({len(PHONE_SAMPLES)} base templates x {variations_per_sample} variations)...")

    for base in PHONE_SAMPLES:
        for var_idx in range(variations_per_sample):
            dark_mode = (var_idx % 2 == 1)
            mode_str = "dark" if dark_mode else "light"
            filename = f"phone_{base['channel']}_{base['label']}_{sample_idx:04d}_{mode_str}.png"
            img_path = os.path.join(output_img_dir, base["channel"], base["label"], filename)

            render_phone_screenshot(base, img_path, dark_mode=dark_mode)

            render_tasks.append({
                "img_path": img_path,
                "filename": filename,
                "channel": base["channel"],
                "label": base["label"],
                "dark_mode": dark_mode,
            })
            sample_idx += 1

    print(f"2. Running parallel OCR on {len(render_tasks)} phone screenshots ({max_workers} workers)...")
    records = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(_ocr_single_task, render_tasks):
            records.append(res)

    df_phone = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_phone.to_csv(output_csv_path, index=False)
    print(f"Rendered and OCR'd {len(df_phone)} phone screenshots -> {output_csv_path}")
    return df_phone


if __name__ == "__main__":
    generate_and_ocr_phone_samples()
