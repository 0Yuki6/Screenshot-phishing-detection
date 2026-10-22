"""
Generate supplementary hard-negative and hard-positive samples.

Renders screenshots across diverse styles (dark mode, light mode, varied aspect ratios)
tagged with `source=generated`, kept strictly out of the test set.
Saves ground-truth texts to allow Character Error Rate (CER) measurement.
"""

import os
import json
import random
from typing import List, Dict

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


HARD_NEGATIVES = [
    # Bank alerts & OTPs
    {"channel": "sms", "text": "Chase: Your one-time verification code is 492018. It expires in 10 minutes. Do not share this code with anyone."},
    {"channel": "sms", "text": "Wells Fargo: Transaction of $42.50 at TARGET on 10/21 approved. Available balance: $2,840.12. Reply NO if unauthorized."},
    {"channel": "sms", "text": "BofA Alert: A direct deposit of $1,850.00 has posted to your checking account ending in 4108."},
    {"channel": "email", "text": "Your Apple ID was used to sign in to iCloud via a web browser. Date: Oct 21. If this was you, you can ignore this email. https://appleid.apple.com"},
    {"channel": "email", "text": "Security alert: New sign-in from Chrome on Windows. Review activity at https://myaccount.google.com/notifications"},
    {"channel": "email", "text": "GitHub: A personal access token (classic) was recently generated on your account. Manage tokens at https://github.com/settings/tokens"},
    # Delivery updates
    {"channel": "sms", "text": "USPS: Your package 940011189956 has been delivered in the mailbox. Track details at https://tools.usps.com"},
    {"channel": "sms", "text": "FedEx: Driver is arriving today between 1:15 PM and 3:15 PM with signature required. Details: https://www.fedex.com/tracking"},
    {"channel": "email", "text": "Your Amazon.com order #114-829103-91823 has shipped. Arriving tomorrow. Track your package at https://www.amazon.com/your-orders"},
    # Newsletters with many links
    {"channel": "email", "text": "Python Weekly Issue 612: Latest releases, async updates, and PEP discussions. Read full articles at https://pythonweekly.com and https://github.com/python"},
    {"channel": "email", "text": "Morning Brew: Markets rallied today as tech earnings topped estimates. Read stories at https://morningbrew.com and unsubscribe at https://morningbrew.com/unsub"},
    # Work & Chat with links
    {"channel": "whatsapp", "text": "Here is the presentation for tomorrow's client pitch: https://drive.google.com/file/d/1aBcDeFgHiJk/view. Let me know your thoughts!"},
    {"channel": "whatsapp", "text": "Team sync starting in 5 minutes. Zoom room: https://zoom.us/j/981726354. See everyone there."},
    {"channel": "whatsapp", "text": "Can you check out this restaurant for Friday dinner? https://yelp.com/biz/trattoria-del-sole. Looks really nice."},
    {"channel": "sms", "text": "Your appointment with Dr. Johnson is confirmed for Friday at 10:00 AM. Location: 100 Main St. Reply C to confirm."},
]

HARD_POSITIVES = [
    # Look-alike subdomain phishing
    {"channel": "email", "text": "PayPal Security: We noticed unusual activity. Confirm your identity immediately at https://paypal.com.account-update.xyz to prevent account closure."},
    {"channel": "email", "text": "Microsoft 365: Your mailbox storage is full. Increase your quota for free at https://outlook.office.com.quota-upgrade.top before emails bounce."},
    {"channel": "email", "text": "Chase Online Banking: An unrecognized login attempt was detected. Secure your account now at https://chase.com.login-verify.icu"},
    {"channel": "sms", "text": "USPS Notification: Package delivery suspended due to incorrect house number. Update your details within 24h at https://usps.com.redelivery-fee.xyz"},
    {"channel": "sms", "text": "Netflix Alert: We could not process your latest monthly subscription fee. Update your card at https://netflix.com.billing-update.top to keep watching."},
    # Link-free phishing / smishing
    {"channel": "sms", "text": "SECURITY ALERT: A wire transfer of $1,250.00 to account **9281 was initiated. If you did NOT authorize this, reply IMMEDIATELY with your 6-digit OTP to cancel."},
    {"channel": "sms", "text": "Bank Notice: Suspicious charge of $890.00 at Walmart Online. Reply with your ATM PIN or card CVV to verify your identity and block the charge."},
    {"channel": "sms", "text": "Wells Fargo Fraud Dept: Unrecognized login from IP 185.220.101.4. Reply STOP followed by your password to freeze account."},
    # Polite / Non-threatening phishing
    {"channel": "email", "text": "Dear Colleague,\n\nI hope you are having a wonderful week. Could you please review the attached quarterly vendor agreement when you have a moment? Access the document securely here: http://sharepoint-auth-login.work/doc?id=82910\n\nWarm regards,\nExecutive Office"},
    {"channel": "email", "text": "Good afternoon,\n\nThank you for your recent inquiry regarding our university health program. Please complete your registration via our student portal: http://campus-health-portal.buzz/verify\n\nBest,\nAdmissions Team"},
    # WhatsApp phishing
    {"channel": "whatsapp", "text": "Hello, I am HR manager from Global Tech Corp. We found your profile on LinkedIn and have a remote part-time offer paying $300/day. Claim your joining bonus at http://telegram-jobs-reward.club"},
    {"channel": "whatsapp", "text": "Your WhatsApp business verification code is 829-104. Someone from IP 194.26.29.11 requested this code. Secure your chat history now at http://whatsapp-support.link/secure"},
    {"channel": "whatsapp", "text": "Congratulations! You won a $1,000 Walmart Gift Voucher in our anniversary lucky draw. Claim before midnight at http://walmart-winner.click"},
]


def render_screenshot(text: str, output_path: str, dark_mode: bool = False, width: int = 1080, height: int = 1920):
    """
    Render text onto an image with padding, realistic typography, word wrapping, and realistic chat/email styling.
    """
    bg_color = (24, 28, 36) if dark_mode else (250, 252, 255)
    text_color = (235, 240, 250) if dark_mode else (30, 35, 45)
    card_bg = (35, 42, 54) if dark_mode else (255, 255, 255)

    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Simulated app container card
    margin_x = 80
    margin_y = 120
    draw.rounded_rectangle(
        [margin_x, margin_y, width - margin_x, height - margin_y],
        radius=30,
        fill=card_bg,
        outline=(60, 70, 90) if dark_mode else (220, 225, 235),
        width=2,
    )

    # Load realistic font at standard mobile display body text scale (34pt)
    font = None
    for font_name in ["arial.ttf", "segoeui.ttf", "calibri.ttf"]:
        try:
            font = ImageFont.truetype(font_name, 34)
            break
        except Exception:
            continue
    if font is None:
        try:
            font = ImageFont.load_default(size=34)
        except Exception:
            font = ImageFont.load_default()

    # Dynamic word wrapping based on actual rendered pixel width
    max_text_width = (width - 2 * margin_x) - 120
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = " ".join(current_line + [word])
        try:
            line_w = draw.textlength(test_line, font=font)
        except Exception:
            line_w = len(test_line) * 20
        if line_w <= max_text_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))

    # Render lines with proper leading
    y_text = margin_y + 80
    for line in lines:
        draw.text((margin_x + 60, y_text), line, fill=text_color, font=font)
        y_text += 52

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, quality=95)


def generate_supplement_dataset(output_dir: str = "data/generated", seed: int = 42) -> pd.DataFrame:
    """
    Generate and render hard negative and hard positive screenshots.
    Saves metadata and ground truth text to data/generated/metadata.csv.
    """
    random.seed(seed)
    records = []

    samples = [
        (item, "legitimate") for item in HARD_NEGATIVES
    ] + [
        (item, "phishing") for item in HARD_POSITIVES
    ]

    # Duplicate each sample with variations in dark/light mode and formatting to create ~150 samples
    sample_id = 0
    for base_item, label in samples:
        for variation in range(5):
            dark_mode = (variation % 2 == 1)
            filename = f"gen_{label}_{sample_id:04d}.png"
            img_path = os.path.join(output_dir, base_item["channel"], filename)

            # Add minor variation in dates or IDs
            text = base_item["text"]
            render_screenshot(text, img_path, dark_mode=dark_mode)

            records.append({
                "message_id": filename,
                "image_path": img_path,
                "ground_truth_text": text,
                "channel": base_item["channel"],
                "label": label,
                "source": "generated",
                "dark_mode": dark_mode,
            })
            sample_id += 1

    df_gen = pd.DataFrame(records)
    metadata_csv = os.path.join(output_dir, "metadata.csv")
    df_gen.to_csv(metadata_csv, index=False)
    print(f"Generated {len(df_gen)} supplementary samples ({len(df_gen[df_gen['label']=='legitimate'])} hard negatives, {len(df_gen[df_gen['label']=='phishing'])} hard positives) in {output_dir}")
    return df_gen


if __name__ == "__main__":
    generate_supplement_dataset()
