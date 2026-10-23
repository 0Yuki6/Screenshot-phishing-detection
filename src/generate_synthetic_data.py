"""
Synthetic data generator for prototype pipeline validation.

Generates ~300 synthetic text samples (50 per channel × class bucket),
runs feature extraction on each, trains a placeholder Random Forest,
and saves everything to disk.

Usage:
    python -m src.generate_synthetic_data
"""

import os
import sys
import random
import logging
import json

import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extraction import extract_features, FEATURE_NAMES
from src.model import train_model, evaluate_model, save_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)


# ── Synthetic text templates ─────────────────────────────────────────

_PHISHING_EMAIL_TEMPLATES = [
    "Subject: Urgent! Your account has been suspended\nFrom: security@{fake_domain}\nTo: user@example.com\n\nDear Customer,\n\nWe have detected unauthorized activity on your account. Please verify your identity immediately by clicking the link below:\n\nhttp://{phish_url}\n\nFailure to verify within 24 hours will result in permanent account closure.\n\nRegards,\nSecurity Team",
    "Subject: Action Required - Verify Your Payment\nFrom: billing@{fake_domain}\nTo: customer@example.com\n\nDear Valued Customer,\n\nYour recent payment could not be processed. Please update your credit card information at:\n\nhttps://{phish_url}\n\nEnter your credentials to confirm payment. Your account will be locked if not resolved immediately.\n\nBilling Department",
    "Subject: Your password will expire soon\nFrom: admin@{fake_domain}\nTo: user@company.com\n\nDear User,\n\nYour password is set to expire. Click here to update your login credentials:\n\nhttp://{phish_url}\n\nThis is your final notice. Act now to avoid losing access.\n\nIT Support",
    "Subject: Invoice #INV-{num} Attached\nFrom: accounts@{fake_domain}\nTo: finance@example.com\n\nDear Sir/Madam,\n\nPlease review the attached invoice and confirm wire transfer payment to the account specified.\n\nAlternatively, verify your bank details at: http://{phish_url}\n\nThis is urgent and requires immediate attention.\n\nAccounts Payable",
    "Subject: Security Alert - Unusual Activity Detected\nFrom: noreply@{fake_domain}\nTo: valued.customer@example.com\n\nDear Account Holder,\n\nWe noticed unusual activity on your account from an unrecognized device. Your account has been temporarily restricted.\n\nConfirm your identity: http://{phish_url}\n\nSSN or OTP verification may be required. Failure to respond will result in account deactivation.\n\nSecurity Team",
    "Subject: Urgent: Verify 2FA to prevent account suspension\nFrom: support@{fake_domain}\nTo: user@example.com\n\nDear Customer,\n\nUnusual sign-in activity detected on your {service} account. Please verify your identity immediately.\n\nReply to this email with your 6-digit OTP and current password, or submit them at: http://{phish_url}\n\nFailure to verify will lead to immediate account termination.\n\nSecurity Department",
    "Subject: Refund Pending - Verification Code Required\nFrom: refunds@{fake_domain}\nTo: user@example.com\n\nDear Customer,\n\nA pending refund of INR {amount} could not be credited. Enter your one-time password and debit card PIN at http://{phish_url} to claim your refund.\n\nFinance Team",
    "Subject: Account Deactivation Notice\nFrom: accounts@{fake_domain}\nTo: user@example.com\n\nYour {service} account is scheduled for deactivation due to non-verification. To keep your account active, enter your login password and OTP here: http://{phish_url}\n\nCustomer Care",
]

_LEGIT_EMAIL_TEMPLATES = [
    "Subject: Your order has shipped!\nFrom: orders@amazon.com\nTo: customer@gmail.com\n\nHi there,\n\nGreat news! Your order #12345 has shipped and is on its way.\n\nTrack your package: https://www.amazon.com/track/12345\n\nExpected delivery: Monday, October 15\n\nThanks for shopping with us!\nAmazon Customer Service",
    "Subject: Weekly Newsletter - Tech Updates\nFrom: newsletter@techcrunch.com\nTo: subscriber@example.com\n\nHello,\n\nHere are this week's top stories:\n\n1. New AI breakthrough in language models\n2. Startup raises $50M for cloud infrastructure\n3. Review: Latest smartphone comparison\n\nRead more at https://techcrunch.com/weekly\n\nUnsubscribe: https://techcrunch.com/unsubscribe",
    "Subject: Meeting reminder - Team standup tomorrow\nFrom: calendar@google.com\nTo: team@company.com\n\nReminder: Team standup meeting\nWhen: Tomorrow at 10:00 AM\nWhere: Conference Room B / https://meet.google.com/abc-defg-hij\n\nAgenda:\n- Sprint review\n- Blockers discussion\n- Next sprint planning",
    "Subject: Your monthly statement is ready\nFrom: statements@chase.com\nTo: customer@gmail.com\n\nDear John,\n\nYour October statement is now available. Log in to your Chase account to view it:\n\nhttps://www.chase.com/statements\n\nTotal balance: $2,450.32\nPayment due: November 15\n\nThank you for being a valued Chase customer.",
    "Subject: Re: Project proposal feedback\nFrom: manager@company.com\nTo: employee@company.com\n\nHi Sarah,\n\nThanks for sending over the proposal. I've reviewed it and have a few suggestions:\n\n1. The timeline looks realistic\n2. Budget section needs more detail\n3. Let's discuss the staffing plan\n\nCan we schedule a call this week?\n\nBest,\nMike",
    "Subject: Your verification code for {service}\nFrom: no-reply@{service_domain}\nTo: user@example.com\n\nHi,\n\nYour one-time verification code is {otp}. This code is valid for 10 minutes.\n\nIf you did not request this code, you can safely ignore this email.\n\nThanks,\n{service} Security Team",
    "Subject: Security alert: New sign-in on {device}\nFrom: security@{service_domain}\nTo: user@example.com\n\nSecurity alert for your account. We noticed a new sign-in from a {device} in {city} at {time} PM.\n\nIf this was you, you do not need to take any action. If this was not you, review your active sessions in account settings.\n\n{service} Account Team",
    "Subject: Your order #{num} has been completed successfully\nFrom: support@{service_domain}\nTo: user@example.com\n\nThank you for your purchase! Your payment of INR {amount} has been received and your order #{num} is confirmed.\n\nDelivery details: Standard shipping to your registered address.\n\nCustomer Support",
    "Subject: You shared some Google Account data with {app_name}\nFrom: no-reply@accounts.google.com\nTo: user@gmail.com\n\nKeep track of your Google Account data. You recently granted {app_name} permission to access your basic profile information.\n\nYou can review or revoke third-party app permissions anytime in your Google Account security settings.",
    "Subject: [GitHub] {app_name} is requesting updated permissions\nFrom: notifications@github.com\nTo: developer@example.com\n\nThe GitHub App {app_name} is requesting additional permissions for your repositories.\n\nYou can review this request directly on GitHub: https://github.com/settings/installations",
    "Subject: Quarterly reminder about Terms of Service and Privacy Policy\nFrom: updates@{service_domain}\nTo: user@example.com\n\nDear User,\n\nThis is a routine reminder regarding our updated Terms of Service and Community Guidelines.\n\nNo action is required on your part. Thank you for being a valued member of our community.",
]

_PHISHING_SMS_TEMPLATES = [
    "Text Message\n+1{phone}\nToday {time} PM\n\nURGENT: Your bank account has been compromised. Verify your identity immediately at http://{phish_url} or your account will be suspended within 24 hours.",
    "SMS\n+44{phone}\n{time} AM\n\nYour package delivery failed. Confirm your address and pay the $1.99 fee at http://{phish_url}. Enter your card details to proceed. Act now!",
    "Text Message\n{phone}\nToday {time} PM\n\nAlert: Unusual login detected on your account. If this wasn't you, secure your account now: http://{phish_url}. Your password and OTP are required.",
    "SMS\n+1{phone}\n{time} AM\n\nCongratulations! You've won a $500 gift card. Claim your prize now at http://{phish_url}. Limited time offer - expires immediately!",
    "Text Message\n+91{phone}\nToday {time} PM\n\nYour OTP for bank transaction is 847291. If you did not request this, block your account at http://{phish_url}. Reply with your PIN to cancel.",
    "SMS\n+91{phone}\nToday {time} PM\n\nSBI Alert: Unrecognized transaction of Rs {amount} initiated on card. To block payment immediately, reply with the 6-digit OTP sent to your phone.",
    "SMS\n+91{phone}\nToday {time} PM\n\nDear Customer, your electricity power will be disconnected today. Update bill payment and enter debit card PIN at http://{phish_url} to avoid cut-off.",
    "SMS\n+91{phone}\nToday {time} PM\n\nDear User, your SIM KYC is expired. Visit http://{phish_url} and enter your mobile number and OTP within 2 hours to avoid service suspension.",
    "SMS\n+91{phone}\nToday {time} PM\n\nBank Notice: Suspicious charge of Rs {amount} on account. Reply STOP followed by your netbanking password to freeze card.",
]

_LEGIT_SMS_TEMPLATES = [
    "Text Message\n+1{phone}\nToday {time} PM\n\nYour Uber ride is arriving in 3 minutes. Driver: Mike in a white Toyota Camry. Track at https://uber.com/trip",
    "SMS\n+44{phone}\n{time} AM\n\nYour verification code is 482910. This code expires in 10 minutes. Do not share this code with anyone.",
    "Text Message\n{phone}\nToday {time} PM\n\nAppointment reminder: Dr. Smith on Tuesday at 2:30 PM. Reply YES to confirm or call (555) 123-4567 to reschedule.",
    "SMS\n+1{phone}\n{time} AM\n\nYour Amazon order has been delivered to your front door. Track at https://amazon.com/track/ABC123",
    "Text Message\n+1{phone}\nToday {time} PM\n\nHey! Are you still coming to dinner tonight? The reservation is at 7:30 at Olive Garden. Let me know!",
    "SMS\n+91{phone}\nToday {time} PM\n\nDear Customer, INR {amount} debited from A/c XX{num} on {date} at {merchant} via UPI Ref {ref}. Avail Bal: INR {balance}. -SBI",
    "SMS\n+91{phone}\nToday {time} PM\n\nDear SBI User, your A/c X{num} credited by INR {amount} on {date} transfer from {sender} Ref No {ref}. Check balance on YONO app.",
    "SMS\n+91{phone}\nToday {time} PM\n\nHello User, Your OTP for login is {otp}. Please do not share with anyone. Regards, OneStepLink.",
    "SMS\n+91{phone}\nToday {time} PM\n\nRecharge of INR {amount} is successful for your Mobile {num} on {date}. Current pack validity has been updated. -Airtel",
    "SMS\n+91{phone}\nToday {time} PM\n\nDear Student NL{num}, NSP offers easy renewal option for your scholarship for academic year 2026-27. Login at scholarships.gov.in.",
    "SMS\n+91{phone}\nToday {time} PM\n\nBiometric access for YONO app successfully enabled. Please ensure that your phone contains only your biometric credentials.",
]

_PHISHING_WHATSAPP_TEMPLATES = [
    "WhatsApp\n{phone}\nonline\n\nUrgent message from your bank: We've detected suspicious activity. Verify your account immediately at http://{phish_url}\n\nYour account will be blocked if you don't respond within 24 hours.\n\nForwarded\n✓✓",
    "WhatsApp\n{phone}\nlast seen today at {time}\n\nDear Customer, your WhatsApp account will expire soon. Renew now at http://{phish_url}\n\nEnter your payment card details to keep your account active. This is your final notice.\n\n✓✓ Delivered",
    "WhatsApp\nGroup: Family\n{phone}\n\n⚠️ WARNING: New government scheme - get Rs.5000 directly! Apply now: http://{phish_url}\n\nForwarded many times\n✓✓ Read",
    "WhatsApp\n{phone}\nonline\n\nYour OTP for login is 739201. Someone is trying to access your account. If this wasn't you, secure your account at http://{phish_url}\n\nEnd-to-end encrypted\n✓✓",
    "WhatsApp\n{phone}\nlast seen yesterday\n\nCongratulations! You've been selected for a special Bitcoin investment opportunity. Invest $100 and earn $10,000 in a week: http://{phish_url}\n\nForwarded\n✓✓ Delivered",
    "WhatsApp\n{phone}\nonline\n\nUrgent: Your account is being transferred to another device. Enter your 6-digit verification code here: http://{phish_url} or reply with the OTP to cancel.\n\nForwarded\n✓✓",
    "WhatsApp\n{phone}\nlast seen today at {time}\n\nCongratulations! You won Rs 25,00,000 in KBC Lucky Draw. To claim your prize, send your bank account number and UPI PIN to this number.\n\n✓✓ Delivered",
    "WhatsApp\n{phone}\nonline\n\nSecurity alert: WhatsApp Web session active. If you did not initiate this, reply with your 6-digit registration PIN immediately to terminate session.\n\n✓✓",
]

_LEGIT_WHATSAPP_TEMPLATES = [
    "WhatsApp\n{phone}\nonline\n\nHey! Can you pick up some groceries on your way home? We need milk, bread, and eggs. Thanks!\n\n✓✓ Read",
    "WhatsApp\nGroup: Work Team\n{phone}\n\nMeeting moved to 3 PM today. Same Zoom link as before: https://zoom.us/j/123456789\n\nSee you there!\n\n✓✓ Delivered",
    "WhatsApp\n{phone}\nlast seen today at {time}\n\nHappy birthday! 🎉🎂 Hope you have an amazing day! Let's celebrate this weekend.\n\nEnd-to-end encrypted\n✓✓ Read",
    "WhatsApp\n{phone}\nonline\n\nJust shared the photos from the trip on Google Drive: https://drive.google.com/folder/abc123\n\nLet me know which ones you like!\n\n✓✓",
    "WhatsApp\nGroup: Apartment\n{phone}\n\nReminder: Building maintenance this Saturday from 9 AM to 12 PM. Water will be shut off during this time.\n\n✓✓ Delivered",
    "WhatsApp\n{phone}\nonline\n\nYour verification code is {otp}. This code is valid for 10 minutes. Do not share this code with anyone.\n\nEnd-to-end encrypted\n✓✓ Read",
    "WhatsApp\n{phone}\nlast seen today at {time}\n\nPayment of INR {amount} to {merchant} was completed successfully. Transaction Ref: {ref}. Thank you!\n\n✓✓ Delivered",
    "WhatsApp\nGroup: Class 2026\n{phone}\n\nHey everyone, New note 2026-09-08.pdf uploaded to the shared drive. Let me know if you can view the notes.\n\n✓✓ Read",
    "WhatsApp\n{phone}\nonline\n\nTwo-step verification PIN reset was successful for your account. If you did not do this, review your settings.\n\n✓✓ Delivered",
    "WhatsApp\n{phone}\nlast seen yesterday\n\nHere is the order update: Your delivery of 1 item from {merchant} is scheduled for tomorrow by 4 PM.\n\n✓✓",
]


# ── Phishing URL components ──────────────────────────────────────────
_FAKE_DOMAINS = [
    "secure-bankverify.com", "accounts-update-center.com", "login-security.net",
    "paypa1-verify.com", "amaz0n-support.com", "micr0soft-security.net",
    "g00gle-alert.com", "apple-id-confirm.net", "chase-verify-account.com",
    "netfl1x-billing.com", "faceb00k-security.com", "dropb0x-share.net",
]

_PHISH_PATHS = [
    "verify-account?id=", "login/confirm?token=", "secure/update?ref=",
    "billing/payment?session=", "identity/check?user=", "password/reset?key=",
]


def _random_phone():
    return "".join([str(random.randint(0, 9)) for _ in range(10)])


def _random_time():
    h = random.randint(1, 12)
    m = random.randint(0, 59)
    return f"{h}:{m:02d}"


def _random_phish_url():
    domain = random.choice(_FAKE_DOMAINS)
    path = random.choice(_PHISH_PATHS)
    return f"{domain}/{path}{random.randint(10000, 99999)}"


def _random_num():
    return str(random.randint(1000, 9999))


_SERVICES = [
    ("Claude", "anthropic.com"),
    ("Steam", "steampowered.com"),
    ("Hugging Face", "huggingface.co"),
    ("NiceHash", "nicehash.com"),
    ("Radient", "radient.io"),
    ("Dab Arcade", "dabarcade.in"),
    ("Binance", "binance.com"),
    ("GitHub", "github.com"),
]

_MERCHANTS = ["Amazon", "Zentro Store", "Flipkart", "Swiggy", "EaseMyTrip", "Zomato", "Myntra"]
_SENDERS = ["Rajesh Kumar", "TEPUDI SOLE", "Anita Roy", "Manoj Singh", "TechCorp Ltd"]
_APPS = ["Cursor", "Cursor.sh", "Grammarly", "Figma", "Zoom", "Notion", "Slack"]
_DEVICES = ["Chrome on Windows", "iPhone 15", "Samsung Galaxy S24", "MacBook Pro", "Steam Mobile"]
_CITIES = ["New Delhi, IN", "Mumbai, IN", "Bangalore, IN", "Assam, IN", "Mizoram, IN"]
_DATES = ["19-Feb", "14-May", "10-Jun", "22-Jul", "18-Oct", "26-Sep", "12-May"]


def _fill_template(template: str, is_phishing: bool) -> str:
    """Fill a template with random values."""
    text = template
    text = text.replace("{phone}", _random_phone())
    text = text.replace("{time}", _random_time())
    text = text.replace("{num}", _random_num())
    text = text.replace("{otp}", f"{random.randint(100000, 999999)}")
    text = text.replace("{amount}", f"{random.choice([11, 49, 100, 199, 499, 786, 1250, 2450, 5000])}")
    text = text.replace("{balance}", f"{random.choice(['1,420.50', '3,850.00', '15,240.00', '780.00'])}")
    text = text.replace("{ref}", f"{random.randint(100000000000, 999999999999)}")
    text = text.replace("{date}", random.choice(_DATES))
    service, s_domain = random.choice(_SERVICES)
    text = text.replace("{service}", service)
    text = text.replace("{service_domain}", s_domain)
    text = text.replace("{merchant}", random.choice(_MERCHANTS))
    text = text.replace("{sender}", random.choice(_SENDERS))
    text = text.replace("{app_name}", random.choice(_APPS))
    text = text.replace("{device}", random.choice(_DEVICES))
    text = text.replace("{city}", random.choice(_CITIES))
    if is_phishing:
        text = text.replace("{phish_url}", _random_phish_url())
        text = text.replace("{fake_domain}", random.choice(_FAKE_DOMAINS))
    return text


def generate_samples(n_per_bucket: int = 50) -> pd.DataFrame:
    """
    Generate synthetic text samples for all 6 channel × class buckets.

    Args:
        n_per_bucket: Number of samples per bucket (3 channels × 2 classes = 6 buckets).

    Returns:
        DataFrame with columns: message_id, ocr_text, channel, label
    """
    records = []
    msg_id = 0

    template_map = {
        ("email", "phishing"): _PHISHING_EMAIL_TEMPLATES,
        ("email", "legitimate"): _LEGIT_EMAIL_TEMPLATES,
        ("sms", "phishing"): _PHISHING_SMS_TEMPLATES,
        ("sms", "legitimate"): _LEGIT_SMS_TEMPLATES,
        ("whatsapp", "phishing"): _PHISHING_WHATSAPP_TEMPLATES,
        ("whatsapp", "legitimate"): _LEGIT_WHATSAPP_TEMPLATES,
    }

    for (channel, label), templates in template_map.items():
        for i in range(n_per_bucket):
            template = templates[i % len(templates)]
            text = _fill_template(template, is_phishing=(label == "phishing"))
            records.append({
                "message_id": f"synth_{msg_id:04d}",
                "ocr_text": text,
                "channel": channel,
                "label": label,
            })
            msg_id += 1

    df = pd.DataFrame(records)
    logger.info("Generated %d synthetic samples (%d per bucket)", len(df), n_per_bucket)
    return df


def _extract_features_for_row(row: dict) -> list:
    """Helper for multiprocessing feature extraction."""
    feats = extract_features(row["ocr_text"], row["channel"], skip_whois=True)
    return [feats[name] for name in FEATURE_NAMES]


def build_feature_matrix(df: pd.DataFrame) -> tuple:
    """
    Extract features for all samples and build the full feature matrix.

    Returns:
        (X, y, feature_names, tfidf_vectorizer)
        Where X is a numpy array (scalar features + TF-IDF),
        y is a numpy array of binary labels (0=legitimate, 1=phishing),
        feature_names is the combined list of feature names,
        tfidf_vectorizer is the fitted TfidfVectorizer.
    """
    import concurrent.futures

    # Extract scalar features
    rows = df.to_dict('records')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        scalar_features = list(executor.map(_extract_features_for_row, rows))

    scalar_matrix = np.array(scalar_features, dtype=np.float64)

    # Build TF-IDF features
    tfidf = TfidfVectorizer(
        max_features=200,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    tfidf_matrix = tfidf.fit_transform(df["ocr_text"])

    # Combine scalar + TF-IDF
    X = hstack([csr_matrix(scalar_matrix), tfidf_matrix]).toarray()

    # Build combined feature name list
    tfidf_names = [f"tfidf_{name}" for name in tfidf.get_feature_names_out()]
    all_feature_names = FEATURE_NAMES + tfidf_names

    # Encode labels
    y = (df["label"] == "phishing").astype(int).values

    logger.info(
        "Feature matrix: %d samples × %d features (%d scalar + %d TF-IDF)",
        X.shape[0], X.shape[1], len(FEATURE_NAMES), len(tfidf_names),
    )
    return X, y, all_feature_names, tfidf


def main():
    """Generate synthetic data, train placeholder model, and save artifacts."""
    # Ensure output directories exist
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)

    # Step 1: Generate synthetic data
    logger.info("=== Step 1: Generating synthetic data ===")
    df = generate_samples(n_per_bucket=50)
    df.to_csv("data/processed/synthetic_samples.csv", index=False)
    logger.info("Saved synthetic samples to data/processed/synthetic_samples.csv")

    # Step 2: Build feature matrix
    logger.info("=== Step 2: Extracting features ===")
    X, y, feature_names, tfidf = build_feature_matrix(df)

    # Save features CSV
    feature_df = pd.DataFrame(X, columns=feature_names)
    feature_df["label"] = y
    feature_df["channel_name"] = df["channel"].values
    feature_df.to_csv("data/processed/features.csv", index=False)
    logger.info("Saved feature matrix to data/processed/features.csv")

    # Step 3: Train/test split and train model
    logger.info("=== Step 3: Training placeholder model ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    model = train_model(X_train, y_train)

    # Step 4: Evaluate
    logger.info("=== Step 4: Evaluating model ===")
    results = evaluate_model(model, X_test, y_test, label_names=["legitimate", "phishing"])
    print("\n" + results["classification_report"])
    print(f"Confusion Matrix:\n{results['confusion_matrix']}\n")

    # Step 5: Save model and TF-IDF vectorizer
    logger.info("=== Step 5: Saving artifacts ===")
    save_model(model, "data/models/model.pkl")
    joblib.dump(tfidf, "data/models/tfidf_vectorizer.pkl")
    logger.info("Saved TF-IDF vectorizer to data/models/tfidf_vectorizer.pkl")

    # Save feature names for inference
    with open("data/models/feature_names.json", "w") as f:
        json.dump(feature_names, f)
    logger.info("Saved feature names to data/models/feature_names.json")

    # Step 6: Permutation importance (top 15)
    logger.info("=== Step 6: Feature importance ===")
    importance = get_permutation_importance(model, X_test, y_test, feature_names)
    print("\nTop 15 Features by Permutation Importance:")
    for name, mean, std in importance[:15]:
        print(f"  {name:40s} {mean:+.4f} ± {std:.4f}")

    logger.info("\n✓ Pipeline complete. Artifacts saved to data/models/")


# Import here to avoid circular imports at module level
from src.model import get_permutation_importance


if __name__ == "__main__":
    main()
