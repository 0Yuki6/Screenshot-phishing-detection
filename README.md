# Phishing Detection System

Screenshot-based phishing detection using OCR, feature extraction, and machine learning. Upload a screenshot of an email, SMS, or WhatsApp message and get a phishing/legitimate verdict with confidence score.

## Architecture

```
Image Upload → Preprocessing → OCR → Channel Detection → Feature Extraction → Random Forest → Verdict
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Tesseract OCR (required for image-based detection)

- **Windows**: Download from [UB-Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki), install, and add to PATH
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

> **Note**: Tesseract is only needed for processing actual screenshots. You can test the pipeline without it using the `bypass_ocr` flag (see below).

### 3. Generate Synthetic Data & Train Placeholder Model

```bash
python -m src.generate_synthetic_data
```

This generates ~300 synthetic samples, extracts features, trains a Random Forest, and saves:
- `data/processed/synthetic_samples.csv` — synthetic text samples
- `data/processed/features.csv` — extracted feature matrix
- `data/models/model.pkl` — trained model
- `data/models/tfidf_vectorizer.pkl` — fitted TF-IDF vectorizer
- `data/models/feature_names.json` — feature name ordering

### 4. Start the API Server

```bash
uvicorn src.main:app --reload
```

The API is now available at `http://localhost:8000`. Swagger UI at `http://localhost:8000/docs`.

### 5. Test Predictions

**With an image** (requires Tesseract):
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@screenshot.png"
```

**Without Tesseract** (bypass OCR, send raw text):
```bash
curl -X POST "http://localhost:8000/predict?bypass_ocr=true" \
  -F "text=Subject: Urgent! Your account has been suspended. Click http://bit.ly/verify-now to restore access."
```

**Health check**:
```bash
curl http://localhost:8000/health
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Upload screenshot or text → phishing verdict |
| `/health` | GET | Service health check |
| `/feedback` | POST | Submit label correction for retraining |
| `/docs` | GET | Interactive Swagger UI |

### `/predict` Response

```json
{
  "prediction_id": "uuid-string",
  "channel": "email",
  "label": "phishing",
  "confidence": 0.9234,
  "flagged_features": ["has_url_shortener=1", "urgent_keyword_count=5"],
  "channel_scores": {"email": 8, "sms": 0, "whatsapp": 0}
}
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
phishing-detector/
├── data/
│   ├── raw/                    # Screenshots organized by channel
│   │   ├── email/
│   │   ├── sms/
│   │   └── whatsapp/
│   ├── processed/              # Extracted feature CSVs
│   └── models/                 # Saved model artifacts (.pkl)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Image preprocessing pipeline
│   ├── ocr.py                  # Tesseract OCR wrapper
│   ├── channel_detector.py     # Rule-based channel detection
│   ├── feature_extraction/
│   │   ├── __init__.py
│   │   ├── feature_extractor.py  # Main feature orchestrator
│   │   ├── url_utils.py        # URL parsing and analysis
│   │   ├── domain_utils.py     # WHOIS, typosquatting, whitelist/blacklist
│   │   ├── text_utils.py       # Keyword counting and text analysis
│   │   └── keyword_lists.py    # Centralized keyword/domain config
│   ├── model.py                # Training, evaluation, persistence
│   ├── generate_synthetic_data.py  # Synthetic data + placeholder model
│   └── main.py                 # FastAPI service
├── tests/
│   ├── test_channel_detector.py
│   ├── test_feature_extraction.py
│   └── test_model.py
├── notebooks/                  # Exploratory analysis
├── requirements.txt
└── README.md
```

## Feature List

| Category | Feature | Type |
|---|---|---|
| Channel | `channel` | categorical (encoded) |
| URL | `url_length`, `subdomain_count`, `special_char_count` | numeric |
| URL | `has_raw_ip`, `has_https`, `has_url_shortener` | binary |
| Domain | `domain_age_days`, `typosquatting_score` | numeric |
| Domain | `blacklist_status`, `whitelist_match` | binary |
| Text | `total_word_count`, `urgent_keyword_count`, `login_payment_keyword_count` | numeric |
| Text | `urgent_keyword_ratio`, `login_payment_keyword_ratio` | numeric |
| Text | `generic_greeting_flag` | binary |
| Text | TF-IDF vector (200 features) | sparse |

## Design Decisions

- **Random Forest** over CNN: no GPU dependency, stable with correlated TF-IDF features
- **Single unified model** with `channel` as a feature: cross-channel pattern learning with less data
- **Automatic channel detection**: rule-based, no user input needed
- **Brand impersonation** via Levenshtein distance, not image-based logo matching
- **WHOIS lookups** are slow — skipped during bulk processing with `skip_whois=True`
- **Blacklist API** is a placeholder interface — replace with Google Safe Browsing / PhishTank when ready
