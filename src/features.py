"""
Unified feature engineering module for phishing detection.

Adheres to repository layout (src/features.py).
Provides extract_features, FEATURE_NAMES, and split-safe feature matrix building.
Guarantees 100% deterministic, offline execution by default.
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.feature_extraction import extract_features, FEATURE_NAMES
from src.masking import strip_template_chrome, mask_text


def extract_features_single(
    text: str,
    channel: str,
    allow_network: bool = False,
    skip_whois: bool = True,
) -> Dict[str, Union[int, float]]:
    """
    Extract scalar feature dictionary from OCR text.
    Offline-safe and deterministic by default.
    """
    return extract_features(text, channel, allow_network=allow_network, skip_whois=skip_whois)


def build_feature_matrix(
    df: pd.DataFrame,
    tfidf_vectorizer: Optional[TfidfVectorizer] = None,
    fit_tfidf: bool = False,
    include_tfidf: bool = True,
    max_tfidf_features: int = 300,
    text_column: str = "clean_text",
    use_char_ngrams: bool = True,
    min_df: int = 10,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str], Optional[TfidfVectorizer]]:
    """
    Build the full feature matrix combining scalar features and optional TF-IDF n-grams.
    
    Guarantees:
    - Zero data leakage (vectorizer strictly fitted on training split when fit_tfidf=True).
    - Tokens are masked with standard semantic placeholders (<URL>, <EMAIL>, <DATE>, <YEAR>, <TIME>, <UI>, <NUM>)
      before vectorization, preventing models from learning timestamp or date artifacts.
    - Uses character n-grams (analyzer='char_wb', ngram_range=(3,5)) with higher min_df to eliminate
      noise and typo vulnerability.
    - include_tfidf toggle enables engineered-only feature comparison.

    Args:
        df: DataFrame containing at least 'channel' and text_column (or 'ocr_text').
        tfidf_vectorizer: Pre-fitted vectorizer (required if fit_tfidf=False and include_tfidf=True).
        fit_tfidf: True to fit a new vectorizer (training split only).
        include_tfidf: Whether to append TF-IDF features to scalar features.
        max_tfidf_features: Maximum number of n-gram features (default 300).
        text_column: Column name to extract features from ('clean_text' or 'masked_text').
        use_char_ngrams: Use character n-grams inside word boundaries (default True).
        min_df: Minimum document frequency threshold (default 10).

    Returns:
        (X, y, all_feature_names, tfidf_vectorizer)
    """
    # Fallback column resolution
    if text_column not in df.columns:
        if "ocr_text" in df.columns:
            clean_series = df["ocr_text"].fillna("").apply(strip_template_chrome)
        else:
            raise KeyError(f"Neither '{text_column}' nor 'ocr_text' found in DataFrame columns")
    else:
        clean_series = df[text_column].fillna("").apply(strip_template_chrome)

    # Always mask text before NLP vectorization to hide timestamps, years, and mobile chrome
    masked_series = clean_series.apply(mask_text)

    # Extract scalar features row by row
    scalar_rows = []
    for idx, row in df.iterrows():
        t = str(clean_series.iloc[len(scalar_rows)])
        ch = str(row.get("channel", "email"))
        feats = extract_features(t, ch, allow_network=False, skip_whois=True)
        scalar_rows.append([feats[name] for name in FEATURE_NAMES])

    scalar_matrix = np.array(scalar_rows, dtype=np.float64)

    # TF-IDF N-grams (if enabled)
    tfidf_matrix = None
    if include_tfidf:
        if fit_tfidf:
            n_docs = len(masked_series)
            effective_min_df = min(min_df, max(1, n_docs // 10))
            effective_max_df = 1.0 if n_docs < 20 else 0.90

            if use_char_ngrams:
                tfidf_vectorizer = TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=max_tfidf_features,
                    min_df=effective_min_df,
                    max_df=effective_max_df,
                    sublinear_tf=True,
                )
            else:
                tfidf_vectorizer = TfidfVectorizer(
                    max_features=max_tfidf_features,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=effective_min_df,
                    max_df=effective_max_df,
                )
            try:
                tfidf_matrix = tfidf_vectorizer.fit_transform(masked_series)
            except ValueError:
                tfidf_vectorizer = TfidfVectorizer(
                    max_features=max_tfidf_features,
                    min_df=1,
                    max_df=1.0,
                )
                tfidf_matrix = tfidf_vectorizer.fit_transform(masked_series)
        elif tfidf_vectorizer is not None:
            tfidf_matrix = tfidf_vectorizer.transform(masked_series)

    if tfidf_matrix is not None:
        X = hstack([csr_matrix(scalar_matrix), tfidf_matrix]).toarray()
        tfidf_names = [f"tfidf_{name}" for name in tfidf_vectorizer.get_feature_names_out()]
        all_feature_names = FEATURE_NAMES + tfidf_names
    else:
        X = scalar_matrix
        all_feature_names = list(FEATURE_NAMES)
        tfidf_vectorizer = None

    # Encode label if present
    y = None
    if "label" in df.columns:
        y = (df["label"] == "phishing").astype(int).values

    return X, y, all_feature_names, tfidf_vectorizer


__all__ = [
    "extract_features",
    "extract_features_single",
    "build_feature_matrix",
    "FEATURE_NAMES",
]
