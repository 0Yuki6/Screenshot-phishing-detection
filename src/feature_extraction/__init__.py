"""
Feature extraction package for the phishing detection pipeline.

Re-exports the main entry points so consumers can do:
    from src.feature_extraction import extract_features, FEATURE_NAMES
"""

from src.feature_extraction.feature_extractor import extract_features, FEATURE_NAMES

__all__ = ["extract_features", "FEATURE_NAMES"]
