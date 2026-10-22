"""
Dataset preparation, template parsing, group clustering (group_id), and split generation.

Guarantees zero data leakage by grouping near-duplicate texts (MinHash/TF-IDF cosine >= 0.85)
and keeping groups strictly within a single partition.
"""

import os
import re
import json
import random
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

from src.masking import strip_template_chrome, mask_text


def extract_template_label(message_id: str, channel: str) -> str:
    """
    Extract template name from filename or channel.
    Email encodes 4 client templates: 'gmail', 'outlook', 'yahoo', 'applemail'.
    SMS and WhatsApp have exactly ONE template each.
    """
    if channel == "email":
        msg_lower = message_id.lower()
        for template in ["gmail", "outlook", "yahoo", "applemail"]:
            if f"_{template}." in msg_lower:
                return template
        return "outlook"  # default fallback
    elif channel == "sms":
        return "standard_sms"
    elif channel == "whatsapp":
        return "standard_whatsapp"
    return "unknown"


class UnionFind:
    """Disjoint Set Union (Union-Find) with path compression."""
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        path = []
        while self.parent[i] != i:
            path.append(i)
            i = self.parent[i]
        for node in path:
            self.parent[node] = i
        return i

    def union(self, i: int, j: int):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j


def compute_group_ids(texts: List[str], similarity_threshold: float = 0.85, batch_size: int = 500) -> List[int]:
    """
    Cluster texts into connected components where pairwise cosine similarity >= threshold.
    
    Args:
        texts: List of normalized/masked strings.
        similarity_threshold: Minimum cosine similarity to merge two texts into the same group.
        batch_size: Block size for sparse matrix multiplication.
        
    Returns:
        List of integer group IDs corresponding to each input text.
    """
    n = len(texts)
    if n == 0:
        return []

    # Vectorize with character 3-5 grams for typo/noise-resilient text similarity
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    X = tfidf.fit_transform(texts)

    uf = UnionFind(n)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_sim = X[start:end].dot(X.T).tocsr()

        for i_local in range(end - start):
            i_global = start + i_local
            row = batch_sim[i_local]
            for j_global, sim in zip(row.indices, row.data):
                if j_global > i_global and sim >= similarity_threshold:
                    uf.union(i_global, j_global)

    raw_groups = [uf.find(i) for i in range(n)]

    # Renumber sequentially
    group_map: Dict[int, int] = {}
    renumbered: List[int] = []
    for g in raw_groups:
        if g not in group_map:
            group_map[g] = len(group_map)
        renumbered.append(group_map[g])

    return renumbered


def create_grouped_splits(
    df: pd.DataFrame,
    seed: int = 42,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
) -> Dict[str, List[int]]:
    """
    Partition dataframe indices into train, val, and test splits without breaking groups.
    Stratifies primarily by (channel, label) bucket at the group level.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Map each group to its dominant channel and label
    group_metadata: Dict[int, Tuple[str, str, List[int]]] = {}
    for idx, row in df.iterrows():
        g = row["group_id"]
        ch = row["channel"]
        lbl = row["label"]
        if g not in group_metadata:
            group_metadata[g] = (ch, lbl, [])
        group_metadata[g][2].append(idx)

    # Bucket groups by (channel, label)
    buckets: Dict[Tuple[str, str], List[int]] = {}
    for g, (ch, lbl, indices) in group_metadata.items():
        key = (ch, lbl)
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(g)

    train_indices = []
    val_indices = []
    test_indices = []

    for key, groups in buckets.items():
        random.shuffle(groups)
        n_groups = len(groups)
        n_test = max(1, int(n_groups * test_ratio))
        n_val = max(1, int(n_groups * val_ratio))

        test_groups = set(groups[:n_test])
        val_groups = set(groups[n_test : n_test + n_val])
        train_groups = set(groups[n_test + n_val :])

        for g in test_groups:
            test_indices.extend(group_metadata[g][2])
        for g in val_groups:
            val_indices.extend(group_metadata[g][2])
        for g in train_groups:
            train_indices.extend(group_metadata[g][2])

    return {
        "train": sorted(train_indices),
        "val": sorted(val_indices),
        "test": sorted(test_indices),
    }


def prepare_dataset(
    input_csv: str = "data/processed/real_samples.csv",
    output_csv: str = "data/processed/dataset_with_groups.csv",
    splits_json: str = "data/processed/splits.json",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Full Stage 1 dataset pipeline.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} samples from {input_csv}")

    # Extract template
    df["template"] = [
        extract_template_label(row["message_id"], row["channel"])
        for _, row in df.iterrows()
    ]

    # Clean and mask text
    print("Stripping template chrome and masking variable tokens...")
    df["clean_text"] = df["ocr_text"].fillna("").apply(strip_template_chrome)
    df["masked_text"] = df["clean_text"].apply(lambda t: mask_text(t, mask_urls=True))

    # Compute group_id via Union-Find connected components
    print("Computing group IDs using TF-IDF cosine similarity >= 0.85...")
    df["group_id"] = compute_group_ids(df["masked_text"].tolist(), similarity_threshold=0.85)

    # Print group statistics
    counts = Counter(df["group_id"])
    print(f"Total unique groups: {len(counts)} across {len(df)} samples")
    print(f"Largest 5 groups: {[count for _, count in counts.most_common(5)]}")

    # Generate grouped splits
    print("Generating grouped train / val / test splits...")
    splits = create_grouped_splits(df, seed=seed)
    print(
        f"Split counts -> Train: {len(splits['train'])} ({len(splits['train'])/len(df)*100:.1f}%), "
        f"Val: {len(splits['val'])} ({len(splits['val'])/len(df)*100:.1f}%), "
        f"Test: {len(splits['test'])} ({len(splits['test'])/len(df)*100:.1f}%)"
    )

    # Assign split column
    df["split"] = "train"
    df.loc[splits["val"], "split"] = "val"
    df.loc[splits["test"], "split"] = "test"
    df["source"] = "main_dataset"

    # Save output
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    with open(splits_json, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Saved dataset with group IDs to {output_csv}")
    print(f"Saved split indices to {splits_json}")
    return df


def recompute_dataset_with_dual_ocr(
    cache_csv: str = "data/processed/dual_ocr_cache.csv",
    input_csv: str = "data/processed/real_samples.csv",
    output_csv: str = "data/processed/dataset_with_groups.csv",
    splits_json: str = "data/processed/splits.json",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Recompute group IDs and splits using the new dual-variant OCR texts.
    Guarantees zero overlap across train, val, and test splits.
    """
    cache_df = pd.read_csv(cache_csv)
    print(f"Loaded {len(cache_df)} dual-variant OCR records from {cache_csv}")

    real_df = pd.read_csv(input_csv)
    # Merge cache with real_samples on channel, label, and message_id
    merged = pd.merge(
        real_df[["channel", "label", "message_id"]],
        cache_df[["channel", "label", "message_id", "ocr_text_dual", "confidence", "winning_variant"]],
        on=["channel", "label", "message_id"],
        how="inner",
    )
    print(f"Merged dataset rows: {len(merged)}")

    # Extract template
    merged["template"] = [
        extract_template_label(row["message_id"], row["channel"])
        for _, row in merged.iterrows()
    ]

    # Clean and mask text using new OCR text
    print("Stripping template chrome and masking variable tokens on new OCR text...")
    merged["ocr_text"] = merged["ocr_text_dual"].fillna("")
    merged["clean_text"] = merged["ocr_text"].apply(strip_template_chrome)
    merged["masked_text"] = merged["clean_text"].apply(lambda t: mask_text(t, mask_urls=True))

    # Compute group_id
    print("Computing new group IDs using TF-IDF cosine similarity >= 0.85...")
    merged["group_id"] = compute_group_ids(merged["masked_text"].tolist(), similarity_threshold=0.85)

    counts = Counter(merged["group_id"])
    print(f"Total unique groups: {len(counts)} across {len(merged)} samples")
    print(f"Largest 5 groups: {[count for _, count in counts.most_common(5)]}")

    # Generate grouped splits
    print("Generating grouped train / val / test splits...")
    splits = create_grouped_splits(merged, seed=seed)

    # Verify zero overlap
    train_groups = set(merged.loc[splits["train"], "group_id"])
    val_groups = set(merged.loc[splits["val"], "group_id"])
    test_groups = set(merged.loc[splits["test"], "group_id"])

    train_val_overlap = train_groups & val_groups
    train_test_overlap = train_groups & test_groups
    val_test_overlap = val_groups & test_groups

    assert len(train_val_overlap) == 0, f"Data leakage! Train-Val overlap: {len(train_val_overlap)}"
    assert len(train_test_overlap) == 0, f"Data leakage! Train-Test overlap: {len(train_test_overlap)}"
    assert len(val_test_overlap) == 0, f"Data leakage! Val-Test overlap: {len(val_test_overlap)}"
    print("Zero data leakage verified! Zero group overlap between train, val, and test.")

    merged["split"] = "train"
    merged.loc[splits["val"], "split"] = "val"
    merged.loc[splits["test"], "split"] = "test"
    merged["source"] = "main_dataset"

    # Save output
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    merged.to_csv(output_csv, index=False)
    with open(splits_json, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Saved updated dataset with dual OCR group IDs to {output_csv}")
    print(f"Saved split indices to {splits_json}")
    return merged


if __name__ == "__main__":
    prepare_dataset()

