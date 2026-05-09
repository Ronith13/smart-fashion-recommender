"""
feature_engineering.py
───────────────────────
Transforms raw CSV data into ML-ready feature matrices:
  - Item feature matrix (TF-IDF on clothing attributes)
  - User-item rating matrix (pivot table)
  - Enriched user features (interaction stats + scaling)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer


# ──────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Label-encode categorical columns (adds *_encoded suffix)."""
    df = df.copy()
    le = LabelEncoder()
    for col in columns:
        df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
    return df


def build_item_feature_matrix(items: pd.DataFrame):
    """
    Combine clothing attributes into a single text blob per item,
    then apply TF-IDF to get a numeric feature matrix.
    """
    items = items.copy()
    items["features"] = (
        items["category"] + " " +
        items["style"]    + " " +
        items["color"]    + " " +
        items["season"]   + " " +
        items["material"] + " " +
        items["price_range"]
    )
    tfidf  = TfidfVectorizer()
    matrix = tfidf.fit_transform(items["features"])
    print(f"  Item feature matrix  : {matrix.shape}")
    return matrix, tfidf, items


def build_user_item_matrix(interactions: pd.DataFrame) -> pd.DataFrame:
    """Pivot interactions → user × item rating matrix (0 = no rating)."""
    matrix = interactions.pivot_table(
        index="user_id", columns="item_id", values="rating", fill_value=0
    )
    sparsity = 1.0 - (len(interactions) / (matrix.shape[0] * matrix.shape[1]))
    print(f"  User-item matrix     : {matrix.shape}  |  sparsity: {sparsity:.2%}")
    return matrix


def engineer_user_features(users: pd.DataFrame,
                            interactions: pd.DataFrame) -> pd.DataFrame:
    """Add aggregated interaction stats to each user, then scale."""
    stats = interactions.groupby("user_id").agg(
        avg_rating         =("rating",    "mean"),
        total_interactions =("item_id",   "count"),
        purchase_rate      =("purchased", "mean"),
    ).reset_index()

    enriched = users.merge(stats, on="user_id", how="left")

    scaler = MinMaxScaler()
    numeric = ["age", "avg_rating", "total_interactions", "purchase_rate"]
    enriched[numeric] = scaler.fit_transform(enriched[numeric].fillna(0))

    print(f"  Enriched user matrix : {enriched.shape}")
    return enriched


# ──────────────────────────────────────────────
def run_feature_engineering(data_dir: str = "data") -> dict:
    print("\n" + "=" * 50)
    print("  FEATURE ENGINEERING")
    print("=" * 50)

    users        = pd.read_csv(f"{data_dir}/users.csv")
    items        = pd.read_csv(f"{data_dir}/items.csv")
    interactions = pd.read_csv(f"{data_dir}/interactions.csv")

    # Encode categoricals
    items_enc = encode_categoricals(
        items, ["category", "style", "color", "season", "price_range", "material"]
    )
    users_enc = encode_categoricals(
        users, ["gender", "preferred_style", "size", "budget_range"]
    )

    # Build matrices
    item_feat_matrix, tfidf, items_with_feat = build_item_feature_matrix(items_enc)
    user_item_matrix = build_user_item_matrix(interactions)
    users_enriched   = engineer_user_features(users_enc, interactions)

    print("\n✅  Feature engineering complete!")
    return {
        "users":               users_enriched,
        "items":               items_enc,
        "item_feature_matrix": item_feat_matrix,
        "user_item_matrix":    user_item_matrix,
        "tfidf":               tfidf,
        "items_with_features": items_with_feat,
        "interactions":        interactions,
    }


if __name__ == "__main__":
    run_feature_engineering()
