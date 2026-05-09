"""
models.py
─────────
Three recommendation models:
  1. ContentBasedRecommender   – TF-IDF cosine similarity on item attributes
  2. CollaborativeFilteringRecommender – KNN on the user-item matrix
  3. MatrixFactorizationRecommender    – TruncatedSVD latent factors
 
Evaluation helpers: Precision@K, Recall@K, F1@K
"""
 
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings("ignore")
 
 
# ══════════════════════════════════════════════════════════
# 1. Content-Based Recommender
# ══════════════════════════════════════════════════════════
class ContentBasedRecommender:
    """Recommend items similar to a given item using cosine similarity."""
 
    def __init__(self):
        self.sim_matrix = None
        self.items_df   = None
 
    def fit(self, item_feature_matrix, items_df: pd.DataFrame):
        self.items_df   = items_df.reset_index(drop=True)
        self.sim_matrix = cosine_similarity(item_feature_matrix)
        print("  ✅ ContentBased fitted.")
 
    def recommend(self, item_id: int, top_n: int = 5) -> list:
        idx_list = self.items_df.index[self.items_df["item_id"] == item_id].tolist()
        if not idx_list:
            return []
        idx    = idx_list[0]
        scores = sorted(enumerate(self.sim_matrix[idx]), key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_n]
        return self.items_df.iloc[[s[0] for s in scores]][
            ["item_id", "item_name", "category", "style", "color"]
        ].to_dict("records")
 
 
# ══════════════════════════════════════════════════════════
# 2. Collaborative Filtering – KNN
# ══════════════════════════════════════════════════════════
class CollaborativeFilteringRecommender:
    """User-user collaborative filtering via KNN on the rating matrix."""
 
    def __init__(self, n_neighbors: int = 10, metric: str = "cosine"):
        self.n_neighbors    = n_neighbors
        self.knn            = NearestNeighbors(n_neighbors=n_neighbors,
                                               metric=metric,
                                               algorithm="brute")
        self.user_item_mat  = None
 
    def fit(self, user_item_matrix: pd.DataFrame):
        self.user_item_mat = user_item_matrix
        self.knn.fit(user_item_matrix.values)
        print(f"  ✅ CF-KNN fitted  (k={self.n_neighbors}).")
 
    def recommend(self, user_id: int, top_n: int = 5) -> list:
        if user_id not in self.user_item_mat.index:
            return []
        uid   = self.user_item_mat.index.get_loc(user_id)
        vec   = self.user_item_mat.iloc[uid].values.reshape(1, -1)
        _, nn = self.knn.kneighbors(vec)
        nbrs  = nn[0][1:]                             # exclude self
 
        avg_ratings = self.user_item_mat.iloc[nbrs].mean(axis=0)
        already     = self.user_item_mat.iloc[uid]
        unrated     = already[already == 0].index
        top         = avg_ratings[unrated].nlargest(top_n)
        return list(top.index)
 
    def predict_top_n(self, user_id: int, top_n: int = 10) -> list:
        """Return top-n items by predicted rating (ALL items, for evaluation)."""
        if user_id not in self.user_item_mat.index:
            return []
        uid  = self.user_item_mat.index.get_loc(user_id)
        vec  = self.user_item_mat.iloc[uid].values.reshape(1, -1)
        _, nn = self.knn.kneighbors(vec)
        nbrs = nn[0][1:]
        avg  = self.user_item_mat.iloc[nbrs].mean(axis=0)
        return list(avg.nlargest(top_n).index)
 
 
# ══════════════════════════════════════════════════════════
# 3. Matrix Factorization – TruncatedSVD
# ══════════════════════════════════════════════════════════
class MatrixFactorizationRecommender:
    """Latent factor model via TruncatedSVD (fast collaborative filtering)."""
 
    def __init__(self, n_components: int = 50):
        self.n_components  = n_components
        self.svd           = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_item_mat = None
        self.reconstructed = None
 
    def fit(self, user_item_matrix: pd.DataFrame):
        self.user_item_mat   = user_item_matrix
        mat                  = user_item_matrix.values
        U                    = self.svd.fit_transform(mat)
        V                    = self.svd.components_
        self.reconstructed   = np.dot(U, V)
        ev = self.svd.explained_variance_ratio_.sum()
        print(f"  ✅ SVD fitted  (components={self.n_components}, "
              f"explained variance={ev:.2%}).")
 
    def recommend(self, user_id: int, top_n: int = 5) -> list:
        if user_id not in self.user_item_mat.index:
            return []
        uid   = self.user_item_mat.index.get_loc(user_id)
        preds = self.reconstructed[uid].copy()
 
        # Mask already-rated items so we don't recommend them again
        rated = self.user_item_mat.iloc[uid].values
        preds[rated > 0] = -999
 
        top_idx = preds.argsort()[::-1][:top_n]
        return list(self.user_item_mat.columns[top_idx])
 
    def predict_top_n(self, user_id: int, top_n: int = 10) -> list:
        """Return top-n items by predicted rating (ALL items, for evaluation)."""
        if user_id not in self.user_item_mat.index:
            return []
        uid     = self.user_item_mat.index.get_loc(user_id)
        preds   = self.reconstructed[uid]
        top_idx = preds.argsort()[::-1][:top_n]
        return list(self.user_item_mat.columns[top_idx])
 
 
# ══════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════
def precision_at_k(recommended: list, relevant: list, k: int = 5) -> float:
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / k
 
 
def recall_at_k(recommended: list, relevant: list, k: int = 5) -> float:
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / len(relevant)
 
 
def evaluate_model(model,
                   user_item_matrix: pd.DataFrame,
                   model_name: str,
                   k: int = 5,
                   sample_users: int = 200) -> dict:
    """
    Evaluate a CF model by predicting ratings for ALL items and checking
    how many of each user's highly-rated items appear in the top-K predictions.
    Relevant items = items the user rated >= 3.5.
    """
    precisions, recalls = [], []
 
    for user_id in user_item_matrix.index[:sample_users]:
        ratings  = user_item_matrix.loc[user_id]
        relevant = list(ratings[ratings >= 3.5].index)
        if not relevant:
            continue
 
        recs = model.predict_top_n(user_id, top_n=k * 2)
        if not recs:
            continue
 
        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
 
    p  = float(np.mean(precisions)) if precisions else 0.0
    r  = float(np.mean(recalls))    if recalls    else 0.0
    f1 = (2 * p * r) / (p + r + 1e-9)
 
    print(f"\n  📊 {model_name}  @k={k}")
    print(f"     Precision@{k} : {p:.4f}")
    print(f"     Recall@{k}    : {r:.4f}")
    print(f"     F1-Score      : {f1:.4f}")
 
    return {"model": model_name, "precision": p, "recall": r, "f1": f1}