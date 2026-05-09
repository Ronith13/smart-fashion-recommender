"""
recommender.py
──────────────
Trains all three models, runs hyperparameter tuning, compares results,
and saves a summary chart + CSV.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import matplotlib.pyplot as plt

from models import (
    ContentBasedRecommender,
    CollaborativeFilteringRecommender,
    MatrixFactorizationRecommender,
    evaluate_model,
)


# ──────────────────────────────────────────────
def compare_models(feature_data: dict,
                   k: int = 5,
                   out_dir: str = "outputs") -> pd.DataFrame:
    """
    Train & evaluate all models (including hyperparameter variants).
    Returns a DataFrame ranked by F1-Score.
    """
    os.makedirs(out_dir, exist_ok=True)

    uim   = feature_data["user_item_matrix"]
    ifm   = feature_data["item_feature_matrix"]
    items = feature_data["items_with_features"]

    results = []

    print("\n─── Content-Based Recommender ───")
    cb = ContentBasedRecommender()
    cb.fit(ifm, items)
    # Content-based can't be scored with evaluate_model directly
    # (it recommends by item, not by user) — we skip numeric eval for it
    # and just confirm it fits.
    print("  (Content-based evaluated qualitatively; see recommend() demo in main.py)")

    print("\n─── Collaborative Filtering – KNN (hyperparameter tuning) ───")
    for k_nbr in [5, 10, 15, 20]:
        cf = CollaborativeFilteringRecommender(n_neighbors=k_nbr)
        cf.fit(uim)
        res = evaluate_model(cf, uim, f"CF-KNN  k={k_nbr}", k=k)
        results.append(res)

    print("\n─── Matrix Factorization – SVD (hyperparameter tuning) ───")
    for n_comp in [20, 50, 100, 150]:
        svd = MatrixFactorizationRecommender(n_components=n_comp)
        svd.fit(uim)
        res = evaluate_model(svd, uim, f"SVD  components={n_comp}", k=k)
        results.append(res)

    df = pd.DataFrame(results).sort_values("f1", ascending=False).reset_index(drop=True)

    # Save results CSV
    csv_path = f"{out_dir}/model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved → {csv_path}")

    # Save bar chart
    _plot_comparison(df, out_dir, k)

    return df, cb          # return content-based model too for the demo


def _plot_comparison(df: pd.DataFrame, out_dir: str, k: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Model Comparison @ k={k}", fontsize=14, fontweight="bold")

    colors = ["#4C72B0" if "KNN" in m else "#DD8452" for m in df["model"]]

    for ax, metric in zip(axes, ["precision", "recall", "f1"]):
        ax.barh(df["model"][::-1], df[metric][::-1], color=colors[::-1])
        ax.set_title(metric.capitalize() + f"@{k}")
        ax.set_xlim(0, df[metric].max() * 1.25)
        for i, v in enumerate(df[metric][::-1]):
            ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    path = f"{out_dir}/model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved    → {path}")
