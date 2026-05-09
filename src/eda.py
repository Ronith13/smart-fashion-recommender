"""
eda.py
──────
Exploratory Data Analysis for the Fashion Recommender dataset.
Produces plots saved to outputs/eda/ and prints key stats.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")


# ──────────────────────────────────────────────
def load_data(data_dir: str = "data"):
    users        = pd.read_csv(f"{data_dir}/users.csv")
    items        = pd.read_csv(f"{data_dir}/items.csv")
    interactions = pd.read_csv(f"{data_dir}/interactions.csv")
    return users, items, interactions


def check_missing(df: pd.DataFrame, name: str) -> None:
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print(f"  [{name}] No missing values ✅")
    else:
        print(f"  [{name}] Missing values found:")
        print(missing[missing > 0].to_string())


def plot_overview(users, items, interactions, out_dir: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("EDA Overview – Smart Fashion Recommender", fontsize=16, fontweight="bold")

    # 1. Age distribution
    sns.histplot(users["age"], bins=20, ax=axes[0, 0], color="#4C72B0")
    axes[0, 0].set_title("User Age Distribution")

    # 2. Preferred style
    style_counts = users["preferred_style"].value_counts()
    axes[0, 1].bar(style_counts.index, style_counts.values, color="#DD8452")
    axes[0, 1].set_title("Preferred Style")
    axes[0, 1].tick_params(axis="x", rotation=30)

    # 3. Rating distribution
    sns.histplot(interactions["rating"], bins=25, ax=axes[0, 2], color="#55A868")
    axes[0, 2].set_title("Rating Distribution")

    # 4. Item categories
    cat_counts = items["category"].value_counts()
    axes[1, 0].barh(cat_counts.index, cat_counts.values, color="#C44E52")
    axes[1, 0].set_title("Item Categories")

    # 5. Purchase rate pie
    pc = interactions["purchased"].value_counts()
    axes[1, 1].pie(pc.values,
                   labels=["Not Purchased", "Purchased"],
                   autopct="%1.1f%%",
                   colors=["#ff9999", "#66b3ff"],
                   startangle=90)
    axes[1, 1].set_title("Purchase Rate")

    # 6. Budget distribution
    budget_counts = users["budget_range"].value_counts()
    axes[1, 2].bar(budget_counts.index, budget_counts.values, color="#8172B2")
    axes[1, 2].set_title("Budget Range")

    plt.tight_layout()
    path = f"{out_dir}/eda_overview.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {path}")


def plot_interactions_per_user(interactions, out_dir: str) -> None:
    counts = interactions.groupby("user_id").size()
    plt.figure(figsize=(10, 4))
    sns.histplot(counts, bins=30, color="#64B5CD")
    plt.title("Interactions per User")
    plt.xlabel("Number of Interactions")
    path = f"{out_dir}/interactions_per_user.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {path}")
    print(f"\n  Avg interactions / user : {counts.mean():.1f}")
    print(f"  Min / Max               : {counts.min()} / {counts.max()}")


def plot_correlation_heatmap(interactions, out_dir: str) -> None:
    numeric = interactions[["rating", "purchased"]]
    corr = numeric.corr()
    plt.figure(figsize=(5, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation – Ratings vs Purchase")
    path = f"{out_dir}/correlation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {path}")


# ──────────────────────────────────────────────
def run_eda(data_dir: str = "data", out_dir: str = "outputs/eda"):
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    users, items, interactions = load_data(data_dir)

    print(f"\n  Rows  →  users: {len(users):,} | items: {len(items):,} | "
          f"interactions: {len(interactions):,}")

    print("\n--- Missing Values ---")
    check_missing(users, "users")
    check_missing(items, "items")
    check_missing(interactions, "interactions")

    print("\n--- Rating Summary ---")
    print(interactions["rating"].describe().round(2).to_string())

    print("\n--- Generating Plots ---")
    plot_overview(users, items, interactions, out_dir)
    plot_interactions_per_user(interactions, out_dir)
    plot_correlation_heatmap(interactions, out_dir)

    print("\n✅  EDA complete!")
    return users, items, interactions


if __name__ == "__main__":
    run_eda()
