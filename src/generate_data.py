"""
generate_data.py
────────────────
Generates a synthetic Fashion dataset:
  - 10,000 users  (preferences, demographics)
  - 500   items   (clothing attributes)
  - ~150,000 interactions (ratings + purchase flag)
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)


# ──────────────────────────────────────────────
def generate_users(n_users: int = 10_000) -> pd.DataFrame:
    styles  = ["Casual", "Formal", "Sporty", "Bohemian", "Streetwear"]
    sizes   = ["XS", "S", "M", "L", "XL"]
    genders = ["Male", "Female", "Non-binary"]
    budgets = ["Low", "Medium", "High"]

    return pd.DataFrame({
        "user_id":         range(1, n_users + 1),
        "age":             np.random.randint(18, 61, n_users),
        "gender":          np.random.choice(genders, n_users, p=[0.45, 0.45, 0.10]),
        "preferred_style": np.random.choice(styles,  n_users),
        "size":            np.random.choice(sizes,   n_users, p=[0.10, 0.20, 0.35, 0.25, 0.10]),
        "budget_range":    np.random.choice(budgets, n_users, p=[0.30, 0.50, 0.20]),
    })


def generate_items(n_items: int = 500) -> pd.DataFrame:
    categories = ["Top", "Bottom", "Dress", "Outerwear", "Footwear", "Accessory"]
    styles     = ["Casual", "Formal", "Sporty", "Bohemian", "Streetwear"]
    colors     = ["Black", "White", "Blue", "Red", "Green",
                  "Yellow", "Pink", "Grey", "Brown", "Purple"]
    seasons    = ["Spring", "Summer", "Fall", "Winter", "All"]
    price_ranges = ["Low", "Medium", "High"]
    materials  = ["Cotton", "Polyester", "Denim", "Wool", "Silk", "Linen", "Synthetic"]

    return pd.DataFrame({
        "item_id":     range(1, n_items + 1),
        "item_name":   [f"Item_{i}" for i in range(1, n_items + 1)],
        "category":    np.random.choice(categories,   n_items),
        "style":       np.random.choice(styles,       n_items),
        "color":       np.random.choice(colors,       n_items),
        "season":      np.random.choice(seasons,      n_items),
        "price_range": np.random.choice(price_ranges, n_items, p=[0.30, 0.50, 0.20]),
        "material":    np.random.choice(materials,    n_items),
    })


def generate_interactions(users: pd.DataFrame,
                           items: pd.DataFrame,
                           avg_interactions: int = 15) -> pd.DataFrame:
    user_styles  = dict(zip(users["user_id"], users["preferred_style"]))
    user_budgets = dict(zip(users["user_id"], users["budget_range"]))
    item_styles  = dict(zip(items["item_id"], items["style"]))
    item_prices  = dict(zip(items["item_id"], items["price_range"]))
    item_ids     = items["item_id"].values

    rows = []
    for user_id in users["user_id"]:
        n = int(np.clip(np.random.poisson(avg_interactions), 5, 50))
        sampled = np.random.choice(item_ids, n, replace=False)

        for item_id in sampled:
            base = np.random.uniform(1.0, 5.0)
            if user_styles[user_id]  == item_styles[item_id]:
                base += np.random.uniform(0.5, 1.5)
            if user_budgets[user_id] == item_prices[item_id]:
                base += np.random.uniform(0.3, 0.7)
            rating    = round(min(5.0, base), 1)
            purchased = int(rating >= 3.5)
            rows.append((user_id, int(item_id), rating, purchased))

    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "purchased"])


# ──────────────────────────────────────────────
def generate_and_save(output_dir: str = "data") -> tuple:
    os.makedirs(output_dir, exist_ok=True)

    print("Generating users …")
    users = generate_users(10_000)

    print("Generating items …")
    items = generate_items(500)

    print("Generating interactions …")
    interactions = generate_interactions(users, items)

    users.to_csv(f"{output_dir}/users.csv",        index=False)
    items.to_csv(f"{output_dir}/items.csv",         index=False)
    interactions.to_csv(f"{output_dir}/interactions.csv", index=False)

    print(f"\n✅  Dataset saved to '{output_dir}/'")
    print(f"    Users:        {len(users):>10,}")
    print(f"    Items:        {len(items):>10,}")
    print(f"    Interactions: {len(interactions):>10,}")
    return users, items, interactions


if __name__ == "__main__":
    generate_and_save()

