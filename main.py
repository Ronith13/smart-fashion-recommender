"""
main.py
───────
Entry point for the Smart Fashion Recommender project.

Run:
    python main.py

Pipeline:
    1. Generate synthetic dataset  (10 000 users, 500 items)
    2. EDA                         (plots saved to outputs/eda/)
    3. Feature Engineering         (TF-IDF matrix, user-item pivot)
    4. Model training + tuning     (CF-KNN, SVD)
    5. Print best model + demo     (content-based sample recommendation)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from generate_data        import generate_and_save
from eda                  import run_eda
from feature_engineering  import run_feature_engineering
from recommender          import compare_models

DATA_DIR = "data"
OUT_DIR  = "outputs"


def main():
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  SMART FASHION RECOMMENDER  –  Full Pipeline Rebuild  ║")
    print("╚" + "═" * 58 + "╝")

    # ── Step 1: Data Generation ──────────────────────────────
    print("\n[STEP 1 / 4]  Data Generation")
    if not os.path.exists(f"{DATA_DIR}/users.csv"):
        generate_and_save(DATA_DIR)
    else:
        print("  Dataset already exists – skipping generation.")

    # ── Step 2: EDA ──────────────────────────────────────────
    print("\n[STEP 2 / 4]  Exploratory Data Analysis")
    run_eda(data_dir=DATA_DIR, out_dir=f"{OUT_DIR}/eda")

    # ── Step 3: Feature Engineering ─────────────────────────
    print("\n[STEP 3 / 4]  Feature Engineering")
    feat = run_feature_engineering(data_dir=DATA_DIR)

    # ── Step 4: Model Training & Evaluation ─────────────────
    print("\n[STEP 4 / 4]  Model Training & Hyperparameter Tuning")
    results, cb_model = compare_models(feat, k=5, out_dir=OUT_DIR)

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON RESULTS  (sorted by F1-Score ↓)")
    print("=" * 60)
    print(results.to_string(index=False))

    best = results.iloc[0]
    print(f"\n  🏆  Best Model  : {best['model']}")
    print(f"      Precision@5 : {best['precision']:.4f}")
    print(f"      Recall@5    : {best['recall']:.4f}")
    print(f"      F1-Score    : {best['f1']:.4f}")

    # ── Content-Based Demo ───────────────────────────────────
    print("\n── Content-Based Demo (item_id = 1) ──")
    recs = cb_model.recommend(item_id=1, top_n=5)
    print(f"  Items similar to Item_1:")
    for r in recs:
        print(f"    • {r['item_name']:10s}  [{r['category']}, {r['style']}, {r['color']}]")

    print("\n✅  All done!  Check the 'outputs/' folder for plots and results.\n")


if __name__ == "__main__":
    main()
