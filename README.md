# 👗 Smart Fashion Recommender

Machine learning recommendation system trained on 10,000+ user records, covering EDA, 
feature engineering, and collaborative filtering. Evaluated multiple models via 
hyperparameter tuning, reaching a Precision@5 of 0.87 on the top-performing model.

**Stack:** Python, Scikit-learn, Pandas
---

## 📌 Project Overview

- Built on a synthetic dataset of **10,000+ users** and **500 clothing items**
- Performed **Exploratory Data Analysis (EDA)** to understand data patterns
- Applied **Feature Engineering** to improve model input quality
- Evaluated **3 ML models** with hyperparameter tuning
- Best model achieved **Precision@5: 0.87 | F1-Score: 0.75**

---

## 🗂️ Project Structure
smart-fashion-recommender/
├── src/
│   ├── generate_data.py        # Generates synthetic dataset
│   ├── eda.py                  # Exploratory Data Analysis
│   ├── feature_engineering.py  # Feature transformations
│   ├── models.py               # ML models + evaluation
│   └── recommender.py          # Model comparison + tuning
├── main.py                     # Run the full pipeline
└── requirements.txt            # Dependencies
---

## ⚙️ How to Run

**1. Clone the repository**
git clone https://github.com/Ronith13/smart-fashion-recommender.git
cd smart-fashion-recommender

**2. Create virtual environment**
python -m venv venv
venv\Scripts\activate

**3. Install dependencies**
pip install -r requirements.txt

**4. Run the project**
python main.py

---

## 🤖 Models Used

| Model | Precision@5 | Recall@5 | F1-Score |
|-------|------------|----------|----------|
| CF-KNN (k=20) | 0.871 | 0.665 | 0.754 |
| CF-KNN (k=15) | 0.849 | 0.647 | 0.734 |
| SVD (150 components) | 0.810 | 0.616 | 0.699 |
| SVD (100 components) | 0.736 | 0.555 | 0.633 |

---

## 🛠️ Tech Stack

- **Python 3.11**
- **Pandas & NumPy** — Data manipulation
- **Scikit-learn** — ML models
- **Matplotlib & Seaborn** — Visualizations

---
