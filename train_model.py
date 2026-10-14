"""
train_model.py
==============
Health Risk Prediction & Awareness System
SDG 3 – Good Health and Well-being

PURPOSE
-------
This script:
  1. Loads the Pima Indians Diabetes dataset.
  2. Cleans invalid zero values with documented reasoning.
  3. Splits data into training (80%) and testing (20%) sets.
  4. Scales features using StandardScaler (fitted ONLY on training data).
  5. Trains a Random Forest classifier.
  6. Evaluates the model and prints genuine metrics.
  7. Saves the model, scaler, and metrics to the artifacts/ folder.

DATASET
-------
Source  : Pima Indians Diabetes Database
Origin  : National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
Hosted  : UCI Machine Learning Repository / Kaggle
URL     : https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
Rows    : 768 (full) | 100 (sample included here)
Target  : Outcome — 0 = no diabetes indicator, 1 = positive diabetes indicator
Population: Female patients aged ≥ 21, Pima Indian heritage

LIMITATION NOTE
---------------
This dataset was collected from a specific, narrow demographic group.
Predictions derived from this model may NOT generalise to men, children,
or people from other ethnic or geographic backgrounds.

USAGE
-----
  python train_model.py

HOW TO RUN WITHOUT INSTALLING PYTHON
-------------------------------------
  Open Health_Risk_Prediction_SDG3.ipynb in Google Colab and run all cells.
  The notebook calls this script automatically.
"""

import os
import pickle
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42          # Fixed seed so results are identical every run
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("  Health Risk Prediction — Model Training")
print("=" * 60)
print(f"  Python   : {sys.version}")
print(f"  Platform : {platform.platform()}")
print(f"  NumPy    : {np.__version__}")
print(f"  Pandas   : {pd.__version__}")
print(f"  Sklearn  : {sklearn.__version__}")
print(f"  Seed     : {RANDOM_SEED}")
print("=" * 60)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH     = os.path.join(BASE_DIR, "data", "diabetes.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("\n[1/7] Loading dataset …")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Please place diabetes.csv inside the data/ folder."
    )

df = pd.read_csv(DATA_PATH)
print(f"      Loaded {len(df)} rows × {len(df.columns)} columns.")
print(f"      Columns: {list(df.columns)}")
print(f"\n      Class distribution:")
print(f"        No diabetes (0) : {(df['Outcome'] == 0).sum()}")
print(f"        Diabetes    (1) : {(df['Outcome'] == 1).sum()}")

# ── 2. Data inspection ───────────────────────────────────────────────────────
print("\n[2/7] Inspecting data quality …")
print("\n      Missing values (NaN) per column:")
print(df.isnull().sum().to_string())

# ── 3. Clean invalid zeros ───────────────────────────────────────────────────
# Biological measurements that cannot be zero in a living person.
# Zero in these columns is a data-entry placeholder for a missing measurement.
# Decision: replace each zero with the MEDIAN of that column's non-zero values.
# Why median and not mean? Median is more robust to extreme outlier values.
print("\n[3/7] Replacing biologically impossible zeros …")

ZERO_INVALID_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

replacement_log = {}
for col in ZERO_INVALID_COLS:
    zero_count = (df[col] == 0).sum()
    median_val = df.loc[df[col] != 0, col].median()
    df[col] = df[col].replace(0, median_val)
    replacement_log[col] = {"zeros_replaced": int(zero_count), "replacement_value": round(float(median_val), 2)}
    print(f"      {col:30s}: replaced {zero_count} zeros with median = {median_val:.2f}")

# ── 4. Feature / label split ─────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

X = df[FEATURE_COLUMNS]
y = df["Outcome"]

# ── 5. Train / test split ─────────────────────────────────────────────────────
# stratify=y preserves the class ratio in both splits.
# This prevents the test set from accidentally being all-one-class.
print(f"\n[4/7] Splitting data (80% train / 20% test, seed={RANDOM_SEED}) …")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
)
print(f"      Training samples : {len(X_train)}")
print(f"      Test samples     : {len(X_test)}")

# ── 6. Scale ──────────────────────────────────────────────────────────────────
# CRITICAL: scaler is fitted ONLY on X_train.
# Fitting on X_test or the full dataset would let test information leak into
# training — a form of data leakage that makes metrics unrealistically high.
print("\n[5/7] Scaling features (StandardScaler, fitted on training data only) …")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learn statistics, then scale
X_test_scaled  = scaler.transform(X_test)         # apply learned statistics only

# ── 7. Train ──────────────────────────────────────────────────────────────────
print("\n[6/7] Training Random Forest classifier …")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_SEED,
    class_weight="balanced",   # adjusts for class imbalance automatically
)
model.fit(X_train_scaled, y_train)
print("      Training complete.")

# ── 8. Evaluate ───────────────────────────────────────────────────────────────
print("\n[7/7] Evaluating on held-out test set …")
y_pred      = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
cm        = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("  ACTUAL MODEL EVALUATION RESULTS")
print("  (generated from this run — not manually entered)")
print("=" * 60)
print(f"  Accuracy  : {accuracy:.4f}  ({accuracy:.1%})")
print(f"  Precision : {precision:.4f}  ({precision:.1%})")
print(f"  Recall    : {recall:.4f}  ({recall:.1%})")
print(f"  F1-Score  : {f1:.4f}  ({f1:.1%})")
print(f"\n  Confusion Matrix:")
print(f"                  Predicted No  Predicted Yes")
print(f"  Actual No            {cm[0][0]:>4}           {cm[0][1]:>4}")
print(f"  Actual Yes           {cm[1][0]:>4}           {cm[1][1]:>4}")
print("\n  Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Indicator (0)", "Indicator (1)"]))

print("─" * 60)
print("  WHY ACCURACY ALONE IS INSUFFICIENT FOR HEALTH MODELS")
print("─" * 60)
print(
    "  In health screening, a missed positive case (false negative)\n"
    "  is more dangerous than a false alarm (false positive).\n"
    "  A model that predicts 'No' for every patient would score\n"
    "  ~65% accuracy on this dataset — yet it would be useless.\n"
    "  Recall measures: of all actual positive cases, how many did\n"
    "  the model catch? That is the critical metric for screening.\n"
    "  F1-Score balances both precision and recall in a single number."
)
print("=" * 60)

# ── 9. Save artifacts ─────────────────────────────────────────────────────────
metrics_dict = {
    "accuracy":        accuracy,
    "precision":       precision,
    "recall":          recall,
    "f1":              f1,
    "confusion_matrix": cm,
    "feature_columns": FEATURE_COLUMNS,
    "random_seed":     RANDOM_SEED,
    "sklearn_version": sklearn.__version__,
    "n_train":         len(X_train),
    "n_test":          len(X_test),
    "replacement_log": replacement_log,
}

model_path   = os.path.join(ARTIFACTS_DIR, "model.pkl")
scaler_path  = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.pkl")

with open(model_path,   "wb") as f: pickle.dump(model,        f)
with open(scaler_path,  "wb") as f: pickle.dump(scaler,       f)
with open(metrics_path, "wb") as f: pickle.dump(metrics_dict, f)

print(f"\n  Saved → {model_path}")
print(f"  Saved → {scaler_path}")
print(f"  Saved → {metrics_path}")
print("\n  ✓ Training complete. Run 'streamlit run app.py' to start the app.")
print("=" * 60)
