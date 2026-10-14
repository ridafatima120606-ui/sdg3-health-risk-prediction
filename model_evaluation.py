"""
model_evaluation.py
===================
Health Risk Prediction & Awareness System
SDG 3 – Good Health and Well-being

PURPOSE
-------
Standalone script that:
  - Loads the saved model artifacts.
  - Re-runs evaluation on the test set.
  - Produces and saves evaluation charts as PNG files.
  - Prints a full reproducibility report.

Run this AFTER train_model.py has been executed:
    python model_evaluation.py

The script will create a docs/evaluation/ folder with charts.
"""

import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for servers and Colab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DATA_PATH     = os.path.join(BASE_DIR, "data", "diabetes.csv")
OUTPUT_DIR    = os.path.join(BASE_DIR, "docs", "evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load artifacts ────────────────────────────────────────────────────────────
print("=" * 60)
print("  Model Evaluation Report")
print("=" * 60)

for name, path in [("model.pkl", os.path.join(ARTIFACTS_DIR, "model.pkl")),
                   ("scaler.pkl", os.path.join(ARTIFACTS_DIR, "scaler.pkl")),
                   ("metrics.pkl", os.path.join(ARTIFACTS_DIR, "metrics.pkl"))]:
    if not os.path.exists(path):
        print(f"\n  ERROR: {name} not found at {path}")
        print("  Please run train_model.py first.")
        sys.exit(1)

with open(os.path.join(ARTIFACTS_DIR, "model.pkl"),   "rb") as f: model   = pickle.load(f)
with open(os.path.join(ARTIFACTS_DIR, "scaler.pkl"),  "rb") as f: scaler  = pickle.load(f)
with open(os.path.join(ARTIFACTS_DIR, "metrics.pkl"), "rb") as f: metrics = pickle.load(f)

print(f"\n  Model trained with scikit-learn {metrics.get('sklearn_version', 'unknown')}")
print(f"  Current scikit-learn             : {sklearn.__version__}")
print(f"  Random seed used during training : {metrics['random_seed']}")
print(f"  Training samples : {metrics['n_train']}")
print(f"  Test samples     : {metrics['n_test']}")

# ── Reproduce the test set ────────────────────────────────────────────────────
# We rebuild the exact same test set using the stored seed and split ratio.
ZERO_INVALID_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
FEATURE_COLUMNS   = metrics["feature_columns"]
RANDOM_SEED       = metrics["random_seed"]

df = pd.read_csv(DATA_PATH)
for col in ZERO_INVALID_COLS:
    median_val = df.loc[df[col] != 0, col].median()
    df[col] = df[col].replace(0, median_val)

X = df[FEATURE_COLUMNS]
y = df["Outcome"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y)
X_test_scaled = scaler.transform(X_test)

y_pred      = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

try:
    roc_auc = roc_auc_score(y_test, y_pred_prob)
except Exception:
    roc_auc = None

# ── Print metrics ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  REPRODUCED EVALUATION METRICS")
print("─" * 60)
print(f"  Accuracy  : {metrics['accuracy']:.4f}")
print(f"  Precision : {metrics['precision']:.4f}")
print(f"  Recall    : {metrics['recall']:.4f}")
print(f"  F1-Score  : {metrics['f1']:.4f}")
if roc_auc is not None:
    print(f"  ROC-AUC   : {roc_auc:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=["No Indicator (0)", "Indicator (1)"]))

# ── Chart 1: Confusion Matrix ────────────────────────────────────────────────
print("\n  Saving confusion matrix chart …")
fig1, ax1 = plt.subplots(figsize=(5, 4))
cm = metrics["confusion_matrix"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Indicator", "Indicator"])
disp.plot(ax=ax1, colorbar=True, cmap="Blues")
ax1.set_title("Confusion Matrix\n(held-out test set)")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
fig1.savefig(cm_path, dpi=150)
plt.close(fig1)
print(f"  Saved → {cm_path}")

# ── Chart 2: Feature Importance ──────────────────────────────────────────────
print("  Saving feature importance chart …")
importances = model.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]

fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.bar(
    [FEATURE_COLUMNS[i] for i in sorted_idx],
    importances[sorted_idx],
    color="#3b82d4",
)
ax2.set_title("Feature Importance (Random Forest)")
ax2.set_ylabel("Importance Score")
ax2.set_xlabel("Feature")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
fi_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
fig2.savefig(fi_path, dpi=150)
plt.close(fig2)
print(f"  Saved → {fi_path}")

# ── Chart 3: ROC Curve ────────────────────────────────────────────────────────
if roc_auc is not None:
    print("  Saving ROC curve chart …")
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_pred_prob, ax=ax3, name="Random Forest")
    ax3.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.50)")
    ax3.set_title("ROC Curve")
    ax3.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    fig3.savefig(roc_path, dpi=150)
    plt.close(fig3)
    print(f"  Saved → {roc_path}")

# ── Chart 4: Metrics bar chart ────────────────────────────────────────────────
print("  Saving metrics summary chart …")
metric_names  = ["Accuracy", "Precision", "Recall", "F1-Score"]
metric_values = [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]]

fig4, ax4 = plt.subplots(figsize=(6, 3.5))
bars = ax4.bar(metric_names, metric_values, color=["#3b82d4", "#7c5cd8", "#f59e0b", "#10b981"])
ax4.set_ylim(0, 1.0)
ax4.set_ylabel("Score")
ax4.set_title("Model Performance Summary (Test Set)")
for bar, val in zip(bars, metric_values):
    ax4.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
             f"{val:.2f}", ha="center", va="bottom", fontweight="bold")
plt.tight_layout()
ms_path = os.path.join(OUTPUT_DIR, "metrics_summary.png")
fig4.savefig(ms_path, dpi=150)
plt.close(fig4)
print(f"  Saved → {ms_path}")

print("\n  ✓ All evaluation charts saved to:", OUTPUT_DIR)
print("=" * 60)

# ── Data cleaning log ─────────────────────────────────────────────────────────
print("\n  DATA CLEANING DECISIONS")
print("─" * 60)
if "replacement_log" in metrics:
    for col, info in metrics["replacement_log"].items():
        print(f"  {col:30s} : {info['zeros_replaced']} zeros → median {info['replacement_value']}")
else:
    print("  (replacement log not available — retrain with latest train_model.py)")
print("=" * 60)
