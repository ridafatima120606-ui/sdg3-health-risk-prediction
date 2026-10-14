"""
app.py
======
Health Risk Prediction & Awareness System
SDG 3 – Good Health and Well-being

Streamlit web application.

PRIVACY NOTE
------------
This application does NOT store, log, transmit, or share any health
measurements entered by the user. All processing happens locally in
the running Python process. No data is sent to any external server.
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Health Risk Prediction | SDG 3",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DATA_PATH     = os.path.join(BASE_DIR, "data", "diabetes.csv")

# ── Auto-train if artifacts are missing ───────────────────────────────────────
def _train_and_save():
    """Train the model from scratch and save artifacts. Called automatically."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 f1_score, precision_score, recall_score)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import sklearn

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    ZERO_INVALID_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    replacement_log = {}
    for col in ZERO_INVALID_COLS:
        zero_count = (df[col] == 0).sum()
        median_val = df.loc[df[col] != 0, col].median()
        df[col] = df[col].replace(0, median_val)
        replacement_log[col] = {"zeros_replaced": int(zero_count),
                                 "replacement_value": round(float(median_val), 2)}

    FEATURE_COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    X = df[FEATURE_COLUMNS]
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED,
                                   class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    metrics_dict = {
        "accuracy":         accuracy_score(y_test, y_pred),
        "precision":        precision_score(y_test, y_pred, zero_division=0),
        "recall":           recall_score(y_test, y_pred, zero_division=0),
        "f1":               f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "feature_columns":  FEATURE_COLUMNS,
        "random_seed":      RANDOM_SEED,
        "sklearn_version":  sklearn.__version__,
        "n_train":          len(X_train),
        "n_test":           len(X_test),
        "replacement_log":  replacement_log,
    }

    with open(os.path.join(ARTIFACTS_DIR, "model.pkl"),   "wb") as f: pickle.dump(model,        f)
    with open(os.path.join(ARTIFACTS_DIR, "scaler.pkl"),  "wb") as f: pickle.dump(scaler,       f)
    with open(os.path.join(ARTIFACTS_DIR, "metrics.pkl"), "wb") as f: pickle.dump(metrics_dict, f)


@st.cache_resource(show_spinner="Training model for first run — please wait …")
def load_artifacts():
    model_path   = os.path.join(ARTIFACTS_DIR, "model.pkl")
    scaler_path  = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.pkl")

    # If any artifact is missing, train now
    if not all(os.path.exists(p) for p in [model_path, scaler_path, metrics_path]):
        _train_and_save()

    with open(model_path,   "rb") as f: model   = pickle.load(f)
    with open(scaler_path,  "rb") as f: scaler  = pickle.load(f)
    with open(metrics_path, "rb") as f: metrics = pickle.load(f)
    return model, scaler, metrics, []

model, scaler, metrics, missing_files = load_artifacts()

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; padding:1.2rem 0 0.4rem 0;">
        <span style="font-size:3rem;">🏥</span>
        <h1 style="margin:0.3rem 0 0 0; font-size:1.9rem;">
            Health Risk Prediction &amp; Awareness
        </h1>
        <p style="color:#555; margin-top:0.3rem; font-size:1rem;">
            <strong>SDG 3 – Good Health and Well-being</strong>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("📌 What is SDG 3?", expanded=False):
    st.markdown(
        """
        **UN Sustainable Development Goal 3** calls for ensuring healthy lives and promoting
        well-being for all at all ages. This includes reducing non-communicable diseases such
        as diabetes, which affects over **537 million adults** worldwide (IDF, 2021).

        This educational prototype demonstrates how machine learning can be used for **health
        awareness and risk communication** — one of many possible contributions to SDG 3.
        """
    )

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — CRITICAL DISCLAIMER
# ─────────────────────────────────────────────────────────────────────────────
st.error(
    "⚠️ **IMPORTANT DISCLAIMER — Please read before using this tool.**\n\n"
    "This application is an **educational prototype built by a student** as an academic "
    "exercise. It is **not a medical device**, **not a diagnostic tool**, and has **not been "
    "reviewed or approved by any healthcare authority**.\n\n"
    "- It **cannot diagnose, confirm, or rule out** diabetes or any other condition.\n"
    "- Its predictions are based on a **small sample dataset** (100 rows) from a **specific "
    "demographic group** (Pima Indian women aged 21+) and may not apply to other populations.\n"
    "- **Do not use this tool to make any health or treatment decisions.**\n"
    "- If you have concerns about your health, please consult a **qualified medical professional**."
)

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — PRIVACY NOTE
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🔒 Privacy Note", expanded=False):
    st.markdown(
        """
        **Your data stays on your device.**

        This application processes all inputs locally within the running Python process.
        No health measurements, personal information, or results are stored, logged, or
        transmitted to any external server or third party. When you close the browser tab,
        your inputs are discarded.

        *Note: If this app is deployed on Streamlit Community Cloud, standard web server
        access logs (IP address, timestamp) may be recorded by Streamlit's hosting
        infrastructure. No health data is included in these logs.*
        """
    )

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — INPUT FIELDS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Enter Health Measurements")
st.markdown(
    "Enter the values below. All fields are required. "
    "If you do not have a specific measurement, use the default value provided — "
    "but note that this reduces the reliability of the result."
)

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Number of Pregnancies",
        min_value=0, max_value=20, value=1, step=1,
        help="Total number of times the patient has been pregnant. Enter 0 if none.",
    )
    glucose = st.number_input(
        "Glucose Level (mg/dL)",
        min_value=50, max_value=300, value=120, step=1,
        help="Plasma glucose concentration from a 2-hour oral glucose tolerance test.",
    )
    blood_pressure = st.number_input(
        "Blood Pressure (mm Hg)",
        min_value=30, max_value=150, value=70, step=1,
        help="Diastolic blood pressure (the lower number in a reading).",
    )
    skin_thickness = st.number_input(
        "Skin Thickness (mm)",
        min_value=5, max_value=100, value=20, step=1,
        help="Triceps skin-fold thickness — an indirect measure of body fat.",
    )

with col2:
    insulin = st.number_input(
        "Insulin Level (µU/mL)",
        min_value=1, max_value=900, value=80, step=1,
        help="2-hour serum insulin level.",
    )
    bmi = st.number_input(
        "BMI (kg/m²)",
        min_value=10.0, max_value=70.0, value=28.0, step=0.1,
        help="Body Mass Index = weight in kg ÷ (height in metres)².",
    )
    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.05, max_value=3.0, value=0.35, step=0.01,
        help=(
            "A score (0.05–3.0) that represents diabetes likelihood based on family history. "
            "Higher values indicate a stronger family history. "
            "If unknown, leave at 0.35 (approximate dataset median)."
        ),
    )
    age = st.number_input(
        "Age (years)",
        min_value=18, max_value=100, value=30, step=1,
        help="Patient age in years. The dataset covers ages 21 and above.",
    )

if age < 21:
    st.warning(
        "⚠️ The training dataset only included patients aged 21 and above. "
        "Results for younger ages may be less reliable."
    )

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🔍 Predict Risk", use_container_width=True, type="primary")

if predict_btn:
    input_array = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                              insulin, bmi, dpf, age]])
    input_scaled    = scaler.transform(input_array)
    prediction      = model.predict(input_scaled)[0]
    probabilities   = model.predict_proba(input_scaled)[0]
    prob_no  = probabilities[0] * 100
    prob_yes = probabilities[1] * 100

    st.subheader("📊 Model Classification Result")

    if prediction == 1:
        st.error(
            f"**Model Classification: Higher-Risk Pattern**\n\n"
            f"The model estimates a **{prob_yes:.1f}%** probability that these measurements "
            f"match patterns associated with a positive diabetes indicator in the training data."
        )
    else:
        st.success(
            f"**Model Classification: Lower-Risk Pattern**\n\n"
            f"The model estimates a **{prob_no:.1f}%** probability that these measurements "
            f"match patterns associated with a negative diabetes indicator in the training data."
        )

    st.info(
        "**What this result means:**\n\n"
        "This is a machine-learning classification, not a medical diagnosis. "
        "The model compares your input values to patterns it learned from 100 training "
        "samples from a specific population group. The result indicates which pattern "
        "your measurements most resemble — it does not confirm or exclude any condition.\n\n"
        "**A 'Higher-Risk' result does NOT mean you have diabetes.**\n"
        "**A 'Lower-Risk' result does NOT mean you do not have diabetes.**\n\n"
        "Please speak with a healthcare professional for a proper assessment."
    )

    fig_p, ax_p = plt.subplots(figsize=(5, 2))
    ax_p.barh(["Lower-Risk Pattern", "Higher-Risk Pattern"],
              [prob_no, prob_yes],
              color=["#10b981", "#ef4444"])
    ax_p.set_xlim(0, 100)
    ax_p.set_xlabel("Model Probability (%)")
    ax_p.set_title("Estimated Pattern Probability")
    for y_pos, val in zip([0, 1], [prob_no, prob_yes]):
        ax_p.text(val + 1, y_pos, f"{val:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_p)
    plt.close(fig_p)

    st.markdown("---")
    st.subheader("💡 General Health Awareness Information")
    st.markdown(
        """
        The following are **general lifestyle recommendations** based on established
        public health guidance (WHO, CDC). They are not personalised medical advice.

        | Factor | Guidance |
        |---|---|
        | 🥦 Diet | Choose whole grains, vegetables, and lean proteins. Limit refined sugar and processed foods. |
        | 🏃 Physical activity | At least 150 minutes of moderate activity per week (e.g. brisk walking). |
        | ⚖️ Body weight | Maintaining a healthy BMI (18.5–24.9) reduces type 2 diabetes risk. |
        | 💧 Hydration | Prefer water over sugary drinks. |
        | 🩺 Regular check-ups | Blood glucose screening is recommended for adults with risk factors. |
        | 🚭 Avoid smoking | Smoking significantly increases cardiovascular and metabolic disease risk. |

        *Sources: World Health Organization (WHO), Centers for Disease Control and Prevention (CDC)*
        """
    )

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Model Performance (Genuine Evaluation Results)")

st.markdown(
    "These metrics were computed by running the trained model on the **held-out test set** — "
    "data the model never saw during training (20% of the dataset). "
    "They reflect real performance, not ideal performance."
)

m = metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy",  f"{m['accuracy']:.1%}")
c2.metric("Precision", f"{m['precision']:.1%}")
c3.metric("Recall",    f"{m['recall']:.1%}")
c4.metric("F1-Score",  f"{m['f1']:.1%}")

with st.expander("📖 What do these metrics mean?"):
    st.markdown(
        """
| Metric | Plain-English Meaning | Why it Matters Here |
|---|---|---|
| **Accuracy** | % of all predictions that were correct | Useful but misleading with imbalanced classes |
| **Precision** | Of all predicted high-risk cases, what % actually were? | High precision = fewer unnecessary alarms |
| **Recall** | Of all actual high-risk cases, what % did the model catch? | **Critical for health screening** — missing a case is costly |
| **F1-Score** | Harmonic mean of Precision and Recall | Balances both concerns into one number |

> **Why accuracy alone is not enough:** If 65% of patients in a dataset have no diabetes,
> a model that always predicts "No" would achieve 65% accuracy — but it would miss every
> single at-risk patient. In health applications, **Recall** is the most important metric
> because failing to identify someone at risk has serious consequences.
        """
    )

cm = m["confusion_matrix"]
st.markdown("**Confusion Matrix**")
st.markdown(
    "Each cell shows how many predictions the model made in each category. "
    "The diagonal (top-left, bottom-right) represents correct predictions."
)
fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.5))
im = ax_cm.imshow(cm, cmap="Blues", interpolation="nearest")
ax_cm.set_title("Confusion Matrix (Test Set)")
ax_cm.set_xlabel("Predicted Label")
ax_cm.set_ylabel("True Label")
ax_cm.set_xticks([0, 1]); ax_cm.set_xticklabels(["No Indicator (0)", "Indicator (1)"], fontsize=8)
ax_cm.set_yticks([0, 1]); ax_cm.set_yticklabels(["No Indicator (0)", "Indicator (1)"], fontsize=8)
for i in range(2):
    for j in range(2):
        ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold",
                   color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.colorbar(im, ax=ax_cm)
plt.tight_layout()
st.pyplot(fig_cm)
plt.close(fig_cm)

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 7 — METHODOLOGY AND LIMITATIONS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔬 Methodology & Limitations")

with st.expander("Dataset Source and Description"):
    st.markdown(
        """
        **Dataset:** Pima Indians Diabetes Database
        **Origin:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
        **Hosted at:** UCI Machine Learning Repository / Kaggle
        **URL:** https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

        | Column | Description |
        |---|---|
        | Pregnancies | Number of pregnancies |
        | Glucose | 2-hour plasma glucose (mg/dL) from OGTT |
        | BloodPressure | Diastolic blood pressure (mm Hg) |
        | SkinThickness | Triceps skin-fold thickness (mm) |
        | Insulin | 2-hour serum insulin (µU/mL) |
        | BMI | Body Mass Index (kg/m²) |
        | DiabetesPedigreeFunction | Family history score |
        | Age | Age in years |
        | Outcome | 0 = negative indicator, 1 = positive indicator |

        **Important limitations:**
        - Collected from Pima Indian **women aged 21 and above** only.
        - Results may not generalise to men, children, or other ethnic groups.
        - Several columns contain zeros that represent missing data (not true zero readings).
        """
    )

with st.expander("Data Cleaning Decisions"):
    if "replacement_log" in metrics:
        st.markdown(
            "The following columns had biologically impossible zero values that were replaced "
            "with the median of each column's non-zero values to avoid misleading the model:"
        )
        log_df = pd.DataFrame(metrics["replacement_log"]).T.reset_index()
        log_df.columns = ["Column", "Zeros Replaced", "Replacement Value (Median)"]
        st.dataframe(log_df, use_container_width=True)
    else:
        st.markdown("Replacement log not available.")

with st.expander("Model Architecture"):
    st.markdown(
        f"""
        **Algorithm:** Random Forest Classifier
        **Library:** scikit-learn {metrics.get('sklearn_version', '')}
        **Hyperparameters:** n_estimators=100, class_weight='balanced', random_state={metrics['random_seed']}
        **Train/Test split:** 80% / 20%, stratified by class label
        **Feature scaling:** StandardScaler (fitted on training data only — no data leakage)
        **Training samples:** {metrics['n_train']}
        **Test samples:** {metrics['n_test']}
        """
    )

with st.expander("Known Limitations and Caveats"):
    st.markdown(
        """
        1. **Small sample size:** The included sample dataset has only 100 rows.
           Small datasets produce models that may not generalise well.
        2. **Population specificity:** The dataset represents a narrow demographic.
           The model's predictions may be less reliable for individuals outside that group.
        3. **Missing data handling:** Zero values were replaced with medians.
           This is a reasonable but imperfect approach.
        4. **No clinical validation:** This model has not been tested in clinical settings
           and has not been reviewed by medical professionals.
        5. **Binary classification only:** The model outputs one of two classes.
           Real diabetes risk exists on a spectrum and involves many additional factors.
        6. **Educational use only:** This prototype is built for learning purposes.
           It should never influence healthcare decisions.
        """
    )

# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:0.82rem;'>"
    "🌍 <b>SDG 3 – Good Health and Well-being</b> &nbsp;|&nbsp; "
    "Educational Student Project &nbsp;|&nbsp; "
    "Not a medical device &nbsp;|&nbsp; "
    "No health data is stored or transmitted"
    "</p>",
    unsafe_allow_html=True,
)
