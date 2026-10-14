# 🏥 Health Risk Prediction & Awareness System
### SDG 3 – Good Health and Well-being

> **Student Project | Educational Prototype**
> ⚠️ This application is not a medical diagnostic tool. See the [Disclaimer](#disclaimer) section.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[YOUR-GITHUB-USERNAME]/sdg3-health-risk-prediction/blob/main/Health_Risk_Prediction_SDG3.ipynb)
[![Live App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://[YOUR-APP-URL].streamlit.app)

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement and Objectives](#problem-statement-and-objectives)
3. [SDG 3 Connection](#sdg-3-connection)
4. [Dataset](#dataset)
5. [System Architecture](#system-architecture)
6. [Technologies Used](#technologies-used)
7. [How to Run in Google Colab](#how-to-run-in-google-colab)
8. [Model Training and Evaluation](#model-training-and-evaluation)
9. [Results](#results)
10. [Screenshots](#screenshots)
11. [Deployment](#deployment)
12. [Disclaimer](#disclaimer)
13. [Credits and License](#credits-and-license)

---

## Project Overview

This project builds an educational machine-learning web application that demonstrates
how health measurements can be used to classify patterns associated with diabetes risk.
It is designed as a student learning exercise aligned with **United Nations Sustainable
Development Goal 3 (SDG 3) — Good Health and Well-being**.

The application:
- Accepts 8 standard health measurements as input.
- Classifies the input as a "higher-risk" or "lower-risk" pattern using a trained Random Forest model.
- Displays the model's estimated probability for each class.
- Shows genuine model evaluation metrics (accuracy, precision, recall, F1-score, confusion matrix).
- Provides general health awareness information.
- Includes a full methodology and limitations section.

---

## Problem Statement and Objectives

**Problem:** Diabetes is a major global health challenge affecting over 537 million adults
(International Diabetes Federation, 2021). Early awareness of risk factors can support
lifestyle changes that reduce incidence of type 2 diabetes. However, awareness tools are
often complex or inaccessible.

**Objectives:**

| # | Objective |
|---|---|
| 1 | Demonstrate a complete machine-learning pipeline from raw data to a deployed web application. |
| 2 | Build a transparent, honest educational tool that communicates both predictions and their limitations. |
| 3 | Apply responsible AI principles: disclose data sources, limitations, and population specificity. |
| 4 | Align the project with SDG 3 by promoting health awareness through technology. |

---

## SDG 3 Connection

**Goal 3: Good Health and Well-being** targets include:

- **3.4** — Reduce premature mortality from non-communicable diseases (NCDs) by one third by 2030.
- **3.8** — Achieve universal health coverage and access to quality healthcare services.
- **3.d** — Strengthen capacities for health risk reduction and management.

This project addresses SDG 3 by:
1. **Raising awareness** about diabetes risk factors using an accessible web tool.
2. **Demonstrating** how technology can support health education.
3. **Practising responsible AI** — being transparent about limitations and not overstating model capabilities.

---

## Dataset

| Item | Details |
|---|---|
| **Name** | Pima Indians Diabetes Database |
| **Source** | National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), USA |
| **Hosted at** | UCI Machine Learning Repository / Kaggle |
| **URL** | https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database |
| **Licence** | CC0 1.0 Universal (Public Domain) |
| **Full size** | 768 rows × 9 columns |
| **Sample included** | 100 rows (in `data/diabetes.csv`) |
| **Target variable** | `Outcome` — 0 = no diabetes indicator, 1 = positive diabetes indicator |

### Column Descriptions

| Column | Type | Description |
|---|---|---|
| Pregnancies | Integer | Number of times pregnant |
| Glucose | Float | Plasma glucose (mg/dL) — 2-hour oral glucose tolerance test |
| BloodPressure | Float | Diastolic blood pressure (mm Hg) |
| SkinThickness | Float | Triceps skin-fold thickness (mm) |
| Insulin | Float | 2-hour serum insulin (µU/mL) |
| BMI | Float | Body Mass Index (kg/m²) |
| DiabetesPedigreeFunction | Float | Family history score (0.05–3.0) |
| Age | Integer | Age in years |
| Outcome | Integer | **Target** — 0 or 1 |

### Data Limitations

- Collected from **Pima Indian women aged 21 and above** only.
- Several columns contain zeros that represent **missing measurements** (not true zeros).
  These were replaced with column medians during preprocessing.
- Results may **not generalise** to men, children, or people from other ethnic or geographic groups.

---

## System Architecture

```
diabetes.csv  ──►  train_model.py  ──►  artifacts/
(data/)              (clean, train,       model.pkl
                      evaluate, save)     scaler.pkl
                                          metrics.pkl
                           │
                           ▼
                        app.py  ──►  Streamlit Cloud  ──►  Browser
                     (web interface)   (deployment)       (user)

                  Health_Risk_Prediction_SDG3.ipynb
                  (Google Colab — runs everything above,
                   packages ZIP, no local install needed)
```

---

## Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core programming language |
| Pandas | ≥1.5 | Data loading and cleaning |
| NumPy | ≥1.23 | Numerical operations |
| Scikit-learn | ≥1.2 | Model training and evaluation |
| Matplotlib | ≥3.6 | Charts and visualisations |
| Streamlit | ≥1.28 | Web application interface |
| Google Colab | — | Cloud notebook (no local install) |
| GitHub | — | Version control and hosting |
| Streamlit Community Cloud | — | Free web deployment |

---

## How to Run in Google Colab

**No Python installation required.** Google Colab runs everything in your browser for free.

### Steps

1. Go to **https://colab.research.google.com**
2. Click **File → Open notebook → GitHub**
3. Paste this repository URL: `https://github.com/[YOUR-GITHUB-USERNAME]/sdg3-health-risk-prediction`
4. Select `Health_Risk_Prediction_SDG3.ipynb`
5. Click **Runtime → Run all** (or press `Ctrl + F9`)
6. Wait for all cells to finish (look for ✓ checkmarks)
7. The notebook will:
   - Install all required packages
   - Load the dataset
   - Train and evaluate the model
   - Display evaluation metrics and charts
   - Package all files into `project.zip`
   - Provide a download link

### Save the notebook to your Google Drive

Click **File → Save a copy in Drive** to keep your own copy with all outputs visible.

---

## Model Training and Evaluation

### Algorithm

**Random Forest Classifier** — an ensemble method that builds many decision trees and
combines their votes. It was chosen because:
- It handles mixed feature types well.
- It is relatively robust to outliers.
- It provides feature importances.
- It performs well on tabular data without extensive tuning.

### Pipeline

1. Load dataset
2. Replace invalid zeros with column medians
3. Split: 80% training / 20% test (stratified, seed=42)
4. Scale features with `StandardScaler` — **fitted on training data only** (no data leakage)
5. Train `RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)`
6. Evaluate on held-out test set
7. Save model, scaler, and metrics as pickle files

### Why Accuracy Alone Is Insufficient

If 65% of patients in the dataset have no diabetes indicator, a model that always
predicts "No" achieves 65% accuracy — but catches zero at-risk cases.
**Recall** (the fraction of actual positive cases the model correctly identifies)
is the most critical metric for a health-screening context.

---

## Results

> **These metrics are generated by running the code. They reflect actual model performance.**
> The exact numbers will appear in your notebook output after running all cells.
> Add them here after you run the notebook.

| Metric | Score |
|---|---|
| Accuracy | _Run the notebook to see your result_ |
| Precision | _Run the notebook to see your result_ |
| Recall | _Run the notebook to see your result_ |
| F1-Score | _Run the notebook to see your result_ |

**Training seed:** 42 | **Train samples:** 80 | **Test samples:** 20 (from 100-row sample)

> **Note:** Using the full 768-row dataset from Kaggle will produce more stable and reliable metrics.
> Download instructions are in the notebook.

---

## Screenshots

> Add your screenshots here after running the project. Upload images to `docs/screenshots/`
> and reference them below.

| Screenshot | Description |
|---|---|
| `docs/screenshots/colab_output.png` | Google Colab notebook with executed cells and metrics |
| `docs/screenshots/app_home.png` | Streamlit app — home screen with input fields |
| `docs/screenshots/app_prediction.png` | Streamlit app — prediction result after clicking Predict Risk |
| `docs/screenshots/github_repo.png` | GitHub repository with all files uploaded |
| `docs/screenshots/confusion_matrix.png` | Confusion matrix chart |

---

## Deployment

### Live Application

> 🔗 **App URL:** `https://[YOUR-APP-URL].streamlit.app`
> _(Replace this with your real URL after deploying)_

### How to Deploy (Streamlit Community Cloud)

1. Push all files (including `artifacts/*.pkl`) to your GitHub repository.
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **New app**.
4. Select your repository, branch `main`, and main file `app.py`.
5. Click **Deploy**.
6. Copy the public URL and add it to this README.

---

## Disclaimer

⚠️ **This application is an educational prototype created by a student as an academic exercise.**

- It is **not a medical device** and has not been reviewed by any healthcare authority.
- It **cannot diagnose, confirm, or rule out** diabetes or any medical condition.
- Predictions are based on a small sample dataset from a **specific demographic group**
  and may not apply to other populations.
- **Do not use this tool to make any health or treatment decisions.**
- Always consult a **qualified medical professional** for health advice.

The model's output is a classification based on pattern recognition in historical data.
It is fundamentally different from a clinical assessment by a trained healthcare provider.

**Privacy:** This application does not store, log, or transmit any user-entered data.
All processing happens locally in the running application. No personal or health
information is collected.

---

## Credits and License

| Item | Details |
|---|---|
| **Student** | [Your Full Name] |
| **Institution** | [Your College / University Name] |
| **Course** | [Your Course Name] |
| **Supervisor** | [Supervisor Name, if applicable] |
| **Project type** | Academic student project |
| **Year** | 2024 |

**Third-party credits:**

- Dataset: Pima Indians Diabetes Database — NIDDK / UCI / Kaggle (CC0 licence)
- Machine learning: scikit-learn (BSD licence)
- Web framework: Streamlit (Apache 2.0 licence)

**Project licence:** MIT — see [LICENSE](LICENSE) for full terms.

---

*Built as part of an academic SDG project. Not for clinical use.*
