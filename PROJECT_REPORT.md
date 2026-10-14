# 📄 Project Report
## Health Risk Prediction & Awareness System
### SDG 3 – Good Health and Well-being

---

| Field | Details |
|---|---|
| **Student Name** | [Your Full Name] |
| **Student ID** | [Your Student ID] |
| **Institution** | [Your College / University Name] |
| **Course** | [Course Name and Code] |
| **Supervisor** | [Supervisor Name, if applicable — leave blank if none] |
| **Submission Date** | [Date] |
| **Project Type** | Academic SDG student project |

> **Replace all bracketed placeholders above with your real details before submitting.**

---

## 1. Executive Summary

This report documents the design, development, and evaluation of a machine-learning web
application built as a student academic project aligned with **United Nations Sustainable
Development Goal 3 (SDG 3) — Good Health and Well-being**.

The application uses the Pima Indians Diabetes Database to demonstrate how health measurements
can be used in a machine-learning pipeline to classify patterns associated with diabetes risk.
A Random Forest classifier was trained, evaluated, and deployed as an interactive Streamlit
web application hosted on Streamlit Community Cloud.

The project is explicitly an educational prototype. It does not diagnose, treat, or screen
for diabetes in real patients, and has not been evaluated in a clinical setting.

---

## 2. Problem Statement

Diabetes is one of the fastest-growing non-communicable diseases globally, affecting over
537 million adults as of 2021 (International Diabetes Federation). Many cases remain
undiagnosed due to lack of awareness and limited access to health information.

SDG 3 specifically targets (3.4) the reduction of premature mortality from NCDs and (3.d)
strengthening capacities for health risk awareness. Technology-based tools that communicate
risk factors in an accessible, transparent way can support this goal — provided they are
honest about their limitations.

**This project explores the question:** Can a simple machine-learning model, built from
publicly available health data, be used to create an educational awareness tool that
communicates both predictions and their limitations responsibly?

---

## 3. Objectives

| # | Objective | Status |
|---|---|---|
| 1 | Build a complete ML pipeline: data loading, cleaning, training, evaluation, deployment | Completed |
| 2 | Train a classifier on the Pima Indians Diabetes dataset | Completed |
| 3 | Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix | Completed |
| 4 | Deploy an interactive web app using Streamlit | Completed |
| 5 | Document data sources, cleaning decisions, and model limitations transparently | Completed |
| 6 | Align the project with SDG 3 | Completed |
| 7 | Provide a reproducible, version-controlled codebase on GitHub | Completed |

---

## 4. Dataset Description

**Name:** Pima Indians Diabetes Database
**Source:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), USA
**Repository:** UCI Machine Learning Repository / Kaggle
**URL:** https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
**Licence:** CC0 1.0 Universal (Public Domain)

### 4.1 Population

The dataset was collected from female patients of Pima Indian heritage, aged 21 years and
above, living near Phoenix, Arizona, USA. This is a population group known to have a
genetically elevated risk of type 2 diabetes.

**Population Limitation:** Because the dataset represents a specific and narrow demographic,
predictions derived from a model trained on it may not apply to:
- Men
- Children under 21
- People from other ethnic backgrounds
- People from different geographic regions

### 4.2 Features

| Column | Type | Range (approx.) | Notes |
|---|---|---|---|
| Pregnancies | Integer | 0–17 | Number of pregnancies |
| Glucose | Float | 44–199 | Plasma glucose (mg/dL) |
| BloodPressure | Float | 24–122 | Diastolic BP (mm Hg) |
| SkinThickness | Float | 7–99 | Triceps skin-fold (mm) |
| Insulin | Float | 14–846 | Serum insulin (µU/mL) |
| BMI | Float | 18–67 | Body Mass Index |
| DiabetesPedigreeFunction | Float | 0.078–2.42 | Family history score |
| Age | Integer | 21–81 | Age in years |
| Outcome | Integer | 0 or 1 | **Target variable** |

### 4.3 Class Distribution (Full Dataset)

| Class | Count | Percentage |
|---|---|---|
| 0 (no indicator) | 500 | 65.1% |
| 1 (positive indicator) | 268 | 34.9% |

The dataset is mildly imbalanced. The model used `class_weight='balanced'` to address this.

### 4.4 Data Cleaning

Several columns contained zeros that are biologically impossible (a person cannot have 0
glucose, 0 blood pressure, or 0 BMI). These zeros represent missing data entries.

**Decision:** Replace each zero in the affected columns with the **median** of the non-zero
values in that column. The median was chosen over the mean because it is more robust to
extreme values (outliers).

Affected columns: Glucose, BloodPressure, SkinThickness, Insulin, BMI.

The `Pregnancies` column legitimately contains zeros (a patient may never have been pregnant)
and was not cleaned.

---

## 5. Methodology

### 5.1 Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.10 |
| Data processing | Pandas, NumPy |
| Machine learning | scikit-learn |
| Visualisation | Matplotlib |
| Web interface | Streamlit |
| Version control | Git / GitHub |
| Cloud notebook | Google Colab |
| Deployment | Streamlit Community Cloud |

### 5.2 Machine Learning Pipeline

```
Raw CSV
  │
  ▼
Load with Pandas
  │
  ▼
Replace invalid zeros with column medians
  │
  ▼
Select 8 feature columns + target column
  │
  ▼
Train/test split (80/20, stratified, seed=42)
  │
  ├── Training set ──► StandardScaler.fit_transform() ──► RandomForestClassifier.fit()
  │                           │
  └── Test set    ──► StandardScaler.transform()  ──► model.predict() ──► Evaluate
```

### 5.3 Data Leakage Prevention

Data leakage occurs when information from the test set influences the training process,
producing artificially optimistic evaluation results. This project prevents leakage by:

1. Fitting the `StandardScaler` **only on training data** (`fit_transform` on X_train).
2. Applying the **same scaler** to test data without refitting (`transform` on X_test).
3. Performing the train/test split **before** any scaling.

### 5.4 Algorithm Selection: Random Forest

A **Random Forest classifier** was chosen as the baseline model for this project because:

- It performs well on tabular data with mixed feature types.
- It requires minimal hyperparameter tuning for a baseline.
- It provides feature importance scores.
- It handles mild class imbalance well with `class_weight='balanced'`.
- It is interpretable enough for an educational context.

**Hyperparameters used:**
- `n_estimators=100` (100 decision trees)
- `class_weight='balanced'` (adjusts for class imbalance)
- `random_state=42` (reproducibility)

### 5.5 Reproducibility

The following fixed values ensure identical results across runs:
- Random seed: `42` (applied to `train_test_split` and `RandomForestClassifier`)
- Package versions recorded in `requirements.txt`
- Seed and package version saved in `artifacts/metrics.pkl`

---

## 6. Results

> **Note:** The metrics below are placeholders. Replace them with your actual results
> after running the notebook. The code generates these numbers — do not invent them.

| Metric | Score | Interpretation |
|---|---|---|
| Accuracy | _[run notebook]_ | Overall fraction of correct predictions |
| Precision | _[run notebook]_ | Of predicted positives, fraction truly positive |
| Recall | _[run notebook]_ | Of actual positives, fraction correctly identified |
| F1-Score | _[run notebook]_ | Harmonic mean of Precision and Recall |

**Confusion Matrix:** (replace with your actual numbers)

|  | Predicted No | Predicted Yes |
|---|---|---|
| **Actual No** | _TN_ | _FP_ |
| **Actual Yes** | _FN_ | _TP_ |

### 6.1 Why Accuracy Alone Is Insufficient

Consider a model that always predicts "No indicator" for every patient. On this dataset
(~65% negative), it would achieve approximately 65% accuracy — but it would miss every
single at-risk patient. In a health-screening context:

- **False Negatives (missed cases)** are more costly than **False Positives (false alarms)**.
- **Recall** (sensitivity) measures the model's ability to catch true positive cases.
- A high Recall is more desirable for screening tools even if it comes at the cost of lower Precision.
- **F1-Score** provides a balanced summary when both Precision and Recall matter.

### 6.2 Feature Importance

(Add the `docs/evaluation/feature_importance.png` chart here after running the notebook.)

The Random Forest model assigns an importance score to each feature based on how much it
reduces impurity across all decision trees. Higher importance = stronger predictor in this model.

Expected result: Glucose and BMI typically rank as the most important features in this dataset.

---

## 7. Streamlit Application

The web application (`app.py`) provides:

| Section | Content |
|---|---|
| Header | Project title and SDG 3 explanation |
| Disclaimer | Prominent warning — educational tool only |
| Privacy Note | Explains no data is stored or transmitted |
| Input Fields | 8 health measurements with help tooltips |
| Predict Risk button | Triggers the model prediction |
| Result | Classification, probability, interpretation |
| Awareness Info | General health guidance (WHO/CDC based) |
| Model Metrics | Accuracy, Precision, Recall, F1, Confusion Matrix |
| Methodology | Data source, cleaning decisions, model architecture |
| Limitations | 6 documented caveats |

---

## 8. Ethical Considerations

### 8.1 Responsible AI Principles Applied

| Principle | How This Project Addresses It |
|---|---|
| Transparency | Data source, cleaning decisions, and model architecture are fully documented |
| Honesty | Metrics are generated by code, not manually selected |
| Limitations | A dedicated section discloses all known limitations |
| Non-maleficence | The app prominently states it is not a diagnostic tool |
| Fairness | Population limitations are disclosed; users are warned that results may not apply to them |
| Privacy | No user data is stored or transmitted |

### 8.2 What This Model Cannot Do

- Diagnose diabetes.
- Replace a clinical assessment.
- Guarantee accurate predictions for any individual.
- Generalise reliably to populations outside the training dataset's demographic.

### 8.3 Dataset Ethics

The dataset is publicly available under CC0 licence. The NIDDK collected it for research
purposes. The population studied (Pima Indian heritage, female, aged 21+) is a real
community. Any tool built from this data must acknowledge that its predictions apply
most reliably to individuals similar to those in the dataset.

---

## 9. Limitations

1. **Small sample:** The included 100-row sample is adequate for demonstration but
   insufficient for a reliable production model. The full 768-row dataset should be used.

2. **Population specificity:** The model was trained on a narrow demographic.

3. **Binary output:** Diabetes risk exists on a spectrum. A binary Yes/No classification
   oversimplifies a complex medical reality.

4. **No clinical validation:** The model has not been tested with real patients or reviewed
   by medical professionals.

5. **Missing data imputation:** Replacing zeros with medians is a reasonable but imperfect
   approach. More sophisticated imputation methods exist.

6. **Feature selection:** The model uses all 8 available features. A medical practitioner
   might identify a different set of clinically meaningful features.

---

## 10. Future Improvements

| Improvement | Benefit |
|---|---|
| Use full 768-row dataset | More reliable model performance |
| Add cross-validation | More robust evaluation than a single split |
| Hyperparameter tuning (GridSearchCV) | Potentially better model performance |
| Try Logistic Regression as comparison | Understand which algorithm suits this data better |
| Add SHAP explanations | Show why the model made a specific prediction |
| Multilingual interface | Improve accessibility |
| Link to real health screening services | Direct users to professional help |

---

## 11. Personal Reflection

> Write 2–4 sentences in your own words about what you learned from this project.
> Example: "This project taught me how a machine-learning pipeline works from data loading
> to deployment. I learned that accuracy alone is not a good measure for health models,
> and that documenting limitations is as important as building the model itself."

[Write your own reflection here.]

---

## 12. References

1. International Diabetes Federation (2021). *IDF Diabetes Atlas, 10th edition*.
   https://www.diabetesatlas.org

2. Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988).
   Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.
   *Proceedings of the Symposium on Computer Applications and Medical Care*, 261–265.

3. UCI Machine Learning Repository — Pima Indians Diabetes Database.
   https://archive.ics.uci.edu/dataset/34/diabetes

4. Kaggle Dataset (CC0 licence):
   https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

5. World Health Organization. *Diabetes Fact Sheet*.
   https://www.who.int/news-room/fact-sheets/detail/diabetes

6. scikit-learn documentation. https://scikit-learn.org/stable/

7. Streamlit documentation. https://docs.streamlit.io

---

*This report was prepared as part of an academic SDG student project.
It represents the student's own work and is not a peer-reviewed research paper.*
