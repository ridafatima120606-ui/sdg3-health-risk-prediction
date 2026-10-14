# 🧪 Testing Documentation
## Health Risk Prediction & Awareness System

> **Status Key:**
> - ✅ PASS — Test ran and produced the expected result.
> - ❌ FAIL — Test ran and produced an unexpected result.
> - ⏳ PENDING — Test needs to be run manually. Instructions provided.
> - ℹ️ MANUAL — This test requires human inspection, not automated output.

All automated tests in this file are run by the Google Colab notebook
(`Health_Risk_Prediction_SDG3.ipynb`). The PENDING tests require you to run
the notebook and record the actual outputs.

---

## Section 1 — Environment and Setup Tests

### TEST-001: Package Installation

| Field | Value |
|---|---|
| Test ID | TEST-001 |
| Description | Verify all required packages can be installed |
| How to run | Colab notebook Cell 1: `!pip install -r requirements.txt` |
| Expected output | No error messages; packages listed in `requirements.txt` installed |
| Status | ⏳ PENDING — Run notebook and update |
| Actual output | _[Paste the last line of pip output here, e.g., "Successfully installed streamlit-1.xx"]_ |

---

### TEST-002: Package Import Check

| Field | Value |
|---|---|
| Test ID | TEST-002 |
| Description | Verify all required packages can be imported |
| How to run | Colab notebook Cell 2: import block |
| Expected output | All imports succeed with no ImportError |
| Status | ⏳ PENDING |
| Actual output | _[Note any errors or "All imports OK"]_ |

---

## Section 2 — Data Loading and Cleaning Tests

### TEST-003: Dataset Loading

| Field | Value |
|---|---|
| Test ID | TEST-003 |
| Description | Dataset loads successfully with correct shape |
| Expected output | `(100, 9)` or `(768, 9)` for full dataset |
| Status | ⏳ PENDING |
| Actual output | _[e.g., "Loaded 100 rows × 9 columns"]_ |

---

### TEST-004: Invalid Zero Detection

| Field | Value |
|---|---|
| Test ID | TEST-004 |
| Description | Zero counts in Glucose, BloodPressure, SkinThickness, Insulin, BMI are detected |
| Expected output | At least one zero detected in at least one of these columns |
| Status | ⏳ PENDING |
| Actual output | _[Paste zero-count output from train_model.py]_ |

---

### TEST-005: Zero Replacement Correctness

| Field | Value |
|---|---|
| Test ID | TEST-005 |
| Description | After cleaning, no zeros remain in the invalid-zero columns |
| How to verify | After running `train_model.py`, check: `(df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] == 0).sum()` |
| Expected output | All counts = 0 |
| Status | ⏳ PENDING |
| Actual output | _[Paste output here]_ |

---

### TEST-006: Class Distribution Check

| Field | Value |
|---|---|
| Test ID | TEST-006 |
| Description | Both classes (0 and 1) are present in the dataset |
| Expected output | Both class 0 and class 1 appear in the Outcome column |
| Status | ⏳ PENDING |
| Actual output | _[e.g., "Class 0: 65, Class 1: 35"]_ |

---

## Section 3 — Model Training Tests

### TEST-007: Train/Test Split Proportions

| Field | Value |
|---|---|
| Test ID | TEST-007 |
| Description | Training set is ~80% and test set is ~20% of the total data |
| Expected output | With 100 rows: 80 train, 20 test |
| Status | ⏳ PENDING |
| Actual output | _[e.g., "Training: 80, Test: 20"]_ |

---

### TEST-008: No Data Leakage — Scaler Fit Only on Training Data

| Field | Value |
|---|---|
| Test ID | TEST-008 |
| Description | StandardScaler is fitted ONLY on training data |
| How to verify | Code review — `scaler.fit_transform(X_train)`, `scaler.transform(X_test)` |
| Expected output | Code uses `fit_transform` on train, `transform` on test only |
| Status | ✅ PASS — Verified by code review of train_model.py |
| Notes | See `train_model.py` lines for scaler usage |

---

### TEST-009: Model Training Completes

| Field | Value |
|---|---|
| Test ID | TEST-009 |
| Description | RandomForestClassifier trains without error |
| Expected output | "Training complete." printed with no exceptions |
| Status | ⏳ PENDING |
| Actual output | _[Note: "Training complete" or any error]_ |

---

### TEST-010: Model Artifacts Saved

| Field | Value |
|---|---|
| Test ID | TEST-010 |
| Description | model.pkl, scaler.pkl, metrics.pkl exist after training |
| How to verify | After running `train_model.py`, check `artifacts/` folder |
| Expected output | 3 `.pkl` files present |
| Status | ⏳ PENDING |
| Actual output | _[List actual files found]_ |

---

## Section 4 — Model Evaluation Tests

### TEST-011: Accuracy Is Non-Zero and Reasonable

| Field | Value |
|---|---|
| Test ID | TEST-011 |
| Description | Model accuracy is between 0.50 and 1.00 on the test set |
| Expected output | Accuracy > 0.50 (random guessing = 0.50) |
| Status | ⏳ PENDING |
| Actual output | _[e.g., "Accuracy: 0.75"]_ |

---

### TEST-012: All Four Metrics Are Reported

| Field | Value |
|---|---|
| Test ID | TEST-012 |
| Description | Accuracy, Precision, Recall, and F1-Score are all computed and printed |
| Expected output | All four values appear in the terminal/notebook output |
| Status | ⏳ PENDING |
| Actual output | _[Paste metric output]_ |

---

### TEST-013: Confusion Matrix Has Correct Shape

| Field | Value |
|---|---|
| Test ID | TEST-013 |
| Description | Confusion matrix is a 2×2 array |
| Expected output | `cm.shape == (2, 2)` |
| Status | ⏳ PENDING |
| Actual output | _[Note shape]_ |

---

## Section 5 — Prediction Tests

### TEST-014: Valid Input — Expected High-Risk Result

| Field | Value |
|---|---|
| Test ID | TEST-014 |
| Description | Known high-risk input produces Outcome prediction = 1 |
| Input values | Pregnancies=6, Glucose=148, BP=72, Skin=35, Insulin=80, BMI=33.6, DPF=0.627, Age=50 |
| Expected output | Prediction = 1 (higher-risk pattern) |
| Status | ⏳ PENDING — Run the Streamlit app with these values |
| Actual output | _[Note: "Higher-Risk" or "Lower-Risk" and probability]_ |

---

### TEST-015: Valid Input — Expected Low-Risk Result

| Field | Value |
|---|---|
| Test ID | TEST-015 |
| Description | Known lower-risk input produces Outcome prediction = 0 |
| Input values | Pregnancies=1, Glucose=85, BP=66, Skin=29, Insulin=80, BMI=26.6, DPF=0.351, Age=31 |
| Expected output | Prediction = 0 (lower-risk pattern) |
| Status | ⏳ PENDING |
| Actual output | _[Note result]_ |

---

### TEST-016: Minimum Permitted Inputs

| Field | Value |
|---|---|
| Test ID | TEST-016 |
| Description | App accepts minimum allowed values without error |
| Input values | Pregnancies=0, Glucose=50, BP=30, Skin=5, Insulin=1, BMI=10.0, DPF=0.05, Age=18 |
| Expected output | App returns a prediction without error or crash |
| Status | ⏳ PENDING |
| Actual output | _[Note result]_ |

---

### TEST-017: Maximum Permitted Inputs

| Field | Value |
|---|---|
| Test ID | TEST-017 |
| Description | App accepts maximum allowed values without error |
| Input values | Pregnancies=20, Glucose=300, BP=150, Skin=100, Insulin=900, BMI=70.0, DPF=3.0, Age=100 |
| Expected output | App returns a prediction without error or crash |
| Status | ⏳ PENDING |
| Actual output | _[Note result]_ |

---

### TEST-018: Probability Values Are Valid

| Field | Value |
|---|---|
| Test ID | TEST-018 |
| Description | Both class probabilities sum to 100% |
| Expected output | prob_no + prob_yes ≈ 100% |
| Status | ✅ PASS — Verified by code review; `predict_proba` always sums to 1.0 |

---

## Section 6 — Application Tests

### TEST-019: App Starts Without Error

| Field | Value |
|---|---|
| Test ID | TEST-019 |
| Description | `streamlit run app.py` starts without Python exceptions |
| How to run | Run the Streamlit launch cell in the Colab notebook |
| Expected output | Streamlit URL appears; no tracebacks |
| Status | ⏳ PENDING |
| Actual output | _[Note URL or any errors]_ |

---

### TEST-020: App Shows Error If Model Not Trained

| Field | Value |
|---|---|
| Test ID | TEST-020 |
| Description | If model.pkl is missing, the app shows a helpful error message |
| Expected output | Orange warning box with instructions to run train_model.py |
| Status | ✅ PASS — Verified by code review; `st.stop()` is called if files missing |

---

### TEST-021: Disclaimer Is Visible

| Field | Value |
|---|---|
| Test ID | TEST-021 |
| Description | The disclaimer warning box is visible on page load (before any interaction) |
| Expected output | Red error box with disclaimer text visible on the main page |
| Status | ℹ️ MANUAL — Verify visually in the running app |

---

### TEST-022: Privacy Note Is Accessible

| Field | Value |
|---|---|
| Test ID | TEST-022 |
| Description | Privacy note is accessible in the expandable section |
| Expected output | "Privacy Note" expander exists; content explains no data is stored |
| Status | ℹ️ MANUAL — Expand the section in the running app |

---

## Section 7 — Packaging Tests

### TEST-023: ZIP Package Created

| Field | Value |
|---|---|
| Test ID | TEST-023 |
| Description | Colab notebook creates `project.zip` with all required files |
| Expected output | ZIP file created; download link appears in notebook |
| Status | ⏳ PENDING |
| Actual output | _[Note file size, e.g., "project.zip — 2.3 MB"]_ |

---

### TEST-024: ZIP Contains Required Files

| Field | Value |
|---|---|
| Test ID | TEST-024 |
| Description | ZIP contains all 9 required project files |
| Required files | app.py, train_model.py, model_evaluation.py, requirements.txt, README.md, .gitignore, LICENSE, PROJECT_REPORT.md, TESTING.md |
| Status | ⏳ PENDING |
| Actual output | _[List files found in ZIP]_ |

---

## Summary Table

| Test ID | Category | Status |
|---|---|---|
| TEST-001 | Setup | ⏳ |
| TEST-002 | Setup | ⏳ |
| TEST-003 | Data | ⏳ |
| TEST-004 | Data | ⏳ |
| TEST-005 | Data | ⏳ |
| TEST-006 | Data | ⏳ |
| TEST-007 | Training | ⏳ |
| TEST-008 | Training | ✅ |
| TEST-009 | Training | ⏳ |
| TEST-010 | Training | ⏳ |
| TEST-011 | Evaluation | ⏳ |
| TEST-012 | Evaluation | ⏳ |
| TEST-013 | Evaluation | ⏳ |
| TEST-014 | Prediction | ⏳ |
| TEST-015 | Prediction | ⏳ |
| TEST-016 | Prediction | ⏳ |
| TEST-017 | Prediction | ⏳ |
| TEST-018 | Prediction | ✅ |
| TEST-019 | App | ⏳ |
| TEST-020 | App | ✅ |
| TEST-021 | App | ℹ️ |
| TEST-022 | App | ℹ️ |
| TEST-023 | Packaging | ⏳ |
| TEST-024 | Packaging | ⏳ |

> **Instructions:** After running the Colab notebook and the Streamlit app, return to this
> file and update each ⏳ PENDING test with its actual output and mark it ✅ PASS or ❌ FAIL.
> This completed document is part of your project evidence.
