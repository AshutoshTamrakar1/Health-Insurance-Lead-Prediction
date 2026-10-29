# Analytics Vidhya Job-A-Thon (Feb 2021) — Health Insurance Lead Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![CatBoost](https://img.shields.io/badge/Library-CatBoost-yellow.svg)](https://catboost.ai/)
[![LightGBM](https://img.shields.io/badge/Library-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning classification project developed for the **Analytics Vidhya Job-A-Thon (February 2021)** competition. The objective is to predict customer interest in a recommended health insurance policy (lead cross-selling).

---

## 📌 Problem Statement

FinMan, a financial services company, provides various financial products such as loans and insurance to its customers. The company aims to cross-sell health insurance to its existing customers as well as new leads. 

Given customer demographic details, existing holding policy information, and policy recommendation details, the goal is to predict whether a customer will be interested in the recommended policy (`Response = 1`) or not (`Response = 0`).

- **Target Variable**: `Response` (Binary: `0` = Not Interested, `1` = Interested)
- **Evaluation Metric**: **ROC-AUC Score**
- **Class Imbalance**: ~76% Class 0 vs. ~24% Class 1

---

## 📊 Dataset Overview

The dataset contains over 50,000 training records:

| Dataset | Rows | Columns | Description |
|---|---|---|---|
| **Train Set** | 50,882 | 14 | Historical customer profile and policy responses |
| **Test Set** | 21,805 | 13 | Customers for whom `Response` probability must be predicted |

### Data Dictionary

| Feature | Description | Data Type | Notes |
|---|---|---|---|
| `ID` | Unique customer identifier | Integer | Excluded from modeling |
| `City_Code` | Code identifier for the customer's city | Categorical (`C1`–`C36`) | 36 unique cities |
| `Region_Code` | Specific region code within city | Numeric / Categorical | High cardinality location feature |
| `Accomodation_Type` | Housing status (`Owned` vs. `Rented`) | Categorical | Most customers own their home |
| `Reco_Insurance_Type` | Recommended insurance type (`Individual` vs. `Joint`) | Categorical | Individual is predominant (~80%) |
| `Upper_Age` | Maximum age of insured customer(s) | Numeric | Age range: 18 – 75 |
| `Lower_Age` | Minimum age of insured customer(s) | Numeric | Equals `Upper_Age` for individual policies |
| `Is_Spouse` | Indicates whether customer is married to nominee | Binary (`Yes` / `No`) | Mostly `No` (~83%) |
| `Health Indicator` | Health rating category (`X1` to `X9`) | Categorical | ~23% missing values |
| `Holding_Policy_Duration` | Duration of previous health policy held | Mixed / String | Values `1.0` to `14.0`, plus `'14+'`; ~39.8% missing |
| `Holding_Policy_Type` | Policy category of previous health policy | Float (`1.0`–`4.0`) | ~39.8% missing |
| `Reco_Policy_Cat` | Category of recommended policy | Integer (`1`–`22`) | Segmented policy offerings |
| `Reco_Policy_Premium` | Annual premium of recommended policy (INR) | Continuous | Premium amounts |
| `Response` *(Target)* | Customer interest in recommendation | Binary (`0` or `1`) | 76% `0`, 24% `1` (Imbalanced) |

---

## 🏆 Model Benchmarks & Results Comparison

| Model | Validation Strategy | Metric (ROC-AUC) | Key Notes |
|---|---|:---:|---|
| **Original 2021 Random Forest** | Train/Validation Split (80/20) | **~0.6115** *(~0.50 on binary)* | `Holding_Policy_Duration` dropped; evaluated on accuracy; hard `predict` predicted only 60 positives out of 21,805 |
| **Corrected Random Forest** | Stratified 5-Fold CV | **0.6582** | Fixed `'14+'` drop; continuous `predict_proba` |
| **LightGBM (Baseline)** | Stratified 5-Fold CV | **0.6910** | Gradient boosting with one-hot & frequency features |
| **CatBoost (Modern Optimized)** | **Stratified 5-Fold CV** | **0.8005** 🚀 | Native target-rate encoding on high-cardinality features (`Region_Code`, `City_Code`); engineered age gap and tenure indicators |

> **Net Performance Gain**: **+18.90 percentage points** improvement in ROC-AUC over the original baseline.

---

## 🛠️ Retrospective Analysis & Key Technical Fixes

### 1. Recovering the Dropped Feature (`Holding_Policy_Duration`)
* **Problem**: In `Holding_Policy_Duration`, customers with >14 years tenure had values entered as the string `'14+'`. Replacing `NaN` with `0` left the column as an `object` data type. Calling `train.select_dtypes(exclude='object')` silently removed the column entirely.
* **Fix**: Parsed `'14+'` to `15.0` and cast to numeric float so policy tenure signal is preserved.

### 2. Metric Misalignment (Accuracy vs. ROC-AUC)
* **Problem**: The original baseline tracked classification accuracy (~75.7%). Since 76% of all records belong to Class 0, a model predicting all zeros scores 76% accuracy without learning patterns.
* **Fix**: Evaluated all iterations using **Stratified 5-Fold Cross Validation** strictly on **ROC-AUC Score**.

### 3. Probability Calibration (`predict_proba` vs. `predict`)
* **Problem**: Calling `model.predict(test)` with default 0.5 decision thresholds resulted in 21,745 zeros and only 60 ones predicted in the test set.
* **Fix**: Generated test predictions using continuous probabilities (`predict_proba[:, 1]`) averaged across 5 folds.

### 4. High-Cardinality Categorical Encoding
* **Problem**: Fitting `LabelEncoder` independently on train and test introduced data drift risk, and dummy one-hot encoding on high-cardinality `Region_Code` exploded dimensionality.
* **Fix**: Leveraged **CatBoost**'s ordered target statistics to handle high-cardinality categorical interactions natively without data leakage.

### 5. High-Impact Feature Engineering
* `Age_Gap = Upper_Age - Lower_Age` (distinguishes individual plans from joint plans with significant partner age gaps).
* `Is_New_Customer = Holding_Policy_Type.isna().astype(int)` (explicit signal for customers with no holding history).
* `Premium_Per_Age = Reco_Policy_Premium / (Upper_Age + 1)`.
* `Health Indicator` treated with an explicit `'Missing'` category instead of mode imputation.

---

## 📂 Repository Structure

```text
├── Job-A-Thon.ipynb      # Complete, executed end-to-end modernized notebook
├── Approach.docx         # Original competition approach document
├── .gitignore            # Excludes datasets and generated artifacts
└── README.md             # Project documentation and benchmark report
```

> **Note**: Datasets (`train_Df64byy.csv`, `test_YCcRUnU.csv`) and output submission files are excluded from git per data privacy and repository hygiene practices.
