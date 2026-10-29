# Analytics Vidhya Job-A-Thon (Feb 2021) — Health Insurance Lead Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning classification project developed for the **Analytics Vidhya Job-A-Thon (February 2021)** competition. The objective is to predict customer interest in a recommended health insurance policy (lead cross-selling).

---

## 📌 Problem Statement

FinMan, a financial services company, provides various financial products such as loans and insurance to its customers. The company aims to cross-sell health insurance to its existing customers as well as new leads. 

Given customer demographic details, existing holding policy information, and policy recommendation details, the goal is to predict whether a customer will be interested in the recommended policy (`Response = 1`) or not (`Response = 0`).

- **Target Variable**: `Response` (Binary: `0` = Not Interested, `1` = Interested)
- **Evaluation Metric**: **ROC-AUC Score**

---

## 📊 Dataset Overview

The dataset contains over 50,000 training records with an imbalanced target distribution (~76% Class 0 vs. ~24% Class 1).

| Dataset | Rows | Columns | Description |
|---|---|---|---|
| **Train Set** | 50,882 | 14 | Historical customer profile and policy responses |
| **Test Set** | 21,805 | 13 | Customers for whom `Response` must be predicted |

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

## 🔍 Exploratory Data Analysis & Key Insights

1. **Class Imbalance**: Around 24% of customers accepted the recommended policy. Models evaluated strictly on raw accuracy risk predicting majority class (`0`) blindly.
2. **Missing Existing Policy Data**: Both `Holding_Policy_Duration` and `Holding_Policy_Type` share the exact same missing percentage (~39.8%). These missing values directly signal **New Customers** who do not currently hold a policy with the firm.
3. **Health Indicator**: ~23% missing data; distribution is heavily skewed toward `X1` and `X2` (better health ratings).
4. **Age Distribution**: `Upper_Age` and `Lower_Age` are identical for individual plans. Their divergence signifies joint/family plans where age gap analysis can serve as a strong predictive signal.

---

## 🛠️ Retrospective Analysis & Lessons Learned

Revisiting this project highlights valuable lessons in machine learning engineering:

### 1. The Silently Dropped Feature (`Holding_Policy_Duration`)
- In `Holding_Policy_Duration`, the duration was stored as mixed string types because tenure exceeding 14 years was encoded as `'14+'`.
- Replacing `NaN` with `0` left the column as an `object` data type.
- During preprocessing, `train.select_dtypes(exclude='object')` was called, which **silently dropped `Holding_Policy_Duration` completely** from the training pipeline.
- *Remedy*: Convert `'14+'` to a numeric value (`15.0`) and cast to `float` so tenure information is retained.

### 2. Metric Misalignment (Accuracy vs. ROC-AUC)
- The initial baseline reported **100% training accuracy** and **~75.7% validation accuracy**.
- Since 76% of all responses are `0`, a trivial baseline predicting all zeros scores 76% accuracy without learning patterns. The 100% training score indicated unconstrained tree overfitting.
- Because the competition metric was **ROC-AUC**, models must optimize rank discrimination rather than arbitrary 0.5 classification threshold.

### 3. Binary Predictions vs. Predicted Probabilities
- Calling `RF.predict(test)` with default 0.5 cutoffs on an imbalanced dataset caused the classifier to predict only **60 positive responses out of 21,805 test cases** (99.7% zeros).
- For ROC-AUC competitions, submissions should use **predicted probabilities** (`predict_proba(X_test)[:, 1]`).

### 4. Categorical Encoding Pipeline
- Running `LabelEncoder()` separately on train and test data introduces label mismatch risk if categories are ordered or distributed differently.
- Calling `pd.get_dummies` after label encoding created redundant integer-named columns. A unified `OneHotEncoder` or native categorical handling (e.g. LightGBM / CatBoost) avoids feature space drift.

---

## 🚀 Recommended Modern Roadmap

If modernizing or iterating further on this benchmark:

1. **Feature Engineering**:
   - `Age_Gap = Upper_Age - Lower_Age` (signals couples with larger age differences).
   - `Is_New_Customer = Holding_Policy_Type.isna().astype(int)`.
   - `Premium_Per_Age = Reco_Policy_Premium / Upper_Age`.
   - Frequency / Target Encoding for high-cardinality `City_Code` and `Region_Code`.
2. **Modeling**:
   - Replace unregularized Random Forest with **LightGBM**, **XGBoost**, or **CatBoost**.
   - Use `scale_pos_weight` or class weighting to handle target imbalance.
3. **Validation Strategy**:
   - Implement **Stratified 5-Fold Cross Validation** evaluating on `roc_auc_score`.
4. **Outputs**:
   - Generate test probabilities using `predict_proba` averaged across out-of-fold models.

---

## 📂 Repository Structure

```text
├── Job-A-Thon.ipynb      # Main analysis and baseline modeling notebook
├── Approach.docx         # Original competition approach document
├── .gitignore            # Excludes datasets and generated artifacts
└── README.md             # Project documentation and retrospective review
```

> **Note**: Datasets (`train_Df64byy.csv`, `test_YCcRUnU.csv`) and output submission files are excluded from this repository per data privacy and repository hygiene practices.
