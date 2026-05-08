"""
handover.py
===========
Phase 5 – The Handover
Trains the final champion model, exports it as .joblib,
and produces a complete JSON schema for Person B / the API team.

Outputs
-------
  model/champion_xgb.joblib   – production-ready model artifact
  model/preprocessor.joblib   – fitted preprocessing pipeline
  model/feature_names.json    – ordered list of feature columns
  model/input_schema.json     – full API schema with types, ranges & examples
  model/sample_request.json   – copy-paste ready single API request body
  model/sample_batch.json     – copy-paste ready batch API request body
  model/model_card.md         – human-readable model card for the handover doc

Usage
-----
    python handover.py
"""

import warnings
warnings.filterwarnings("ignore")

import os, json, joblib, datetime
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, classification_report
)

# ─────────────────────────────────────────────────────────────
# 0. Config
# ─────────────────────────────────────────────────────────────
DATA_PATH    = "credit_risk_dataset.csv"
OUT_DIR      = "model"
RANDOM_STATE = 42
TEST_SIZE    = 0.20

# Best params found by Optuna (Phase 4)
BEST_PARAMS = {
    "n_estimators"     : 600,
    "max_depth"        : 5,
    "learning_rate"    : 0.06774593731668856,
    "subsample"        : 0.7951977427742329,
    "colsample_bytree" : 0.5622693192496626,
    "min_child_weight" : 4,
    "gamma"            : 0.6580168264470879,
    "reg_alpha"        : 0.1681308748581626,
    "reg_lambda"       : 0.5769713635627621,
}

os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1. Preprocessing  (returns fitted encoder map too)
# ─────────────────────────────────────────────────────────────
class CreditRiskPreprocessor:
    """Stateful preprocessor — fit once, transform many times."""

    CAT_COLS = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
    ]

    def fit(self, df: pd.DataFrame):
        self.medians_ = {
            "person_age"       : df.loc[df.person_age <= 100, "person_age"].median(),
            "person_emp_length": df.loc[df.person_emp_length <= 60,
                                        "person_emp_length"].median(),
        }
        tmp = df.copy()
        tmp.loc[tmp.person_emp_length > 60, "person_emp_length"] = np.nan
        tmp["person_emp_length"].fillna(self.medians_["person_emp_length"], inplace=True)

        self.int_rate_medians_ = (
            tmp.groupby("loan_grade")["loan_int_rate"].median().to_dict()
        )
        self.global_int_rate_median_ = tmp["loan_int_rate"].median()

        self.label_encoders_ = {}
        for col in self.CAT_COLS:
            le = LabelEncoder()
            le.fit(df[col].str.strip().str.upper())
            self.label_encoders_[col] = le
            # store classes for schema documentation
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Fix outliers
        df.loc[df.person_age > 100, "person_age"] = self.medians_["person_age"]
        df.loc[df.person_emp_length > 60, "person_emp_length"] = np.nan
        df["person_emp_length"].fillna(self.medians_["person_emp_length"], inplace=True)

        # Impute interest rate by grade
        def fill_rate(row):
            if pd.isna(row["loan_int_rate"]):
                return self.int_rate_medians_.get(row["loan_grade"],
                                                  self.global_int_rate_median_)
            return row["loan_int_rate"]
        df["loan_int_rate"] = df.apply(fill_rate, axis=1)

        # Encode categoricals
        for col in self.CAT_COLS:
            df[col] = df[col].str.strip().str.upper()
            df[col] = self.label_encoders_[col].transform(df[col])

        # Feature engineering
        df["loan_to_income"]    = df["loan_amnt"] / (df["person_income"] + 1)
        df["income_per_emp_yr"] = df["person_income"] / (df["person_emp_length"] + 1)
        df["risk_score"]        = df["loan_int_rate"] * df["loan_percent_income"]

        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)


# ─────────────────────────────────────────────────────────────
# 2. Load, preprocess, split
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  PHASE 5 — MODEL HANDOVER")
    print("=" * 60)

    print("\n[1/5] Loading & preprocessing …")
    df_raw = pd.read_csv(DATA_PATH)

    prep = CreditRiskPreprocessor()
    df   = prep.fit_transform(df_raw.drop(columns=["loan_status"]))
    y    = df_raw["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    spw = round((y_train == 0).sum() / (y_train == 1).sum(), 4)
    feature_names = df.columns.tolist()
    print(f"      Features : {len(feature_names)}")
    print(f"      Train    : {len(X_train):,}   Test: {len(X_test):,}")
    print(f"      SPW      : {spw}")

    # ── Train champion ─────────────────────────────────────────
    print("\n[2/5] Training Optuna champion model …")
    champion = XGBClassifier(
        **BEST_PARAMS,
        scale_pos_weight  = spw,
        use_label_encoder = False,
        eval_metric       = "logloss",
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
    )
    champion.fit(X_train, y_train,
                 eval_set=[(X_test, y_test)], verbose=False)

    y_pred  = champion.predict(X_test)
    y_proba = champion.predict_proba(X_test)[:, 1]

    final_metrics = {
        "accuracy"  : round(accuracy_score(y_test, y_pred),  4),
        "roc_auc"   : round(roc_auc_score(y_test, y_proba),  4),
        "f1"        : round(f1_score(y_test, y_pred),         4),
        "precision" : round(precision_score(y_test, y_pred),  4),
        "recall"    : round(recall_score(y_test, y_pred),     4),
    }

    print(f"\n      {'Metric':<12} {'Score':>8}")
    print(f"      {'─'*22}")
    for k, v in final_metrics.items():
        print(f"      {k:<12} {v:>8.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Default','Default'])}")

    # ── Save artifacts ─────────────────────────────────────────
    print("[3/5] Saving model artifacts …")

    model_path = os.path.join(OUT_DIR, "champion_xgb.joblib")
    prep_path  = os.path.join(OUT_DIR, "preprocessor.joblib")
    feat_path  = os.path.join(OUT_DIR, "feature_names.json")

    joblib.dump(champion, model_path, compress=3)
    joblib.dump(prep,     prep_path,  compress=3)
    with open(feat_path, "w") as f:
        json.dump(feature_names, f, indent=2)

    model_size = os.path.getsize(model_path) / 1024
    print(f"      champion_xgb.joblib   → {model_size:.1f} KB")
    print(f"      preprocessor.joblib   → saved")
    print(f"      feature_names.json    → {len(feature_names)} features")

    # ── Build input schema ─────────────────────────────────────
    print("\n[4/5] Generating input schema & sample payloads …")

    # Encoder classes for documentation
    enc = prep.label_encoders_

    input_schema = {
        "_meta": {
            "description"  : "Credit Risk Prediction API — Input Schema",
            "version"      : "1.0.0",
            "generated_at" : datetime.datetime.utcnow().isoformat() + "Z",
            "model"        : "XGBClassifier (Optuna-tuned)",
            "target"       : "loan_status",
            "target_classes": {"0": "No Default", "1": "Default"},
            "output_fields": {
                "prediction" : "int  — 0 (No Default) or 1 (Default)",
                "probability": "float — P(Default), range [0.0, 1.0]",
                "risk_tier"  : "str  — LOW / MEDIUM / HIGH (threshold-based)"
            }
        },
        "fields": {
            "person_age": {
                "type"       : "integer",
                "description": "Age of the applicant in years",
                "min"        : 18,
                "max"        : 100,
                "example"    : 29,
                "required"   : True
            },
            "person_income": {
                "type"       : "integer",
                "description": "Annual income of the applicant in USD",
                "min"        : 4_000,
                "max"        : 6_000_000,
                "example"    : 58_000,
                "required"   : True
            },
            "person_home_ownership": {
                "type"        : "string",
                "description" : "Applicant's home ownership status",
                "allowed"     : sorted(enc["person_home_ownership"].classes_.tolist()),
                "example"     : "RENT",
                "required"    : True
            },
            "person_emp_length": {
                "type"       : "float",
                "description": "Employment length in years (0 = less than 1 year)",
                "min"        : 0.0,
                "max"        : 60.0,
                "example"    : 4.0,
                "required"   : True,
                "nullable"   : True,
                "note"       : "Pass null if unknown; will be imputed with training median"
            },
            "loan_intent": {
                "type"        : "string",
                "description" : "Purpose / intent of the loan",
                "allowed"     : sorted(enc["loan_intent"].classes_.tolist()),
                "example"     : "PERSONAL",
                "required"    : True
            },
            "loan_grade": {
                "type"        : "string",
                "description" : "Loan grade assigned by the lender (A = best, G = worst)",
                "allowed"     : sorted(enc["loan_grade"].classes_.tolist()),
                "example"     : "B",
                "required"    : True
            },
            "loan_amnt": {
                "type"       : "integer",
                "description": "Loan amount requested in USD",
                "min"        : 500,
                "max"        : 35_000,
                "example"    : 8_000,
                "required"   : True
            },
            "loan_int_rate": {
                "type"       : "float",
                "description": "Annual interest rate of the loan (%)",
                "min"        : 5.0,
                "max"        : 24.0,
                "example"    : 11.49,
                "required"   : True,
                "nullable"   : True,
                "note"       : "Pass null if unknown; will be imputed from loan_grade median"
            },
            "loan_percent_income": {
                "type"       : "float",
                "description": "Loan amount as a fraction of annual income",
                "min"        : 0.0,
                "max"        : 0.83,
                "example"    : 0.14,
                "required"   : True,
                "note"       : "Computed as loan_amnt / person_income — supply directly or let API compute"
            },
            "cb_person_default_on_file": {
                "type"        : "string",
                "description" : "Whether the applicant has a historical default on credit bureau file",
                "allowed"     : sorted(enc["cb_person_default_on_file"].classes_.tolist()),
                "example"     : "N",
                "required"    : True
            },
            "cb_person_cred_hist_length": {
                "type"       : "integer",
                "description": "Length of credit history in years",
                "min"        : 2,
                "max"        : 30,
                "example"    : 6,
                "required"   : True
            }
        },
        "engineered_features": {
            "_note"            : "These are computed server-side from the raw fields above. Do NOT pass these in the request.",
            "loan_to_income"   : "loan_amnt / (person_income + 1)",
            "income_per_emp_yr": "person_income / (person_emp_length + 1)",
            "risk_score"       : "loan_int_rate × loan_percent_income"
        }
    }

    schema_path = os.path.join(OUT_DIR, "input_schema.json")
    with open(schema_path, "w") as f:
        json.dump(input_schema, f, indent=2)

    # ── Sample request bodies ──────────────────────────────────
    sample_single = {
        "person_age"                : 29,
        "person_income"             : 58_000,
        "person_home_ownership"     : "RENT",
        "person_emp_length"         : 4.0,
        "loan_intent"               : "PERSONAL",
        "loan_grade"                : "B",
        "loan_amnt"                 : 8_000,
        "loan_int_rate"             : 11.49,
        "loan_percent_income"       : 0.14,
        "cb_person_default_on_file" : "N",
        "cb_person_cred_hist_length": 6
    }

    sample_batch = {
        "records": [
            {
                "person_age": 29, "person_income": 58_000,
                "person_home_ownership": "RENT", "person_emp_length": 4.0,
                "loan_intent": "PERSONAL", "loan_grade": "B",
                "loan_amnt": 8_000, "loan_int_rate": 11.49,
                "loan_percent_income": 0.14,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 6
            },
            {
                "person_age": 55, "person_income": 34_000,
                "person_home_ownership": "MORTGAGE", "person_emp_length": 2.0,
                "loan_intent": "DEBTCONSOLIDATION", "loan_grade": "E",
                "loan_amnt": 18_000, "loan_int_rate": 19.75,
                "loan_percent_income": 0.53,
                "cb_person_default_on_file": "Y",
                "cb_person_cred_hist_length": 11
            },
            {
                "person_age": 22, "person_income": 28_000,
                "person_home_ownership": "OWN", "person_emp_length": None,
                "loan_intent": "EDUCATION", "loan_grade": "A",
                "loan_amnt": 3_500, "loan_int_rate": None,
                "loan_percent_income": 0.13,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 3
            }
        ]
    }

    single_path = os.path.join(OUT_DIR, "sample_request.json")
    batch_path  = os.path.join(OUT_DIR, "sample_batch.json")
    with open(single_path, "w") as f:
        json.dump(sample_single, f, indent=2)
    with open(batch_path, "w") as f:
        json.dump(sample_batch, f, indent=2)

    # ── Model card ─────────────────────────────────────────────
    print("[5/5] Writing model card …")
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    model_card = f"""# Model Card — Credit Risk Champion (XGBoost + Optuna)

**Generated:** {now}  
**Version:** 1.0.0  
**Owner:** ML Team (Person A)  
**Consumer:** API Team (Person B)

---

## 1. Model Overview

| Property | Value |
|---|---|
| Algorithm | XGBoost (`XGBClassifier`) |
| Tuning | Optuna — TPE sampler, Hyperband pruner, 60 trials |
| Task | Binary classification — credit default prediction |
| Target | `loan_status` → 0 = No Default, 1 = Default |
| Training rows | {len(X_train):,} |
| Test rows | {len(X_test):,} |
| Class imbalance correction | `scale_pos_weight = {spw}` |

---

## 2. Performance on Hold-out Test Set

| Metric | Score |
|---|---|
| ROC-AUC | **{final_metrics['roc_auc']}** |
| Accuracy | {final_metrics['accuracy']} |
| F1 (Default class) | {final_metrics['f1']} |
| Precision (Default) | {final_metrics['precision']} |
| Recall (Default) | {final_metrics['recall']} |

Interpretation: The model correctly flags **{final_metrics['recall']*100:.1f}%** of actual defaulters
while keeping false alarms at **{(1-final_metrics['precision'])*100:.1f}%**.

---

## 3. Best Hyperparameters (Optuna)

```json
{json.dumps(BEST_PARAMS, indent=2)}
```

---

## 4. Input Features

### Raw features (sent in the API request)

| Field | Type | Required | Example | Notes |
|---|---|---|---|---|
| `person_age` | int | ✅ | `29` | 18–100 |
| `person_income` | int | ✅ | `58000` | Annual USD |
| `person_home_ownership` | string | ✅ | `"RENT"` | RENT / OWN / MORTGAGE / OTHER |
| `person_emp_length` | float | ✅ | `4.0` | Years; pass `null` if unknown |
| `loan_intent` | string | ✅ | `"PERSONAL"` | See schema for full list |
| `loan_grade` | string | ✅ | `"B"` | A–G |
| `loan_amnt` | int | ✅ | `8000` | USD |
| `loan_int_rate` | float | ✅ | `11.49` | %; pass `null` if unknown |
| `loan_percent_income` | float | ✅ | `0.14` | loan_amnt / person_income |
| `cb_person_default_on_file` | string | ✅ | `"N"` | Y / N |
| `cb_person_cred_hist_length` | int | ✅ | `6` | Years |

### Engineered features (computed server-side — do NOT send these)

| Feature | Formula |
|---|---|
| `loan_to_income` | `loan_amnt / (person_income + 1)` |
| `income_per_emp_yr` | `person_income / (person_emp_length + 1)` |
| `risk_score` | `loan_int_rate × loan_percent_income` |

---

## 5. Expected API Output

```json
{{
  "prediction"  : 0,
  "probability" : 0.073,
  "risk_tier"   : "LOW"
}}
```

| Field | Type | Description |
|---|---|---|
| `prediction` | int | 0 = No Default, 1 = Default |
| `probability` | float | P(Default) — range [0.0, 1.0] |
| `risk_tier` | string | LOW (<0.3), MEDIUM (0.3–0.6), HIGH (>0.6) |

---

## 6. Files Handed Over

| File | Purpose |
|---|---|
| `champion_xgb.joblib` | Trained XGBoost model — load with `joblib.load()` |
| `preprocessor.joblib` | Fitted preprocessor — **must** be applied before prediction |
| `feature_names.json` | Ordered column list the model expects |
| `input_schema.json` | Full schema with types, ranges, allowed values |
| `sample_request.json` | Single-record request body — copy-paste for testing |
| `sample_batch.json` | Batch request body — 3 diverse examples |
| `model_card.md` | This document |

---

## 7. How to Load and Use the Model

```python
import joblib, json
import pandas as pd

# Load artifacts
preprocessor   = joblib.load("model/preprocessor.joblib")
model          = joblib.load("model/champion_xgb.joblib")
feature_names  = json.load(open("model/feature_names.json"))

def predict(record: dict) -> dict:
    df  = pd.DataFrame([record])
    X   = preprocessor.transform(df)[feature_names]
    pred  = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0, 1])
    tier  = "HIGH" if proba > 0.6 else ("MEDIUM" if proba > 0.3 else "LOW")
    return {{"prediction": pred, "probability": round(proba, 4), "risk_tier": tier}}
```

---

## 8. Caveats & Known Limitations

- Model was trained on data with **US-centric** loan records; performance may degrade on other geographies.
- `person_emp_length` and `loan_int_rate` nulls are imputed with training-set medians — upstream systems should aim to provide these.
- Class imbalance (~78% No Default / 22% Default) is handled via `scale_pos_weight`; consider recalibrating the 0.5 decision threshold for different business cost trade-offs.
- Re-train recommended if data distribution drifts >6 months.
"""

    card_path = os.path.join(OUT_DIR, "model_card.md")
    with open(card_path, "w") as f:
        f.write(model_card)

    # ── Final summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  HANDOVER COMPLETE")
    print("=" * 60)
    for fname in [
        "champion_xgb.joblib", "preprocessor.joblib",
        "feature_names.json",  "input_schema.json",
        "sample_request.json", "sample_batch.json",
        "model_card.md"
    ]:
        path = os.path.join(OUT_DIR, fname)
        size = os.path.getsize(path) / 1024
        print(f"  ✅  model/{fname:<30}  ({size:.1f} KB)")

    print(f"\n  Champion ROC-AUC : {final_metrics['roc_auc']}")
    print(f"  Champion Accuracy: {final_metrics['accuracy']}")
    print(f"\n  Hand this folder to Person B. They only need:")
    print(f"    1. champion_xgb.joblib  (the model)")
    print(f"    2. preprocessor.joblib  (the transforms)")
    print(f"    3. input_schema.json    (what to send)")
    print(f"    4. model_card.md        (how to use it)\n")


if __name__ == "__main__":
    main()
