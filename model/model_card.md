# Model Card — Credit Risk Champion (XGBoost + Optuna)

**Generated:** 2026-05-08 21:20 UTC  
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
| Training rows | 26,064 |
| Test rows | 6,517 |
| Class imbalance correction | `scale_pos_weight = 3.5839` |

---

## 2. Performance on Hold-out Test Set

| Metric | Score |
|---|---|
| ROC-AUC | **0.9514** |
| Accuracy | 0.9234 |
| F1 (Default class) | 0.8199 |
| Precision (Default) | 0.8421 |
| Recall (Default) | 0.7989 |

Interpretation: The model correctly flags **79.9%** of actual defaulters
while keeping false alarms at **15.8%**.

---

## 3. Best Hyperparameters (Optuna)

```json
{
  "n_estimators": 600,
  "max_depth": 5,
  "learning_rate": 0.06774593731668856,
  "subsample": 0.7951977427742329,
  "colsample_bytree": 0.5622693192496626,
  "min_child_weight": 4,
  "gamma": 0.6580168264470879,
  "reg_alpha": 0.1681308748581626,
  "reg_lambda": 0.5769713635627621
}
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
{
  "prediction"  : 0,
  "probability" : 0.073,
  "risk_tier"   : "LOW"
}
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
    return {"prediction": pred, "probability": round(proba, 4), "risk_tier": tier}
```

---

## 8. Caveats & Known Limitations

- Model was trained on data with **US-centric** loan records; performance may degrade on other geographies.
- `person_emp_length` and `loan_int_rate` nulls are imputed with training-set medians — upstream systems should aim to provide these.
- Class imbalance (~78% No Default / 22% Default) is handled via `scale_pos_weight`; consider recalibrating the 0.5 decision threshold for different business cost trade-offs.
- Re-train recommended if data distribution drifts >6 months.
