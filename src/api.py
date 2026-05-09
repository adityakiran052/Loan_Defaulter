from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
import joblib
import os
from typing import Optional

app = FastAPI(
    title="Loan Default Prediction Service",
    description="Predicts whether a loan applicant will default based on Person A's trained model.",
    version="1.0.0",
)

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️  No model found at {MODEL_PATH}. Predictions will use mock logic.")


# ── Input schema ──────────────────────────────────────────────────────────────
# Mirrors the raw/cleaned columns that Person A's notebook feeds into the model
# (before feature engineering & one-hot encoding — those happen here in the API).

VALID_HOME_OWNERSHIP = {"RENT", "OWN", "MORTGAGE", "OTHER"}
VALID_LOAN_INTENT    = {"PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
                        "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"}
VALID_LOAN_GRADE     = {"A", "B", "C", "D", "E", "F", "G"}
VALID_CB_DEFAULT     = {"Y", "N"}


class LoanApplication(BaseModel):
    # ── Numerical inputs ──────────────────────────────────────────────────────
    person_age: int = Field(..., ge=18, le=100, examples=[28],
                            description="Applicant age (18–100)")
    person_income: float = Field(..., gt=0, examples=[54000],
                                 description="Annual income in USD")
    person_emp_length: float = Field(..., ge=0, le=60, examples=[5.0],
                                     description="Employment length in years (0–60)")
    loan_amnt: float = Field(..., gt=0, examples=[12000],
                             description="Requested loan amount in USD")
    loan_int_rate: float = Field(..., gt=0, le=30, examples=[11.49],
                                 description="Loan interest rate (%)")
    loan_percent_income: float = Field(..., ge=0, le=1, examples=[0.22],
                                       description="Loan amount as fraction of annual income")
    cb_person_cred_hist_length: int = Field(..., ge=0, examples=[3],
                                            description="Credit history length in years")

    # ── Categorical inputs (raw strings — encoded internally) ─────────────────
    person_home_ownership: str = Field(..., examples=["RENT"],
                                       description="RENT | OWN | MORTGAGE | OTHER")
    loan_intent: str = Field(..., examples=["PERSONAL"],
                             description="PERSONAL | EDUCATION | MEDICAL | VENTURE | HOMEIMPROVEMENT | DEBTCONSOLIDATION")
    loan_grade: str = Field(..., examples=["B"],
                            description="A | B | C | D | E | F | G")
    cb_person_default_on_file: str = Field(..., examples=["N"],
                                           description="Prior default on credit bureau file: Y | N")

    # ── Validators ────────────────────────────────────────────────────────────
    @validator("person_home_ownership")
    def validate_home_ownership(cls, v):
        v = v.strip().upper()
        if v not in VALID_HOME_OWNERSHIP:
            raise ValueError(f"person_home_ownership must be one of {VALID_HOME_OWNERSHIP}")
        return v

    @validator("loan_intent")
    def validate_loan_intent(cls, v):
        v = v.strip().upper()
        if v not in VALID_LOAN_INTENT:
            raise ValueError(f"loan_intent must be one of {VALID_LOAN_INTENT}")
        return v

    @validator("loan_grade")
    def validate_loan_grade(cls, v):
        v = v.strip().upper()
        if v not in VALID_LOAN_GRADE:
            raise ValueError(f"loan_grade must be one of {VALID_LOAN_GRADE}")
        return v

    @validator("cb_person_default_on_file")
    def validate_cb_default(cls, v):
        v = v.strip().upper()
        if v not in VALID_CB_DEFAULT:
            raise ValueError(f"cb_person_default_on_file must be 'Y' or 'N'")
        return v


# ── Feature engineering — mirrors Person A's notebook cells 35–37 ─────────────
def build_feature_vector(app: LoanApplication) -> pd.DataFrame:
    """
    Replicates the exact preprocessing pipeline from credit_risk_eda.ipynb:
      Cell 35 — engineered features
      Cell 36 — binary encode cb_person_default_on_file → cb_default_flag
      Cell 37 — pd.get_dummies(drop_first=True) on home_ownership, loan_intent, loan_grade
    """
    d = app.model_dump()

    # ── Step 1: base numeric features ─────────────────────────────────────────
    row = {
        "person_age":                 d["person_age"],
        "person_income":              d["person_income"],
        "person_emp_length":          d["person_emp_length"],
        "loan_amnt":                  d["loan_amnt"],
        "loan_int_rate":              d["loan_int_rate"],
        "loan_percent_income":        d["loan_percent_income"],
        "cb_person_cred_hist_length": d["cb_person_cred_hist_length"],
    }

    # ── Step 2: engineered features (Cell 35) ─────────────────────────────────
    row["loan_to_income_ratio"] = round(d["loan_amnt"] / d["person_income"], 4)
    row["income_per_emp_year"]  = round(d["person_income"] / (d["person_emp_length"] + 1), 2)

    # ── Step 3: binary encode cb default flag (Cell 36) ───────────────────────
    row["cb_default_flag"] = 1 if d["cb_person_default_on_file"] == "Y" else 0

    # ── Step 4: one-hot encode categoricals with drop_first=True (Cell 37) ────
    # Mirrors pd.get_dummies(columns=['person_home_ownership','loan_intent','loan_grade'], drop_first=True)
    # The reference (dropped) category is the alphabetically first value in each column.
    # person_home_ownership → drops MORTGAGE (first alphabetically among MORTGAGE/OTHER/OWN/RENT)
    for cat in ["OWN", "RENT", "OTHER"]:
        row[f"person_home_ownership_{cat}"] = int(d["person_home_ownership"] == cat)

    # loan_intent → drops DEBTCONSOLIDATION
    for cat in ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]:
        row[f"loan_intent_{cat}"] = int(d["loan_intent"] == cat)

    # loan_grade → drops A
    for cat in ["B", "C", "D", "E", "F", "G"]:
        row[f"loan_grade_{cat}"] = int(d["loan_grade"] == cat)

    return pd.DataFrame([row])


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"message": "Loan Default API is running. Use /docs for the interactive UI."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(application: LoanApplication):
    try:
        features = build_feature_vector(application)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {e}")

    if model is not None:
        prediction  = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])
    else:
        # ── Mock logic until Person A's model.pkl is available ────────────────
        prediction  = 0
        probability = 0.15

    return {
        "status":            "Success",
        "default_prediction": prediction,   # 1 = likely default, 0 = likely repay
        "probability":        round(probability, 4),
        "model_used":         "trained" if model is not None else "mock",
    }