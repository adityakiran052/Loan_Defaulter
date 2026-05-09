"""
train_xgb.py
============
Phase 3 – Step 2 | Advanced Model: XGBoost
Goal: Beat the baseline using gradient-boosted trees with hyperparameter tuning.
All runs tracked with MLflow.

Usage:
    python train_xgb.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    RandomizedSearchCV, cross_val_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
    PrecisionRecallDisplay
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ─────────────────────────────────────────────
# 0. Config
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "raw", "credit_risk_dataset.csv")
EXPERIMENT   = "credit_risk_xgboost"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5
N_ITER       = 30          # RandomizedSearchCV iterations

mlflow.set_tracking_uri(f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}")

# ─────────────────────────────────────────────
# 1. Data Loading & Preprocessing
# ─────────────────────────────────────────────
def load_and_preprocess(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    # Fix outliers
    df.loc[df["person_age"] > 100, "person_age"] = df.loc[
        df["person_age"] <= 100, "person_age"
    ].median()
    df.loc[df["person_emp_length"] > 60, "person_emp_length"] = np.nan

    # Impute missing
    df["person_emp_length"].fillna(df["person_emp_length"].median(), inplace=True)
    df["loan_int_rate"] = df.groupby("loan_grade")["loan_int_rate"].transform(
        lambda x: x.fillna(x.median())
    )
    df["loan_int_rate"].fillna(df["loan_int_rate"].median(), inplace=True)

    # Encode categoricals
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].str.strip().str.upper()
        df[col] = le.fit_transform(df[col])

    # Feature engineering
    df["loan_to_income"]    = df["loan_amnt"] / (df["person_income"] + 1)
    df["income_per_emp_yr"] = df["person_income"] / (df["person_emp_length"] + 1)
    df["age_income_ratio"]  = df["person_age"] / (df["person_income"] / 10000 + 1)
    df["risk_score"]        = (
        df["loan_int_rate"] * df["loan_percent_income"]
    )  # high rate × high income ratio → risky

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]
    return X, y


# ─────────────────────────────────────────────
# 2. Class imbalance weight
# ─────────────────────────────────────────────
def compute_scale_pos_weight(y_train: pd.Series) -> float:
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    return round(neg / pos, 4)


# ─────────────────────────────────────────────
# 3. Hyperparameter search space
# ─────────────────────────────────────────────
XGB_PARAM_DIST = {
    "n_estimators"     : [200, 300, 400, 500, 600],
    "max_depth"        : [3, 4, 5, 6, 7],
    "learning_rate"    : [0.01, 0.05, 0.1, 0.15, 0.2],
    "subsample"        : [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree" : [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight" : [1, 3, 5, 7],
    "gamma"            : [0, 0.1, 0.2, 0.3],
    "reg_alpha"        : [0, 0.01, 0.1, 1.0],
    "reg_lambda"       : [0.5, 1.0, 1.5, 2.0],
}


# ─────────────────────────────────────────────
# 4. Plotting helpers
# ─────────────────────────────────────────────
def plot_diagnostics(model, X_test, y_test, best_params: dict, run_name: str):
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ── Confusion matrix ──────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["No Default", "Default"],
        cmap="Blues", ax=ax0
    )
    ax0.set_title("Confusion Matrix", fontweight="bold")

    # ── ROC curve ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax1, color="#4C72B0")
    ax1.set_title("ROC Curve", fontweight="bold")
    ax1.plot([0, 1], [0, 1], "k--", lw=1)

    # ── Precision-Recall curve ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax2, color="#DD8452")
    ax2.set_title("Precision-Recall Curve", fontweight="bold")

    # ── Feature importance ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    importances = pd.Series(model.feature_importances_,
                            index=X_test.columns).sort_values(ascending=True)
    top15 = importances.tail(15)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(top15)))
    top15.plot(kind="barh", ax=ax3, color=colors, edgecolor="white")
    ax3.set_title("Top-15 Feature Importances (gain)", fontweight="bold")
    ax3.set_xlabel("Importance Score")

    # ── Prediction probability distribution ───────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(y_proba[y_test == 0], bins=40, alpha=0.6, color="#4CAF50",
             label="No Default", density=True)
    ax4.hist(y_proba[y_test == 1], bins=40, alpha=0.6, color="#F44336",
             label="Default", density=True)
    ax4.axvline(0.5, ls="--", color="black", lw=1.5)
    ax4.set_title("Predicted Probability Distribution", fontweight="bold")
    ax4.set_xlabel("P(Default)")
    ax4.legend()

    fig.suptitle(f"XGBoost Diagnostic Report — {run_name}",
                 fontsize=15, fontweight="bold", y=1.01)

    path = f"xgb_diagnostics_{run_name.replace(' ','_').lower()}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


# ─────────────────────────────────────────────
# 5. Train & evaluate one run
# ─────────────────────────────────────────────
def train_and_log(run_name: str, params: dict, X_train, X_test,
                  y_train, y_test, scale_pos_weight: float) -> dict:

    model = XGBClassifier(
        **params,
        scale_pos_weight = scale_pos_weight,
        use_label_encoder=False,
        eval_metric      = "logloss",
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
    )

    # Cross-val on train set
    cv   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs = cross_val_score(model, X_train, y_train,
                           scoring="roc_auc", cv=cv, n_jobs=-1)
    print(f"\n  CV AUC: {aucs.mean():.4f} ± {aucs.std():.4f}")

    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy"    : round(accuracy_score(y_test, y_pred),  4),
        "roc_auc"     : round(roc_auc_score(y_test, y_proba),  4),
        "f1"          : round(f1_score(y_test, y_pred),         4),
        "precision"   : round(precision_score(y_test, y_pred),  4),
        "recall"      : round(recall_score(y_test, y_pred),     4),
        "cv_auc_mean" : round(aucs.mean(), 4),
        "cv_auc_std"  : round(aucs.std(),  4),
    }

    print(f"  Test AUC : {metrics['roc_auc']:.4f}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Default','Default'])}")

    diag_path = plot_diagnostics(model, X_test, y_test, params, run_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("run_type",          run_name)
        mlflow.log_param("scale_pos_weight",  scale_pos_weight)
        mlflow.log_param("cv_folds",          CV_FOLDS)
        mlflow.log_param("test_size",         TEST_SIZE)
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_artifact(diag_path)
        mlflow.xgboost.log_model(model, artifact_path="model",
                                 registered_model_name="xgboost_credit_risk")

    return metrics, model


# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────
def main():
    print("Loading and preprocessing data …")
    X, y = load_and_preprocess(DATA_PATH)
    print(f"  Features : {X.shape[1]} | Samples : {X.shape[0]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    spw = compute_scale_pos_weight(y_train)
    print(f"  scale_pos_weight (class imbalance correction): {spw}")

    mlflow.set_experiment(EXPERIMENT)

    # ── Run 1: Default XGBoost ─────────────────────────────────────
    print("\n" + "="*55)
    print("  RUN 1: Default XGBoost (no tuning)")
    print("="*55)
    default_params = {
        "n_estimators" : 300,
        "max_depth"    : 6,
        "learning_rate": 0.1,
        "subsample"    : 0.8,
        "colsample_bytree": 0.8,
    }
    metrics_default, _ = train_and_log(
        "XGB_Default", default_params,
        X_train, X_test, y_train, y_test, spw
    )

    # ── Run 2: Hyperparameter Search ───────────────────────────────
    print("\n" + "="*55)
    print(f"  RUN 2: RandomizedSearchCV ({N_ITER} iterations, {CV_FOLDS}-fold CV)")
    print("="*55)

    base_xgb = XGBClassifier(
        scale_pos_weight=spw,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    cv_strat = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search   = RandomizedSearchCV(
        base_xgb, XGB_PARAM_DIST,
        n_iter=N_ITER,
        scoring="roc_auc",
        cv=cv_strat,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_params = {k.replace("xgbclassifier__", ""): v
                   for k, v in search.best_params_.items()}
    print(f"\n  Best CV AUC : {search.best_score_:.4f}")
    print(f"  Best params : {best_params}")

    metrics_tuned, best_model = train_and_log(
        "XGB_Tuned", best_params,
        X_train, X_test, y_train, y_test, spw
    )

    # ── Final Comparison ───────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL COMPARISON")
    print("="*60)
    compare = pd.DataFrame({
        "XGB Default" : metrics_default,
        "XGB Tuned"   : metrics_tuned,
    }).T
    compare.index.name = "Model"
    print(compare.to_string())

    # ── Improvement summary chart ──────────────────────────────────
    metric_names = ["accuracy", "roc_auc", "f1", "precision", "recall"]
    x     = np.arange(len(metric_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (label, res) in enumerate(
        [("XGB Default", metrics_default), ("XGB Tuned", metrics_tuned)]
    ):
        vals = [res[m] for m in metric_names]
        bars = ax.bar(x + i * width, vals, width, label=label,
                      color=["#4C72B0", "#2ecc71"][i], edgecolor="white", alpha=0.9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_title("XGBoost: Default vs Tuned Performance", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("xgb_comparison.png", dpi=120)
    plt.close()
    print("\nComparison chart saved -> xgb_comparison.png")

    print(f"\nMLflow experiment: '{EXPERIMENT}'")
    print("Run `mlflow ui` to explore all runs in your browser.\n")

    # ── Save best model features for downstream use ────────────────
    pd.Series(X.columns.tolist()).to_csv("feature_names.csv", index=False, header=False)
    print("Feature names saved -> feature_names.csv\n")


if __name__ == "__main__":
    main()
