"""
train_baseline.py
=================
Phase 3 – Step 1 | Baseline Modeling Pipeline
Goal: Establish a "floor" accuracy using Logistic Regression & Decision Tree.
All runs tracked with MLflow.

Usage:
    python train_baseline.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────
# 0. Config
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "raw", "credit_risk_dataset.csv")
EXPERIMENT   = "credit_risk_baseline"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

mlflow.set_tracking_uri(f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}")

# ─────────────────────────────────────────────
# 1. Data Loading & Preprocessing
# ─────────────────────────────────────────────
def load_and_preprocess(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    # --- Fix outliers ---
    df.loc[df["person_age"] > 100, "person_age"] = df.loc[
        df["person_age"] <= 100, "person_age"
    ].median()
    df.loc[df["person_emp_length"] > 60, "person_emp_length"] = np.nan

    # --- Impute missing values ---
    df["person_emp_length"].fillna(df["person_emp_length"].median(), inplace=True)
    df["loan_int_rate"] = df.groupby("loan_grade")["loan_int_rate"].transform(
        lambda x: x.fillna(x.median())
    )
    df["loan_int_rate"].fillna(df["loan_int_rate"].median(), inplace=True)

    # --- Encode categoricals ---
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].str.strip().str.upper()
        df[col] = le.fit_transform(df[col])

    # --- Feature engineering ---
    df["loan_to_income"]     = df["loan_amnt"] / (df["person_income"] + 1)
    df["income_per_emp_yr"]  = df["person_income"] / (df["person_emp_length"] + 1)

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]
    return X, y


# ─────────────────────────────────────────────
# 2. Evaluation helper
# ─────────────────────────────────────────────
def evaluate(name: str, model, X_train, X_test, y_train, y_test) -> dict:
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy"  : round(accuracy_score(y_test, y_pred),           4),
        "roc_auc"   : round(roc_auc_score(y_test, y_proba),           4),
        "f1"        : round(f1_score(y_test, y_pred),                  4),
        "precision" : round(precision_score(y_test, y_pred),           4),
        "recall"    : round(recall_score(y_test, y_pred),              4),
    }

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Default','Default'])}")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["No Default", "Default"],
        cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix — {name}", fontweight="bold")
    plt.tight_layout()
    cm_path = f"cm_{name.replace(' ', '_').lower()}.png"
    plt.savefig(cm_path, dpi=100)
    plt.close()

    return metrics, cm_path


# ─────────────────────────────────────────────
# 3. Main
# ─────────────────────────────────────────────
def main():
    print("Loading and preprocessing data …")
    X, y = load_and_preprocess(DATA_PATH)
    print(f"  Features : {X.shape[1]} | Samples : {X.shape[0]}")
    print(f"  Class balance — No Default: {(y==0).sum()}  Default: {(y==1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    mlflow.set_experiment(EXPERIMENT)

    # ── Model definitions ──────────────────────────────────────────
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
            ))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=20,
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    }

    all_results = {}

    for name, model in models.items():
        with mlflow.start_run(run_name=name):

            # Cross-validation AUC (on training set)
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            cv_aucs = cross_val_score(model, X_train, y_train,
                                      scoring="roc_auc", cv=cv, n_jobs=-1)
            print(f"\n[{name}] CV AUC: {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")

            metrics, cm_path = evaluate(name, model, X_train, X_test, y_train, y_test)
            metrics["cv_auc_mean"] = round(cv_aucs.mean(), 4)
            metrics["cv_auc_std"]  = round(cv_aucs.std(),  4)

            # ── MLflow logging ────────────────────────────────────
            mlflow.log_param("model_type",    name)
            mlflow.log_param("test_size",     TEST_SIZE)
            mlflow.log_param("cv_folds",      CV_FOLDS)
            mlflow.log_param("random_state",  RANDOM_STATE)
            if "Decision Tree" in name:
                mlflow.log_param("max_depth",        8)
                mlflow.log_param("min_samples_leaf", 20)
            else:
                mlflow.log_param("max_iter",    1000)
                mlflow.log_param("class_weight","balanced")

            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            mlflow.log_artifact(cm_path)
            mlflow.sklearn.log_model(model, artifact_path="model",
                                     registered_model_name=f"baseline_{name.replace(' ','_').lower()}")

            all_results[name] = metrics

    # ── Summary table ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("  BASELINE RESULTS SUMMARY")
    print("="*60)
    summary = pd.DataFrame(all_results).T
    summary.index.name = "Model"
    print(summary.to_string())

    # ── Comparison bar chart ───────────────────────────────────────
    metrics_to_plot = ["accuracy", "roc_auc", "f1", "precision", "recall"]
    x      = np.arange(len(metrics_to_plot))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#4C72B0", "#DD8452"]
    for i, (name, res) in enumerate(all_results.items()):
        vals = [res[m] for m in metrics_to_plot]
        bars = ax.bar(x + i * width, vals, width, label=name, color=colors[i],
                      edgecolor="white", alpha=0.9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics_to_plot, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Baseline Models — Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend(fontsize=11)
    ax.axhline(0.5, ls="--", color="grey", lw=0.8, alpha=0.6, label="Random baseline")
    plt.tight_layout()
    plt.savefig("baseline_comparison.png", dpi=120)
    plt.close()
    print("\nComparison chart saved -> baseline_comparison.png")

    print(f"\nMLflow experiment: '{EXPERIMENT}'")
    print("Run `mlflow ui` to explore run history in your browser.\n")


if __name__ == "__main__":
    main()
