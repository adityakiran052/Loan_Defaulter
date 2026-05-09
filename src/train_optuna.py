"""
train_optuna.py
===============
Phase 4 – Optimization with Optuna
Goal: Find the globally optimal XGBoost hyperparameters using Bayesian search,
      then compare every run — baseline, default XGB, and Optuna champion —
      side-by-side in the MLflow UI.

Usage:
    python train_optuna.py
    mlflow ui          # open http://127.0.0.1:5000 to see all runs
"""

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("optuna").setLevel(logging.WARNING)

import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import mlflow.sklearn
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import os

# ─────────────────────────────────────────────────────────────
# 0. Config
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH     = os.path.join(PROJECT_ROOT, "data", "raw", "credit_risk_dataset.csv")
EXPERIMENT    = "credit_risk_optimization"
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
CV_FOLDS      = 5
N_TRIALS      = 60          # Optuna trials
OPTUNA_STUDY  = "xgb_credit_risk_study"

mlflow.set_tracking_uri(f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────
# 1. Preprocessing (shared across all models)
# ─────────────────────────────────────────────────────────────
def load_and_preprocess(path: str):
    df = pd.read_csv(path)

    df.loc[df["person_age"] > 100, "person_age"] = (
        df.loc[df["person_age"] <= 100, "person_age"].median()
    )
    df.loc[df["person_emp_length"] > 60, "person_emp_length"] = np.nan
    df["person_emp_length"].fillna(df["person_emp_length"].median(), inplace=True)
    df["loan_int_rate"] = df.groupby("loan_grade")["loan_int_rate"].transform(
        lambda x: x.fillna(x.median())
    )
    df["loan_int_rate"].fillna(df["loan_int_rate"].median(), inplace=True)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].str.strip().str.upper()
        df[col] = le.fit_transform(df[col])

    # Feature engineering
    df["loan_to_income"]    = df["loan_amnt"] / (df["person_income"] + 1)
    df["income_per_emp_yr"] = df["person_income"] / (df["person_emp_length"] + 1)
    df["age_income_ratio"]  = df["person_age"] / (df["person_income"] / 10_000 + 1)
    df["risk_score"]        = df["loan_int_rate"] * df["loan_percent_income"]

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]
    return X, y


def scale_pos_weight(y_train):
    return round((y_train == 0).sum() / (y_train == 1).sum(), 4)


# ─────────────────────────────────────────────────────────────
# 2. Metrics helper
# ─────────────────────────────────────────────────────────────
def compute_metrics(model, X_test, y_test) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy"  : round(accuracy_score(y_test, y_pred),  4),
        "roc_auc"   : round(roc_auc_score(y_test, y_proba),  4),
        "f1"        : round(f1_score(y_test, y_pred),         4),
        "precision" : round(precision_score(y_test, y_pred),  4),
        "recall"    : round(recall_score(y_test, y_pred),     4),
    }


# ─────────────────────────────────────────────────────────────
# 3. Diagnostic plot
# ─────────────────────────────────────────────────────────────
def diagnostic_plot(model, X_test, y_test, title: str, fname: str):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

    # Confusion matrix
    ax0 = fig.add_subplot(gs[0, 0])
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["No Default", "Default"],
        cmap="Blues", ax=ax0)
    ax0.set_title("Confusion Matrix", fontweight="bold")

    # ROC
    ax1 = fig.add_subplot(gs[0, 1])
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax1, color="#4C72B0")
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set_title("ROC Curve", fontweight="bold")

    # Precision-Recall
    ax2 = fig.add_subplot(gs[0, 2])
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax2, color="#DD8452")
    ax2.set_title("Precision-Recall Curve", fontweight="bold")

    # Feature importance (XGB only)
    ax3 = fig.add_subplot(gs[1, :2])
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=X_test.columns).sort_values()
        top = imp.tail(15)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(top)))
        top.plot(kind="barh", ax=ax3, color=colors, edgecolor="white")
    else:
        # Logistic regression coefficients
        coefs = pd.Series(np.abs(model.named_steps["clf"].coef_[0]),
                          index=X_test.columns).sort_values()
        coefs.tail(15).plot(kind="barh", ax=ax3, color="#4C72B0", edgecolor="white")
    ax3.set_title("Feature Importances", fontweight="bold")

    # Probability distribution
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(y_proba[y_test == 0], bins=40, alpha=0.6, color="#4CAF50",
             label="No Default", density=True)
    ax4.hist(y_proba[y_test == 1], bins=40, alpha=0.6, color="#F44336",
             label="Default", density=True)
    ax4.axvline(0.5, ls="--", color="black", lw=1.5)
    ax4.set_title("Predicted Probability Distribution", fontweight="bold")
    ax4.set_xlabel("P(Default)")
    ax4.legend()

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    return fname


# ─────────────────────────────────────────────────────────────
# 4. Optuna objective
# ─────────────────────────────────────────────────────────────
def make_objective(X_train, y_train, spw, cv):
    def objective(trial: optuna.Trial) -> float:
        params = {
            # Core architecture
            "n_estimators"      : trial.suggest_int("n_estimators",      100, 1000, step=50),
            "max_depth"         : trial.suggest_int("max_depth",          2,   8),
            "learning_rate"     : trial.suggest_float("learning_rate",    0.005, 0.3, log=True),
            # Regularisation
            "subsample"         : trial.suggest_float("subsample",        0.5, 1.0),
            "colsample_bytree"  : trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bylevel" : trial.suggest_float("colsample_bylevel",0.4, 1.0),
            "min_child_weight"  : trial.suggest_int("min_child_weight",   1,   20),
            "gamma"             : trial.suggest_float("gamma",            0.0, 1.0),
            "reg_alpha"         : trial.suggest_float("reg_alpha",        1e-4, 10.0, log=True),
            "reg_lambda"        : trial.suggest_float("reg_lambda",       1e-4, 10.0, log=True),
            # Sampling method
            "grow_policy"       : trial.suggest_categorical("grow_policy",
                                    ["depthwise", "lossguide"]),
        }

        model = XGBClassifier(
            **params,
            scale_pos_weight  = spw,
            use_label_encoder = False,
            eval_metric       = "logloss",
            random_state      = RANDOM_STATE,
            n_jobs            = -1,
        )
        scores = cross_val_score(model, X_train, y_train,
                                 scoring="roc_auc", cv=cv, n_jobs=-1)
        return scores.mean()

    return objective


# ─────────────────────────────────────────────────────────────
# 5. Optuna visualisation plots
# ─────────────────────────────────────────────────────────────
def plot_optuna_history(study: optuna.Study, fname="optuna_history.png"):
    trials_df = study.trials_dataframe()
    trials_df = trials_df[trials_df["state"] == "COMPLETE"].reset_index(drop=True)
    trials_df["best_so_far"] = trials_df["value"].cummax()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Trial scores + running best
    ax = axes[0]
    ax.scatter(trials_df.index, trials_df["value"],
               alpha=0.5, s=25, color="#4C72B0", label="Trial AUC", zorder=3)
    ax.plot(trials_df.index, trials_df["best_so_far"],
            color="#E74C3C", lw=2.5, label="Best so far")
    ax.set_xlabel("Trial #", fontsize=11)
    ax.set_ylabel("CV ROC-AUC", fontsize=11)
    ax.set_title("Optuna Optimization History", fontsize=13, fontweight="bold")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    # Hyperparameter importance (top 8)
    ax2 = axes[1]
    importances = optuna.importance.get_param_importances(study)
    imp_series  = pd.Series(importances).sort_values()[-8:]
    colors      = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(imp_series)))
    imp_series.plot(kind="barh", ax=ax2, color=colors, edgecolor="white")
    ax2.set_title("Hyperparameter Importance (Optuna)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Relative Importance")
    ax2.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    plt.suptitle("Optuna Study Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()
    return fname


def plot_param_relationships(study: optuna.Study, fname="optuna_params.png"):
    trials_df = study.trials_dataframe()
    trials_df = trials_df[trials_df["state"] == "COMPLETE"]

    param_cols = [c for c in trials_df.columns if c.startswith("params_")]
    key_params = [c for c in param_cols
                  if any(k in c for k in ["n_estimators","max_depth",
                                           "learning_rate","subsample"])]

    fig, axes = plt.subplots(1, len(key_params), figsize=(5 * len(key_params), 4))
    if len(key_params) == 1:
        axes = [axes]

    for ax, col in zip(axes, key_params):
        param_name = col.replace("params_", "")
        sc = ax.scatter(trials_df[col], trials_df["value"],
                        c=trials_df["value"], cmap="RdYlGn",
                        alpha=0.7, s=30, edgecolors="none")
        ax.set_xlabel(param_name, fontsize=10)
        ax.set_ylabel("CV AUC", fontsize=10)
        ax.set_title(f"AUC vs {param_name}", fontsize=10, fontweight="bold")
        plt.colorbar(sc, ax=ax)

    plt.suptitle("Key Hyperparameter vs AUC Relationship", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()
    return fname


# ─────────────────────────────────────────────────────────────
# 6. Grand comparison chart
# ─────────────────────────────────────────────────────────────
def plot_grand_comparison(results: dict, fname="grand_comparison.png"):
    metrics   = ["accuracy", "roc_auc", "f1", "precision", "recall"]
    models    = list(results.keys())
    x         = np.arange(len(metrics))
    width     = 0.18
    palette   = ["#95A5A6", "#3498DB", "#E67E22", "#2ECC71"]

    fig, ax = plt.subplots(figsize=(15, 7))
    for i, (model_name, res) in enumerate(results.items()):
        vals = [res[m] for m in metrics]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=model_name, color=palette[i % len(palette)],
                      edgecolor="white", alpha=0.92)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("All Models — Grand Performance Comparison\n"
                 "(Baseline → XGB Default → XGB Tuned → Optuna Champion)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.axhline(0.5, ls="--", color="grey", lw=0.8, alpha=0.5)

    # Annotate best per metric
    for j, metric in enumerate(metrics):
        best_val  = max(results[m][metric] for m in models)
        best_mdl  = max(models, key=lambda m: results[m][metric])
        ax.annotate(f"★ {best_mdl}", xy=(j, best_val + 0.05),
                    ha="center", fontsize=7, color="black", alpha=0.7)

    plt.tight_layout()
    plt.savefig(fname, dpi=130)
    plt.close()
    return fname


# ─────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  PHASE 4 — OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    print("\n[1/6] Loading & preprocessing data …")
    X, y = load_and_preprocess(DATA_PATH)
    print(f"      Features: {X.shape[1]}  |  Samples: {X.shape[0]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    spw = scale_pos_weight(y_train)
    cv  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    mlflow.set_experiment(EXPERIMENT)
    all_results = {}

    # ── Step A: Re-run baselines in THIS experiment so MLflow can compare ──
    print("\n[2/6] Training baselines (logged to same experiment for comparison) …")
    baseline_models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=RANDOM_STATE))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=20,
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    }
    for name, model in baseline_models.items():
        model.fit(X_train, y_train)
        m = compute_metrics(model, X_test, y_test)
        all_results[name] = m
        fname = diagnostic_plot(model, X_test, y_test, name,
                                f"diag_{name.replace(' ','_').lower()}.png")
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", name)
            for k, v in m.items():
                mlflow.log_metric(k, v)
            mlflow.log_artifact(fname)
            mlflow.sklearn.log_model(model, artifact_path="model")
        print(f"      {name}: AUC={m['roc_auc']:.4f}  Acc={m['accuracy']:.4f}")

    # ── Step B: XGBoost default (reference point) ──────────────────────────
    print("\n[3/6] Training default XGBoost (reference) …")
    xgb_default = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, use_label_encoder=False,
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
    )
    xgb_default.fit(X_train, y_train)
    m_def = compute_metrics(xgb_default, X_test, y_test)
    all_results["XGB Default"] = m_def
    fname_def = diagnostic_plot(xgb_default, X_test, y_test,
                                "XGB Default", "diag_xgb_default.png")
    with mlflow.start_run(run_name="XGB_Default"):
        mlflow.log_param("model_type",    "XGBClassifier")
        mlflow.log_param("n_estimators",  300)
        mlflow.log_param("max_depth",     6)
        mlflow.log_param("learning_rate", 0.1)
        for k, v in m_def.items():
            mlflow.log_metric(k, v)
        mlflow.log_artifact(fname_def)
        mlflow.xgboost.log_model(xgb_default, artifact_path="model")
    print(f"      XGB Default: AUC={m_def['roc_auc']:.4f}  Acc={m_def['accuracy']:.4f}")

    # ── Step C: Optuna study ────────────────────────────────────────────────
    print(f"\n[4/6] Running Optuna study ({N_TRIALS} trials) …")
    print(f"      Sampler : TPE  |  Pruner : Hyperband  |  Objective : CV ROC-AUC")
    print(f"      Search space: n_estimators, max_depth, learning_rate,")
    print(f"                    subsample, colsample_bytree/bylevel,")
    print(f"                    min_child_weight, gamma, reg_alpha, reg_lambda\n")

    study = optuna.create_study(
        study_name = OPTUNA_STUDY,
        direction  = "maximize",
        sampler    = TPESampler(seed=RANDOM_STATE, n_startup_trials=10,
                                multivariate=True),
        pruner     = HyperbandPruner(),
    )

    # Progress callback
    def progress_cb(study, trial):
        if trial.number % 10 == 0 or trial.number == N_TRIALS - 1:
            print(f"      Trial {trial.number:>3} | "
                  f"value={trial.value:.5f} | "
                  f"best={study.best_value:.5f}")

    study.optimize(
        make_objective(X_train, y_train, spw, cv),
        n_trials  = N_TRIALS,
        callbacks = [progress_cb],
        show_progress_bar = False,
    )

    best_params = study.best_params
    print(f"\n  [OK] Best CV AUC : {study.best_value:.5f}")
    print(f"  [OK] Best params :")
    for k, v in best_params.items():
        print(f"      {k:<25}: {v}")

    # ── Step D: Train champion model ────────────────────────────────────────
    print("\n[5/6] Training Optuna champion on full train set …")
    champion = XGBClassifier(
        **best_params,
        scale_pos_weight  = spw,
        use_label_encoder = False,
        eval_metric       = "logloss",
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
    )
    champion.fit(X_train, y_train,
                 eval_set=[(X_test, y_test)], verbose=False)

    m_opt = compute_metrics(champion, X_test, y_test)
    all_results["Optuna Champion"] = m_opt

    print(f"\n  {'Metric':<12} {'XGB Default':>12} {'Optuna':>12} {'Delta':>8}")
    print(f"  {'-'*46}")
    for metric in ["accuracy", "roc_auc", "f1", "precision", "recall"]:
        delta = m_opt[metric] - m_def[metric]
        sign  = "+" if delta >= 0 else ""
        print(f"  {metric:<12} {m_def[metric]:>12.4f} {m_opt[metric]:>12.4f} "
              f"{sign}{delta:>7.4f}")

    print(f"\n{classification_report(y_test, champion.predict(X_test), target_names=['No Default','Default'])}")

    # Diagnostic plot
    fname_opt = diagnostic_plot(champion, X_test, y_test,
                                "Optuna Champion XGBoost", "diag_optuna_champion.png")

    # Optuna-specific plots
    print("[5/6] Generating Optuna analysis plots …")
    hist_path   = plot_optuna_history(study)
    params_path = plot_param_relationships(study)

    # MLflow log
    with mlflow.start_run(run_name="Optuna_Champion"):
        mlflow.log_param("model_type",     "XGBClassifier_Optuna")
        mlflow.log_param("n_trials",        N_TRIALS)
        mlflow.log_param("sampler",        "TPE_multivariate")
        mlflow.log_param("pruner",         "Hyperband")
        mlflow.log_param("optuna_best_cv_auc", round(study.best_value, 5))
        for k, v in best_params.items():
            mlflow.log_param(k, v)
        for k, v in m_opt.items():
            mlflow.log_metric(k, v)
        mlflow.log_artifact(fname_opt)
        mlflow.log_artifact(hist_path)
        mlflow.log_artifact(params_path)
        mlflow.xgboost.log_model(
            champion, artifact_path="model",
            registered_model_name="xgboost_optuna_champion"
        )

    # ── Step E: Grand comparison ────────────────────────────────────────────
    print("\n[6/6] Generating grand comparison …")
    grand_path = plot_grand_comparison(all_results)

    # Log comparison as shared artifact in a dedicated run
    with mlflow.start_run(run_name="Grand_Comparison"):
        mlflow.log_artifact(grand_path)
        mlflow.log_artifact(hist_path)
        mlflow.log_artifact(params_path)
        # Log all model scores as metrics for parallel-coordinate view in MLflow UI
        for model_name, res in all_results.items():
            safe = model_name.replace(" ", "_")
            for metric, val in res.items():
                mlflow.log_metric(f"{safe}_{metric}", val)

    # ── Print final leaderboard ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL LEADERBOARD")
    print("=" * 65)
    leaderboard = pd.DataFrame(all_results).T
    leaderboard.index.name = "Model"
    leaderboard_sorted = leaderboard.sort_values("roc_auc", ascending=False)
    print(leaderboard_sorted.to_string())

    winner = leaderboard_sorted.index[0]
    print(f"\n  [CHAMPION] {winner}")
    print(f"      ROC-AUC : {leaderboard_sorted.loc[winner,'roc_auc']:.4f}")
    print(f"      Accuracy: {leaderboard_sorted.loc[winner,'accuracy']:.4f}")
    print(f"      F1      : {leaderboard_sorted.loc[winner,'f1']:.4f}")

    print(f"\n  Plots saved:")
    for p in [grand_path, hist_path, params_path, fname_opt]:
        print(f"    -> {p}")

    print(f"\n  MLflow experiment : '{EXPERIMENT}'")
    print("  Run `mlflow ui` then open http://127.0.0.1:5000")
    print("  -> Go to the experiment -> Select all runs -> 'Compare'")
    print("  -> Parallel Coordinates plot shows every hyperparameter vs AUC\n")


if __name__ == "__main__":
    main()
