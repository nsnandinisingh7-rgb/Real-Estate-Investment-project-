"""
Step 3 - Model Training, Evaluation and MLflow Tracking
Project: Real Estate Investment Advisor

Trains and compares:
  Classification (target Good_Investment):  Logistic Regression, Random Forest, XGBoost
  Regression     (target Future_Price_5Y_Lakhs): Ridge, Random Forest, XGBoost (+ naive baseline)

Every run is logged to MLflow (parameters, metrics, training time, model).
The best model of each task is saved for the Streamlit app.

Run (after 01_preprocessing.py):
    python 03_train_models.py --data outputs --outdir outputs/models

View experiments:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import json
import os
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
SEED = 42


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def importance(model, cols):
    """Feature importance for tree models, absolute coefficients for linear models."""
    if hasattr(model, "feature_importances_"):
        v = model.feature_importances_
    else:
        v = np.abs(np.ravel(model.coef_))
    return pd.Series(v, index=cols).sort_values(ascending=False)


def plot_importance(imp, title, path, top=20):
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.head(top)[::-1].plot.barh(ax=ax, color="#2a6f97")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs")
    ap.add_argument("--outdir", default="outputs/models")
    ap.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    plots = os.path.join(args.outdir, "plots")
    os.makedirs(plots, exist_ok=True)

    # ------------------------------------------------------------------ data
    state = joblib.load(os.path.join(args.data, "preprocessor.joblib"))
    df = pd.read_csv(os.path.join(args.data, "model_ready_real_estate.csv"))
    price_raw = pd.read_csv(os.path.join(args.data, "cleaned_real_estate.csv"),
                            usecols=["Price_in_Lakhs"])["Price_in_Lakhs"]
    fc, fr = state["feature_cols_classification"], state["feature_cols_regression"]
    y_c, y_r = df["Good_Investment"].astype(int), df["Future_Price_5Y_Lakhs"]
    print(f"Rows: {len(df):,} | classification features: {len(fc)} | regression features: {len(fr)}")
    print(f"Good_Investment share: {y_c.mean():.1%}")

    # one split used by both tasks: 80% train / 20% test, stratified on the class label
    idx_tr, idx_te = train_test_split(df.index, test_size=0.2, random_state=SEED, stratify=y_c)
    Xc_tr, Xc_te = df.loc[idx_tr, fc], df.loc[idx_te, fc]
    Xr_tr, Xr_te = df.loc[idx_tr, fr], df.loc[idx_te, fr]
    yc_tr, yc_te = y_c.loc[idx_tr], y_c.loc[idx_te]
    yr_tr, yr_te = y_r.loc[idx_tr], y_r.loc[idx_te]
    print(f"Train {len(idx_tr):,} | Test {len(idx_te):,}")

    mlflow.set_tracking_uri(args.tracking_uri)

    # ------------------------------------------------------ CLASSIFICATION
    clf_models = {
        "LogisticRegression": LogisticRegression(max_iter=300, C=1.0, n_jobs=2),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=10,
                                               max_samples=0.5, n_jobs=2, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.8, n_jobs=2,
                                 random_state=SEED, eval_metric="logloss"),
    }
    mlflow.set_experiment("RealEstate_Classification")
    clf_rows, clf_fitted, clf_proba = [], {}, {}
    print("\n=== CLASSIFICATION: Is this a Good Investment? ===")
    for name, model in clf_models.items():
        with mlflow.start_run(run_name=name):
            t0 = time.time()
            model.fit(Xc_tr, yc_tr)
            secs = time.time() - t0
            proba = model.predict_proba(Xc_te)[:, 1]
            pred = (proba >= 0.5).astype(int)
            m = dict(accuracy=accuracy_score(yc_te, pred),
                     precision=precision_score(yc_te, pred),
                     recall=recall_score(yc_te, pred),
                     f1=f1_score(yc_te, pred),
                     roc_auc=roc_auc_score(yc_te, proba),
                     train_accuracy=accuracy_score(yc_tr, model.predict(Xc_tr)))
            mlflow.log_params({k: v for k, v in model.get_params().items()
                               if isinstance(v, (int, float, str, bool)) or v is None})
            mlflow.log_param("n_features", len(fc))
            mlflow.log_metrics({**m, "train_seconds": secs})
            mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
        clf_rows.append({"model": name, **m, "train_seconds": secs})
        clf_fitted[name], clf_proba[name] = model, proba
        print(f"{name:<20} acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f} "
              f"(train acc {m['train_accuracy']:.4f}, {secs:.0f}s)")
    clf_res = pd.DataFrame(clf_rows).set_index("model")
    best_c = clf_res["f1"].idxmax()
    best_clf = clf_fitted[best_c]

    # cross-validation of the best classifier (60k sample, 3 folds) for stability evidence
    samp = Xc_tr.sample(60000, random_state=SEED)
    cv = cross_val_score(clone(clf_models[best_c]), samp, yc_tr.loc[samp.index],
                         cv=StratifiedKFold(3, shuffle=True, random_state=SEED), scoring="f1")
    print(f"Best classifier: {best_c} | 3-fold CV F1 = {cv.mean():.4f} +/- {cv.std():.4f}")

    # classification plots
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(yc_te, (clf_proba[best_c] >= .5).astype(int))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                xticklabels=["Not good", "Good"], yticklabels=["Not good", "Good"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {best_c}")
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "clf_confusion_matrix.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in clf_proba.items():
        fpr, tpr, _ = roc_curve(yc_te, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(yc_te, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curves - Good Investment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "clf_roc_curves.png"), dpi=130)
    plt.close(fig)

    imp_c = importance(best_clf, fc)
    plot_importance(imp_c, f"Top 20 Features - {best_c} (Good Investment)",
                    os.path.join(plots, "clf_feature_importance.png"))

    # --------------------------------------------------------- REGRESSION
    reg_models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=10,
                                              max_samples=0.5, n_jobs=2, random_state=SEED),
        "XGBoost": XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8,
                                colsample_bytree=0.8, n_jobs=2, random_state=SEED),
    }
    mlflow.set_experiment("RealEstate_Regression")
    reg_rows, reg_fitted, reg_pred = [], {}, {}
    print("\n=== REGRESSION: Price after 5 years (Lakhs) ===")
    # naive baseline: every property grows 5% a year (no learning). Shows what the models add.
    naive = price_raw.loc[idx_te] * 1.05 ** 5
    nb = dict(rmse=rmse(yr_te, naive), mae=float(mean_absolute_error(yr_te, naive)),
              r2=float(r2_score(yr_te, naive)))
    print(f"{'Naive 5%/yr baseline':<20} rmse={nb['rmse']:.3f} mae={nb['mae']:.3f} r2={nb['r2']:.4f}")
    with mlflow.start_run(run_name="NaiveBaseline_5pct"):
        mlflow.log_metrics(nb)
    reg_rows.append({"model": "Naive baseline (5%/yr)", **nb, "train_seconds": 0.0})
    for name, model in reg_models.items():
        with mlflow.start_run(run_name=name):
            t0 = time.time()
            model.fit(Xr_tr, yr_tr)
            secs = time.time() - t0
            p = model.predict(Xr_te)
            m = dict(rmse=rmse(yr_te, p), mae=float(mean_absolute_error(yr_te, p)),
                     r2=float(r2_score(yr_te, p)),
                     mape_pct=float(np.mean(np.abs((yr_te - p) / yr_te)) * 100),
                     train_r2=float(r2_score(yr_tr, model.predict(Xr_tr))))
            mlflow.log_params({k: v for k, v in model.get_params().items()
                               if isinstance(v, (int, float, str, bool)) or v is None})
            mlflow.log_param("n_features", len(fr))
            mlflow.log_metrics({**m, "train_seconds": secs})
            mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
        reg_rows.append({"model": name, **m, "train_seconds": secs})
        reg_fitted[name], reg_pred[name] = model, p
        print(f"{name:<20} rmse={m['rmse']:.3f} mae={m['mae']:.3f} r2={m['r2']:.4f} "
              f"(train r2 {m['train_r2']:.4f}, {secs:.0f}s)")
    reg_res = pd.DataFrame(reg_rows).set_index("model")
    best_r = reg_res.drop(index="Naive baseline (5%/yr)")["rmse"].idxmin()
    best_reg = reg_fitted[best_r]
    print(f"Best regressor: {best_r}")

    # regression plots
    p = reg_pred[best_r]
    s = np.random.RandomState(SEED).choice(len(yr_te), 5000, replace=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].scatter(yr_te.values[s], p[s], s=6, alpha=.3)
    lim = [yr_te.min(), yr_te.max()]
    ax[0].plot(lim, lim, "r--")
    ax[0].set_xlabel("Actual 5-year price (Lakhs)")
    ax[0].set_ylabel("Predicted")
    ax[0].set_title(f"Actual vs Predicted - {best_r}")
    ax[1].hist(yr_te.values - p, bins=60, color="#e07a5f")
    ax[1].set_title("Residuals (actual - predicted)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "reg_actual_vs_pred_residuals.png"), dpi=130)
    plt.close(fig)

    imp_r = importance(best_reg, fr)
    plot_importance(imp_r, f"Top 20 Features - {best_r} (5-Year Price)",
                    os.path.join(plots, "reg_feature_importance.png"))

    # ------------------------------------------------------------- SAVE
    clf_res.round(4).to_csv(os.path.join(args.outdir, "classification_results.csv"))
    reg_res.round(4).to_csv(os.path.join(args.outdir, "regression_results.csv"))
    imp_c.to_csv(os.path.join(args.outdir, "feature_importance_classifier.csv"), header=["importance"])
    imp_r.to_csv(os.path.join(args.outdir, "feature_importance_regressor.csv"), header=["importance"])
    joblib.dump(best_clf, os.path.join(args.outdir, "classifier.joblib"), compress=3)
    joblib.dump(best_reg, os.path.join(args.outdir, "regressor.joblib"), compress=3)
    meta = dict(best_classifier=best_c, best_regressor=best_r,
                classifier_metrics=clf_res.loc[best_c].round(4).to_dict(),
                regressor_metrics=reg_res.loc[best_r].round(4).to_dict(),
                naive_baseline=nb, cv_f1_mean=float(cv.mean()), cv_f1_std=float(cv.std()),
                n_train=int(len(idx_tr)), n_test=int(len(idx_te)),
                good_investment_share=float(y_c.mean()))
    with open(os.path.join(args.outdir, "model_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print("\n=== SUMMARY ===")
    print(clf_res.round(4).to_string())
    print(reg_res.round(4).to_string())
    print(f"\nSaved models, metrics and plots in {args.outdir}")
    print("MLflow UI:  mlflow ui --backend-store-uri", args.tracking_uri)


if __name__ == "__main__":
    main()
