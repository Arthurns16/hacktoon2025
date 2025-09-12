# xgb_model.py — FAST baseline (no tuning)
import numpy as np
from scipy import sparse
import shap
from xgboost import XGBClassifier

from utils_ml import (
    evaluate_all, ensure_dirs, save_model, top_k_from_shap_values
)

def _vstack(a, b):
    return sparse.vstack([a, b]) if sparse.issparse(a) or sparse.issparse(b) else np.vstack([a, b])

def _index_rows(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]

def _sample_rows(X, max_rows=800, random_state=42):
    n = X.shape[0]
    if n <= max_rows:
        return X
    rng = np.random.RandomState(random_state)
    idx = rng.choice(n, size=max_rows, replace=False)
    return _index_rows(X, idx)

def _to_dense_if_sparse(X):
    return X.toarray() if sparse.issparse(X) else X

def run_xgboost(
    X_train, y_train, X_valid, y_valid, X_test, y_test,
    feature_names, class_labels,
    model_dir="tuned_best_ML_pkl", shap_dir="SHAP_performance_summary"
):
    ensure_dirs(model_dir, shap_dir)

    clf = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=1.0,
        colsample_bytree=1.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_valid)
    val_proba = clf.predict_proba(X_valid)
    _ = evaluate_all(y_valid, val_pred, val_proba, class_labels)

    X_tv = _vstack(X_train, X_valid)
    y_tv = np.concatenate([y_train, y_valid])
    best = clf.__class__(**clf.get_params())
    best.fit(X_tv, y_tv)

    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)
    metrics = evaluate_all(y_test, y_pred, y_proba, class_labels)
    metrics["model"] = "XGBoost"
    metrics["best_params"] = {k: best.get_params()[k] for k in
                              ["n_estimators","learning_rate","max_depth","subsample","colsample_bytree"]}

    try:
        X_shap = _to_dense_if_sparse(_sample_rows(X_tv, 600))
        expl = shap.TreeExplainer(best)
        shap_vals = expl.shap_values(X_shap)
        if isinstance(shap_vals, list):
            abs_mean = np.mean([np.mean(np.abs(sv), axis=0) for sv in shap_vals], axis=0)
        else:
            abs_mean = np.mean(np.abs(shap_vals), axis=0)
        top10 = top_k_from_shap_values(abs_mean, feature_names, k=10)
        top10["model"] = "XGBoost"
    except Exception as e:
        top10 = None
        metrics["shap_error"] = str(e)

    save_model(best, f"{model_dir}/best_xgboost.pkl")
    return metrics, top10
