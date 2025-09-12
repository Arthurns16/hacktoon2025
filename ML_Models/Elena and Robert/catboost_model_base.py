# catboost_model.py — FAST baseline (early stop, lighter model, no SHAP)
import numpy as np
from scipy import sparse
from catboost import CatBoostClassifier

from utils_ml import (
    evaluate_all, ensure_dirs, save_model, top_k_from_shap_values  # top_k still imported if you need it elsewhere
)

def _vstack(a, b):
    return sparse.vstack([a, b]) if sparse.issparse(a) or sparse.issparse(b) else np.vstack([a, b])

def _index_rows(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]

def _to_dense_if_sparse(X):
    return X.toarray() if sparse.issparse(X) else X

def run_catboost(
    X_train, y_train, X_valid, y_valid, X_test, y_test,
    feature_names, class_labels,
    model_dir="tuned_best_ML_pkl", shap_dir="SHAP_performance_summary",
    use_gpu=False
):
    ensure_dirs(model_dir, shap_dir)

    def _maybe_dense(X): 
        # catboost can handle dense faster; keep as is
        return _to_dense_if_sparse(X)

    # --- FAST baseline hyperparams ---
    base_params = dict(
        loss_function="MultiClass",
        eval_metric="TotalF1",
        iterations=300,              # ↓ from 500
        learning_rate=0.15,          # ↑ a bit to compensate
        depth=5,                     # ↓ from 6
        bootstrap_type="Bernoulli",  # subsampling for speed
        subsample=0.8,
        rsm=0.8,                     # column subsampling
        random_seed=42,
        use_best_model=True,         # keep best iteration found on eval_set
        od_type="Iter",              # early stopping
        od_wait=50,                  # patience
        thread_count=-1,             # or set to a smaller int if needed
        verbose=False,
        allow_writing_files=False,   # avoid disk I/O overhead
    )
    if use_gpu:
        base_params.update(dict(task_type="GPU"))  # optionally: devices="0"

    clf = CatBoostClassifier(**base_params)

    # Train WITH validation for early stopping
    clf.fit(
        _maybe_dense(X_train), y_train,
        eval_set=(_maybe_dense(X_valid), y_valid)
    )

    # Validate
    val_pred = clf.predict(_maybe_dense(X_valid))
    val_proba = clf.predict_proba(_maybe_dense(X_valid))
    _ = evaluate_all(y_valid, val_pred, val_proba, class_labels)

    # OPTIONAL: skip the expensive second fit on (train+valid) to save time.
    # If you really need it, uncomment below — but it doubles training time.
    # X_tv = _vstack(X_train, X_valid)
    # y_tv = np.concatenate([y_train, y_valid])
    # best = clf.__class__(**clf.get_params())
    # best.fit(_maybe_dense(X_tv), y_tv)
    # model_for_test = best

    model_for_test = clf  # fast path: use early-stopped model

    # Test
    y_pred = model_for_test.predict(_maybe_dense(X_test))
    y_proba = model_for_test.predict_proba(_maybe_dense(X_test))
    metrics = evaluate_all(y_test, y_pred, y_proba, class_labels)
    metrics["model"] = "CatBoost"
    metrics["best_params"] = {
        "iterations": model_for_test.get_params()["iterations"],
        "learning_rate": model_for_test.get_params()["learning_rate"],
        "depth": model_for_test.get_params()["depth"]
    }

    # FAST feature importance instead of SHAP
    try:
        import numpy as np
        import pandas as pd
        # PredictionValuesChange is fast and correlates well with SHAP for many cases
        importances = model_for_test.get_feature_importance(type="PredictionValuesChange")
        top_idx = np.argsort(importances)[::-1][:10]
        top10 = {
            "model": "CatBoost",
            "top_features": [
                {"feature": feature_names[i], "importance": float(importances[i])} 
                for i in top_idx
            ]
        }
    except Exception as e:
        top10 = None
        metrics["importance_error"] = str(e)

    save_model(model_for_test, f"{model_dir}/best_catboost.pkl")
    return metrics, top10
