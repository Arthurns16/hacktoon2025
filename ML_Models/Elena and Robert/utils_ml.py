# utils_ml.py
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score
)
from sklearn.model_selection import StratifiedKFold, KFold
from joblib import dump

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def save_model(model, path):
    dump(model, path)


def make_safe_stratified_kfold(y, max_splits=3, random_state=42):
    """Choose n_splits <= max_splits so every class has at least 2 samples per fold."""
    y = np.asarray(y)
    # count per class
    _, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    # upper bound so each fold receives >=1 sample per class (be conservative)
    n_splits = min(max_splits, max(2, min_count))  # at least 2
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def make_robust_cv(y, max_splits=3, random_state=42):
    """
    Return a CV splitter that is stratified when feasible, else plain KFold.
    """
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    if min_count >= 2:
        n_splits = min(max_splits, min_count)
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        # Can't stratify at all — fall back to plain KFold
        return KFold(n_splits=2, shuffle=True, random_state=random_state)

def cv_mask_min_count(y, min_count=2):
    """
    Boolean mask: keep only samples of classes that appear at least `min_count` times in y.
    Use this to build a clean subset for CV *only*.
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    keep_classes = classes[counts >= min_count]
    return np.isin(y, keep_classes)


def evaluate_all(y_true, y_pred, y_proba, labels=None):
    """
    Robust metrics computation for multiclass.
    - Handles cases where some classes are absent in y_true for a given split.
    - For AUC, tries full class set from y_proba; falls back to present classes if needed.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {}
    metrics["precision_weighted"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["recall_weighted"]    = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["f1_weighted"]        = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["precision_macro"]    = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"]       = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_macro"]           = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["kappa"]              = cohen_kappa_score(y_true, y_pred)

    # Confusion matrix labels: prefer provided labels if they match proba width; else fall back
    n_classes = None
    if y_proba is not None and hasattr(y_proba, "shape") and y_proba.ndim == 2:
        n_classes = y_proba.shape[1]

    labels_cm = None
    if labels is not None:
        labels_arr = np.asarray(labels)
        if (n_classes is None) or (len(labels_arr) == n_classes):
            labels_cm = labels_arr

    if labels_cm is None:
        labels_cm = np.unique(np.concatenate([y_true, y_pred]))

    cm = confusion_matrix(y_true, y_pred, labels=labels_cm)
    metrics["confusion_matrix"] = cm

    # AUC (macro & weighted). Need probs and at least 2 unique classes in y_true
    metrics["auc_macro"] = np.nan
    metrics["auc_weighted"] = np.nan
    if y_proba is not None and n_classes is not None and len(np.unique(y_true)) > 1:
        # Try with full class set (columns of y_proba)
        all_labels = np.arange(n_classes)
        try:
            metrics["auc_macro"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro", labels=all_labels
            )
            metrics["auc_weighted"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="weighted", labels=all_labels
            )
        except ValueError:
            # Fallback: restrict to classes present in y_true
            present = np.unique(y_true)
            if present.size >= 2:
                try:
                    y_score_sub = y_proba[:, present]
                    metrics["auc_macro"] = roc_auc_score(
                        y_true, y_score_sub, multi_class="ovr", average="macro", labels=present
                    )
                    metrics["auc_weighted"] = roc_auc_score(
                        y_true, y_score_sub, multi_class="ovr", average="weighted", labels=present
                    )
                except Exception:
                    pass  # leave NaN
    return metrics

def top_k_from_shap_values(abs_mean_importance, feature_names, k=10):
    order = np.argsort(abs_mean_importance)[::-1]
    top_idx = order[:k]
    return pd.DataFrame({
        "feature": np.array(feature_names)[top_idx],
        "abs_mean_shap": abs_mean_importance[top_idx]
    })
