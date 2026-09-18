from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .constants import CANONICAL_CLASSES


def evaluate_predictions(rows: list[dict]) -> dict:
    rows = [row for row in rows if bool(row.get("valid", True))]
    if not rows:
        return {"status": "invalid", "reason": "no valid predictions"}
    truth = np.asarray([row["target"] for row in rows], dtype=np.int64)
    pred = np.asarray([row["prediction"] for row in rows], dtype=np.int64)
    support = np.bincount(truth, minlength=4)
    if np.any(support == 0):
        return {"status": "invalid", "reason": "missing class support", "support": support.tolist()}
    precision, recall, f1, class_support = precision_recall_fscore_support(truth, pred, labels=np.arange(4), zero_division=0)
    cm = confusion_matrix(truth, pred, labels=np.arange(4))
    normalized = (cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).tolist()
    majority_class = int(np.bincount(truth, minlength=4).argmax())
    majority_prediction = np.full_like(truth, majority_class)
    result = {
        "status": "complete",
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, labels=np.arange(4), average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(truth, pred, labels=np.arange(4))),
        "majority_class_baseline": {
            "class_index": majority_class,
            "class_name": CANONICAL_CLASSES[majority_class],
            "accuracy": float(accuracy_score(truth, majority_prediction)),
            "macro_f1": float(f1_score(truth, majority_prediction, labels=np.arange(4), average="macro", zero_division=0)),
        },
        "per_class": {
            name: {"precision": float(p), "recall": float(r), "f1": float(score), "support": int(n)}
            for name, p, r, score, n in zip(CANONICAL_CLASSES, precision, recall, f1, class_support)
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_row_normalized": normalized,
    }
    result.update(temporal_metrics(rows))
    return result


def temporal_metrics(rows: list[dict]) -> dict:
    by_subject: dict[str, list[dict]] = {}
    for row in rows:
        by_subject.setdefault(str(row["subject_id"]), []).append(row)
    transition_truth, transition_pred, stable_truth, stable_pred = [], [], [], []
    rate_errors, true_rates, predicted_rates = [], [], []
    for subject_rows in by_subject.values():
        subject_rows.sort(key=lambda row: int(row["epoch_index"]))
        epochs = np.asarray([row["epoch_index"] for row in subject_rows], dtype=np.int64)
        truth = np.asarray([row["target"] for row in subject_rows], dtype=np.int64)
        pred = np.asarray([row["prediction"] for row in subject_rows], dtype=np.int64)
        adjacent = np.diff(epochs) == 1
        if not adjacent.any():
            continue
        left_truth, right_truth = truth[:-1][adjacent], truth[1:][adjacent]
        left_pred, right_pred = pred[:-1][adjacent], pred[1:][adjacent]
        changed = right_truth != left_truth
        transition_truth.extend(right_truth[changed]); transition_pred.extend(right_pred[changed])
        stable_truth.extend(right_truth[~changed]); stable_pred.extend(right_pred[~changed])
        true_rates.append(float(changed.mean()))
        predicted_rates.append(float((right_pred != left_pred).mean()))
        rate_errors.append(abs(float((right_pred != left_pred).mean()) - float(changed.mean())))
    def macro(left, right):
        if not left:
            return None
        return float(f1_score(left, right, labels=np.arange(4), average="macro", zero_division=0))
    return {
        "transition_macro_f1": macro(transition_truth, transition_pred),
        "stable_macro_f1": macro(stable_truth, stable_pred),
        "true_transition_rate": None if not true_rates else float(np.mean(true_rates)),
        "predicted_transition_rate": None if not predicted_rates else float(np.mean(predicted_rates)),
        "transition_rate_aggregation": "subject-mean over adjacent valid epoch pairs",
        "transition_rate_error": None if not rate_errors else float(np.mean(rate_errors)),
    }
