from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    y_true = y_true.astype(int)
    y_hat = y_hat.astype(int)
    tn = int(((y_true == 0) & (y_hat == 0)).sum())
    fp = int(((y_true == 0) & (y_hat == 1)).sum())
    fn = int(((y_true == 1) & (y_hat == 0)).sum())
    tp = int(((y_true == 1) & (y_hat == 1)).sum())
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def binary_report(y_true: np.ndarray, y_hat: np.ndarray) -> Dict[str, float]:
    c = confusion_matrix(y_true, y_hat)
    tn, fp = int(c[0, 0]), int(c[0, 1])
    fn, tp = int(c[1, 0]), int(c[1, 1])

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, (precision + recall))

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion": c.tolist(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def roc_curve(y_true: np.ndarray, prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve points by sorting thresholds descending."""
    y_true = y_true.astype(int)
    prob = prob.astype(float)

    order = np.argsort(-prob)
    y = y_true[order]
    p = prob[order]

    P = max(1, int((y == 1).sum()))
    N = max(1, int((y == 0).sum()))

    tpr = []
    fpr = []
    thr = []

    tp = 0
    fp = 0
    last_p = None

    for yi, pi in zip(y, p):
        if last_p is None or pi != last_p:
            tpr.append(tp / P)
            fpr.append(fp / N)
            thr.append(pi)
            last_p = pi

        if yi == 1:
            tp += 1
        else:
            fp += 1

    # final point
    tpr.append(tp / P)
    fpr.append(fp / N)
    thr.append(-np.inf)

    return np.asarray(fpr), np.asarray(tpr), np.asarray(thr)


def pr_curve(y_true: np.ndarray, prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PR curve by sorting thresholds descending."""
    y_true = y_true.astype(int)
    prob = prob.astype(float)

    order = np.argsort(-prob)
    y = y_true[order]
    p = prob[order]

    P = max(1, int((y == 1).sum()))

    prec = []
    rec = []
    thr = []

    tp = 0
    fp = 0
    last_p = None

    for yi, pi in zip(y, p):
        if last_p is None or pi != last_p:
            precision = tp / max(1, tp + fp)
            recall = tp / P
            prec.append(precision)
            rec.append(recall)
            thr.append(pi)
            last_p = pi

        if yi == 1:
            tp += 1
        else:
            fp += 1

    precision = tp / max(1, tp + fp)
    recall = tp / P
    prec.append(precision)
    rec.append(recall)
    thr.append(-np.inf)

    return np.asarray(prec), np.asarray(rec), np.asarray(thr)


def auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoid AUC assuming x is monotonic increasing."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return 0.0
    return float(np.trapezoid(y, x))
