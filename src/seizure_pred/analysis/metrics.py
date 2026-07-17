from __future__ import annotations

from typing import Dict, Optional, Tuple

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


def moving_average_segmented(probs: np.ndarray, y: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Apply moving average smoothing within contiguous regions of equal labels.

    This avoids smoothing across boundary transitions between interictal and preictal.
    """
    if len(probs) < window_size or window_size <= 1:
        return probs.copy()

    y = np.asarray(y)
    smoothed_probs = probs.copy().astype(float)

    # Find contiguous regions of identical labels
    regions = []
    start = 0
    for i in range(1, len(y)):
        if y[i] != y[i - 1]:
            regions.append((start, i))
            start = i
    regions.append((start, len(y)))  # Last region

    # Apply moving average within each region
    for start, end in regions:
        region_len = end - start
        if region_len >= window_size:
            for i in range(start, end):
                win_start = max(start, i - window_size + 1)
                win_end = i + 1
                smoothed_probs[i] = np.mean(probs[win_start:win_end])
        else:
            for i in range(start, end):
                win_start = start
                win_end = i + 1
                smoothed_probs[i] = np.mean(probs[win_start:win_end])

    return smoothed_probs


def clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sampling_period: float = 5.0,
    suppression_duration: Optional[int] = None,
) -> dict[str, float]:
    """Compute clinical event-level metrics: sensitivity and FPR per hour.

    - Sensitivity: 1.0 if at least one preictal segment (y_true == 1) has a
      positive prediction (y_pred == 1), 0.0 otherwise. NaN if no preictal.
    - fpr_per_hour: false positives per hour computed over interictal samples
      (y_true == 0) only.
    - fpr_per_hour_suppressed (optional): same as fpr_per_hour but after applying
      a suppression window. Once an alarm fires, subsequent positives within
      ``suppression_duration`` windows are not counted as new false alarms
      (mirrors the legacy "FPR_sup" metric). Only computed when
      ``suppression_duration`` is a positive int.

    Parameters
    ----------
    suppression_duration:
        Number of consecutive windows to suppress after any alarm (positive
        prediction). ``None`` or <= 0 disables suppression.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict[str, float] = {}

    # Sensitivity: at least one preictal detected
    has_preictal = np.any(y_true == 1)
    if has_preictal:
        detected = np.any((y_true == 1) & (y_pred == 1))
        metrics["sensitivity"] = 1.0 if detected else 0.0
    else:
        metrics["sensitivity"] = float("nan")

    # FPR per hour - computed over interictal time only
    def _fpr_per_hour(pred: np.ndarray) -> float:
        interictal_mask = (y_true == 0)
        false_positives = np.sum(interictal_mask & (pred == 1))
        interictal_samples = np.sum(interictal_mask)
        interictal_hours = (interictal_samples * sampling_period) / 3600.0
        return float(false_positives / interictal_hours) if interictal_hours > 0 else float("nan")

    metrics["fpr_per_hour"] = _fpr_per_hour(y_pred)

    # Suppression-based FPR: after any alarm, suppress the next
    # `suppression_duration` windows from counting as new alarms.
    if suppression_duration is not None and int(suppression_duration) > 0:
        sup = int(suppression_duration)
        sup_pred = y_pred.copy()
        i = 0
        n = len(sup_pred)
        while i < n:
            if sup_pred[i] == 1:
                # suppress following `sup` windows (turn them off for counting)
                end = min(n, i + 1 + sup)
                sup_pred[i + 1:end] = 0
                i = end
            else:
                i += 1
        metrics["fpr_per_hour_suppressed"] = _fpr_per_hour(sup_pred)
    else:
        metrics["fpr_per_hour_suppressed"] = metrics["fpr_per_hour"]

    return metrics
