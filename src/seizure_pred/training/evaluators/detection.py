
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from seizure_pred.inference.postprocess import (
    Postprocessor,
    Threshold,
    events_from_binary,
    match_events_iou,
)
from seizure_pred.training.registries import EVALUATORS, POSTPROCESSORS


@dataclass
class DetectionEventMetrics:
    window_f1: float
    window_precision: float
    window_recall: float
    event_f1: float
    event_precision: float
    event_recall: float
    tp_events: int
    fp_events: int
    fn_events: int


def _binary_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


@EVALUATORS.register("detection_events", help="Detection evaluator: window metrics + event-based metrics after postprocessing.")
class DetectionEventsEvaluator:
    """Evaluator for detection tasks based on window labels.

    Assumptions / contracts:
      - Model outputs window logits for seizure-vs-not (binary)
      - y_true is window-level 0/1 seizure label
      - Event metrics are derived by grouping consecutive 1 windows into events.
    """

    def __init__(
        self,
        loss_fn: nn.Module,
        postprocess: Optional[Dict[str, Any]] = None,
        iou_threshold: float = 0.1,
    ):
        self.loss_fn = loss_fn
        self.iou_threshold = float(iou_threshold)

        if postprocess is None:
            self.postprocessor: Postprocessor = Threshold(0.5)
        else:
            # postprocess: {"name":"hysteresis","kwargs":{...}}
            name = postprocess.get("name", "threshold")
            kwargs = postprocess.get("kwargs", {}) or {}
            self.postprocessor = POSTPROCESSORS.create(name, **kwargs)

    @torch.no_grad()
    def evaluate(self, model: nn.Module, loader, device: str) -> Dict[str, Any]:
        model.eval()
        losses = []
        all_probs = []
        all_y = []

        for x, y, meta in loader:
            x = x.to(device)
            y = y.to(device).float()
            logits = model(x).squeeze(-1)
            loss = self.loss_fn(logits, y)
            losses.append(loss.detach().float().cpu().item())

            probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
            all_probs.append(probs)
            all_y.append(y.detach().cpu().numpy().astype(np.int64))

        probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0,), dtype=np.float32)
        y_true = np.concatenate(all_y, axis=0) if all_y else np.zeros((0,), dtype=np.int64)

        post = self.postprocessor(probs)
        y_pred = post.pred.astype(np.int64)

        # window-level metrics
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        w_prec, w_rec, w_f1 = _binary_prf(tp, fp, fn)

        # event-level metrics
        pred_events = events_from_binary(y_pred)
        true_events = events_from_binary(y_true)
        tp_e, fp_e, fn_e = match_events_iou(pred_events, true_events, iou_threshold=self.iou_threshold)
        e_prec, e_rec, e_f1 = _binary_prf(tp_e, fp_e, fn_e)

        out = {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "window_precision": w_prec,
            "window_recall": w_rec,
            "window_f1": w_f1,
            "event_precision": e_prec,
            "event_recall": e_rec,
            "event_f1": e_f1,
            "tp_events": tp_e,
            "fp_events": fp_e,
            "fn_events": fn_e,
            "iou_threshold": self.iou_threshold,
            "postprocess": getattr(self.postprocessor, "__class__", type(self.postprocessor)).__name__,
        }
        return out
