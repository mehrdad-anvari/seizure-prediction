
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float]]


@dataclass
class PostprocessResult:
    """Outputs of a postprocessing pipeline.

    - probs: float array in [0,1] after smoothing/hysteresis (if applicable)
    - pred:  int array {0,1} final decision
    """
    probs: np.ndarray
    pred: np.ndarray


class Postprocessor:
    """Base class for postprocessing window-level probabilities.

    Input:
        probs: shape (N,) float in [0,1]
        meta: optional per-window metadata (list of dicts) or any auxiliary info

    Output:
        PostprocessResult(probs=..., pred=...)
    """
    def __call__(self, probs: ArrayLike, meta: Any = None) -> PostprocessResult:  # pragma: no cover
        raise NotImplementedError


class Threshold(Postprocessor):
    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)

    def __call__(self, probs: ArrayLike, meta: Any = None) -> PostprocessResult:
        p = np.asarray(probs, dtype=np.float32)
        pred = (p >= self.threshold).astype(np.int64)
        return PostprocessResult(probs=p, pred=pred)


class MovingAverage(Postprocessor):
    """Simple causal moving average smoother + threshold."""
    def __init__(self, window: int = 5, threshold: float = 0.5, centered: bool = False):
        self.window = int(max(1, window))
        self.threshold = float(threshold)
        self.centered = bool(centered)

    def __call__(self, probs: ArrayLike, meta: Any = None) -> PostprocessResult:
        p = np.asarray(probs, dtype=np.float32)
        if p.size == 0:
            return PostprocessResult(probs=p, pred=p.astype(np.int64))

        w = self.window
        kernel = np.ones(w, dtype=np.float32) / w
        if self.centered:
            # centered convolution (pads on both sides)
            pad = w // 2
            pp = np.pad(p, (pad, w - 1 - pad), mode="edge")
            sm = np.convolve(pp, kernel, mode="valid")
        else:
            # causal: average over last w values
            pp = np.pad(p, (w - 1, 0), mode="edge")
            sm = np.convolve(pp, kernel, mode="valid")
        pred = (sm >= self.threshold).astype(np.int64)
        return PostprocessResult(probs=sm, pred=pred)


class Hysteresis(Postprocessor):
    """Hysteresis thresholding + duration constraints.

    A robust detector for noisy window probabilities:
    - turns ON when probs >= on_threshold
    - stays ON until probs <= off_threshold
    - optional min_on/min_off enforce minimum run lengths (in windows)
    """
    def __init__(
        self,
        on_threshold: float = 0.6,
        off_threshold: float = 0.4,
        min_on: int = 1,
        min_off: int = 1,
        smoothing_window: int = 1,
        smoothing_centered: bool = False,
    ):
        self.on_threshold = float(on_threshold)
        self.off_threshold = float(off_threshold)
        self.min_on = int(max(1, min_on))
        self.min_off = int(max(1, min_off))
        self.smoothing_window = int(max(1, smoothing_window))
        self.smoothing_centered = bool(smoothing_centered)

        if self.off_threshold > self.on_threshold:
            raise ValueError("off_threshold must be <= on_threshold for hysteresis.")

    def _smooth(self, p: np.ndarray) -> np.ndarray:
        if self.smoothing_window <= 1:
            return p
        return MovingAverage(self.smoothing_window, threshold=0.5, centered=self.smoothing_centered)(p).probs

    @staticmethod
    def _enforce_min_run(pred: np.ndarray, value: int, min_len: int) -> np.ndarray:
        """Remove short runs of `value` by flipping them."""
        if pred.size == 0:
            return pred
        out = pred.copy()
        n = len(out)
        i = 0
        while i < n:
            if out[i] != value:
                i += 1
                continue
            j = i
            while j < n and out[j] == value:
                j += 1
            run_len = j - i
            if run_len < min_len:
                out[i:j] = 1 - value
            i = j
        return out

    def __call__(self, probs: ArrayLike, meta: Any = None) -> PostprocessResult:
        p0 = np.asarray(probs, dtype=np.float32)
        p = self._smooth(p0)

        pred = np.zeros_like(p, dtype=np.int64)
        state = 0
        for i, val in enumerate(p):
            if state == 0:
                if val >= self.on_threshold:
                    state = 1
            else:
                if val <= self.off_threshold:
                    state = 0
            pred[i] = state

        # enforce min on/off durations
        pred = self._enforce_min_run(pred, value=1, min_len=self.min_on)
        pred = self._enforce_min_run(pred, value=0, min_len=self.min_off)
        return PostprocessResult(probs=p, pred=pred)


class Compose(Postprocessor):
    """Compose multiple postprocessors; last one determines final pred."""
    def __init__(self, steps: Sequence[Postprocessor]):
        self.steps = list(steps)

    def __call__(self, probs: ArrayLike, meta: Any = None) -> PostprocessResult:
        p = np.asarray(probs, dtype=np.float32)
        last = PostprocessResult(probs=p, pred=(p >= 0.5).astype(np.int64))
        for step in self.steps:
            last = step(last.probs, meta=meta)
        return last


def events_from_binary(pred: np.ndarray) -> list[tuple[int, int]]:
    """Convert a 0/1 sequence into [(start_idx, end_idx_exclusive), ...] for runs of 1."""
    pred = np.asarray(pred, dtype=np.int64)
    events: list[tuple[int, int]] = []
    n = pred.size
    i = 0
    while i < n:
        if pred[i] == 0:
            i += 1
            continue
        j = i
        while j < n and pred[j] == 1:
            j += 1
        events.append((i, j))
        i = j
    return events


def match_events_iou(
    pred_events: list[tuple[int, int]],
    true_events: list[tuple[int, int]],
    iou_threshold: float = 0.1,
) -> tuple[int, int, int]:
    """Greedy IoU matching: returns (tp, fp, fn)."""
    used = [False] * len(true_events)

    def iou(a, b) -> float:
        inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        union = (a[1] - a[0]) + (b[1] - b[0]) - inter
        return 0.0 if union == 0 else inter / union

    tp = 0
    for pe in pred_events:
        best = -1
        best_iou = 0.0
        for i, te in enumerate(true_events):
            if used[i]:
                continue
            v = iou(pe, te)
            if v > best_iou:
                best_iou = v
                best = i
        if best >= 0 and best_iou >= iou_threshold:
            used[best] = True
            tp += 1

    fp = len(pred_events) - tp
    fn = len(true_events) - tp
    return tp, fp, fn
