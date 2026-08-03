"""Plots comparing nested-CV ensemble and inner-fold predictions."""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .io import read_jsonl
from .plots import plot_preictal_prob


_INNER_SPLIT_RE = re.compile(r"^inner_split_(\d+)$")
_UNSAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_PredictionKey = Tuple[str, int]


def _find_inner_prediction_files(split_path: Path) -> List[Tuple[int, Path]]:
    files: List[Tuple[int, Path]] = []
    for path in split_path.glob("inner_split_*/predictions.jsonl"):
        match = _INNER_SPLIT_RE.match(path.parent.name)
        if match:
            files.append((int(match.group(1)), path))
    return sorted(files, key=lambda item: item[0])


def _probability(row: Dict[str, Any]) -> float:
    if "prob" in row:
        return float(row["prob"])
    if "logit" in row:
        logit = float(row["logit"])
        # Branching keeps the sigmoid stable for very large-magnitude logits.
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        exp_logit = math.exp(logit)
        return exp_logit / (1.0 + exp_logit)
    raise ValueError("prediction row has neither 'prob' nor 'logit'")


def _preictal_rows_by_key(path: Path) -> Dict[_PredictionKey, Dict[str, Any]]:
    """Load positive rows keyed by event and stable global sample ID."""
    predictions: Dict[_PredictionKey, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        label = row.get("y_true", row.get("target"))
        if label is None or int(label) != 1:
            continue

        meta = row.get("meta")
        if not isinstance(meta, dict):
            continue
        event_id = meta.get("event_id")
        global_epoch_id = meta.get("global_epoch_id")
        epoch_index = meta.get("epoch_index_within_event")
        if event_id is None or global_epoch_id is None or epoch_index is None:
            continue

        key = (str(event_id), int(global_epoch_id))
        if key in predictions:
            raise ValueError(f"duplicate preictal sample key {key!r} in {path}")
        predictions[key] = {
            "prob": _probability(row),
            "epoch_index_within_event": int(epoch_index),
        }
    return predictions


def _safe_filename_component(value: str) -> str:
    safe = _UNSAFE_FILENAME_RE.sub("_", value).strip("._")
    return safe or "event"


def analyze_preictal_prob(
    split_dir: str | Path,
    *,
    out_dir: Optional[str | Path] = None,
    sampling_period: float = 5.0,
) -> Dict[str, Any]:
    """Plot preictal probabilities for an ensemble and every inner fold.

    Rows are aligned across files with ``(event_id, global_epoch_id)`` and are
    ordered within each event by ``epoch_index_within_event``. One plot is
    written per preictal event so unrelated events are never connected.
    """
    split_path = Path(split_dir)
    outer_path = split_path / "predictions.jsonl"
    if not outer_path.exists():
        return {"status": "missing_outer_predictions", "split_dir": str(split_path)}

    inner_files = _find_inner_prediction_files(split_path)
    if not inner_files:
        return {"status": "no_inner_splits", "split_dir": str(split_path)}

    if sampling_period <= 0:
        raise ValueError("sampling_period must be greater than zero")

    outer = _preictal_rows_by_key(outer_path)
    inner = {
        f"inner_split_{inner_idx}": _preictal_rows_by_key(path)
        for inner_idx, path in inner_files
    }
    if not outer:
        return {"status": "no_preictal_predictions", "split_dir": str(split_path)}

    outer_keys = set(outer)
    inner_keys = {name: set(predictions) for name, predictions in inner.items()}
    common_keys = set(outer_keys)
    for predictions in inner.values():
        common_keys.intersection_update(predictions)
    missing_from_inner = {
        name: len(outer_keys - keys) for name, keys in inner_keys.items()
    }
    extra_in_inner = {
        name: len(keys - outer_keys) for name, keys in inner_keys.items()
    }
    alignment = {
        "outer_preictal_samples": len(outer_keys),
        "common_preictal_samples": len(common_keys),
        "missing_from_inner": missing_from_inner,
        "extra_in_inner": extra_in_inner,
    }
    if not common_keys:
        return {
            "status": "no_aligned_preictal_predictions",
            "split_dir": str(split_path),
            "alignment": alignment,
        }
    if any(missing_from_inner.values()) or any(extra_in_inner.values()):
        warnings.warn(
            f"{split_path}: plotting {len(common_keys)} of {len(outer_keys)} "
            "outer preictal samples because prediction keys differ across "
            "inner splits; inspect the returned alignment summary",
            RuntimeWarning,
            stacklevel=2,
        )

    event_ids = sorted({event_id for event_id, _ in common_keys})
    output_path = Path(out_dir) if out_dir is not None else split_path / "analysis"
    output_path.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, str] = {}
    aligned_samples: Dict[str, int] = {}
    for event_id in event_ids:
        event_keys = sorted(
            (key for key in common_keys if key[0] == event_id),
            key=lambda key: (
                outer[key]["epoch_index_within_event"],
                key[1],
            ),
        )
        epoch_index = np.asarray(
            [outer[key]["epoch_index_within_event"] for key in event_keys],
            dtype=np.float64,
        )
        x_minutes = epoch_index * float(sampling_period) / 60.0
        ensemble_prob = np.asarray([outer[key]["prob"] for key in event_keys])
        inner_prob = {
            name: np.asarray([predictions[key]["prob"] for key in event_keys])
            for name, predictions in inner.items()
        }

        split_name = _safe_filename_component(split_path.name)
        if len(event_ids) == 1:
            filename = f"preictal_prob_{split_name}.png"
        else:
            filename = (
                f"preictal_prob_{split_name}_"
                f"{_safe_filename_component(event_id)}.png"
            )
        save_path = output_path / filename
        plot_preictal_prob(
            x_minutes,
            ensemble_prob,
            inner_prob,
            save_path=str(save_path),
            title=f"{split_path.name} / {event_id}: preictal probabilities",
        )
        artifacts[event_id] = str(save_path)
        aligned_samples[event_id] = len(event_keys)

    return {
        "status": "ok",
        "split_dir": str(split_path),
        "n_inner_splits": len(inner),
        "n_events": len(event_ids),
        "aligned_samples": aligned_samples,
        "alignment": alignment,
        "artifacts": artifacts,
    }


# Keep the requested spelling available as a short public alias.
ananlyze_preictal_prob = analyze_preictal_prob
