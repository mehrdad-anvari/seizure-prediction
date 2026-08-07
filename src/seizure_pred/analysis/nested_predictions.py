"""Plots comparing nested-CV ensemble and inner-fold predictions."""

from __future__ import annotations

import json
import math
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .io import read_jsonl
from .plots import (
    plot_interictal_combined,
    plot_preictal_prob,
    plot_prob_vs_pp_scatter,
    plot_prob_vs_pp_scatter_combined,
)


_INNER_SPLIT_RE = re.compile(r"^inner_split_(\d+)$")
_UNSAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_EventKey = Tuple[str, Optional[int], Optional[float], Optional[float]]
_PredictionKey = Tuple[_EventKey, int]


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
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        exp_logit = math.exp(logit)
        return exp_logit / (1.0 + exp_logit)
    raise ValueError("prediction row has neither 'prob' nor 'logit'")


def _rows_by_key(
    path: Path,
    *,
    target: int,
    event_type: str,
) -> Dict[_PredictionKey, Dict[str, Any]]:
    """Load rows for one class, keyed by event and stable global sample ID."""
    predictions: Dict[_PredictionKey, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        label = row.get("y_true", row.get("target"))
        if label is None or int(label) != target:
            continue

        meta = row.get("meta")
        if not isinstance(meta, dict):
            continue
        meta_label = meta.get("label")
        if event_type == "interictal" and meta_label not in (None, "interictal"):
            continue
        event_id = meta.get("event_id")
        global_epoch_id = meta.get("global_epoch_id")
        epoch_index = meta.get("epoch_index_within_event")
        if event_id is None or global_epoch_id is None or epoch_index is None:
            continue

        n_segments = meta.get("n_segments_in_event")
        onset_sec = meta.get("onset_sec")
        duration_sec = meta.get("duration_sec")
        event_key: _EventKey = (
            str(event_id),
            int(n_segments) if n_segments is not None else None,
            float(onset_sec) if onset_sec is not None else None,
            float(duration_sec) if duration_sec is not None else None,
        )
        key = (event_key, int(global_epoch_id))
        if key in predictions:
            raise ValueError(f"duplicate {event_type} sample key {key!r} in {path}")
        predictions[key] = {
            "prob": _probability(row),
            "epoch_index_within_event": int(epoch_index),
        }
    return predictions


def _safe_filename_component(value: str) -> str:
    safe = _UNSAFE_FILENAME_RE.sub("_", value).strip("._")
    return safe or "event"


def _event_sort_key(event_key: _EventKey) -> Tuple[Any, ...]:
    event_id, n_segments, onset_sec, duration_sec = event_key
    return (
        event_id,
        onset_sec if onset_sec is not None else float("inf"),
        duration_sec if duration_sec is not None else float("inf"),
        n_segments if n_segments is not None else float("inf"),
    )


def _analyze_event_prob(
    split_dir: str | Path,
    *,
    target: int,
    event_type: str,
    out_dir: Optional[str | Path] = None,
    sampling_period: float = 5.0,
) -> Dict[str, Any]:
    """Plot model probabilities for one event type across nested-CV folds.

    Rows are aligned across files with ``(event_id, global_epoch_id)`` and are
    ordered within each event by ``epoch_index_within_event``. One plot is
    written per event so unrelated time sequences are never connected.
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

    outer = _rows_by_key(outer_path, target=target, event_type=event_type)
    inner = {
        f"inner_split_{inner_idx}": _rows_by_key(
            path,
            target=target,
            event_type=event_type,
        )
        for inner_idx, path in inner_files
    }
    if not outer:
        return {
            "status": f"no_{event_type}_predictions",
            "split_dir": str(split_path),
        }

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
        f"outer_{event_type}_samples": len(outer_keys),
        f"common_{event_type}_samples": len(common_keys),
        "missing_from_inner": missing_from_inner,
        "extra_in_inner": extra_in_inner,
    }
    if not common_keys:
        return {
            "status": f"no_aligned_{event_type}_predictions",
            "split_dir": str(split_path),
            "alignment": alignment,
        }
    if any(missing_from_inner.values()) or any(extra_in_inner.values()):
        warnings.warn(
            f"{split_path}: plotting {len(common_keys)} of {len(outer_keys)} "
            f"outer {event_type} samples because prediction keys differ across "
            "inner splits; inspect the returned alignment summary",
            RuntimeWarning,
            stacklevel=2,
        )

    event_keys = sorted(
        {event_key for event_key, _ in common_keys},
        key=_event_sort_key,
    )
    event_id_counts = Counter(event_key[0] for event_key in event_keys)
    event_occurrences: Dict[str, int] = defaultdict(int)
    output_path = Path(out_dir) if out_dir is not None else split_path / "analysis"
    output_path.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, str] = {}
    aligned_samples: Dict[str, int] = {}

    # ---- interictal: combine all events into a single figure ----------
    if event_type == "interictal":
        combined_events: list = []
        cumulative_offset = 0
        for event_key in event_keys:
            raw_event_id = event_key[0]
            event_occurrences[raw_event_id] += 1
            if event_id_counts[raw_event_id] == 1:
                display_event_id = raw_event_id
            else:
                display_event_id = (
                    f"{raw_event_id}_event_{event_occurrences[raw_event_id]}"
                )

            sample_keys = sorted(
                (key for key in common_keys if key[0] == event_key),
                key=lambda key: (
                    outer[key]["epoch_index_within_event"],
                    key[1],
                ),
            )
            # Use raw epoch index (not minutes) for the x-axis.
            epoch_index = np.asarray(
                [outer[key]["epoch_index_within_event"] for key in sample_keys],
                dtype=np.float64,
            )
            x_index = epoch_index + cumulative_offset
            cumulative_offset += int(epoch_index[-1]) + 1

            ensemble_prob = np.asarray([outer[key]["prob"] for key in sample_keys])
            inner_prob = {
                name: np.asarray([predictions[key]["prob"] for key in sample_keys])
                for name, predictions in inner.items()
            }

            combined_events.append({
                "x_index": x_index,
                "ensemble_prob": ensemble_prob,
                "inner_prob": inner_prob,
                "label": display_event_id,
            })
            aligned_samples[display_event_id] = len(sample_keys)

        split_name = _safe_filename_component(split_path.name)
        filename = f"interictal_prob_combined_{split_name}.png"
        save_path = output_path / filename
        plot_interictal_combined(
            combined_events,
            save_path=str(save_path),
            title=(
                f"{split_path.name}: "
                "interictal predicted preictal probability"
            ),
        )
        artifacts["interictal_combined"] = str(save_path)

    else:
        # ---- preictal: one plot per event (original behaviour) ----------
        for event_key in event_keys:
            raw_event_id = event_key[0]
            event_occurrences[raw_event_id] += 1
            if event_id_counts[raw_event_id] == 1:
                display_event_id = raw_event_id
            else:
                display_event_id = (
                    f"{raw_event_id}_event_{event_occurrences[raw_event_id]}"
                )

            sample_keys = sorted(
                (key for key in common_keys if key[0] == event_key),
                key=lambda key: (
                    outer[key]["epoch_index_within_event"],
                    key[1],
                ),
            )
            epoch_index = np.asarray(
                [outer[key]["epoch_index_within_event"] for key in sample_keys],
                dtype=np.float64,
            )
            x_minutes = epoch_index * float(sampling_period) / 60.0
            ensemble_prob = np.asarray([outer[key]["prob"] for key in sample_keys])
            inner_prob = {
                name: np.asarray([predictions[key]["prob"] for key in sample_keys])
                for name, predictions in inner.items()
            }

            split_name = _safe_filename_component(split_path.name)
            if len(event_keys) == 1:
                filename = f"{event_type}_prob_{split_name}.png"
            else:
                filename = (
                    f"{event_type}_prob_{split_name}_"
                    f"{_safe_filename_component(display_event_id)}.png"
                )
            save_path = output_path / filename
            plot_preictal_prob(
                x_minutes,
                ensemble_prob,
                inner_prob,
                save_path=str(save_path),
                title=(
                    f"{split_path.name} / {display_event_id}: "
                    "predicted preictal probability"
                ),
                event_type=event_type,
            )
            artifacts[display_event_id] = str(save_path)
            aligned_samples[display_event_id] = len(sample_keys)

    return {
        "status": "ok",
        "split_dir": str(split_path),
        "n_inner_splits": len(inner),
        "n_events": len(event_keys),
        "aligned_samples": aligned_samples,
        "alignment": alignment,
        "artifacts": artifacts,
    }


def analyze_preictal_prob(
    split_dir: str | Path,
    *,
    out_dir: Optional[str | Path] = None,
    sampling_period: float = 5.0,
) -> Dict[str, Any]:
    """Compare preictal-event predictions across inner folds and ensemble."""
    return _analyze_event_prob(
        split_dir,
        target=1,
        event_type="preictal",
        out_dir=out_dir,
        sampling_period=sampling_period,
    )


def analyze_interictal_prob(
    split_dir: str | Path,
    *,
    out_dir: Optional[str | Path] = None,
    sampling_period: float = 5.0,
) -> Dict[str, Any]:
    """Compare interictal-event predictions across inner folds and ensemble."""
    return _analyze_event_prob(
        split_dir,
        target=0,
        event_type="interictal",
        out_dir=out_dir,
        sampling_period=sampling_period,
    )


# ---- P-P scatter helpers ----------------------------------------------------


def _extract_pp_data(
    path: Path,
    *,
    target: int,
    event_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Extract ``(prob, pp_max, pp_mean, epoch_index)`` from predictions.jsonl.

    Filters rows by *target* and *event_type*, then reads the model
    probability and EEG peak-to-peak features (pp_max, pp_mean) from meta.
    Returns None if no matching rows exist.
    """
    probs: list[float] = []
    pp_max_list: list[float] = []
    pp_mean_list: list[float] = []
    epoch_indices: list[int] = []

    for row in read_jsonl(path):
        label = row.get("y_true", row.get("target"))
        if label is None or int(label) != target:
            continue

        meta = row.get("meta")
        if not isinstance(meta, dict):
            continue
        meta_label = meta.get("label")
        if event_type == "interictal" and meta_label not in (None, "interictal"):
            continue

        pp_max_val = meta.get("pp_max")
        pp_mean_val = meta.get("pp_mean")
        epoch_idx = meta.get("epoch_index_within_event")
        if pp_max_val is None or pp_mean_val is None or epoch_idx is None:
            continue

        probs.append(_probability(row))
        pp_max_list.append(float(pp_max_val))
        pp_mean_list.append(float(pp_mean_val))
        epoch_indices.append(int(epoch_idx))

    if not probs:
        return None

    return (
        np.asarray(probs, dtype=np.float64),
        np.asarray(pp_max_list, dtype=np.float64),
        np.asarray(pp_mean_list, dtype=np.float64),
        np.asarray(epoch_indices, dtype=np.float64),
    )


def analyze_interictal_pp_scatter(
    split_dir: str | Path,
    *,
    out_dir: Optional[str | Path] = None,
    sampling_period: float = 5.0,
) -> Dict[str, Any]:
    """Scatter plot: model prob vs EEG pp_max / pp_mean (interictal only).

    Points are coloured by epoch index for temporal context.
    """
    split_path = Path(split_dir)
    outer_path = split_path / "predictions.jsonl"
    if not outer_path.exists():
        return {"status": "missing_outer_predictions", "split_dir": str(split_path)}

    if sampling_period <= 0:
        raise ValueError("sampling_period must be greater than zero")

    data = _extract_pp_data(outer_path, target=0, event_type="interictal")
    if data is None:
        return {"status": "no_interictal_predictions", "split_dir": str(split_path)}

    prob, pp_max, pp_mean, epoch_index = data

    output_path = Path(out_dir) if out_dir is not None else split_path / "analysis"
    output_path.mkdir(parents=True, exist_ok=True)

    split_name = _safe_filename_component(split_path.name)
    filename = f"interictal_pp_scatter_{split_name}.png"
    save_path = output_path / filename
    plot_prob_vs_pp_scatter(
        prob, pp_max, pp_mean,
        save_path=str(save_path),
        title=(
            f"{split_path.name}: "
            "interictal — model prob vs EEG P-P"
        ),
        event_type="interictal",
        x_index=epoch_index,
    )

    return {
        "status": "ok",
        "split_dir": str(split_path),
        "n_interictal_samples": len(prob),
        "artifacts": {"interictal_pp_scatter": str(save_path)},
    }


def analyze_pp_scatter_combined(
    split_dir: str | Path,
    *,
    out_dir: Optional[str | Path] = None,
    sampling_period: float = 5.0,
) -> Dict[str, Any]:
    """Scatter plot with interictal and preictal on the same axes.

    Extracts model prob and EEG pp_max / pp_mean for both event types and
    plots them colour-coded by event type (interictal=blue, preictal=red).
    """
    split_path = Path(split_dir)
    outer_path = split_path / "predictions.jsonl"
    if not outer_path.exists():
        return {"status": "missing_outer_predictions", "split_dir": str(split_path)}

    if sampling_period <= 0:
        raise ValueError("sampling_period must be greater than zero")

    interictal_data = _extract_pp_data(outer_path, target=0, event_type="interictal")
    preictal_data = _extract_pp_data(outer_path, target=1, event_type="preictal")

    if interictal_data is None and preictal_data is None:
        return {"status": "no_predictions", "split_dir": str(split_path)}

    # Keep only (prob, pp_max, pp_mean) — drop epoch_index.
    interictal_tuple = interictal_data[:3] if interictal_data is not None else None
    preictal_tuple = preictal_data[:3] if preictal_data is not None else None

    output_path = Path(out_dir) if out_dir is not None else split_path / "analysis"
    output_path.mkdir(parents=True, exist_ok=True)

    split_name = _safe_filename_component(split_path.name)
    filename = f"pp_scatter_combined_{split_name}.png"
    save_path = output_path / filename
    plot_prob_vs_pp_scatter_combined(
        interictal_tuple,
        preictal_tuple,
        save_path=str(save_path),
        title=(
            f"{split_path.name}: "
            "model prob vs EEG P-P"
        ),
    )

    n_interictal = len(interictal_data[0]) if interictal_data is not None else 0
    n_preictal = len(preictal_data[0]) if preictal_data is not None else 0

    return {
        "status": "ok",
        "split_dir": str(split_path),
        "n_interictal_samples": n_interictal,
        "n_preictal_samples": n_preictal,
        "artifacts": {"pp_scatter_combined": str(save_path)},
    }


# Keep the requested spelling available as a short public alias.
ananlyze_preictal_prob = analyze_preictal_prob
