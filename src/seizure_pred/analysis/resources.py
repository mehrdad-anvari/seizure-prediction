from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _numeric_summary(values: list[Any]) -> Dict[str, float]:
    numeric = [float(value) for value in values if isinstance(value, (int, float)) and np.isfinite(value)]
    if not numeric:
        return {}
    return {
        "mean": float(np.mean(numeric)),
        "min": float(np.min(numeric)),
        "max": float(np.max(numeric)),
    }


def summarize_resource_metrics(run_dir: str, *, out_dir: str | None = None) -> Dict[str, Any]:
    """Aggregate per-training-run resource artifacts for analysis."""
    run_path = Path(run_dir)
    resource_files = sorted(run_path.glob("**/resource_metrics.json"))
    records = []
    for path in resource_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                records.append(json.load(f))
        except (OSError, ValueError):
            continue

    summary: Dict[str, Any] = {
        "run_dir": str(run_path),
        "training_runs": len(records),
        "model": records[0].get("model", {}) if records else {},
        "hardware": records[0].get("hardware", {}) if records else {},
        "training": {},
    }
    if records:
        training_keys = [
            "total_wall_time_seconds",
            "total_process_cpu_time_seconds",
            "process_cpu_utilization_percent_mean",
            "system_cpu_utilization_percent_mean",
            "process_memory_rss_mb_peak",
            "gpu_utilization_percent_mean",
            "gpu_utilization_percent_peak",
            "gpu_memory_allocated_mb_peak",
            "gpu_memory_reserved_mb_peak",
        ]
        def _training_value(record: Dict[str, Any], key: str) -> Any:
            training = record.get("training", {})
            value = training.get(key)
            if key in {"gpu_memory_allocated_mb_peak", "gpu_memory_reserved_mb_peak"}:
                epoch_values = [epoch.get(key) for epoch in training.get("epochs", [])]
                numeric = [candidate for candidate in [value, *epoch_values] if isinstance(candidate, (int, float))]
                return max(numeric) if numeric else None
            return value

        summary["training"] = {
            key: _numeric_summary([_training_value(record, key) for record in records])
            for key in training_keys
        }
        summary["training"]["total_wall_time_seconds_sum"] = float(sum(
            record.get("training", {}).get("total_wall_time_seconds", 0.0) for record in records
        ))

    output_path = Path(out_dir) if out_dir is not None else run_path / "analysis"
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "resource_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
