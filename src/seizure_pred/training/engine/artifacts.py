from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    return str(x)


class ArtifactWriter:
    """Stable artifact writer for training and inference."""

    def __init__(self, run_dir: str, *, schema_version: int = SCHEMA_VERSION):
        self.run_dir = run_dir
        self.schema_version = int(schema_version)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    @property
    def checkpoints_dir(self) -> str:
        d = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(d, exist_ok=True)
        return d

    def _path(self, name: str) -> str:
        return os.path.join(self.run_dir, name)

    def write_schema(self) -> None:
        with open(self._path("schema.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": self.schema_version,
                    "created_at": _utc_now_iso(),
                    "artifacts": {
                        "config": "config.json",
                        "history": "history.jsonl",
                        "metrics": "metrics.json",
                        "predictions": "predictions.jsonl",
                        "checkpoints_dir": "checkpoints/",
                    },
                },
                f,
                indent=2,
                default=_json_default,
            )

    def write_config(self, cfg: Any) -> None:
        payload = asdict(cfg) if is_dataclass(cfg) else cfg
        with open(self._path("config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"schema_version": self.schema_version, "written_at": _utc_now_iso(), "config": payload},
                f,
                indent=2,
                default=_json_default,
            )

    def append_history(self, row: Dict[str, Any]) -> None:
        out = dict(row)
        out.setdefault("schema_version", self.schema_version)
        out.setdefault("written_at", _utc_now_iso())
        with open(self._path("history.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(out, default=_json_default) + "\n")

    def write_metrics(self, metrics: Dict[str, Any], *, filename: str = "metrics.json") -> None:
        """Write a metrics JSON file.

        By default writes to `metrics.json`, but callers may write additional
        metric files (e.g., `test_metrics.json`) by providing `filename`.
        """

        with open(self._path(filename), "w", encoding="utf-8") as f:
            json.dump(
                {"schema_version": self.schema_version, "written_at": _utc_now_iso(), "metrics": dict(metrics)},
                f,
                indent=2,
                default=_json_default,
            )

    def save_best_checkpoint(
        self,
        *,
        model,
        optimizer=None,
        scheduler=None,
        epoch: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        filename: str = "best.pt",
    ) -> str:
        import torch

        path = os.path.join(self.checkpoints_dir, filename)
        ckpt = {
            "schema_version": self.schema_version,
            "saved_at": _utc_now_iso(),
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "metrics": metrics or {},
        }
        if optimizer is not None:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            ckpt["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(ckpt, path)
        return path

    def write_predictions(
        self,
        rows: Optional[Iterable[Dict[str, Any]]] = None,
        *,
        filename: str = "predictions.jsonl",
        # Back-compat: tensor-style API used by Trainer
        logits=None,
        targets=None,
        meta=None,
        split_name: Optional[str] = None,
    ) -> str:
        """Write predictions.

        Supports two calling styles:
          1) rows: an iterable of dicts
          2) (logits, targets, meta): tensors/arrays + metadata list
        """

        # Style (2): convert tensors -> row dicts
        if rows is None and logits is not None and targets is not None:
            import torch

            lg = logits.detach().cpu() if isinstance(logits, torch.Tensor) else logits
            tg = targets.detach().cpu() if isinstance(targets, torch.Tensor) else targets
            lg_list = lg.tolist() if hasattr(lg, "tolist") else list(lg)
            tg_list = tg.tolist() if hasattr(tg, "tolist") else list(tg)
            m_list = list(meta) if meta is not None else [None] * len(tg_list)
            s = split_name or ""
            rows = (
                {"split": s, "logit": float(lg_list[i]), "target": int(tg_list[i]), "meta": m_list[i]}
                for i in range(min(len(lg_list), len(tg_list)))
            )

        if rows is None:
            rows = []

        path = self._path(filename)
        with open(path, "w", encoding="utf-8") as f:
            for i, row in enumerate(rows):
                out = dict(row)
                out.setdefault("schema_version", self.schema_version)
                out.setdefault("index", i)
                f.write(json.dumps(out, default=_json_default) + "\n")
        return path
