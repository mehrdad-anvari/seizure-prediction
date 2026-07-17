from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional
import math

import os
import logging
import torch
from torch import nn
from torch.utils.data import DataLoader

from seizure_pred.core.config import TrainConfig
from seizure_pred.training.engine.callbacks import CallbackList
from seizure_pred.training.engine.metrics import binary_classification_metrics
from seizure_pred.training.engine.artifacts import ArtifactWriter


def _monitor_value(cfg: TrainConfig, val_loss: float, val_metrics: Dict[str, Any]) -> Optional[float]:
    """Resolve the monitored metric value for best-checkpoint selection.

    Returns None when the configured metric is unavailable/NaN for this epoch.
    """
    monitor = (getattr(cfg, "monitor", "val_loss") or "val_loss").strip()
    if monitor == "val_loss":
        return float(val_loss)
    v = val_metrics.get(monitor)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    if math.isnan(f):
        return None
    return f


def _monitor_mode(cfg: TrainConfig) -> str:
    monitor = (getattr(cfg, "monitor", "val_loss") or "val_loss").strip()
    if monitor == "val_loss":
        return "min"
    mode = (getattr(cfg, "monitor_mode", "max") or "max").strip().lower()
    return "max" if mode not in {"min", "max"} else mode


def _is_better(new: float, best: float, mode: str) -> bool:
    return new < best if mode == "min" else new > best


# Metrics that can be used for best-checkpoint selection. Anything else is
# rejected at trainer init so a typo doesn't silently disable checkpointing.
_MONITOR_METRICS = {"val_loss", "auc", "ap", "f1", "acc", "precision", "recall", "loss"}


def _validate_monitor(cfg: TrainConfig) -> None:
    monitor = (getattr(cfg, "monitor", "val_loss") or "val_loss").strip()
    if monitor not in _MONITOR_METRICS:
        raise ValueError(
            f"Unknown monitor metric '{monitor}'. "
            f"Expected one of {sorted(_MONITOR_METRICS)}."
        )
    mode = (getattr(cfg, "monitor_mode", "min") or "min").strip().lower()
    if mode not in {"min", "max"}:
        raise ValueError(f"monitor_mode must be 'min' or 'max', got '{mode}'.")


class Trainer:
    """Standard (non-MIL) trainer.

    Expected batch: (x, y, meta)
      - x: (B, C, T)
      - y: (B,) with 0/1
      - meta: list[dict] or any metadata
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object],
        cfg: TrainConfig,
        run_dir: str,
        artifact_writer: Optional[ArtifactWriter] = None,
        callbacks: Optional[list[Any]] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the standard (non-MIL) Trainer.

        Args:
            model: PyTorch neural network module.
            loss_fn: Criterion/loss module.
            optimizer: PyTorch optimizer instance.
            scheduler: Optional learning rate scheduler.
            cfg: Resolved training configuration.
            run_dir: Path to save run outputs, logs, and checkpoints.
            artifact_writer: Optional custom ArtifactWriter (instantiated inside if None).
            callbacks: Optional list of callback instances.
            device: Optional target device string (defaults to cuda/cpu configured in cfg).
        """
        self.cfg = cfg
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.run_dir = run_dir
        self.device = torch.device(device or cfg.device if torch.cuda.is_available() else "cpu")
        self.writer = artifact_writer or ArtifactWriter(run_dir)
        self.callbacks = CallbackList(callbacks or [])

        _validate_monitor(cfg)

        os.makedirs(self.run_dir, exist_ok=True)
        self.model.to(self.device)

        # Write schema/config once
        try:
            self.writer.write_schema()
            self.writer.write_config(asdict(cfg))
        except Exception:
            # Writer is best-effort; do not crash training for I/O
            pass

    def fit(self, *, train_loader: DataLoader, val_loader: DataLoader, write_best_predictions: bool = True) -> str:
        """Execute the full training and validation loop.

        Args:
            train_loader: Dataloader yielding training batches.
            val_loader: Dataloader yielding validation batches.
            write_best_predictions: Whether to save predicted probabilities for the best epoch.

        Returns:
            The file path to the best saved model checkpoint.
        """
        state: Dict[str, Any] = {"trainer": self, "epoch": 0, "best_val_loss": float("inf")}
        self.callbacks.on_train_start(state)

        mode = _monitor_mode(self.cfg)
        best_value = float("inf") if mode == "min" else float("-inf")
        state["monitor"] = getattr(self.cfg, "monitor", "val_loss") or "val_loss"
        state["monitor_mode"] = mode
        state["best_monitor_value"] = best_value

        logger = logging.getLogger("seizure_pred")
        best_ckpt_path = ""
        last_ckpt_path = ""
        for epoch in range(1, int(self.cfg.epochs) + 1):
            state["epoch"] = epoch
            self.callbacks.on_epoch_start(state)

            train_loss = self._train_one_epoch(train_loader, state)
            val_out = self.evaluate(val_loader, state)

            val_loss = float(val_out["loss"])
            state["train_loss"] = train_loss
            state["val_loss"] = val_loss
            state["val_metrics"] = {k: v for k, v in val_out.items() if k not in {"val_logits", "val_targets", "val_meta"}}

            # Log once per epoch (consolidated)
            try:
                parts = [
                    f"Epoch {epoch:03d}/{int(self.cfg.epochs):03d}",
                    f"train_loss={train_loss:.5f}",
                    f"val_loss={val_loss:.5f}",
                ]
                # Include common validation metrics if present
                for k in ("auc", "acc", "f1", "precision", "recall"):
                    v = state["val_metrics"].get(k)
                    if isinstance(v, (int, float)):
                        parts.append(f"{k}={float(v):.4f}")
                logger.info(" | ".join(parts))
            except Exception:
                # Logging must never crash training
                pass

            # history row
            try:
                self.writer.append_history(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        **{f"val_{k}": v for k, v in state["val_metrics"].items() if isinstance(v, (int, float))},
                    }
                )
            except Exception:
                pass

            # scheduler step (epoch mode)
            if self.scheduler is not None and getattr(self.cfg.sched, "step", "epoch") == "epoch":
                try:
                    self.scheduler.step()
                except TypeError:
                    # schedulers like ReduceLROnPlateau may need val_loss
                    try:
                        self.scheduler.step(val_loss)
                    except Exception:
                        pass

            # best checkpoint (by configured monitor metric)
            mon_val = _monitor_value(self.cfg, val_loss, state["val_metrics"])
            improved = mon_val is not None and _is_better(mon_val, best_value, mode)
            if improved:
                best_value = mon_val  # type: ignore[assignment]
                state["best_val_loss"] = val_loss
                state["best_monitor_value"] = best_value
                try:
                    best_ckpt_path = self.writer.save_best_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        metrics=state["val_metrics"],
                    )
                except Exception:
                    best_ckpt_path = ""

                # write metrics + optional predictions
                try:
                    self.writer.write_metrics({"val_loss": val_loss, **state["val_metrics"]})
                except Exception:
                    pass

                if write_best_predictions:
                    try:
                        self.writer.write_predictions(
                            logits=val_out["val_logits"],
                            targets=val_out["val_targets"],
                            meta=val_out["val_meta"],
                            split_name="val",
                        )
                    except Exception:
                        pass

                # Track last checkpoint as well
                last_ckpt_path = best_ckpt_path or last_ckpt_path

            self.callbacks.on_epoch_end(state)

            # Early stopping support: callbacks may set state["stop"]=True
            if state.get("stop", False):
                break

        # If we never managed to write a "best" checkpoint (e.g., writer failure),
        # try to write one last checkpoint so callers always get a valid path.
        if not best_ckpt_path:
            try:
                last_ckpt_path = self.writer.save_best_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=int(state.get("epoch", 0)),
                    metrics=state.get("val_metrics", {}),
                    filename="last.pt",
                )
            except Exception:
                pass

        self.callbacks.on_train_end(state)
        return best_ckpt_path or last_ckpt_path

    def _train_one_epoch(self, train_loader: DataLoader, state: Dict[str, Any]) -> float:
        """Run a single epoch of training optimization steps over train_loader.

        Args:
            train_loader: Training DataLoader.
            state: The shared training state dictionary.

        Returns:
            Average training loss over the epoch.
        """
        self.model.train()
        losses = []

        for step, batch in enumerate(train_loader):
            x, y, meta = batch
            x = x.to(self.device)
            y = y.to(self.device).float().view(-1)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self._to_binary_logits(self.model(x))
            
            if self.loss_fn.__class__.__name__ == "PreictalWeightedLoss":
                temporal_weights = self._extract_temporal_weights(meta).to(self.device)
                loss = self.loss_fn(logits, y, temporal_weights)
            else:
                loss = self.loss_fn(logits, y)
                
            loss.backward()

            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.grad_clip_norm))

            self.optimizer.step()

            # scheduler step (step mode)
            if self.scheduler is not None and getattr(self.cfg.sched, "step", "epoch") == "step":
                try:
                    self.scheduler.step()
                except Exception:
                    pass

            losses.append(float(loss.item()))
            state["step"] = step
            state["batch_loss"] = float(loss.item())
            self.callbacks.on_batch_end(state)

        return sum(losses) / max(1, len(losses))

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform evaluation/validation on the provided dataloader.

        Args:
            val_loader: Evaluation DataLoader.
            state: Optional shared training state dictionary for callback hooks.

        Returns:
            Dictionary containing computed metrics (e.g., loss, acc, auc, f1), 
            along with raw logits, targets, and metadata.
        """
        self.model.eval()
        losses = []
        logits_all = []
        targets_all = []
        meta_all = []

        st = state if state is not None else {}
        self.callbacks.on_val_start(st)

        for step, batch in enumerate(val_loader):
            x, y, meta = batch
            x = x.to(self.device)
            y = y.to(self.device).float().view(-1)

            logits = self._to_binary_logits(self.model(x))
            loss = self.loss_fn(logits, y)

            losses.append(float(loss.item()))
            logits_all.append(logits.detach().cpu())
            targets_all.append(y.detach().cpu())
            meta_all.extend(meta if isinstance(meta, list) else [meta])

            st["val_step"] = step
            self.callbacks.on_val_batch_end(st)

        logits_t = torch.cat(logits_all) if logits_all else torch.empty(0)
        targets_t = torch.cat(targets_all) if targets_all else torch.empty(0)

        m = binary_classification_metrics(logits_t, targets_t, threshold=0.5)
        out = {
            "loss": sum(losses) / max(1, len(losses)),
            **m,
            "val_logits": logits_t,
            "val_targets": targets_t,
            "val_meta": meta_all,
        }

        st["val_out"] = out
        self.callbacks.on_val_end(st)
        return out

    @staticmethod
    def _to_binary_logits(raw: torch.Tensor) -> torch.Tensor:
        """Normalize model output to shape (B,) binary logits.

        Supported outputs:
          - (B,)             -> returned as-is
          - (B,1)            -> squeeze dim=1
          - (B,2)            -> convert to single binary logit using (logit1 - logit0)
          - (B,*,...)        -> tries to squeeze/flatten conservatively
        """

        if not isinstance(raw, torch.Tensor):
            raw = torch.as_tensor(raw)

        if raw.ndim == 1:
            return raw

        if raw.ndim == 2:
            if raw.shape[1] == 1:
                return raw[:, 0]
            if raw.shape[1] == 2:
                return raw[:, 1] - raw[:, 0]
            # Unexpected: treat as already a vector if possible
            if raw.shape[0] == 1:
                return raw.reshape(-1)
            raise ValueError(f"Expected binary logits with shape (B,), (B,1) or (B,2) but got {tuple(raw.shape)}")

        # Higher-dim: try squeezing singleton dims after batch.
        x = raw
        while x.ndim > 1 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.ndim == 1:
            return x
        # As a last resort, flatten everything except batch then reduce if 2-class.
        x2 = x.reshape(x.shape[0], -1)
        if x2.shape[1] == 1:
            return x2[:, 0]
        if x2.shape[1] == 2:
            return x2[:, 1] - x2[:, 0]
        raise ValueError(f"Could not coerce logits to binary: got {tuple(raw.shape)}")

    def _extract_temporal_weights(self, meta: Any) -> torch.Tensor:
        """Extract temporal weights from metadata.
        Supports both List[Dict] and Dict[str, List/Tensor].
        """
        weights = []
        if isinstance(meta, dict):
            labels = meta.get("label", [])
            epoch_indices = meta.get("epoch_index_within_event", [])
            n_segs = meta.get("n_segments_in_event", [])
            
            n_samples = len(labels)
            for i in range(n_samples):
                label_val = labels[i]
                if label_val == "preictal":
                    idx = int(epoch_indices[i]) if i < len(epoch_indices) else 0
                    total = int(n_segs[i]) if i < len(n_segs) else 1
                    weight = idx / max(total - 1, 1)
                else:
                    weight = 0.0
                weights.append(weight)
        elif isinstance(meta, list):
            for m in meta:
                if isinstance(m, dict) and m.get("label") == "preictal":
                    idx = m.get("epoch_index_within_event", 0)
                    total = m.get("n_segments_in_event", 1)
                    weight = idx / max(total - 1, 1)
                else:
                    weight = 0.0
                weights.append(weight)
        else:
            return torch.zeros(1)
            
        return torch.tensor(weights, dtype=torch.float)
