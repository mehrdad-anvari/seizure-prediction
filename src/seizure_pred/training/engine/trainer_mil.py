from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from seizure_pred.core.config import TrainConfig
from seizure_pred.training.engine.callbacks import CallbackList
from seizure_pred.training.engine.metrics import binary_classification_metrics
from seizure_pred.training.engine.artifacts import ArtifactWriter


class TrainerMIL:
    """MIL trainer.

    Expected batch: (x, y, meta)
      - x: (B, bag, C, T) (or already aggregated by the model)
      - y: (B,)
      - meta: list
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
        self.cfg = cfg
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.run_dir = run_dir
        self.device = torch.device(device or cfg.device if torch.cuda.is_available() else "cpu")
        self.writer = artifact_writer or ArtifactWriter(run_dir)
        self.callbacks = CallbackList(callbacks or [])

        os.makedirs(self.run_dir, exist_ok=True)
        self.model.to(self.device)

        try:
            self.writer.write_schema()
            self.writer.write_config(asdict(cfg))
        except Exception:
            pass

    def fit(self, *, train_loader: DataLoader, val_loader: DataLoader, write_best_predictions: bool = True) -> str:
        state: Dict[str, Any] = {"trainer": self, "epoch": 0, "best_val_loss": float("inf")}
        self.callbacks.on_train_start(state)

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

            if self.scheduler is not None and getattr(self.cfg.sched, "step", "epoch") == "epoch":
                try:
                    self.scheduler.step()
                except TypeError:
                    try:
                        self.scheduler.step(val_loss)
                    except Exception:
                        pass

            if val_loss < float(state["best_val_loss"]):
                state["best_val_loss"] = val_loss
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

                last_ckpt_path = best_ckpt_path or last_ckpt_path

            self.callbacks.on_epoch_end(state)
            if state.get("stop", False):
                break

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
        self.model.train()
        losses = []

        for step, batch in enumerate(train_loader):
            x, y, meta = batch
            x = x.to(self.device)
            y = y.to(self.device).float().view(-1)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x).view(-1)
            loss = self.loss_fn(logits, y)
            loss.backward()

            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.grad_clip_norm))

            self.optimizer.step()

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

            logits = self.model(x).view(-1)
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


# Backwards-compatible alias (older scripts imported MILTrainer)
MILTrainer = TrainerMIL
