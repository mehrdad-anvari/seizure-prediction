"""Lightweight experiment grid runner.

Expands a small grid of config overrides (dot-paths) over a single data split
and trains one model per combination. This mirrors the behaviour of the
``seizure-pred train`` CLI but is convenient for quick hyper-parameter sweeps.

Example::

    seizure-pred experiments --config examples/config_prediction.yaml \
        --grid '{"optim.lr": [1e-3, 3e-4], "loss.name": ["bce_logits", "focal"]}'
"""

from __future__ import annotations

import itertools
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from seizure_pred.core.config import TrainConfig
from seizure_pred.core.io import load_dict, from_dict
from seizure_pred.core.validate import validate_train_config_dict
from seizure_pred.core.seed import seed_everything
from seizure_pred.training.engine.pipeline import build_dataset, iter_splits, build_loader
from seizure_pred.training.engine.artifacts import ArtifactWriter
from seizure_pred.training.engine.trainer import Trainer
from seizure_pred.training.engine.trainer_mil import TrainerMIL
from seizure_pred.training.registries import (
    CALLBACKS,
    DATALOADERS,
    LOSSES,
    MODELS,
    OPTIMIZERS,
    SCHEDULERS,
)


def _expand_grid(grid: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    combos: List[Dict[str, Any]] = []
    for prod in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, prod)})
    return combos


def _apply_overrides(cfg_dict: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    # deep copy via json round-trip
    out = json.loads(json.dumps(cfg_dict))
    for path, value in overrides.items():
        parts = path.split(".")
        cur = out
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value
    return out


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _build_components(cfg: TrainConfig):
    """Build model/loss/optimizer/scheduler/callbacks from registries."""
    model = MODELS.create(cfg.model.name, cfg.model)
    loss_fn = LOSSES.create(cfg.loss.name, **(cfg.loss.kwargs or {}))
    optimizer = OPTIMIZERS.create(
        cfg.optim.name,
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        **(cfg.optim.kwargs or {}),
    )
    scheduler = None
    if cfg.sched.name:
        scheduler = SCHEDULERS.create(cfg.sched.name, optimizer, **(cfg.sched.kwargs or {}))

    cb_list = []
    for item in getattr(cfg, "callbacks", []) or []:
        name = item["name"] if isinstance(item, dict) else getattr(item, "name", None)
        kwargs = item.get("kwargs", {}) if isinstance(item, dict) else dict(getattr(item, "kwargs", {}) or {})
        if not name or name not in CALLBACKS:
            continue
        cb_list.append(CALLBACKS.create(name, **kwargs))
    return model, loss_fn, optimizer, scheduler, cb_list


def run_grid(
    base_config_path: str,
    grid: Mapping[str, Sequence[Any]],
    *,
    split_index: int = 0,
    n_folds: Optional[int] = None,
    mil: bool = False,
    save_root: Optional[str] = None,
) -> List[str]:
    """Run a simple grid of experiments.

    ``grid`` keys are dot-paths into the config, e.g.::

        - "optim.lr": [1e-3, 3e-4]
        - "loss.name": ["bce_logits", "focal"]
        - "model.name": ["simple_cnn"]

    Returns: list of run_dir paths (one per grid combination).
    """
    # Ensure plugins are registered before any registry lookups.
    import seizure_pred.training as training
    import seizure_pred.models as models

    training.register_all()
    models.register_all()

    base = load_dict(base_config_path)
    validate_train_config_dict(base)

    run_dirs: List[str] = []
    combos = _expand_grid(grid)
    stamp = _utc_stamp()

    for i, overrides in enumerate(combos):
        cfg_dict = _apply_overrides(base, overrides)
        validate_train_config_dict(cfg_dict)
        cfg: TrainConfig = from_dict(TrainConfig, cfg_dict)

        if save_root is not None:
            cfg.save_dir = save_root
        if n_folds is not None:
            cfg.data.n_folds = int(n_folds)

        # embed override summary into run name
        override_tag = "_".join(f"{k.replace('.', '-')}-{str(v)}" for k, v in overrides.items())
        cfg.run_name = f"{cfg.run_name}__grid{i:03d}__{override_tag}"[:160]

        seed_everything(None, seed=cfg.seed)

        ds = build_dataset(cfg)
        splits = list(iter_splits(ds, cfg.data))
        if not splits:
            raise RuntimeError("No splits produced for grid experiment")
        if split_index < 0 or split_index >= len(splits):
            raise ValueError(f"split_index {split_index} out of range (0..{len(splits) - 1})")
        train_set, val_set = splits[split_index]

        dl_name = cfg.data.dataloader_type or "torch"
        if dl_name not in DATALOADERS:
            raise ValueError(f"Unknown dataloader '{dl_name}'. Use `seizure-pred list`.")

        train_loader = build_loader(dl_name, train_set, cfg, shuffle=True)
        val_loader = build_loader(dl_name, val_set, cfg, shuffle=False)

        model, loss_fn, optimizer, scheduler, cb_list = _build_components(cfg)

        run_dir = os.path.join(cfg.save_dir, cfg.run_name, stamp, f"split_{split_index}")
        os.makedirs(run_dir, exist_ok=True)
        writer = ArtifactWriter(run_dir)
        writer.write_schema()
        writer.write_config(_as_serializable_cfg(cfg))

        trainer_cls = TrainerMIL if mil else Trainer
        trainer = trainer_cls(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            run_dir=run_dir,
            artifact_writer=writer,
            callbacks=cb_list,
        )

        best_ckpt = trainer.fit(train_loader=train_loader, val_loader=val_loader)
        run_dirs.append(run_dir)
        print(f"[grid {i + 1}/{len(combos)}] overrides={overrides} -> {best_ckpt}")

    return run_dirs


def _as_serializable_cfg(cfg: TrainConfig) -> Dict[str, Any]:
    from dataclasses import asdict

    return asdict(cfg)
