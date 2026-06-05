from __future__ import annotations

import itertools
import json
import os
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from seizure_pred.core.config import TrainConfig
from seizure_pred.core.io import load_dict, merge_dict, from_dict
from seizure_pred.core.validate import validate_train_config_dict
from seizure_pred.training.engine.pipeline import build_dataset, iter_splits, build_dataloader
from seizure_pred.training.engine.trainer import Trainer
from seizure_pred.training.engine.trainer_mil import TrainerMIL
from seizure_pred.training.registries import MODELS, LOSSES, OPTIMIZERS, SCHEDULERS


def _expand_grid(grid: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    combos = []
    for prod in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, prod)})
    return combos


def _apply_overrides(cfg_dict: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    # overrides are dict paths like {"optim.lr": 1e-4, "loss.name": "bce_logits"}
    out = json.loads(json.dumps(cfg_dict))  # deep copy
    for path, value in overrides.items():
        parts = path.split(".")
        cur = out
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value
    return out


def run_grid(
    base_config_path: str,
    grid: Mapping[str, Sequence[Any]],
    *,
    split_index: int = 0,
    dataloader: str = "torch",
    mil: bool = False,
    save_root: str | None = None,
) -> List[str]:
    """Run a simple grid of experiments.

    grid keys are dot-paths into config, e.g.:
      - "optim.lr": [1e-3, 3e-4]
      - "loss.name": ["bce_logits", "focal"]
      - "model.name": ["simple_cnn"]

    Returns: list of run_dir paths.
    """
    base = load_dict(base_config_path)
    validate_train_config_dict(base)

    run_dirs: List[str] = []
    combos = _expand_grid(grid)

    for i, overrides in enumerate(combos):
        cfg_dict = _apply_overrides(base, overrides)
        validate_train_config_dict(cfg_dict)
        cfg = from_dict(TrainConfig, cfg_dict)

        # optionally override save_dir root
        if save_root is not None:
            cfg.save_dir = save_root

        # build data
        ds = build_dataset(cfg)
        splits = list(iter_splits(ds))
        if split_index < 0 or split_index >= len(splits):
            raise ValueError(f"split_index {split_index} out of range (0..{len(splits)-1})")
        train_set, val_set = splits[split_index]

        train_loader = build_dataloader(dataloader, train_set, cfg.data, shuffle=True)
        val_loader = build_dataloader(dataloader, val_set, cfg.data, shuffle=False)

        # build model/loss/optim/sched via registries
        model = MODELS.create(cfg.model.name, cfg.model)
        loss_fn = LOSSES.create(cfg.loss.name, cfg.loss)
        optim = OPTIMIZERS.create(cfg.optim.name, model.parameters(), cfg.optim)
        sched = SCHEDULERS.maybe_create(cfg.sched.name, optim, cfg.sched)

        trainer = TrainerMIL(cfg, model, loss_fn, optim, sched) if mil else Trainer(cfg, model, loss_fn, optim, sched)

        # embed override summary into run name
        override_tag = "_".join(f"{k.replace('.', '-')}-{str(v)}" for k, v in overrides.items())
        cfg.run_name = f"{cfg.run_name}__grid{i:03d}__{override_tag}"[:160]

        run_dir = trainer.fit(train_loader, val_loader, split_index=split_index)
        run_dirs.append(run_dir)

    return run_dirs
