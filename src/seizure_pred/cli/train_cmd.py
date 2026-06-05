from __future__ import annotations

import argparse
import json
import os
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from seizure_pred.core.config import TrainConfig
from seizure_pred.core.io import load_dict, merge_dict, from_dict
from seizure_pred.core.seed import seed_everything
from seizure_pred.core.validate import validate_config_dict
from seizure_pred.training.registries import CALLBACKS, DATALOADERS, MODELS, LOSSES, OPTIMIZERS, SCHEDULERS
from seizure_pred.training.engine.pipeline import build_dataset, iter_splits, build_loader
from seizure_pred.training.engine.artifacts import ArtifactWriter
from seizure_pred.training.engine.trainer import Trainer
from seizure_pred.training.engine.trainer_mil import TrainerMIL
from seizure_pred.core.logging import setup_logging


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _make_run_dir(cfg: TrainConfig, split_index: int) -> str:
    # <save_dir>/<run_name>/<stamp>/split_<k>/
    root = os.path.join(cfg.save_dir, cfg.run_name, _utc_stamp(), f"split_{split_index}")
    os.makedirs(root, exist_ok=True)
    return root


def add_train_cmd(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("train", help="Train a model (prediction/detection/MIL)")
    p.add_argument("--config", required=True, help="YAML/JSON config file")
    p.add_argument("--override", default=None, help="Optional YAML/JSON override file merged on top")
    p.add_argument("--split-index", type=int, default=0, help="Which split fold to run")
    p.add_argument("--n-folds", type=int, default=5, help="Number of folds for leave-one-out style split")
    p.add_argument("--dataloader", default=None, help="Override dataloader strategy name")
    p.add_argument("--mil", action="store_true", help="Use MIL trainer")
    p.add_argument("--strict", action="store_true", help="Fail fast if requested components are missing")
    p.add_argument("--print-config", action="store_true", help="Print merged config and exit")

    p.set_defaults(func=run_train)


def run_train(args: argparse.Namespace) -> None:
    raw = load_dict(args.config)
    if args.override:
        raw = merge_dict(raw, load_dict(args.override))

    # Validate user config dict against TrainConfig schema (clear errors)
    validate_config_dict(raw, TrainConfig)

    cfg: TrainConfig = from_dict(TrainConfig, raw)

    # Register built-in plugins only when training is invoked (avoids heavy imports on `--help`).
    import seizure_pred.training as training
    training.register_all()
    import seizure_pred.models as models
    models.register_all()
    
    # Optional overrides
    if args.dataloader is not None:
        cfg.data.kwargs = dict(cfg.data.kwargs or {})
        cfg.data.kwargs["dataloader_name"] = args.dataloader  # keep provenance
    dl_name = args.dataloader or getattr(cfg.data, "dataloader", None) or "torch"

    if args.print_config:
        print(json.dumps(asdict(cfg), indent=2, default=str))
        return

    # Determinism / seeding
    seed_everything(getattr(cfg, "determinism", None), seed=cfg.seed)

    # Run directory + logger
    run_dir = _make_run_dir(cfg, args.split_index)
    logger = setup_logging(run_dir)
    logger.info("[train] run_dir=%s", run_dir)
    logger.info("[train] config_path=%s", args.config)
    if args.override:
        logger.info("[train] override_path=%s", args.override)

    # Build dataset + split
    dataset = build_dataset(cfg)
    splits = list(iter_splits(dataset, n_folds=args.n_folds))
    if args.split_index < 0 or args.split_index >= len(splits):
        raise SystemExit(f"--split-index {args.split_index} out of range (0..{len(splits)-1})")

    train_set, val_set = splits[args.split_index]

    # Build loaders via registry (factory pattern)
    # (If strict and not present -> raise with helpful registry error)
    if args.strict and dl_name not in DATALOADERS:
        raise SystemExit(f"Unknown dataloader '{dl_name}'. Use `seizure-pred list`.")

    train_loader = build_loader(dl_name, train_set, cfg, shuffle=True)
    val_loader = build_loader(dl_name, val_set, cfg, shuffle=False)

    # Build model/loss/optim/sched from registries
    if args.strict and cfg.model.name not in MODELS:
        raise SystemExit(f"Unknown model '{cfg.model.name}'. Use `seizure-pred list`.")
    if args.strict and cfg.loss.name not in LOSSES:
        raise SystemExit(f"Unknown loss '{cfg.loss.name}'. Use `seizure-pred list`.")
    if args.strict and cfg.optim.name not in OPTIMIZERS:
        raise SystemExit(f"Unknown optimizer '{cfg.optim.name}'. Use `seizure-pred list`.")
    if args.strict and cfg.sched.name and cfg.sched.name not in SCHEDULERS:
        raise SystemExit(f"Unknown scheduler '{cfg.sched.name}'. Use `seizure-pred list`.")

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

    # Callbacks: cfg.callbacks = [{name, kwargs}, ...]
    cb_list = []
    for item in getattr(cfg, "callbacks", []) or []:
        name = item["name"] if isinstance(item, dict) else getattr(item, "name", None)
        kwargs = item.get("kwargs", {}) if isinstance(item, dict) else dict(getattr(item, "kwargs", {}) or {})
        if not name:
            continue
        if args.strict and name not in CALLBACKS:
            raise SystemExit(f"Unknown callback '{name}'. Use `seizure-pred list`.")
        cb_list.append(CALLBACKS.create(name, **kwargs))

    # Artifact writer
    writer = ArtifactWriter(run_dir)
    writer.write_schema()
    writer.write_config(asdict(cfg))

    # Train
    if args.mil:
        trainer = TrainerMIL(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            run_dir=run_dir,
            artifact_writer=writer,
            callbacks=cb_list,
        )
    else:
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            run_dir=run_dir,
            artifact_writer=writer,
            callbacks=cb_list,
        )

    logger.info("[train] dl_name=%s batch_size=%s num_workers=%s", dl_name, cfg.data.batch_size, cfg.data.num_workers)
    logger.info("[train] model=%s loss=%s optim=%s sched=%s", cfg.model.name, cfg.loss.name, cfg.optim.name, cfg.sched.name)

    best_ckpt = trainer.fit(train_loader=train_loader, val_loader=val_loader)

    logger.info("[train] best_checkpoint=%s", best_ckpt)
    logger.info("[train] done")


def train_from_config(
    config_path: os.PathLike[str] | str,
    *,
    split_index: int = 0,
    dataloader: str | None = None,
    mil: bool = False,
    override_path: os.PathLike[str] | str | None = None,
    strict: bool = False,
) -> None:
    """Programmatic entrypoint used by legacy scripts.

    This mirrors the CLI behavior of ``seizure-pred train`` but is convenient for
    back-compat wrappers that historically imported a function.
    """

    ns = argparse.Namespace(
        config=str(config_path),
        override=None if override_path is None else str(override_path),
        split_index=int(split_index),
        n_folds=5,
        dataloader=dataloader,
        mil=bool(mil),
        strict=bool(strict),
        print_config=False,
    )
    run_train(ns)
