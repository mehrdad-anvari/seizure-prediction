from __future__ import annotations

import argparse
from pathlib import Path

import torch

from seizure_pred.core.config import TrainConfig
from seizure_pred.core.io import from_dict, load_dict, merge_dict
from seizure_pred.training import registries  # ensure registrations
from seizure_pred.training.engine.pipeline import build_dataset, iter_splits, build_dataloader
from seizure_pred.training.engine.trainer import Trainer
from seizure_pred.training.engine.trainer_mil import MILTrainer
from seizure_pred.training.registries import MODELS, LOSSES, OPTIMIZERS, SCHEDULERS, DATALOADERS


def _set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a seizure prediction/detection model")
    p.add_argument("--config", type=str, required=True, help="Path to yaml/json config")
    p.add_argument("--override", type=str, default=None, help="Optional yaml/json override file")
    p.add_argument("--split-index", type=int, default=0, help="Which split to train (0..N-1)")
    p.add_argument("--dataloader", type=str, default="torch", help="dataloader strategy: torch|undersample|mil|...")
    p.add_argument("--mil", action="store_true", help="Use MIL training (bags)")
    return p


def run_train(args: argparse.Namespace) -> None:
    cfg_dict = load_dict(args.config)
    if args.override:
        cfg_dict = merge_dict(cfg_dict, load_dict(args.override))
    cfg = from_dict(TrainConfig, cfg_dict)

    _set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")

    dataset = build_dataset(cfg)
    splits = list(iter_splits(dataset))
    if not splits:
        raise RuntimeError("No splits produced. Check dataset and split config.")
    if args.split_index < 0 or args.split_index >= len(splits):
        raise ValueError(f"split-index out of range: {args.split_index} (0..{len(splits)-1})")

    train_set, val_set = splits[args.split_index]

    # dataloaders
    train_loader = build_dataloader(args.dataloader, train_set, cfg.data, shuffle=True)
    val_loader = build_dataloader(args.dataloader, val_set, cfg.data, shuffle=False)

    # model
    model = MODELS.create(cfg.model.name, cfg.model).to(device)

    # loss
    loss_fn = LOSSES.create(cfg.loss.name, **cfg.loss.kwargs).to(device)

    # optimizer
    optimizer = OPTIMIZERS.create(cfg.optim.name, model.parameters(), cfg.optim)

    # scheduler
    scheduler = None
    if cfg.sched.name:
        scheduler = SCHEDULERS.create(cfg.sched.name, optimizer, cfg.sched)

    if args.mil or args.dataloader == "mil":
        trainer = MILTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            amp=cfg.amp,
            grad_clip_norm=cfg.grad_clip_norm,
        )
        hist = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=cfg.epochs,
            log_every=cfg.log_every,
            save_dir=cfg.save_dir,
            run_name=cfg.run_name,
            scheduler=scheduler,
        )
    else:
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            amp=cfg.amp,
            grad_clip_norm=cfg.grad_clip_norm,
        )
        hist = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=cfg.epochs,
            log_every=cfg.log_every,
            save_dir=cfg.save_dir,
            run_name=cfg.run_name,
            scheduler=scheduler,
        )

    # save training history
    out_dir = Path(cfg.save_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history.json").write_text(__import__("json").dumps(hist, indent=2), encoding="utf-8")
    print(f"Done. Artifacts saved to: {out_dir}")


def main(argv=None) -> None:
    args = build_argparser().parse_args(argv)
    run_train(args)
