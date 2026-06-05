"""Legacy-style nested CV training script.

This reproduces the high-level workflow of the old `train.py`:
  - outer CV: leave-one-preictal-event-out (leave_one_preictal)
  - inner CV: stratified K-fold
  - optional "balanced" training via undersampling

Run:
  python examples/scripts/legacy_nested_cv.py --help
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Any, Tuple

import torch

from seizure_pred.core.config import TrainConfig, DataConfig, ModelConfig, OptimConfig, LossConfig, SchedConfig
from seizure_pred.core.logging import setup_logging
from seizure_pred.core.seed import seed_everything
from seizure_pred.data.chbmit_npz import CHBMITDataset
from seizure_pred.data.splits import make_cv_splitter
from seizure_pred.training.engine.artifacts import ArtifactWriter
from seizure_pred.training.engine.trainer import Trainer
from seizure_pred.training.engine.pipeline import build_loader
from seizure_pred.training.registries import MODELS, LOSSES, OPTIMIZERS, SCHEDULERS


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _two_digit_subject(subject_id: str | int) -> str:
    s = str(subject_id).strip()
    if s.isdigit():
        return f"{int(s):02d}"
    return s


def _build_cfg(args: argparse.Namespace) -> TrainConfig:
    data = DataConfig(
        name="chbmit_npz",
        dataset_dir=args.dataset_dir,
        subject_id=_two_digit_subject(args.subject_id),
        use_uint16=bool(args.use_uint16),
        suffix=args.suffix,
        task=args.task,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=not bool(args.no_pin_memory),
        persistent_workers=not bool(args.no_persistent_workers),
        kwargs={},
    )

    # Legacy flag: apply_normalization
    if args.apply_normalization:
        # Instance-wise z-score normalization
        data.kwargs["online_transforms"] = ["instance_norm"]

    model = ModelConfig(
        name=args.model,
        num_classes=2,
        kwargs={},
    )
    # pass through optional model kwargs from CLI
    if args.model_kwargs:
        for kv in args.model_kwargs:
            k, v = kv.split("=", 1)
            model.kwargs[k] = _coerce_scalar(v)

    optim = OptimConfig(name=args.optimizer, lr=float(args.lr), weight_decay=float(args.weight_decay), kwargs={})
    loss = LossConfig(name=args.loss, kwargs={})
    sched = SchedConfig(name=args.scheduler, step="epoch", kwargs={})

    return TrainConfig(
        task=args.task,
        seed=int(args.seed),
        device=args.device,
        epochs=int(args.epochs),
        amp=not bool(args.no_amp),
        log_every=int(args.log_every),
        val_every=1,
        save_dir=args.save_dir,
        run_name=args.run_name,
        data=data,
        model=model,
        loss=loss,
        optim=optim,
        sched=sched,
        callbacks=[],
    )


def _coerce_scalar(v: str) -> Any:
    vv = v.strip()
    for t in (int, float):
        try:
            return t(vv)
        except Exception:
            pass
    if vv.lower() in {"true", "false"}:
        return vv.lower() == "true"
    return vv


def _build_components(cfg: TrainConfig):
    # Lazy plugin registration
    import seizure_pred.training as training
    import seizure_pred.models as models

    training.register_all()
    models.register_all()

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
    return model, loss_fn, optimizer, scheduler


def main() -> None:
    p = argparse.ArgumentParser(description="Legacy-style nested CV training (outer+inner)")

    # Dataset
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--subject-id", required=True)
    p.add_argument("--suffix", default="fd_5s_szx5_prex5")
    p.add_argument("--task", default="prediction", choices=["prediction", "detection"])
    p.add_argument("--use-uint16", action="store_true")
    p.add_argument("--apply-normalization", action="store_true")

    # Training basics
    p.add_argument("--model", default="eegwavenet")
    p.add_argument("--model-kwargs", nargs="*", default=None, help="Extra model kwargs like key=value")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", default="adam")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--loss", default="bce_logits")
    p.add_argument("--scheduler", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--log-every", type=int, default=25)

    # Dataloader options
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--no-persistent-workers", action="store_true")

    # Nested CV
    p.add_argument("--outer-cv-mode", default="leave_one_preictal", choices=["leave_one_preictal", "leave_one_out"]) 
    p.add_argument("--outer-cv-method", default="balanced")
    p.add_argument("--inner-cv-mode", default="stratified", choices=["stratified"])
    p.add_argument("--inner-cv-method", default="balanced")
    p.add_argument("--n-fold", type=int, default=5)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--random-state", type=int, default=0)

    # Output
    p.add_argument("--save-dir", default="runs")
    p.add_argument("--run-name", default="legacy_nested_cv")

    args = p.parse_args()

    cfg = _build_cfg(args)
    seed_everything(getattr(cfg, "determinism", None), seed=cfg.seed)

    sub = _two_digit_subject(args.subject_id)
    run_root = os.path.join(cfg.save_dir, cfg.run_name, _stamp(), f"sub-{sub}")
    os.makedirs(run_root, exist_ok=True)
    logger = setup_logging(run_root)

    logger.info("[legacy_nested_cv] run_root=%s", run_root)
    logger.info("[legacy_nested_cv] cfg=%s", asdict(cfg))

    # Build dataset directly (keeps script self-contained)
    online_transforms = cfg.data.kwargs.get("online_transforms")
    if isinstance(online_transforms, list) and online_transforms and isinstance(online_transforms[0], str):
        from seizure_pred.transforms.registry import create_transform

        online_transforms = [create_transform(name) for name in online_transforms]

    ds = CHBMITDataset(
        dataset_dir=cfg.data.dataset_dir,
        subject_id=cfg.data.subject_id,
        use_uint16=cfg.data.use_uint16,
        suffix=cfg.data.suffix,
        task=cfg.data.task,
        online_transforms=online_transforms,
        offline_transforms=None,
        print_events=True,
    )

    outer_splits = list(
        make_cv_splitter(
            ds,
            mode=args.outer_cv_mode,
            method=args.outer_cv_method,
            shuffle=args.shuffle,
            random_state=args.random_state,
            n_fold=args.n_fold,
        )
    )
    logger.info("[legacy_nested_cv] outer_folds=%d", len(outer_splits))

    for outer_i, (train_val_set, test_set) in enumerate(outer_splits, start=1):
        outer_dir = os.path.join(run_root, f"outer_{outer_i}")
        os.makedirs(outer_dir, exist_ok=True)
        logger_outer = setup_logging(outer_dir)
        logger_outer.info("===== OUTER FOLD %d/%d =====", outer_i, len(outer_splits))

        inner_splits = list(
            make_cv_splitter(
                train_val_set,
                mode=args.inner_cv_mode,
                method=args.inner_cv_method,
                n_fold=args.n_fold,
                shuffle=args.shuffle,
                random_state=args.random_state,
            )
        )
        logger_outer.info("inner_folds=%d", len(inner_splits))

        for inner_i, (train_set, val_set) in enumerate(inner_splits, start=1):
            split_dir = os.path.join(outer_dir, f"inner_{inner_i}")
            os.makedirs(split_dir, exist_ok=True)
            logger_split = setup_logging(split_dir)
            logger_split.info("--- INNER FOLD %d/%d ---", inner_i, len(inner_splits))

            # Build a fresh model per inner fold
            model, loss_fn, optimizer, scheduler = _build_components(cfg)

            writer = ArtifactWriter(split_dir)
            writer.write_schema()
            writer.write_config(asdict(cfg))

            trainer = Trainer(
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                run_dir=split_dir,
                artifact_writer=writer,
                callbacks=[],
                device=cfg.device,
            )

            # Legacy "balanced" method: undersample negatives during training
            train_dl_name = "undersample" if args.inner_cv_method == "balanced" else "torch"
            train_loader = build_loader(train_dl_name, train_set, cfg, shuffle=True)
            val_loader = build_loader("torch", val_set, cfg, shuffle=False)
            test_loader = build_loader("torch", test_set, cfg, shuffle=False)

            logger_split.info(
                "train=%d val=%d test=%d | train_loader=%s",
                len(train_set),
                len(val_set),
                len(test_set),
                train_dl_name,
            )

            best_ckpt = trainer.fit(train_loader=train_loader, val_loader=val_loader)
            logger_split.info("best_ckpt=%s", best_ckpt)

            # Evaluate on the outer test set and persist predictions
            test_out = trainer.evaluate(test_loader)
            writer.write_metrics(
                {
                    "outer": outer_i,
                    "inner": inner_i,
                    **{k: v for k, v in test_out.items() if k not in {"val_logits", "val_targets", "val_meta"}},
                },
                filename="test_metrics.json",
            )
            writer.write_predictions(
                logits=test_out["val_logits"],
                targets=test_out["val_targets"],
                meta=test_out["val_meta"],
                split_name="test",
                filename="test_predictions.jsonl",
            )
            logger_split.info("test_loss=%.4f test_acc=%.4f", float(test_out["loss"]), float(test_out.get("acc", float("nan"))))


if __name__ == "__main__":
    main()
