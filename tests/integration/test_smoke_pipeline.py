from __future__ import annotations

import os
import tempfile

import torch

import seizure_pred.training  # noqa: F401
import seizure_pred.models  # noqa: F401

from seizure_pred.core.config import TrainConfig
from seizure_pred.training.engine.pipeline import build_dataset, iter_splits, build_loader
from seizure_pred.training.engine.artifacts import ArtifactWriter
from seizure_pred.training.registries import MODELS, LOSSES, OPTIMIZERS, SCHEDULERS
from seizure_pred.training.engine.trainer import Trainer
from seizure_pred.inference.predictor import predict
from seizure_pred.analysis.runner import analyze_run


def test_smoke_pipeline_end_to_end():
    # Keep the smoke test fast and deterministic in constrained/CI environments.
    # (The default thread counts can make even tiny conv nets surprisingly slow.)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    cfg = TrainConfig()
    cfg.device = "cpu"
    cfg.amp = False
    cfg.epochs = 1
    cfg.task = "prediction"

    cfg.data.name = "synthetic"
    cfg.data.batch_size = 4
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False
    cfg.data.persistent_workers = False
    cfg.data.kwargs = {"n": 32, "c": 8, "t": 64, "pos_frac": 0.25, "seed": 1}

    cfg.model.name = "simple_cnn"
    cfg.model.num_classes = 1
    cfg.model.in_channels = 8
    cfg.model.kwargs = {"hidden": 8}

    dataset = build_dataset(cfg)
    train_set, val_set = next(iter(iter_splits(dataset, cfg.data)))
    train_loader = build_loader("torch", train_set, cfg, shuffle=True)
    val_loader = build_loader("torch", val_set, cfg, shuffle=False)

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

    with tempfile.TemporaryDirectory() as td:
        writer = ArtifactWriter(td)
        writer.write_schema()
        writer.write_config(cfg)

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            run_dir=td,
            artifact_writer=writer,
        )
        best = trainer.fit(train_loader=train_loader, val_loader=val_loader)
        assert os.path.exists(best)

        rows = list(predict(trainer.model, val_loader, device="cpu", threshold=0.5))
        writer.write_predictions(rows)

        # Plotting can be slow on first import of matplotlib (font cache) and is not
        # required for this end-to-end smoke test.
        rep = analyze_run(td, threshold=0.5, make_plots=False)
        assert "report" in rep and "f1" in rep["report"]
        assert os.path.exists(os.path.join(td, "analysis", "report.json"))


def test_smoke_train_from_config_end_to_end():
    # Keep the smoke test fast and deterministic
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    import yaml
    from seizure_pred.cli.train_cmd import train_from_config

    with tempfile.TemporaryDirectory() as td:
        config_data = {
            "device": "cpu",
            "amp": False,
            "epochs": 1,
            "task": "prediction",
            "save_dir": td,
            "run_name": "test_run",
            "data": {
                "name": "synthetic",
                "batch_size": 4,
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
                "split_method": "stratified",
                "n_folds": 2,
                "kwargs": {"n": 16, "c": 8, "t": 32, "pos_frac": 0.25, "seed": 1}
            },
            "model": {
                "name": "simple_cnn",
                "num_classes": 1,
                "in_channels": 8,
                "kwargs": {"hidden": 4}
            }
        }
        
        cfg_file = os.path.join(td, "config.yaml")
        with open(cfg_file, "w") as f:
            yaml.dump(config_data, f)

        # Call train_from_config which should train both splits
        train_from_config(cfg_file, strict=True)

        # check that runs/test_run/<timestamp>/split_0/checkpoints/best.pt exists
        run_name_dir = os.path.join(td, "test_run")
        assert os.path.exists(run_name_dir)
        timestamps = os.listdir(run_name_dir)
        assert len(timestamps) == 1
        stamp_dir = os.path.join(run_name_dir, timestamps[0])
        
        assert os.path.exists(os.path.join(stamp_dir, "split_0", "checkpoints", "best.pt"))
        assert os.path.exists(os.path.join(stamp_dir, "split_1", "checkpoints", "best.pt"))

