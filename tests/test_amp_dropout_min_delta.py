from __future__ import annotations

import math

import pytest
import torch

from seizure_pred.core.config import TrainConfig


def _make_trainer(tmp_path, *, amp: bool, device: str):
    import seizure_pred.models as models
    from seizure_pred.training.engine.pipeline import build_dataset, iter_splits
    from seizure_pred.training.engine.trainer import Trainer
    from seizure_pred.training.registries import LOSSES, MODELS, OPTIMIZERS
    from torch.utils.data import DataLoader

    models.register_all()
    cfg = TrainConfig()
    cfg.device = device
    cfg.amp = amp
    cfg.epochs = 2
    cfg.data.name = "synthetic"
    cfg.data.kwargs = {"n": 32, "c": 8, "t": 64, "pos_frac": 0.5, "seed": 1}
    cfg.model.name = "simple_cnn"
    cfg.model.in_channels = 8

    ds = build_dataset(cfg)
    train_set, val_set = next(iter(iter_splits(ds, cfg.data)))
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)
    model = MODELS.create(cfg.model.name, cfg.model)
    trainer = Trainer(
        model=model,
        loss_fn=LOSSES.create("bce_logits"),
        optimizer=OPTIMIZERS.create("adam", model.parameters(), lr=1e-3),
        scheduler=None,
        cfg=cfg,
        run_dir=str(tmp_path),
    )
    return trainer, train_loader, val_loader


def test_amp_on_cpu_falls_back_to_fp32(tmp_path, caplog):
    trainer, train_loader, val_loader = _make_trainer(tmp_path, amp=True, device="cpu")
    assert trainer.device.type == "cpu"
    assert trainer.use_amp is False
    assert trainer.scaler.is_enabled() is False
    loss = trainer._train_one_epoch(train_loader, {})
    assert math.isfinite(loss)
    out = trainer.evaluate(val_loader)
    assert out["val_logits"].dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_amp_cuda_smoke(tmp_path):
    trainer, train_loader, val_loader = _make_trainer(tmp_path, amp=True, device="cuda")
    assert trainer.use_amp is True
    assert trainer.scaler.is_enabled()
    loss = trainer._train_one_epoch(train_loader, {})
    assert math.isfinite(loss)
    out = trainer.evaluate(val_loader)
    assert out["val_logits"].dtype == torch.float32
    assert math.isfinite(out["loss"])


def _ce(**kw):
    import seizure_pred.models as models
    from seizure_pred.core.config import ModelConfig
    from seizure_pred.training.registries import MODELS

    models.register_all()
    if "ce_stsenet" not in MODELS:
        pytest.skip("ce_stsenet not registered (missing scipy)")
    return MODELS.create("ce_stsenet", ModelConfig(name="ce_stsenet", in_channels=18, num_classes=1, kwargs=kw))


def test_ce_stsenet_dropout():
    base = _ce()
    drop = _ce(dropout=0.5)
    assert list(base.state_dict().keys()) == list(drop.state_dict().keys())
    assert sum(p.numel() for p in base.parameters()) == sum(p.numel() for p in drop.parameters())
    x = torch.randn(4, 18, 640)
    drop.train()
    assert drop(x).shape == (4, 1, 1, 1)
    drop.eval()
    base.load_state_dict(drop.state_dict())
    base.eval()
    assert torch.allclose(base(x), drop(x), atol=1e-5)
    assert base.conv_classifier.dropout is None


def test_early_stopping_min_delta():
    from seizure_pred.training.callbacks.early_stopping import EarlyStopping

    def run(min_delta):
        es = EarlyStopping(monitor="val_loss", mode="min", patience=2, min_delta=min_delta)
        state = {}
        for v in (1.0, 0.95, 0.90):
            state["val_loss"] = v
            es.on_epoch_end(state)
        return state.get("stop", False)

    assert run(0.0) is False
    assert run(0.1) is True
