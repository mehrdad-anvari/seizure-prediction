"""Checks for the example model configs in ``configs/models/``.

Every config is parsed the way ``seizure-pred train --strict`` parses it, then
each component named in it is actually constructed: model, loss, optimizer,
scheduler, and the offline transform list. This catches the failure mode these
files are most prone to -- a kwarg or registry name that drifted away from the
builder -- without needing the CHB-MIT data to be present.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "models"

CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))

# The window every study uses: 18 CHB-MIT channels, 5 s at 128 Hz.
CHANNELS = 18
SAMPLES = 640


def _config_ids() -> list[str]:
    return [p.stem for p in CONFIGS]


def test_config_dir_is_populated():
    assert CONFIGS, f"no example configs found in {CONFIG_DIR}"


@pytest.fixture(scope="module")
def registries():
    import seizure_pred.models as models
    import seizure_pred.training as training

    training.register_all()
    models.register_all()
    from seizure_pred.training import registries as reg

    return reg


@pytest.fixture(params=CONFIGS, ids=_config_ids())
def cfg(request):
    from seizure_pred.core.config import TrainConfig
    from seizure_pred.core.io import from_dict, load_dict
    from seizure_pred.core.validate import validate_config_dict

    yaml = pytest.importorskip("yaml")
    _ = yaml
    raw = load_dict(request.param)
    validate_config_dict(raw, TrainConfig)  # what --strict does
    return from_dict(TrainConfig, raw)


def test_window_shape_matches_the_studies(cfg):
    """`in_channels` / `chunk_size`, when given, must match the real windows."""
    if cfg.model.in_channels is not None:
        assert int(cfg.model.in_channels) == CHANNELS
    chunk = (cfg.model.kwargs or {}).get("chunk_size")
    if chunk is not None:
        assert int(chunk) == SAMPLES


def test_binary_head(cfg):
    """These configs pair a single logit with a binary loss."""
    assert int(cfg.model.num_classes) == 1


def test_offline_transforms_resolve(cfg):
    """Transform names must exist, and none of these models wants a filterbank."""
    from seizure_pred.transforms.registry import create_transform

    names = (cfg.data.kwargs or {}).get("offline_transforms", [])
    assert isinstance(names, list)
    for name in names:
        assert "filterbank" not in name, (
            f"{cfg.model.name} consumes raw EEG; {name} rewrites the time axis"
        )
        try:
            create_transform(name)
        except ImportError as exc:  # optional scipy/mne extras
            pytest.skip(f"transform '{name}' needs an optional dependency: {exc}")


def test_model_builds(cfg, registries):
    if cfg.model.name not in registries.MODELS:
        pytest.skip(f"model '{cfg.model.name}' is not registered (missing optional dependency)")
    model = registries.MODELS.create(cfg.model.name, cfg.model)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_training_components_build(cfg, registries):
    torch = pytest.importorskip("torch")

    loss_fn = registries.LOSSES.create(cfg.loss.name, **(cfg.loss.kwargs or {}))
    assert callable(loss_fn)

    params = [torch.nn.Parameter(torch.zeros(1))]
    optimizer = registries.OPTIMIZERS.create(
        cfg.optim.name,
        params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        **(cfg.optim.kwargs or {}),
    )
    assert optimizer.param_groups

    if cfg.sched.name:
        scheduler = registries.SCHEDULERS.create(
            cfg.sched.name, optimizer, **(cfg.sched.kwargs or {})
        )
        assert hasattr(scheduler, "step")


def test_callbacks_build(cfg, registries):
    """Callbacks named in a config must be registered and constructible.

    ``from_dict`` leaves the list as plain dicts (it does not recurse into
    ``list[CallbackConfig]``), which is also what ``train_cmd`` expects.
    """
    for item in cfg.callbacks or []:
        name = item["name"] if isinstance(item, dict) else item.name
        kwargs = item.get("kwargs", {}) if isinstance(item, dict) else (item.kwargs or {})
        assert name in registries.CALLBACKS, f"unknown callback '{name}'"
        assert registries.CALLBACKS.create(name, **kwargs) is not None


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("exponential", {"gamma": 0.95}),
        ("cosine_warm_restarts", {"T_0": 10, "T_mult": 2, "eta_min": 0.0003}),
    ],
)
def test_new_schedulers_change_lr(name, kwargs, registries):
    """The two schedulers added for the paper recipes actually move the LR."""
    torch = pytest.importorskip("torch")

    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = registries.OPTIMIZERS.create("adamw", [param], lr=0.1)
    scheduler = registries.SCHEDULERS.create(name, optimizer, **kwargs)

    start = optimizer.param_groups[0]["lr"]
    for _ in range(5):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] != start
