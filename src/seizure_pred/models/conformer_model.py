from __future__ import annotations

from seizure_pred.core.config import ModelConfig
from seizure_pred.training.registries import MODELS

# Original implementation lives in seizure_pred.models.conformer.model
from .conformer.model import Conformer  # type: ignore


@MODELS.register("conformer", help="Conformer model imported from original seizure-prediction-main/models/conformer")
def build_conformer(cfg: ModelConfig):
    kw = dict(cfg.kwargs or {})
    if cfg.in_channels is not None and "in_channels" not in kw:
        kw["in_channels"] = cfg.in_channels
    if cfg.num_classes is not None and "num_classes" not in kw:
        kw["num_classes"] = cfg.num_classes
    return Conformer(**kw)
