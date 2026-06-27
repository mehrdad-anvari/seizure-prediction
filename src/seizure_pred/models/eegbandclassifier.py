from __future__ import annotations

import torch
import torch.nn as nn
from seizure_pred.models.eeg_band_classifier import (
    BandFeatureExtractor,
    EEGBandEmbeddingNet,
    EEGBandClassifier,
)

# ---- seizure_pred registry glue ----
from seizure_pred.core.config import ModelConfig
from seizure_pred.training.registries import MODELS

@MODELS.register("eegbandclassifier", help="Imported from original seizure-prediction-main/models/eegbandclassifier.py")
def build_eegbandclassifier(cfg: ModelConfig):
    kw = dict(cfg.kwargs or {})
    # Map common config fields
    n_classes = int(getattr(cfg, "num_classes", 2))
    n_bands = int(kw.get("n_bands", 5))
    return EEGBandClassifier(n_classes=n_classes, n_bands=n_bands)
