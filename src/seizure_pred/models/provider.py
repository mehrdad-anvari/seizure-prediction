from __future__ import annotations

"""Legacy model-provider helpers.

The modern API uses :mod:`seizure_pred.training.registries` (MODELS.create(...)).
This module is retained for back-compat with older scripts that called
``get_builder(...)`` to obtain a callable returning a fresh model each fold.

Key design goal: importing this module should *not* depend on the old repo's
``models.*`` top-level package. All imports are resolved from the current
:mod:`seizure_pred.models` package and are performed lazily.
"""

from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn


channels = [
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FZ-CZ",
    "CZ-PZ",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
]


def initialize_edge_weights(num_nodes: int, seed: int = 42, diag_value: float = 1.0):
    """Initialize a fully-connected (including self-loops) edge_index/edge_weight."""
    torch.manual_seed(seed)
    row, col = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing="ij")
    edge_index = torch.stack([row.reshape(-1), col.reshape(-1)], dim=0)
    edge_weight = torch.empty(num_nodes, num_nodes)
    torch.nn.init.xavier_normal_(edge_weight)
    edge_weight.fill_diagonal_(diag_value)
    return edge_index, edge_weight.reshape(-1)


def model_builder(model_class: type[nn.Module], **kwargs) -> Callable[[], nn.Module]:
    """Return a callable that builds a fresh model instance."""

    def build() -> nn.Module:
        return model_class(**kwargs)

    return build


def _import_or_raise(module: str, symbol: str):
    """Lazy import helper with a clearer error message."""
    try:
        mod = __import__(module, fromlist=[symbol])
        return getattr(mod, symbol)
    except Exception as e:  # pragma: no cover
        raise ImportError(f"Could not import {symbol} from {module}.") from e


def get_builder(model: str = "CE-stSENet") -> Callable[[], nn.Module]:
    """Return a builder callable for a given legacy model name."""

    match model:
        case "EEGNET" | "EEGNet":
            EEGNet = _import_or_raise("seizure_pred.models.eegnet", "EEGNet")
            return model_builder(
                EEGNet,
                chunk_size=640,
                num_electrodes=18,
                dropout=0.5,
                kernel_1=64,
                kernel_2=16,
                F1=8,
                F2=16,
                D=2,
                num_classes=2,
            )

        case "CE-stSENet" | "ce_stsenet":
            CE_stSENet = _import_or_raise("seizure_pred.models.ce_stsenet.ce_stsenet", "CE_stSENet")
            return model_builder(CE_stSENet, inc=18, class_num=2, si=128)

        case "cspnet" | "CSPNet":
            CSPNet = _import_or_raise("seizure_pred.models.cspnet", "CSPNet")
            return model_builder(
                CSPNet,
                chunk_size=128 * 5,
                num_electrodes=18,
                num_classes=2,
                dropout=0.5,
                num_filters_t=20,
                filter_size_t=25,
                num_filters_s=2,
                filter_size_s=-1,
                pool_size_1=100,
                pool_stride_1=25,
            )

        case "stnet" | "STNet":
            STNet = _import_or_raise("seizure_pred.models.stnet", "STNet")
            return model_builder(STNet, chunk_size=128 * 5, grid_size=(9, 9), num_classes=2, dropout=0.2)

        case "simple-vit" | "simplevit" | "SimpleViT":
            SimpleViT = _import_or_raise("seizure_pred.models.simplevit", "SimpleViT")
            return model_builder(
                SimpleViT,
                chunk_size=128 * 5,
                grid_size=(9, 9),
                t_patch_size=32,
                s_patch_size=(3, 3),
                hid_channels=32,
                depth=3,
                heads=4,
                head_channels=8,
                mlp_channels=32,
                num_classes=2,
            )

        case "TSception":
            TSception = _import_or_raise("seizure_pred.models.tsception", "TSception")
            return model_builder(
                TSception,
                num_classes=2,
                input_size=(18, 640),
                sampling_rate=256,
                num_T=9,
                num_S=6,
                hidden=128,
                dropout_rate=0.2,
            )

        case "FBMSNet":
            FBMSNet = _import_or_raise("seizure_pred.models.fbmsnet", "FBMSNet")
            return model_builder(FBMSNet, nChan=18, nTime=640, nClass=2)

        case "LaBraM":
            LaBraM = _import_or_raise("seizure_pred.models.labram", "LaBraM")
            return model_builder(
                LaBraM,
                chunk_size=128 * 5,
                patch_size=80,
                embed_dim=80,
                depth=6,
                num_heads=6,
                mlp_ratio=1,
                qk_norm=partial(nn.LayerNorm, eps=1e-6),
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                init_values=0.1,
                drop_rate=0.1,
                electrodes=channels,
            )

        case "RGNN" | "rgnn":
            RGNN_Model = _import_or_raise("seizure_pred.models.rgnn", "RGNN_Model")
            return model_builder(RGNN_Model, num_channels=18, num_classes=2)

        case "DGCNN2" | "dgcnn2":
            DGCNN_Model = _import_or_raise("seizure_pred.models.dgcnn2", "DGCNN_Model")
            return model_builder(DGCNN_Model, num_channels=18, num_classes=2)

        case "DGCNN" | "dgcnn":
            DGCNN = _import_or_raise("seizure_pred.models.dgcnn", "DGCNN")
            return model_builder(DGCNN, num_channels=18, num_classes=2)

        case "Conformer" | "conformer":
            Conformer = _import_or_raise("seizure_pred.models.conformer.model", "Conformer")
            return model_builder(Conformer, in_channels=18, num_classes=2)

        case "TSLANet":
            TSLANet = _import_or_raise("seizure_pred.models.tslanet", "TSLANet")
            return model_builder(TSLANet, in_channels=18, num_classes=2)

        case "LMDA":
            LMDA = _import_or_raise("seizure_pred.models.lmda", "LMDA")
            return model_builder(LMDA, in_channels=18, num_classes=2)

        case "MB_dMGC_CWTFFNet":
            MB_dMGC_CWTFFNet = _import_or_raise("seizure_pred.models.mb_dmgc_cwtffnet", "MB_dMGC_CWTFFNet")
            return model_builder(MB_dMGC_CWTFFNet, in_channels=18, num_classes=2)

        case "EEGBandClassifier":
            EEGBandClassifier = _import_or_raise("seizure_pred.models.eeg_band_classifier", "EEGBandClassifier")
            return model_builder(EEGBandClassifier, in_channels=18, num_classes=2)

        case "EEGWaveNet":
            EEGWaveNet = _import_or_raise("seizure_pred.models.eegwavenet", "EEGWaveNet")
            return model_builder(EEGWaveNet, in_channels=18, num_classes=2)

        case other:
            raise ValueError(
                f"Unknown legacy model '{other}'. Prefer the registry API: MODELS.create(name, cfg)."
            )
