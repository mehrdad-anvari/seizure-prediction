from __future__ import annotations

from seizure_pred.core.config import DataConfig
from seizure_pred.data.chbmit_npz import CHBMITDataset
from seizure_pred.training.registries import DATASETS


def _maybe_build_transforms(spec):
    """Allow config files to specify transforms by name.

    Examples
    --------
    data:
      kwargs:
        online_transforms: ["instance_norm"]
    """
    if spec is None:
        return None
    if isinstance(spec, list) and spec and isinstance(spec[0], str):
        from seizure_pred.transforms.registry import create_transform

        return [create_transform(name) for name in spec]
    return spec


@DATASETS.register("chbmit_npz", help="CHB-MIT processed NPZ sessions (float/uint16)")
def build_chbmit_npz_dataset(cfg: DataConfig) -> CHBMITDataset:
    kwargs = dict(cfg.kwargs or {})
    if "online_transforms" in kwargs:
        kwargs["online_transforms"] = _maybe_build_transforms(kwargs.get("online_transforms"))
    if "offline_transforms" in kwargs:
        kwargs["offline_transforms"] = _maybe_build_transforms(kwargs.get("offline_transforms"))

    return CHBMITDataset(
        dataset_dir=cfg.dataset_dir,
        subject_id=cfg.subject_id,
        use_uint16=cfg.use_uint16,
        suffix=cfg.suffix,
        task=cfg.task,
        **kwargs,
    )
