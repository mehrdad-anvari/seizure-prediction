from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

TaskType = Literal["prediction", "detection"]
SchedStep = Literal["epoch", "step"]


@dataclass
class DataConfig:
    name: str = "chbmit_npz"
    dataset_dir: str = "data/BIDS_CHB-MIT"
    subject_id: str = "01"
    use_uint16: bool = False
    suffix: str = "fd_5s_szx5_prex5"
    task: TaskType = "prediction"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str = "simple_cnn"
    num_classes: int = 2
    in_channels: Optional[int] = None
    sfreq: Optional[float] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimConfig:
    name: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedConfig:
    name: Optional[str] = None
    step: SchedStep = "epoch"
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig:
    name: str = "bce_logits"
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CallbackConfig:
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    task: TaskType = "prediction"
    seed: int = 42
    device: str = "cuda"
    epochs: int = 50
    grad_clip_norm: Optional[float] = 1.0
    amp: bool = True
    log_every: int = 25
    val_every: int = 1
    save_dir: str = "runs"
    run_name: str = "default"

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    sched: SchedConfig = field(default_factory=SchedConfig)

    callbacks: list[CallbackConfig] = field(default_factory=list)


def asdict_shallow(dc_obj: Any) -> Dict[str, Any]:
    if not hasattr(dc_obj, "__dataclass_fields__"):
        raise TypeError("asdict_shallow expects a dataclass instance")
    out: Dict[str, Any] = {}
    for k in dc_obj.__dataclass_fields__.keys():
        v = getattr(dc_obj, k)
        if hasattr(v, "__dataclass_fields__"):
            out[k] = asdict_shallow(v)
        elif isinstance(v, list) and v and hasattr(v[0], "__dataclass_fields__"):
            out[k] = [asdict_shallow(x) for x in v]
        else:
            out[k] = v
    return out
