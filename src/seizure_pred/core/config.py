from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

TaskType = Literal["prediction", "detection"]
SchedStep = Literal["epoch", "step"]


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing.

    Attributes:
        name: Name of the dataset builder plugin (e.g., "chbmit_npz", "synthetic").
        dataset_dir: Path to the root BIDS/preprocessed dataset directory.
        subject_id: Subject ID to train/evaluate on (e.g., "01").
        use_uint16: Whether to load preprocessed float data scaled to uint16 (to save memory).
        suffix: Suffix pattern for matching preprocessed files (e.g., "fd_5s_szx5_prex5").
        task: Target task type ("prediction" or "detection").
        batch_size: Minibatch size for training and evaluation.
        num_workers: Number of CPU workers for dataloader multiprocessing.
        pin_memory: Whether to copy tensors into CUDA pinned memory.
        persistent_workers: Whether dataloader workers remain alive between epochs.
        split_method: Method for train/validation/test splitting ("kfold", "loo").
        dataloader_type: Type of data loader ("undersample", "mil").
        kwargs: Additional dataset-specific builder keyword arguments.
    """
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
    split_method: Literal["leave_one_preictal", "leave_one_out", "stratified"] = "stratified"
    n_folds: int = 5,
    shuffle: bool = True,
    dataloader_type: Literal["undersample", "mil", "torch"] = "undersample"
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for the neural network model.

    Attributes:
        name: Name of the model plugin registered in the MODELS registry (e.g., "simple_cnn", "eegwavenet").
        num_classes: Number of classification target classes (usually 1 for binary logits).
        in_channels: Number of input signal channels (inferred from data if None).
        sfreq: Signal sampling frequency in Hz (inferred if None).
        kwargs: Additional model-specific initialization keyword arguments.
    """
    name: str = "simple_cnn"
    num_classes: int = 2
    in_channels: Optional[int] = None
    sfreq: Optional[float] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimConfig:
    """Configuration for the optimizer.

    Attributes:
        name: Name of the optimizer plugin (e.g., "adam", "sgd").
        lr: Base learning rate.
        weight_decay: Weight decay (L2 penalty) coefficient.
        kwargs: Additional optimizer-specific parameters.
    """
    name: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedConfig:
    """Configuration for the learning rate scheduler.

    Attributes:
        name: Name of the scheduler plugin (e.g., "cosine", "step", or null for no scheduler).
        step: Frequency of scheduler step calls ("epoch" or "step").
        kwargs: Additional scheduler-specific parameters.
    """
    name: Optional[str] = None
    step: SchedStep = "epoch"
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig:
    """Configuration for the loss function.

    Attributes:
        name: Name of the loss function plugin registered in LOSSES (e.g., "bce_logits", "focal").
        kwargs: Additional loss-specific parameters.
    """
    name: str = "bce_logits"
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CallbackConfig:
    """Configuration for training callbacks.

    Attributes:
        name: Registered name of the callback in CALLBACKS registry.
        kwargs: Initialization keyword arguments passed to build the callback.
    """
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    """Overall training pipeline configuration.

    Attributes:
        task: Problem type, either "prediction" (preictal vs interictal) or "detection" (seizure vs non-seizure).
        seed: Random seed for repeatability.
        device: Target execution device (e.g., "cuda", "cpu").
        epochs: Number of complete epochs to train.
        grad_clip_norm: Maximum gradient norm for clipping (disabled if None).
        amp: Whether to use Automatic Mixed Precision (FP16) training.
        log_every: Step frequency for training progress logging.
        val_every: Epoch frequency for validation evaluation.
        save_dir: Base directory for saving training run folders.
        run_name: Custom run tag/prefix.
        data: Nested dataset configuration.
        model: Nested model configuration.
        loss: Nested loss configuration.
        optim: Nested optimizer configuration.
        sched: Nested learning rate scheduler configuration.
        callbacks: List of callback configurations to instantiate.
    """
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
