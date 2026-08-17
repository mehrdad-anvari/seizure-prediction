from __future__ import annotations

import copy
import os
import platform
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None

try:
    from thop import profile as thop_profile  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    thop_profile = None


_MB = 1024 ** 2


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    valid = [float(value) for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def _maximum(values: Sequence[Optional[float]]) -> Optional[float]:
    valid = [float(value) for value in values if value is not None]
    return float(max(valid)) if valid else None


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def collect_model_metrics(model: torch.nn.Module, input_shape: Sequence[int]) -> Dict[str, Any]:
    """Collect static model complexity without modifying the training model."""
    parameters = list(model.parameters())
    total_params = sum(parameter.numel() for parameter in parameters)
    trainable_params = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())

    metrics: Dict[str, Any] = {
        "class_name": model.__class__.__name__,
        "input_shape_per_sample": [int(value) for value in input_shape],
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "non_trainable_parameters": int(total_params - trainable_params),
        "parameter_memory_mb": float(parameter_bytes / _MB),
        "buffer_memory_mb": float(buffer_bytes / _MB),
        "macs_per_sample": None,
        "flops_per_sample_estimate": None,
        "operation_count_method": None,
    }

    if thop_profile is None:
        return metrics

    try:
        profile_model = copy.deepcopy(model).cpu().eval()
        dummy = torch.zeros((1, *input_shape), dtype=torch.float32)
        macs, _ = thop_profile(profile_model, inputs=(dummy,), verbose=False)
        metrics.update({
            "macs_per_sample": int(macs),
            "flops_per_sample_estimate": int(2 * macs),
            "operation_count_method": "thop_macs; flops estimated as 2 * macs",
        })
    except Exception as exc:
        metrics["operation_count_error"] = str(exc)
    return metrics


@dataclass
class _Sample:
    process_cpu_percent: Optional[float]
    system_cpu_percent: Optional[float]
    process_rss_mb: Optional[float]
    gpu_utilization_percent: Optional[float]
    gpu_memory_allocated_mb: Optional[float]
    gpu_memory_reserved_mb: Optional[float]


class TrainingResourceMonitor:
    """Sample process/device utilization and record per-epoch training costs."""

    def __init__(self, model: torch.nn.Module, device: torch.device, input_shape: Sequence[int]) -> None:
        self.model = model
        self.device = device
        self.input_shape = tuple(int(value) for value in input_shape)
        self._samples: list[_Sample] = []
        self._epoch_sample_start = 0
        self._epoch_start_wall = 0.0
        self._epoch_start_cpu = 0.0
        self._train_start_wall = 0.0
        self._validation_start_wall = 0.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process(os.getpid()) if psutil is not None else None
        self._started_wall = 0.0
        self._started_cpu = 0.0
        self.epochs: list[Dict[str, Any]] = []

    def hardware_metrics(self) -> Dict[str, Any]:
        cpu_name = platform.processor() or platform.machine()
        hardware: Dict[str, Any] = {
            "platform": platform.platform(),
            "cpu_model": cpu_name,
            "cpu_logical_count": os.cpu_count(),
            "cpu_physical_count": psutil.cpu_count(logical=False) if psutil is not None else None,
            "system_memory_total_mb": float(psutil.virtual_memory().total / _MB) if psutil is not None else None,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "training_device": str(self.device),
        }
        if self.device.type == "cuda":
            properties = torch.cuda.get_device_properties(self.device)
            hardware.update({
                "gpu_name": properties.name,
                "gpu_total_memory_mb": float(properties.total_memory / _MB),
                "gpu_compute_capability": f"{properties.major}.{properties.minor}",
            })
        else:
            hardware.update({"gpu_name": None, "gpu_total_memory_mb": None, "gpu_compute_capability": None})
        return hardware

    def start(self) -> None:
        self._started_wall = time.perf_counter()
        self._started_cpu = time.process_time()
        if self._process is not None:
            self._process.cpu_percent(None)
            psutil.cpu_percent(None)
        self._thread = threading.Thread(target=self._sample_loop, name="training-resource-monitor", daemon=True)
        self._thread.start()

    def start_epoch(self) -> None:
        _cuda_sync(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._epoch_sample_start = len(self._samples)
        self._epoch_start_wall = time.perf_counter()
        self._epoch_start_cpu = time.process_time()
        self._train_start_wall = self._epoch_start_wall

    def finish_train(self) -> float:
        _cuda_sync(self.device)
        self._validation_start_wall = time.perf_counter()
        return float(self._validation_start_wall - self._train_start_wall)

    def finish_epoch(self, epoch: int, train_seconds: float, train_samples: int, validation_samples: int) -> Dict[str, Any]:
        _cuda_sync(self.device)
        finished = time.perf_counter()
        epoch_samples = self._samples[self._epoch_sample_start:]
        wall_seconds = finished - self._epoch_start_wall
        cpu_seconds = time.process_time() - self._epoch_start_cpu
        validation_seconds = finished - self._validation_start_wall
        row: Dict[str, Any] = {
            "epoch": int(epoch),
            "wall_time_seconds": float(wall_seconds),
            "train_time_seconds": float(train_seconds),
            "validation_time_seconds": float(validation_seconds),
            "process_cpu_time_seconds": float(cpu_seconds),
            "train_samples": int(train_samples),
            "validation_samples": int(validation_samples),
            "train_samples_per_second": float(train_samples / train_seconds) if train_seconds > 0 else None,
            "process_cpu_utilization_percent_mean": _mean([sample.process_cpu_percent for sample in epoch_samples]),
            "system_cpu_utilization_percent_mean": _mean([sample.system_cpu_percent for sample in epoch_samples]),
            "process_memory_rss_mb_peak": _maximum([sample.process_rss_mb for sample in epoch_samples]),
            "gpu_utilization_percent_mean": _mean([sample.gpu_utilization_percent for sample in epoch_samples]),
            "gpu_memory_allocated_mb_peak": float(torch.cuda.max_memory_allocated(self.device) / _MB) if self.device.type == "cuda" else None,
            "gpu_memory_reserved_mb_peak": float(torch.cuda.max_memory_reserved(self.device) / _MB) if self.device.type == "cuda" else None,
        }
        self.epochs.append(row)
        return row

    def finish(self) -> Dict[str, Any]:
        _cuda_sync(self.device)
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        total_wall = time.perf_counter() - self._started_wall
        total_cpu = time.process_time() - self._started_cpu
        gpu_allocated_peaks = [sample.gpu_memory_allocated_mb for sample in self._samples]
        gpu_allocated_peaks.extend(epoch["gpu_memory_allocated_mb_peak"] for epoch in self.epochs)
        gpu_reserved_peaks = [sample.gpu_memory_reserved_mb for sample in self._samples]
        gpu_reserved_peaks.extend(epoch["gpu_memory_reserved_mb_peak"] for epoch in self.epochs)
        return {
            "model": collect_model_metrics(self.model, self.input_shape),
            "hardware": self.hardware_metrics(),
            "training": {
                "total_wall_time_seconds": float(total_wall),
                "total_process_cpu_time_seconds": float(total_cpu),
                "process_cpu_utilization_percent_mean": _mean([sample.process_cpu_percent for sample in self._samples]),
                "system_cpu_utilization_percent_mean": _mean([sample.system_cpu_percent for sample in self._samples]),
                "process_memory_rss_mb_peak": _maximum([sample.process_rss_mb for sample in self._samples]),
                "gpu_utilization_percent_mean": _mean([sample.gpu_utilization_percent for sample in self._samples]),
                "gpu_utilization_percent_peak": _maximum([sample.gpu_utilization_percent for sample in self._samples]),
                "gpu_memory_allocated_mb_peak": _maximum(gpu_allocated_peaks),
                "gpu_memory_reserved_mb_peak": _maximum(gpu_reserved_peaks),
                "epochs": self.epochs,
            },
        }

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(0.2):
            process_cpu = self._process.cpu_percent(None) if self._process is not None else None
            system_cpu = psutil.cpu_percent(None) if psutil is not None else None
            rss = float(self._process.memory_info().rss / _MB) if self._process is not None else None
            gpu_utilization = None
            gpu_allocated = None
            gpu_reserved = None
            if self.device.type == "cuda":
                gpu_allocated = float(torch.cuda.memory_allocated(self.device) / _MB)
                gpu_reserved = float(torch.cuda.memory_reserved(self.device) / _MB)
                try:
                    gpu_utilization = float(torch.cuda.utilization(self.device))
                except Exception:
                    pass
            self._samples.append(_Sample(process_cpu, system_cpu, rss, gpu_utilization, gpu_allocated, gpu_reserved))


def infer_input_shape(loader: Any, *, mil: bool = False) -> tuple[int, ...]:
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        dataset = getattr(loader, "base", None)
    if dataset is None:
        raise ValueError("Cannot infer model input shape from loader")
    x, _, _ = dataset[0]
    shape = tuple(int(value) for value in x.shape)
    return shape[1:] if mil else shape


def loader_sample_count(loader: Any) -> int:
    indices = getattr(loader, "all_indices", None)
    if indices is not None:
        return int(len(indices))
    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        return int(len(dataset))
    return 0
