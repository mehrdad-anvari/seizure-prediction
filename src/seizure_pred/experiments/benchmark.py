"""Model benchmarking utility (ported from the original ``benchmark_models.py``).

Measures, per model and batch size:

- total / trainable parameter count
- FLOPs and MACs (requires the optional ``thop`` package)
- CPU and (if available) GPU inference latency + throughput (samples/sec)
- GPU peak memory and CPU->GPU speedup
- optional ``torchinfo`` layer summary

Models are built through the :data:`seizure_pred.training.registries.MODELS`
registry, so any registered model name can be benchmarked.

Example::

    seizure-pred benchmark --models eegnet eegwavenet --batch-sizes 1 32

Programmatic::

    from seizure_pred.experiments.benchmark import benchmark_all_models
    df = benchmark_all_models(["eegnet", "eegwavenet"], batch_sizes=[1, 32])
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from seizure_pred.core.config import ModelConfig

try:
    from thop import profile, clever_format  # type: ignore

    _HAS_THOP = True
except Exception:  # pragma: no cover
    _HAS_THOP = False

try:
    from torchinfo import summary as _torchinfo_summary  # type: ignore

    _HAS_TORCHINFO = True
except Exception:  # pragma: no cover
    _HAS_TORCHINFO = False


def _build_model(model_name: str, *, in_channels: int = 18, chunk_size: int = 640,
                 num_classes: int = 2, kwargs: Optional[dict] = None):
    from seizure_pred.training.registries import MODELS

    cfg = ModelConfig(
        name=model_name,
        num_classes=num_classes,
        in_channels=in_channels,
        kwargs=kwargs or {},
    )
    return MODELS.create(model_name, cfg)


class ModelBenchmark:
    """Benchmark a single registered model."""

    def __init__(self, model_name: str, input_shape: Tuple[int, int, int] = (1, 18, 640),
                 *, model_kwargs: Optional[dict] = None):
        self.model_name = model_name
        self.input_shape = input_shape
        self.model_kwargs = model_kwargs or {}

    def _fresh_model(self) -> torch.nn.Module:
        _, channels, t = self.input_shape
        return _build_model(self.model_name, in_channels=channels, chunk_size=t, kwargs=self.model_kwargs)

    @staticmethod
    def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    def measure_flops(self, model: torch.nn.Module, device: str = "cpu"):
        if not _HAS_THOP:
            return None, None
        model = model.to(device).eval()
        dummy = torch.randn(self.input_shape).to(device)
        try:
            flops, params = profile(model, inputs=(dummy,), verbose=False)
            return clever_format([flops, params], "%.3f")
        except Exception as e:
            print(f"  Warning: could not calculate FLOPs for {self.model_name}: {e}")
            return None, None

    def measure_inference_time(self, model: torch.nn.Module, device: str = "cpu",
                               n_runs: int = 100, warmup_runs: int = 10) -> Dict[str, float]:
        model = model.to(device).eval()
        dummy = torch.randn(self.input_shape).to(device)
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = model(dummy)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000.0)
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
            "median": float(np.median(times)),
        }

    def benchmark(self, n_runs: int = 100, batch_size: int = 32) -> Dict[str, Any]:
        self.input_shape = (batch_size, self.input_shape[1], self.input_shape[2])
        results: Dict[str, Any] = {
            "model_name": self.model_name,
            "batch_size": batch_size,
            "input_shape": str(self.input_shape),
        }

        model = self._fresh_model()
        total_params, trainable_params = self.count_parameters(model)
        results["total_params"] = total_params
        results["trainable_params"] = trainable_params

        cpu_times = self.measure_inference_time(model, device="cpu", n_runs=n_runs)
        results.update({
            "cpu_mean_ms": cpu_times["mean"], "cpu_std_ms": cpu_times["std"],
            "cpu_min_ms": cpu_times["min"], "cpu_max_ms": cpu_times["max"],
            "cpu_median_ms": cpu_times["median"],
            "cpu_throughput_samples_per_sec": (batch_size * 1000.0) / max(cpu_times["mean"], 1e-9),
        })

        if torch.cuda.is_available():
            model_gpu = self._fresh_model()
            gpu_times = self.measure_inference_time(model_gpu, device="cuda", n_runs=n_runs)
            results.update({
                "gpu_mean_ms": gpu_times["mean"], "gpu_std_ms": gpu_times["std"],
                "gpu_min_ms": gpu_times["min"], "gpu_max_ms": gpu_times["max"],
                "gpu_median_ms": gpu_times["median"],
                "gpu_throughput_samples_per_sec": (batch_size * 1000.0) / max(gpu_times["mean"], 1e-9),
                "gpu_speedup": cpu_times["mean"] / max(gpu_times["mean"], 1e-9),
            })
            try:
                torch.cuda.reset_peak_memory_stats()
                model_gpu = model_gpu.cuda().eval()
                dummy = torch.randn(self.input_shape).cuda()
                with torch.no_grad():
                    _ = model_gpu(dummy)
                results["gpu_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
            except Exception as e:
                print(f"  Could not measure GPU memory: {e}")
        else:
            results["gpu_mean_ms"] = None
            results["gpu_speedup"] = None

        single_shape = (1, self.input_shape[1], self.input_shape[2])
        self.input_shape = single_shape
        flops, params_str = self.measure_flops(self._fresh_model(), device="cpu")
        results["flops"] = flops
        results["params_formatted"] = params_str

        if _HAS_TORCHINFO:
            try:
                _torchinfo_summary(model, input_size=single_shape, verbose=0)
            except Exception as e:
                print(f"  Could not generate torchinfo summary: {e}")

        return results


def benchmark_all_models(
    models: List[str],
    batch_sizes: List[int] = [1, 16, 32],
    n_runs: int = 100,
    output_dir: str = "benchmark_results",
    *,
    channels: int = 18,
    time_points: int = 640,
    model_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """Benchmark multiple models across batch sizes; save a timestamped CSV."""
    import seizure_pred.models as models_mod

    models_mod.register_all()

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    all_results: List[Dict[str, Any]] = []

    for model_name in models:
        try:
            for batch_size in batch_sizes:
                print(f"\n{'#' * 60}\nModel: {model_name} | Batch: {batch_size}\n{'#' * 60}")
                bench = ModelBenchmark(model_name, input_shape=(batch_size, channels, time_points),
                                       model_kwargs=model_kwargs)
                all_results.append(bench.benchmark(n_runs=n_runs, batch_size=batch_size))
        except Exception as e:
            print(f"\nError benchmarking {model_name}: {e}")
            import traceback

            traceback.print_exc()

    df = pd.DataFrame(all_results)
    csv_path = output_path / f"benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    return df
