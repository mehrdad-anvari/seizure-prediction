from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DeterminismConfig:
    seed: int = 42
    deterministic: bool = False
    cudnn_benchmark: bool = False
    cuda_deterministic: bool = False


def seed_everything(cfg: DeterminismConfig | None = None, *, seed: int | None = None) -> None:
    """Seed python/numpy/torch and optionally enable deterministic algorithms.

    Notes:
      - Deterministic mode may reduce performance and some ops may still be nondeterministic
        depending on CUDA/cuDNN versions.
    """
    # Backwards/CLI-friendly behavior:
    # - If cfg is None, use default DeterminismConfig()
    # - If `seed` is provided, it overrides cfg.seed
    if cfg is None:
        cfg = DeterminismConfig()
    if seed is not None:
        cfg = DeterminismConfig(
            seed=int(seed),
            deterministic=cfg.deterministic,
            cudnn_benchmark=cfg.cudnn_benchmark,
            cuda_deterministic=cfg.cuda_deterministic,
        )

    seed = int(cfg.seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)

    if cfg.deterministic:
        # torch>=1.8
        torch.use_deterministic_algorithms(True, warn_only=True)

    if cfg.cuda_deterministic:
        # cuBLAS determinism (best-effort)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
