import torch
import numpy as np

class InstanceNormTransform:
    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def __call__(self, eeg: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(eeg, np.ndarray):
            x = torch.from_numpy(eeg)
        else:
            x = eeg

        # Normalize per channel across time (common in EEG)
        # shape expected: (channels, time)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / (std + self.eps)

        return x.numpy()  # keep dataset outputs consistent

class RobustNormTransform:
    def __init__(
        self,
        eps: float = 1e-5,
        mad_scale: float = 1.4826,
    ):
        self.eps = eps
        self.mad_scale = mad_scale

    def __call__(self, eeg: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(eeg, np.ndarray):
            x = torch.from_numpy(eeg)
        else:
            x = eeg

        # Expected shape: (channels, time)

        median = x.median(dim=-1, keepdim=True).values

        mad = (x - median).abs().median(dim=-1, keepdim=True).values

        robust_std = self.mad_scale * mad

        x = (x - median) / (robust_std + self.eps)

        return x.numpy()