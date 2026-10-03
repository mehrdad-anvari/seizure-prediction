import torch
import numpy as np

_ABS_FLOOR = 1e-12


def _denominator(scale: torch.Tensor, eps: float, eps_mode: str) -> torch.Tensor:
    """Per-channel denominator for normalization.

    "absolute": scale + eps (legacy; dominates for data stored in volts, ~1e-5).
    "relative": scale + eps * ref, ref = median of the per-channel scales of the
    window (clamped to a tiny floor so all-zero windows stay finite). The result
    is invariant to the unit of the input (V vs uV).
    """
    if eps_mode == "absolute":
        return scale + eps
    if eps_mode != "relative":
        raise ValueError(f"eps_mode must be 'relative' or 'absolute', got {eps_mode!r}")
    ref = scale.median(dim=-2, keepdim=True).values if scale.ndim >= 2 else scale.median()
    return scale + eps * ref.clamp_min(_ABS_FLOOR) + _ABS_FLOOR


class InstanceNormTransform:
    def __init__(self, eps: float = 1e-5, eps_mode: str = "relative"):
        if eps_mode not in ("relative", "absolute"):
            raise ValueError(f"eps_mode must be 'relative' or 'absolute', got {eps_mode!r}")
        self.eps = eps
        self.eps_mode = eps_mode

    def __call__(self, eeg: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(eeg, np.ndarray):
            x = torch.from_numpy(eeg)
        else:
            x = eeg

        # Normalize per channel across time (common in EEG)
        # shape expected: (channels, time)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / _denominator(std, self.eps, self.eps_mode)

        return x.numpy()  # keep dataset outputs consistent

class RobustNormTransform:
    def __init__(
        self,
        eps: float = 1e-5,
        mad_scale: float = 1.4826,
        eps_mode: str = "relative",
    ):
        if eps_mode not in ("relative", "absolute"):
            raise ValueError(f"eps_mode must be 'relative' or 'absolute', got {eps_mode!r}")
        self.eps = eps
        self.eps_mode = eps_mode
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

        x = (x - median) / _denominator(robust_std, self.eps, self.eps_mode)

        return x.numpy()