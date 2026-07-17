"""Explainability utilities (optional, requires Captum).

Ported (and trimmed) from the original repository's ``xai.py``. It computes
input attributions with Captum's :class:`IntegratedGradients` and aggregates
them per channel for EEG models.

Because Captum is an optional dependency, importing this module does not fail;
the heavy import only happens when an attribution function is actually called.
Install with ``pip install captum``.

Typical use::

    from seizure_pred.inference.xai import channel_attributions

    attrs = channel_attributions(model, x, n_steps=50)   # (B, C, T)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def _require_captum():
    try:
        from captum.attr import IntegratedGradients  # type: ignore

        return IntegratedGradients
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "XAI requires the optional 'captum' package. Install with: pip install captum"
        ) from e


@torch.no_grad()
def _baseline(x: torch.Tensor) -> torch.Tensor:
    """A zero (black) baseline of the same shape as x."""
    return torch.zeros_like(x)


def integrated_gradients(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    target: int = 1,
    n_steps: int = 50,
    device: str = "cpu",
) -> torch.Tensor:
    """Return Captum IntegratedGradients attributions (same shape as inputs).

    ``model`` must produce binary logits (or (B,2) logits). ``target`` selects
    the output class to attribute (1 = positive/preictal class).
    """
    IntegratedGradients = _require_captum()
    model = model.to(device).eval()
    inputs = inputs.to(device)

    def forward(x):
        out = model(x)
        if out.ndim == 1:
            return out.unsqueeze(1)
        if out.shape[-1] == 1:
            return torch.cat([-out, out], dim=-1)
        return out

    ig = IntegratedGradients(forward)
    baseline = _baseline(inputs)
    attr = ig.attribute(inputs, baselines=baseline, target=target, n_steps=int(n_steps))
    return attr.detach().cpu()


def channel_attributions(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    target: int = 1,
    n_steps: int = 50,
    device: str = "cpu",
) -> np.ndarray:
    """Return per-channel mean absolute attribution, shape (B, C).

    Useful for ranking the most influential EEG channels for the positive class.
    """
    attr = integrated_gradients(model, inputs, target=target, n_steps=n_steps, device=device)
    return attr.abs().mean(dim=tuple(range(2, attr.ndim))).cpu().numpy()


def plot_topomap(
    channel_attr: np.ndarray,
    ch_names: list[str],
    *,
    save_path: Optional[str] = None,
    title: str = "Channel attributions",
):
    """Plot a 2-D topographic map of channel attributions (requires mne)."""
    try:
        import mne  # type: ignore
        import matplotlib  # type: ignore

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Topomap plotting requires 'mne' and 'matplotlib'") from e

    montage = mne.channels.make_standard_montage("standard_1020")
    info = mne.create_info(ch_names, 1000.0, "eeg").set_montage(montage)
    fig, ax = plt.subplots(figsize=(5, 5))
    mne.viz.plot_topomap(np.asarray(channel_attr, dtype=float), info, axes=ax, show=False)
    ax.set_title(title)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return fig
