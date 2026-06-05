"""Minimal XAI demo for seizure_pred models (new structure).

The original repo had exploratory explainability code with several environment
assumptions. This updated script integrates with the new config + registry
system and keeps heavy explainability libs optional.

Current demo:
  - Load a checkpoint and run a simple gradient-based saliency map on a single
    EEG sample (user-provided .npy/.npz).

Optional dependencies:
  - `captum` (recommended) for Integrated Gradients.

Example:
  python examples/scripts/xai.py --checkpoint path/to/best.pt --input eeg.npy --model-name eegnet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XAI demo (saliency / integrated gradients)")
    p.add_argument("--checkpoint", required=True, help="Path to a model checkpoint (.pt)")
    p.add_argument("--input", required=True, help="Path to .npy or .npz (key 'eeg') with shape (C,T) or (B,C,T)")
    p.add_argument("--model-name", required=True, help="Model registry name (e.g. eegnet, tsception, simplevit)")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="Output .npy saliency path")
    return p.parse_args()


def _load_eeg(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        x = np.load(path)
    elif path.suffix.lower() == ".npz":
        d = np.load(path)
        if "eeg" not in d:
            raise KeyError(".npz must contain key 'eeg'")
        x = d["eeg"]
    else:
        raise ValueError("Input must be .npy or .npz")

    if x.ndim == 3:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"Expected (C,T) or (B,C,T), got {x.shape}")
    return x.astype(np.float32)


def _try_integrated_gradients(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor | None:
    try:
        from captum.attr import IntegratedGradients  # type: ignore
    except Exception:
        return None

    model.eval()
    ig = IntegratedGradients(model)
    # Target: class 1 by default (seizure). Users can modify.
    attributions = ig.attribute(x, target=1)
    return attributions


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    # Register models
    import seizure_pred.models as models
    models.register_all()

    # Build model skeleton
    from seizure_pred.core.config import ModelConfig
    from seizure_pred.training.registries import MODELS

    cfg = ModelConfig(name=args.model_name, num_classes=args.num_classes)
    model = MODELS.create(args.model_name, cfg)
    model.to(device)

    # Load checkpoint (supports common patterns)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)

    eeg = _load_eeg(Path(args.input))
    x = torch.from_numpy(eeg).unsqueeze(0).to(device)  # (B,C,T)
    x.requires_grad_(True)

    # Try captum IG, else fallback to plain gradients
    sal = _try_integrated_gradients(model, x)
    if sal is None:
        model.eval()
        out = model(x)
        # assume logits (B, num_classes)
        score = out[:, 1].sum()
        score.backward()
        sal = x.grad.detach().clone()
        print("[xai] captum not installed -> used plain gradients. Install with: pip install captum")
    else:
        print("[xai] used Integrated Gradients (captum)")

    sal_np = sal.squeeze(0).detach().cpu().numpy()  # (C,T)
    out_path = Path(args.out) if args.out else Path(args.input).with_suffix(".saliency.npy")
    np.save(out_path, sal_np)
    print(f"[xai] wrote {out_path} (shape={sal_np.shape})")


if __name__ == "__main__":
    main()
