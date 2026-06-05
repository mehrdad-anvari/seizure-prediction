"""Transform helpers.

The old repo exposed transforms as plain Python classes. In the new repo we keep
them under `seizure_pred.transforms` and provide a small, dependency-safe
factory for example scripts and user code.

This is intentionally *not* a heavy plugin system: it avoids import-time
side-effects and only imports transform modules when a transform is requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Dict, Optional, Tuple, Type


@dataclass(frozen=True)
class TransformSpec:
    """A lazy transform spec."""

    module: str
    attr: str
    extra_hint: Optional[str] = None


_SIGNAL: Dict[str, TransformSpec] = {
    # signal
    "instance_norm": TransformSpec("seizure_pred.transforms.signal.normalize", "InstanceNormTransform"),
    "to_grid": TransformSpec("seizure_pred.transforms.signal.rearrange", "ToGrid"),
    "filterbank": TransformSpec("seizure_pred.transforms.signal.filterbank", "FilterBank", extra_hint="signal"),
    "wavelet_filterbank": TransformSpec(
        "seizure_pred.transforms.signal.wavletfilterbank", "WaveletFilterBank", extra_hint="signal"
    ),
}

_FEATURE: Dict[str, TransformSpec] = {
    # feature
    "band_power": TransformSpec("seizure_pred.transforms.feature.band_power", "BandPowerTransform", extra_hint="signal"),
    "line_length": TransformSpec("seizure_pred.transforms.feature.basic_stats", "LineLength"),
    "mean_abs_corr": TransformSpec("seizure_pred.transforms.feature.connectivity", "MeanAbsCorrelation"),
    "mean_coh": TransformSpec("seizure_pred.transforms.feature.connectivity", "MeanCoh", extra_hint="eeg"),
    "mean_plv": TransformSpec("seizure_pred.transforms.feature.connectivity", "MeanPLV", extra_hint="eeg"),
    "mean_pli": TransformSpec("seizure_pred.transforms.feature.connectivity", "MeanPLI", extra_hint="eeg"),
    "wpli": TransformSpec("seizure_pred.transforms.feature.connectivity", "MeanWPLI", extra_hint="eeg"),
    "spectral_entropy": TransformSpec("seizure_pred.transforms.feature.spectral_summary", "SpectralEntropy", extra_hint="signal"),
    "diff_entropy": TransformSpec("seizure_pred.transforms.feature.differential_entropy", "BandDifferentialEntropy", extra_hint="signal"),
}


def list_transforms(kind: str | None = None) -> Dict[str, TransformSpec]:
    """Return available transforms.

    kind:
      - None: all
      - "signal": only signal transforms
      - "feature": only feature transforms
    """

    if kind is None:
        return {**_SIGNAL, **_FEATURE}
    if kind == "signal":
        return dict(_SIGNAL)
    if kind == "feature":
        return dict(_FEATURE)
    raise ValueError("kind must be one of: None, 'signal', 'feature'")


def _require_extra(extra_hint: str) -> None:
    if extra_hint == "signal":
        try:
            import scipy  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "This transform requires SciPy. Install with: pip install seizure-pred[signal] (or pip install scipy)"
            ) from e
    if extra_hint == "eeg":
        # connectivity transforms rely on mne-connectivity
        try:
            import mne_connectivity  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "This transform requires mne-connectivity. Install with: pip install seizure-pred[eeg]"
            ) from e


def create_transform(name: str, **kwargs: Any) -> Any:
    """Instantiate a transform by name.

    This factory is used by example scripts and is safe with optional deps.
    """

    all_t = list_transforms(None)
    if name not in all_t:
        available = ", ".join(sorted(all_t.keys())) or "<none>"
        raise KeyError(f"Unknown transform '{name}'. Available: {available}")
    spec = all_t[name]
    if spec.extra_hint:
        _require_extra(spec.extra_hint)
    mod = import_module(spec.module)
    cls = getattr(mod, spec.attr)
    return cls(**kwargs)
