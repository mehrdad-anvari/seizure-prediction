from __future__ import annotations

"""Feature-extraction utilities adapted to the refactored library.

This is intentionally under `examples/` (not import-time library code).
"""

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from seizure_pred.transforms.feature.basic_stats import (
    MeanAmplitude,
    StandardDeviation,
    Skewness,
    Kurtosis,
    RootMeanSquare,
    LineLength,
    ZeroCrossingRate,
    HjorthActivity,
    HjorthMobility,
    HjorthComplexity,
)


@dataclass
class FeatureSpec:
    name: str
    transform: object
    # If transform returns multiple columns (e.g., band-wise), we expand them.
    subnames: List[str] | None = None


def _try_make(transform_ctor, name: str, *, strict: bool = False):
    try:
        return transform_ctor()
    except ImportError as e:
        if strict:
            raise
        print(f"[features] skipping {name}: {e}")
        return None


def build_feature_specs(
    sfreq: int = 128,
    include_mne_connectivity: bool = True,
    include_scipy_features: bool = True,
    strict: bool = False,
) -> Tuple[List[FeatureSpec], List[str]]:
    """Return feature specs + flattened feature names."""

    specs: List[FeatureSpec] = []

    # Basic time-domain features (numpy-only)
    specs.extend(
        [
            FeatureSpec("MeanAmplitude", MeanAmplitude()),
            FeatureSpec("StandardDeviation", StandardDeviation()),
            FeatureSpec("Skewness", Skewness()),
            FeatureSpec("Kurtosis", Kurtosis()),
            FeatureSpec("RootMeanSquare", RootMeanSquare()),
            FeatureSpec("LineLength", LineLength()),
            FeatureSpec("ZeroCrossingRate", ZeroCrossingRate()),
            FeatureSpec("HjorthActivity", HjorthActivity()),
            FeatureSpec("HjorthMobility", HjorthMobility()),
            FeatureSpec("HjorthComplexity", HjorthComplexity()),
        ]
    )

    # SciPy-based spectral features
    if include_scipy_features:
        from seizure_pred.transforms.feature.band_power import (
            DeltaPower,
            ThetaPower,
            AlphaPower,
            BetaPower,
            GammaPower,
        )
        from seizure_pred.transforms.feature.spectral_summary import (
            SpectralEntropy,
            IntensityWeightedMeanFrequency,
            SpectralEdgeFrequency,
            PeakFrequency,
        )
        from seizure_pred.transforms.feature.differential_entropy import BandDifferentialEntropy

        # Band power
        for cls in [DeltaPower, ThetaPower, AlphaPower, BetaPower, GammaPower]:
            t = _try_make(lambda c=cls: c(sfreq), cls.__name__, strict=strict)
            if t is not None:
                specs.append(FeatureSpec(cls.__name__, t))

        # Spectral summary
        for cls in [SpectralEntropy, IntensityWeightedMeanFrequency, SpectralEdgeFrequency, PeakFrequency]:
            t = _try_make(lambda c=cls: c(sfreq), cls.__name__, strict=strict)
            if t is not None:
                specs.append(FeatureSpec(cls.__name__, t))

        # Band differential entropy (multi-output)
        bde = _try_make(lambda: BandDifferentialEntropy(sampling_rate=sfreq), "BandDifferentialEntropy", strict=strict)
        if bde is not None:
            # keep names stable
            band_names = list(getattr(bde, "band_dict", {}).keys())
            subnames = [f"BDE_{b}" for b in band_names] if band_names else None
            specs.append(FeatureSpec("BandDifferentialEntropy", bde, subnames=subnames))

    # Connectivity features (optional: mne + mne-connectivity)
    if include_mne_connectivity:
        from seizure_pred.transforms.feature.connectivity import (
            MeanAbsCorrelation,
            MeanCoh,
            MeanPLV,
            MeanImCoh,
            MeanPLI,
            MeanWPLI,
        )

        # Always-available correlation
        specs.append(FeatureSpec("MeanAbsCorrelation", MeanAbsCorrelation()))

        # MNE-based measures may fail at construction if deps missing
        for cls in [MeanCoh, MeanPLV, MeanImCoh, MeanPLI, MeanWPLI]:
            t = _try_make(lambda c=cls: c(sfreq=sfreq), cls.__name__, strict=strict)
            if t is not None:
                specs.append(FeatureSpec(cls.__name__, t))

    # Flatten names
    feature_names: List[str] = []
    for s in specs:
        if s.subnames:
            feature_names.extend(s.subnames)
        else:
            feature_names.append(s.name)

    return specs, feature_names


def extract_features_for_channel(
    X: np.ndarray,
    channel: int,
    specs: Iterable[FeatureSpec],
) -> np.ndarray:
    """Extract features for a fixed channel for each segment.

    X: (n_segments, n_channels, n_times)
    returns: (n_segments, n_features)
    """

    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"Expected X as (segments, channels, time); got shape={X.shape}")

    n_segments = X.shape[0]

    rows: List[np.ndarray] = []
    for i in range(n_segments):
        eeg = X[i]
        feats: List[float] = []
        for spec in specs:
            out = spec.transform.apply(eeg=eeg) if hasattr(spec.transform, "apply") else spec.transform(eeg)
            arr = np.asarray(out)
            # common shapes: (n_channels, 1) or (n_channels,) or (n_channels, n_bands)
            if arr.ndim == 1:
                feats.append(float(arr[channel]))
            elif arr.ndim == 2:
                if arr.shape[1] == 1:
                    feats.append(float(arr[channel, 0]))
                else:
                    feats.extend([float(v) for v in arr[channel, :].tolist()])
            else:
                raise ValueError(f"Unexpected output shape from {spec.name}: {arr.shape}")
        rows.append(np.array(feats, dtype=float))

    return np.stack(rows, axis=0)
