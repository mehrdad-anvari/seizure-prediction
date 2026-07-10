# Transforms

Transforms live under `src/seizure_pred/transforms/`. They are plain Python
classes (a `SignalTransform` maps signal→signal; a `FeatureTransform` maps
signal→feature). Instantiate directly or via the small factory:

```python
from seizure_pred.transforms.registry import create_transform, list_transforms

print(list_transforms())                      # all available specs
norm = create_transform("instance_norm")      # InstanceNormTransform()
```

Some transforms require optional extras (`.[signal]` → scipy, `.[eeg]` →
mne/mne-connectivity); the factory raises a clear `ImportError` when the
needed dependency is missing.

## Signal transforms (`transforms.signal`)

| Name | Class | Description |
|------|-------|-------------|
| `instance_norm` | `InstanceNormTransform` | per-channel z-score across time |
| `to_grid` | `ToGrid` | project channels → (9,9) spatial grid for STNet/SimpleViT |
| `filterbank` | `FilterBank` | butter bandpass band stacking (δ/θ/α/β/γ) for band models |
| `wavelet_filterbank` | `WaveletFilterBank` | Db4 DWT component stacking for EEGWaveNet |

## Feature transforms (`transforms.feature`)

Over 35 features, all returning `(n_channels, 1)` or scalar summaries:

- **Basic stats**: `MeanAmplitude`, `StandardDeviation`, `Skewness`,
  `Kurtosis`, `RootMeanSquare`, `LineLength`, `ZeroCrossingRate`,
  `HjorthActivity`, `HjorthMobility`, `HjorthComplexity`.
- **Band power**: `DeltaPower`, `ThetaPower`, `AlphaPower`, `BetaPower`,
  `GammaPower` (Welch PSD).
- **Spectral summaries**: `SpectralEntropy`, `IntensityWeightedMeanFrequency`,
  `SpectralEdgeFrequency` (95th pct), `PeakFrequency`.
- **Connectivity** (`.[eeg]`): `MeanCoh`, `MeanPLV`, `MeanImCoh`, `MeanPLI`,
  `MeanWPLI`; plus `MeanAbsCorrelation` (numpy only).
- **Differential entropy**: `BandDifferentialEntropy`.

## Using transforms in configs

The `chbmit_npz` dataset builder resolves `data.kwargs.online_transforms` /
`offline_transforms` when they are lists of **names**:

```yaml
data:
  name: chbmit_npz
  kwargs:
    offline_transforms: ["filterbank"]   # applied once at load
    online_transforms: ["instance_norm"] # applied per __getitem__
```

Model-specific offline transforms (e.g. `filterbank` for `fbmsnet`,
`wavelet_filterbank` for `eegwavenet`) should be wired this way; the original
repo wired them automatically in the training script, the library makes them
explicit and pluggable.
