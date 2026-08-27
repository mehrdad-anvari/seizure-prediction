# Model zoo

All models register through `MODELS` and are built from a `ModelConfig`
(`name`, `num_classes`, `in_channels`, `sfreq`, `kwargs`). Register plugins with
`import seizure_pred.models; seizure_pred.models.register_all()`.

Use `seizure-pred list` to see exactly what is available in your environment
(some models are skipped if optional deps are missing).

## Built-in models

| Name | Type | Notes |
|------|------|-------|
| `simple_cnn` | 1D CNN | baseline; `in_channels`, `hidden` |
| `eegnet` | CNN | EEGNet; `num_electrodes`, `chunk_size`, `F1`, `F2`, `D`, `kernel_1/2`, `dropout` |
| `eegwavenet` / `eegwavenet_tiny` | WaveNet | wavelet-component branches; `model_size` |
| `tsception` | T/S CNN | `sampling_rate`, `num_T`, `num_S`, `hidden`, `dropout_rate` |
| `fbmsnet` | filter-bank CNN | expects offline `filterbank` transform |
| `lmda` | depth-attention CNN | `chans`, `samples`, `depth`, `kernel` |
| `tslanet` | spectral + inception | `patch_size`, `emb_dim`, `depth` |
| `cspnet` | CSP CNN | `num_filters_t/s`, `filter_size_t/s` |
| `stnet` | grid CNN | requires `to_grid` transform |
| `conformer` | Conformer | `encoder_dim`, `num_encoder_layers` |
| `simplevit` / `simple_vit` | ViT | requires `to_grid`; patch/grid dims |
| `eeg_band_classifier` / `eegbandclassifier` | band classifier | expects offline `filterbank` |
| `ce_stsenet` | multi-band ST-SENet | loads Db4 `scaling_filter.mat` (needs scipy) |
| `darnet` | dual attention refinement | temporal self-attention + conv refinement; `chunk_size`, `d_model`, `num_heads`, `attn_dropout` |
| `mhanet` | hybrid attention | channel + multi-scale global attention (needs einops); `chunk_size` must equal `T`, `num_heads` must divide the channel count |
| `mb_dmgc_cwtffnet` | multi-branch graph + CWT | flagship; uses `ch_locs.npy` adjacency |

## Models reconstructed from papers

These come from papers in `papers/` that publish **no reference code**, so each is
a reconstruction from the paper text: block structure and stated dimensions are
followed, and every inferred choice is listed in the module docstring. Treat them
as faithful-in-structure, not numerically equivalent to the authors' models.

| Name | Paper | Notes |
|------|-------|-------|
| `fapex` | FAPEX, NeurIPS 2025 | fractional neural frame operator → amplitude/phase cross-encoding over bidirectional SSMs → linear attention across electrodes; `patch_size`, `d_model`, `d_state`, `depth`. Channel-count agnostic |
| `md_rescapsnet` | MD-ResCapsNet, BSPC 2026 | CSP projection → STFT image → SE-SA ResNet → capsule routing; needs `sfreq`, optional `csp_filters` from `compute_csp_filters`; pair with the `capsule_margin` loss |
| `seizurenet_kan` | SeizureNet-KAN, JESTCH 2026 | PLV graph over channels + KAN-enhanced GCN (B-splines, degree 3, grid 20); torch-only, no `torch_geometric` needed |
| `seresnet3d` | 3D-SERESNet, iScience 2025 | channel-stacked STFT volume → three 3D SE residual modules; pair with the `focal` loss |
| `sbtm` | SBTM, Sci. Reports 2026 | spectral + Hjorth + statistical features per sub-window → Bi-LSTM → dense head. The paper's Spizella metaheuristic optimiser is **not** implemented (see docstring) |

The STFT/graph/feature front-ends run inside these models, so they all consume
ordinary `(B, C, T)` windows like the rest of the zoo. `md_rescapsnet`, `seresnet3d`
and `sbtm` read `sfreq`, so set it on the config.

## Graph / GNN models (optional `.[gnn]`)

| Name | Notes |
|------|-------|
| `rgnn` | RGNN; expects 5-dim DE features, learnable edge weights, optional domain adaptation |
| `dgcnn2` | DCRNN-style; learnable edge weights |
| `eeg_gnn_ssl` | DCRNN graph RNN classifier; builds IMT adjacency from electrode distances |
| `dgcnn` | Chebyshev spectral GCN (torch-only) |

## Adding a model

```python
from seizure_pred.core.config import ModelConfig
from seizure_pred.training.registries import MODELS

@MODELS.register("my_model", help="My model")
def build_my_model(cfg: ModelConfig):
    kw = dict(cfg.kwargs or {})
    return MyModel(in_channels=cfg.in_channels, num_classes=cfg.num_classes, **kw)
```

Graph models should guard `torch_geometric` import via
`seizure_pred.core.optional_deps.is_torch_geometric_available()`.

## Legacy provider

`seizure_pred.models.provider.get_builder(name)` is retained for back-compat
with older scripts that obtained a fresh-model factory per fold. Prefer the
registry API (`MODELS.create(name, cfg)`) for new code.
