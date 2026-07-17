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
| `mb_dmgc_cwtffnet` | multi-branch graph + CWT | flagship; uses `ch_locs.npy` adjacency |

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
