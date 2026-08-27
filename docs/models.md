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
| `fapex` | FAPEX, NeurIPS 2025 | fractional neural frame operator → amplitude/phase cross-encoding over bidirectional SSMs → linear attention across electrodes; `patch_size`, `d_model`, `d_state`, `depth`, `frnfo_per_layer`. Defaults are FAPEX-Small (4 layers, `d_model=128`, 2 FrNFOs per layer); FAPEX-Base is `depth=6, d_model=256`. Channel-count agnostic |
| `md_rescapsnet` | MD-ResCapsNet, BSPC 2026 | CSP projection → STFT image → SE-SA ResNet → capsule routing; needs `sfreq`, optional `csp_filters` from `compute_csp_filters`; pair with the `capsule_margin` loss |
| `seizurenet_kan` | SeizureNet-KAN, JESTCH 2026 | PLV graph over channels + KAN-enhanced GCN (B-splines, degree 3, grid 20); torch-only, no `torch_geometric` needed |
| `seresnet3d` | 3D-SERESNet, iScience 2025 | channel-stacked STFT volume → three 3D SE residual modules; pair with the `focal` loss |
| `sbtm` | SBTM, Sci. Reports 2026 | spectral + Hjorth + statistical features per sub-window → Bi-LSTM → dense head. The paper's Spizella metaheuristic optimiser is **not** implemented (see docstring) |

The STFT/graph/feature front-ends run inside these models, so they all consume
ordinary `(B, C, T)` windows like the rest of the zoo. `md_rescapsnet`, `seresnet3d`
and `sbtm` read `sfreq`, so set it on the config.

## Example configs

`configs/models/` holds one runnable config per recently added model. All seven
take `configs/studies/study003.yaml` as their base — same CHB-MIT subject, same
nested CV block, same `monitor: auc` — and change only what the model or its
paper requires:

```text
conda run -n torch-gpu seizure-pred train --config configs/models/seresnet3d.yaml --strict
```

| Config | Offline transforms | Loss | Optimizer | LR | Batch | Epochs | Scheduler |
|--------|--------------------|------|-----------|---:|------:|-------:|-----------|
| `darnet.yaml` | `robust_norm` | `bce_logits` | `adamw` | 1e-4 | 32 | 50 | – |
| `mhanet.yaml` | `robust_norm` | `bce_logits` | `adamw` | 1e-4 | 32 | 50 | – |
| `fapex.yaml` | `robust_norm` | `bce_logits` | `adamw` | 1e-4 | 32 | 50 | – |
| `md_rescapsnet.yaml` | `robust_norm` | `capsule_margin` | `adam` | 2e-3 | 16 | 100 | `exponential` |
| `seizurenet_kan.yaml` | `robust_norm` | `bce_logits` | `adam` | 1e-4 | 64 | 200 | – |
| `seresnet3d.yaml` | *(none)* | `focal` | `adamw` | 3e-3 | 32 | 100 | `cosine_warm_restarts` |
| `sbtm.yaml` | `robust_norm` | `bce_logits` | `adam` | 1e-3 | 32 | 50 | – |

Three conventions are worth knowing before copying one of these:

- **No `wavelet_filterbank`.** The studies stack it after normalization, but it
  combines its Db4 sub-bands with `combine_mode="concat_time"`: the `(18, 640)`
  shape survives while the time axis becomes a concatenation of dyadic
  sub-bands. Every model here reads real time — internal STFT (`md_rescapsnet`,
  `seresnet3d`), analytic phase (`seizurenet_kan`), fractional Fourier
  (`fapex`), Hjorth parameters (`sbtm`), temporal attention (`darnet`,
  `mhanet`) — so the filterbank is dropped and `robust_norm` (the winner of
  STUDY-002) is carried forward alone. `seresnet3d` gets an empty list because
  its Eq. 3 per-channel z-score runs inside the model.
- **`in_channels` and `sfreq` are set explicitly.** Neither is inferred from the
  data anywhere in the pipeline; the study configs omit them only because
  `eegwavenet` ignores both. `fapex` and `seizurenet_kan` genuinely need neither
  and leave them unset.
- **Training recipes follow each paper where the paper states one**, and every
  value the paper leaves unstated is called out in a comment in the file itself.
  `darnet` and `mhanet` keep the study003 recipe: both papers target auditory
  attention decoding, not seizure prediction. `fapex.yaml` follows Appendix H of
  FAPEX's supplementary (`papers/supplementaries/`), which gives the optimizer,
  epoch budget and early-stopping patience outright and gives the learning rate,
  weight decay, dropout and batch size only as search grids — the file picks one
  value from each grid and says so. Two things Appendix H specifies are *not*
  implemented: an EMA of the weights (decay 0.995) and its per-forward-pass
  augmentation suite.

`amp` is off in the five configs whose front-end is an FFT or STFT; fp16
autocast over complex tensors is the first thing to break there, and it buys
little next to the 3D convolutions and routing iterations.

`fapex.yaml` is the memory-hungry one: at Appendix H's sizes its state-space scan
keeps `(batch × channels, patches, d_inner, d_state)` tensors for the backward
pass, which runs to roughly 10 GB at `batch_size: 32`. The file explains the
arithmetic and names `batch_size: 16` as the fallback that stays inside the
paper's grid.

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
