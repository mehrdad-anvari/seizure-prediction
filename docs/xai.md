# Explainability (XAI)

`seizure_pred.inference.xai` provides input attribution for EEG models using
Captum's **IntegratedGradients**. Captum is an *optional* dependency — the
module imports fine without it; the heavy import only happens when an
attribution function is called.

Install with `pip install captum`.

## API

```python
from seizure_pred.inference.xai import integrated_gradients, channel_attributions, plot_topomap

# attr: same shape as inputs (B, C, T)
attr = integrated_gradients(model, x, target=1, n_steps=50, device="cpu")

# mean absolute attribution per channel: (B, C)
ch = channel_attributions(model, x, target=1, n_steps=50)

# 2-D topomap (requires mne + matplotlib)
plot_topomap(ch[0], ch_names, save_path="topo.png", title="Channel attributions")
```

`integrated_gradients` handles models returning `(B,)`, `(B,1)`, or `(B,2)`
logits; `target` selects the output class to attribute (1 = positive/preictal).
A zero baseline is used by default.

## Notes

- Attribution quality depends on the model and a sensible baseline; for
  wavelet/waveform models you may prefer random interictal baselines (as the
  legacy `xai.py` did).
- `channel_attributions` aggregates `|attr|` over the time dimension, giving a
  per-channel importance ranking — useful for explaining *which* electrodes
  drive a preictal alarm.
- `plot_topomap` builds a `standard_1020` montage; supply bipolar channel names
  accordingly.
