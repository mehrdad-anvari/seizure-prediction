# Benchmarking

`seizure_pred.experiments.benchmark` measures, per model and batch size:

- total / trainable parameter count
- FLOPs and MACs (requires `thop`; `clever_format`-ed)
- CPU and (if available) GPU inference latency (mean/std/min/max/median)
- throughput (samples/sec) on CPU/GPU
- CPU→GPU speedup
- GPU peak memory (`torch.cuda.max_memory_allocated`)
- optional `torchinfo` layer summary

Models are built through the `MODELS` registry, so any registered name works.

## CLI

```bash
seizure-pred benchmark \
  --models eegnet eegwavenet mb_dmgc_cwtffnet \
  --batch-sizes 1 32 \
  --n-runs 100 \
  --output-dir benchmark_results \
  --channels 18 --time-points 640
```

Results are saved to `benchmark_results/benchmark_results_<timestamp>.csv`.

## Programmatic

```python
from seizure_pred.experiments.benchmark import benchmark_all_models, ModelBenchmark
import seizure_pred.models as models

models.register_all()

# Bench several models
df = benchmark_all_models(["eegnet", "eegwavenet"], batch_sizes=[1, 32],
                          n_runs=50, output_dir="benchmark_results")

# Single model, custom input
bench = ModelBenchmark("eegnet", input_shape=(1, 18, 640))
res = bench.benchmark(n_runs=100, batch_size=32)
```

## Optional dependencies

- `thop` — FLOPs/MACs (skipped gracefully if absent).
- `torchinfo` — layer summary (skipped gracefully if absent).
- A CUDA device — GPU metrics are only reported when available.
