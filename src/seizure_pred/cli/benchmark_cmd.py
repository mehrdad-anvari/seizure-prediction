from __future__ import annotations

import argparse


def add_benchmark_cmd(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("benchmark", help="Benchmark registered models (params, FLOPs, latency, memory)")
    p.add_argument("--models", nargs="+", required=True, help="Registered model names to benchmark")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 16, 32], help="Batch sizes to test")
    p.add_argument("--n-runs", type=int, default=100, help="Inference timing runs")
    p.add_argument("--output-dir", default="benchmark_results", help="Output directory for the CSV")
    p.add_argument("--channels", type=int, default=18, help="Number of input channels")
    p.add_argument("--time-points", type=int, default=640, help="Number of time samples per window")
    p.set_defaults(func=run_benchmark)


def run_benchmark(args: argparse.Namespace) -> None:
    from seizure_pred.experiments.benchmark import benchmark_all_models

    benchmark_all_models(
        models=args.models,
        batch_sizes=args.batch_sizes,
        n_runs=args.n_runs,
        output_dir=args.output_dir,
        channels=args.channels,
        time_points=args.time_points,
    )
