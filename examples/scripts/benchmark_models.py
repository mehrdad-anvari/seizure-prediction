"""Benchmark multiple models using the new training pipeline.

This is a rewritten version of the old `benchmark_models.py`, adapted to the
new registry-based training API.

It runs short trainings for a list of model names against a single config
template, overriding `model.name` for each run.

Example:
  python examples/scripts/benchmark_models.py \
    --config configs/train.yaml \
    --models eegnet tsception simplevit \
    --epochs 5

Notes:
  - For full control (grids, sweeps), prefer the CLI + overrides.
  - This script is intentionally minimal and side-effect-free.
"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from seizure_pred.core.config import TrainConfig
from seizure_pred.core.io import load_dict, from_dict, save_config
from seizure_pred.cli.train_cmd import run_train


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark models (new seizure_pred pipeline)")
    p.add_argument("--config", required=True, help="Base YAML/JSON config")
    p.add_argument("--models", nargs="+", required=True, help="List of model registry names")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs")
    p.add_argument("--split-index", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--mil", action="store_true", help="Use MIL trainer")
    p.add_argument("--strict", action="store_true", help="Fail fast if components are missing")
    return p.parse_args()


def _make_override(base: Dict[str, Any], model_name: str, epochs: int | None) -> Dict[str, Any]:
    o = {"model": {"name": model_name}}
    if epochs is not None:
        o["epochs"] = int(epochs)
    return o


def _run_one(base_cfg_path: str, override: Dict[str, Any], *, split_index: int, n_folds: int, mil: bool, strict: bool) -> None:
    # Write merged config to a temp file and call the same CLI implementation used by `seizure-pred train`.
    base = load_dict(base_cfg_path)
    merged = copy.deepcopy(base)
    # shallow merge is enough here (only overriding model.name + epochs). For deep merges use CLI overrides.
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k].update(v)
        else:
            merged[k] = v

    # Validate by building TrainConfig (raises readable error if invalid)
    _ = from_dict(TrainConfig, merged)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "bench_config.yaml"
        save_config(from_dict(TrainConfig, merged), tmp)

        ns = argparse.Namespace(
            config=str(tmp),
            override=None,
            split_index=split_index,
            n_folds=n_folds,
            dataloader=None,
            mil=mil,
            strict=strict,
            print_config=False,
        )
        run_train(ns)


def main() -> None:
    args = _parse_args()

    # Print a quick plan
    print(json.dumps({
        "base_config": args.config,
        "models": args.models,
        "epochs_override": args.epochs,
        "split_index": args.split_index,
        "n_folds": args.n_folds,
        "mil": args.mil,
        "strict": args.strict,
    }, indent=2))

    for name in args.models:
        print(f"\n[benchmark] model={name}")
        override = _make_override(load_dict(args.config), name, args.epochs)
        _run_one(
            args.config,
            override,
            split_index=args.split_index,
            n_folds=args.n_folds,
            mil=args.mil,
            strict=args.strict,
        )


if __name__ == "__main__":
    main()
