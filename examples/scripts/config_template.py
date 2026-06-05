"""Generate a starter training config for the new library.

This replaces the old `config_template.py` and generates a YAML/JSON file that
matches `seizure_pred.core.config.TrainConfig`.

Example:
  python examples/scripts/config_template.py --out train.yaml

Then train:
  python -m seizure_pred.cli.main train --config train.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from seizure_pred.core.config import TrainConfig
from seizure_pred.core.io import save_config


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a starter seizure_pred training config")
    p.add_argument("--out", required=True, help="Output path (.yaml/.yml/.json)")
    p.add_argument("--task", choices=["prediction", "detection"], default="prediction")
    p.add_argument("--dataset-dir", default=None, help="Override data.dataset_dir")
    p.add_argument("--subject", "--subject-id", dest="subject", default=None, help="Override data.subject_id")
    p.add_argument("--model", default=None, help="Override model.name")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = TrainConfig(task=args.task)
    cfg.data.task = args.task

    if args.dataset_dir is not None:
        cfg.data.dataset_dir = args.dataset_dir
    if args.subject is not None:
        cfg.data.subject_id = args.subject
    if args.model is not None:
        cfg.model.name = args.model

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out)
    print(f"[config] wrote {out}")


if __name__ == "__main__":
    main()
