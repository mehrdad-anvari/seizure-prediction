from __future__ import annotations

import argparse
from pathlib import Path

from seizure_pred.cli.train_cmd import train_from_config


def main(argv: list[str] | None = None) -> None:
    """Back-compat entrypoint for old train.py scripts."""
    p = argparse.ArgumentParser(description="Legacy wrapper: delegates to seizure-pred train")
    p.add_argument("--config", required=True)
    p.add_argument("--split-index", type=int, default=0)
    p.add_argument("--dataloader", default=None)
    p.add_argument("--mil", action="store_true")
    args = p.parse_args(argv)
    train_from_config(Path(args.config), split_index=args.split_index, dataloader=args.dataloader, mil=args.mil)


if __name__ == "__main__":
    main()
