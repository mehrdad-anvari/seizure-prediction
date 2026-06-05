from __future__ import annotations

import argparse

# NOTE:
# We intentionally do NOT import torch-heavy modules (training/models) at CLI import time.
# Each subcommand is responsible for registering plugins when it needs them.


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="seizure-pred", description="Seizure prediction/detection CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    from .list_cmd import add_list_cmd
    from .train_cmd import add_train_cmd
    from .predict_cmd import add_predict_cmd
    from .preprocess_cmd import add_preprocess_cmd
    from .analyze_cmd import add_analyze_cmd
    from .experiments_cmd import add_experiments_cmd

    add_list_cmd(sub)
    add_train_cmd(sub)
    add_predict_cmd(sub)
    add_preprocess_cmd(sub)
    add_analyze_cmd(sub)
    add_experiments_cmd(sub)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        raise SystemExit(2)

    args.func(args)
