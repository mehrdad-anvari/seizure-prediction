from __future__ import annotations

import argparse
import json
from typing import Any, Dict



def add_experiments_cmd(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("experiments", help="Run a simple grid of experiments from a base config")
    p.add_argument("--config", required=True, help="Path to base YAML/JSON config")
    p.add_argument(
        "--grid",
        required=True,
        help='JSON dict of dot-path -> list. Example: {"optim.lr":[1e-3,3e-4]}',
    )
    p.add_argument("--split-index", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--dataloader", default="torch")
    p.add_argument("--mil", action="store_true")
    p.add_argument("--save-root", default=None, help="Override config.save_dir")
    p.set_defaults(func=run_experiments)


def run_experiments(args: argparse.Namespace) -> None:
    grid: Dict[str, Any] = json.loads(args.grid)
    from seizure_pred.experiments.grid import run_grid
    run_dirs = run_grid(
        args.config,
        grid,
        split_index=args.split_index,
        n_folds=args.n_folds,
        dataloader=args.dataloader,
        mil=args.mil,
        save_root=args.save_root,
    )
    print(json.dumps({"runs": run_dirs}, indent=2))
