from __future__ import annotations

import argparse
import json


def add_list_cmd(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("list", help="List registered datasets/models/losses/optimizers/etc")
    p.set_defaults(func=run_list)


def run_list(_: argparse.Namespace) -> None:
    import seizure_pred.training as training
    training.register_all()
    import seizure_pred.models as models
    models.register_all()

    # Register models (model files register into training registries)
    
    from seizure_pred.training.registries import list_all

    print(json.dumps(list_all(), indent=2))
