from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, Iterable, Optional

import torch

from seizure_pred.core.config import TrainConfig
from seizure_pred.core.io import load_dict, merge_dict, from_dict
from seizure_pred.core.seed import seed_everything
from seizure_pred.core.validate import validate_config_dict
from seizure_pred.training.engine.pipeline import build_dataset, iter_splits, build_loader
from seizure_pred.training.engine.artifacts import ArtifactWriter
from seizure_pred.training.engine.checkpoint import restore_checkpoint
from seizure_pred.training.registries import MODELS, POSTPROCESSORS, DATALOADERS
from seizure_pred.inference.predictor import predict  # generator of rows


def add_predict_cmd(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("predict", help="Run inference on a split and write predictions artifact")
    p.add_argument("--config", required=True, help="YAML/JSON config file")
    p.add_argument("--override", default=None, help="Optional YAML/JSON override file merged on top")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt)")
    p.add_argument("--split-index", type=int, default=0, help="Which split fold to run")
    p.add_argument("--n-folds", type=int, default=5, help="Number of folds for splits")
    p.add_argument("--dataloader", default=None, help="Override dataloader strategy name")
    p.add_argument("--mil", action="store_true", help="Treat batches as MIL bags")
    p.add_argument("--strict", action="store_true", help="Fail fast if requested components are missing")
    p.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for label")
    p.add_argument("--out-dir", default=None, help="Where to write predictions artifacts")
    p.add_argument("--apply-postprocess", action="store_true", help="Apply cfg.postprocess to labels")

    p.set_defaults(func=run_predict)


def _build_postprocessor(cfg: TrainConfig, strict: bool) -> Optional[object]:
    pp_cfg = getattr(cfg, "postprocess", None)
    if not pp_cfg or not getattr(pp_cfg, "name", None):
        return None
    name = pp_cfg.name
    kwargs: Dict[str, Any] = dict(getattr(pp_cfg, "kwargs", {}) or {})
    if strict and name not in POSTPROCESSORS:
        raise SystemExit(f"Unknown postprocessor '{name}'. Use `seizure-pred list`.")
    return POSTPROCESSORS.create(name, **kwargs)


def run_predict(args: argparse.Namespace) -> None:
    raw = load_dict(args.config)
    if args.override:
        raw = merge_dict(raw, load_dict(args.override))

    validate_config_dict(raw, TrainConfig)
    cfg: TrainConfig = from_dict(TrainConfig, raw)

    # Register built-in plugins only when prediction is invoked.
    import seizure_pred.training as training
    training.register_all()
    import seizure_pred.models as models
    models.register_all()
    
    seed_everything(getattr(cfg, "determinism", None), seed=cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")

    # Dataset + split
    ds = build_dataset(cfg)
    splits = list(iter_splits(ds, n_folds=args.n_folds))
    if args.split_index < 0 or args.split_index >= len(splits):
        raise SystemExit(f"--split-index {args.split_index} out of range (0..{len(splits)-1})")

    _, val_set = splits[args.split_index]

    dl_name = args.dataloader or getattr(cfg.data, "dataloader", None) or "torch"
    if args.strict and dl_name not in DATALOADERS:
        raise SystemExit(f"Unknown dataloader '{dl_name}'. Use `seizure-pred list`.")
    loader = build_loader(dl_name, val_set, cfg, shuffle=False)

    # Model
    if args.strict and cfg.model.name not in MODELS:
        raise SystemExit(f"Unknown model '{cfg.model.name}'. Use `seizure-pred list`.")
    model = MODELS.create(cfg.model.name, cfg.model)
    model.to(device)

    # Restore weights
    restore_checkpoint(args.checkpoint, model=model)

    # Optional postprocess (applied to predicted labels)
    postproc = _build_postprocessor(cfg, args.strict) if args.apply_postprocess else None

    # Output directory
    out_dir = args.out_dir
    if out_dir is None:
        # default: sibling folder next to checkpoint
        out_dir = os.path.join(os.path.dirname(args.checkpoint), f"predict_split_{args.split_index}")
    os.makedirs(out_dir, exist_ok=True)

    writer = ArtifactWriter(out_dir)
    writer.write_schema()
    writer.write_config(asdict(cfg))

    # Generate rows
    rows = predict(
        model=model,
        loader=loader,
        device=str(device),
        is_mil=args.mil,
        threshold=args.threshold,
        postprocess=postproc,
    )

    # Write standardized predictions artifact
    writer.write_predictions(rows)

    print(json.dumps(
        {
            "out_dir": out_dir,
            "checkpoint": args.checkpoint,
            "split_index": args.split_index,
            "threshold": args.threshold,
            "postprocess": getattr(getattr(cfg, "postprocess", None), "name", None) if args.apply_postprocess else None,
        },
        indent=2
    ))
