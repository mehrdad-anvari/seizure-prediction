from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from seizure_pred.core.config import TrainConfig
from seizure_pred.core.io import from_dict, load_dict, merge_dict
from seizure_pred.inference import predict
from seizure_pred.training import registries  # ensure registrations
from seizure_pred.training.engine.pipeline import build_dataset, iter_splits, build_dataloader
from seizure_pred.training.engine.checkpoint import restore
from seizure_pred.training.registries import MODELS


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run inference (predict) on a split")
    p.add_argument("--config", type=str, required=True, help="Path to yaml/json config")
    p.add_argument("--override", type=str, default=None, help="Optional yaml/json override file")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (best.pt/last.pt)")
    p.add_argument("--split-index", type=int, default=0)
    p.add_argument("--dataloader", type=str, default="torch")
    p.add_argument("--mil", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    return p


def main(argv=None) -> None:
    args = build_argparser().parse_args(argv)

    cfg_dict = load_dict(args.config)
    if args.override:
        cfg_dict = merge_dict(cfg_dict, load_dict(args.override))
    cfg = from_dict(TrainConfig, cfg_dict)

    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")

    dataset = build_dataset(cfg)
    splits = list(iter_splits(dataset))
    train_set, test_set = splits[args.split_index]

    loader = build_dataloader(args.dataloader, test_set, cfg.data, shuffle=False)

    model = MODELS.create(cfg.model.name, cfg.model).to(device)
    restore(Path(args.checkpoint), model=model, optimizer=None, scheduler=None)

    out = predict(model, loader, device, is_mil=args.mil or args.dataloader == "mil", threshold=args.threshold)

    out_dir = Path(cfg.save_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "probs": out.probs.tolist(),
        "preds": out.preds.tolist(),
        "targets": None if out.targets is None else out.targets.tolist(),
        "meta_count": len(out.meta),
    }
    (out_dir / "predictions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved predictions to: {out_dir / 'predictions.json'}")
