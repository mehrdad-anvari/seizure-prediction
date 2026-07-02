from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RunPaths:
    """Standard run directory layout.

    We create:
        <save_dir>/<run_name>/<YYYYmmdd_HHMMSS>/split_<k>/
    """
    run_dir: Path
    analysis_dir: Path
    checkpoints_dir: Path

    @staticmethod
    def create(save_dir: str, run_name: str, split_index: int, *, timestamp: Optional[str] = None) -> "RunPaths":
        save_root = Path(save_dir)
        if timestamp is None:
            # UTC to avoid timezone confusion between machines
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = save_root / run_name / timestamp / f"split_{split_index}"
        analysis_dir = run_dir / "analysis"
        checkpoints_dir = run_dir / "checkpoints"
        return RunPaths(run_dir=run_dir, analysis_dir=analysis_dir, checkpoints_dir=checkpoints_dir)

    def ensure(self) -> "RunPaths":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        return self


def find_splits(dir_path: str) -> list[tuple[int, str]]:
    """Finds subdirectories matching split_X under dir_path.

    Returns:
        List of tuples (split_index, split_dir_path) sorted by split_index.
    """
    import os
    import re
    splits = []
    if not os.path.isdir(dir_path):
        return []
    for name in os.listdir(dir_path):
        sub_path = os.path.join(dir_path, name)
        if os.path.isdir(sub_path):
            m = re.match(r"^split_(\d+)$", name)
            if m:
                splits.append((int(m.group(1)), sub_path))
    return sorted(splits, key=lambda x: x[0])
