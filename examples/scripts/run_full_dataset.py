#!/usr/bin/env python
"""Run one configuration on every CHB-MIT subject (a "full-dataset run").

Companion script for the "Full-Dataset Runs" section of docs/studies.md
(see RUN-001).

For each subject it:

1. writes a fully resolved config to
   ``<save_dir>/<group>/sub-<ID>/config_used.yaml``,
2. runs ``seizure-pred train`` on that config, so the results of the subject
   land in ``<save_dir>/<group>/sub-<ID>/<timestamp>/``,
3. unless ``--no-analyze`` is given, runs ``seizure-pred analyze`` on the new
   run so that ``analysis/variant_summary.csv`` (the file the studies tables are
   built from) is produced.

A complete config is written per subject rather than passing ``--override``:
``--override`` deep-merges dictionaries and cannot remove keys that are already
present in ``sched.kwargs``, so a full config is the only way to guarantee that
the scheduler recorded next to the results is the one that actually ran.

Examples
--------
Each run has its own base config under ``configs/full_dataset/``, and the config's
``run_name`` becomes the run group::

    python examples/scripts/run_full_dataset.py --config configs/full_dataset/run001.yaml

Show the plan for two subjects without training anything::

    python examples/scripts/run_full_dataset.py \
        --config configs/full_dataset/run001.yaml --subjects 01,02 --dry-run
"""

from __future__ import annotations

import argparse
import copy
import re
import subprocess
import sys
from pathlib import Path

import yaml

STAMP_RE = re.compile(r"^\d{8}_\d{6}$")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train one configuration on every subject, one run per subject.",
    )
    p.add_argument("--config", required=True, help="Base YAML/JSON config to run (see configs/full_dataset/ for full-dataset run configs).")
    p.add_argument(
        "--group",
        default=None,
        help="Run group directory under save_dir (default: the config's run_name).",
    )
    p.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Subject IDs, e.g. `01 02` or `01,02`. Default: every subject with processed data.",
    )
    p.add_argument("--dataset-dir", default=None, help="Override data.dataset_dir.")
    p.add_argument("--suffix", default=None, help="Processed suffix (default: data.suffix from the config).")
    p.add_argument("--python", default=sys.executable, help="Interpreter used to launch the CLI.")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip subjects whose newest run already has raw_predictions.pkl.",
    )
    p.add_argument("--no-analyze", action="store_true", help="Skip `seizure-pred analyze` after training.")
    p.add_argument("--fail-fast", action="store_true", help="Stop at the first failing subject.")
    return p.parse_args(argv)


def norm_id(value) -> str:
    text = str(value).strip()
    return f"{int(text):02d}" if text.isdigit() else text


def norm_subjects(values):
    out = []
    for value in values or []:
        out.extend(norm_id(part) for part in str(value).replace(",", " ").split())
    return out


def resolve(path, repo_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def subject_has_data(dataset_dir: Path, subject: str, suffix: str) -> bool:
    sub_dir = dataset_dir / f"sub-{subject}"
    return sub_dir.is_dir() and any(sub_dir.glob(f"ses-*/eeg/*{suffix}_float.npz"))


def discover_subjects(dataset_dir: Path, suffix: str):
    subjects = []
    for sub_dir in sorted(dataset_dir.glob("sub-*")):
        subject = sub_dir.name[len("sub-"):]
        if subject_has_data(dataset_dir, subject, suffix):
            subjects.append(subject)
    return subjects


def build_subject_config(base: dict, subject: str, group: str) -> dict:
    """Return a copy of ``base`` retargeted at one subject and run group."""
    cfg = copy.deepcopy(base)
    cfg.setdefault("data", {})["subject_id"] = subject
    cfg["run_name"] = f"{group}/sub-{subject}"
    return cfg

def write_config(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)


def newest_run_dir(subject_dir: Path):
    if not subject_dir.is_dir():
        return None
    stamps = [d for d in subject_dir.iterdir() if d.is_dir() and STAMP_RE.match(d.name)]
    return max(stamps, key=lambda d: d.name) if stamps else None


def preflight(cli, repo_root: Path) -> None:
    """Fail fast if the CLI does not actually dispatch.

    ``python -m seizure_pred.cli.main`` silently does nothing when the module has
    no ``__main__`` guard, which would turn a whole sweep into an unnoticed
    no-op, so make the CLI answer once before touching any subject.
    """
    probe = subprocess.run(
        cli + ["--help"],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if probe.returncode != 0 or "usage:" not in (probe.stdout or ""):
        raise SystemExit(
            "The training CLI did not respond to `--help` and would have done nothing.\n"
            f"  command : {' '.join(cli)}\n"
            f"  exit    : {probe.returncode}\n"
            "Check that the package is installed for this interpreter "
            '(pip install -e ".[cli]").'
        )


def run_cli(cmd, log_path: Path, repo_root: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("  $ " + " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()
    return proc.returncode


def main(argv=None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]

    from seizure_pred.core.io import load_dict  # same loader the CLI uses

    base = load_dict(args.config)
    data_cfg = base.get("data") or {}
    dataset_dir = resolve(args.dataset_dir or data_cfg.get("dataset_dir", "data/BIDS_CHB-MIT"), repo_root)
    suffix = args.suffix or data_cfg.get("suffix", "")
    save_dir = base.get("save_dir", "runs")
    group = args.group or base.get("run_name", "run")

    subjects = norm_subjects(args.subjects) or discover_subjects(dataset_dir, suffix)

    print(f"config      : {args.config}")
    print(f"dataset_dir : {dataset_dir}")
    print(f"suffix      : {suffix}")
    print(f"group       : {group}  (under {save_dir}/)")
    print(f"sched       : {base.get('sched')}")
    print(f"subjects    : {len(subjects)} -> {', '.join(subjects) if subjects else '(none)'}")
    print()

    if not subjects:
        print("Nothing to do: no subject was selected and none was found with processed data.")
        return 1

    if args.dry_run:
        for subject in subjects:
            has_data = subject_has_data(dataset_dir, subject, suffix)
            target = Path(save_dir) / group / f"sub-{subject}"
            print(f"[dry-run] sub-{subject}: has_data={has_data} -> {target}")
        return 0

    cli = [args.python, "-m", "seizure_pred.cli.main"]
    preflight(cli, repo_root)

    results = []
    for index, subject in enumerate(subjects, start=1):
        subject_dir = resolve(str(Path(save_dir) / group / f"sub-{subject}"), repo_root)
        print()
        print("=" * 72)
        print(f"[{index}/{len(subjects)}] subject {subject} -> {subject_dir}")
        print("=" * 72)

        if not subject_has_data(dataset_dir, subject, suffix):
            print(f"  skip: no processed data for sub-{subject} (suffix `{suffix}`)")
            results.append((subject, "nodata", "no processed data"))
            if args.fail_fast:
                break
            continue

        if args.skip_existing:
            existing = newest_run_dir(subject_dir)
            if existing is not None and (existing / "raw_predictions.pkl").exists():
                print(f"  skip: already completed ({existing})")
                results.append((subject, "skipped", f"existing run {existing.name}"))
                continue

        cfg = build_subject_config(base, subject, group)
        cfg_path = subject_dir / "config_used.yaml"
        write_config(cfg, cfg_path)

        code = run_cli(
            cli + ["train", "--config", str(cfg_path), "--strict"],
            subject_dir / "train.log",
            repo_root,
        )
        if code != 0:
            results.append((subject, "failed", f"train exited with {code}"))
            if args.fail_fast:
                break
            continue

        run_dir = newest_run_dir(subject_dir)
        if run_dir is None:
            results.append((subject, "failed", "train finished but no run dir was found"))
            continue

        if not args.no_analyze:
            code = run_cli(
                cli + ["analyze", "--run-dir", str(run_dir)],
                subject_dir / "analyze.log",
                repo_root,
            )
            if code != 0:
                results.append((subject, "failed", f"analyze exited with {code}"))
                if args.fail_fast:
                    break
                continue

        try:
            shown = run_dir.relative_to(repo_root)
        except ValueError:
            shown = run_dir
        results.append((subject, "ok", str(shown)))

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    width = max((len(subject) for subject, _, _ in results), default=2)
    for subject, kind, detail in results:
        print(f"  sub-{subject:<{width}}  [{kind:<7}] {detail}")

    failed = sum(1 for _, kind, _ in results if kind == "failed")
    ok = sum(1 for _, kind, _ in results if kind == "ok")
    skipped = sum(1 for _, kind, _ in results if kind in ("skipped", "nodata"))
    print()
    print(f"{ok} ok, {skipped} skipped, {failed} failed (of {len(subjects)} requested).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
