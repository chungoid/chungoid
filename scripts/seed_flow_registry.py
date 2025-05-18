"""Seed the Flow Registry with baseline Stage-Flow YAML files.

Usage
-----
python dev/scripts/seed_flow_registry.py [--stages-dir path] [--overwrite] [--mode persistent|memory] [--dry-run]

By default it scans `chungoid-core/server_prompts/stages/*.yaml` and stores
records in the persistent Chroma DB (unless `--mode memory` is passed).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure src import path
PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJ_ROOT / "chungoid-core" / "src"))

from chungoid.utils.flow_registry import FlowRegistry, FlowCard  # noqa: E402

DEFAULT_STAGE_DIR = PROJ_ROOT / "chungoid-core" / "server_prompts" / "stages"


def seed(
    stages_dir: Path = DEFAULT_STAGE_DIR,
    *,
    overwrite: bool = False,
    mode: str = "persistent",
    dry_run: bool = False,
) -> None:
    if not stages_dir.exists():
        raise FileNotFoundError(stages_dir)

    registry = FlowRegistry(project_root=PROJ_ROOT, chroma_mode=mode)

    count = 0
    for path in sorted(stages_dir.glob("*.yaml")):
        flow_id = path.stem  # use filename sans extension
        yaml_text = path.read_text(encoding="utf-8")
        card = FlowCard(flow_id=flow_id, name=flow_id, yaml_text=yaml_text, tags=["baseline"])
        if dry_run:
            print(f"[dry-run] Would add {flow_id}")
            continue
        try:
            registry.add(card, overwrite=overwrite)
            print(f"Added {flow_id}")
            count += 1
        except ValueError as exc:
            print(f"Skip {flow_id}: {exc}")
    print(f"Seed complete ({count} flows added)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Seed Flow Registry with baseline flows")
    ap.add_argument("--stages-dir", type=Path, default=DEFAULT_STAGE_DIR)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--mode", choices=["persistent", "memory"], default="persistent")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    seed(args.stages_dir, overwrite=args.overwrite, mode=args.mode, dry_run=args.dry_run) 