#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from workflow_runner import WorkflowRunner, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto performance optimization workflow")
    parser.add_argument("--config", default="workflow/config.toml", help="Path to config TOML")
    parser.add_argument("--rounds", type=int, default=None, help="Override [workflow].rounds")
    args = parser.parse_args()

    repo_root = Path(os.getcwd())
    config_path = repo_root / args.config
    cfg = load_config(config_path)
    if args.rounds is not None:
        cfg["workflow"]["rounds"] = args.rounds

    runner = WorkflowRunner(repo_root=repo_root, config=cfg)
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

