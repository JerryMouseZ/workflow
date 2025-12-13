#!/usr/bin/env python3
"""工作流入口脚本"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from workflow import WorkflowRunner, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto performance optimization workflow")
    parser.add_argument("--config", default="workflow/config.toml", help="Path to config TOML")
    parser.add_argument("--rounds", type=int, default=None, help="Override [workflow].rounds")
    parser.add_argument("--skip", action="append", default=[], help="Skip steps by selector")
    parser.add_argument("--only", action="append", default=[], help="Run only selected steps")
    parser.add_argument("--list-steps", action="store_true", help="List steps and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print step plan and exit")
    parser.add_argument("--resume", action="store_true", help="Resume from last completed step")
    args = parser.parse_args()

    repo_root = Path(os.getcwd())
    cfg = load_config(repo_root / args.config)
    if args.rounds is not None:
        cfg["workflow"]["rounds"] = args.rounds

    runner = WorkflowRunner(
        repo_root=repo_root, config=cfg,
        skip_selectors=args.skip, only_selectors=args.only, resume=args.resume,
    )
    if args.list_steps:
        print(runner.format_steps())
        return 0
    if args.dry_run:
        print(runner.format_step_plan())
        return 0
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
