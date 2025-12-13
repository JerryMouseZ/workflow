#!/usr/bin/env python3
"""工作流入口脚本"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from workflow import WorkflowRunner, load_config


def discover_repo_root(cwd: Path, script_dir: Path) -> Path:
    # 优先从当前工作目录向上找项目根目录（包含 polardb/ 与 test/）
    for p in [cwd, *cwd.parents]:
        if (p / "polardb").is_dir() and (p / "test").is_dir():
            return p
    # 其次：按脚本所在位置推断（常见：repo_root/workflow/run.py）
    parent = script_dir.parent
    if (parent / "polardb").is_dir() and (parent / "test").is_dir():
        return parent
    return cwd


def resolve_config_path(cfg_arg: str, *, cwd: Path, repo_root: Path, script_dir: Path) -> Path:
    p = Path(cfg_arg)
    if p.is_absolute():
        if p.exists():
            return p
        raise FileNotFoundError(f"config not found: {p}")

    for base in (cwd, repo_root, script_dir, script_dir.parent):
        cand = base / p
        if cand.exists():
            return cand
    raise FileNotFoundError(f"config not found: tried {cfg_arg} under cwd/repo_root/script_dir")


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

    cwd = Path(os.getcwd())
    script_dir = Path(__file__).resolve().parent
    repo_root = discover_repo_root(cwd, script_dir)

    config_path = resolve_config_path(args.config, cwd=cwd, repo_root=repo_root, script_dir=script_dir)
    cfg = load_config(config_path)
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
