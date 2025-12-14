"""WorkflowRunner 和 WorkflowContext"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .steps import STEP_REGISTRY
from .utils import RunLogger, git, json_dump, latest_run_id, now_id


@dataclass
class WorkflowContext:
    """工作流执行上下文"""
    repo_root: Path
    run_id: str
    round_idx: int
    logger: RunLogger
    state_file: Path
    state: dict[str, Any]
    config: dict[str, Any] = field(default_factory=dict)
    workflow_cfg: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)


class WorkflowRunner:
    """工作流执行器"""

    def __init__(
        self, *, repo_root: Path, config: dict[str, Any],
        skip_selectors: list[str] | None = None,
        only_selectors: list[str] | None = None,
        resume: bool = False,
    ) -> None:
        self.repo_root = repo_root
        self.config = config
        self.steps_cfg: list[dict[str, Any]] = list(config.get("steps", []))
        self.workflow_cfg: dict[str, Any] = dict(config.get("workflow", {}))
        self.skip_selectors = self._normalize_selectors(skip_selectors)
        self.only_selectors = self._normalize_selectors(only_selectors)
        self.resume = resume

    @staticmethod
    def _normalize_selectors(raw: list[str] | None) -> list[str]:
        if not raw:
            return []
        out: list[str] = []
        for item in raw:
            if item is not None:
                out.extend(tok.strip() for tok in str(item).split(",") if tok.strip())
        return out

    @staticmethod
    def _matches_selector(*, step_cfg: dict[str, Any], idx: int, selector: str) -> bool:
        sel = selector.strip()
        if not sel:
            return False
        step_type = str(step_cfg.get("type", "")).strip()
        step_name = str(step_cfg.get("name", "")).strip()

        if sel.startswith("#"):
            return sel[1:].strip().isdigit() and int(sel[1:].strip()) == idx

        for prefix in ("type:", "type=", "name:", "name=", "idx:", "idx="):
            if sel.startswith(prefix):
                key, value = prefix[:-1], sel[len(prefix):].strip()
                if key == "type":
                    return value == step_type
                if key == "name":
                    return value == step_name
                if key == "idx":
                    return value.isdigit() and int(value) == idx
                return False

        if sel.isdigit() and int(sel) == idx:
            return True
        return sel == step_name or sel == step_type

    def _should_run_step(self, *, step_cfg: dict[str, Any], idx: int) -> tuple[bool, str | None]:
        if step_cfg.get("enabled", True) is False:
            return False, "disabled (enabled=false)"
        if self.only_selectors:
            if not any(self._matches_selector(step_cfg=step_cfg, idx=idx, selector=s) for s in self.only_selectors):
                return False, "not selected (--only)"
        if self.skip_selectors:
            if any(self._matches_selector(step_cfg=step_cfg, idx=idx, selector=s) for s in self.skip_selectors):
                return False, "skipped (--skip)"
        return True, None

    def format_steps(self) -> str:
        lines = ["idx  enabled  type                      name"]
        for i, s in enumerate(self.steps_cfg, 1):
            lines.append(f"{i:>3}  {str(bool(s.get('enabled', True))).lower():<7}  {str(s.get('type', '')).strip():<24}  {str(s.get('name', '')).strip()}")
        return "\n".join(lines)

    def format_step_plan(self) -> str:
        lines = ["idx  action   type                      name  reason"]
        for i, s in enumerate(self.steps_cfg, 1):
            run, reason = self._should_run_step(step_cfg=s, idx=i)
            lines.append(f"{i:>3}  {'run' if run else 'skip':<6}  {str(s.get('type', '')).strip():<24}  {str(s.get('name', '')).strip():<20}  {reason or ''}".rstrip())
        return "\n".join(lines)

    def run(self) -> None:
        rounds = int(self.workflow_cfg.get("rounds", 1))
        log_dir = self.repo_root / str(self.workflow_cfg.get("log_dir", "workflow/logs"))
        state_file = self.repo_root / str(self.workflow_cfg.get("state_file", "workflow/state.json"))

        state: dict[str, Any] = {}
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
            except Exception:
                state = {}

        # 当 initial_qps 和 initial_recall 都为 0 且没有历史 benchmark 数据时，先运行一次 benchmark 初始化
        self._maybe_run_initial_benchmark(state, state_file, log_dir)

        last_completed_idx = state.get("last_completed_step_idx", 0) if self.resume else 0
        persisted_data = state.get("data", {}) if self.resume else {}

        for round_idx in range(1, rounds + 1):
            run_id = self._resolve_run_id(state, log_dir, round_idx, last_completed_idx)
            run_dir = log_dir / run_id
            logger = RunLogger(run_dir)

            state["active_run_id"], state["active_round_idx"] = run_id, round_idx
            json_dump(state_file, state)

            self._log_env(logger, run_id, round_idx, last_completed_idx)

            ctx = WorkflowContext(
                repo_root=self.repo_root, run_id=run_id, round_idx=round_idx,
                logger=logger, state_file=state_file, state=state,
                config=self.config, workflow_cfg=self.workflow_cfg, data=dict(persisted_data),
            )

            step_plan = self._build_step_plan(last_completed_idx)
            self._log_step_plan(logger, step_plan)

            for p in step_plan:
                if p["action"] != "run":
                    continue
                idx, t = int(p["idx"]), str(p["type"])
                STEP_REGISTRY[t](self.steps_cfg[idx - 1]).run(ctx)
                state["last_completed_step_idx"], state["data"] = idx, ctx.data
                json_dump(state_file, state)

            if self._is_workflow_completed(step_plan):
                state.update({"last_completed_step_idx": 0, "data": {}})
                state.pop("active_run_id", None)
                state.pop("active_round_idx", None)
                json_dump(state_file, state)

    def _resolve_run_id(self, state: dict, log_dir: Path, round_idx: int, last_completed_idx: int) -> str:
        if self.resume:
            active = state.get("active_run_id")
            if isinstance(active, str) and state.get("active_round_idx") == round_idx and active.endswith(f"_r{round_idx}"):
                return active
            if last_completed_idx > 0:
                return latest_run_id(log_dir, round_idx=round_idx) or f"{now_id()}_r{round_idx}"
        return f"{now_id()}_r{round_idx}"

    def _log_env(self, logger: RunLogger, run_id: str, round_idx: int, last_completed_idx: int) -> None:
        env_name = f"00_env_resume_{now_id()}.json" if self.resume and logger.path("00_env.json").exists() else "00_env.json"
        json_dump(logger.path(env_name), {
            "run_id": run_id, "round": round_idx, "cwd": str(self.repo_root),
            "git_head": git("git rev-parse HEAD", self.repo_root),
            "skip_selectors": self.skip_selectors, "only_selectors": self.only_selectors,
            "resume": self.resume,
            "resume_from_idx": last_completed_idx + 1 if self.resume and last_completed_idx > 0 else None,
        })

    def _build_step_plan(self, last_completed_idx: int) -> list[dict[str, Any]]:
        plan: list[dict[str, Any]] = []
        for idx, s_cfg in enumerate(self.steps_cfg, 1):
            t = s_cfg.get("type")
            if t not in STEP_REGISTRY:
                raise ValueError(f"unknown step type: {t}")
            run_it, reason = self._should_run_step(step_cfg=s_cfg, idx=idx)
            if run_it and self.resume and idx <= last_completed_idx:
                run_it, reason = False, f"resumed (completed in previous run, idx<={last_completed_idx})"
            plan.append({
                "idx": idx, "type": str(t), "name": str(s_cfg.get("name", "")).strip(),
                "enabled": bool(s_cfg.get("enabled", True)), "action": "run" if run_it else "skip", "reason": reason,
            })
        return plan

    def _log_step_plan(self, logger: RunLogger, step_plan: list[dict[str, Any]]) -> None:
        name = f"00_step_plan_resume_{now_id()}.json" if self.resume and logger.path("00_step_plan.json").exists() else "00_step_plan.json"
        json_dump(logger.path(name), step_plan)

    def _is_workflow_completed(self, step_plan: list[dict[str, Any]]) -> bool:
        for p in step_plan:
            if not p.get("enabled", True):
                continue
            if p["action"] == "run":
                continue
            reason = str(p.get("reason") or "")
            if self.resume and reason.startswith("resumed (completed in previous run"):
                continue
            return False
        return True

    def _maybe_run_initial_benchmark(self, state: dict[str, Any], state_file: Path, log_dir: Path) -> None:
        """当 initial_qps 和 initial_recall 都为 0 且没有历史 benchmark 数据时，运行一次 benchmark 初始化"""
        init_qps = float(self.workflow_cfg.get("initial_qps", 0.0))
        init_recall = float(self.workflow_cfg.get("initial_recall", 0.0))
        if init_qps != 0.0 or init_recall != 0.0 or state.get("best_benchmark_summary"):
            return

        # 找到 benchmark step 配置
        bench_cfg = next((s for s in self.steps_cfg if s.get("type") == "benchmark"), None)
        if not bench_cfg:
            return

        print("[init] initial_qps=0 and initial_recall=0, running initial benchmark...")
        run_id = f"{now_id()}_init"
        logger = RunLogger(log_dir / run_id)
        ctx = WorkflowContext(
            repo_root=self.repo_root, run_id=run_id, round_idx=0,
            logger=logger, state_file=state_file, state=state,
            config=self.config, workflow_cfg=self.workflow_cfg, data={},
        )
        STEP_REGISTRY["benchmark"](bench_cfg).run(ctx)
        summary = ctx.data.get("benchmark_summary")
        if summary:
            state["best_benchmark_summary"] = summary
            json_dump(state_file, state)
            print(f"[init] initial benchmark done: QPS={summary.get('qps', {}).get('median')}, recall={summary.get('recall', {}).get('mean')}")
