"""TargetSelectStep 实现"""
from __future__ import annotations

from typing import TYPE_CHECKING

from . import Step
from ..utils import load_changelog, pick_targets

if TYPE_CHECKING:
    from ..runner import WorkflowContext


class TargetSelectStep(Step):
    type_name = "target_select"

    def run(self, ctx: WorkflowContext) -> None:
        filtered = ctx.data.get("profile", {}).get("filtered") or []
        strategy = str(self.cfg.get("strategy", "top_self_percent"))
        pick_top_k = int(self.cfg.get("pick_top_k", 3))
        max_attempts = int(self.cfg.get("max_attempts_per_func", 3))
        changelog = load_changelog(ctx.state_file)
        targets = pick_targets(filtered, strategy=strategy, k=pick_top_k,
                               changelog=changelog, max_attempts_per_func=max_attempts)
        if not targets:
            raise RuntimeError("no target functions after filtering")
        ctx.data["targets"] = targets
        ctx.logger.write_text(
            "02_targets.txt",
            "\n".join([f"{t['function']}  self%={t['self_pct']} samples={t['samples']}" for t in targets]) + "\n",
        )
