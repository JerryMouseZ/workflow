"""CommandStep 实现"""
from __future__ import annotations

from typing import TYPE_CHECKING

from . import Step
from ..utils import run_shell

if TYPE_CHECKING:
    from ..runner import WorkflowContext


class CommandStep(Step):
    type_name = "command"

    def run(self, ctx: WorkflowContext) -> None:
        command = str(self.cfg.get("command", "")).strip()
        if not command:
            raise ValueError("command step requires command")
        name = str(self.cfg.get("name", "command")).strip() or "command"
        tee = bool(self.cfg.get("tee", True))
        log_path = ctx.logger.path(f"cmd/{name}.log")
        res = run_shell(command=command, cwd=ctx.repo_root, env=None, log_path=log_path, tee=tee)
        ctx.logger.append_command(command, res.returncode, log_path)
        if int(self.cfg.get("require_success", 1)) == 1 and res.returncode != 0:
            raise RuntimeError(f"command step failed (rc={res.returncode}): {command}, see {log_path}")
        ctx.data.setdefault("commands", {})[name] = {"returncode": res.returncode, "log": str(log_path)}
