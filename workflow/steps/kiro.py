"""KiroApplyAndTestStep 实现"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from . import Step
from ..utils import run_agent_file, run_shell, which

if TYPE_CHECKING:
    from ..runner import WorkflowContext


class KiroApplyAndTestStep(Step):
    type_name = "kiro_apply_and_test"

    def run(self, ctx: WorkflowContext) -> None:
        kiro_cmd = self.cfg.get("kiro_cmd")
        if not isinstance(kiro_cmd, list) or not kiro_cmd:
            raise ValueError("kiro_apply_and_test requires kiro_cmd=[...]")
        if which(kiro_cmd[0]) is None:
            raise RuntimeError(f"kiro-cli not found in PATH: {kiro_cmd[0]}")

        retry_prompt_path = ctx.repo_root / self.cfg.get("retry_prompt_file", "")
        if not retry_prompt_path.exists():
            raise FileNotFoundError(f"retry prompt file not found: {retry_prompt_path}")
        retry_prompt_template = retry_prompt_path.read_text(encoding="utf-8")

        build_cmds = list(self.cfg.get("build_cmds", []))
        test_cmds = list(self.cfg.get("test_cmds", []))
        max_iterations = int(self.cfg.get("max_iterations", 5))

        recall_cmd = str(self.cfg.get("recall_cmd", "")).strip()
        recall_regex = re.compile(str(self.cfg.get("recall_regex", r"recall.*:\s*([0-9.]+)")), re.IGNORECASE)
        min_recall = float(self.cfg.get("min_recall", ctx.workflow_cfg.get("initial_recall", 0.85)))

        prompt_file = ctx.data.get("kiro_prompt_file")
        if not prompt_file:
            raise RuntimeError("missing kiro_prompt_file from previous step")
        prompt_path = Path(prompt_file)
        if not prompt_path.exists():
            raise RuntimeError(f"kiro prompt file not found: {prompt_path}")

        for i in range(1, max_iterations + 1):
            kiro_log = ctx.logger.path(f"05_kiro_iter_{i}.log")
            res = run_agent_file(cmd=kiro_cmd, input_file=prompt_path, cwd=ctx.repo_root, env=None, log_path=kiro_log)
            ctx.logger.append_command(f"{' '.join(kiro_cmd)} < {prompt_path}", res.returncode, kiro_log)
            if res.returncode != 0:
                raise RuntimeError(f"kiro-cli failed (rc={res.returncode}), see {kiro_log}")

            build_ok, build_logs = self._run_commands(ctx, build_cmds, f"05_build_iter_{i}", "build")
            test_ok, test_logs = (True, []) if not build_ok else self._run_commands(ctx, test_cmds, f"05_test_iter_{i}", "test")
            recall_ok, recall_log = self._check_recall(ctx, i, recall_cmd, recall_regex, min_recall) if test_ok else (True, None)

            if test_ok and recall_ok:
                ctx.data["tests_passed"] = True
                ctx.logger.write_text("05_test_status.txt", f"PASS after iteration {i}\n")
                return

            last_log = recall_log if (recall_log and not recall_ok) else test_logs[-1] if test_logs else build_logs[-1] if build_logs else kiro_log
            err_tail = last_log.read_text(encoding="utf-8", errors="ignore").splitlines()[-120:]
            retry_content = retry_prompt_template.format(error_log="\n".join(err_tail))
            prompt_path = ctx.logger.path(f"05_kiro_retry_{i}.md")
            prompt_path.write_text(retry_content, encoding="utf-8")

        raise RuntimeError(f"tests did not pass after {max_iterations} iterations; see {ctx.logger.run_dir}")

    def _run_commands(self, ctx: WorkflowContext, cmds: list, prefix: str, phase: str) -> tuple[bool, list[Path]]:
        logs: list[Path] = []
        for j, c in enumerate(cmds, 1):
            lp = ctx.logger.path(f"{prefix}_{j}.log")
            r = run_shell(command=c, cwd=ctx.repo_root, env=None, log_path=lp)
            ctx.logger.append_command(c, r.returncode, lp)
            logs.append(lp)
            if r.returncode != 0:
                return False, logs
        return True, logs

    def _check_recall(self, ctx: WorkflowContext, i: int, recall_cmd: str, recall_regex: re.Pattern, min_recall: float) -> tuple[bool, Path | None]:
        if not recall_cmd:
            return True, None
        recall_log = ctx.logger.path(f"05_recall_iter_{i}.log")
        r = run_shell(command=recall_cmd, cwd=ctx.repo_root, env=None, log_path=recall_log, tee=True)
        ctx.logger.append_command(recall_cmd, r.returncode, recall_log)
        if r.returncode != 0:
            return False, recall_log
        recall_val = None
        for m in recall_regex.finditer(r.output_text):
            recall_val = float(m.group(1))
        if recall_val is not None and recall_val < min_recall:
            ctx.logger.write_text(f"05_recall_iter_{i}_fail.txt", f"recall={recall_val} < min={min_recall}\n")
            return False, recall_log
        return True, recall_log
