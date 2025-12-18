"""CodexGenerateKiroPromptStep 实现"""
from __future__ import annotations

from typing import TYPE_CHECKING

from . import Step
from ..utils import load_changelog, run_agent, which

if TYPE_CHECKING:
    from ..runner import WorkflowContext


class CodexGenerateKiroPromptStep(Step):
    type_name = "codex_generate_kiro_prompt"

    def run(self, ctx: WorkflowContext) -> None:
        codex_cmd = self.cfg.get("codex_cmd")
        if not isinstance(codex_cmd, list) or not codex_cmd:
            raise ValueError("codex_generate_kiro_prompt requires codex_cmd=[...]")
        if which(codex_cmd[0]) is None:
            raise RuntimeError(f"codex not found in PATH: {codex_cmd[0]}")

        prompt_file = self.cfg.get("prompt_file")
        if not prompt_file:
            raise ValueError("codex_generate_kiro_prompt requires prompt_file")
        prompt_path = ctx.repo_root / prompt_file
        if not prompt_path.exists():
            raise FileNotFoundError(f"prompt file not found: {prompt_path}")
        prompt_template = prompt_path.read_text(encoding="utf-8")

        max_context_chars = int(self.cfg.get("max_context_chars", 12000))
        profile_filtered = ctx.data.get("profile", {}).get("filtered", [])[:20]
        targets = ctx.data.get("targets") or profile_filtered[:3]
        context_data = ctx.data.get("context", {})
        if context_data.get("md_path"):
            context_md = ctx.logger.path(context_data["md_path"]).read_text(encoding="utf-8")[:max_context_chars]
        else:
            context_md = "(no context collected)"

        kiro_prompt_path = ctx.logger.path("04_kiro_prompt.md")
        changelog_text = self._format_changelog(load_changelog(ctx.state_file))

        codex_prompt = prompt_template.format(
            target_lines="\n".join([f"- {t['function']} (self%={t['self_pct']}, samples={t['samples']})" for t in targets]),
            prof_lines="\n".join([f"- {e['function']} (self%={e['self_pct']}, samples={e['samples']})" for e in profile_filtered]),
            context_md=context_md,
            output_file=str(kiro_prompt_path),
            last_decision=ctx.state.get("last_decision") or "(无上一轮决策)",
            run_id=ctx.logger.run_id,
            changelog=changelog_text,
        )

        log_path = ctx.logger.path("04_codex_generate_kiro_prompt.log")
        res = run_agent(cmd=codex_cmd, prompt=codex_prompt, cwd=ctx.repo_root, env=None, log_path=log_path)
        ctx.logger.append_command(" ".join(codex_cmd), res.returncode, log_path)
        if res.returncode != 0:
            raise RuntimeError(f"codex exec failed (rc={res.returncode}), see {log_path}")

        if not kiro_prompt_path.exists():
            raise RuntimeError(f"codex did not create kiro prompt file: {kiro_prompt_path}")
        ctx.data["kiro_prompt_file"] = str(kiro_prompt_path)

    def _format_changelog(self, changelog: list) -> str:
        if not changelog:
            return "(无历史变更记录)"
        lines = []
        for e in changelog[-10:]:
            outcome_icon = "✓" if e.get("outcome") == "commit" else "✗"
            qps_b, qps_a = e.get("qps_before"), e.get("qps_after")
            qps_delta = f"{qps_a - qps_b:+.1f}" if qps_b and qps_a else "N/A"
            lines.append(f"- [{outcome_icon}] R{e.get('round')}: {e.get('target_func')} | QPS {qps_delta} | {', '.join(e.get('files_changed', [])[:3])}")
            if e.get("changes_summary"):
                lines.append(f"  摘要: {e['changes_summary'][:100]}...")
        return "\n".join(lines)
