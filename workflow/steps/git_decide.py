"""CodexGitDecideStep 实现"""
from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from . import Step
from ..utils import git, load_changelog, run_agent, run_shell, save_changelog, which

if TYPE_CHECKING:
    from ..runner import WorkflowContext


class CodexGitDecideStep(Step):
    type_name = "codex_git_decide"

    def run(self, ctx: WorkflowContext) -> None:
        codex_cmd = self.cfg.get("codex_cmd")
        if not isinstance(codex_cmd, list) or not codex_cmd:
            raise ValueError("codex_git_decide requires codex_cmd=[...]")
        if which(codex_cmd[0]) is None:
            raise RuntimeError(f"codex not found in PATH: {codex_cmd[0]}")

        mode = str(self.cfg.get("mode", "codex"))
        best, cur = ctx.state.get("best_benchmark_summary"), ctx.data.get("benchmark_summary")
        if not best:
            init_qps = ctx.workflow_cfg.get("initial_qps", 0.0)
            init_recall = ctx.workflow_cfg.get("initial_recall", 0.85)
            best = {"qps": {"mean": init_qps}, "recall": {"mean": init_recall}, "_initial": True}

        head = git("git rev-parse HEAD", ctx.repo_root)
        diff_stat = git("git diff --stat", ctx.repo_root)
        status = git("git status --porcelain", ctx.repo_root)

        ctx.logger.write_text("07_git_diff_stat.txt", diff_stat + "\n")
        ctx.logger.write_text("07_git_status.txt", status + "\n")

        prompt_key = "summarize_prompt_file" if mode == "local" else "prompt_file"
        prompt_path = ctx.repo_root / self.cfg.get(prompt_key, "")
        if not prompt_path.exists():
            raise FileNotFoundError(f"prompt file not found: {prompt_path}")
        prompt_template = prompt_path.read_text(encoding="utf-8")

        codex_prompt = self._build_prompt(mode, prompt_template, best, cur, diff_stat, status, head, ctx)

        lp = ctx.logger.path("07_codex_git_decide.log")
        res = run_agent(cmd=codex_cmd, prompt=codex_prompt, cwd=ctx.repo_root, env=None, log_path=lp)
        ctx.logger.append_command(" ".join(codex_cmd), res.returncode, lp)
        if res.returncode != 0:
            raise RuntimeError(f"codex git decide failed (rc={res.returncode}), see {lp}")

        ctx.logger.write_text("07_codex_git_decide.output.txt", res.output_text.strip() + "\n")

        outcome = self._decide_and_execute_git(ctx, best, cur, diff_stat, status) if mode == "local" else None

        ctx.state["last_git_head"] = git("git rev-parse HEAD", ctx.repo_root)
        if cur and outcome == "commit":
            ctx.state["best_benchmark_summary"] = cur
            self._run_profile_after_commit(ctx)

        decision_path = ctx.logger.path("07_decision.md")
        decision_text = decision_path.read_text(encoding="utf-8") if decision_path.exists() else ""
        if decision_text:
            ctx.state["last_decision"] = decision_text

        self._append_changelog(ctx, cur, best, diff_stat, decision_text, outcome)

    def _build_prompt(self, mode: str, template: str, best: dict | None, cur: dict | None,
                      diff_stat: str, status: str, head: str, ctx: WorkflowContext) -> str:
        if mode == "local":
            return template.format(
                best_summary=json.dumps(best, ensure_ascii=False, indent=2) if best else "null",
                cur_summary=json.dumps(cur, ensure_ascii=False, indent=2) if cur else "null",
                diff_stat=diff_stat or "(no diff)",
                run_dir=ctx.logger.run_dir,
            )
        return template.format(
            head=head,
            best_summary=json.dumps(best, ensure_ascii=False, indent=2) if best else "null",
            cur_summary=json.dumps(cur, ensure_ascii=False, indent=2) if cur else "null",
            diff_stat=diff_stat or "(no diff)",
            status=status or "(clean)",
            run_dir=ctx.logger.run_dir,
            mode=mode,
        )

    def _run_profile_after_commit(self, ctx: WorkflowContext) -> None:
        profile_cmd = self.cfg.get("profile_cmd")
        if not profile_cmd:
            return
        cmd = profile_cmd if isinstance(profile_cmd, str) else " ".join(profile_cmd)
        lp = ctx.logger.path("07_profile_after_commit.log")
        print(f"[profile] running profile after commit: {cmd}")
        run_shell(command=cmd, cwd=ctx.repo_root, env=None, log_path=lp, tee=True)

    def _decide_and_execute_git(self, ctx: WorkflowContext, best: dict | None, cur: dict | None,
                                 diff_stat: str, status: str) -> str:
        min_recall = float(self.cfg.get("min_recall", 0.85))
        cur_qps = cur.get("qps", {}).get("median") if cur else None
        cur_recall = cur.get("recall", {}).get("mean") if cur else None
        best_qps = best.get("qps", {}).get("median") if best else None

        should_commit = (
            cur_qps is not None and cur_recall is not None and
            cur_recall >= min_recall and
            (best_qps is None or cur_qps >= best_qps)
        )

        if should_commit and status.strip():
            git("git add -A", ctx.repo_root)
            msg = f"perf: QPS median {best_qps:.1f} -> {cur_qps:.1f}, recall {cur_recall:.3f}"
            git(f'git commit -m "{msg}"', ctx.repo_root)
            ctx.logger.write_text("07_git_action.txt", f"COMMIT: {msg}\n")
            return "commit"
        elif status.strip():
            git("git checkout -- .", ctx.repo_root)
            ctx.logger.write_text("07_git_action.txt", f"CHECKOUT: QPS {cur_qps}, recall {cur_recall}\n")
            return "checkout"
        else:
            ctx.logger.write_text("07_git_action.txt", "NO_CHANGE: working tree clean\n")
            return "no_change"

    def _append_changelog(self, ctx: WorkflowContext, cur: dict | None, prev: dict | None,
                          diff_stat: str, decision: str, outcome: str | None = None) -> None:
        changelog = load_changelog(ctx.state_file)
        summary_path = ctx.logger.path("05_changes_summary.md")
        changes_summary = summary_path.read_text(encoding="utf-8").strip() if summary_path.exists() else ""

        if outcome is None:
            outcome = "checkout"
            if decision and "commit" in decision.lower() and "不 commit" not in decision and "不commit" not in decision:
                outcome = "commit"

        files_changed = [line.split("|")[0].strip() for line in diff_stat.splitlines() if "|" in line]

        entry = {
            "round": ctx.round_idx,
            "run_id": ctx.run_id,
            "timestamp": datetime.now().isoformat(),
            "target_func": ctx.data.get("targets", [{}])[0].get("function", ""),
            "files_changed": files_changed,
            "changes_summary": changes_summary,
            "qps_before": prev.get("qps", {}).get("mean") if prev else None,
            "qps_after": cur.get("qps", {}).get("mean") if cur else None,
            "outcome": outcome,
        }
        changelog.append(entry)
        save_changelog(ctx.state_file, changelog)
