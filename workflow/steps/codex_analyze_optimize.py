"""CodexAnalyzeAndOptimizeStep 实现 - 合并规划和执行"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from . import Step
from ..utils import load_changelog, run_agent, run_shell, which

if TYPE_CHECKING:
    from ..runner import WorkflowContext


class CodexAnalyzeAndOptimizeStep(Step):
    type_name = "codex_analyze_and_optimize"

    def run(self, ctx: WorkflowContext) -> None:
        # 1. 准备 codex 命令
        codex_cmd = self.cfg.get("codex_cmd")
        if not isinstance(codex_cmd, list) or not codex_cmd:
            raise ValueError("codex_analyze_and_optimize requires codex_cmd=[...]")
        if which(codex_cmd[0]) is None:
            raise RuntimeError(f"codex not found in PATH: {codex_cmd[0]}")

        # 2. 加载提示词模板
        prompt_file = self.cfg.get("prompt_file")
        if not prompt_file:
            raise ValueError("codex_analyze_and_optimize requires prompt_file")
        prompt_path = ctx.repo_root / prompt_file
        if not prompt_path.exists():
            raise FileNotFoundError(f"prompt file not found: {prompt_path}")
        prompt_template = prompt_path.read_text(encoding="utf-8")

        # 3. 加载历史变更和性能数据
        changelog = load_changelog(ctx.state_file)
        changelog_text = self._format_changelog(changelog)
        performance_history = self._format_performance_history(changelog)

        # 获取上一轮的性能变化
        last_perf_change = self._get_last_performance_change(changelog)

        # 4. 生成 codex prompt
        codex_prompt = prompt_template.format(
            polardb_dir=str(ctx.repo_root / "polardb"),
            last_decision=ctx.state.get("last_decision") or "(无上一轮决策)",
            run_id=ctx.logger.run_id,
            changelog=changelog_text,
            performance_history=performance_history,
            last_perf_change=last_perf_change,
        )

        # 6. 执行 codex 进行分析和优化（带迭代）
        max_iterations = int(self.cfg.get("max_iterations", 5))
        build_cmds = list(self.cfg.get("build_cmds", []))
        test_cmds = list(self.cfg.get("test_cmds", []))

        recall_cmd = str(self.cfg.get("recall_cmd", "")).strip()
        recall_regex = re.compile(str(self.cfg.get("recall_regex", r"recall.*:\s*([0-9.]+)")), re.IGNORECASE)
        min_recall = float(self.cfg.get("min_recall", ctx.workflow_cfg.get("initial_recall", 0.85)))

        # 迭代执行优化
        for i in range(1, max_iterations + 1):
            # 执行 codex
            log_path = ctx.logger.path(f"04_codex_analyze_optimize_iter_{i}.log")
            res = run_agent(cmd=codex_cmd, prompt=codex_prompt, cwd=ctx.repo_root, env=None, log_path=log_path)
            ctx.logger.append_command(" ".join(codex_cmd), res.returncode, log_path)
            if res.returncode != 0:
                raise RuntimeError(f"codex exec failed (rc={res.returncode}), see {log_path}")

            # 执行编译
            build_ok, build_logs = self._run_commands(ctx, build_cmds, f"04_build_iter_{i}", "build")

            # 如果编译成功，执行测试
            test_ok, test_logs = (True, []) if not build_ok else self._run_commands(ctx, test_cmds, f"04_test_iter_{i}", "test")

            # 如果测试成功，检查 recall
            recall_ok, recall_log = self._check_recall(ctx, i, recall_cmd, recall_regex, min_recall) if test_ok else (True, None)

            # 如果全部通过，标记成功并退出
            if test_ok and recall_ok:
                ctx.data["tests_passed"] = True
                ctx.logger.write_text("04_test_status.txt", f"PASS after iteration {i}\n")

                # 保存优化摘要
                summary_path = ctx.logger.path("04_optimization_summary.md")
                if summary_path.exists():
                    ctx.data["optimization_summary"] = summary_path.read_text(encoding="utf-8")

                return

            # 失败时，准备重试 prompt
            last_log = recall_log if (recall_log and not recall_ok) else test_logs[-1] if test_logs else build_logs[-1] if build_logs else log_path
            err_tail = last_log.read_text(encoding="utf-8", errors="ignore").splitlines()[-120:]

            # 更新 prompt 加入错误信息
            retry_section = f"\n\n## 上次尝试失败信息\n\n```\n{chr(10).join(err_tail)}\n```\n\n请分析失败原因，调整优化方案后重试。"
            codex_prompt = codex_prompt + retry_section

        raise RuntimeError(f"optimization did not pass after {max_iterations} iterations; see {ctx.logger.run_dir}")

    def _run_commands(self, ctx: WorkflowContext, cmds: list, prefix: str, phase: str) -> tuple[bool, list[Path]]:
        """执行一组命令"""
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
        """检查 recall 正确性"""
        if not recall_cmd:
            return True, None
        recall_log = ctx.logger.path(f"04_recall_iter_{i}.log")
        r = run_shell(command=recall_cmd, cwd=ctx.repo_root, env=None, log_path=recall_log, tee=True)
        ctx.logger.append_command(recall_cmd, r.returncode, recall_log)
        if r.returncode != 0:
            return False, recall_log
        recall_val = None
        for m in recall_regex.finditer(r.output_text):
            recall_val = float(m.group(1))
        if recall_val is not None and recall_val < min_recall:
            ctx.logger.write_text(f"04_recall_iter_{i}_fail.txt", f"recall={recall_val} < min={min_recall}\n")
            return False, recall_log
        return True, recall_log

    def _format_changelog(self, changelog: list) -> str:
        """格式化历史变更记录"""
        if not changelog:
            return "(无历史变更记录)"
        lines = []
        for e in changelog[-10:]:
            outcome_icon = "✓" if e.get("outcome") == "commit" else "✗"
            qps_b, qps_a = e.get("qps_before"), e.get("qps_after")
            qps_delta = f"{qps_a - qps_b:+.1f}" if qps_b and qps_a else "N/A"
            recall_b, recall_a = e.get("recall_before"), e.get("recall_after")
            recall_delta = f"{recall_a - recall_b:+.4f}" if recall_b and recall_a else "N/A"
            idx_b, idx_a = e.get("index_build_time_s_before"), e.get("index_build_time_s_after")
            idx_delta = f"{idx_a - idx_b:+.3f}s" if isinstance(idx_b, (int, float)) and isinstance(idx_a, (int, float)) else "N/A"
            lines.append(
                f"- [{outcome_icon}] R{e.get('round')}: {e.get('target_func')} | "
                f"IndexBuild {idx_delta} | QPS {qps_delta} | Recall {recall_delta} | "
                f"{', '.join(e.get('files_changed', [])[:3])}"
            )
            if e.get("changes_summary"):
                lines.append(f"  摘要: {e['changes_summary'][:100]}...")
        return "\n".join(lines)

    def _format_performance_history(self, changelog: list) -> str:
        """格式化性能历史趋势"""
        if not changelog:
            return "(无性能历史数据)"

        lines = ["## 性能历史趋势\n"]

        # 获取最近10轮的数据
        recent = changelog[-10:]

        qps_after_vals = [e.get("qps_after") for e in recent if isinstance(e.get("qps_after"), (int, float))]
        recall_after_vals = [e.get("recall_after") for e in recent if isinstance(e.get("recall_after"), (int, float))]
        index_after_vals = [e.get("index_build_time_s_after") for e in recent if isinstance(e.get("index_build_time_s_after"), (int, float))]

        best_qps = max(qps_after_vals) if qps_after_vals else 0
        best_recall = max(recall_after_vals) if recall_after_vals else 0
        best_index = min(index_after_vals) if index_after_vals else None

        best_parts = [f"QPS={best_qps:.1f}", f"Recall={best_recall:.4f}"]
        if isinstance(best_index, (int, float)):
            best_parts.insert(0, f"IndexBuild={best_index:.3f}s")
        lines.append(f"**历史最佳**: {', '.join(best_parts)}\n")

        # 显示每轮的性能数据
        lines.append("**最近10轮性能**:")
        for e in recent:
            outcome = "成功提交" if e.get("outcome") == "commit" else "回滚"
            qps_b, qps_a = e.get("qps_before", 0), e.get("qps_after", 0)
            recall_b, recall_a = e.get("recall_before", 0), e.get("recall_after", 0)
            qps_change = ((qps_a - qps_b) / qps_b * 100) if qps_b > 0 else 0
            idx_b, idx_a = e.get("index_build_time_s_before"), e.get("index_build_time_s_after")
            idx_text = "IndexBuild N/A"
            if isinstance(idx_b, (int, float)) and isinstance(idx_a, (int, float)):
                idx_text = f"IndexBuild {idx_b:.3f}s→{idx_a:.3f}s"
            lines.append(
                f"- R{e.get('round')}: {idx_text}, "
                f"QPS {qps_b:.1f}→{qps_a:.1f} ({qps_change:+.2f}%), "
                f"Recall {recall_b:.4f}→{recall_a:.4f}, {outcome}"
            )

        return "\n".join(lines)

    def _get_last_performance_change(self, changelog: list) -> str:
        """获取上一轮的性能变化"""
        if not changelog:
            return "(无上一轮数据)"

        last = changelog[-1]
        qps_b, qps_a = last.get("qps_before", 0), last.get("qps_after", 0)
        recall_b, recall_a = last.get("recall_before", 0), last.get("recall_after", 0)
        idx_b, idx_a = last.get("index_build_time_s_before"), last.get("index_build_time_s_after")

        qps_change = ((qps_a - qps_b) / qps_b * 100) if qps_b > 0 else 0
        recall_change = recall_a - recall_b
        outcome = "成功提交" if last.get("outcome") == "commit" else "回滚"

        return f"""上一轮 (R{last.get('round')}):
- 优化目标: {last.get('target_func')}
- Index build time: {idx_b} → {idx_a}
- QPS变化: {qps_b:.1f} → {qps_a:.1f} ({qps_change:+.2f}%)
- Recall变化: {recall_b:.4f} → {recall_a:.4f} ({recall_change:+.4f})
- 结果: {outcome}
- 修改文件: {', '.join(last.get('files_changed', [])[:5])}
"""
