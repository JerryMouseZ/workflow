"""Step 基类和所有 Step 实现"""
from __future__ import annotations

import json
import re
import statistics
from typing import TYPE_CHECKING, Any

from .utils import (
    collect_context_md, filter_profile_entries, git, json_dump, latest_folded_file,
    load_changelog, parse_self_time_table, pick_targets, run_agent, run_agent_file,
    run_cmd, run_shell, save_changelog, which,
)

if TYPE_CHECKING:
    from .runner import WorkflowContext


class Step:
    """Step 基类"""
    type_name: str = ""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def run(self, ctx: WorkflowContext) -> None:
        raise NotImplementedError


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


class ProfileAnalyzeStep(Step):
    type_name = "profile_analyze"

    def run(self, ctx: WorkflowContext) -> None:
        analyze_cmd = self.cfg.get("analyze_cmd")
        top_n = int(self.cfg.get("top_n", 20))
        ignore_regex = list(self.cfg.get("ignore_regex", [r"^\[.*\]$", r"(?i)unknown"]))
        if not isinstance(analyze_cmd, list) or not analyze_cmd:
            raise ValueError("profile_analyze requires analyze_cmd=[...]")

        latest = latest_folded_file(ctx.repo_root)
        if latest is None:
            self._bootstrap_profile_artifacts(ctx)
            latest = latest_folded_file(ctx.repo_root)
            if latest is None:
                raise RuntimeError(
                    "no .folded or .perf.data files found under test/profile_output/ "
                    "(bootstrap attempted but still missing; see 00_bootstrap_benchmark.log / 00_bootstrap_profile.log)"
                )

        log_path = ctx.logger.path("01_profile_analyze.log")
        res = run_cmd(cmd=analyze_cmd + [str(latest), str(top_n)], cwd=ctx.repo_root, env=None, stdin_text=None, log_path=log_path)
        ctx.logger.append_command(" ".join(analyze_cmd), res.returncode, log_path)
        if res.returncode != 0:
            raise RuntimeError(f"profile analyze failed (rc={res.returncode}), see {log_path}")

        entries = parse_self_time_table(res.output_text)
        filtered = filter_profile_entries(entries, ignore_regex)
        json_dump(ctx.logger.path("01_profile_entries.raw.json"), entries)
        json_dump(ctx.logger.path("01_profile_entries.filtered.json"), filtered)
        ctx.data["profile"] = {"raw": entries, "filtered": filtered, "ignore_regex": ignore_regex}

    def _bootstrap_profile_artifacts(self, ctx: WorkflowContext) -> None:
        """
        当首次运行时 test/profile_output 为空（或不存在）时：
        先跑一次性能测试（benchmark），再跑一次 profile 生成，确保产出 .folded/.perf.data。
        """

        def _as_shell_cmd(v: Any) -> str:
            if isinstance(v, str):
                return v.strip()
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                return " ".join(x.strip() for x in v if x.strip())
            return ""

        # 优先从 profile_analyze step 自己的配置读取；否则尝试从整体 config 推断。
        bench_cmd = _as_shell_cmd(self.cfg.get("bootstrap_bench_cmd"))
        profile_cmd = _as_shell_cmd(self.cfg.get("bootstrap_profile_cmd"))

        steps = []
        if isinstance(getattr(ctx, "config", None), dict):
            steps = ctx.config.get("steps", []) if isinstance(ctx.config.get("steps"), list) else []

        if not bench_cmd:
            for s in steps:
                if isinstance(s, dict) and s.get("type") == "benchmark":
                    bench_cmd = _as_shell_cmd(s.get("command"))
                    if bench_cmd:
                        break

        if not profile_cmd:
            for s in steps:
                if isinstance(s, dict) and s.get("type") == "codex_git_decide":
                    profile_cmd = _as_shell_cmd(s.get("profile_cmd"))
                    if profile_cmd:
                        break

        # 兜底：如果 benchmark 命令看起来是 run_benchmark.sh 且未带 --profile，则拼上一次 profile。
        if not profile_cmd and bench_cmd and "run_benchmark.sh" in bench_cmd and "--profile" not in bench_cmd:
            profile_cmd = bench_cmd + " --profile"

        if not bench_cmd or not profile_cmd:
            raise RuntimeError(
                "missing profile artifacts and cannot infer bootstrap commands; "
                "set profile_analyze.bootstrap_bench_cmd/bootstrap_profile_cmd in config.toml "
                "or add a command step to generate profile first"
            )

        # 确保目录存在（profile 脚本通常会创建，但这里先建好）
        (ctx.repo_root / "test" / "profile_output").mkdir(parents=True, exist_ok=True)

        bench_log = ctx.logger.path("00_bootstrap_benchmark.log")
        prof_log = ctx.logger.path("00_bootstrap_profile.log")

        print("[profile] missing profile artifacts, bootstrapping: benchmark -> profile")

        r1 = run_shell(command=bench_cmd, cwd=ctx.repo_root, env=None, log_path=bench_log, tee=True)
        ctx.logger.append_command(bench_cmd, r1.returncode, bench_log)
        if r1.returncode != 0:
            raise RuntimeError(f"bootstrap benchmark failed (rc={r1.returncode}), see {bench_log}")

        r2 = run_shell(command=profile_cmd, cwd=ctx.repo_root, env=None, log_path=prof_log, tee=True)
        ctx.logger.append_command(profile_cmd, r2.returncode, prof_log)
        if r2.returncode != 0:
            raise RuntimeError(f"bootstrap profile failed (rc={r2.returncode}), see {prof_log}")


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


class ContextCollectStep(Step):
    type_name = "context_collect"

    def run(self, ctx: WorkflowContext) -> None:
        target_func = str(ctx.data["targets"][0]["function"])
        roots = list(self.cfg.get("search_roots", ["polardb"]))
        snippet_radius = int(self.cfg.get("snippet_radius", 50))
        max_files = int(self.cfg.get("max_files", 20))
        max_total_matches = int(self.cfg.get("max_total_matches", 120))

        md, matches = collect_context_md(
            repo_root=ctx.repo_root, target_func=target_func, roots=roots,
            snippet_radius=snippet_radius, max_files=max_files, max_total_matches=max_total_matches,
        )
        ctx.logger.write_text("03_context.md", md)
        json_dump(ctx.logger.path("03_context.matches.json"), matches)
        ctx.data["context"] = {"target_func": target_func, "roots": roots, "md_path": "03_context.md"}


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

        # codex 将 kiro prompt 写入此文件
        kiro_prompt_path = ctx.logger.path("04_kiro_prompt.md")

        # 格式化历史 changelog
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
        for e in changelog[-10:]:  # 只保留最近 10 条
            outcome_icon = "✓" if e.get("outcome") == "commit" else "✗"
            qps_b, qps_a = e.get("qps_before"), e.get("qps_after")
            qps_delta = f"{qps_a - qps_b:+.1f}" if qps_b and qps_a else "N/A"
            lines.append(f"- [{outcome_icon}] R{e.get('round')}: {e.get('target_func')} | QPS {qps_delta} | {', '.join(e.get('files_changed', [])[:3])}")
            if e.get("changes_summary"):
                lines.append(f"  摘要: {e['changes_summary'][:100]}...")
        return "\n".join(lines)


class KiroApplyAndTestStep(Step):
    type_name = "kiro_apply_and_test"

    def run(self, ctx: WorkflowContext) -> None:
        from pathlib import Path as _Path

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

        # recall 检查配置
        recall_cmd = str(self.cfg.get("recall_cmd", "")).strip()
        recall_regex = re.compile(str(self.cfg.get("recall_regex", r"recall.*:\s*([0-9.]+)")), re.IGNORECASE)
        min_recall = float(self.cfg.get("min_recall", ctx.workflow_cfg.get("initial_recall", 0.85)))

        # 首次使用 codex 生成的 prompt 文件
        prompt_file = ctx.data.get("kiro_prompt_file")
        if not prompt_file:
            raise RuntimeError("missing kiro_prompt_file from previous step")
        prompt_path = _Path(prompt_file)
        if not prompt_path.exists():
            raise RuntimeError(f"kiro prompt file not found: {prompt_path}")

        for i in range(1, max_iterations + 1):
            kiro_log = ctx.logger.path(f"05_kiro_iter_{i}.log")
            res = run_agent_file(cmd=kiro_cmd, input_file=prompt_path, cwd=ctx.repo_root, env=None, log_path=kiro_log)
            ctx.logger.append_command(f"{' '.join(kiro_cmd)} < {prompt_path}", res.returncode, kiro_log)
            if res.returncode != 0:
                raise RuntimeError(f"kiro-cli failed (rc={res.returncode}), see {kiro_log}")

            # build & test
            build_ok, build_logs = True, []
            for j, c in enumerate(build_cmds, 1):
                lp = ctx.logger.path(f"05_build_iter_{i}_{j}.log")
                r = run_shell(command=c, cwd=ctx.repo_root, env=None, log_path=lp)
                ctx.logger.append_command(c, r.returncode, lp)
                build_logs.append(lp)
                if r.returncode != 0:
                    build_ok = False
                    break

            test_ok, test_logs = build_ok, []
            if build_ok:
                for j, c in enumerate(test_cmds, 1):
                    lp = ctx.logger.path(f"05_test_iter_{i}_{j}.log")
                    r = run_shell(command=c, cwd=ctx.repo_root, env=None, log_path=lp)
                    ctx.logger.append_command(c, r.returncode, lp)
                    test_logs.append(lp)
                    if r.returncode != 0:
                        test_ok = False
                        break

            # recall 检查
            recall_ok, recall_log = True, None
            if test_ok and recall_cmd:
                recall_log = ctx.logger.path(f"05_recall_iter_{i}.log")
                r = run_shell(command=recall_cmd, cwd=ctx.repo_root, env=None, log_path=recall_log, tee=True)
                ctx.logger.append_command(recall_cmd, r.returncode, recall_log)
                if r.returncode != 0:
                    recall_ok = False
                else:
                    recall_val = None
                    for m in recall_regex.finditer(r.output_text):
                        recall_val = float(m.group(1))
                    if recall_val is not None and recall_val < min_recall:
                        recall_ok = False
                        ctx.logger.write_text(f"05_recall_iter_{i}_fail.txt", f"recall={recall_val} < min={min_recall}\n")

            if test_ok and recall_ok:
                ctx.data["tests_passed"] = True
                ctx.logger.write_text("05_test_status.txt", f"PASS after iteration {i}\n")
                return

            # 失败时生成 retry prompt 文件
            last_log = recall_log if (recall_log and not recall_ok) else test_logs[-1] if test_logs else build_logs[-1] if build_logs else kiro_log
            err_tail = last_log.read_text(encoding="utf-8", errors="ignore").splitlines()[-120:]
            retry_content = retry_prompt_template.format(error_log="\n".join(err_tail))
            prompt_path = ctx.logger.path(f"05_kiro_retry_{i}.md")
            prompt_path.write_text(retry_content, encoding="utf-8")

        raise RuntimeError(f"tests did not pass after {max_iterations} iterations; see {ctx.logger.run_dir}")


class BenchmarkStep(Step):
    type_name = "benchmark"

    def run(self, ctx: WorkflowContext) -> None:
        warmup = int(self.cfg.get("warmup", 3))
        runs = int(self.cfg.get("runs", 10))
        command = str(self.cfg.get("command", "")).strip()
        if not command:
            raise ValueError("benchmark requires command")

        qps_re = re.compile(str(self.cfg.get("qps_regex", r"QPS:\s*([0-9.]+)")))
        recall_re = re.compile(str(self.cfg.get("recall_regex", r"recall.*:\s*([0-9.]+)")), re.IGNORECASE)

        for i in range(1, warmup + 1):
            lp = ctx.logger.path(f"06_bench_warmup_{i}.log")
            r = run_shell(command=command, cwd=ctx.repo_root, env=None, log_path=lp, tee=True)
            ctx.logger.append_command(command, r.returncode, lp)
            if r.returncode != 0:
                raise RuntimeError(f"benchmark warmup failed (rc={r.returncode}), see {lp}")

        results: list[dict[str, Any]] = []
        jsonl_path = ctx.logger.path("06_bench_results.jsonl")
        if jsonl_path.exists():
            jsonl_path.unlink()
        for i in range(1, runs + 1):
            lp = ctx.logger.path(f"06_bench_run_{i}.log")
            r = run_shell(command=command, cwd=ctx.repo_root, env=None, log_path=lp, tee=True)
            ctx.logger.append_command(command, r.returncode, lp)
            if r.returncode != 0:
                raise RuntimeError(f"benchmark run failed (rc={r.returncode}), see {lp}")

            qps = recall = None
            for m in qps_re.finditer(r.output_text):
                qps = float(m.group(1))
            for m in recall_re.finditer(r.output_text):
                recall = float(m.group(1))

            row = {"run": i, "qps": qps, "recall": recall, "log": lp.name}
            results.append(row)
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        qps_vals = [r["qps"] for r in results if isinstance(r.get("qps"), (int, float))]
        recall_vals = [r["recall"] for r in results if isinstance(r.get("recall"), (int, float))]
        summary = {
            "runs": runs,
            "qps": {"n": len(qps_vals), "mean": statistics.mean(qps_vals) if qps_vals else None,
                    "median": statistics.median(qps_vals) if qps_vals else None,
                    "stdev": statistics.pstdev(qps_vals) if len(qps_vals) > 1 else 0.0 if qps_vals else None},
            "recall": {"n": len(recall_vals), "mean": statistics.mean(recall_vals) if recall_vals else None,
                       "median": statistics.median(recall_vals) if recall_vals else None,
                       "stdev": statistics.pstdev(recall_vals) if len(recall_vals) > 1 else 0.0 if recall_vals else None},
        }
        json_dump(ctx.logger.path("06_bench_summary.json"), summary)
        ctx.data["benchmark_summary"] = summary


class CodexGitDecideStep(Step):
    type_name = "codex_git_decide"

    def run(self, ctx: WorkflowContext) -> None:
        codex_cmd = self.cfg.get("codex_cmd")
        if not isinstance(codex_cmd, list) or not codex_cmd:
            raise ValueError("codex_git_decide requires codex_cmd=[...]")
        if which(codex_cmd[0]) is None:
            raise RuntimeError(f"codex not found in PATH: {codex_cmd[0]}")

        mode = str(self.cfg.get("mode", "codex"))
        # 使用历史最佳作为比较基准（而非上一轮，因为上一轮可能被回滚）
        best, cur = ctx.state.get("best_benchmark_summary"), ctx.data.get("benchmark_summary")
        # 如果没有历史最佳数据，使用配置的初始基准值
        if not best:
            init_qps = ctx.workflow_cfg.get("initial_qps", 0.0)
            init_recall = ctx.workflow_cfg.get("initial_recall", 0.85)
            best = {"qps": {"mean": init_qps}, "recall": {"mean": init_recall}, "_initial": True}
        head = git("git rev-parse HEAD", ctx.repo_root)
        diff_stat = git("git diff --stat", ctx.repo_root)
        status = git("git status --porcelain", ctx.repo_root)

        ctx.logger.write_text("07_git_diff_stat.txt", diff_stat + "\n")
        ctx.logger.write_text("07_git_status.txt", status + "\n")

        # 根据 mode 选择不同的 prompt 文件
        prompt_key = "summarize_prompt_file" if mode == "local" else "prompt_file"
        prompt_path = ctx.repo_root / self.cfg.get(prompt_key, "")
        if not prompt_path.exists():
            raise FileNotFoundError(f"prompt file not found: {prompt_path}")
        prompt_template = prompt_path.read_text(encoding="utf-8")

        if mode == "local":
            codex_prompt = prompt_template.format(
                best_summary=json.dumps(best, ensure_ascii=False, indent=2) if best else "null",
                cur_summary=json.dumps(cur, ensure_ascii=False, indent=2) if cur else "null",
                diff_stat=diff_stat or "(no diff)",
                run_dir=ctx.logger.run_dir,
            )
        else:
            codex_prompt = prompt_template.format(
                head=head,
                best_summary=json.dumps(best, ensure_ascii=False, indent=2) if best else "null",
                cur_summary=json.dumps(cur, ensure_ascii=False, indent=2) if cur else "null",
                diff_stat=diff_stat or "(no diff)",
                status=status or "(clean)",
                run_dir=ctx.logger.run_dir,
                mode=mode,
            )

        lp = ctx.logger.path("07_codex_git_decide.log")
        res = run_agent(cmd=codex_cmd, prompt=codex_prompt, cwd=ctx.repo_root, env=None, log_path=lp)
        ctx.logger.append_command(" ".join(codex_cmd), res.returncode, lp)
        if res.returncode != 0:
            raise RuntimeError(f"codex git decide failed (rc={res.returncode}), see {lp}")

        ctx.logger.write_text("07_codex_git_decide.output.txt", res.output_text.strip() + "\n")

        # local 模式：python 执行 git 决策
        if mode == "local":
            outcome = self._decide_and_execute_git(ctx, best, cur, diff_stat, status)
        else:
            outcome = None

        ctx.state["last_git_head"] = git("git rev-parse HEAD", ctx.repo_root)
        # 只有 commit 时才更新历史最佳
        if cur and outcome == "commit":
            ctx.state["best_benchmark_summary"] = cur
            # commit 成功后运行 profile 生成命令
            self._run_profile_after_commit(ctx)
        # 保存 decision 供下一轮参考
        decision_path = ctx.logger.path("07_decision.md")
        decision_text = decision_path.read_text(encoding="utf-8") if decision_path.exists() else ""
        if decision_text:
            ctx.state["last_decision"] = decision_text

        # 追加 changelog 条目
        self._append_changelog(ctx, cur, best, diff_stat, decision_text, outcome)

    def _run_profile_after_commit(self, ctx: WorkflowContext) -> None:
        """commit 成功后运行 profile 生成命令"""
        profile_cmd = self.cfg.get("profile_cmd")
        if not profile_cmd:
            return
        cmd = profile_cmd if isinstance(profile_cmd, str) else " ".join(profile_cmd)
        lp = ctx.logger.path("07_profile_after_commit.log")
        print(f"[profile] running profile after commit: {cmd}")
        run_shell(command=cmd, cwd=ctx.repo_root, env=None, log_path=lp, tee=True)

    def _decide_and_execute_git(self, ctx: WorkflowContext, best: dict | None, cur: dict | None, diff_stat: str, status: str) -> str:
        """根据 benchmark 结果决定 commit 或 checkout，返回 outcome"""
        min_recall = float(self.cfg.get("min_recall", 0.85))
        cur_qps = cur.get("qps", {}).get("median") if cur else None
        cur_recall = cur.get("recall", {}).get("mean") if cur else None
        best_qps = best.get("qps", {}).get("median") if best else None

        # 决策逻辑：recall >= min_recall 且 qps 超过历史最佳则 commit
        should_commit = (
            cur_qps is not None and cur_recall is not None and
            cur_recall >= min_recall and
            (best_qps is None or cur_qps >= best_qps)
        )

        if should_commit and status.strip():
            # 执行 commit
            git("git add -A", ctx.repo_root)
            msg = f"perf: QPS median {best_qps:.1f} -> {cur_qps:.1f}, recall {cur_recall:.3f}"
            git(f'git commit -m "{msg}"', ctx.repo_root)
            ctx.logger.write_text("07_git_action.txt", f"COMMIT: {msg}\n")
            return "commit"
        elif status.strip():
            # 执行 checkout
            git("git checkout -- .", ctx.repo_root)
            ctx.logger.write_text("07_git_action.txt", f"CHECKOUT: QPS {cur_qps}, recall {cur_recall}\n")
            return "checkout"
        else:
            ctx.logger.write_text("07_git_action.txt", "NO_CHANGE: working tree clean\n")
            return "no_change"

    def _append_changelog(self, ctx: WorkflowContext, cur: dict | None, prev: dict | None, diff_stat: str, decision: str, outcome: str | None = None) -> None:
        from datetime import datetime
        changelog = load_changelog(ctx.state_file)
        # 读取 changes_summary（kiro 生成的修改摘要）
        summary_path = ctx.logger.path("05_changes_summary.md")
        changes_summary = summary_path.read_text(encoding="utf-8").strip() if summary_path.exists() else ""
        # 解析 decision 中的 outcome（commit/checkout），如果未传入则从 decision 文本推断
        if outcome is None:
            outcome = "checkout"
            if decision:
                if "commit" in decision.lower() and "不 commit" not in decision and "不commit" not in decision:
                    outcome = "commit"
        # 提取修改的文件列表
        files_changed = [line.split("|")[0].strip() for line in diff_stat.splitlines() if "|" in line]
        # 构建 changelog 条目
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


# Step 注册表
STEP_REGISTRY: dict[str, type[Step]] = {
    cls.type_name: cls for cls in [
        CommandStep, ProfileAnalyzeStep, TargetSelectStep, ContextCollectStep,
        CodexGenerateKiroPromptStep, KiroApplyAndTestStep, BenchmarkStep, CodexGitDecideStep,
    ]
}
