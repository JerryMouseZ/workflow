"""Step 基类和所有 Step 实现"""
from __future__ import annotations

import json
import re
import statistics
from typing import TYPE_CHECKING, Any

from .utils import (
    collect_context_md, filter_profile_entries, git, json_dump, latest_folded_file,
    parse_self_time_table, pick_targets, run_agent, run_cmd, run_shell, which,
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
            raise RuntimeError("no .folded or .perf.data files found under test/profile_output/")

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


class TargetSelectStep(Step):
    type_name = "target_select"

    def run(self, ctx: WorkflowContext) -> None:
        filtered = ctx.data.get("profile", {}).get("filtered") or []
        strategy = str(self.cfg.get("strategy", "top_self_percent"))
        pick_top_k = int(self.cfg.get("pick_top_k", 3))
        targets = pick_targets(filtered, strategy=strategy, k=pick_top_k)
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
        targets = ctx.data["targets"]
        profile_filtered = ctx.data["profile"]["filtered"][:20]
        context_md = ctx.logger.path(ctx.data["context"]["md_path"]).read_text(encoding="utf-8")[:max_context_chars]

        codex_prompt = prompt_template.format(
            target_lines="\n".join([f"- {t['function']} (self%={t['self_pct']}, samples={t['samples']})" for t in targets]),
            prof_lines="\n".join([f"- {e['function']} (self%={e['self_pct']}, samples={e['samples']})" for e in profile_filtered]),
            context_md=context_md,
        )

        log_path = ctx.logger.path("04_codex_generate_kiro_prompt.log")
        res = run_agent(cmd=codex_cmd, prompt=codex_prompt, cwd=ctx.repo_root, env=None, log_path=log_path)
        ctx.logger.append_command(" ".join(codex_cmd), res.returncode, log_path)
        if res.returncode != 0:
            raise RuntimeError(f"codex exec failed (rc={res.returncode}), see {log_path}")

        ctx.logger.write_text("04_kiro_prompt.txt", res.output_text.strip() + "\n")
        ctx.data["kiro_prompt"] = res.output_text


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
        prompt = str(ctx.data.get("kiro_prompt", "")).strip()
        if not prompt:
            raise RuntimeError("missing kiro_prompt from previous step")

        for i in range(1, max_iterations + 1):
            kiro_log = ctx.logger.path(f"05_kiro_iter_{i}.log")
            res = run_agent(cmd=kiro_cmd, prompt=prompt + "\n", cwd=ctx.repo_root, env=None, log_path=kiro_log)
            ctx.logger.append_command(" ".join(kiro_cmd), res.returncode, kiro_log)
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

            if test_ok:
                ctx.data["tests_passed"] = True
                ctx.logger.write_text("05_test_status.txt", f"PASS after iteration {i}\n")
                return

            last_log = test_logs[-1] if test_logs else build_logs[-1] if build_logs else kiro_log
            err_tail = last_log.read_text(encoding="utf-8", errors="ignore").splitlines()[-120:]
            prompt = retry_prompt_template.format(error_log="\n".join(err_tail))

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

        prompt_path = ctx.repo_root / self.cfg.get("prompt_file", "")
        if not prompt_path.exists():
            raise FileNotFoundError(f"prompt file not found: {prompt_path}")
        prompt_template = prompt_path.read_text(encoding="utf-8")

        mode = str(self.cfg.get("mode", "codex"))
        prev, cur = ctx.state.get("last_benchmark_summary"), ctx.data.get("benchmark_summary")
        head = git("git rev-parse HEAD", ctx.repo_root)
        diff_stat = git("git diff --stat", ctx.repo_root)
        status = git("git status --porcelain", ctx.repo_root)

        ctx.logger.write_text("07_git_diff_stat.txt", diff_stat + "\n")
        ctx.logger.write_text("07_git_status.txt", status + "\n")

        codex_prompt = prompt_template.format(
            head=head,
            prev_summary=json.dumps(prev, ensure_ascii=False, indent=2) if prev else "null",
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
        ctx.state["last_git_head"] = git("git rev-parse HEAD", ctx.repo_root)
        if cur:
            ctx.state["last_benchmark_summary"] = cur


# Step 注册表
STEP_REGISTRY: dict[str, type[Step]] = {
    cls.type_name: cls for cls in [
        CommandStep, ProfileAnalyzeStep, TargetSelectStep, ContextCollectStep,
        CodexGenerateKiroPromptStep, KiroApplyAndTestStep, BenchmarkStep, CodexGitDecideStep,
    ]
}
