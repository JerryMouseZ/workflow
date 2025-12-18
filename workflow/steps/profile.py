"""ProfileAnalyzeStep 实现"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import Step
from ..utils import (
    filter_profile_entries, json_dump, latest_folded_file,
    parse_self_time_table, run_cmd, run_shell,
)

if TYPE_CHECKING:
    from ..runner import WorkflowContext


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
        """当首次运行时 test/profile_output 为空时，先跑 benchmark 再跑 profile"""

        def _as_shell_cmd(v: Any) -> str:
            if isinstance(v, str):
                return v.strip()
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                return " ".join(x.strip() for x in v if x.strip())
            return ""

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

        if not profile_cmd and bench_cmd and "run_benchmark.sh" in bench_cmd and "--profile" not in bench_cmd:
            profile_cmd = bench_cmd + " --profile"

        if not bench_cmd or not profile_cmd:
            raise RuntimeError(
                "missing profile artifacts and cannot infer bootstrap commands; "
                "set profile_analyze.bootstrap_bench_cmd/bootstrap_profile_cmd in config.toml"
            )

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
