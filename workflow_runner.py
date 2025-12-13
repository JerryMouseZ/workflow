from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    if "workflow" not in cfg:
        raise ValueError("config missing [workflow]")
    if "steps" not in cfg or not isinstance(cfg["steps"], list):
        raise ValueError("config missing [[steps]] list")
    return cfg


def _now_id() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _latest_run_id(log_dir: Path, *, round_idx: int) -> str | None:
    if not log_dir.exists():
        return None
    suffix = f"_r{round_idx}"
    candidates: list[str] = []
    for p in log_dir.iterdir():
        if p.is_dir() and p.name.endswith(suffix):
            candidates.append(p.name)
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclasses.dataclass
class CmdResult:
    returncode: int
    output_text: str


class RunLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.commands_log = self.run_dir / "commands.log"

    def path(self, name: str) -> Path:
        return self.run_dir / name

    def write_text(self, name: str, text: str) -> Path:
        p = self.path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def append_command(self, cmd_display: str, returncode: int, log_path: Path) -> None:
        ts = _dt.datetime.now().isoformat(timespec="seconds")
        with self.commands_log.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] rc={returncode} cmd={cmd_display} log={log_path}\n")


def _run_cmd(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str] | None,
    stdin_text: str | None,
    log_path: Path,
    tee: bool = True,
) -> CmdResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdin=subprocess.PIPE if stdin_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    if stdin_text is not None:
        assert proc.stdin is not None
        proc.stdin.write(stdin_text)
        proc.stdin.close()

    out_chunks: list[str] = []
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write(f"$ {' '.join(cmd)}\n\n")
        for line in proc.stdout:
            out_chunks.append(line)
            lf.write(line)
            if tee:
                sys.stdout.write(line)
        proc.wait()

    return CmdResult(returncode=proc.returncode, output_text="".join(out_chunks))


def _run_shell(
    *,
    command: str,
    cwd: Path,
    env: dict[str, str] | None,
    log_path: Path,
    tee: bool = True,
) -> CmdResult:
    return _run_cmd(
        cmd=["bash", "-lc", command],
        cwd=cwd,
        env=env,
        stdin_text=None,
        log_path=log_path,
        tee=tee,
    )


def _run_agent(
    *,
    cmd: list[str],
    prompt: str,
    cwd: Path,
    env: dict[str, str] | None,
    log_path: Path,
    tee: bool = True,
) -> CmdResult:
    return _run_cmd(cmd=cmd, cwd=cwd, env=env, stdin_text=prompt, log_path=log_path, tee=tee)


def _git(cmd: str, repo_root: Path) -> str:
    out = subprocess.check_output(["bash", "-lc", cmd], cwd=str(repo_root), text=True)
    return out.strip()


def _which(exe: str) -> str | None:
    return shutil.which(exe)


def _latest_folded_file(repo_root: Path) -> Path | None:
    """获取最新的 .folded 或 .perf.data 文件，与 analyze_self_time.py 保持一致"""
    prof_dir = repo_root / "test" / "profile_output"
    if not prof_dir.exists():
        return None
    all_files = list(prof_dir.glob("*.folded")) + list(prof_dir.glob("*.perf.data"))
    if not all_files:
        return None
    return max(all_files, key=lambda p: p.stat().st_mtime)


def _parse_self_time_table(text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    in_table = False
    has_samples_col = False
    for line in text.splitlines():
        if line.strip().startswith("Rank") and "Function" in line:
            in_table = True
            has_samples_col = "Samples" in line
            continue
        if in_table and re.fullmatch(r"-{10,}", line.strip()):
            continue
        if in_table and line.strip().startswith("Top "):
            break
        if not in_table:
            continue
        if has_samples_col:
            # folded: "1  12.34%  12345  Function"
            m = re.match(r"^\s*(\d+)\s+(\d+(?:\.\d+)?)%\s+(\d+)\s+(.*)$", line)
            if not m:
                continue
            rank = int(m.group(1))
            self_pct = float(m.group(2))
            samples = int(m.group(3))
            func = m.group(4).strip()
        else:
            # perf.data: "1  12.34%  Function"
            m = re.match(r"^\s*(\d+)\s+(\d+(?:\.\d+)?)%\s+(.*)$", line)
            if not m:
                continue
            rank = int(m.group(1))
            self_pct = float(m.group(2))
            samples = 0
            func = m.group(3).strip()
        entries.append({"rank": rank, "self_pct": self_pct, "samples": samples, "function": func})
    return entries


def _filter_profile_entries(entries: list[dict[str, Any]], ignore_regex: list[str]) -> list[dict[str, Any]]:
    regs = [re.compile(p) for p in ignore_regex]
    out: list[dict[str, Any]] = []
    for e in entries:
        func = str(e["function"])
        if any(r.search(func) for r in regs):
            continue
        out.append(e)
    return out


def _pick_targets(entries: list[dict[str, Any]], strategy: str, k: int) -> list[dict[str, Any]]:
    if not entries:
        return []
    if strategy == "top_samples":
        ranked = sorted(entries, key=lambda x: (-int(x["samples"]), float(x["self_pct"])))
    else:  # top_self_percent
        ranked = sorted(entries, key=lambda x: (-float(x["self_pct"]), -int(x["samples"])))
    return ranked[: max(1, k)]


def _rg_matches(repo_root: Path, roots: list[str], needle: str, max_total: int) -> list[dict[str, Any]]:
    rg = _which("rg")
    matches: list[dict[str, Any]] = []
    if rg:
        pattern = rf"\\b{re.escape(needle)}\\b"
        cmd = ["rg", "-n", "-S", "--no-heading", "--glob", "!.git/**", pattern] + roots
        proc = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)
        for line in proc.stdout.splitlines():
            # path:line:content
            parts = line.split(":", 2)
            if len(parts) != 3:
                continue
            p, ln, content = parts
            try:
                lineno = int(ln)
            except ValueError:
                continue
            matches.append({"path": p, "line": lineno, "content": content})
            if len(matches) >= max_total:
                break
        return matches

    # fallback
    for root in roots:
        base = repo_root / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if needle in line:
                    matches.append({"path": str(path.relative_to(repo_root)), "line": i, "content": line})
                    if len(matches) >= max_total:
                        return matches
    return matches


_C_DEF_RE = re.compile(
    r"^\s*(?!if\b|for\b|while\b|switch\b|return\b)"
    r"(?:[A-Za-z_]\w*[\w\s\*\(\)]*\s+)?"
    r"([A-Za-z_]\w*)\s*\([^;]*\)\s*\{\s*$"
)


def _enclosing_c_function(lines: list[str], one_based_line: int, lookback: int = 250) -> str | None:
    idx = max(0, one_based_line - 1)
    start = max(0, idx - lookback)
    for j in range(idx, start - 1, -1):
        m = _C_DEF_RE.match(lines[j])
        if m:
            return m.group(1)
    return None


def _snippet(path: Path, line: int, radius: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    start = max(1, line - radius)
    end = min(len(lines), line + radius)
    out_lines = []
    for ln in range(start, end + 1):
        prefix = ">>" if ln == line else "  "
        out_lines.append(f"{prefix}{ln:6d} {lines[ln-1]}")
    return "\n".join(out_lines)


def _collect_context_md(
    *,
    repo_root: Path,
    target_func: str,
    roots: list[str],
    snippet_radius: int,
    max_files: int,
    max_total_matches: int,
) -> tuple[str, list[dict[str, Any]]]:
    matches = _rg_matches(repo_root, roots, target_func, max_total_matches)
    by_file: dict[str, list[dict[str, Any]]] = {}
    for m in matches:
        by_file.setdefault(m["path"], []).append(m)
    ranked_files = sorted(by_file.items(), key=lambda kv: -len(kv[1]))[:max_files]

    callers: dict[str, int] = {}
    md_lines = [f"# Context for `{target_func}`", ""]
    md_lines.append("## Candidate files")
    for p, ms in ranked_files:
        md_lines.append(f"- `{p}` (matches={len(ms)})")
    md_lines.append("")

    md_lines.append("## Call-site hints (heuristic)")
    for p, ms in ranked_files:
        abs_p = repo_root / p
        try:
            file_lines = abs_p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for m in ms[:10]:
            caller = _enclosing_c_function(file_lines, int(m["line"]))
            if caller:
                callers[caller] = callers.get(caller, 0) + 1
    for caller, cnt in sorted(callers.items(), key=lambda kv: -kv[1])[:20]:
        md_lines.append(f"- `{caller} -> {target_func}` (hits={cnt})")
    md_lines.append("")

    md_lines.append("## Snippets")
    for p, ms in ranked_files:
        abs_p = repo_root / p
        md_lines.append(f"### `{p}`")
        for m in ms[:5]:
            md_lines.append(f"\n**match @ L{m['line']}**\n")
            md_lines.append("```")
            md_lines.append(_snippet(abs_p, int(m["line"]), snippet_radius))
            md_lines.append("```")
        md_lines.append("")

    return "\n".join(md_lines), matches


class Step:
    type_name: str = ""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def run(self, ctx: "WorkflowContext") -> None:  # pragma: no cover
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
        res = _run_shell(command=command, cwd=ctx.repo_root, env=None, log_path=log_path, tee=tee)
        ctx.logger.append_command(command, res.returncode, log_path)
        if int(self.cfg.get("require_success", 1)) == 1 and res.returncode != 0:
            raise RuntimeError(f"command step failed (rc={res.returncode}): {command}, see {log_path}")
        ctx.data.setdefault("commands", {})[name] = {"returncode": res.returncode, "log": str(log_path)}


@dataclasses.dataclass
class WorkflowContext:
    repo_root: Path
    run_id: str
    round_idx: int
    logger: RunLogger
    state_file: Path
    state: dict[str, Any]
    data: dict[str, Any]


class ProfileAnalyzeStep(Step):
    type_name = "profile_analyze"

    def run(self, ctx: WorkflowContext) -> None:
        analyze_cmd = self.cfg.get("analyze_cmd")
        top_n = int(self.cfg.get("top_n", 20))
        ignore_regex = list(self.cfg.get("ignore_regex", [r"^\[.*\]$", r"(?i)unknown"]))
        if not isinstance(analyze_cmd, list) or not analyze_cmd:
            raise ValueError("profile_analyze requires analyze_cmd=[...]")

        latest = _latest_folded_file(ctx.repo_root)
        if latest is None:
            raise RuntimeError("no .folded or .perf.data files found under test/profile_output/")

        log_path = ctx.logger.path("01_profile_analyze.log")
        res = _run_cmd(
            cmd=analyze_cmd + [str(latest), str(top_n)],
            cwd=ctx.repo_root,
            env=None,
            stdin_text=None,
            log_path=log_path,
        )
        ctx.logger.append_command(" ".join(analyze_cmd), res.returncode, log_path)
        if res.returncode != 0:
            raise RuntimeError(f"profile analyze failed (rc={res.returncode}), see {log_path}")

        entries = _parse_self_time_table(res.output_text)
        filtered = _filter_profile_entries(entries, ignore_regex)

        _json_dump(ctx.logger.path("01_profile_entries.raw.json"), entries)
        _json_dump(ctx.logger.path("01_profile_entries.filtered.json"), filtered)
        ctx.data["profile"] = {"raw": entries, "filtered": filtered, "ignore_regex": ignore_regex}


class TargetSelectStep(Step):
    type_name = "target_select"

    def run(self, ctx: WorkflowContext) -> None:
        prof = ctx.data.get("profile", {})
        filtered = prof.get("filtered") or []
        strategy = str(self.cfg.get("strategy", "top_self_percent"))
        pick_top_k = int(self.cfg.get("pick_top_k", 3))
        targets = _pick_targets(filtered, strategy=strategy, k=pick_top_k)
        if not targets:
            raise RuntimeError("no target functions after filtering; adjust ignore_regex or ensure profile exists")
        ctx.data["targets"] = targets
        ctx.logger.write_text(
            "02_targets.txt",
            "\n".join([f"{t['function']}  self%={t['self_pct']} samples={t['samples']}" for t in targets]) + "\n",
        )


class ContextCollectStep(Step):
    type_name = "context_collect"

    def run(self, ctx: WorkflowContext) -> None:
        targets = ctx.data["targets"]
        target_func = str(targets[0]["function"])
        roots = list(self.cfg.get("search_roots", ["polardb"]))
        snippet_radius = int(self.cfg.get("snippet_radius", 50))
        max_files = int(self.cfg.get("max_files", 20))
        max_total_matches = int(self.cfg.get("max_total_matches", 120))

        md, matches = _collect_context_md(
            repo_root=ctx.repo_root,
            target_func=target_func,
            roots=roots,
            snippet_radius=snippet_radius,
            max_files=max_files,
            max_total_matches=max_total_matches,
        )
        ctx.logger.write_text("03_context.md", md)
        _json_dump(ctx.logger.path("03_context.matches.json"), matches)
        ctx.data["context"] = {"target_func": target_func, "roots": roots, "md_path": "03_context.md"}


class CodexGenerateKiroPromptStep(Step):
    type_name = "codex_generate_kiro_prompt"

    def run(self, ctx: WorkflowContext) -> None:
        codex_cmd = self.cfg.get("codex_cmd")
        if not isinstance(codex_cmd, list) or not codex_cmd:
            raise ValueError("codex_generate_kiro_prompt requires codex_cmd=[...]")
        if _which(codex_cmd[0]) is None:
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
        context_md = (ctx.logger.path(ctx.data["context"]["md_path"]).read_text(encoding="utf-8"))[:max_context_chars]

        target_lines = "\n".join(
            [f"- {t['function']} (self%={t['self_pct']}, samples={t['samples']})" for t in targets]
        )
        prof_lines = "\n".join(
            [f"- {e['function']} (self%={e['self_pct']}, samples={e['samples']})" for e in profile_filtered]
        )

        codex_prompt = prompt_template.format(
            target_lines=target_lines,
            prof_lines=prof_lines,
            context_md=context_md,
        )

        log_path = ctx.logger.path("04_codex_generate_kiro_prompt.log")
        res = _run_agent(cmd=codex_cmd, prompt=codex_prompt, cwd=ctx.repo_root, env=None, log_path=log_path)
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
        if _which(kiro_cmd[0]) is None:
            raise RuntimeError(f"kiro-cli not found in PATH: {kiro_cmd[0]}")

        retry_prompt_file = self.cfg.get("retry_prompt_file")
        if not retry_prompt_file:
            raise ValueError("kiro_apply_and_test requires retry_prompt_file")
        retry_prompt_path = ctx.repo_root / retry_prompt_file
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
            res = _run_agent(cmd=kiro_cmd, prompt=prompt + "\n", cwd=ctx.repo_root, env=None, log_path=kiro_log)
            ctx.logger.append_command(" ".join(kiro_cmd), res.returncode, kiro_log)
            if res.returncode != 0:
                raise RuntimeError(f"kiro-cli failed (rc={res.returncode}), see {kiro_log}")

            # build
            build_ok = True
            build_logs: list[Path] = []
            for j, c in enumerate(build_cmds, 1):
                lp = ctx.logger.path(f"05_build_iter_{i}_{j}.log")
                r = _run_shell(command=c, cwd=ctx.repo_root, env=None, log_path=lp)
                ctx.logger.append_command(c, r.returncode, lp)
                build_logs.append(lp)
                if r.returncode != 0:
                    build_ok = False
                    break

            # tests
            test_ok = build_ok
            test_logs: list[Path] = []
            if build_ok:
                for j, c in enumerate(test_cmds, 1):
                    lp = ctx.logger.path(f"05_test_iter_{i}_{j}.log")
                    r = _run_shell(command=c, cwd=ctx.repo_root, env=None, log_path=lp)
                    ctx.logger.append_command(c, r.returncode, lp)
                    test_logs.append(lp)
                    if r.returncode != 0:
                        test_ok = False
                        break

            if test_ok:
                ctx.data["tests_passed"] = True
                ctx.logger.write_text("05_test_status.txt", f"PASS after iteration {i}\n")
                return

            # failure: craft retry prompt
            last_log = (test_logs[-1] if test_logs else build_logs[-1] if build_logs else kiro_log)
            err_tail = last_log.read_text(encoding="utf-8", errors="ignore").splitlines()[-120:]
            prompt = retry_prompt_template.format(error_log="\n".join(err_tail))

        raise RuntimeError(f"tests did not pass after {max_iterations} iterations; see workflow logs in {ctx.logger.run_dir}")


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

        # warmup
        for i in range(1, warmup + 1):
            lp = ctx.logger.path(f"06_bench_warmup_{i}.log")
            r = _run_shell(command=command, cwd=ctx.repo_root, env=None, log_path=lp, tee=True)
            ctx.logger.append_command(command, r.returncode, lp)
            if r.returncode != 0:
                raise RuntimeError(f"benchmark warmup failed (rc={r.returncode}), see {lp}")

        # measured runs
        results: list[dict[str, Any]] = []
        jsonl_path = ctx.logger.path("06_bench_results.jsonl")
        if jsonl_path.exists():
            jsonl_path.unlink()
        for i in range(1, runs + 1):
            lp = ctx.logger.path(f"06_bench_run_{i}.log")
            r = _run_shell(command=command, cwd=ctx.repo_root, env=None, log_path=lp, tee=True)
            ctx.logger.append_command(command, r.returncode, lp)
            if r.returncode != 0:
                raise RuntimeError(f"benchmark run failed (rc={r.returncode}), see {lp}")

            qps = None
            recall = None
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
            "qps": {
                "n": len(qps_vals),
                "mean": statistics.mean(qps_vals) if qps_vals else None,
                "median": statistics.median(qps_vals) if qps_vals else None,
                "stdev": statistics.pstdev(qps_vals) if len(qps_vals) > 1 else 0.0 if qps_vals else None,
            },
            "recall": {
                "n": len(recall_vals),
                "mean": statistics.mean(recall_vals) if recall_vals else None,
                "median": statistics.median(recall_vals) if recall_vals else None,
                "stdev": statistics.pstdev(recall_vals) if len(recall_vals) > 1 else 0.0 if recall_vals else None,
            },
        }
        _json_dump(ctx.logger.path("06_bench_summary.json"), summary)
        ctx.data["benchmark_summary"] = summary


class CodexGitDecideStep(Step):
    type_name = "codex_git_decide"

    def run(self, ctx: WorkflowContext) -> None:
        codex_cmd = self.cfg.get("codex_cmd")
        if not isinstance(codex_cmd, list) or not codex_cmd:
            raise ValueError("codex_git_decide requires codex_cmd=[...]")
        if _which(codex_cmd[0]) is None:
            raise RuntimeError(f"codex not found in PATH: {codex_cmd[0]}")

        prompt_file = self.cfg.get("prompt_file")
        if not prompt_file:
            raise ValueError("codex_git_decide requires prompt_file")
        prompt_path = ctx.repo_root / prompt_file
        if not prompt_path.exists():
            raise FileNotFoundError(f"prompt file not found: {prompt_path}")
        prompt_template = prompt_path.read_text(encoding="utf-8")

        mode = str(self.cfg.get("mode", "codex"))
        prev = ctx.state.get("last_benchmark_summary")
        cur = ctx.data.get("benchmark_summary")
        head = _git("git rev-parse HEAD", ctx.repo_root)
        diff_stat = _git("git diff --stat", ctx.repo_root)
        status = _git("git status --porcelain", ctx.repo_root)

        ctx.logger.write_text("07_git_diff_stat.txt", diff_stat + "\n")
        ctx.logger.write_text("07_git_status.txt", status + "\n")

        codex_prompt = prompt_template.format(
            head=head,
            prev_summary=json.dumps(prev, ensure_ascii=False, indent=2) if prev else "null",
            cur_summary=json.dumps(cur, ensure_ascii=False, indent=2) if cur else "null",
            diff_stat=diff_stat if diff_stat else "(no diff)",
            status=status if status else "(clean)",
            run_dir=ctx.logger.run_dir,
            mode=mode,
        )

        lp = ctx.logger.path("07_codex_git_decide.log")
        res = _run_agent(cmd=codex_cmd, prompt=codex_prompt, cwd=ctx.repo_root, env=None, log_path=lp)
        ctx.logger.append_command(" ".join(codex_cmd), res.returncode, lp)
        if res.returncode != 0:
            raise RuntimeError(f"codex git decide failed (rc={res.returncode}), see {lp}")

        ctx.logger.write_text("07_codex_git_decide.output.txt", res.output_text.strip() + "\n")

        # 更新 state：不依赖 codex 是否真的执行了 git
        new_head = _git("git rev-parse HEAD", ctx.repo_root)
        ctx.state["last_git_head"] = new_head
        if cur:
            ctx.state["last_benchmark_summary"] = cur


STEP_REGISTRY: dict[str, type[Step]] = {
    CommandStep.type_name: CommandStep,
    ProfileAnalyzeStep.type_name: ProfileAnalyzeStep,
    TargetSelectStep.type_name: TargetSelectStep,
    ContextCollectStep.type_name: ContextCollectStep,
    CodexGenerateKiroPromptStep.type_name: CodexGenerateKiroPromptStep,
    KiroApplyAndTestStep.type_name: KiroApplyAndTestStep,
    BenchmarkStep.type_name: BenchmarkStep,
    CodexGitDecideStep.type_name: CodexGitDecideStep,
}


class WorkflowRunner:
    def __init__(
        self,
        *,
        repo_root: Path,
        config: dict[str, Any],
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
            if item is None:
                continue
            for tok in str(item).split(","):
                tok = tok.strip()
                if tok:
                    out.append(tok)
        return out

    @staticmethod
    def _matches_selector(*, step_cfg: dict[str, Any], idx: int, selector: str) -> bool:
        sel = selector.strip()
        if not sel:
            return False

        step_type = str(step_cfg.get("type", "")).strip()
        step_name = str(step_cfg.get("name", "")).strip()

        def norm_prefix(s: str) -> tuple[str, str] | None:
            for p in ("type:", "type=", "name:", "name=", "idx:", "idx="):
                if s.startswith(p):
                    return p[:-1], s[len(p) :].strip()
            return None

        if sel.startswith("#"):
            return sel[1:].strip().isdigit() and int(sel[1:].strip()) == idx

        pref = norm_prefix(sel)
        if pref is not None:
            key, value = pref
            if key == "type":
                return value == step_type
            if key == "name":
                return value == step_name
            if key == "idx":
                return value.isdigit() and int(value) == idx
            return False

        if sel.isdigit() and int(sel) == idx:
            return True

        # plain token: match name or type
        return sel == step_name or sel == step_type

    def _should_run_step(self, *, step_cfg: dict[str, Any], idx: int) -> tuple[bool, str | None]:
        enabled = step_cfg.get("enabled", True)
        if enabled is False:
            return False, "disabled (enabled=false)"

        if self.only_selectors:
            ok = any(self._matches_selector(step_cfg=step_cfg, idx=idx, selector=s) for s in self.only_selectors)
            if not ok:
                return False, "not selected (--only)"

        if self.skip_selectors:
            hit = any(self._matches_selector(step_cfg=step_cfg, idx=idx, selector=s) for s in self.skip_selectors)
            if hit:
                return False, "skipped (--skip)"

        return True, None

    def format_steps(self) -> str:
        lines = ["idx  enabled  type                      name"]
        for i, s in enumerate(self.steps_cfg, 1):
            enabled = s.get("enabled", True)
            t = str(s.get("type", "")).strip()
            n = str(s.get("name", "")).strip()
            lines.append(f"{i:>3}  {str(bool(enabled)).lower():<7}  {t:<24}  {n}")
        return "\n".join(lines)

    def format_step_plan(self) -> str:
        lines = ["idx  action   type                      name  reason"]
        for i, s in enumerate(self.steps_cfg, 1):
            run, reason = self._should_run_step(step_cfg=s, idx=i)
            action = "run" if run else "skip"
            t = str(s.get("type", "")).strip()
            n = str(s.get("name", "")).strip()
            lines.append(f"{i:>3}  {action:<6}  {t:<24}  {n:<20}  {reason or ''}".rstrip())
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

        # resume: 获取上次完成的步骤索引和数据
        last_completed_idx = state.get("last_completed_step_idx", 0) if self.resume else 0
        persisted_data = state.get("data", {}) if self.resume else {}

        for round_idx in range(1, rounds + 1):
            # resume: 使用同一个 run_id/log 目录，避免日志分散在多个目录
            run_id: str
            if self.resume:
                active_run_id = state.get("active_run_id")
                active_round_idx = state.get("active_round_idx")
                if (
                    isinstance(active_run_id, str)
                    and isinstance(active_round_idx, int)
                    and active_round_idx == round_idx
                    and active_run_id.endswith(f"_r{round_idx}")
                ):
                    run_id = active_run_id
                elif last_completed_idx > 0:
                    run_id = _latest_run_id(log_dir, round_idx=round_idx) or f"{_now_id()}_r{round_idx}"
                else:
                    run_id = f"{_now_id()}_r{round_idx}"
            else:
                run_id = f"{_now_id()}_r{round_idx}"
            run_dir = log_dir / run_id
            logger = RunLogger(run_dir)

            # 记录当前活跃 run，供 --resume 继续使用同一个日志目录
            state["active_run_id"] = run_id
            state["active_round_idx"] = round_idx
            _json_dump(state_file, state)

            env_name = "00_env.json"
            if self.resume and logger.path(env_name).exists():
                env_name = f"00_env_resume_{_now_id()}.json"
            env_info = {
                "run_id": run_id,
                "round": round_idx,
                "cwd": str(self.repo_root),
                "git_head": _git("git rev-parse HEAD", self.repo_root),
                "skip_selectors": self.skip_selectors,
                "only_selectors": self.only_selectors,
                "resume": self.resume,
                "resume_from_idx": last_completed_idx + 1 if self.resume and last_completed_idx > 0 else None,
            }
            _json_dump(logger.path(env_name), env_info)

            ctx = WorkflowContext(
                repo_root=self.repo_root,
                run_id=run_id,
                round_idx=round_idx,
                logger=logger,
                state_file=state_file,
                state=state,
                data=dict(persisted_data),
            )

            step_plan: list[dict[str, Any]] = []
            for idx, s_cfg in enumerate(self.steps_cfg, 1):
                t = s_cfg.get("type")
                if t not in STEP_REGISTRY:
                    raise ValueError(f"unknown step type: {t}")
                run_it, reason = self._should_run_step(step_cfg=s_cfg, idx=idx)
                # resume: 跳过已完成的步骤
                if run_it and self.resume and idx <= last_completed_idx:
                    run_it = False
                    reason = f"resumed (completed in previous run, idx<={last_completed_idx})"
                step_plan.append(
                    {
                        "idx": idx,
                        "type": str(t),
                        "name": str(s_cfg.get("name", "")).strip(),
                        "enabled": bool(s_cfg.get("enabled", True)),
                        "action": "run" if run_it else "skip",
                        "reason": reason,
                    }
                )
            step_plan_name = "00_step_plan.json"
            if self.resume and logger.path(step_plan_name).exists():
                step_plan_name = f"00_step_plan_resume_{_now_id()}.json"
            _json_dump(logger.path(step_plan_name), step_plan)

            for p in step_plan:
                if p["action"] != "run":
                    continue
                idx = int(p["idx"])
                s_cfg = self.steps_cfg[idx - 1]
                t = str(p["type"])
                STEP_REGISTRY[t](s_cfg).run(ctx)
                # 记录完成的步骤和数据
                state["last_completed_step_idx"] = idx
                state["data"] = ctx.data
                _json_dump(state_file, state)

            # 全部步骤都完成（或因 resume 已完成）时，才重置 state。
            # 这样 `--only idx:1` 之后可用 `--resume --only idx:2` 接着跑。
            completed_workflow = True
            for p in step_plan:
                if not p.get("enabled", True):
                    continue
                if p["action"] == "run":
                    continue
                reason = str(p.get("reason") or "")
                if self.resume and reason.startswith("resumed (completed in previous run"):
                    continue
                completed_workflow = False
                break

            if completed_workflow:
                state["last_completed_step_idx"] = 0
                state["data"] = {}
                state.pop("active_run_id", None)
                state.pop("active_round_idx", None)
                _json_dump(state_file, state)
