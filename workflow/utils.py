"""工具函数模块"""
from __future__ import annotations

import datetime as _dt
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore


def load_config(path: Path) -> dict[str, Any]:
    """加载 TOML 配置文件"""
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    if "workflow" not in cfg:
        raise ValueError("config missing [workflow]")
    if "steps" not in cfg or not isinstance(cfg["steps"], list):
        raise ValueError("config missing [[steps]] list")
    return cfg


def now_id() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def latest_run_id(log_dir: Path, *, round_idx: int) -> str | None:
    if not log_dir.exists():
        return None
    suffix = f"_r{round_idx}"
    candidates = [p.name for p in log_dir.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    return sorted(candidates)[-1] if candidates else None


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_changelog(state_file: Path) -> list[dict[str, Any]]:
    """从独立文件加载 changelog"""
    changelog_file = state_file.parent / "changelog.json"
    if changelog_file.exists():
        try:
            return json.loads(changelog_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def save_changelog(state_file: Path, changelog: list[dict[str, Any]]) -> None:
    """保存 changelog 到独立文件"""
    changelog_file = state_file.parent / "changelog.json"
    json_dump(changelog_file, changelog)


@dataclass
class CmdResult:
    returncode: int
    output_text: str


class RunLogger:
    """运行日志管理器"""
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.commands_log = self.run_dir / "commands.log"

    @property
    def run_id(self) -> str:
        return self.run_dir.name

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


def run_cmd(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str] | None,
    stdin_text: str | None,
    log_path: Path,
    tee: bool = True,
) -> CmdResult:
    """执行命令并记录输出"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), env=env,
        stdin=subprocess.PIPE if stdin_text is not None else None,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
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


def run_shell(*, command: str, cwd: Path, env: dict[str, str] | None, log_path: Path, tee: bool = True) -> CmdResult:
    return run_cmd(cmd=["bash", "-lc", command], cwd=cwd, env=env, stdin_text=None, log_path=log_path, tee=tee)


def run_agent(*, cmd: list[str], prompt: str, cwd: Path, env: dict[str, str] | None, log_path: Path, tee: bool = True) -> CmdResult:
    return run_cmd(cmd=cmd, cwd=cwd, env=env, stdin_text=prompt, log_path=log_path, tee=tee)


def run_agent_file(*, cmd: list[str], input_file: Path, cwd: Path, env: dict[str, str] | None, log_path: Path, tee: bool = True) -> CmdResult:
    """使用 shell 重定向从文件输入运行 agent: cmd < input_file"""
    shell_cmd = " ".join(cmd) + f" < {input_file}"
    return run_cmd(cmd=["bash", "-lc", shell_cmd], cwd=cwd, env=env, stdin_text=None, log_path=log_path, tee=tee)


def git(cmd: str, repo_root: Path) -> str:
    return subprocess.check_output(["bash", "-lc", cmd], cwd=str(repo_root), text=True).strip()


def which(exe: str) -> str | None:
    return shutil.which(exe)


def latest_folded_file(repo_root: Path) -> Path | None:
    """获取最新的 .folded 或 .perf.data 文件"""
    prof_dir = repo_root / "test" / "profile_output"
    if not prof_dir.exists():
        return None
    all_files = list(prof_dir.glob("*.folded")) + list(prof_dir.glob("*.perf.data"))
    return max(all_files, key=lambda p: p.stat().st_mtime) if all_files else None


def parse_self_time_table(text: str) -> list[dict[str, Any]]:
    """解析 profile 分析输出表格"""
    entries: list[dict[str, Any]] = []
    in_table = False
    has_samples_col = False
    for line in text.splitlines():
        if line.strip().startswith("Rank") and "Function" in line:
            in_table, has_samples_col = True, "Samples" in line
            continue
        if in_table and re.fullmatch(r"-{10,}", line.strip()):
            continue
        if in_table and line.strip().startswith("Top "):
            break
        if not in_table:
            continue
        if has_samples_col:
            m = re.match(r"^\s*(\d+)\s+(\d+(?:\.\d+)?)%\s+(\d+)\s+(.*)$", line)
            if not m:
                continue
            rank, self_pct, samples, func = int(m.group(1)), float(m.group(2)), int(m.group(3)), m.group(4).strip()
        else:
            m = re.match(r"^\s*(\d+)\s+(\d+(?:\.\d+)?)%\s+(.*)$", line)
            if not m:
                continue
            rank, self_pct, samples, func = int(m.group(1)), float(m.group(2)), 0, m.group(3).strip()
        entries.append({"rank": rank, "self_pct": self_pct, "samples": samples, "function": func})
    return entries


def filter_profile_entries(entries: list[dict[str, Any]], ignore_regex: list[str]) -> list[dict[str, Any]]:
    regs = [re.compile(p) for p in ignore_regex]
    return [e for e in entries if not any(r.search(str(e["function"])) for r in regs)]


def pick_targets(entries: list[dict[str, Any]], strategy: str, k: int, 
                 changelog: list[dict] | None = None, max_attempts_per_func: int = 3) -> list[dict[str, Any]]:
    """选择优化目标，避免重复优化同一函数太多次"""
    if not entries:
        return []
    
    # 统计每个函数被尝试的次数（只计算 checkout 的，commit 的说明有效）
    attempt_counts: dict[str, int] = {}
    if changelog:
        for entry in changelog:
            func = entry.get("target_func", "")
            if entry.get("outcome") == "checkout":
                attempt_counts[func] = attempt_counts.get(func, 0) + 1
    
    # 过滤掉尝试次数过多的函数
    filtered = [e for e in entries if attempt_counts.get(e["function"], 0) < max_attempts_per_func]
    if not filtered:
        filtered = entries  # 如果全部被过滤，回退到原始列表
    
    if strategy == "top_samples":
        ranked = sorted(filtered, key=lambda x: (-int(x["samples"]), float(x["self_pct"])))
    else:
        ranked = sorted(filtered, key=lambda x: (-float(x["self_pct"]), -int(x["samples"])))
    return ranked[:max(1, k)]


def rg_matches(repo_root: Path, roots: list[str], needle: str, max_total: int) -> list[dict[str, Any]]:
    """使用 ripgrep 搜索匹配"""
    matches: list[dict[str, Any]] = []
    rg = which("rg")
    if rg:
        existing_roots = [r for r in roots if (repo_root / r).exists()]
        search_roots = existing_roots or ["."]

        def _parse_rg_stdout(stdout: str) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for line in stdout.splitlines():
                parts = line.split(":", 2)
                if len(parts) != 3:
                    continue
                try:
                    out.append({"path": parts[0], "line": int(parts[1]), "content": parts[2]})
                except ValueError:
                    continue
                if len(out) >= max_total:
                    break
            return out

        # 优先：对 C/C++ 标识符用单词边界精确匹配；否则退化为固定字符串搜索。
        is_identifier = re.fullmatch(r"[A-Za-z_]\w*", needle) is not None
        if is_identifier:
            pattern = rf"\b{re.escape(needle)}\b"
            cmd = ["rg", "-n", "-S", "--no-heading", "--glob", "!.git/**", pattern] + search_roots
            proc = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)
            matches = _parse_rg_stdout(proc.stdout)

        if not matches:
            cmd = ["rg", "-n", "-S", "--no-heading", "--glob", "!.git/**", "-F", needle] + search_roots
            proc = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)
            matches = _parse_rg_stdout(proc.stdout)

        return matches

    # fallback: 纯 Python 搜索
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


def enclosing_c_function(lines: list[str], one_based_line: int, lookback: int = 250) -> str | None:
    idx = max(0, one_based_line - 1)
    for j in range(idx, max(0, idx - lookback) - 1, -1):
        m = _C_DEF_RE.match(lines[j])
        if m:
            return m.group(1)
    return None


def snippet(path: Path, line: int, radius: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    start, end = max(1, line - radius), min(len(lines), line + radius)
    return "\n".join(f"{'>>':>2}{ln:6d} {lines[ln-1]}" if ln == line else f"  {ln:6d} {lines[ln-1]}" for ln in range(start, end + 1))


def collect_context_md(
    *, repo_root: Path, target_func: str, roots: list[str],
    snippet_radius: int, max_files: int, max_total_matches: int,
) -> tuple[str, list[dict[str, Any]]]:
    """收集目标函数的上下文信息，生成精简 Markdown（只保留调用链和关键代码）"""
    matches = rg_matches(repo_root, roots, target_func, max_total_matches)
    if not matches:
        return f"# Context for `{target_func}`\n\n_No matches found._\n", matches

    by_file: dict[str, list[dict[str, Any]]] = {}
    for m in matches:
        by_file.setdefault(m["path"], []).append(m)
    ranked_files = sorted(by_file.items(), key=lambda kv: -len(kv[1]))[:max_files]

    # 收集调用链信息
    callers: dict[str, list[tuple[str, int]]] = {}  # caller -> [(file, line), ...]
    for p, ms in ranked_files:
        try:
            file_lines = (repo_root / p).read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for m in ms[:10]:
            caller = enclosing_c_function(file_lines, int(m["line"]))
            if caller and caller != target_func:
                callers.setdefault(caller, []).append((p, int(m["line"])))

    md_lines = [f"# Context for `{target_func}`", ""]

    # 调用链（最重要）
    md_lines.append("## Call chain")
    if callers:
        for caller, locs in sorted(callers.items(), key=lambda kv: -len(kv[1]))[:10]:
            loc_str = ", ".join(f"{p}:{ln}" for p, ln in locs[:3])
            md_lines.append(f"- `{caller}` -> `{target_func}` @ {loc_str}")
    else:
        md_lines.append("_(no callers found)_")
    md_lines.append("")

    # 关键代码片段（精简：每文件最多 2 个，radius 限制为 8 行）
    effective_radius = min(snippet_radius, 8)
    md_lines.append("## Key snippets")
    for p, ms in ranked_files[:7]:  # 最多 7 个文件
        abs_p = repo_root / p
        md_lines.append(f"### `{p}`")
        for m in ms[:2]:  # 每文件最多 2 个片段
            md_lines.extend([f"L{m['line']}:", "```c", snippet(abs_p, int(m["line"]), effective_radius), "```"])
        md_lines.append("")

    return "\n".join(md_lines), matches
