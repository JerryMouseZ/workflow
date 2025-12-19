"""BenchmarkStep 实现"""
from __future__ import annotations

import json
import re
import statistics
from typing import TYPE_CHECKING, Any

from . import Step
from ..utils import json_dump, run_shell

if TYPE_CHECKING:
    from ..runner import WorkflowContext


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
        index_time_re = re.compile(str(self.cfg.get(
            "index_build_time_regex",
            r"Index build time:\s*([0-9.]+)s",
        )), re.IGNORECASE)

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

            qps = recall = index_build_time_s = None
            for m in qps_re.finditer(r.output_text):
                qps = float(m.group(1))
            for m in recall_re.finditer(r.output_text):
                recall = float(m.group(1))
            for m in index_time_re.finditer(r.output_text):
                index_build_time_s = float(m.group(1))

            row = {
                "run": i,
                "qps": qps,
                "recall": recall,
                "index_build_time_s": index_build_time_s,
                "log": lp.name,
            }
            results.append(row)
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary = self._compute_summary(results, runs)
        json_dump(ctx.logger.path("06_bench_summary.json"), summary)
        ctx.data["benchmark_summary"] = summary
        # Persist latest benchmark summary (incl. index build time) into state.json.
        ctx.state["last_benchmark_summary"] = summary

    def _compute_summary(self, results: list[dict[str, Any]], runs: int) -> dict[str, Any]:
        qps_vals = [r["qps"] for r in results if isinstance(r.get("qps"), (int, float))]
        recall_vals = [r["recall"] for r in results if isinstance(r.get("recall"), (int, float))]
        index_time_vals = [
            r["index_build_time_s"] for r in results
            if isinstance(r.get("index_build_time_s"), (int, float))
        ]
        return {
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
            "index_build_time_s": {
                "n": len(index_time_vals),
                "mean": statistics.mean(index_time_vals) if index_time_vals else None,
                "median": statistics.median(index_time_vals) if index_time_vals else None,
                "stdev": statistics.pstdev(index_time_vals) if len(index_time_vals) > 1 else 0.0 if index_time_vals else None,
            },
        }
