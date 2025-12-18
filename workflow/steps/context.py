"""ContextCollectStep 实现"""
from __future__ import annotations

from typing import TYPE_CHECKING

from . import Step
from ..utils import collect_context_md, json_dump

if TYPE_CHECKING:
    from ..runner import WorkflowContext


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
