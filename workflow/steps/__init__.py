"""Step 基类和注册表"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..runner import WorkflowContext


class Step:
    """Step 基类"""
    type_name: str = ""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def run(self, ctx: WorkflowContext) -> None:
        raise NotImplementedError


# 延迟导入各 Step 实现，避免循环依赖
def _get_registry() -> dict[str, type[Step]]:
    from .command import CommandStep
    from .profile import ProfileAnalyzeStep
    from .target import TargetSelectStep
    from .context import ContextCollectStep
    from .codex_kiro import CodexGenerateKiroPromptStep
    from .kiro import KiroApplyAndTestStep
    from .benchmark import BenchmarkStep
    from .git_decide import CodexGitDecideStep

    return {
        cls.type_name: cls for cls in [
            CommandStep, ProfileAnalyzeStep, TargetSelectStep, ContextCollectStep,
            CodexGenerateKiroPromptStep, KiroApplyAndTestStep, BenchmarkStep, CodexGitDecideStep,
        ]
    }


class _LazyRegistry:
    """延迟加载的 Step 注册表"""
    _cache: dict[str, type[Step]] | None = None

    def __getitem__(self, key: str) -> type[Step]:
        if self._cache is None:
            self._cache = _get_registry()
        return self._cache[key]

    def __contains__(self, key: str) -> bool:
        if self._cache is None:
            self._cache = _get_registry()
        return key in self._cache


STEP_REGISTRY = _LazyRegistry()
