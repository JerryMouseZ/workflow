"""向后兼容的重导出模块 - 所有 Step 实现已迁移到 steps/ 子包"""
from __future__ import annotations

# 重导出所有公开接口，保持向后兼容
from .steps import Step, STEP_REGISTRY
from .steps.command import CommandStep
from .steps.profile import ProfileAnalyzeStep
from .steps.target import TargetSelectStep
from .steps.context import ContextCollectStep
from .steps.codex_kiro import CodexGenerateKiroPromptStep
from .steps.kiro import KiroApplyAndTestStep
from .steps.benchmark import BenchmarkStep
from .steps.git_decide import CodexGitDecideStep

__all__ = [
    "Step",
    "STEP_REGISTRY",
    "CommandStep",
    "ProfileAnalyzeStep",
    "TargetSelectStep",
    "ContextCollectStep",
    "CodexGenerateKiroPromptStep",
    "KiroApplyAndTestStep",
    "BenchmarkStep",
    "CodexGitDecideStep",
]
