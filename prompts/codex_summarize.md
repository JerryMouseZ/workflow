你是性能优化归因助手。
请基于本轮 benchmark 结果（与上轮对比）以及当前代码变更，给出归因分析与下一轮演化方向建议。

上轮 benchmark summary（可能为空）:
{prev_summary}

本轮 benchmark summary:
{cur_summary}

git diff --stat:
{diff_stat}

要求：
1) 分析本轮优化的效果（QPS/recall 变化）
2) 归因判断：本轮修改是否带来了正向收益
3) 给出下一轮演化方向建议（简短但具体）

请将分析结果写入 {run_dir}/07_decision.md
