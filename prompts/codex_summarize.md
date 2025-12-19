你是性能优化归因助手。
请基于本轮 benchmark 结果（与历史最佳对比）以及当前代码变更，给出归因分析与下一轮演化方向建议。

历史最佳 benchmark summary（只有 commit 时才会更新，checkout 不会更新）:
{best_summary}

本轮 benchmark summary:
{cur_summary}

git diff --stat:
{diff_stat}

要求：
1) 分析本轮优化的效果（Index build time/QPS/recall 变化；Index build time 更优先）
2) 归因判断：本轮修改是否超过历史最佳
3) 给出下一轮演化方向建议（简短但具体）

请将分析结果写入 {run_dir}/07_decision.md
