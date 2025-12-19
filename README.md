# 自动性能优化工作流（可插拔）

该目录提供一个“profile→制定优化方案→自动改代码/跑测试→跑benchmark→归因并决定commit/checkout→进入下一轮”的自动化工作流，所有外部脚本与命令都通过配置文件进行组合。

## 快速开始

1) 复制并编辑配置：

```bash
cp workflow/config.example.toml workflow/config.toml
```

2) 运行一轮工作流：

```bash
python3 workflow/run.py --config workflow/config.toml
```

## 跳过/只运行部分步骤（用于调试）

先查看 config 里有哪些 step：

```bash
python3 workflow/run.py --config workflow/config.toml --list-steps
```

只打印“本次会跑哪些步骤”，但不实际执行：

```bash
python3 workflow/run.py --config workflow/config.toml --dry-run --skip bench,codex_git
```

实际执行时跳过某些步骤（selector 可用 `name:<name>` / `type:<type>` / `idx:<i>` / `#<i>`，或直接写 token 匹配 name/type）：

```bash
python3 workflow/run.py --config workflow/config.toml --skip bench,codex_git
python3 workflow/run.py --config workflow/config.toml --only name:profile,name:target,name:context
```

也可以在 `[[steps]]` 里加 `enabled = false` 直接禁用某一步。

## 从中断位置继续执行（--resume）

如果 workflow 执行中断（如手动停止或出错），可以使用 `--resume` 从上次完成的步骤继续：

```bash
python3 workflow/run.py --config workflow/config.toml --resume
```

- 执行进度保存在 `state.json` 的 `last_completed_step_idx` 字段
- `--resume` 会跳过已完成的步骤，从下一个步骤开始执行
- `--resume` 会复用同一个 `workflow/logs/<run_id>/` 目录继续写日志（`run_id` 保存在 `state.json` 的 `active_run_id`/`active_round_idx` 字段）
- 当工作流“完整跑完全部启用步骤”（包含因 `--resume` 被跳过的已完成步骤）后，进度会自动重置；如果使用 `--only`/`--skip` 只跑子集步骤，进度不会重置，便于分段调试与继续执行
- 可与 `--skip`/`--only` 组合使用

## 关键点

- Profile 分析：强制调用 `python3 test/analyze_self_time.py`，并在工作流侧对结果进行过滤（默认跳过 `^\[.*\]$` / `unknown` 之类的“未知项”）。
- Orchestrator：根据 profile Top 列表选择候选热点函数，结合 `rg` 搜索得到的调用点/代码片段，调用 `codex exec --dangerously-bypass-approvals-and-sandbox` 产出给 `kiro-cli` 的执行 prompt。
- Kiro 迭代：执行修改后自动跑编译/单测命令；失败则携带错误日志继续喂给 kiro 迭代，直到通过或达到最大迭代次数。
- Benchmark：先 warmup 3 次（丢弃），再跑 10 次并解析/保存结果（QPS/Recall/Index build time 等由正则可配置；Index build time 日志格式：`Index build time: 42.5s`）。
- Git 决策：由 codex 结合“本轮 vs 上轮”的 benchmark 汇总与 `git diff --stat` 归因，决定是否提交或回退；无论 commit/checkout 都会保留本轮 `workflow/logs/<run_id>/` 日志。

## Agent 调用格式（等价）

- Codex：`echo "prompt" | codex exec --dangerously-bypass-approvals-and-sandbox`
- Kiro：`echo "prompt" | kiro-cli chat -a`

## 日志与状态

- 日志：`workflow/logs/<run_id>/`（包含每步的 stdout/stderr、提取的 context、benchmark 原始输出与汇总等）
- 状态：`workflow/state.json`（保存上一次 benchmark 汇总与 git HEAD，用于对比归因）

> 提示：默认 `workflow/.gitignore` 会忽略 `logs/` 与 `state.json`，避免误提交。
