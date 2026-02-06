# vLLM：最近 50 次提交（本地 `main`）— 趋势与热点

[English](vllm-last-50-commits-trends.md) | [简体中文](vllm-last-50-commits-trends.zh-CN.md)

背景：这是对 `/home/oldzhu/vllm` 最近 50 次提交的轻量活动报告（HEAD `02080179a`）。

## 高层趋势（来自提交标题）

- **Bugfix 为主**：提交标题标签多为 `Bugfix` / `BugFix`，也有不少未标注的提交。
- **异步调度 / 分布式正确性**：多次提到异步调度与 torchrun / PP（pipeline parallel）行为。
- **量化与 MoE**：反复出现 **FP8**、**MoE**、**TRTLLM**、**quantized** 等。
- **前端/入口演进**：请求 schema 与服务入口持续调整（包含 “pooling entrypoints” 与 realtime 相关测试/模型）。
- **配置整理与重构**：配置相关变更与移除弃用的环境变量。

提交标题中的高频关键词（大致）：async, backend, fp8, moe, transformers, model, torchrun, scheduling, pooling, request, decoding。

## Top 5 “工作项”（最活跃领域）

按**最近 50 次非 merge 提交触及次数**排序：

1. **tests** — 17 次提交
2. **vllm/model_executor** — 13 次提交
3. **vllm/v1** — 8 次提交
4. **vllm/entrypoints** — 7 次提交
5. **vllm/config** — 6 次提交

这些领域的代表性提交标题：

- **tests**
  - Fix torchrun PP broadcast deadlock with async scheduling
  - Pooling entrypoints request schema consensus
  - Fix negative accepted tokens metric crash
- **vllm/model_executor**
  - Fix sparse MLA metadata building
  - Disable TRTLLM FP8 MoE for specific routing/router logits dtype combinations
  - Fix tensor parallelism for quantized Mamba models
- **vllm/v1**
  - Async scheduling / metrics / scheduler-related changes
- **vllm/entrypoints**
  - Pooling entrypoints request schema changes
  - Voxtral realtime warm-up behavior adjustments
- **vllm/config**
  - `@config` → `dataclass_transform`
  - Removal of deprecated env vars

## 热点文件（触及最多）

按**触及频次**排序：

- `vllm/v1/core/sched/scheduler.py`
- `vllm/model_executor/layers/quantization/fp8.py`
- `vllm/forward_context.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/config/compilation.py`
- `tests/distributed/test_torchrun_example.py`
- `tests/distributed/test_torchrun_example_moe.py`
- `tests/entrypoints/openai/test_realtime_validation.py`
- `tests/models/multimodal/generation/test_voxtral_realtime.py`

## Churn 峰值（大改动）

某些路径在这段时间内有极大的新增+删除行数（不一定是提交次数多）：

- `csrc/cpu/cpu_types_arm.hpp`
- `mooncake/mooncake_connector.py` 及 `examples/online_serving/disaggregated_serving/mooncake_connector/*`
- `vllm/_ipex_ops.py` 与 `vllm/model_executor/layers/quantization/ipex_quant.py`
- `vllm/model_executor/models/interns1_pro.py`

## 复现 / 扩展该报告

如果你想调整“区域归类”或换一个窗口：

- `git --no-pager log -50 --oneline`
- `git --no-pager log -50 --no-merges --numstat`
- `git --no-pager log -50 --no-merges --name-only`

（如果需要，我也可以生成按领域的“top commits”列表，或按周趋势。）
