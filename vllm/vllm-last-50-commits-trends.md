# vLLM: last 50 commits (local `main`) — trends & hot areas

Context: this is a lightweight activity report over the most recent 50 commits in `/home/oldzhu/vllm` (HEAD `02080179a`).

## High-level trends (from commit subjects)

- **Bugfix-heavy**: commit subject tags are dominated by `Bugfix` / `BugFix`, plus many untagged commits.
- **Async scheduling / distributed correctness**: multiple changes reference async scheduling and torchrun / PP (pipeline parallel) behaviors.
- **Quantization & MoE work**: repeated mentions of **FP8**, **MoE**, **TRTLLM**, and **quantized** models.
- **Frontend/entrypoints evolution**: ongoing changes around request schemas and serving entrypoints (including “pooling entrypoints” and realtime-related tests/models).
- **Config hygiene & refactors**: config-related changes and removal of deprecated environment variables.

Top keywords (roughly) from subjects: async, backend, fp8, moe, transformers, model, torchrun, scheduling, pooling, request, decoding.

## Top 5 “work items” (most active areas)

Ranked by **how many of the last 50 non-merge commits touched the area**:

1. **tests** — 17 commits
2. **vllm/model_executor** — 13 commits
3. **vllm/v1** — 8 commits
4. **vllm/entrypoints** — 7 commits
5. **vllm/config** — 6 commits

Representative commit subjects seen in those areas:

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

## Hot files (most touched)

By **touch frequency** in the last 50 non-merge commits:

- `vllm/v1/core/sched/scheduler.py`
- `vllm/model_executor/layers/quantization/fp8.py`
- `vllm/forward_context.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/config/compilation.py`
- `tests/distributed/test_torchrun_example.py`
- `tests/distributed/test_torchrun_example_moe.py`
- `tests/entrypoints/openai/test_realtime_validation.py`
- `tests/models/multimodal/generation/test_voxtral_realtime.py`

## Churn spikes (large edits)

Some paths show very large added+deleted line counts over the window (not necessarily many commits):

- `csrc/cpu/cpu_types_arm.hpp`
- `mooncake/mooncake_connector.py` and related `examples/online_serving/disaggregated_serving/mooncake_connector/*`
- `vllm/_ipex_ops.py` and `vllm/model_executor/layers/quantization/ipex_quant.py`
- `vllm/model_executor/models/interns1_pro.py`

## Reproduce / extend the report

If you want to tweak the “area bucketing” or look at a different window size:

- `git --no-pager log -50 --oneline`
- `git --no-pager log -50 --no-merges --numstat`
- `git --no-pager log -50 --no-merges --name-only`

(If you want, I can also generate a per-area “top commits” list or a week-by-week trend line.)
