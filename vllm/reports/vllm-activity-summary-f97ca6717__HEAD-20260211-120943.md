# What changed (custom ops + native kernels)

[English](vllm-activity-summary-f97ca6717__HEAD-20260211-120943.md) | [Chinese (ZH-CN)](vllm-activity-summary-f97ca6717__HEAD-20260211-120943.zh-CN.md)

Source report: [vllm-activity-n10-f97ca6717__HEAD-20260211-120943.md](vllm-activity-n10-f97ca6717__HEAD-20260211-120943.md)
Generated: 2026-02-11

## Highlights

- Qwen3-Next MTP accuracy bugfix.
- Faster topK-per-row decode kernel for DeepSeek-V3.2 sparse attention (large churn in `csrc/topk.cu`).
- CPU backend fixes: w8a8 oneDNN weight prepack and MLA decode build fix on x86 without AVX512.

## Scope

- Path filter: `vllm/_custom_ops.py`, `vllm/_aiter_ops.py`, `csrc`, `vllm/v1/attention`.
- Commits covered: f97ca6717..HEAD (4 commits in report window).
