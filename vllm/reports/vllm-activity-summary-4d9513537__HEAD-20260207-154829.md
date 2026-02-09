# What changed (custom ops + native kernels)

[English](vllm-activity-summary-4d9513537__HEAD-20260207-154829.md) | [Chinese (ZH-CN)](vllm-activity-summary-4d9513537__HEAD-20260207-154829.zh-CN.md)

Source report: [vllm-activity-n10-4d9513537__HEAD-20260207-154829.md](vllm-activity-n10-4d9513537__HEAD-20260207-154829.md)
Generated: 2026-02-07

## Highlights

- CPU attention work dominates: NEON BFMMLA BF16 paged-attention implementation and s390x BF16 kernel type.
- ROCm AITER import regression fixed for explicit backend selection.
- Attention stack churn includes FA3 swizzle revert and TRTLLM attention disablement when KV transfer is enabled.
- MoE permute kernel refactor removes align block size logic.
- FP8 kernel updates (wvSplitK) add padding and performance tweaks.

## Scope

- Path filter: `vllm/_custom_ops.py`, `vllm/_aiter_ops.py`, `csrc`, `vllm/v1/attention`.
- Commits covered: 4d9513537..HEAD (8 commits in report window).
