# What changed (custom ops + native kernels)

Source report: [vllm-activity-n10-3920cafdd__HEAD-20260208-120459.md](vllm-activity-n10-3920cafdd__HEAD-20260208-120459.md)
Generated: 2026-02-08

## Highlights

- wvSplitKrc kernels: performance tuning and broader case coverage.
- Unified KV cache update: moved checks out of the custom op for leaner kernel path.
- CUTLASS: enable SM121 (DGX Spark) with `enable_sm120_or_later`.
- ROCm attention backends saw the largest churn (skinny gemms + backend files).
- Minor cleanup: spelling fixes in attention-related code paths.

## Scope

- Path filter: `vllm/_custom_ops.py`, `vllm/_aiter_ops.py`, `csrc`, `vllm/v1/attention`.
- Commits covered: 3920cafdd..HEAD (4 commits in report window).
