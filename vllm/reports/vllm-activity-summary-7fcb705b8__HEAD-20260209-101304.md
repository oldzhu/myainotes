# What changed (custom ops + native kernels)

[English](vllm-activity-summary-7fcb705b8__HEAD-20260209-101304.md) | [Chinese (ZH-CN)](vllm-activity-summary-7fcb705b8__HEAD-20260209-101304.zh-CN.md)

Source report: [vllm-activity-n10-7fcb705b8__HEAD-20260209-101304.md](vllm-activity-n10-7fcb705b8__HEAD-20260209-101304.md)
Generated: 2026-02-09

## Highlights

- Reverted a performance regression fix affecting GLM-4.7-GPTQ decode and MTP acceptance rate.
- Changes limited to attention backend code in `vllm/v1/attention/backends/flashinfer.py`.

## Scope

- Path filter: `vllm/_custom_ops.py`, `vllm/_aiter_ops.py`, `csrc`, `vllm/v1/attention`.
- Commits covered: 7fcb705b8..HEAD (1 commit in report window).
