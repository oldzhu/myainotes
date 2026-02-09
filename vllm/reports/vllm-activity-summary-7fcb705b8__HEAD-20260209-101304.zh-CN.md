# 变更摘要（自定义算子 + 原生内核）

[English](vllm-activity-summary-7fcb705b8__HEAD-20260209-101304.md) | [简体中文](vllm-activity-summary-7fcb705b8__HEAD-20260209-101304.zh-CN.md)

来源报告：[vllm-activity-n10-7fcb705b8__HEAD-20260209-101304.md](vllm-activity-n10-7fcb705b8__HEAD-20260209-101304.md)
生成时间：2026-02-09

## 要点

- 回滚了影响 GLM-4.7-GPTQ decode 与 MTP 接受率的性能回归修复。
- 变更集中在 `vllm/v1/attention/backends/flashinfer.py` 的注意力后端代码。

## 范围

- 路径过滤：`vllm/_custom_ops.py`, `vllm/_aiter_ops.py`, `csrc`, `vllm/v1/attention`。
- 覆盖提交：7fcb705b8..HEAD（报告窗口内 1 条提交）。
