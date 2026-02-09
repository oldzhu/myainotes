# 变更摘要（自定义算子 + 原生内核）

[English](vllm-activity-summary-3920cafdd__HEAD-20260208-120459.md) | [简体中文](vllm-activity-summary-3920cafdd__HEAD-20260208-120459.zh-CN.md)

来源报告：[vllm-activity-n10-3920cafdd__HEAD-20260208-120459.md](vllm-activity-n10-3920cafdd__HEAD-20260208-120459.md)
生成时间：2026-02-08

## 要点

- wvSplitKrc 内核：性能调优并扩展覆盖用例。
- 统一 KV cache 更新：将检查逻辑移出自定义算子，使内核路径更精简。
- CUTLASS：为 SM121（DGX Spark）启用 `enable_sm120_or_later`。
- ROCm 注意力后端变更量最大（skinny gemms + 后端文件）。
- 小幅清理：修正注意力相关路径中的拼写问题。

## 范围

- 路径过滤：`vllm/_custom_ops.py`, `vllm/_aiter_ops.py`, `csrc`, `vllm/v1/attention`。
- 覆盖提交：3920cafdd..HEAD（报告窗口内 4 条提交）。
