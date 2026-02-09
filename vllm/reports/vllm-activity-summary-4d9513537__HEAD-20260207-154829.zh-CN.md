# 变更摘要（自定义算子 + 原生内核）

[English](vllm-activity-summary-4d9513537__HEAD-20260207-154829.md) | [简体中文](vllm-activity-summary-4d9513537__HEAD-20260207-154829.zh-CN.md)

来源报告：[vllm-activity-n10-4d9513537__HEAD-20260207-154829.md](vllm-activity-n10-4d9513537__HEAD-20260207-154829.md)
生成时间：2026-02-07

## 要点

- CPU attention 工作占主导：NEON BFMMLA BF16 分页注意力实现与 s390x BF16 kernel 类型。
- ROCm AITER 导入回归已修复，支持显式后端选择。
- 注意力栈变更包括 FA3 swizzle 回滚与 KV 传输启用时禁用 TRTLLM attention。
- MoE permute 内核重构，移除对齐 block size 逻辑。
- FP8 内核更新（wvSplitK）增加 padding 与性能优化。

## 范围

- 路径过滤：`vllm/_custom_ops.py`, `vllm/_aiter_ops.py`, `csrc`, `vllm/v1/attention`。
- 覆盖提交：4d9513537..HEAD（报告窗口内 8 条提交）。
