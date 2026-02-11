# 变更摘要（自定义算子 + 原生内核）

[English](vllm-activity-summary-f97ca6717__HEAD-20260211-120943.md) | [简体中文](vllm-activity-summary-f97ca6717__HEAD-20260211-120943.zh-CN.md)

来源报告：[vllm-activity-n10-f97ca6717__HEAD-20260211-120943.md](vllm-activity-n10-f97ca6717__HEAD-20260211-120943.md)
生成时间：2026-02-11

## 要点

- 修复 Qwen3-Next MTP 精度问题。
- DeepSeek-V3.2 稀疏注意力新增更快的每行 topK decode 内核（`csrc/topk.cu` 变更量最大）。
- CPU 后端修复：w8a8 oneDNN 权重预打包与 x86 无 AVX512 的 MLA decode 编译修复。

## 范围

- 路径过滤：`vllm/_custom_ops.py`, `vllm/_aiter_ops.py`, `csrc`, `vllm/v1/attention`。
- 覆盖提交：f97ca6717..HEAD（报告窗口内 4 条提交）。
