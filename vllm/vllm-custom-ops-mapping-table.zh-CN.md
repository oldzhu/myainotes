# vLLM 自定义算子映射表（总览）

[English](vllm-custom-ops-mapping-table.md) | [简体中文](vllm-custom-ops-mapping-table.zh-CN.md)

本总览链接到按流水线阶段拆分的映射表。每个表格将 Python 调用点与原生内核
以及 GPT 推理中的用途对应起来。

## 流水线阶段

- KV cache 阶段：[vllm-custom-ops-mapping-kv-cache.md](vllm-custom-ops-mapping-kv-cache.md)
- 注意力阶段：[vllm-custom-ops-mapping-attention.md](vllm-custom-ops-mapping-attention.md)
- 归一化 + RoPE + 激活阶段：[vllm-custom-ops-mapping-norm-activation.md](vllm-custom-ops-mapping-norm-activation.md)

## 备注

- 后续阶段（量化/GEMM、分布式/AR、CPU 后端）会新增独立文档。
