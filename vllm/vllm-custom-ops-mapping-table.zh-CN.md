# vLLM 自定义算子映射表（总览）

[English](vllm-custom-ops-mapping-table.md) | [简体中文](vllm-custom-ops-mapping-table.zh-CN.md)

本总览链接到按流水线阶段拆分的映射表。每个表格将 Python 调用点与原生内核
以及 GPT 推理中的用途对应起来。

## 流水线阶段

- KV cache 阶段：[vllm-custom-ops-mapping-kv-cache.md](vllm-custom-ops-mapping-kv-cache.md)
- 注意力阶段：[vllm-custom-ops-mapping-attention.md](vllm-custom-ops-mapping-attention.md)
- 归一化 + RoPE + 激活阶段：[vllm-custom-ops-mapping-norm-activation.md](vllm-custom-ops-mapping-norm-activation.md)
- 采样 + logits 阶段：[vllm-custom-ops-mapping-sampling.md](vllm-custom-ops-mapping-sampling.md)
- 量化 + GEMM 阶段：[vllm-custom-ops-mapping-quant-gemm.md](vllm-custom-ops-mapping-quant-gemm.md)
- 分布式 + all-reduce 阶段：[vllm-custom-ops-mapping-distributed-ar.md](vllm-custom-ops-mapping-distributed-ar.md)
- CPU 后端阶段：[vllm-custom-ops-mapping-cpu-backends.md](vllm-custom-ops-mapping-cpu-backends.md)

## 备注

- 各阶段文档保持单一关注点，跨阶段算子以其主要调用点为准。
