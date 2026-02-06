# vLLM 自定义算子映射表：注意力阶段

[English](vllm-custom-ops-mapping-attention.md) | [简体中文](vllm-custom-ops-mapping-attention.zh-CN.md)

范围：`torch.ops._C` 注意力算子（paged attention + merge），用于 decode 以及分块注意力的合并。

## 映射表：`torch.ops._C` 注意力算子（paged attention + merge）

| 算子 | Python 调用点 + GPT 推理用途 | 原生实现 + 说明/伪代码 |
|---|---|---|
| `paged_attention_v1` | 封装见 [vllm/vllm/_custom_ops.py](vllm/vllm/_custom_ops.py#L33)。用途：在分页 KV cache 上做 decode 注意力，按 $A = \text{softmax}(QK^T/\sqrt{d})V$ 计算，依赖 block_table 与 seq_lens。 | 实现： [vllm/csrc/attention/paged_attention_v1.cu](vllm/csrc/attention/paged_attention_v1.cu#L20) 调度到核心 kernel [vllm/csrc/attention/attention_kernels.cuh](vllm/csrc/attention/attention_kernels.cuh#L41)。支持 head_size/block_size 特化与 block-sparse 参数。伪代码：`对每个 seq/head 遍历 blocks; 计算 QK 与 softmax; 累加 V`。
| `paged_attention_v2` | 封装见 [vllm/vllm/_custom_ops.py](vllm/vllm/_custom_ops.py#L77)。用途：分块（partition）softmax 的 paged attention，先做分片，再归并。 | 实现： [vllm/csrc/attention/paged_attention_v2.cu](vllm/csrc/attention/paged_attention_v2.cu#L20) 启动分片 kernel + reduce kernel；核心数学在 [vllm/csrc/attention/attention_kernels.cuh](vllm/csrc/attention/attention_kernels.cuh#L41)。伪代码：`每个 partition 计算局部 softmax 与 logits; 记录 exp_sums/max_logits; reduce 得到最终输出`。
| `merge_attn_states` | 封装见 [vllm/vllm/_custom_ops.py](vllm/vllm/_custom_ops.py#L187)，通过调度器 [vllm/vllm/v1/attention/ops/merge_attn_states.py](vllm/vllm/v1/attention/ops/merge_attn_states.py#L9) 合并部分注意力结果（prefix/suffix 或 split-KV）。用途：用 log-sum-exp 稳定合并输出。 | 实现：Triton 版本在 [vllm/vllm/v1/attention/ops/triton_merge_attn_states.py](vllm/vllm/v1/attention/ops/triton_merge_attn_states.py#L11)（参考 arXiv:2501.01005 §2.2）。若 CUDA 支持，则调用 `torch.ops._C.merge_attn_states` 的自定义扩展实现。伪代码：`max_lse=max(p_lse,s_lse); out=(p_out*exp(p_lse-max_lse)+s_out*exp(s_lse-max_lse))/(exp(p_lse-max_lse)+exp(s_lse-max_lse))`。
