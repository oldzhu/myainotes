# vLLM custom ops mapping table: attention stage

[English](vllm-custom-ops-mapping-attention.md) | [Chinese (ZH-CN)](vllm-custom-ops-mapping-attention.zh-CN.md)

Scope: `torch.ops._C` attention ops (paged attention + merge), used during decode and
partitioned attention merging.

## Mapping table: `torch.ops._C` attention ops (paged attention + merge)

| Op | Python call site(s) + GPT inference role | Native implementation + notes/pseudo code |
|---|---|---|
| `paged_attention_v1` | Wrapper in [vllm/vllm/_custom_ops.py](vllm/vllm/_custom_ops.py#L33). Role: decode-time attention over paged KV cache blocks using $A = \text{softmax}(QK^T/\sqrt{d})V$ with block tables and sequence lengths. | Implementation: [vllm/csrc/attention/paged_attention_v1.cu](vllm/csrc/attention/paged_attention_v1.cu#L20) dispatches into the kernel in [vllm/csrc/attention/attention_kernels.cuh](vllm/csrc/attention/attention_kernels.cuh#L41). Supports head-size and block-size specializations and optional block-sparse parameters. Pseudocode: `for each seq/head: iterate blocks via block_tables; compute QK and softmax in shared memory; accumulate V`.
| `paged_attention_v2` | Wrapper in [vllm/vllm/_custom_ops.py](vllm/vllm/_custom_ops.py#L77). Role: paged attention with partitioned softmax (split-KV) to reduce memory, then reduce across partitions. | Implementation: [vllm/csrc/attention/paged_attention_v2.cu](vllm/csrc/attention/paged_attention_v2.cu#L20) launches a partitioned kernel plus a reduce kernel; core math in [vllm/csrc/attention/attention_kernels.cuh](vllm/csrc/attention/attention_kernels.cuh#L41). Pseudocode: `for partition: compute partial softmax/logits; store exp_sums + max_logits; reduce partitions to final output`.
| `merge_attn_states` | Wrapper in [vllm/vllm/_custom_ops.py](vllm/vllm/_custom_ops.py#L187); used via the dispatcher in [vllm/vllm/v1/attention/ops/merge_attn_states.py](vllm/vllm/v1/attention/ops/merge_attn_states.py#L9) to combine partial attention outputs (prefix/suffix or split-KV). Role: numerically stable merge of attention outputs using log-sum-exp. | Implementation: Triton fallback in [vllm/vllm/v1/attention/ops/triton_merge_attn_states.py](vllm/vllm/v1/attention/ops/triton_merge_attn_states.py#L11) (references arXiv:2501.01005 §2.2). When CUDA supports it, a custom C++ op is used via `torch.ops._C.merge_attn_states` (source in extension). Pseudocode: `max_lse = max(p_lse, s_lse); out = (p_out*exp(p_lse-max_lse) + s_out*exp(s_lse-max_lse)) / (exp(p_lse-max_lse)+exp(s_lse-max_lse))`.
