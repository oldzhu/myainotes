# vLLM custom ops mapping table: CPU backends stage

[English](vllm-custom-ops-mapping-cpu-backends.md) | [Chinese (ZH-CN)](vllm-custom-ops-mapping-cpu-backends.zh-CN.md)

Scope: CPU attention kernels and oneDNN-backed matrix multiplication ops.

## Mapping table: CPU backends

| Op | Python call site(s) + GPT inference role | Native implementation + notes/pseudo code |
|---|---|---|
| `cpu_attn_get_scheduler_metadata` | Called in CPU attention backend [vllm/vllm/v1/attention/backends/cpu_attn.py](vllm/vllm/v1/attention/backends/cpu_attn.py#L177). Role: compute scheduler metadata for CPU attention execution. | CPU kernel in [vllm/csrc/cpu/cpu_attn.cpp](vllm/csrc/cpu/cpu_attn.cpp#L3). Computes tile sizes and scheduling params. Pseudocode: `metadata = calc_schedule(seqlens, heads)`.
| `cpu_attn_reshape_and_cache` | Called during CPU prefill [vllm/vllm/v1/attention/backends/cpu_attn.py](vllm/vllm/v1/attention/backends/cpu_attn.py#L326). Role: reshape QKV and populate CPU KV cache blocks. | CPU kernel in [vllm/csrc/cpu/cpu_attn.cpp](vllm/csrc/cpu/cpu_attn.cpp#L66). Writes keys/values into cache layout. Pseudocode: `cache_kv(key, value, slots)`.
| `cpu_attention_with_kv_cache` | Called during CPU decode [vllm/vllm/v1/attention/backends/cpu_attn.py](vllm/vllm/v1/attention/backends/cpu_attn.py#L349). Role: compute attention over CPU KV cache. | CPU kernel in [vllm/csrc/cpu/cpu_attn.cpp](vllm/csrc/cpu/cpu_attn.cpp#L122). Runs attention with cached KV. Pseudocode: `softmax(QK^T) * V`.
| `create_onednn_mm` | Called when setting CPU linear handler [vllm/vllm/model_executor/layers/utils.py](vllm/vllm/model_executor/layers/utils.py#L261). Role: create oneDNN GEMM handler for CPU linear layers. | CPU entry in [vllm/csrc/cpu/dnnl_kernels.cpp](vllm/csrc/cpu/dnnl_kernels.cpp#L497). Prepares oneDNN primitive and returns a handler. Pseudocode: `handler = dnnl_prepare(weight)`.
| `onednn_mm` | Called by CPU linear lambda [vllm/vllm/model_executor/layers/utils.py](vllm/vllm/model_executor/layers/utils.py#L262). Role: run oneDNN GEMM with optional bias on CPU. | CPU entry in [vllm/csrc/cpu/dnnl_kernels.cpp](vllm/csrc/cpu/dnnl_kernels.cpp#L519). Executes oneDNN matmul. Pseudocode: `C = A * B + bias`.
| `create_onednn_scaled_mm` | Called in CPU scaled MM path [vllm/vllm/model_executor/layers/quantization/kernels/scaled_mm/cpu.py](vllm/vllm/model_executor/layers/quantization/kernels/scaled_mm/cpu.py#L127). Role: create oneDNN handler for scaled GEMM. | CPU entry in [vllm/csrc/cpu/dnnl_kernels.cpp](vllm/csrc/cpu/dnnl_kernels.cpp#L303). Prepares scaled matmul primitive. Pseudocode: `handler = dnnl_prepare_scaled(weight)`.
| `onednn_scaled_mm` | Called in CPU scaled MM path [vllm/vllm/model_executor/layers/quantization/kernels/scaled_mm/cpu.py](vllm/vllm/model_executor/layers/quantization/kernels/scaled_mm/cpu.py#L199). Role: run oneDNN scaled GEMM on CPU. | CPU entry in [vllm/csrc/cpu/dnnl_kernels.cpp](vllm/csrc/cpu/dnnl_kernels.cpp#L354). Executes scaled matmul with scales/zero-points. Pseudocode: `C = (A * a_scale) * (B * b_scale) + bias`.
