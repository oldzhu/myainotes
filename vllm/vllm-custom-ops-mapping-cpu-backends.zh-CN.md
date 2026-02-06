# vLLM 自定义算子映射表：CPU 后端阶段

[English](vllm-custom-ops-mapping-cpu-backends.md) | [简体中文](vllm-custom-ops-mapping-cpu-backends.zh-CN.md)

范围：CPU 注意力内核与 oneDNN 矩阵乘法算子。

## 映射表：CPU 后端

| 算子 | Python 调用点 + GPT 推理用途 | 原生实现 + 说明/伪代码 |
|---|---|---|
| `cpu_attn_get_scheduler_metadata` | CPU 注意力后端调用 [vllm/vllm/v1/attention/backends/cpu_attn.py](vllm/vllm/v1/attention/backends/cpu_attn.py#L177)。用途：计算 CPU attention 的调度元数据。 | CPU 内核在 [vllm/csrc/cpu/cpu_attn.cpp](vllm/csrc/cpu/cpu_attn.cpp#L3)。计算 tile 与调度参数。伪代码：`metadata = calc_schedule(seqlens, heads)`。
| `cpu_attn_reshape_and_cache` | CPU prefill 调用 [vllm/vllm/v1/attention/backends/cpu_attn.py](vllm/vllm/v1/attention/backends/cpu_attn.py#L326)。用途：reshape QKV 并写入 CPU KV cache。 | CPU 内核在 [vllm/csrc/cpu/cpu_attn.cpp](vllm/csrc/cpu/cpu_attn.cpp#L66)。将 K/V 写入缓存布局。伪代码：`cache_kv(key, value, slots)`。
| `cpu_attention_with_kv_cache` | CPU decode 调用 [vllm/vllm/v1/attention/backends/cpu_attn.py](vllm/vllm/v1/attention/backends/cpu_attn.py#L349)。用途：在 CPU KV cache 上计算 attention。 | CPU 内核在 [vllm/csrc/cpu/cpu_attn.cpp](vllm/csrc/cpu/cpu_attn.cpp#L122)。执行 attention 主循环。伪代码：`softmax(QK^T) * V`。
| `create_onednn_mm` | 设置 CPU 线性层 handler 调用 [vllm/vllm/model_executor/layers/utils.py](vllm/vllm/model_executor/layers/utils.py#L261)。用途：创建 oneDNN GEMM handler。 | CPU 入口在 [vllm/csrc/cpu/dnnl_kernels.cpp](vllm/csrc/cpu/dnnl_kernels.cpp#L497)。准备 oneDNN primitive 并返回 handler。伪代码：`handler = dnnl_prepare(weight)`。
| `onednn_mm` | CPU 线性层调用 [vllm/vllm/model_executor/layers/utils.py](vllm/vllm/model_executor/layers/utils.py#L262)。用途：执行 oneDNN GEMM（可带 bias）。 | CPU 入口在 [vllm/csrc/cpu/dnnl_kernels.cpp](vllm/csrc/cpu/dnnl_kernels.cpp#L519)。执行 oneDNN matmul。伪代码：`C = A * B + bias`。
| `create_onednn_scaled_mm` | CPU scaled MM 调用 [vllm/vllm/model_executor/layers/quantization/kernels/scaled_mm/cpu.py](vllm/vllm/model_executor/layers/quantization/kernels/scaled_mm/cpu.py#L127)。用途：创建 scaled GEMM 的 oneDNN handler。 | CPU 入口在 [vllm/csrc/cpu/dnnl_kernels.cpp](vllm/csrc/cpu/dnnl_kernels.cpp#L303)。准备 scaled matmul primitive。伪代码：`handler = dnnl_prepare_scaled(weight)`。
| `onednn_scaled_mm` | CPU scaled MM 调用 [vllm/vllm/model_executor/layers/quantization/kernels/scaled_mm/cpu.py](vllm/vllm/model_executor/layers/quantization/kernels/scaled_mm/cpu.py#L199)。用途：执行 oneDNN scaled GEMM。 | CPU 入口在 [vllm/csrc/cpu/dnnl_kernels.cpp](vllm/csrc/cpu/dnnl_kernels.cpp#L354)。执行带 scale/zero-point 的 matmul。伪代码：`C = (A * a_scale) * (B * b_scale) + bias`。
