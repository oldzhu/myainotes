# vLLM 自定义算子映射表：归一化 + RoPE + 激活阶段

[English](vllm-custom-ops-mapping-norm-activation.md) | [简体中文](vllm-custom-ops-mapping-norm-activation.zh-CN.md)

范围：Transformer 块内部的归一化、位置编码和激活相关自定义算子。

## 映射表：归一化 + RoPE + 激活

| 算子 | Python 调用点 + GPT 推理用途 | 原生实现 + 说明/伪代码 |
|---|---|---|
| `rms_norm` | 封装见 [vllm/vllm/_custom_ops.py](vllm/vllm/_custom_ops.py#L332)；RMSNorm 层调用 [vllm/vllm/model_executor/layers/layernorm.py](vllm/vllm/model_executor/layers/layernorm.py#L18-L31)。用途：Transformer 归一化，$y = x / \sqrt{\text{mean}(x^2)+\epsilon} * w$。 | CUDA 内核在 [vllm/csrc/layernorm_kernels.cu](vllm/csrc/layernorm_kernels.cu)。先做方差归约，再缩放并乘权重；对 FP16/BF16 做向量化。伪代码：`var=mean(x^2); y = x * rsqrt(var+eps) * w`。
| `fused_add_rms_norm` | 封装见 [vllm/vllm/_custom_ops.py](vllm/vllm/_custom_ops.py)；用于残差 + RMSNorm 融合路径 [vllm/vllm/model_executor/layers/layernorm.py](vllm/vllm/model_executor/layers/layernorm.py)。用途：把残差相加与 RMSNorm 合并为一次内存遍历。 | CUDA 内核在 [vllm/csrc/layernorm_kernels.cu](vllm/csrc/layernorm_kernels.cu)（`fused_add_rms_norm_kernel`）。先计算 $x \leftarrow x + r$，再执行 RMSNorm。伪代码：`x = x + r; var=mean(x^2); x = x * rsqrt(var+eps) * w`。
| `rotary_embedding` | 封装见 [vllm/vllm/_custom_ops.py](vllm/vllm/_custom_ops.py#L318)；RoPE 前向调用 [vllm/vllm/model_executor/layers/rotary_embedding/base.py](vllm/vllm/model_executor/layers/rotary_embedding/base.py#L200-L225)。用途：对 Q/K 应用 RoPE，使用缓存的 $\cos$/$\sin$ 旋转前 `rot_dim`。 | CUDA 内核在 [vllm/csrc/pos_encoding_kernels.cu](vllm/csrc/pos_encoding_kernels.cu)。支持 GPT‑NeoX 或 GPT‑J 风格旋转，原地更新 Q/K。伪代码：`x' = x*cos - y*sin; y' = y*cos + x*sin`。
| `silu_and_mul` | SwiGLU 激活使用 [vllm/vllm/model_executor/layers/activation.py](vllm/vllm/model_executor/layers/activation.py#L115-L150)。用途：MLP gating，$\text{silu}(x)=x\sigma(x)$，输出 $\text{silu}(x_1) * x_2$。 | CUDA 内核在 [vllm/csrc/activation_kernels.cu](vllm/csrc/activation_kernels.cu)。对齐时使用 128-bit 向量化加载；计算 silu 后相乘（或 `mul_and_silu` 反向顺序）。伪代码：`out[i] = silu(x[i]) * y[i]`。
| `mul_and_silu` | SwiGLU 变体 [vllm/vllm/model_executor/layers/activation.py](vllm/vllm/model_executor/layers/activation.py)。用途：$x_1 * \text{silu}(x_2)$（顺序相反）用于门控 MLP。 | CUDA 内核在 [vllm/csrc/activation_kernels.cu](vllm/csrc/activation_kernels.cu)。与 `silu_and_mul` 同族内核，仅调整激活顺序。伪代码：`out[i] = x[i] * silu(y[i])`。
| `gelu_and_mul` | GeGLU 激活 [vllm/vllm/model_executor/layers/activation.py](vllm/vllm/model_executor/layers/activation.py)。用途：$\text{GELU}(x_1) * x_2$。 | CUDA 内核在 [vllm/csrc/activation_kernels.cu](vllm/csrc/activation_kernels.cu)（`gelu_and_mul`）。使用精确 GELU（erf）后相乘。伪代码：`out[i] = gelu(x[i]) * y[i]`。
| `gelu_tanh_and_mul` | GeGLU 的 tanh 近似 [vllm/vllm/model_executor/layers/activation.py](vllm/vllm/model_executor/layers/activation.py)。用途：$\text{GELU}_\tanh(x_1) * x_2$（近似加速）。 | CUDA 内核在 [vllm/csrc/activation_kernels.cu](vllm/csrc/activation_kernels.cu)（`gelu_tanh_and_mul`）。伪代码：`out[i] = gelu_tanh(x[i]) * y[i]`。
| `fatrelu_and_mul` | FATReLU 门控 [vllm/vllm/model_executor/layers/activation.py](vllm/vllm/model_executor/layers/activation.py)。用途：先对 gate 做阈值 ReLU，再与另一半相乘，$\max(x_1, t) * x_2$。 | CUDA 内核在 [vllm/csrc/activation_kernels.cu](vllm/csrc/activation_kernels.cu)（`act_and_mul_kernel_with_param` + `fatrelu_kernel`）。伪代码：`out[i] = max(x[i], threshold) * y[i]`。
| `swigluoai_and_mul` | GPT‑OSS 的 SwiGLU 变体 [vllm/vllm/model_executor/layers/activation.py](vllm/vllm/model_executor/layers/activation.py)。用途：带 clamp 的 OpenAI 风格 SwiGLU（alpha/limit）。 | CUDA 内核在 [vllm/csrc/activation_kernels.cu](vllm/csrc/activation_kernels.cu)（`swigluoai_and_mul_kernel`）。伪代码：`gate=clamp(gate); up=clamp(up); out=(up+1) * gate * sigmoid(gate*alpha)`。
