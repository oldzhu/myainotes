# vLLM 深入：`_custom_ops.py` → 原生内核（`csrc/`）→ 推理加速

[English](vllm-custom-ops-and-native-kernels-deep-dive.md) | [简体中文](vllm-custom-ops-and-native-kernels-deep-dive.zh-CN.md)

本笔记把 **vLLM 的 Python↔原生“胶水层”** 映射到 GPT 推理的关键组件，并解释原生内核如何注册/被调用，以及为什么比“纯 PyTorch”更快。

范围：
- `vllm/_custom_ops.py` 中的自定义算子接口
- 这些算子如何通过 `torch.ops.*` 暴露，并在 Python 中被调用
- CUDA + ROCm + CPU 相关细节（包含内存布局与分发）
- 与编译/图捕获的关系（必要处）

---

## 0) 心智模型（1 分钟）

vLLM 的最热推理路径主要由以下部分主导：
- **KV cache 写入**（prefill 与 decode 阶段）：把 K/V 写入 *分页/分块* 缓存
- **Attention 读取**（尤其是 decode）：高效读取 K/V，避免多次 gather/copy
- **带宽受限的逐元素**操作：RMSNorm、RoPE、activation（SiLU·mul）
- **GEMM**（QKV、MLP、输出）：常常量化，需要特殊布局

`vllm/_custom_ops.py` 是“单一 Python 门面”，负责：
1) 强制加载 C++/CUDA 扩展，
2) 为注册算子提供可调用的 Python 封装，
3) 为编译工具提供 fake/meta 实现。

---

## 1) `_custom_ops.py` 实际在做什么

### 1.1 import 时加载内核
`_custom_ops.py` 在 import 时执行：

- `current_platform.import_kernels()`（见 `vllm/platforms/interface.py`）

这会尝试导入编译好的扩展模块：
- `import vllm._C`（主扩展；注册**大多数**算子）
- `import vllm._moe_C`（可选；MoE/Marlin 相关算子）
- ROCm 会增加 `import vllm._rocm_C`（见 `vllm/platforms/rocm.py`）

一旦导入，C++ 注册代码就会运行，算子会出现在 `torch.ops.*` 下。

### 1.2 封装函数
`_custom_ops.py` 中的大多数函数是 `torch.ops.<namespace>.<op>` 的**薄封装**。

示例：
- Attention：`paged_attention_v1/v2`, `merge_attn_states`
- KV cache：`reshape_and_cache`, `reshape_and_cache_flash`（命名空间为 `_C_cache_ops`）
- CPU attention：`cpu_attention_with_kv_cache`, `cpu_attn_reshape_and_cache`
- RoPE：`rotary_embedding`
- Norm：`rms_norm`, `fused_add_rms_norm`
- Activation：`silu_and_mul`
- 量化 GEMM/打包：`cutlass_scaled_mm`, `marlin_gemm`, GPTQ/AWQ 相关辅助等

### 1.3 Fake/meta 注册
`_custom_ops.py` 会有条件地导入 `torch.library.register_fake` 并为部分算子定义 fake 实现。

意义：
- `torch.compile` / Inductor / AOTAutograd 需要**形状传播**与“抽象执行”。
- 对于返回 tensor 的算子，fake 实现返回形状正确的占位 tensor。
- 对于 in-place / `-> ()` 的算子，fake 实现通常是 no-op。

编译支持的显式引用可见：
- `vllm/compilation/fix_functionalization.py`
- `vllm/compilation/matcher_utils.py`

---

## 2) 原生算子如何注册（以及命名空间为何“奇怪”）

主要 C++ 注册入口：
- `csrc/torch_bindings.cpp`

它使用：
- `TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) { ... }`

如果 `TORCH_EXTENSION_NAME` 是 `_C`（通常如此），则：
- 此块中注册的算子会出现在 `torch.ops._C.<op_name>`

vLLM 还会通过拼接后缀注册子库：

- `TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cache_ops), cache_ops)`
  - 变为 `torch.ops._C_cache_ops.<op_name>`
  - 如 `reshape_and_cache`, `reshape_and_cache_flash`, `swap_blocks`, …

CPU 上 `csrc/cpu/torch_bindings.cpp` 注册：
- CPU attention 算子在 `torch.ops._C.*`
- 以及 `torch.ops._C_cpu.*`：
  - `TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cpu), cpu_ops)`
  - 例如 `mla_decode_kvcache` → `torch.ops._C_cpu.mla_decode_kvcache`

ROCm 有独立注册文件：
- `csrc/rocm/torch_bindings.cpp`
- 由 `vllm._rocm_C` 导入，通常暴露为 `torch.ops._rocm_C.*`

---

## 3) 映射：自定义算子 → GPT 推理组件

下面是理解推理性能最有用的“谁调用谁”地图。

### 3.1 KV cache 写入路径（vLLM 最核心特性）

**调用位置（CUDA 路径）：**
- `vllm/v1/attention/backends/fa_utils.py` 导入 `reshape_and_cache_flash`
- `vllm/v1/attention/backends/flash_attn.py` 调用 `reshape_and_cache_flash(...)`
- `vllm/v1/attention/backends/flashinfer.py` 也调用 `torch.ops._C_cache_ops.reshape_and_cache_flash(...)`
- `vllm/v1/attention/backends/tree_attn.py` 使用 `ops.reshape_and_cache_flash(...)`
- `vllm/v1/attention/backends/flex_attention.py` 调用 `torch.ops._C_cache_ops.reshape_and_cache_flash(...)`

**调用位置（CPU attention backend）：**
- `vllm/v1/attention/backends/cpu_attn.py` 调用 `ops.cpu_attn_reshape_and_cache(...)`

**它做什么：**
- 接收新计算出的 K/V（通常形状 `[num_tokens, num_kv_heads, head_dim]`）
- 将其写入分页/分块 KV cache（形状类似 `[num_blocks, num_kv_heads, block_size, head_dim]`）
- 使用 `slot_mapping`（每个 token 的“slot 地址”）

#### 3.1.1 三个关键“寻址”张量

这些名字在 Python 元数据对象和内核签名中都会出现：

- `slot_mapping`：**写入地址**（每个 token 的目标 slot）。
  - 概念上 `slot = block_id * block_size + offset`（或等价编码）。

- `block_table` / `block_tables`：**每个序列的读取地址**。
  - 每个序列的物理 cache block 列表。
  - 使 attention 读取 KV 无需连续存储。

- `seq_lens`：**每个序列需要注意的历史长度**。
  - 通常是 int32 长度；用于 mask attention 并限制循环。

经验法则：
- 若算子**写 KV**，通常需要 `slot_mapping`
- 若算子**读 KV 做 attention**，通常需要 `block_tables` + `seq_lens`（有时还有 `query_start_loc`）

#### 3.1.2 KV cache 打包（为何形状看起来很奇怪）

在 CUDA 类平台，vLLM 常用打包布局提升内存合并与向量化加载。

一个具体例子：`vllm/v1/attention/ops/paged_attn.py::PagedAttention.split_kv_cache`：
- `x = 16 // kv_cache.element_size()`（每 16 字节的元素数）
- `key_cache` 视图：`[num_blocks, num_kv_heads, head_size // x, *, x]`
- `value_cache` 视图：`[num_blocks, num_kv_heads, head_size, *]`

如果你遇到内核签名不匹配，通常是因为 Python 传入的是**内核期望的视图**，而不是“逻辑形状”。

**原生实现（CUDA）：**
- schema + dispatch：`csrc/torch_bindings.cpp`（`_C_cache_ops.reshape_and_cache(_flash)`）
- kernel：`csrc/cache_kernels.cu`（`reshape_and_cache_kernel`, `reshape_and_cache_flash_kernel`）

**为什么比“朴素 PyTorch”快：**
- 避免 Python 级 scatter/gather 与多次 view/transpose
- 一次内核直接写入最终 KV cache 布局
- 可在内核内处理 dtype/量化细节（`kv_cache_dtype`, `k_scale`, `v_scale`）

流程示意：

```text
Model forward
  -> project to Q,K,V
  -> RoPE (in-place)
  -> KV write: reshape_and_cache(_flash)
       (slot_mapping decides where each token lands)
  -> attention reads K/V from paged cache
```

### 3.2 Attention 读取/计算

v1 中 CUDA attention 通常由 **FlashAttention / FlashInfer / Triton** 后端处理，而不是 vLLM 的 legacy `paged_attention_v1/v2` 入口。

ROCm 有专用路径可使用 vLLM 的 ROCm paged attention：
- `vllm/v1/attention/ops/chunked_prefill_paged_decode.py` 调用 `ops.paged_attention_rocm(...)`
- 自定义封装：`vllm/_custom_ops.py::paged_attention_rocm`
- 原生算子是 `torch.ops._rocm_C.paged_attention`

vLLM 也暴露 CUDA paged attention API：
- `torch.ops._C.paged_attention_v1` / `torch.ops._C.paged_attention_v2`
- 在 `csrc/torch_bindings.cpp` 注册，并在 `csrc/attention/paged_attention_v2.cu`（v2）实现

即使当前选用后端不调用它们，也值得理解，因为：
- 它们是“分页 KV cache 读 + attention 计算”的经典示例
- 展示了调度器与 cache 管理器中常见的数据结构（`block_tables`, `seq_lens` 等）

### 3.3 RoPE / 位置编码

RoPE 是**带宽受限**操作，极适合 in-place 向量化内核。

调用处：
- `vllm/model_executor/layers/rotary_embedding/base.py`：
  - CUDA 优先 `torch.ops.vllm.flashinfer_rotary_embedding`
  - 否则使用 `ops.rotary_embedding(...)`（in-place）

为什么快：
- 避免 Python 切片/拼接与额外分配
- 单核融合“half 维旋转 + 组合”模式
- 便于捕获/编译，因为它就地修改预分配张量

### 3.4 归一化（RMSNorm 与融合 add+norm）

调用处：
- `vllm/model_executor/layers/layernorm.py` 使用：
  - `ops.rms_norm(out, x, weight, eps)`
  - `ops.fused_add_rms_norm(x, residual, weight, eps)`（in-place）

为什么快：
- 单内核完成多个逐元素操作/归约
- 融合 residual-add 降低读写次数

ROCm 说明：
- layernorm 分发可选择 `rocm_aiter_ops.rms_norm` 或 `rms_norm2d_with_add`（启用 AITER 时）。

### 3.5 激活融合（SwiGLU / SiLU·mul）

自定义算子：
- `torch.ops._C.silu_and_mul`（在 `csrc/torch_bindings.cpp` 注册）

调用位置：
- 在模型块实现 SwiGLU 时使用，也被编译融合器引用（`vllm/compilation/*`）。

为什么快：
- 融合激活 + 乘法（只读一次输入、写一次输出）
- 减少 kernel 启动与中间张量

### 3.6 量化 + GEMM 内核

这是 `_custom_ops.py` 中很大的一部分，也是后端差异最大的部分。

典型调用处：
- `vllm/model_executor/layers/quantization/*.py`
  - AWQ：`ops.awq_dequantize`, `ops.awq_gemm`
  - GPTQ：`ops.gptq_shuffle`, `ops.gptq_gemm`
  - Marlin：`ops.marlin_*`, `ops.gptq_marlin_*`, `ops.awq_marlin_*`
  - Cutlass scaled MM：`ops.cutlass_scaled_mm(...)`

为什么快：
- 使用专用布局（打包 int4/8 权重、块级 scale/zero）
- 分发到匹配 GPU 架构能力的内核
- 避免先解量化到 fp16/fp32 再 matmul 的带宽浪费

---

## 4) CPU 深入：内存布局、分发与“融合”策略

CPU 性能主要依赖：
- 选择正确 **ISA**（AMX/AVX512/NEON 等）
- 线性、可预测的内存访问
- 降低 decode 内循环开销

### 4.1 KV cache 布局
CPU attention backend 中，KV cache 形状为：
- `[2, num_blocks, num_kv_heads, block_size, head_dim]`（之后解绑为 K 与 V）

`cpu_attn_reshape_and_cache` 写入：
- `key_cache`: `[num_blocks, num_kv_heads, block_size, head_dim]`
- `value_cache`: `[num_blocks, num_kv_heads, block_size, head_dim]`

C++ 实现（`csrc/cpu/cpu_attn.cpp`）要求：
- `key.stride(2) == 1` 且 `value.stride(2) == 1`（head_dim 连续）

### 4.2 ISA + head_dim 分发
CPU attention 使用两级分发：
- head_dim 必须在 {32, 64, 80, 96, 112, 128, 160, 192, 224, 256}
- ISA 通过字符串提示选择：`amx`, `vec`, `vec16`, `neon`

见 `csrc/cpu/cpu_attn.cpp` 的分发宏。

这样能让最热循环内无分支，保持紧凑。

### 4.3 调度元数据（decode tiling）
CPU backend 预计算调度元数据：
- `torch.ops._C.get_scheduler_metadata(...)`

调用处：
- CPU attention backend 生成 metadata 后传给
- `torch.ops._C.cpu_attention_with_kv_cache(...)`

关键技巧：将“怎么切 tile/拆分”的规划提前做完，内核保持紧凑。

### 4.4 CPU 上的“融合”
CPU attention 通常不依赖编译器大融合，而是：
- 使用专用 attention 内核（`cpu_attention_with_kv_cache`）
- 融合 MoE 内核（AVX512 可用时）
- 少量 GEMM/量化内核

`torch.compile` 仍可用于其它部分，但 attention 本身已经是单原生算子。

---

## 5) 图捕获 / 编译说明（实践视角）

vLLM 使用多种策略：
- **CUDA graph capture** 用于稳定 decode（形状相对固定）
- **torch.compile**（Inductor）用于部分图 + 自定义模式融合

对自定义算子而言有两点关键：
1) **API 形状**：很多算子是 `-> ()` 且写入预分配输出，有利于 capture。
2) **编译友好性**：因为自定义 in-place 会影响 AOT 工具，需要 fake/meta 与 functionalization 修复。

若遇到编译/capture 问题，`vllm/compilation/fix_functionalization.py` 往往解释了“为何被 special-case”。

---

## 6) 没有自定义算子会怎样？（性能直觉）

一种有用的思考方式：

- **带宽受限**（RoPE、RMSNorm、SiLU·mul）：
  - 减少内存往返
  - 减少中间张量
  - 减少 kernel 启动

- **布局/打包受限**（KV 写入、量化 GEMM）：
  - 直接写入所需布局
  - 保持打包数据并使用专用数学内核

具体对比：
- `reshape_and_cache_flash` vs 朴素 PyTorch：
  - 朴素做法通常需要 view/transpose + scatter，多次 launch，还会分配临时张量
  - vLLM 内核直接 indexed 写入 paged cache

- `fused_add_rms_norm` vs `x = x + residual; rms_norm(x)`：
  - 朴素：2+ kernel + 中间张量（或至少额外读写）
  - 融合：单 kernel、带宽更省

- 量化 GEMM vs `dequantize -> matmul`：
  - 朴素：先解量化到 fp16/fp32 再 GEMM（浪费带宽/内存）
  - 自定义：保持打包权重，内核内完成缩放

---

## 7) 贡献指南：新增或修改自定义算子

最小且可靠的流程：

1) **先找 Python 调用点**
   - 在 `vllm/model_executor/` 与 `vllm/v1/attention/` 中搜索 `from vllm import _custom_ops as ops`

2) **定位算子注册**
   - CUDA/通用：`csrc/torch_bindings.cpp`
   - CPU：`csrc/cpu/torch_bindings.cpp`
   - ROCm：`csrc/rocm/torch_bindings.cpp`

3) **定位内核实现**
   - attention：`csrc/attention/*`
   - KV cache：`csrc/cache_kernels*.cu`
   - norm：`csrc/layernorm_kernels.cu` / `csrc/layernorm_quant_kernels.cu`
   - quantization：`csrc/quantization/*`
   - CPU attention：`csrc/cpu/cpu_attn*.{cpp,hpp}`

4) **更新 `_custom_ops.py` 封装**（若需要 Python 层 API）
   - 参数顺序/语义与 `ops.def(...)` 保持一致

5) **添加测试/基准**
   - 内核正确性测试通常在 `tests/` 下（先搜索已有测试）
   - 性能 sanity：新内核或新变体建议加 `benchmarks/` 下的小基准

贡献者易踩坑清单：
- schema 不匹配（Python wrapper 与 `ops.def(...)` 签名不同）
- 缺少 fake/meta 实现（编译路径会坏）
- 输入不连续（许多内核要求 `stride(-1)==1`）
- 算子内部有捕获不友好的分配

---

## 8) 接下来该看哪里

若想追踪 v1 attention 端到端：
- attention backend 选择：`vllm/v1/attention/selector.py`
- 具体后端实现：`vllm/v1/attention/backends/*.py`
- KV-cache 辅助：`vllm/v1/attention/backends/fa_utils.py`

若想专注 Python↔原生边界：
- 封装面：`vllm/_custom_ops.py`
- 注册位置：`csrc/torch_bindings.cpp` 与 `csrc/cpu/torch_bindings.cpp`

若想理解 CPU 性能开关：
- CPU attention backend：`vllm/v1/attention/backends/cpu_attn.py`
- C++ 分发 + 布局：`csrc/cpu/cpu_attn.cpp`

---

## 附录：快速 sanity 检查（算子是否加载）

当你不确定 wheel/build 是否包含扩展时，这些检查很有用。

检查主扩展是否加载，以及关键算子是否存在：

```bash
python -c "import vllm, torch; import vllm._custom_ops as ops; \
print('reshape_and_cache_flash:', torch.ops._C_cache_ops.reshape_and_cache_flash); \
print('rms_norm:', torch.ops._C.rms_norm); \
print('cpu_attention_with_kv_cache:', torch.ops._C.cpu_attention_with_kv_cache)"
```

如果某个算子缺失，通常意味着：
- 扩展模块导入失败（看 `current_platform.import_kernels()` 的警告），或
- 该算子被平台/架构条件编译排除。

