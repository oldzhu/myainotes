# vLLM deep dive: `_custom_ops.py` → native kernels (`csrc/`) → inference speed

This note maps **vLLM’s Python↔native “glue” layer** to concrete GPT inference components and explains (at a contributor level) how the native kernels are registered, called, and why they’re faster than “plain PyTorch” equivalents.

Scope:
- vLLM’s custom op surface in `vllm/_custom_ops.py`
- How ops are exposed via `torch.ops.*` and where they’re used from Python
- CUDA + ROCm + CPU specifics (including memory layout and dispatch)
- How this interacts with compilation / graph capture (where relevant)

---

## 0) Mental model (1 minute)

vLLM’s hottest inference path is dominated by:
- **KV cache writes** (during prefill and decode): put K/V into a *paged/block* cache
- **Attention reads** (especially decode): read K/V efficiently, avoid gathers/copies
- **Bandwidth-bound elementwise** ops: RMSNorm, RoPE, activation(SiLU·mul)
- **GEMMs** (QKV, MLP, output): often quantized, requiring special layouts

`vllm/_custom_ops.py` is the “single Python façade” that:
1) forces the C++/CUDA extensions to load,
2) provides Python-callable wrappers around the registered ops,
3) provides *fake/meta* implementations for compiler tooling.

---

## 1) What `_custom_ops.py` actually does

### 1.1 Import-time kernel loading
At import time, `_custom_ops.py` runs:

- `current_platform.import_kernels()` (see `vllm/platforms/interface.py`)

That attempts to import the compiled extension module(s):
- `import vllm._C` (main extension; registers **most** operators)
- `import vllm._moe_C` (optional; MoE/Marlin-related operators)
- ROCm adds `import vllm._rocm_C` (see `vllm/platforms/rocm.py`)

Once imported, the C++ registration hooks run, and ops become available under `torch.ops.*`.

### 1.2 Wrapper functions
Most functions in `_custom_ops.py` are **thin wrappers** around `torch.ops.<namespace>.<op>`.

Examples:
- Attention: `paged_attention_v1/v2`, `merge_attn_states`
- KV cache: `reshape_and_cache`, `reshape_and_cache_flash` (note the namespace is `_C_cache_ops`)
- CPU attention: `cpu_attention_with_kv_cache`, `cpu_attn_reshape_and_cache`
- RoPE: `rotary_embedding`
- Norm: `rms_norm`, `fused_add_rms_norm`
- Activation: `silu_and_mul`
- Quantized GEMMs / packing: `cutlass_scaled_mm`, `marlin_gemm`, GPTQ/AWQ helpers, etc.

### 1.3 Fake/meta registrations
`_custom_ops.py` conditionally imports `torch.library.register_fake` and defines fake implementations for some ops.

Why this matters:
- `torch.compile` / Inductor / AOTAutograd need **shape propagation** and “abstract” execution.
- For ops that return tensors, fake impls return correctly-shaped placeholder tensors.
- For in-place / `-> ()` ops, fake impls are typically no-ops.

You can see vLLM’s compile support explicitly referencing these ops in:
- `vllm/compilation/fix_functionalization.py`
- `vllm/compilation/matcher_utils.py`

---

## 2) How native ops are registered (and why the namespaces look weird)

The main C++ registration entrypoint is:
- `csrc/torch_bindings.cpp`

It uses `TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) { ... }`.

If `TORCH_EXTENSION_NAME` is `_C` (which it typically is), then:
- ops registered in that block become `torch.ops._C.<op_name>`

vLLM also registers *sub-libraries* by concatenating suffixes:

- `TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cache_ops), cache_ops)`
  - becomes `torch.ops._C_cache_ops.<op_name>`
  - e.g. `reshape_and_cache`, `reshape_and_cache_flash`, `swap_blocks`, …

On CPU, `csrc/cpu/torch_bindings.cpp` registers:
- CPU attention ops under `torch.ops._C.*` (same `_C` namespace)
- and a small group under `torch.ops._C_cpu.*` via:
  - `TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cpu), cpu_ops)`
  - e.g. `mla_decode_kvcache` → `torch.ops._C_cpu.mla_decode_kvcache`

ROCm has its own extension registration file:
- `csrc/rocm/torch_bindings.cpp`
- imported by `vllm._rocm_C`, typically exposed as `torch.ops._rocm_C.*`

---

## 3) Mapping: custom ops → GPT inference components

Below is the most useful “what calls what” map for understanding real inference speed.

### 3.1 KV cache write path (the *most* vLLM-specific part)

**Where it’s called from (CUDA-ish paths):**
- `vllm/v1/attention/backends/fa_utils.py` imports `reshape_and_cache_flash`
- `vllm/v1/attention/backends/flash_attn.py` calls `reshape_and_cache_flash(...)`
- `vllm/v1/attention/backends/flashinfer.py` also calls `torch.ops._C_cache_ops.reshape_and_cache_flash(...)`
- `vllm/v1/attention/backends/tree_attn.py` uses `ops.reshape_and_cache_flash(...)`
- `vllm/v1/attention/backends/flex_attention.py` calls `torch.ops._C_cache_ops.reshape_and_cache_flash(...)`

**Where it’s called from (CPU attention backend):**
- `vllm/v1/attention/backends/cpu_attn.py` calls `ops.cpu_attn_reshape_and_cache(...)`

**What it does:**
- takes freshly computed K/V (usually shaped `[num_tokens, num_kv_heads, head_dim]`)
- and writes them into the block/paged KV cache (shaped like `[num_blocks, num_kv_heads, block_size, head_dim]`)
- using `slot_mapping` (a per-token mapping of “token → (block_id, offset)”)

#### 3.1.1 The three key “addressing” tensors you’ll keep seeing

These names show up in both Python metadata objects and kernel signatures:

- `slot_mapping`: **write address** for each incoming token.
  - Conceptually, each token gets a “slot id” in the KV cache.
  - Most kernels treat it like `slot = block_id * block_size + offset` (or an equivalent encoding).

- `block_table` / `block_tables`: **read address** per sequence.
  - For each sequence/request, it lists which physical cache blocks contain that sequence’s tokens.
  - This is how attention reads KV without requiring the cache to be contiguous.

- `seq_lens`: **how much history to attend to** per sequence.
  - Usually int32 lengths; used to mask attention and bound loops.

When you’re mapping Python → native, a good heuristic is:
- if the op **writes KV**, expect `slot_mapping`
- if the op **reads KV for attention**, expect `block_tables` + `seq_lens` (and sometimes `query_start_loc`)

#### 3.1.2 KV cache packing (why key/value shapes look odd)

On CUDA-alike platforms, vLLM often stores KV in a packed layout for better memory coalescing and vectorized loads.

One concrete example is the helper in `vllm/v1/attention/ops/paged_attn.py::PagedAttention.split_kv_cache`, which derives:
- `x = 16 // kv_cache.element_size()` (number of elements per 16-byte chunk)
- `key_cache` view: `[num_blocks, num_kv_heads, head_size // x, *, x]`
- `value_cache` view: `[num_blocks, num_kv_heads, head_size, *]`

If you’re debugging a kernel signature mismatch, it’s often because the Python side is passing the *view* expected by the kernel, not the “logical” KV cache shape.

**Native implementation (CUDA):**
- schema + dispatch: `csrc/torch_bindings.cpp` (`_C_cache_ops.reshape_and_cache(_flash)`)
- kernel: `csrc/cache_kernels.cu` (`reshape_and_cache_kernel`, `reshape_and_cache_flash_kernel`)

**Why it’s fast vs naive PyTorch:**
- avoids Python-level scatter/gather and multiple view/transpose ops
- writes *directly* into the final KV cache layout in one kernel
- can incorporate KV-cache dtype/quantization details (`kv_cache_dtype`, `k_scale`, `v_scale`) without intermediate tensors

Flow sketch:

```text
Model forward
  -> project to Q,K,V
  -> RoPE (in-place)
  -> KV write: reshape_and_cache(_flash)
       (slot_mapping decides where each token lands)
  -> attention reads K/V from paged cache
```

### 3.2 Attention read/compute

In v1, CUDA attention is typically handled by **FlashAttention / FlashInfer / Triton** backends, not by vLLM’s legacy CUDA `paged_attention_v1/v2` entrypoints.

ROCm has a dedicated path that can use vLLM’s ROCm paged attention op:
- `vllm/v1/attention/ops/chunked_prefill_paged_decode.py` calls `ops.paged_attention_rocm(...)`
- custom op wrapper: `vllm/_custom_ops.py::paged_attention_rocm`
- native op is `torch.ops._rocm_C.paged_attention`

There is also a vLLM CUDA paged attention API exposed as:
- `torch.ops._C.paged_attention_v1` / `torch.ops._C.paged_attention_v2`
- registered in `csrc/torch_bindings.cpp` and implemented in `csrc/attention/paged_attention_v2.cu` (v2)

Even if your selected backend doesn’t call these today, they’re still important to understand because:
- they’re the canonical example of “paged KV cache read + attention compute” in native code
- they illustrate the data structures you see throughout the scheduler + cache manager (`block_tables`, `seq_lens`, etc.)

### 3.3 RoPE / positional encoding

RoPE is **bandwidth-bound** and benefits heavily from in-place, vectorized kernels.

Callsite:
- `vllm/model_executor/layers/rotary_embedding/base.py`:
  - on CUDA it prefers `torch.ops.vllm.flashinfer_rotary_embedding` if enabled
  - otherwise it uses `ops.rotary_embedding(...)` (in-place)

Why it’s fast:
- avoids Python slicing/cat overhead and extra allocations
- fuses the “rotate half dims + combine” pattern in one kernel
- is capture/compile friendly because it mutates preallocated tensors

### 3.4 Normalization (RMSNorm and fused add+norm)

Callsite:
- `vllm/model_executor/layers/layernorm.py` uses:
  - `ops.rms_norm(out, x, weight, eps)`
  - `ops.fused_add_rms_norm(x, residual, weight, eps)` (in-place)

Why it’s fast:
- single kernel instead of multiple elementwise reductions + broadcasts
- fused residual-add reduces memory reads/writes

ROCm note:
- layernorm dispatch can choose `rocm_aiter_ops.rms_norm` or `rms_norm2d_with_add` when AITER is enabled.

### 3.5 Activation fusion (SwiGLU / SiLU·mul)

Custom op:
- `torch.ops._C.silu_and_mul` (registered in `csrc/torch_bindings.cpp`)

Where it shows up:
- indirectly used by model blocks that implement SwiGLU, and by compilation fusers (`vllm/compilation/*`).

Why it’s fast:
- fuses activation + multiply (one read of input, one write of output)
- reduces kernel launches and intermediate tensors

### 3.6 Quantization + GEMM kernels

This is a big part of `_custom_ops.py`, and it’s where you’ll see the most backend diversity.

Typical callsites:
- `vllm/model_executor/layers/quantization/*.py`
  - AWQ: `ops.awq_dequantize`, `ops.awq_gemm`
  - GPTQ: `ops.gptq_shuffle`, `ops.gptq_gemm`
  - Marlin: `ops.marlin_*`, `ops.gptq_marlin_*`, `ops.awq_marlin_*`
  - Cutlass scaled MM: `ops.cutlass_scaled_mm(...)`

Why they’re fast:
- use specialized layouts (packed int4/8 weights, block scales/zeros)
- dispatch to kernels that match GPU arch capabilities
- avoid dequantizing to fp16/fp32 just to call `torch.matmul`

---

## 4) CPU deep dive: memory layout, dispatch, and “fusion” strategy

CPU performance is primarily about:
- choosing the right **ISA** (AMX/AVX512/NEON variants)
- keeping memory access linear and predictable
- reducing overhead in the inner decode loop

### 4.1 KV cache layout
In the CPU attention backend, KV cache is shaped:
- `[2, num_blocks, num_kv_heads, block_size, head_dim]` (then unbound to K and V)

`cpu_attn_reshape_and_cache` writes into:
- `key_cache`: `[num_blocks, num_kv_heads, block_size, head_dim]`
- `value_cache`: `[num_blocks, num_kv_heads, block_size, head_dim]`

The C++ implementation (`csrc/cpu/cpu_attn.cpp`) enforces:
- `key.stride(2) == 1` and `value.stride(2) == 1` (contiguous head_dim)

### 4.2 ISA + head_dim dispatch
CPU attention uses a two-level dispatch:
- head_dim must be one of {32, 64, 80, 96, 112, 128, 160, 192, 224, 256}
- ISA selected via string hint: `amx`, `vec`, `vec16`, `neon`

See `csrc/cpu/cpu_attn.cpp` for the dispatch macros.

This keeps inner loops specialized and avoids runtime branches inside the hottest loop.

### 4.3 Scheduler metadata (decode tiling)
The CPU backend precomputes schedule metadata via:
- `torch.ops._C.get_scheduler_metadata(...)`

Callsite:
- used by the CPU attention backend metadata path, then fed into:
- `torch.ops._C.cpu_attention_with_kv_cache(...)`

This is a key CPU trick: do the “how should we tile/split the work?” planning once, and keep the kernel tight.

### 4.4 “Fusion” on CPU
vLLM CPU doesn’t generally rely on giant compiler fusions for attention; instead it provides:
- purpose-built attention kernels (`cpu_attention_with_kv_cache`)
- fused MoE kernels (when AVX512 is available)
- select GEMM/quant kernels

Torch.compile can still be useful on CPU for other parts, but attention itself is already expressed as a single native op.

---

## 5) Graph capture / compilation notes (practical view)

vLLM uses multiple strategies:
- **CUDA graph capture** for steady-state decode (fixed-ish shapes)
- **torch.compile** (Inductor) for some graphs, plus custom pattern fusers

For custom ops, two properties matter:
1) **API shape:** many ops are `-> ()` and write into preallocated outputs; this is good for capture.
2) **Compiler friendliness:** fake/meta implementations and functionalization fixes are needed because custom in-place ops can confuse AOT tooling.

If you’re chasing a compile/capture issue, `vllm/compilation/fix_functionalization.py` is a strong “why is this special-cased?” map.

---

## 6) “What would happen without custom ops?” (performance intuition)

A useful way to reason about this:

- If an op is **bandwidth-bound** (RoPE, RMSNorm, SiLU·mul), the win comes from:
  - fewer passes over memory
  - fewer intermediate tensors
  - fewer kernel launches

- If an op is **layout/packing-bound** (KV cache write, quantized GEMMs), the win comes from:
  - writing directly into the required layout
  - keeping data packed and using specialized math kernels

Concrete comparisons:
- `reshape_and_cache_flash` vs naive PyTorch:
  - naive approach often needs views/transposes + scatter; it’s launch-heavy and tends to allocate temporaries
  - vLLM’s kernel does direct indexed writes into paged cache

- `fused_add_rms_norm` vs `x = x + residual; rms_norm(x)`:
  - naive: 2+ kernels and an intermediate tensor (or at least an extra read/write)
  - fused: one kernel, less bandwidth

- quantized GEMMs vs `dequantize -> matmul`:
  - naive dequantizes to fp16/fp32, then calls GEMM (wastes bandwidth and memory)
  - custom kernels keep packed weights and apply scaling in-kernel

---

## 7) Contribution guide: adding or modifying a custom op

A minimal, reliable workflow:

1) **Find the Python callsite first**
   - Search for `from vllm import _custom_ops as ops` in `vllm/model_executor/` and `vllm/v1/attention/`.

2) **Locate the op registration**
   - CUDA/general ops: `csrc/torch_bindings.cpp`
   - CPU ops: `csrc/cpu/torch_bindings.cpp`
   - ROCm ops: `csrc/rocm/torch_bindings.cpp`

3) **Locate the kernel implementation**
   - attention: `csrc/attention/*`
   - KV cache: `csrc/cache_kernels*.cu`
   - norm: `csrc/layernorm_kernels.cu` / `csrc/layernorm_quant_kernels.cu`
   - quantization: `csrc/quantization/*`
   - CPU attention: `csrc/cpu/cpu_attn*.{cpp,hpp}`

4) **Update `_custom_ops.py` wrapper** (if you need a Python-level API)
   - keep args order and semantics aligned with the C++ schema

5) **Add tests/benchmarks**
   - kernel correctness tests typically live under `tests/` (search for existing kernel tests first)
   - perf sanity: add a microbenchmark under `benchmarks/` if it’s a new kernel or a new variant

A contributor pitfall checklist:
- schema mismatch (Python wrapper args don’t match `ops.def(...)` signature)
- missing fake/meta implementation (compiler path breaks)
- non-contiguous inputs (many kernels require `stride(-1)==1`)
- capture-unfriendly allocations inside the op

---

## 8) Quick “where to look next”

If you want to trace the end-to-end attention flow (v1):
- attention backend selection: `vllm/v1/attention/selector.py`
- specific backend implementations: `vllm/v1/attention/backends/*.py`
- KV-cache op helpers: `vllm/v1/attention/backends/fa_utils.py`

If you want to focus on the Python↔native boundary:
- wrapper surface: `vllm/_custom_ops.py`
- op registration: `csrc/torch_bindings.cpp` and `csrc/cpu/torch_bindings.cpp`

If you want to understand CPU performance knobs:
- CPU attention backend: `vllm/v1/attention/backends/cpu_attn.py`
- C++ dispatch + layout: `csrc/cpu/cpu_attn.cpp`

---

## Appendix: quick sanity checks (ops loaded?)

These are useful when you’re unsure whether a wheel/build actually included the extension.

Check the main extension loads and a few key ops exist:

```bash
python -c "import vllm, torch; import vllm._custom_ops as ops; \
print('reshape_and_cache_flash:', torch.ops._C_cache_ops.reshape_and_cache_flash); \
print('rms_norm:', torch.ops._C.rms_norm); \
print('cpu_attention_with_kv_cache:', torch.ops._C.cpu_attention_with_kv_cache)"
```

If an op is missing, it usually means:
- the extension module failed to import (see the warning emitted by `current_platform.import_kernels()`), or
- it was conditionally compiled out for your platform/arch.

