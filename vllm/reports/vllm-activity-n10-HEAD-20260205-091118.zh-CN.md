# Git 活动报告：最近 10 次提交 (main@4d9513537)

[English](vllm-activity-n10-HEAD-20260205-091118.md) | [简体中文](vllm-activity-n10-HEAD-20260205-091118.zh-CN.md)

仓库：`/home/oldzhu/vllm`
生成时间：`2026-02-05 09:11`
路径过滤：`vllm/_custom_ops.py, vllm/_aiter_ops.py, csrc, vllm/v1/attention`

## 最近提交

| 提交 | 主题 |
|---|---|
| 6e98f6d8b | Implement zero-copy GQA for multimodal and CPU (#33732) |
| 0e9229862 | [Misc] Delay deprecation of CommonAttentionMetadata properties (#33801) |
| 45f8fd6f9 | [Feature] Enable `TRITON_ATTN` for Batch Invariance (#33688) |
| 4dffc5e04 | [CPU] Split attention dispatch by head_dim alignment (#32161) |
| 2267cb1cf | [Attention][FA3] Update FA3 to include new swizzle optimization (#23465) |
| e10604480 | [XPU][1/N] Deprecate ipex and switch to vllm-xpu-kernels for xpu platform (#33379) |
| e69c990c2 | [Feature][CPU Backend]: Optimize ARM vectorization backend (#30329) |
| 089cd4f00 | fix cutlass_3x_gemm_fp8_blockwise on sm103a (#32224) |
| 528e9b149 | [Feature][Core] Support Fabric detection to adapt the MNNVL protocol for the GB series (#33540) |
| 9eb58f8cf | fix[ROCm]: Remove unconditional aiter import (#32902) |

## 趋势

### 提交主题标签/前缀

| 前缀 | 计数 |
|---|---:|
| (no tag) | 3 |
| Feature | 3 |
| Misc | 1 |
| CPU | 1 |
| Attention | 1 |
| XPU | 1 |

### 主题关键词 Top

| 关键词 | 计数 |
|---|---:|
| cpu | 2 |
| fa3 | 2 |
| backend | 2 |
| implement | 1 |
| zero-copy | 1 |
| gqa | 1 |
| multimodal | 1 |
| delay | 1 |
| deprecation | 1 |
| commonattentionmetadata | 1 |
| properties | 1 |
| triton_attn | 1 |
| batch | 1 |
| invariance | 1 |
| split | 1 |

## 主要工作项（区域）

按触及该区域的提交次数排序。

| 区域 | 提交数 | 示例主题 |
|---|---:|---|
| vllm/v1 | 6 | Implement zero-copy GQA for multimodal and CPU (#33732)<br>[Misc] Delay deprecation of CommonAttentionMetadata properties (#33801)<br>[Feature] Enable `TRITON_ATTN` for Batch Invariance (#33688) |
| csrc | 4 | [CPU] Split attention dispatch by head_dim alignment (#32161)<br>[Feature][CPU Backend]: Optimize ARM vectorization backend (#30329)<br>fix cutlass_3x_gemm_fp8_blockwise on sm103a (#32224) |
| vllm/_aiter_ops.py | 1 | fix[ROCm]: Remove unconditional aiter import (#32902) |

## 热点区域（按触及文件数）

| 区域 | 触及文件数 |
|---|---:|
| csrc | 17 |
| vllm/v1 | 9 |
| vllm/_aiter_ops.py | 1 |

## 热点区域（按变更量）

| 区域 | 新增+删除 |
|---|---:|
| csrc | 1711 |
| vllm/v1 | 56 |
| vllm/_aiter_ops.py | 16 |

## 热点文件（按触及次数）

| 文件 | 触及次数 |
|---|---:|
| vllm/v1/attention/backends/cpu_attn.py | 1 |
| vllm/v1/attention/ops/vit_attn_wrappers.py | 1 |
| vllm/v1/attention/backend.py | 1 |
| vllm/v1/attention/ops/triton_unified_attention.py | 1 |
| csrc/cpu/cpu_attn.cpp | 1 |
| csrc/cpu/cpu_attn_amx.hpp | 1 |
| csrc/cpu/cpu_attn_neon.hpp | 1 |
| csrc/cpu/generate_cpu_attn_dispatch.py | 1 |
| vllm/v1/attention/backends/flash_attn.py | 1 |
| vllm/v1/attention/backends/mla/flashattn_mla.py | 1 |

## 热点文件（按变更量）

| 文件 | 新增+删除 |
|---|---:|
| csrc/cpu/cpu_types_arm.hpp | 1130 |
| csrc/cpu/generate_cpu_attn_dispatch.py | 203 |
| csrc/cpu/cpu_attn.cpp | 125 |
| csrc/cutlass_extensions/common.hpp | 94 |
| csrc/quantization/w8a8/cutlass/scaled_mm_c2x.cuh | 35 |
| csrc/quantization/w8a8/cutlass/scaled_mm_c2x_sm89_fp8_dispatch.cuh | 34 |
| csrc/quantization/w8a8/cutlass/scaled_mm_c2x_sm89_int8_dispatch.cuh | 34 |
| csrc/cumem_allocator.cpp | 19 |
| vllm/v1/attention/ops/vit_attn_wrappers.py | 16 |
| vllm/_aiter_ops.py | 16 |

## 分桶规则

可用 `--rules rules.json` 覆盖。格式为 JSON 列表：`[{name, pattern}, ...]`。

```json
[
  {
    "name": "csrc",
    "pattern": "^csrc/"
  },
  {
    "name": "vllm/v1",
    "pattern": "^vllm/v1/"
  },
  {
    "name": "vllm/entrypoints",
    "pattern": "^vllm/entrypoints/"
  },
  {
    "name": "vllm/model_executor",
    "pattern": "^vllm/model_executor/"
  },
  {
    "name": "vllm/compilation",
    "pattern": "^vllm/compilation/"
  },
  {
    "name": "vllm/distributed",
    "pattern": "^vllm/distributed/"
  },
  {
    "name": "vllm/config",
    "pattern": "^vllm/config/"
  },
  {
    "name": "vllm/platforms",
    "pattern": "^vllm/platforms/"
  },
  {
    "name": "vllm/plugins",
    "pattern": "^vllm/plugins/"
  },
  {
    "name": "vllm/utils",
    "pattern": "^vllm/utils/"
  },
  {
    "name": "tests",
    "pattern": "^tests/"
  },
  {
    "name": "docs",
    "pattern": "^docs/"
  },
  {
    "name": "benchmarks",
    "pattern": "^benchmarks/"
  },
  {
    "name": "examples",
    "pattern": "^examples/"
  },
  {
    "name": "tools",
    "pattern": "^tools/"
  },
  {
    "name": "build/packaging",
    "pattern": "^(pyproject\\.toml|setup\\.py|CMakeLists\\.txt|cmake/|MANIFEST\\.in|requirements/)"
  }
]
```
