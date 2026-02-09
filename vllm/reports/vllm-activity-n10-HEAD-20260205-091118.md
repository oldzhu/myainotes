# Git activity report: last 10 commits (main@4d9513537)

[English](vllm-activity-n10-HEAD-20260205-091118.md) | [Chinese (ZH-CN)](vllm-activity-n10-HEAD-20260205-091118.zh-CN.md)

Repo: `/home/oldzhu/vllm`
Generated: `2026-02-05 09:11`
Path filter: `vllm/_custom_ops.py, vllm/_aiter_ops.py, csrc, vllm/v1/attention`

## Recent commits

| Commit | Date (iso) | Subject |
|---|---|---|
| 6e98f6d8b | 2026-02-05T05:11:39+09:00 | Implement zero-copy GQA for multimodal and CPU (#33732) |
| 0e9229862 | 2026-02-04T09:41:57-07:00 | [Misc] Delay deprecation of CommonAttentionMetadata properties (#33801) |
| 45f8fd6f9 | 2026-02-03T21:27:34-08:00 | [Feature] Enable `TRITON_ATTN` for Batch Invariance (#33688) |
| 4dffc5e04 | 2026-02-04T09:07:15+05:30 | [CPU] Split attention dispatch by head_dim alignment (#32161) |
| 2267cb1cf | 2026-02-03T09:08:47-07:00 | [Attention][FA3] Update FA3 to include new swizzle optimization (#23465) |
| e10604480 | 2026-02-03T14:46:10+08:00 | [XPU][1/N] Deprecate ipex and switch to vllm-xpu-kernels for xpu platform (#33379) |
| e69c990c2 | 2026-02-03T04:17:56+00:00 | [Feature][CPU Backend]: Optimize ARM vectorization backend (#30329) |
| 089cd4f00 | 2026-02-02T11:47:46-08:00 | fix cutlass_3x_gemm_fp8_blockwise on sm103a (#32224) |
| 528e9b149 | 2026-02-02T23:55:46+09:00 | [Feature][Core] Support Fabric detection to adapt the MNNVL protocol for the GB series (#33540) |
| 9eb58f8cf | 2026-02-02T19:40:02+05:30 | fix[ROCm]: Remove unconditional aiter import (#32902) |

## Trends

### Commit subject tags/prefixes

| Prefix | Count |
|---|---:|
| (no tag) | 3 |
| Feature | 3 |
| Misc | 1 |
| CPU | 1 |
| Attention | 1 |
| XPU | 1 |

### Top subject keywords

| Keyword | Count |
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

## Top work items (areas)

Ranked by number of commits touching the area.

| Area | Commits | Sample subjects |
|---|---:|---|
| vllm/v1 | 6 | Implement zero-copy GQA for multimodal and CPU (#33732)<br>[Misc] Delay deprecation of CommonAttentionMetadata properties (#33801)<br>[Feature] Enable `TRITON_ATTN` for Batch Invariance (#33688) |
| csrc | 4 | [CPU] Split attention dispatch by head_dim alignment (#32161)<br>[Feature][CPU Backend]: Optimize ARM vectorization backend (#30329)<br>fix cutlass_3x_gemm_fp8_blockwise on sm103a (#32224) |
| vllm/_aiter_ops.py | 1 | fix[ROCm]: Remove unconditional aiter import (#32902) |

## Hot areas (by file touches)

| Area | Touched files |
|---|---:|
| csrc | 17 |
| vllm/v1 | 9 |
| vllm/_aiter_ops.py | 1 |

## Hot areas (by churn)

| Area | Added+Deleted |
|---|---:|
| csrc | 1711 |
| vllm/v1 | 56 |
| vllm/_aiter_ops.py | 16 |

## Hot files (by touch frequency)

| File | Touches |
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

## Hot files (by churn)

| File | Added+Deleted |
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

## Bucketing rules

You can override these with `--rules rules.json`. Format is JSON list: `[{name, pattern}, ...]`.

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
    "pattern": "^(pyproject\\\\.toml|setup\\\\.py|CMakeLists\\\\.txt|cmake/|MANIFEST\\\\.in|requirements/)"
  }
]
```
