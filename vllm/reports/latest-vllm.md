# Git activity report: last 10 commits (main@7fcb705b8)

[English](latest-vllm.md) | [Chinese (ZH-CN)](latest-vllm.zh-CN.md)

Repo: `/home/oldzhu/vllm`
Generated: `2026-02-08 12:04`
Path filter: `vllm/_custom_ops.py, vllm/_aiter_ops.py, csrc, vllm/v1/attention`

## Recent commits

| Commit | Date (iso) | Subject |
|---|---|---|
| [ed17f54c8](https://github.com/oldzhu/vllm/commit/ed17f54c8bccc2f1a19f83f65b36d837698b8792) | 2026-02-07T05:33:11-08:00 | Perf tuning and expansion of cases covered for wvSplitKrc (#33493) |
| [de3869bb4](https://github.com/oldzhu/vllm/commit/de3869bb4db76658bc4775ba4ba5fa6ef546237a) | 2026-02-07T07:30:09-06:00 | move checks out of `unified_kv_cache_update` custom op (#33943) |
| [15a0b9e57](https://github.com/oldzhu/vllm/commit/15a0b9e570dc8bfe716a7d76d50716898123dbae) | 2026-02-07T15:58:50+08:00 | Fix spelling errors (#33978) |
| [bc32444b2](https://github.com/oldzhu/vllm/commit/bc32444b238d2ec3726f599cf3fc67dbaf51a6c6) | 2026-02-06T20:28:01-08:00 | [Kernel] Add enable_sm120_or_later for SM121 (DGX Spark) CUTLASS support (#33517) |

## Trends

### Commit subject tags/prefixes

| Prefix | Count |
|---|---:|
| (no tag) | 3 |
| Kernel | 1 |

### Top subject keywords

| Keyword | Count |
|---|---:|
| perf | 1 |
| tuning | 1 |
| expansion | 1 |
| cases | 1 |
| covered | 1 |
| wvsplitkrc | 1 |
| move | 1 |
| checks | 1 |
| out | 1 |
| unified_kv_cache_update | 1 |
| custom | 1 |
| spelling | 1 |
| errors | 1 |
| enable_sm120_or_later | 1 |
| sm121 | 1 |

## Top work items (areas)

Ranked by number of commits touching the area.

| Area | Commits | Sample subjects |
|---|---:|---|
| csrc | 2 | Perf tuning and expansion of cases covered for wvSplitKrc (#33493)<br>[Kernel] Add enable_sm120_or_later for SM121 (DGX Spark) CUTLASS support (#33517) |
| vllm/v1 | 2 | move checks out of `unified_kv_cache_update` custom op (#33943)<br>Fix spelling errors (#33978) |

## Hot areas (by file touches)

| Area | Touched files |
|---|---:|
| vllm/v1 | 5 |
| csrc | 3 |

## Hot areas (by churn)

| Area | Added+Deleted |
|---|---:|
| csrc | 341 |
| vllm/v1 | 163 |

## Hot files (by touch frequency)

| File | Touches |
|---|---:|
| csrc/rocm/skinny_gemms.cu | 1 |
| vllm/v1/attention/backends/flash_attn.py | 1 |
| vllm/v1/attention/backends/rocm_aiter_unified_attn.py | 1 |
| vllm/v1/attention/backends/rocm_attn.py | 1 |
| vllm/v1/attention/backends/triton_attn.py | 1 |
| vllm/v1/attention/ops/flashmla.py | 1 |
| csrc/cutlass_extensions/common.hpp | 1 |
| csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh | 1 |

## Hot files (by churn)

| File | Added+Deleted |
|---|---:|
| csrc/rocm/skinny_gemms.cu | 327 |
| vllm/v1/attention/backends/rocm_attn.py | 74 |
| vllm/v1/attention/backends/triton_attn.py | 40 |
| vllm/v1/attention/backends/rocm_aiter_unified_attn.py | 31 |
| csrc/cutlass_extensions/common.hpp | 11 |
| vllm/v1/attention/backends/flash_attn.py | 10 |
| vllm/v1/attention/ops/flashmla.py | 8 |
| csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh | 3 |

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
