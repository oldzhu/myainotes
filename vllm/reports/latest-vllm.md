# Git activity report: last 10 commits (main@3920cafdd)

Repo: `/home/oldzhu/vllm`
Generated: `2026-02-07 15:48`
Path filter: `vllm/_custom_ops.py, vllm/_aiter_ops.py, csrc, vllm/v1/attention`

## Recent commits

| Commit | Date (iso) | Subject |
|---|---|---|
| [77c09e113](https://github.com/oldzhu/vllm/commit/77c09e1130661197ccac2d968a28cd4a557922d5) | 2026-02-06T13:57:06-05:00 | [Refactor] Remove align block size logic in `moe_permute` (#33449) |
| [350ca72c0](https://github.com/oldzhu/vllm/commit/350ca72c0423f2b094723dfea479a8f0eb7a46d5) | 2026-02-06T09:08:16-06:00 | [ROCm][AITER] Fix AITER import regression for explicit backend selection (#33749) |
| [1363e3d6d](https://github.com/oldzhu/vllm/commit/1363e3d6d5659b58376fa5284afc2c8be548cc9d) | 2026-02-06T07:01:48+00:00 | [cpu][performance] CPU Paged Attention NEON BFMMLA BF16 Implementation (#32263) |
| [ac04dd374](https://github.com/oldzhu/vllm/commit/ac04dd374f996f8df960933fa076bb5ea53c0c2a) | 2026-02-06T10:27:02+05:30 | [CPU] Add BF16 Kernel type for s390x (#33788) |
| [d5c480011](https://github.com/oldzhu/vllm/commit/d5c4800112c12bbcd4955858ef1b415c16ae16e7) | 2026-02-05T14:16:02-08:00 | Adds padding and perf improvements to wvSplitK_fp8 (#33527) |
| [af3162d3a](https://github.com/oldzhu/vllm/commit/af3162d3aaa559a738396baf5b5134c1ab0742f5) | 2026-02-05T12:37:18-05:00 | [Spec Decode] Unified Parallel Drafting (#32887) |
| [e3bf79ffa](https://github.com/oldzhu/vllm/commit/e3bf79ffa080a5052aa61fce71b70b11fb7f9d1e) | 2026-02-04T22:54:27-05:00 | Revert "[Attention][FA3] Update FA3 to include new swizzle optimization" (#33841) |
| [bbe0574d8](https://github.com/oldzhu/vllm/commit/bbe0574d8e51c1c5935aeff9e92040c61d1d59c5) | 2026-02-04T19:49:18-05:00 | [Bugfix] Disable TRTLLM attention when KV transfer is enabled (#33192) |

## Trends

### Commit subject tags/prefixes

| Prefix | Count |
|---|---:|
| (no tag) | 2 |
| Refactor | 1 |
| ROCm | 1 |
| cpu | 1 |
| CPU | 1 |
| Spec Decode | 1 |
| Bugfix | 1 |

### Top subject keywords

| Keyword | Count |
|---|---:|
| attention | 3 |
| aiter | 2 |
| bf16 | 2 |
| fa3 | 2 |
| align | 1 |
| block | 1 |
| size | 1 |
| logic | 1 |
| moe_permute | 1 |
| import | 1 |
| regression | 1 |
| explicit | 1 |
| backend | 1 |
| selection | 1 |
| performance | 1 |

## Top work items (areas)

Ranked by number of commits touching the area.

| Area | Commits | Sample subjects |
|---|---:|---|
| vllm/v1 | 5 | [ROCm][AITER] Fix AITER import regression for explicit backend selection (#33749)<br>[cpu][performance] CPU Paged Attention NEON BFMMLA BF16 Implementation (#32263)<br>[Spec Decode] Unified Parallel Drafting (#32887) |
| csrc | 4 | [Refactor] Remove align block size logic in `moe_permute` (#33449)<br>[cpu][performance] CPU Paged Attention NEON BFMMLA BF16 Implementation (#32263)<br>[CPU] Add BF16 Kernel type for s390x (#33788) |
| vllm/_aiter_ops.py | 1 | [ROCm][AITER] Fix AITER import regression for explicit backend selection (#33749) |

## Hot areas (by file touches)

| Area | Touched files |
|---|---:|
| csrc | 10 |
| vllm/v1 | 9 |
| vllm/_aiter_ops.py | 1 |

## Hot areas (by churn)

| Area | Added+Deleted |
|---|---:|
| csrc | 1178 |
| vllm/v1 | 173 |
| vllm/_aiter_ops.py | 148 |

## Hot files (by touch frequency)

| File | Touches |
|---|---:|
| vllm/v1/attention/backends/flashinfer.py | 2 |
| csrc/moe/moe_permute_unpermute_op.cu | 1 |
| csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu | 1 |
| csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.h | 1 |
| csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.inl | 1 |
| csrc/moe/torch_bindings.cpp | 1 |
| vllm/_aiter_ops.py | 1 |
| vllm/v1/attention/backends/fa_utils.py | 1 |
| vllm/v1/attention/backends/rocm_aiter_fa.py | 1 |
| csrc/cpu/cpu_attn_impl.hpp | 1 |

## Hot files (by churn)

| File | Added+Deleted |
|---|---:|
| csrc/cpu/cpu_attn_neon_bfmmla.hpp | 682 |
| csrc/rocm/skinny_gemms.cu | 300 |
| vllm/_aiter_ops.py | 148 |
| csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu | 60 |
| vllm/v1/attention/backends/flashinfer.py | 53 |
| csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.inl | 47 |
| vllm/v1/attention/backends/fa_utils.py | 46 |
| csrc/moe/moe_permute_unpermute_op.cu | 44 |
| vllm/v1/attention/backends/utils.py | 32 |
| csrc/cpu/cpu_attn_neon.hpp | 19 |

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
