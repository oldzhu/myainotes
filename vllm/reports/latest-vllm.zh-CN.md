# Git 活动报告：最近 10 次提交 (main@f97ca6717)

[English](latest-vllm.md) | [简体中文](latest-vllm.zh-CN.md)

仓库：`/home/oldzhu/vllm`
生成时间：`2026-02-09 10:13`
路径过滤：`vllm/_custom_ops.py, vllm/_aiter_ops.py, csrc, vllm/v1/attention`

## 最近提交

| 提交 | 日期 (iso) | 主题 |
|---|---|---|
| [179ae7da8](https://github.com/oldzhu/vllm/commit/179ae7da8f48f76b0db4dc3e331153b24a36a96a) | 2026-02-09T00:13:24+08:00 | [Revert] Fix performance regression for GLM-4.7-GPTQ decode and MTP acceptance rate (#33771) |

## 趋势

### 提交主题标签/前缀

| 前缀 | 计数 |
|---|---:|
| Revert | 1 |

### 主题关键词 Top

| 关键词 | 计数 |
|---|---:|
| performance | 1 |
| regression | 1 |
| glm-4 | 1 |
| gptq | 1 |
| decode | 1 |
| mtp | 1 |
| acceptance | 1 |
| rate | 1 |

## 主要工作项（区域）

按触及该区域的提交次数排序。

| 区域 | 提交数 | 示例主题 |
|---|---:|---|
| vllm/v1 | 1 | [Revert] Fix performance regression for GLM-4.7-GPTQ decode and MTP acceptance rate (#33771) |

## 热点区域（按触及文件数）

| 区域 | 触及文件数 |
|---|---:|
| vllm/v1 | 1 |

## 热点区域（按变更量）

| 区域 | 新增+删除 |
|---|---:|
| vllm/v1 | 4 |

## 热点文件（按触及次数）

| 文件 | 触及次数 |
|---|---:|
| vllm/v1/attention/backends/flashinfer.py | 1 |

## 热点文件（按变更量）

| 文件 | 新增+删除 |
|---|---:|
| vllm/v1/attention/backends/flashinfer.py | 4 |

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
