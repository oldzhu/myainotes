# 可复用分析脚本

[English](README.md) | [简体中文](README.zh-CN.md)

这些脚本用于反复生成“最近发生了什么变化？”的快速视图。

## `git_activity_report.py`

从 `git log` 生成 Markdown 报告：

- 提交标题标签/前缀统计（如 `[Bugfix]`, `[CI/Build]`）
- 标题关键词统计
- 活跃区域（按触及该区域的提交数）
- 热门区域（按触及文件数）
- 高 churn 区域（新增+删除）
- 热门文件（触及次数/新增删除）

### 用法

在此 notes 仓库中：

```bash
python3 scripts/git_activity_report.py --repo ~/vllm -n 50 --out /tmp/vllm-last-50.md
```

### 监控特定区域（按路径过滤）

如果你只关注“自定义算子 + 原生内核”的改动，可按路径过滤（语义等同于 `git log -- <paths...>`）：

```bash
python3 scripts/git_activity_report.py \
  --repo ~/vllm -n 10 --show-commits 10 \
  --path vllm/_custom_ops.py \
  --path csrc \
  --path vllm/v1/attention \
  --out /tmp/vllm-kernels-last-10.md
```

也可以把路径写到文件里：

```bash
cat > /tmp/vllm-kernel-paths.txt <<'EOF'
# Custom ops glue + native kernels
vllm/_custom_ops.py
vllm/_aiter_ops.py
csrc
vllm/v1/attention
EOF

python3 scripts/git_activity_report.py --repo ~/vllm -n 10 --show-commits 10 \
  --paths-file /tmp/vllm-kernel-paths.txt --out /tmp/vllm-kernels-last-10.md
```

## `snapshot_git_activity.py`

便捷包装：生成带**时间戳**的报告到 `./reports/`（相对当前工作目录），可选生成 `latest-<repo>.md`。

### 用法

在此 notes 仓库中：

```bash
python3 scripts/snapshot_git_activity.py --repo ~/vllm -n 50 --latest
```

路径过滤快照：

```bash
python3 scripts/snapshot_git_activity.py --repo ~/vllm -n 10 --latest --show-commits 10 \
  --path vllm/_custom_ops.py --path csrc --path vllm/v1/attention
```

写入指定目录：

```bash
python3 scripts/snapshot_git_activity.py --repo ~/vllm -n 50 --out-dir /home/oldzhu/mynotes/vllm/reports --latest
```

## Makefile 快捷命令

如果你在 notes 目录（`/home/oldzhu/mynotes/vllm`），可以运行：

```bash
make vllm-last50
```

监控自定义算子/原生内核：

```bash
make vllm-kernels-last10
```

## `check_bilingual_docs.py`

校验此 notes 仓库的双语规范：

- 每个英文 `.md` 都有对应的 `.zh-CN.md`
- 每个 `.zh-CN.md` 都有对应的英文 `.md`
- 两者在顶部都包含语言切换链接

### 用法

在 notes 仓库内：

```bash
python3 scripts/check_bilingual_docs.py --root /home/oldzhu/mynotes/vllm
```

或通过 Makefile：

```bash
make bilingual-check
```

需要的话可覆盖仓库路径：

```bash
make vllm-last50 VLLM_REPO=/path/to/vllm
```

不同窗口 / 版本区间：

```bash
python3 scripts/git_activity_report.py --repo ~/vllm -n 200 --rev main --out /tmp/vllm-last-200.md
python3 scripts/git_activity_report.py --repo ~/vllm -n 100 --rev v0.6.0..HEAD --out /tmp/vllm-since-v0.6.0.md
```

包含 merge（通常不建议用于 churn 统计）：

```bash
python3 scripts/git_activity_report.py --repo ~/vllm -n 50 --include-merges
```

### 自定义“区域归类”规则

默认规则内置在脚本里，若要覆盖：

1) 创建 JSON：

```json
[
  {"name": "core", "pattern": "^vllm/v1/"},
  {"name": "kernels", "pattern": "^csrc/"},
  {"name": "frontend", "pattern": "^vllm/entrypoints/"}
]
```

2) 运行：

```bash
python3 scripts/git_activity_report.py --repo ~/vllm -n 50 --rules rules.json
```
