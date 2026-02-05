# Reusable analysis scripts

These scripts are meant to be re-run whenever you want a quick “what changed recently?” view.

## `git_activity_report.py`

Generates a Markdown report from `git log`:

- commit subject tag/prefix counts (e.g. `[Bugfix]`, `[CI/Build]`)
- top subject keywords
- top active areas (by commits touching the area)
- hottest areas by touched-file count
- hottest areas by churn (added+deleted)
- hottest files by touch count / churn

### Usage

From this notes repo:

```bash
python3 scripts/git_activity_report.py --repo ~/vllm -n 50 --out /tmp/vllm-last-50.md
```

### Monitoring a specific area (path-filtered)

If you only care about “custom ops + native kernels” changes, filter by paths (same semantics as `git log -- <paths...>`):

```bash
python3 scripts/git_activity_report.py \
  --repo ~/vllm -n 10 --show-commits 10 \
  --path vllm/_custom_ops.py \
  --path csrc \
  --path vllm/v1/attention \
  --out /tmp/vllm-kernels-last-10.md
```

You can also keep a list in a file:

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

Convenience wrapper to generate a **timestamped** report under `./reports/` (relative to your current working directory), with an optional `latest-<repo>.md` copy.

### Usage

From this notes repo:

```bash
python3 scripts/snapshot_git_activity.py --repo ~/vllm -n 50 --latest
```

Path-filtered snapshot:

```bash
python3 scripts/snapshot_git_activity.py --repo ~/vllm -n 10 --latest --show-commits 10 \
  --path vllm/_custom_ops.py --path csrc --path vllm/v1/attention
```

Write into a specific folder:

```bash
python3 scripts/snapshot_git_activity.py --repo ~/vllm -n 50 --out-dir /home/oldzhu/mynotes/vllm/reports --latest
```

## Makefile shortcuts

If you’re in the notes folder (`/home/oldzhu/mynotes/vllm`), you can run:

```bash
make vllm-last50
```

For custom-ops/native-kernels monitoring:

```bash
make vllm-kernels-last10
```

Override the repo path if needed:

```bash
make vllm-last50 VLLM_REPO=/path/to/vllm
```

Different window / revision:

```bash
python3 scripts/git_activity_report.py --repo ~/vllm -n 200 --rev main --out /tmp/vllm-last-200.md
python3 scripts/git_activity_report.py --repo ~/vllm -n 100 --rev v0.6.0..HEAD --out /tmp/vllm-since-v0.6.0.md
```

Include merges (usually you **don’t** want this for churn stats):

```bash
python3 scripts/git_activity_report.py --repo ~/vllm -n 50 --include-merges
```

### Custom “area bucketing” rules

Default rules are embedded in the script. To override:

1) Create a JSON file like:

```json
[
  {"name": "core", "pattern": "^vllm/v1/"},
  {"name": "kernels", "pattern": "^csrc/"},
  {"name": "frontend", "pattern": "^vllm/entrypoints/"}
]
```

2) Run:

```bash
python3 scripts/git_activity_report.py --repo ~/vllm -n 50 --rules rules.json
```
