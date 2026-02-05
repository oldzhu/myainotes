#!/usr/bin/env python3
"""Generate a lightweight git activity report (trends + hot areas/files).

Designed to be:
- dependency-free (stdlib only)
- stable output for note-taking (Markdown)
- reusable across repos and time windows

Example:
  python3 scripts/git_activity_report.py --repo ~/vllm -n 50 --out /tmp/vllm-last-50.md

Notes:
- "Areas" are bucketed by path prefixes (customizable via --rules).
- Churn is computed from `git log --numstat` (added+deleted), skipping binary changes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_RULES = [
    ("csrc", r"^csrc/"),
    ("vllm/v1", r"^vllm/v1/"),
    ("vllm/entrypoints", r"^vllm/entrypoints/"),
    ("vllm/model_executor", r"^vllm/model_executor/"),
    ("vllm/compilation", r"^vllm/compilation/"),
    ("vllm/distributed", r"^vllm/distributed/"),
    ("vllm/config", r"^vllm/config/"),
    ("vllm/platforms", r"^vllm/platforms/"),
    ("vllm/plugins", r"^vllm/plugins/"),
    ("vllm/utils", r"^vllm/utils/"),
    ("tests", r"^tests/"),
    ("docs", r"^docs/"),
    ("benchmarks", r"^benchmarks/"),
    ("examples", r"^examples/"),
    ("tools", r"^tools/"),
    (
        "build/packaging",
        r"^(pyproject\\.toml|setup\\.py|CMakeLists\\.txt|cmake/|MANIFEST\\.in|requirements/)",
    ),
]


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]


def run_git(repo: Path, args: list[str]) -> str:
    cmd = ["git", "-C", str(repo), "--no-pager", *args]
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, errors="replace")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(e.output.strip() or f"git command failed: {' '.join(cmd)}") from e


def _normalize_github_base(remote_url: str) -> str | None:
    s = (remote_url or "").strip()
    if not s:
        return None

    # Common forms:
    # - https://github.com/org/repo
    # - https://github.com/org/repo.git
    # - git@github.com:org/repo.git
    # - ssh://git@github.com/org/repo.git
    if s.startswith("git@github.com:"):
        s = "https://github.com/" + s[len("git@github.com:") :]
    elif s.startswith("ssh://git@github.com/"):
        s = "https://github.com/" + s[len("ssh://git@github.com/") :]

    if s.endswith(".git"):
        s = s[:-4]

    if s.startswith("https://github.com/"):
        return s.rstrip("/")

    return None


def detect_github_base(repo: Path) -> str | None:
    try:
        remote = run_git(repo, ["config", "--get", "remote.origin.url"]).strip()
    except Exception:
        return None
    return _normalize_github_base(remote)


def load_rules(rules_path: Path | None) -> list[Rule]:
    if rules_path is None:
        return [Rule(name, re.compile(rx)) for name, rx in DEFAULT_RULES]

    data = json.loads(rules_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Rules JSON must be a list of objects: [{name, pattern}, ...]")

    rules: list[Rule] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Rules entries must be objects")
        name = item.get("name")
        pattern = item.get("pattern")
        if not isinstance(name, str) or not isinstance(pattern, str):
            raise ValueError("Each rule must have string fields: name, pattern")
        rules.append(Rule(name, re.compile(pattern)))
    return rules


def bucket_path(path: str, rules: list[Rule]) -> str:
    for rule in rules:
        if rule.pattern.search(path):
            return rule.name

    if path.startswith("vllm/"):
        parts = path.split("/")
        return "vllm/" + parts[1] if len(parts) > 1 else "vllm"

    return "other"


def load_pathspec(paths: list[str], paths_file: Path | None) -> list[str]:
    pathspec: list[str] = []
    for p in paths:
        p = (p or "").strip()
        if p:
            pathspec.append(p)

    if paths_file is not None:
        for line in paths_file.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            pathspec.append(s)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for p in pathspec:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _append_pathspec(args: list[str], pathspec: list[str]) -> list[str]:
    if not pathspec:
        return args
    return [*args, "--", *pathspec]


def iter_short_commits(
    repo: Path,
    n: int,
    rev: str,
    include_merges: bool,
    pathspec: list[str],
) -> list[tuple[str, str, str, str]]:
    fmt = "%H%x09%h%x09%ad%x09%s"
    # Use an unambiguous timestamp format for reports.
    args = ["log", f"-{n}", rev, "--date=iso-strict", f"--pretty=format:{fmt}"]
    if not include_merges:
        args.insert(1, "--no-merges")
    out = run_git(repo, _append_pathspec(args, pathspec))

    commits: list[tuple[str, str, str, str]] = []
    for line in out.splitlines():
        if "\t" not in line:
            continue
        parts = line.split("\t", 3)
        if len(parts) != 4:
            continue
        full_sha, short_sha, dt, subj = parts
        commits.append((full_sha.strip(), short_sha.strip(), dt.strip(), subj.strip()))
    return commits


def iter_subjects(
    repo: Path,
    n: int,
    rev: str,
    include_merges: bool,
    pathspec: list[str],
) -> list[str]:
    fmt = "%H%x09%s"
    args = ["log", f"-{n}", rev, f"--pretty=format:{fmt}"]
    if not include_merges:
        args.insert(1, "--no-merges")
    out = run_git(repo, _append_pathspec(args, pathspec))

    subjects: list[str] = []
    for line in out.splitlines():
        if "\t" not in line:
            continue
        _sha, subj = line.split("\t", 1)
        subjects.append(subj.strip())
    return subjects


def parse_numstat(
    repo: Path,
    n: int,
    rev: str,
    include_merges: bool,
    pathspec: list[str],
) -> Iterable[tuple[int | None, int | None, str]]:
    # None churn indicates binary change (added/deleted == '-')
    args = ["log", f"-{n}", rev]
    if not include_merges:
        args.insert(1, "--no-merges")
    args += ["--numstat", "--pretty=format:COMMIT"]
    out = run_git(repo, _append_pathspec(args, pathspec))

    for line in out.splitlines():
        if not line.strip() or line.startswith("COMMIT"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        added_s, deleted_s, path = parts
        path = path.strip()

        # Normalize rename-ish paths: `a/{old => new}/b.py` → `a/new/b.py` (approx)
        if "=>" in path:
            path = path.split("=>")[-1].strip().replace("{", "").replace("}", "")

        if added_s == "-" or deleted_s == "-":
            yield (None, None, path)
            continue
        try:
            yield (int(added_s), int(deleted_s), path)
        except ValueError:
            continue


def parse_name_only(
    repo: Path,
    n: int,
    rev: str,
    include_merges: bool,
    pathspec: list[str],
) -> list[tuple[str, list[str]]]:
    fmt = "COMMIT%x09%H%x09%s"
    args = ["log", f"-{n}", rev]
    if not include_merges:
        args.insert(1, "--no-merges")
    args += ["--name-only", f"--pretty=format:{fmt}"]
    out = run_git(repo, _append_pathspec(args, pathspec))

    commits: list[tuple[str, list[str]]] = []
    cur_subj: str | None = None
    cur_files: list[str] = []

    def flush() -> None:
        nonlocal cur_subj, cur_files
        if cur_subj is None:
            return
        commits.append((cur_subj, cur_files))
        cur_subj = None
        cur_files = []

    for line in out.splitlines():
        if line.startswith("COMMIT\t"):
            flush()
            _c, _sha, subj = line.split("\t", 2)
            cur_subj = subj.strip()
            cur_files = []
            continue
        if line.strip():
            cur_files.append(line.strip())

    flush()
    return commits


def top_keywords(subjects: Iterable[str], k: int) -> list[tuple[str, int]]:
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "without",
        "by",
        "from",
        "into",
        "as",
        "is",
        "are",
        "be",
        "fix",
        "fixes",
        "fixed",
        "bug",
        "bugfix",
        "add",
        "adds",
        "added",
        "remove",
        "removed",
        "update",
        "updates",
        "updated",
        "refactor",
        "refactors",
        "refactored",
        "improve",
        "improves",
        "improved",
        "make",
        "makes",
        "made",
        "support",
        "supports",
        "supporting",
        "enable",
        "enabled",
        "disable",
        "disabled",
    }

    words = Counter()
    for subj in subjects:
        s = re.sub(r"^\[[^\]]+\]\s*", "", subj)
        s = re.sub(r"\(#\d+\)", "", s)
        for w in re.findall(r"[A-Za-z][A-Za-z0-9_\-/]+", s.lower()):
            if len(w) < 3 or w in stop:
                continue
            words[w] += 1
    return words.most_common(k)


def subject_prefix_counts(subjects: Iterable[str], k: int) -> list[tuple[str, int]]:
    prefix = Counter()
    for s in subjects:
        m = re.match(r"^\[([^\]]+)\]", s)
        prefix[m.group(1) if m else "(no tag)"] += 1
    return prefix.most_common(k)


def markdown_table(rows: list[tuple[str, int]], headers: tuple[str, str]) -> str:
    left, right = headers
    out = [f"| {left} | {right} |", "|---|---:|"]
    for key, val in rows:
        out.append(f"| {key} | {val} |")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Generate a git activity report (Markdown).")
    ap.add_argument("--repo", default=".", help="Path to git repo (default: cwd)")
    ap.add_argument("-n", type=int, default=50, help="Number of commits to analyze (default: 50)")
    ap.add_argument(
        "--rev",
        default="HEAD",
        help="Revision range base for log (default: HEAD). Examples: HEAD, main, v0.6.0..HEAD",
    )
    ap.add_argument("--include-merges", action="store_true", help="Include merge commits")
    ap.add_argument(
        "--path",
        action="append",
        default=[],
        help=(
            "Limit analysis to commits that touch this path (file or dir). "
            "May be repeated. Equivalent to appending `-- <path>` to `git log`."
        ),
    )
    ap.add_argument(
        "--paths-file",
        type=str,
        default=None,
        help="Optional newline-delimited list of pathspecs (comments with #).",
    )
    ap.add_argument("--rules", type=str, default=None, help="Path to rules JSON file")
    ap.add_argument("--out", type=str, default="-", help="Output path (default: '-', stdout)")
    ap.add_argument("--top", type=int, default=10, help="Top-K for tables (default: 10)")
    ap.add_argument(
        "--top-areas",
        type=int,
        default=5,
        help="Top-K areas to show in the 'work items' list (default: 5)",
    )
    ap.add_argument(
        "--show-commits",
        type=int,
        default=0,
        help="If >0, include a table listing the most recent matching commits (default: 0).",
    )
    ap.add_argument(
        "--github-base",
        type=str,
        default=None,
        help=(
            "Optional GitHub repo base URL like https://github.com/org/repo. "
            "If omitted, tries to detect from remote.origin.url."
        ),
    )

    args = ap.parse_args(argv)

    repo = Path(args.repo).expanduser().resolve()
    if not (repo / ".git").exists():
        # Support being invoked from a subdir inside the repo
        try:
            top = run_git(repo, ["rev-parse", "--show-toplevel"]).strip()
            repo = Path(top)
        except Exception as e:
            raise SystemExit(f"Not a git repo: {repo} ({e})")

    rules = load_rules(Path(args.rules).expanduser().resolve() if args.rules else None)

    pathspec = load_pathspec(
        args.path,
        Path(args.paths_file).expanduser().resolve() if args.paths_file else None,
    )

    head = run_git(repo, ["rev-parse", "--short", "HEAD"]).strip()
    branch = run_git(repo, ["rev-parse", "--abbrev-ref", "HEAD"]).strip()

    github_base = None
    if args.github_base:
        github_base = _normalize_github_base(args.github_base)
        if github_base is None:
            raise SystemExit(
                "--github-base must be a GitHub repo URL like https://github.com/org/repo"
            )
    else:
        github_base = detect_github_base(repo)

    subjects_all = iter_subjects(
        repo,
        args.n,
        args.rev,
        include_merges=args.include_merges,
        pathspec=pathspec,
    )
    prefixes = subject_prefix_counts(subjects_all, args.top)
    keywords = top_keywords(subjects_all, 15)

    # file touch + churn stats (prefer excluding merges for file metrics)
    numstat = list(
        parse_numstat(
            repo,
            args.n,
            args.rev,
            include_merges=args.include_merges,
            pathspec=pathspec,
        )
    )

    area_freq = Counter()
    area_churn = Counter()
    file_freq = Counter()
    file_churn = Counter()

    for added, deleted, path in numstat:
        area = bucket_path(path, rules)
        area_freq[area] += 1
        file_freq[path] += 1
        if added is not None and deleted is not None:
            churn = added + deleted
            area_churn[area] += churn
            file_churn[path] += churn

    # commits-per-area + sample subjects
    commits = parse_name_only(
        repo,
        args.n,
        args.rev,
        include_merges=args.include_merges,
        pathspec=pathspec,
    )
    area_commits = Counter()
    area_samples: dict[str, list[str]] = defaultdict(list)
    for subj, files in commits:
        areas = {bucket_path(f, rules) for f in files}
        for a in areas:
            area_commits[a] += 1
            if len(area_samples[a]) < 3:
                area_samples[a].append(subj)

    top_work_items = area_commits.most_common(args.top_areas)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = f"Git activity report: last {args.n} commits ({branch}@{head})"

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Repo: `{repo}`")
    lines.append(f"Generated: `{now}`")
    if pathspec:
        lines.append(f"Path filter: `{', '.join(pathspec)}`")
    lines.append("")

    if args.show_commits > 0:
        recent = iter_short_commits(
            repo,
            args.show_commits,
            args.rev,
            include_merges=args.include_merges,
            pathspec=pathspec,
        )
        lines.append("## Recent commits")
        lines.append("")
        lines.append("| Commit | Date (iso) | Subject |")
        lines.append("|---|---|---|")
        for full_sha, short_sha, dt, subj in recent:
            if github_base:
                sha_cell = f"[{short_sha}]({github_base}/commit/{full_sha})"
            else:
                sha_cell = short_sha
            lines.append(f"| {sha_cell} | {dt} | {subj} |")
        lines.append("")

    lines.append("## Trends")
    lines.append("")
    lines.append("### Commit subject tags/prefixes")
    lines.append("")
    lines.append(markdown_table([(k, v) for k, v in prefixes], ("Prefix", "Count")))
    lines.append("")

    lines.append("### Top subject keywords")
    lines.append("")
    lines.append(markdown_table([(k, v) for k, v in keywords], ("Keyword", "Count")))
    lines.append("")

    lines.append("## Top work items (areas)")
    lines.append("")
    lines.append("Ranked by number of commits touching the area.")
    lines.append("")
    lines.append("| Area | Commits | Sample subjects |")
    lines.append("|---|---:|---|")
    for area, count in top_work_items:
        samples = area_samples.get(area, [])
        sample_text = "<br>".join(samples)
        lines.append(f"| {area} | {count} | {sample_text} |")
    lines.append("")

    lines.append("## Hot areas (by file touches)")
    lines.append("")
    lines.append(markdown_table(area_freq.most_common(args.top), ("Area", "Touched files")))
    lines.append("")

    lines.append("## Hot areas (by churn)")
    lines.append("")
    lines.append(markdown_table(area_churn.most_common(args.top), ("Area", "Added+Deleted")))
    lines.append("")

    lines.append("## Hot files (by touch frequency)")
    lines.append("")
    lines.append(markdown_table(file_freq.most_common(args.top), ("File", "Touches")))
    lines.append("")

    lines.append("## Hot files (by churn)")
    lines.append("")
    lines.append(markdown_table(file_churn.most_common(args.top), ("File", "Added+Deleted")))
    lines.append("")

    rules_json = json.dumps([{"name": r.name, "pattern": r.pattern.pattern} for r in rules], indent=2)
    lines.append("## Bucketing rules")
    lines.append("")
    lines.append("You can override these with `--rules rules.json`. Format is JSON list: `[{name, pattern}, ...]`.")
    lines.append("")
    lines.append("```json")
    lines.append(rules_json)
    lines.append("```")
    lines.append("")

    text = "\n".join(lines)

    if args.out == "-":
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
    else:
        out_path = Path(args.out).expanduser()
        out_path.write_text(text, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
