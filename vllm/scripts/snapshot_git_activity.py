#!/usr/bin/env python3
"""Snapshot a git activity report into a timestamped Markdown file.

This is a thin wrapper around `scripts/git_activity_report.py` that:
- picks a default output directory (`./reports`)
- generates a stable timestamped filename
- optionally also writes a `latest-<repo>.md` copy for quick access

Example:
  python3 scripts/snapshot_git_activity.py --repo ~/vllm -n 50
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _sanitize_filename(s: str) -> str:
    # Keep it simple and filesystem-friendly.
    s = s.strip()
    s = s.replace("..", "__")
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-") or "HEAD"


def _detect_repo_name(repo: Path) -> str:
    # Use the top-level folder name (after resolving to repo root).
    return repo.name


def _git_toplevel(repo: Path) -> Path:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.STDOUT,
            errors="replace",
        ).strip()
        return Path(out)
    except subprocess.CalledProcessError as e:
        raise SystemExit((e.output or "").strip() or f"Not a git repo: {repo}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Snapshot a timestamped git activity report (Markdown).")
    ap.add_argument("--repo", default=".", help="Path to git repo (default: cwd)")
    ap.add_argument("-n", type=int, default=50, help="Number of commits to analyze (default: 50)")
    ap.add_argument("--rev", default="HEAD", help="Revision/range for git log (default: HEAD)")
    ap.add_argument("--rules", default=None, help="Optional rules JSON for area bucketing")
    ap.add_argument(
        "--out-dir",
        default="reports",
        help="Directory to write reports into (default: ./reports)",
    )
    ap.add_argument(
        "--latest",
        action="store_true",
        help="Also write/update latest-<repo>.md in the output directory",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=10,
        help="Top-K rows for most tables (default: 10)",
    )
    ap.add_argument(
        "--top-areas",
        type=int,
        default=5,
        help="Top-K areas to show in work-items list (default: 5)",
    )

    args = ap.parse_args(argv)

    repo = Path(args.repo).expanduser().resolve()
    repo_root = _git_toplevel(repo)
    repo_name = _detect_repo_name(repo_root)

    out_dir = Path(args.out_dir).expanduser()
    # If user gave a relative path, treat it as relative to current working directory.
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    rev_slug = _sanitize_filename(args.rev)
    repo_slug = _sanitize_filename(repo_name)

    out_file = out_dir / f"{repo_slug}-activity-n{args.n}-{rev_slug}-{ts}.md"

    script_dir = Path(__file__).resolve().parent
    report_script = script_dir / "git_activity_report.py"

    cmd = [
        sys.executable,
        str(report_script),
        "--repo",
        str(repo_root),
        "-n",
        str(args.n),
        "--rev",
        str(args.rev),
        "--out",
        str(out_file),
        "--top",
        str(args.top),
        "--top-areas",
        str(args.top_areas),
    ]
    if args.rules:
        cmd += ["--rules", str(Path(args.rules).expanduser())]

    subprocess.check_call(cmd)

    if args.latest:
        latest = out_dir / f"latest-{repo_slug}.md"
        shutil.copyfile(out_file, latest)

    print(str(out_file))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
