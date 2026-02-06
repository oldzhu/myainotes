#!/usr/bin/env python3
"""Validate bilingual markdown policy for this notes repo.

Rules enforced:
- Every English .md must have a matching .zh-CN.md.
- Every .zh-CN.md must have a matching English .md.
- Both files must include a language switch link near the top.

Exit code:
  0 if all checks pass, non-zero otherwise.
"""

from __future__ import annotations

import argparse
from pathlib import Path


EXCLUDE_DIRS = {".git", "reports", "__pycache__"}


def _iter_md_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob("*.md"):
        rel = path.relative_to(root)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        out.append(path)
    return out


def _switch_link_present(text: str, other_name: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    head = "\n".join(lines[:12])
    return other_name in head and ("[English]" in head or "[简体中文]" in head or "[Chinese" in head)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Check bilingual markdown policy.")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="Root of notes repo")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    md_files = _iter_md_files(root)

    issues: list[str] = []

    for path in md_files:
        rel = path.relative_to(root)
        name = rel.name
        if name.endswith(".zh-CN.md"):
            eng_name = name.replace(".zh-CN.md", ".md")
            eng_path = path.with_name(eng_name)
            if not eng_path.exists():
                issues.append(f"Missing English file for {rel}")
                continue
            zh_text = path.read_text(encoding="utf-8", errors="replace")
            if not _switch_link_present(zh_text, eng_name):
                issues.append(f"Missing switch link in {rel}")
        else:
            zh_name = name.replace(".md", ".zh-CN.md")
            zh_path = path.with_name(zh_name)
            if not zh_path.exists():
                issues.append(f"Missing Chinese file for {rel}")
                continue
            en_text = path.read_text(encoding="utf-8", errors="replace")
            if not _switch_link_present(en_text, zh_name):
                issues.append(f"Missing switch link in {rel}")

    if issues:
        print("Bilingual policy check failed:")
        for item in issues:
            print(f"- {item}")
        return 2

    print("Bilingual policy check passed.")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
