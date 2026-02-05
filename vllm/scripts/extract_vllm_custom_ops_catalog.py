#!/usr/bin/env python3
"""Extract a catalog of vLLM custom ops exposed to Python.

This script is intentionally dependency-free and aims to answer:
- What operators are registered in the native extensions (torch.ops.*)?
- Which of those are wrapped/used by Python glue (vllm/_custom_ops.py, etc.)?

It parses authoritative sources by scanning vLLM native extension registration sites:
- csrc/** for TORCH_LIBRARY* blocks and their .def("...") registrations

It also scans Python for torch.ops usage to help reconcile what is *used* from Python
versus what is *registered* by vLLM itself.

Output is Markdown.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


_PY_OP_RE = re.compile(r"\btorch\.ops\.(?P<ns>[A-Za-z0-9_]+)\.(?P<op>[A-Za-z0-9_]+)\b")

_TORCH_LIBRARY_RE = re.compile(
    r"\bTORCH_LIBRARY\s*\(\s*(?P<ns>[A-Za-z0-9_]+)\s*,\s*(?P<var>[A-Za-z0-9_]+)\s*\)"
)
_TORCH_LIBRARY_FRAGMENT_RE = re.compile(
    r"\bTORCH_LIBRARY_FRAGMENT\s*\(\s*(?P<ns>[A-Za-z0-9_]+)\s*,\s*(?P<var>[A-Za-z0-9_]+)\s*\)"
)
_TORCH_LIBRARY_IMPL_RE = re.compile(
    r"\bTORCH_LIBRARY_IMPL\s*\(\s*(?P<ns>[A-Za-z0-9_]+)\s*,\s*(?P<key>[^,\)]+)\s*,\s*(?P<var>[A-Za-z0-9_]+)\s*\)"
)

_TORCH_LIBRARY_EXPAND_TOKEN = "TORCH_LIBRARY_EXPAND"


@dataclass(frozen=True)
class CppOp:
    ns: str
    name: str
    file: str


@dataclass(frozen=True)
class CppBindingBlock:
    ns: str
    var: str
    file: str
    start: int
    end: int


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _strip_cpp_comments_and_track_strings(text: str, i: int) -> int:
    """Advance index i over comments/strings; return new i.

    This is used by brace matching to reduce false matches.
    """

    n = len(text)
    if i >= n:
        return i

    ch = text[i]

    if ch == '"':
        i += 1
        while i < n:
            if text[i] == "\\":
                i += 2
                continue
            if text[i] == '"':
                return i + 1
            i += 1
        return i

    if ch == "/" and i + 1 < n:
        nxt = text[i + 1]
        if nxt == "/":
            i += 2
            while i < n and text[i] != "\n":
                i += 1
            return i
        if nxt == "*":
            i += 2
            while i + 1 < n:
                if text[i] == "*" and text[i + 1] == "/":
                    return i + 2
                i += 1
            return i

    return i


def _find_matching_brace(text: str, open_brace_index: int) -> int | None:
    if open_brace_index < 0 or open_brace_index >= len(text) or text[open_brace_index] != "{":
        return None

    depth = 0
    i = open_brace_index
    n = len(text)

    while i < n:
        new_i = _strip_cpp_comments_and_track_strings(text, i)
        if new_i != i:
            i = new_i
            continue

        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _extract_op_name_from_schema(schema: str) -> str | None:
    head = schema.split("(", 1)[0].strip()
    if not head:
        return None
    # Drop any qualifiers like "vllm::op".
    if "::" in head:
        head = head.rsplit("::", 1)[-1]
    m = re.match(r"[A-Za-z0-9_]+", head)
    return m.group(0) if m else None


def _find_torch_library_blocks(text: str, file: Path) -> list[CppBindingBlock]:
    blocks: list[CppBindingBlock] = []
    for lib_re in (_TORCH_LIBRARY_RE, _TORCH_LIBRARY_FRAGMENT_RE, _TORCH_LIBRARY_IMPL_RE):
        for m in lib_re.finditer(text):
            ns = m.group("ns")
            var = m.group("var")
            brace_open = text.find("{", m.end())
            if brace_open < 0:
                continue
            brace_close = _find_matching_brace(text, brace_open)
            if brace_close is None:
                continue
            blocks.append(
                CppBindingBlock(
                    ns=ns,
                    var=var,
                    file=str(file),
                    start=brace_open,
                    end=brace_close,
                ))

    # TORCH_LIBRARY_EXPAND is frequently used in vLLM, and its first argument can be a macro
    # expression with nested parentheses/commas (e.g., CONCAT(TORCH_EXTENSION_NAME, _cache_ops)).
    # Regex is error-prone here; parse the macro call with a tiny argument splitter.
    idx = 0
    n = len(text)
    while True:
        hit = text.find(_TORCH_LIBRARY_EXPAND_TOKEN, idx)
        if hit < 0:
            break

        # Find the opening paren for the macro invocation.
        paren_open = text.find("(", hit + len(_TORCH_LIBRARY_EXPAND_TOKEN))
        if paren_open < 0:
            idx = hit + len(_TORCH_LIBRARY_EXPAND_TOKEN)
            continue

        # Find the matching close paren.
        depth = 0
        i = paren_open
        while i < n:
            new_i = _strip_cpp_comments_and_track_strings(text, i)
            if new_i != i:
                i = new_i
                continue
            ch = text[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    paren_close = i
                    break
            i += 1
        else:
            idx = paren_open + 1
            continue

        inside = text[paren_open + 1 : paren_close]
        args: list[str] = []
        buf: list[str] = []
        depth2 = 0
        j = 0
        while j < len(inside):
            ch = inside[j]
            if ch == "(":
                depth2 += 1
            elif ch == ")":
                depth2 -= 1
            elif ch == "," and depth2 == 0:
                arg = "".join(buf).strip()
                if arg:
                    args.append(arg)
                buf = []
                j += 1
                continue
            buf.append(ch)
            j += 1
        tail = "".join(buf).strip()
        if tail:
            args.append(tail)

        if len(args) >= 2:
            ns_expr = args[0]
            var = args[1]
            # Variable names should be simple identifiers; skip weird parses.
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", var):
                ns = _resolve_torch_library_expand_namespace(file, ns_expr)
                brace_open = text.find("{", paren_close)
                if brace_open >= 0:
                    brace_close = _find_matching_brace(text, brace_open)
                    if brace_close is not None:
                        blocks.append(
                            CppBindingBlock(
                                ns=ns,
                                var=var,
                                file=str(file),
                                start=brace_open,
                                end=brace_close,
                            )
                        )

        idx = paren_close + 1
    # Prefer earlier blocks first; useful for deterministic output.
    blocks.sort(key=lambda b: (b.file, b.start, b.ns, b.var))
    return blocks


def _resolve_torch_library_expand_namespace(file: Path, ns_expr: str) -> str:
    """Best-effort mapping from TORCH_LIBRARY_EXPAND's namespace expression to torch.ops namespace.

    vLLM commonly uses TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ...) where TORCH_EXTENSION_NAME
    varies by extension target (CUDA/CPU/ROCm/MoE). It also uses CONCAT(TORCH_EXTENSION_NAME, _suffix)
    to build sibling namespaces.
    """

    p = str(file).replace("\\", "/")
    clean = re.sub(r"\s+", "", ns_expr)

    # Default base depends on which extension is being built.
    if p.endswith("csrc/rocm/torch_bindings.cpp"):
        base = "_rocm_C"
    elif p.endswith("csrc/moe/torch_bindings.cpp"):
        base = "_moe_C"
    else:
        base = "_C"

    if clean == "TORCH_EXTENSION_NAME":
        return base

    # CONCAT(TORCH_EXTENSION_NAME,_cache_ops) => _C_cache_ops (or base + suffix)
    prefix = "CONCAT(TORCH_EXTENSION_NAME,"
    if clean.startswith(prefix) and clean.endswith(")"):
        suffix = clean[len(prefix) : -1]
        return f"{base}{suffix}"

    # If we can't resolve, keep the expression (still useful as a breadcrumb).
    return clean


def extract_cpp_ops_from_repo(repo: Path) -> tuple[list[CppOp], list[str]]:
    """Scan csrc/** for TORCH_LIBRARY* blocks and extract registered op names."""

    csrc = repo / "csrc"
    if not csrc.exists():
        return [], []

    cpp_files: list[Path] = []
    for ext in ("*.cpp", "*.cc", "*.cxx", "*.cu"):
        cpp_files.extend(csrc.rglob(ext))

    ops: list[CppOp] = []
    source_files_used: set[str] = set()

    for f in sorted(set(cpp_files)):
        text = _read_text(f)
        if "TORCH_LIBRARY" not in text:
            continue

        blocks = _find_torch_library_blocks(text, f)
        if not blocks:
            continue

        rel_file = str(f.relative_to(repo))
        source_files_used.add(rel_file)

        for b in blocks:
            block_text = text[b.start : b.end + 1]
            def_re = re.compile(
                rf"\b{re.escape(b.var)}\s*\.def\(\s*(?:TORCH_SELECTIVE_SCHEMA\s*\(\s*)?\"(?P<sig>[^\"]+)\""
            )
            for m in def_re.finditer(block_text):
                sig = m.group("sig")
                name = _extract_op_name_from_schema(sig)
                if not name:
                    continue
                ops.append(CppOp(ns=b.ns, name=name, file=rel_file))

    return ops, sorted(source_files_used)


def extract_python_ops(py_path: Path) -> dict[str, set[str]]:
    text = _read_text(py_path)
    out: dict[str, set[str]] = {}
    for m in _PY_OP_RE.finditer(text):
        ns = m.group("ns")
        op = m.group("op")
        out.setdefault(ns, set()).add(op)
    return out


def extract_python_ops_from_repo(repo: Path) -> tuple[dict[str, set[str]], dict[str, dict[str, set[str]]]]:
    """Scan vllm/**.py for torch.ops usage.

    Returns:
      - aggregated namespace -> ops
      - per-file mapping (repo-relative file -> namespace -> ops)
    """

    pkg = repo / "vllm"
    if not pkg.exists():
        return {}, {}

    aggregated: dict[str, set[str]] = {}
    per_file: dict[str, dict[str, set[str]]] = {}
    for py in sorted(pkg.rglob("*.py")):
        ns_map = extract_python_ops(py)
        if not ns_map:
            continue
        rel = str(py.relative_to(repo))
        per_file[rel] = ns_map
        for ns, ops in ns_map.items():
            aggregated.setdefault(ns, set()).update(ops)

    return aggregated, per_file


def render_markdown(
    repo: Path,
    cpp_ops: list[CppOp],
    native_sources: list[str],
    python_agg: dict[str, set[str]],
    python_per_file: dict[str, dict[str, set[str]]],
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines: list[str] = []
    lines.append("# vLLM custom ops catalog (Python-exposed)")
    lines.append("")
    lines.append(f"Repo: `{repo}`")
    lines.append(f"Generated: `{now}`")
    lines.append("")
    lines.append(
        "This document lists custom operators exposed to Python via `torch.ops.*` "
        "that are registered by vLLM native extensions, plus a few Python glue files "
        "that call into them."
    )
    lines.append("")

    lines.append("## 1) Native-registered operators (authoritative)")
    lines.append("")
    lines.append(
        "Scanned `csrc/**` for `TORCH_LIBRARY*` blocks and extracted their `.def(\"...\")` registrations."
    )
    lines.append("")
    lines.append(f"Native source files with registrations: **{len(native_sources)}**")
    lines.append("")
    for f in native_sources:
        lines.append(f"- `{f}`")
    lines.append("")

    by_ns: dict[str, set[str]] = {}
    by_ns_files: dict[str, set[str]] = {}
    for op in cpp_ops:
        by_ns.setdefault(op.ns, set()).add(op.name)
        by_ns_files.setdefault(op.ns, set()).add(op.file)

    for ns in sorted(by_ns.keys()):
        lines.append(f"### `torch.ops.{ns}`")
        lines.append("")
        files = sorted(by_ns_files.get(ns, set()))
        if files:
            files_rel: list[str] = []
            for f in files:
                fp = Path(f)
                try:
                    files_rel.append(str(fp.relative_to(repo)))
                except Exception:
                    files_rel.append(str(fp))
            lines.append("Sources: " + ", ".join(f"`{f}`" for f in files_rel))
            lines.append("")
        lines.append(f"Count: **{len(by_ns[ns])}**")
        lines.append("")
        lines.append("| Op |")
        lines.append("|---|")
        for name in sorted(by_ns[ns]):
            lines.append(f"| `{name}` |")
        lines.append("")

    lines.append("## 2) Python `torch.ops.*` usage")
    lines.append("")
    lines.append(
        "Scanned `vllm/**/*.py` for `torch.ops.<namespace>.<op>` references (this includes third-party "
        "namespaces too, e.g. `torch.ops.aten`)."
    )
    lines.append("")

    lines.append("### Aggregated by namespace")
    lines.append("")
    for ns in sorted(python_agg.keys()):
        ops = python_agg[ns]
        lines.append(
            f"- `torch.ops.{ns}` ({len(ops)} ops): " + ", ".join(f"`{o}`" for o in sorted(ops))
        )
    lines.append("")

    lines.append("### Selected call sites (top files)")
    lines.append("")
    # Show a concise subset: files with the most torch.ops references.
    file_scores: list[tuple[int, str]] = []
    for rel, ns_map in python_per_file.items():
        score = sum(len(v) for v in ns_map.values())
        file_scores.append((score, rel))
    for score, rel in sorted(file_scores, reverse=True)[:10]:
        ns_map = python_per_file[rel]
        lines.append(f"- `{rel}` ({score} refs)")
        # Keep per-file summary compact.
        for ns in sorted(ns_map.keys()):
            ops = ns_map[ns]
            lines.append(
                f"  - `torch.ops.{ns}` ({len(ops)} ops): " + ", ".join(f"`{o}`" for o in sorted(ops))
            )
    lines.append("")

    lines.append("## 3) Reconciliation")
    lines.append("")
    native_namespaces = set(by_ns.keys())
    python_namespaces = set(python_agg.keys())
    python_minus_native = sorted(python_namespaces - native_namespaces)
    native_minus_python = sorted(native_namespaces - python_namespaces)

    lines.append("### Namespaces used in Python but not found in vLLM native bindings scan")
    lines.append("")
    if python_minus_native:
        for ns in python_minus_native:
            lines.append(f"- `torch.ops.{ns}`")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("### Namespaces registered natively but not referenced by Python scans")
    lines.append("")
    if native_minus_python:
        for ns in native_minus_python:
            lines.append(f"- `torch.ops.{ns}`")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## 4) Notes / caveats")
    lines.append("")
    lines.append(
        "- Some operators are conditionally compiled (CUDA vs ROCm vs CPU ISA). Presence depends on your build/platform."
    )
    lines.append(
        "- This catalog lists operator names/namespaces. Exact schemas live in the `.def(\"...\")` strings in the binding files."
    )
    lines.append(
        "- Third-party namespaces (e.g. `torch.ops.aiter.*` or `torch.ops.vllm.*`) may appear in Python glue but are not necessarily registered by vLLM’s `_C` extensions."
    )

    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Extract a catalog of vLLM custom ops exposed to Python.")
    ap.add_argument("--repo", default=str(Path.home() / "vllm"), help="Path to vLLM git repo")
    ap.add_argument("--out", default="-", help="Output markdown path (default: '-', stdout)")
    args = ap.parse_args(argv)

    repo = Path(args.repo).expanduser().resolve()

    cpp_ops, native_sources = extract_cpp_ops_from_repo(repo)
    python_agg, python_per_file = extract_python_ops_from_repo(repo)

    md = render_markdown(repo, cpp_ops, native_sources, python_agg, python_per_file)

    if args.out == "-":
        print(md, end="")
    else:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")

    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
