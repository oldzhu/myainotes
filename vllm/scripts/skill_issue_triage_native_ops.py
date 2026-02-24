#!/usr/bin/env python3
"""Select candidate native-ops issues and generate a triage report.

Uses the GitHub Search API to find open issues labeled for native ops and emits
both English and Simplified Chinese summaries for quick selection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable


def _http_json(url: str, token: str | None) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "vllm-issue-triage-script",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc


def _iso_to_ymd(iso_s: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_s.replace("Z", "+00:00"))
        return dt.date().isoformat()
    except ValueError:
        return iso_s


def _status_summary(labels: list[str]) -> str:
    lowered = {l.lower() for l in labels}
    if any("in progress" in l or "in-progress" in l or "wip" in l for l in lowered):
        return "In progress"
    if any("blocked" in l for l in lowered):
        return "Blocked"
    if any("needs" in l for l in lowered) and any("triage" in l for l in lowered):
        return "Needs triage"
    if any("help wanted" in l for l in lowered):
        return "Help wanted"
    return "Open"


@dataclass(frozen=True)
class Issue:
    number: int
    title: str
    url: str
    labels: list[str]
    created_at: str
    updated_at: str
    comments: int
    assignee: str | None


def _iter_issues(
    repo: str,
    labels: list[str],
    query_text: str | None,
    token: str | None,
    per_page: int,
    max_pages: int,
) -> Iterable[Issue]:
    label_query = " OR ".join([f"label:{urllib.parse.quote(l)}" for l in labels])
    base = f"repo:{repo} is:issue is:open"
    query = base
    if labels:
        query += f" ({label_query})"
    if query_text:
        query += f" {query_text}"
    encoded = urllib.parse.quote(query, safe=":()+")

    for page in range(1, max_pages + 1):
        url = (
            "https://api.github.com/search/issues"
            f"?q={encoded}&sort=updated&order=desc&per_page={per_page}&page={page}"
        )
        payload = _http_json(url, token)
        items = payload.get("items", [])
        if not items:
            break
        for item in items:
            if "pull_request" in item:
                continue
            labels_list = [l.get("name", "") for l in item.get("labels", [])]
            assignee = None
            if item.get("assignee"):
                assignee = item["assignee"].get("login")
            yield Issue(
                number=item.get("number"),
                title=item.get("title", ""),
                url=item.get("html_url", ""),
                labels=labels_list,
                created_at=item.get("created_at", ""),
                updated_at=item.get("updated_at", ""),
                comments=int(item.get("comments", 0)),
                assignee=assignee,
            )


def _render_report(
    issues: list[Issue],
    repo: str,
    labels: list[str],
    query_text: str | None,
    date_str: str,
    link_en: str,
    link_zh: str,
) -> str:
    label_text = ", ".join(labels) if labels else "(none)"
    query_text = query_text or "(none)"
    header = [
        "# Native ops issue triage (top 10)",
        "",
        f"[English]({link_en}) | [Chinese (ZH-CN)]({link_zh})",
        "",
        f"Repo: `{repo}`",
        f"Generated: `{date_str}`",
        f"Label filter: `{label_text}`",
        f"Query: `{query_text}`",
        "",
        "## Candidates",
        "",
        "| # | Issue | Labels | Status | Assignee | Updated | Comments |",
        "|---:|---|---|---|---|---|---:|",
    ]
    rows = []
    if not issues:
        rows.append("| - | No matching open, unassigned issues found. | - | - | - | - | - |")
    for issue in issues:
        labels_s = ", ".join(issue.labels)
        status = _status_summary(issue.labels)
        assignee = issue.assignee or "-"
        updated = _iso_to_ymd(issue.updated_at)
        rows.append(
            f"| {issue.number} | [{issue.title}]({issue.url}) | {labels_s} | {status} | {assignee} | {updated} | {issue.comments} |"
        )

    return "\n".join(header + rows + [""])


def _render_report_zh(
    issues: list[Issue],
    repo: str,
    labels: list[str],
    query_text: str | None,
    date_str: str,
    link_en: str,
    link_zh: str,
) -> str:
    label_text = ", ".join(labels) if labels else "(none)"
    query_text = query_text or "(none)"
    header = [
        "# 原生算子问题筛选（前 10 条）",
        "",
        f"[English]({link_en}) | [简体中文]({link_zh})",
        "",
        f"Repo: `{repo}`",
        f"生成时间：`{date_str}`",
        f"标签过滤：`{label_text}`",
        f"搜索条件：`{query_text}`",
        "",
        "## 候选问题",
        "",
        "| # | Issue | 标签 | 状态 | 负责人 | 更新日期 | 评论数 |",
        "|---:|---|---|---|---|---|---:|",
    ]
    rows = []
    if not issues:
        rows.append("| - | 未找到符合条件且未分配的问题。 | - | - | - | - | - |")
    for issue in issues:
        labels_s = ", ".join(issue.labels)
        status = _status_summary(issue.labels)
        status_map = {
            "In progress": "进行中",
            "Blocked": "受阻",
            "Needs triage": "待分流",
            "Help wanted": "需要帮助",
            "Open": "开放",
        }
        status_zh = status_map.get(status, status)
        assignee = issue.assignee or "-"
        updated = _iso_to_ymd(issue.updated_at)
        rows.append(
            f"| {issue.number} | [{issue.title}]({issue.url}) | {labels_s} | {status_zh} | {assignee} | {updated} | {issue.comments} |"
        )

    return "\n".join(header + rows + [""])


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Select native-ops issues and generate a triage report.")
    ap.add_argument("--repo", default="vllm-project/vllm", help="GitHub repo in org/name form")
    ap.add_argument(
        "--labels",
        default="native-kernels,custom-ops",
        help="Comma-separated label names (default: native-kernels,custom-ops)",
    )
    ap.add_argument(
        "--query",
        default=None,
        help="Optional search keywords (GitHub search syntax) to narrow issues.",
    )
    ap.add_argument("--limit", type=int, default=10, help="Number of issues to include (default: 10)")
    ap.add_argument("--out", required=True, help="Output report path (English)")
    ap.add_argument("--out-zh", required=True, help="Output report path (Simplified Chinese)")
    ap.add_argument(
        "--include-assigned",
        action="store_true",
        help="Include assigned issues (default: only unassigned)",
    )
    ap.add_argument("--token", default=os.getenv("GITHUB_TOKEN"), help="GitHub token (or env GITHUB_TOKEN)")
    args = ap.parse_args(argv)

    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    issues: list[Issue] = []
    for issue in _iter_issues(
        args.repo,
        labels,
        args.query,
        args.token,
        per_page=50,
        max_pages=10,
    ):
        if not args.include_assigned and issue.assignee:
            continue
        issues.append(issue)
        if len(issues) >= args.limit:
            break

    link_en = os.path.basename(args.out)
    link_zh = os.path.basename(args.out_zh)
    report = _render_report(issues, args.repo, labels, args.query, date_str, link_en, link_zh)
    report_zh = _render_report_zh(
        issues, args.repo, labels, args.query, date_str, link_en, link_zh
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    with open(args.out_zh, "w", encoding="utf-8") as f:
        f.write(report_zh)

    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
