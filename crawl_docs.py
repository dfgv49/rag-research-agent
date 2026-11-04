#!/usr/bin/env python
"""Build a 300-500 document corpus from the docs.langchain.com Markdown indexes.

Everything is written under $env:TEMP: the DSH sandbox denies Python writes
inside the repo workspace (PermissionError / WinError 5), so this script never
writes to the repository. `src/sample_docs.json` (the 32-doc baseline) is left
untouched.

    python crawl_docs.py                 # full run  -> %TEMP%\docs_large.json
    python crawl_docs.py --index-only    # index only -> %TEMP%\docs_index.json
    python crawl_docs.py --max-pages 5   # smoke test

Only the standard library is used (urllib). Every request carries a
User-Agent, a timeout, and up to two retries with exponential backoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from urllib.parse import urlparse

ROOT_INDEXES = [
    # 308-redirects to https://docs.langchain.com/llms.txt on Python 3.10.
    "https://python.langchain.com/llms.txt",
    "https://docs.langchain.com/llms.txt",
]
ALLOWED_HOSTS = {"docs.langchain.com", "python.langchain.com"}

MAX_INDEX_DEPTH = 5
MAX_INDEX_REQUESTS = 400
MAX_BODY_FETCHES = 700
TIMEOUT = 35
MAX_RETRIES = 2
DELAY = 0.25
MIN_BODY_CHARS = 200

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

TEMP = os.environ.get("TEMP") or os.environ.get("TMP") or "."
OUT_PATH = os.path.join(TEMP, "docs_large.json")
INDEX_PATH = os.path.join(TEMP, "docs_index.json")
STATS_PATH = os.path.join(TEMP, "docs_large_stats.json")

LINK_RE = re.compile(r"https?://[^\s)\"'<>\]]+")
URL_NOISE = ".,;:"

# Path fragments that mark an off-topic or pure-API-reference page.
EXCLUDE_TOKENS = (
    "smith-api",
    "agent-server-api",
    "api-reference",
    "/administration/",
    "/changelog",
    "release-notes",
    "whats-new",
    "pricing",
    "legal",
    "privacy",
    "terms-of",
    "security-policy",
    "self-host",
    "troubleshoot",
    "/contributing/",
    "/releases/",
    "release-policy",
    "versioning",
    "byoc",
    "billing",
    "/fleet/",
)

# On-topic path prefixes: Python-side docs only. The JavaScript tree is a
# translation of the same pages and would make retrieval ground truth
# ambiguous (both language variants answer the same question).
INCLUDE_PREFIXES = (
    "oss/python/",
    "oss/deepagents/",
    "oss/openwiki/",
    "langsmith/python/",
)

# Whole trees dropped for the same duplication reason.
JS_PREFIXES = ("oss/javascript/", "langsmith/javascript/")

BOILERPLATE_PREFIXES = (
    "> ## Documentation Index",
    "> Fetch the complete documentation index at:",
    "> Use this file to discover all available pages before exploring further.",
)

STEPS: list[str] = []
STATS: Counter = Counter()
FAILURES: list[dict] = []


class RedirectHandler(urllib.request.HTTPRedirectHandler):
    """Python 3.10 refuses 308 responses; handle them like 302."""

    http_error_308 = urllib.request.HTTPRedirectHandler.http_error_302

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if code not in (301, 302, 303, 307, 308):
            return None
        forwarded = {k: v for k, v in req.headers.items() if k.lower() not in ("content-length", "content-type")}
        forwarded["User-Agent"] = USER_AGENT
        return urllib.request.Request(
            newurl.replace(" ", "%20"),
            headers=forwarded,
            origin_req_host=req.origin_req_host,
            unverifiable=True,
        )


OPENER = urllib.request.build_opener(RedirectHandler)


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch(url: str, kind: str) -> str | None:
    """GET url with retries; returns decoded text or None after the last failure."""
    last_error = ""
    for attempt in range(MAX_RETRIES + 1):
        STATS[f"{kind}_requests"] += 1
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with OPENER.open(request, timeout=TIMEOUT) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, "replace")
        except Exception as exc:  # urllib raises a wide family here
            last_error = f"{type(exc).__name__}: {exc}"
            STATS[f"{kind}_errors"] += 1
            if attempt < MAX_RETRIES:
                STATS[f"{kind}_retries"] += 1
                time.sleep(1.5 * (2**attempt))
    FAILURES.append({"url": url, "stage": kind, "error": last_error[:160]})
    return None


def extract_links(text: str) -> list[str]:
    found = []
    for match in LINK_RE.finditer(text):
        url = match.group(0).rstrip(URL_NOISE)
        if urlparse(url).netloc in ALLOWED_HOSTS:
            found.append(url.split("#")[0])
    return found


def collect_md_urls() -> tuple[list[str], list[str]]:
    """Walk the .txt index tree and return (md urls in discovery order, txt urls)."""
    seen_txt: set[str] = set()
    md_seen: set[str] = set()
    md_order: list[str] = []
    queue = deque((url, 0) for url in ROOT_INDEXES)

    while queue and len(seen_txt) < MAX_INDEX_REQUESTS:
        url, depth = queue.popleft()
        if url in seen_txt:
            continue
        seen_txt.add(url)
        body = fetch(url, "index")
        if body is None:
            continue
        for link in extract_links(body):
            if link.endswith(".txt"):
                if depth + 1 <= MAX_INDEX_DEPTH and link not in seen_txt:
                    queue.append((link, depth + 1))
            elif link.endswith(".md") and link not in md_seen:
                md_seen.add(link)
                md_order.append(link)
        time.sleep(DELAY)

    log(f"[index] visited {len(seen_txt)} .txt indexes, found {len(md_order)} unique .md urls")
    return md_order, sorted(seen_txt)


def classify(url: str) -> str | None:
    """Return a topic section name when the url is on topic, else None."""
    path = urlparse(url).path.lstrip("/")
    lowered = path.lower()
    if lowered.startswith(JS_PREFIXES):
        STATS["excluded_javascript"] += 1
        return None
    if any(token in lowered for token in EXCLUDE_TOKENS):
        return None
    if not lowered.startswith(INCLUDE_PREFIXES):
        return None
    segments = lowered.split("/")
    for section in ("langgraph", "langchain", "concepts", "integrations", "deepagents", "migrate"):
        if section in segments:
            return section
    if lowered.startswith("oss/openwiki/"):
        return "openwiki"
    if lowered.startswith("langsmith/python/"):
        return "langsmith-python"
    return "oss-misc"


def clean_body(raw: str) -> tuple[str, str]:
    """Strip the shared docs header block; return (title, cleaned markdown)."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    lines = text.split("\n")

    # Drop the identical "Documentation Index" blockquote at the head of every page.
    start = 0
    while start < len(lines) and start < 20:
        stripped = lines[start].strip()
        if not stripped:
            start += 1
            continue
        if stripped.startswith(BOILERPLATE_PREFIXES):
            start += 1
            continue
        break
    lines = lines[start:]

    # A page sometimes repeats its H1 as the first body heading; keep one copy.
    kept: list[str] = []
    leading_title: str | None = None
    for line in lines:
        heading = re.match(r"^(#{1,2})\s+(.+?)\s*$", line)
        if heading and not kept_has_content(kept):
            title_text = heading.group(2)
            if leading_title is None:
                leading_title = title_text
                kept.append(line)
                continue
            if title_text == leading_title:
                continue
        kept.append(line)

    body = "\n".join(kept).replace("\u200b", "")
    # Fenced blocks carry renderer-only attributes: ```bash theme={"theme":{...}}
    body = re.sub(r'(?m)^(```[\w+#-]*)\s+theme=\{.*\}$', r"\1", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    title = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    return (title.group(1).strip() if title else ""), body


def kept_has_content(kept: list[str]) -> bool:
    return any(line.strip() for line in kept)


def short(text: str, limit: int = 120) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Crawl docs.langchain.com into a large RAG corpus.")
    parser.add_argument("--index-only", action="store_true", help="only collect .md urls")
    parser.add_argument("--max-pages", type=int, default=MAX_BODY_FETCHES, help="cap body fetches")
    parser.add_argument("--out", default=OUT_PATH, help="output json path (default: %%TEMP%%\\docs_large.json)")
    args = parser.parse_args()

    started = time.time()
    md_urls, txt_urls = collect_md_urls()

    if args.index_only:
        with open(INDEX_PATH, "w", encoding="utf-8") as handle:
            json.dump({"txt": txt_urls, "md": md_urls}, handle, ensure_ascii=False, indent=1)
        log(f"[index] saved {INDEX_PATH}")
        return 0

    selected = []
    for url in md_urls:
        section = classify(url)
        if section:
            selected.append((url, section))
        else:
            STATS["excluded_paths"] += 1
    log(f"[filter] {len(selected)} of {len(md_urls)} urls kept by topic filter")

    documents: list[dict] = []
    digests: dict[str, str] = {}
    too_short: list[dict] = []

    for url, section in selected:
        if len(documents) >= args.max_pages:
            break
        STATS["body_attempts"] += 1
        raw = fetch(url, "body")
        if raw is None:
            continue
        STATS["body_ok"] += 1
        title, body = clean_body(raw)
        if len(body) < MIN_BODY_CHARS:
            STATS["too_short"] += 1
            too_short.append({"url": url, "chars": len(body)})
            continue
        digest = hashlib.md5(body.encode("utf-8")).hexdigest()
        if digest in digests:
            STATS["duplicates"] += 1
            continue
        digests[digest] = url
        STATS["kept"] += 1
        documents.append(
            {
                "page_content": body,
                "metadata": {
                    "source": url,
                    "source_url": url,
                    "title": title or os.path.basename(urlparse(url).path).replace(".md", ""),
                    "section": section,
                },
            }
        )
        time.sleep(DELAY)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(documents, handle, ensure_ascii=False, indent=1)
    size_mb = os.path.getsize(args.out) / (1024 * 1024)

    lengths = sorted(len(doc["page_content"]) for doc in documents)
    sections = Counter(doc["metadata"]["section"] for doc in documents)
    report = {
        "index": {
            "txt_indexes_visited": len(txt_urls),
            "md_urls_unique": len(md_urls),
            "index_requests": STATS["index_requests"],
            "index_errors": STATS["index_errors"],
        },
        "filter": {
            "excluded_javascript": STATS["excluded_javascript"],
            "excluded_other": STATS["excluded_paths"],
            "kept_by_path": len(selected),
        },
        "fetch": {
            "body_attempted": STATS["body_attempts"],
            "body_ok": STATS["body_ok"],
            "body_failed": STATS["body_errors"],
            "body_retries": STATS["body_retries"],
            "failures": FAILURES[:10],
            "failure_count": len(FAILURES),
        },
        "content": {
            "too_short": STATS["too_short"],
            "duplicates_by_md5": STATS["duplicates"],
            "documents": len(documents),
            "output": args.out,
            "size_mb": round(size_mb, 2),
        },
        "lengths": {
            "min": lengths[0] if lengths else 0,
            "p50": int(statistics.median(lengths)) if lengths else 0,
            "max": lengths[-1] if lengths else 0,
            "mean": int(statistics.fmean(lengths)) if lengths else 0,
        },
        "sections": dict(sections.most_common()),
        "samples": [
            {"source": doc["metadata"]["source"], "title": doc["metadata"]["title"], "head": short(doc["page_content"])}
            for doc in documents[:3]
        ],
        "too_short_urls": too_short[:10],
        "elapsed_sec": round(time.time() - started, 1),
    }
    with open(STATS_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)

    log(f"[done] {len(documents)} docs -> {args.out} ({size_mb:.2f} MB) in {report['elapsed_sec']}s")
    log(f"[done] stats -> {STATS_PATH}")
    log(f"[done] lengths min/p50/max/mean = {report['lengths']}")
    log(f"[done] sections = {report['sections']}")
    for sample in report["samples"]:
        log(f"[sample] {sample['source']} | {sample['head'][:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
