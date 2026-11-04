"""出题辅助工具：按关键词找文档 uuid，并校验测试集。

为什么需要它：`eval/testcases.yaml` 里的 `expected_doc_uuids` 必须真实对应语料文档，
而这个 uuid 是 `page_content` 的 md5（见 `src/shared/state.py:_generate_uuid`），
人眼看不出对应关系，只能靠工具换算。

用法
----
1) 按关键词找文档（出题时最常用）：
   .\\.venv\\Scripts\\python.exe find_doc.py --grep checkpointer

2) 只看标题：
   .\\.venv\\Scripts\\python.exe find_doc.py --grep "并行" --title-only

3) 全文搜 + 限制条数：
   .\\.venv\\Scripts\\python.exe find_doc.py --grep "add_conditional_edges" --limit 5

4) 校验测试集里所有 expected_doc_uuids 是否都存在于语料：
   .\\.venv\\Scripts\\python.exe find_doc.py --validate eval/testcases.yaml

5) 指定语料（默认依次尝试 src/sample_docs_large.json → src/sample_docs.json）：
   .\\.venv\\Scripts\\python.exe find_doc.py --corpus src/sample_docs.json --grep nodes

只读脚本，不写任何文件。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid as uuidlib
from typing import Any, Optional


# 候选语料路径，按优先级尝试（大语料优先）
DEFAULT_CORPORA = ("src/sample_docs_large.json", "src/sample_docs.json")


def doc_uuid(page_content: str) -> str:
    """与 shared.state._generate_uuid 保持一致：page_content 的 md5 → uuid。"""
    return str(uuidlib.UUID(hashlib.md5(page_content.encode()).hexdigest()))


def resolve_corpus(explicit: Optional[str]) -> str:
    """定位语料文件。显式指定优先，否则按 DEFAULT_CORPORA 顺序找第一个存在的。"""
    if explicit:
        if not os.path.exists(explicit):
            raise SystemExit(f"语料不存在：{explicit}")
        return explicit
    for path in DEFAULT_CORPORA:
        if os.path.exists(path):
            return path
    raise SystemExit(
        "找不到语料文件。已尝试：\n  " + "\n  ".join(DEFAULT_CORPORA) +
        "\n请用 --corpus 指定。"
    )


def load_corpus(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise SystemExit(f"语料格式应为 JSON 数组，实际是 {type(data).__name__}：{path}")
    return data


def load_testcases(path: str) -> list[dict[str, Any]]:
    try:
        import yaml
    except ImportError:
        raise SystemExit("需要 pyyaml 才能读 YAML 测试集（已确认环境内有 6.0.3）")
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if isinstance(data, dict):
        data = data.get("cases") or []
    return data or []


def meta_of(doc: dict[str, Any], key: str, default: str = "") -> str:
    metadata = doc.get("metadata") or {}
    return str(metadata.get(key, default) or default)


def cmd_grep(docs: list[dict[str, Any]], keyword: str, limit: int, title_only: bool) -> int:
    """在正文与标题里做大小写不敏感的子串搜索，打印 uuid + 元信息。"""
    needle = keyword.lower()
    hits = []
    for doc in docs:
        content = doc.get("page_content") or ""
        title = meta_of(doc, "title")
        if needle in content.lower() or needle in title.lower():
            hits.append(doc)

    if not hits:
        print(f"没找到包含 {keyword!r} 的文档（共 {len(docs)} 篇）。换个关键词试试。")
        return 0

    print(f"命中 {len(hits)} 篇（共语料 {len(docs)} 篇），显示前 {min(limit, len(hits))} 篇：\n")
    for index, doc in enumerate(hits[:limit], start=1):
        uid = doc_uuid(doc.get("page_content") or "")
        title = meta_of(doc, "title") or "(无标题)"
        source = meta_of(doc, "source") or "(无来源)"
        section = meta_of(doc, "section")
        print(f"[{index}] uuid={uid}")
        print(f"    标题  ：{title}")
        print(f"    来源  ：{source}")
        if section:
            print(f"    分区  ：{section}")
        if not title_only:
            content = " ".join((doc.get("page_content") or "").split())
            print(f"    正文  ：{content[:160]}…")
        print()

    if len(hits) > limit:
        print(f"（还有 {len(hits) - limit} 篇未显示，用 --limit 调整）")
    return 0


def cmd_validate(corpus_path: str, docs: list[dict[str, Any]], testcases_path: str) -> int:
    """校验测试集：每个 expected_doc_uuids 是否都存在于语料；顺带报表统计。"""
    known = {doc_uuid(doc.get("page_content") or "") for doc in docs}
    cases = load_testcases(testcases_path)
    if not cases:
        print(f"测试集为空或格式不对：{testcases_path}")
        return 1

    print(f"语料：{corpus_path}（{len(docs)} 篇）")
    print(f"测试集：{testcases_path}（{len(cases)} 条）\n")

    by_category: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_category.setdefault(str(case.get("category", "?")), []).append(case)

    missing_total = 0
    problems: list[str] = []

    print("按类别统计（真实 / TODO / 缺 expected）")
    print("-" * 58)
    for category in sorted(by_category):
        group = by_category[category]
        todo = sum(1 for c in group if c.get("TODO"))
        real = len(group) - todo
        no_expected = sum(
            1 for c in group
            if not c.get("TODO") and not (c.get("expected_doc_uuids") or [])
        )
        print(f"  {category:16} 共 {len(group):3}  真实 {real:3}  TODO {todo:3}  缺期望 {no_expected:3}")
    print("-" * 58)

    print("\n逐条校验 expected_doc_uuids：")
    for case in cases:
        if case.get("TODO"):
            continue
        cid = case.get("id", "?")
        expected = case.get("expected_doc_uuids") or []
        if not expected:
            if case.get("category") != "no_answer":
                problems.append(f"{cid}: 真实用例但没有任何 expected_doc_uuids")
            continue
        for uid in expected:
            if uid not in known:
                missing_total += 1
                problems.append(f"{cid}: uuid 不在语料中 → {uid}")

    if problems:
        print(f"\n发现 {len(problems)} 个问题：")
        for line in problems[:40]:
            print("  ✗", line)
        if len(problems) > 40:
            print(f"  …（还有 {len(problems) - 40} 条）")
        print(f"\n结论：不通过（缺失 uuid {missing_total} 个）")
        return 1

    print("\n结论：✓ 全部通过 —— 所有 expected_doc_uuids 都能在语料中找到")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="出题辅助：按关键词找文档 uuid / 校验测试集",
    )
    parser.add_argument("--corpus", help="语料 JSON 路径（默认自动找 large → sample）")
    parser.add_argument("--grep", help="按关键词搜索正文与标题")
    parser.add_argument("--limit", type=int, default=10, help="最多显示几条（默认 10）")
    parser.add_argument("--title-only", action="store_true", help="只显示标题，不显示正文摘要")
    parser.add_argument("--validate", metavar="TESTCASES", help="校验测试集 YAML")
    args = parser.parse_args()

    if not args.grep and not args.validate:
        parser.print_help()
        return 1

    corpus_path = resolve_corpus(args.corpus)
    docs = load_corpus(corpus_path)

    if args.grep:
        return cmd_grep(docs, args.grep, args.limit, args.title_only)
    return cmd_validate(corpus_path, docs, args.validate)


if __name__ == "__main__":
    sys.exit(main())
