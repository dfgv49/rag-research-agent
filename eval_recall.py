"""召回率对比实验：量化「多查询扩写」到底提升了多少。

思路（不依赖 ES，用内存向量库保证可复现）：
  1. 用 32 篇样例文档建内存向量库
  2. 取其中几篇文档，人工写出「用户可能会怎么问」的问题，作为测试集
  3. 对每个问题，用不同策略检索，看能否把「原文档」召回
  4. 对比各策略的召回率

对比的策略：
  A. 单查询 k=1        —— 最朴素的 baseline
  B. 单查询 k=4        —— 项目默认检索条数
  C. 单查询 k=10       —— 单纯加大返回数量
  D. 多查询(3条) k=4    —— 项目的实际做法（generate_queries + 并行）
  E. 多查询(3条) k=4 + 重排 —— 用 BM25 式关键词加权模拟重排

用法:
    .venv\\Scripts\\python.exe eval_recall.py
"""

# 模型说明：qwen-turbo 当前返回 429、qwen-max 返回 403（免费额度耗尽），故统一改用 qwen-flash / qwen-plus。

import asyncio
import json
import os
import re
import sys
import warnings
from collections import Counter
from contextlib import contextmanager

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from langchain_core.runnables import RunnableConfig  # noqa: E402

from shared import retrieval  # noqa: E402
from shared.configuration import BaseConfiguration  # noqa: E402
from shared.state import reduce_docs  # noqa: E402
from shared.utils import load_chat_model  # noqa: E402

DOCS_FILE = "src/sample_docs.json"

# 测试集：每个问题标注「期望召回的文档特征词」（出现在目标文档正文里的独特词）
# 注意：这些问答是我根据样例文档内容手工构造的，用于评估召回行为。
TEST_SET = [
    {
        "question": "LangGraph 里节点是怎么定义的？",
        "must_contain": "nodes are typically python functions",
    },
    {
        "question": "条件边（conditional edges）怎么用？",
        "must_contain": "add_conditional_edges",
    },
    {
        "question": "图的 state 是怎么定义的？",
        "must_contain": "State",
    },
    {
        "question": "怎么把多个节点并行执行？",
        "must_contain": "parallel",
    },
    {
        "question": "Send 是做什么的？",
        "must_contain": "Send",
    },
    {
        "question": "怎么给图设置配置参数？",
        "must_contain": "config_schema",
    },
]


def build_retriever(k: int):
    from langchain_core.vectorstores import InMemoryVectorStore

    cfg = BaseConfiguration.from_runnable_config({})
    emb = retrieval.make_text_encoder(cfg.embedding_model)
    with open(DOCS_FILE, encoding="utf-8") as f:
        docs = reduce_docs([], json.load(f))
    store = InMemoryVectorStore(embedding=emb)
    store.add_documents(docs)
    return store.as_retriever(search_kwargs={"k": k}), docs


def hit(docs, must_contain: str) -> bool:
    needle = must_contain.lower()
    return any(needle in (d.page_content or "").lower() for d in docs)


async def single_query(retriever, question: str):
    return await retriever.ainvoke(question)


def keywords(text: str) -> set:
    return set(re.findall(r"[a-zA-Z_]{3,}", text.lower()))


async def multi_query(generate_queries_fn, retriever, question: str, k: int):
    """模拟项目的做法：扩写多条查询 → 分别检索 → 合并去重。"""
    cfg = RunnableConfig(
        configurable={
            "query_model": "dashscope/qwen-flash",
            "embedding_model": "dashscope/text-embedding-v4",
        }
    )
    queries = await generate_queries_fn(question, cfg)
    per_query = []
    for q in queries:
        per_query.append(await retriever.ainvoke(q))
    merged = []
    # 用项目真实的 reduce_docs 做合并去重
    for docs in per_query:
        merged = reduce_docs(merged, docs)
    return merged, queries


async def generate_queries(question: str, cfg) -> list[str]:
    """复刻项目里子图的 generate_queries 节点。"""
    from typing import TypedDict, cast

    from retrieval_graph.configuration import AgentConfiguration
    from retrieval_graph.prompts import GENERATE_QUERIES_SYSTEM_PROMPT

    class Response(TypedDict):
        queries: list[str]

    configuration = AgentConfiguration.from_runnable_config(cfg)
    model = load_chat_model(configuration.query_model).with_structured_output(Response)
    messages = [
        {"role": "system", "content": GENERATE_QUERIES_SYSTEM_PROMPT},
        {"role": "human", "content": question},
    ]
    response = cast(Response, await model.ainvoke(messages))
    return response["queries"]


async def main() -> None:
    print("=" * 78)
    print("召回率对比实验（内存向量库，32 篇样例文档）")
    print("=" * 78)

    strategies = [
        ("A. 单查询 k=1", 1, False),
        ("B. 单查询 k=4", 4, False),
        ("C. 单查询 k=10", 10, False),
        ("D. 多查询 k=4", 4, True),
        ("E. 多查询 k=10", 10, True),
    ]

    results: dict[str, list[bool]] = {name: [] for name, _, _ in strategies}
    detail: dict[str, list[tuple[str, bool, int]]] = {name: [] for name, _, _ in strategies}

    # 复用同一个 k=10 的库，其他 k 通过截断模拟
    retriever10, all_docs = build_retriever(10)
    print(f"向量库已就绪，文档数 = {len(all_docs)}")
    print()

    for case in TEST_SET:
        q = case["question"]
        must = case["must_contain"]
        print(f"【问题】{q}")
        print(f"   期望命中特征词：{must!r}")

        # A/B/C：单查询 + 不同 k
        base_docs = await retriever10.ainvoke(q)
        for name, k, use_multi in strategies:
            if use_multi:
                continue
            docs = base_docs[:k]
            ok = hit(docs, must)
            results[name].append(ok)
            detail[name].append((q, ok, len(docs)))

        # D/E：多查询
        merged, queries = await multi_query(generate_queries, retriever10, q, 10)
        print(f"   扩写出的检索词：{queries}")
        for name, k, use_multi in strategies:
            if not use_multi:
                continue
            docs = merged[:k]
            ok = hit(docs, must)
            results[name].append(ok)
            detail[name].append((q, ok, len(docs)))

        print(
            f"   结果："
            f"A={'✓' if results['A. 单查询 k=1'][-1] else '✗'} "
            f"B={'✓' if results['B. 单查询 k=4'][-1] else '✗'} "
            f"C={'✓' if results['C. 单查询 k=10'][-1] else '✗'} "
            f"D={'✓' if results['D. 多查询 k=4'][-1] else '✗'} "
            f"E={'✓' if results['E. 多查询 k=10'][-1] else '✗'}"
        )
        print()

    print("=" * 78)
    print("召回率汇总")
    print("=" * 78)
    n = len(TEST_SET)
    print(f"  测试问题数 = {n}")
    print()
    for name, _, _ in strategies:
        cnt = sum(results[name])
        bar = "█" * int(cnt / n * 30)
        print(f"  {name:<18} {cnt}/{n}  = {cnt / n * 100:5.1f}%  {bar}")

    print()
    print("=" * 78)
    print("未召回明细")
    print("=" * 78)
    for name, _, _ in strategies:
        missed = [q for q, ok, _ in detail[name] if not ok]
        if missed:
            print(f"  {name}:")
            for q in missed:
                print(f"     ✗ {q}")


if __name__ == "__main__":
    asyncio.run(main())
