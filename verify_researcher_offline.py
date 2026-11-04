"""实测 researcher_graph 的运行细节（内存向量库版，不依赖 Docker/ES）。

验证问题：
  1. 一个研究步骤被扩写成几条检索词？
  2. Send 派发了几个并行任务？每个任务的载荷（arg）是什么类型？
  3. 并行任务真的同时在跑吗？总耗时接近单条还是一串相加？
  4. 并行结果如何被 reduce_docs 合并去重？
  5. 每个并行任务能不能收到 config？

用法:
    .venv\\Scripts\\python.exe verify_researcher_offline.py
"""

# 模型说明：qwen-turbo 当前返回 429、qwen-max 返回 403（免费额度耗尽），故统一改用 qwen-flash / qwen-plus。

import asyncio
import json
import os
import sys
import time
import warnings
from contextlib import contextmanager

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from langchain_core.runnables import RunnableConfig  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

import retrieval_graph.researcher_graph.graph as rg  # noqa: E402
from retrieval_graph.researcher_graph.state import ResearcherState  # noqa: E402
from shared import retrieval  # noqa: E402
from shared.configuration import BaseConfiguration  # noqa: E402
from shared.state import reduce_docs  # noqa: E402

DOCS_FILE = "src/sample_docs.json"

events: list[tuple[str, float]] = []
T0 = time.perf_counter()


def stamp(msg: str) -> None:
    events.append((msg, time.perf_counter() - T0))


def build_memory_retriever():
    from langchain_core.vectorstores import InMemoryVectorStore

    cfg = BaseConfiguration.from_runnable_config({})
    emb = retrieval.make_text_encoder(cfg.embedding_model)
    with open(DOCS_FILE, encoding="utf-8") as f:
        docs = reduce_docs([], json.load(f))
    store = InMemoryVectorStore(embedding=emb)
    store.add_documents(docs)
    print(f"    内存向量库已就绪：{len(docs)} 篇文档")
    return store.as_retriever(search_kwargs={"k": 4})


@contextmanager
def patched_retriever():
    original = retrieval.make_retriever
    retriever = build_memory_retriever()

    @contextmanager
    def fake(config):
        yield retriever

    retrieval.make_retriever = fake  # type: ignore[assignment]
    try:
        yield
    finally:
        retrieval.make_retriever = original  # type: ignore[assignment]


original_generate = rg.generate_queries
original_retrieve = rg.retrieve_documents
original_parallel = rg.retrieve_in_parallel


async def patched_generate(state, *, config):
    stamp("generate_queries 开始")
    out = await original_generate(state, config=config)
    stamp(f"generate_queries 结束，产出 {len(out['queries'])} 条检索词")
    return out


async def patched_retrieve(state, *, config):
    q = state.query
    stamp(f"  ├ retrieve 开始  query={q!r}")
    out = await original_retrieve(state, config=config)
    stamp(f"  ├ retrieve 结束  query={q!r} 命中 {len(out['documents'])} 篇")
    return out


def patched_parallel(state):
    sends = original_parallel(state)
    stamp(f"retrieve_in_parallel 一次派发 {len(sends)} 个 Send")
    print()
    print("    Send 明细：")
    for s in sends:
        print(f"      node = {s.node!r}")
        print(f"      arg  = {s.arg!r}   (类型 {type(s.arg).__name__})")
    return sends


async def main() -> int:
    print("=" * 74)
    print("researcher_graph 运行实测（内存向量库，不依赖 Docker）")
    print("=" * 74)
    print("准备：")
    with patched_retriever():
        builder = StateGraph(ResearcherState)
        builder.add_node("generate_queries", patched_generate)
        builder.add_node("retrieve_documents", patched_retrieve)
        builder.add_edge(START, "generate_queries")
        builder.add_conditional_edges(
            "generate_queries",
            patched_parallel,
            path_map=["retrieve_documents"],
        )
        builder.add_edge("retrieve_documents", END)
        graph = builder.compile()

        config = RunnableConfig(
            configurable={
                "query_model": "dashscope/qwen-flash",
                "embedding_model": "dashscope/text-embedding-v4",
                "retriever_provider": "elastic-local",
            }
        )

        question = "LangGraph 里节点是怎么定义的？"
        print()
        print(f"输入研究步骤：{question}")

        t = time.perf_counter()
        result = await graph.ainvoke({"question": question}, config)
        elapsed = time.perf_counter() - t

    print()
    print("=" * 74)
    print("执行时间线（相对开始，毫秒）")
    print("=" * 74)
    for msg, off in events:
        print(f"  {off * 1000:8.1f} ms  {msg}")

    print()
    print("=" * 74)
    print("产物")
    print("=" * 74)
    queries = result.get("queries") or []
    docs = result.get("documents") or []
    print(f"  检索词 {len(queries)} 条：")
    for i, q in enumerate(queries, 1):
        print(f"     {i}. {q}")
    print()
    print(f"  合并后文档数        = {len(docs)}")
    print(f"  其中唯一 uuid 数    = {len({d.metadata.get('uuid') for d in docs})}")
    print(f"  子图总耗时          = {elapsed:.2f} 秒")

    # 并行性分析：看三个 retrieve 的起止是否重叠
    starts = [o for m, o in events if "retrieve 开始" in m]
    ends = [o for m, o in events if "retrieve 结束" in m]
    print()
    print("=" * 74)
    print("并行性分析")
    print("=" * 74)
    if starts and ends:
        span = max(ends) - min(starts)
        print(f"  第一个 retrieve 开始 -> 最后一个结束 的跨度 = {span * 1000:.0f} ms")
        print(f"  三个任务的时间区间：")
        for i, (s, e) in enumerate(zip(sorted(starts), sorted(ends)), 1):
            print(f"    任务{i}: {s * 1000:7.0f} ms -> {e * 1000:7.0f} ms  (时长 {(e - s) * 1000:.0f} ms)")
        print()
        print("  如果三个任务的区间【互相重叠】，说明是并发执行；")
        print("  如果【首尾相接不重叠】，说明退化成串行了。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
