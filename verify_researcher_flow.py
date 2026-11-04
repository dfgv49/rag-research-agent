"""实测 researcher_graph 的真实运行细节。

验证这些问题：
  1. 一个研究步骤会被扩写成几条检索词？
  2. Send 派发了几个并行任务？它们是同时在跑吗？
  3. 每个并行任务收到的 state 是什么（QueryState 的作用）？
  4. 并行结果如何被 reduce_docs 合并？
  5. 每个 retrieve_documents 任务能收到 config 吗？

用法:
    .venv\\Scripts\\python.exe verify_researcher_flow.py
"""

# 模型说明：qwen-turbo 当前返回 429、qwen-max 返回 403（免费额度耗尽），故统一改用 qwen-flash / qwen-plus。

import asyncio
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
os.environ.setdefault("ELASTICSEARCH_URL", "http://127.0.0.1:9200")

from langchain_core.runnables import RunnableConfig  # noqa: E402

import retrieval_graph.researcher_graph.graph as rg  # noqa: E402

# ---- 给三个函数打上探针，记录真实调用 ----
original_generate = rg.generate_queries
original_retrieve = rg.retrieve_documents
original_parallel = rg.retrieve_in_parallel

events: list[tuple[str, float, str]] = []
T0 = time.perf_counter()


def stamp(msg: str) -> None:
    events.append((msg, time.perf_counter() - T0, ""))


async def patched_generate(state, *, config):
    stamp("generate_queries 开始执行")
    out = await original_generate(state, config=config)
    stamp(f"generate_queries 产出 {len(out['queries'])} 条检索词")
    return out


async def patched_retrieve(state, *, config):
    q = getattr(state, "query", None)
    stamp(f"retrieve_documents 开始 | query={q!r}")
    out = await original_retrieve(state, config=config)
    stamp(f"retrieve_documents 结束 | query={q!r} | 命中 {len(out['documents'])} 篇")
    return out


def patched_parallel(state):
    sends = original_parallel(state)
    stamp(f"retrieve_in_parallel 派发 {len(sends)} 个 Send 任务")
    for s in sends:
        print(f"       Send -> node={s.node!r}, arg={s.arg!r}, arg 类型={type(s.arg).__name__}")
    return sends


async def main() -> None:
    print("=" * 74)
    print("researcher_graph 运行实测")
    print("=" * 74)

    rg.generate_queries = patched_generate
    rg.retrieve_documents = patched_retrieve
    rg.retrieve_in_parallel = patched_parallel

    # 需要重新编译图，让节点指向打过补丁的函数
    from langgraph.graph import END, START, StateGraph

    from retrieval_graph.researcher_graph.state import ResearcherState

    builder = StateGraph(ResearcherState)
    # 注意：必须显式指定节点名，否则 LangGraph 会用函数名（patched_generate）
    # 作为节点名，导致后面的边找不到 generate_queries。
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

    question = "LangGraph 里节点是怎么定义的？"
    print(f"输入的研究步骤：{question}")
    print()
    print("派发的 Send 任务明细：")

    config = RunnableConfig(
        configurable={
            "query_model": "dashscope/qwen-flash",
            "embedding_model": "dashscope/text-embedding-v4",
            "retriever_provider": "elastic-local",
        }
    )

    t = time.perf_counter()
    result = await graph.ainvoke({"question": question}, config)
    elapsed = time.perf_counter() - t

    print()
    print("=" * 74)
    print("执行时间线（相对开始，单位毫秒）")
    print("=" * 74)
    for msg, offset, _ in events:
        print(f"  {offset * 1000:8.1f} ms  {msg}")

    print()
    print("=" * 74)
    print("产物")
    print("=" * 74)
    queries = result.get("queries") or []
    docs = result.get("documents") or []
    print(f"  生成的检索词（{len(queries)} 条）：")
    for i, q in enumerate(queries, 1):
        print(f"     {i}. {q}")
    print()
    print(f"  合并后的文档数 = {len(docs)}")
    print(f"  其中唯一 uuid 数 = {len({d.metadata.get('uuid') for d in docs})}")
    print()
    print(f"  整个子图耗时 = {elapsed:.2f} 秒")


if __name__ == "__main__":
    asyncio.run(main())
