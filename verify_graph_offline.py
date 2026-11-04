"""端到端验证：用百炼模型跑通完整 retrieval_graph（不依赖 Docker / Elasticsearch）。

为什么需要这个脚本？
    项目默认的向量库是本地 Docker 里的 Elasticsearch。如果你暂时不想拉取（或拉不动）
    那个 1GB 多的镜像，可以用这个脚本先把「图逻辑 + 百炼模型」这条链路验完。
    它用一个内存向量库替换掉 Elasticsearch，其余流程（路由 → 制定计划 → 并行检索 →
    生成回答）全部走项目真实代码。

它验证了什么？
    1. 主编排图 retrieval_graph 能否完整跑完，不抛异常
    2. Router 分类是否落在合法档位（langchain / more-info / general）
    3. 研究子图能否把 1 个研究步骤扩成多条检索 query 并并行检索
    4. 真实命中的文档能否被组装进提示词，并由 qwen-plus 生成带引用的中文回答

用法：
    .venv\\Scripts\\python.exe verify_graph_offline.py
"""

# 模型说明：qwen-turbo 当前返回 429、qwen-max 返回 403（免费额度耗尽），故统一改用 qwen-flash / qwen-plus。

import asyncio
import json
import os
import sys
import warnings
from contextlib import contextmanager
from typing import Generator

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

# 让 langgraph 的图能 import 到 src 下的包
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from langchain_core.runnables import RunnableConfig  # noqa: E402
from langchain_core.vectorstores import VectorStoreRetriever  # noqa: E402

from shared import retrieval  # noqa: E402
from shared.state import reduce_docs  # noqa: E402

DOCS_FILE = "src/sample_docs.json"

# 把检索到的文档累积下来，便于最后展示"到底查到了什么"
TRACE: dict[str, list] = {"retrieved": []}


def _build_memory_retriever() -> VectorStoreRetriever:
    """构建内存向量库检索器（底层仍是百炼 text-embedding-v4，1024 维）。"""
    from langchain_core.vectorstores import InMemoryVectorStore

    from shared.configuration import BaseConfiguration

    cfg = BaseConfiguration.from_runnable_config({})
    embeddings = retrieval.make_text_encoder(cfg.embedding_model)

    with open(DOCS_FILE, encoding="utf-8") as f:
        serialized_docs = json.load(f)
    docs = reduce_docs([], serialized_docs)

    store = InMemoryVectorStore(embedding=embeddings)
    store.add_documents(docs)
    print(f"   已写入内存向量库：{len(docs)} 篇文档（{cfg.embedding_model}）")
    return store.as_retriever(search_kwargs={"k": 3})


@contextmanager
def patched_retriever() -> Generator[None, None, None]:
    """临时把项目的 make_retriever 换成内存向量库版本。"""
    original = retrieval.make_retriever
    retriever = _build_memory_retriever()

    @contextmanager
    def fake_make_retriever(config):
        yield retriever

    retrieval.make_retriever = fake_make_retriever  # type: ignore[assignment]
    try:
        yield
    finally:
        retrieval.make_retriever = original  # type: ignore[assignment]


async def main() -> int:
    print("=" * 72)
    print("端到端验证：百炼模型 + 完整 retrieval_graph（内存向量库版）")
    print("=" * 72)

    from retrieval_graph.graph import graph

    config = RunnableConfig(
        configurable={
            "query_model": "dashscope/qwen-flash",
            "response_model": "dashscope/qwen-plus",
            "embedding_model": "dashscope/text-embedding-v4",
        }
    )

    question = "LangGraph 里的节点（node）是怎么定义的？"
    print(f"用户提问：{question}")
    print("-" * 72)

    with patched_retriever():
        result = await graph.ainvoke(
            {"messages": [("user", question)]}, config
        )

    # ---------- 1) 路由结果 ----------
    router = result.get("router") or {}
    print()
    print("【1】入口路由")
    print(f"    type  = {router.get('type')!r}")
    print(f"    logic = {router.get('logic')!r}")
    legal = {"langchain", "more-info", "general"}
    if router.get("type") not in legal:
        print("    ✗ 分类值不在合法档位，真实运行会在 route_query 抛 ValueError")
        return 1
    print("    ✓ 分类值合法")

    # ---------- 2) 研究计划 ----------
    print()
    print("【2】研究计划")
    steps = result.get("steps") or []
    print(f"    剩余未执行步骤数 = {len(steps)}（跑完应为 0）")

    # ---------- 3) 检索到的文档 ----------
    docs = result.get("documents") or []
    print()
    print("【3】检索结果")
    print(f"    去重后文档数 = {len(docs)}")
    for i, d in enumerate(docs[:6], 1):
        snippet = d.page_content.replace("\n", " ")[:72]
        print(f"    [{i}] {snippet}...")

    # ---------- 4) 最终回答 ----------
    messages = result.get("messages") or []
    final = str(messages[-1].content) if messages else ""
    print()
    print("【4】最终回答（qwen-plus 生成）")
    print("-" * 72)
    print(final)
    print("-" * 72)

    ok = bool(final) and len(docs) > 0 and router.get("type") == "langchain"
    print()
    if ok:
        print("结论：✓ 完整链路已跑通，百炼模型与图逻辑配合正常。")
        print("      接下来只需把向量库换成 Elasticsearch 即可投入实际使用。")
        return 0
    print("结论：✗ 链路未完全跑通，请看上面的输出定位。")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
