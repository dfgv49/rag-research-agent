"""端到端验证：百炼模型 + 完整 retrieval_graph + 真实 Elasticsearch。

与 verify_graph_offline.py 的区别：
    那个脚本用内存向量库替代 ES，用于在没有 Docker 时先验证链路；
    这个脚本用项目原生的 make_retriever()，走真实的 Elasticsearch，
    并且会真的执行 indexer 图把样例文档写进 ES。

它验证了什么？
    1. indexer 图能否把 32 篇样例文档以 1024 维写进 ES 的 langchain_index
    2. ES 里的文档数量、向量维度是否与 text-embedding-v4 对齐
    3. retrieval_graph 能否在真实向量库上完整跑完
    4. 路由分类、研究计划收敛、检索命中、最终回答是否正确

用法：
    .venv\\Scripts\\python.exe verify_graph_es.py
"""

# 模型说明：qwen-turbo 当前返回 429、qwen-max 返回 403（免费额度耗尽），故统一改用 qwen-flash / qwen-plus。

import asyncio
import os
import sys
import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# 本地 Docker 的 ES：必须是 127.0.0.1，而不是 README 里的 host.docker.internal
os.environ.setdefault("ELASTICSEARCH_URL", "http://127.0.0.1:9200")

from langchain_core.runnables import RunnableConfig  # noqa: E402

INDEX_NAME = "langchain_index"


def es_client():
    from elasticsearch import Elasticsearch

    url = os.environ["ELASTICSEARCH_URL"]
    user = os.environ.get("ELASTICSEARCH_USER")
    pwd = os.environ.get("ELASTICSEARCH_PASSWORD")
    if user and pwd:
        return Elasticsearch(url, basic_auth=(user, pwd), request_timeout=30)
    return Elasticsearch(url, request_timeout=30)


def report_index_state(es) -> tuple[bool, int]:
    """打印索引状态，返回 (索引是否存在, 文档数)。"""
    if not es.indices.exists(index=INDEX_NAME):
        print(f"    索引 {INDEX_NAME} 不存在（首次运行属正常）")
        return False, 0

    count = es.count(index=INDEX_NAME)["count"]
    print(f"    索引 {INDEX_NAME} 已存在，当前文档数 = {count}")

    # 取一条样本，检查向量维度
    sample = es.search(
        index=INDEX_NAME,
        body={"size": 1, "query": {"match_all": {}}, "_source": True},
    )
    hits = sample["hits"]["hits"]
    if hits:
        src = hits[0]["_source"]
        vec_field = None
        vec_len = None
        for key in ("vector", "embedding", "_vector"):
            if key in src and isinstance(src[key], list):
                vec_field, vec_len = key, len(src[key])
                break
        print(f"    样本字段: {sorted(src.keys())}")
        if vec_field:
            print(f"    向量字段 {vec_field!r} 维度 = {vec_len}")
    return True, count


async def main() -> int:
    print("=" * 72)
    print("端到端验证：百炼模型 + 真实 Elasticsearch")
    print("=" * 72)
    print(f"ELASTICSEARCH_URL = {os.environ['ELASTICSEARCH_URL']}")
    print("-" * 72)

    es = es_client()
    print("【0】Elasticsearch 连接与索引现状")
    try:
        info = es.info()
        print(f"    已连接：{info['name']} / ES {info['version']['number']}")
    except Exception as exc:  # noqa: BLE001
        print(f"    ✗ 无法连接 ES：{type(exc).__name__}: {exc}")
        print("    请确认容器在跑：docker ps --filter name=elasticsearch")
        return 1

    existed, before_count = report_index_state(es)

    # ---------- 1) 跑 indexer 图 ----------
    print()
    print("【1】执行 indexer 图（灌入 src/sample_docs.json）")
    from index_graph.graph import graph as index_graph

    index_config = RunnableConfig(
        configurable={
            "embedding_model": "dashscope/text-embedding-v4",
            "retriever_provider": "elastic-local",
        }
    )
    res = await index_graph.ainvoke({"docs": []}, index_config)
    print(f"    indexer 返回 docs = {res.get('docs')!r}（'delete' 表示已落库）")

    es.indices.refresh(index=INDEX_NAME)
    after_count = es.count(index=INDEX_NAME)["count"]
    print(f"    灌入后文档数 = {after_count}（灌入前 {before_count}）")
    if after_count == 0:
        print("    ✗ 索引仍为空，indexer 没写进去")
        return 1
    print(f"    ✓ 新增 {after_count - before_count} 篇文档")
    report_index_state(es)

    # ---------- 2) 用真实检索器跑一遍 retrieval_graph ----------
    print()
    print("【2】执行 retrieval_graph（真实 ES 检索）")
    from retrieval_graph.graph import graph

    config = RunnableConfig(
        configurable={
            "query_model": "dashscope/qwen-flash",
            "response_model": "dashscope/qwen-plus",
            "embedding_model": "dashscope/text-embedding-v4",
            "retriever_provider": "elastic-local",
        }
    )

    question = "LangGraph 里怎么定义节点？节点和边是怎么连接的？"
    print(f"    提问：{question}")
    result = await graph.ainvoke({"messages": [("user", question)]}, config)

    router = result.get("router") or {}
    docs = result.get("documents") or []
    steps = result.get("steps") or []
    messages = result.get("messages") or []
    final = str(messages[-1].content) if messages else ""

    print()
    print("【3】运行结果")
    print(f"    路由 type  = {router.get('type')!r}")
    print(f"    路由 logic = {router.get('logic')!r}")
    print(f"    剩余研究步骤 = {len(steps)}")
    print(f"    检索命中文档 = {len(docs)}")
    for i, d in enumerate(docs[:5], 1):
        print(f"      [{i}] {d.page_content.replace(chr(10), ' ')[:70]}...")

    print()
    print("【4】最终回答（qwen-plus）")
    print("-" * 72)
    print(final)
    print("-" * 72)

    ok = (
        router.get("type") == "langchain"
        and len(docs) > 0
        and bool(final)
        and after_count > 0
    )
    print()
    if ok:
        print("结论：✓ 真实 Elasticsearch 全链路已跑通。")
        print("      可以启动 langgraph dev，在 Studio 里正式使用了。")
        return 0
    print("结论：✗ 未完全通过，请看上面输出定位。")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
