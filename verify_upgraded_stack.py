"""验证「升级后的技术栈」能否跑通本项目（不动现有 .venv）。

Studio 提示需要 langgraph-api >= 0.11.0，而这会连带把 langgraph / langchain /
langchain-core 一起拉高。为了不破坏已经跑通的 .venv，本脚本在 .venv-test 中运行，
用升级后的依赖验证同一套模板代码是否仍然正常。

用法：
    .venv-test\\Scripts\\python.exe verify_upgraded_stack.py
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
os.environ.setdefault("ELASTICSEARCH_URL", "http://127.0.0.1:9200")

import importlib.metadata as md  # noqa: E402

from langchain_core.runnables import RunnableConfig  # noqa: E402


def show_versions() -> None:
    print("当前测试环境版本：")
    for p in [
        "langgraph",
        "langgraph-api",
        "langgraph-cli",
        "langgraph-runtime-inmem",
        "langgraph-checkpoint",
        "langchain",
        "langchain-core",
        "langchain-openai",
        "langchain-elasticsearch",
    ]:
        try:
            print(f"    {p:<28} {md.version(p)}")
        except Exception:  # noqa: BLE001
            print(f"    {p:<28} 缺失")
    print()


async def main() -> int:
    print("=" * 72)
    print("升级后技术栈回归验证")
    print("=" * 72)
    show_versions()

    failures: list[str] = []

    # ---------- 1) 项目模块能否正常 import ----------
    print("【1】项目模块导入")
    try:
        from retrieval_graph.configuration import AgentConfiguration  # noqa: F401
        from retrieval_graph.graph import graph  # noqa: F401
        from shared.retrieval import make_text_encoder
        from shared.utils import format_docs, load_chat_model

        cfg = AgentConfiguration.from_runnable_config({})
        print(f"    ✓ 导入成功")
        print(f"      query_model     = {cfg.query_model}")
        print(f"      response_model  = {cfg.response_model}")
        print(f"      embedding_model = {cfg.embedding_model}")
    except Exception as exc:  # noqa: BLE001
        print(f"    ✗ 导入失败：{type(exc).__name__}: {exc}")
        return 1

    # ---------- 2) 百炼对话模型 ----------
    print()
    print("【2】百炼对话模型（新 langchain-core 下）")
    try:
        llm = load_chat_model("dashscope/qwen-flash")
        out = llm.invoke("用一个词回答：1+1 等于几？")
        print(f"    ✓ {str(out.content)[:60]}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"对话模型: {exc}")
        print(f"    ✗ {type(exc).__name__}: {exc}")

    # ---------- 3) 结构化输出（版本升级最容易坏的地方）----------
    print()
    print("【3】结构化输出 with_structured_output")
    try:
        from retrieval_graph.state import Router

        r = load_chat_model("dashscope/qwen-flash").with_structured_output(Router)
        got = r.invoke("LangGraph 里怎么定义节点？")
        print(f"    ✓ Router -> {got}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"结构化输出: {exc}")
        print(f"    ✗ {type(exc).__name__}: {exc}")

    # ---------- 4) 百炼向量模型 ----------
    print()
    print("【4】百炼向量模型")
    try:
        emb = make_text_encoder("dashscope/text-embedding-v4")
        vec = emb.embed_query("测试")
        print(f"    ✓ 维度 = {len(vec)}")
        if len(vec) != 1024:
            failures.append(f"向量维度异常: {len(vec)}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"向量模型: {exc}")
        print(f"    ✗ {type(exc).__name__}: {exc}")

    # ---------- 5) 真实 ES 端到端 ----------
    print()
    print("【5】真实 Elasticsearch 端到端（retrieval_graph）")
    try:
        config = RunnableConfig(
            configurable={
                "query_model": "dashscope/qwen-flash",
                "response_model": "dashscope/qwen-plus",
                "embedding_model": "dashscope/text-embedding-v4",
                "retriever_provider": "elastic-local",
            }
        )
        q = "LangGraph 里的条件边怎么用？"
        print(f"    提问：{q}")
        result = await graph.ainvoke({"messages": [("user", q)]}, config)

        router = result.get("router") or {}
        docs = result.get("documents") or []
        msgs = result.get("messages") or []
        final = str(msgs[-1].content) if msgs else ""

        print(f"    路由 type     = {router.get('type')!r}")
        print(f"    命中文档      = {len(docs)}")
        print(f"    剩余研究步骤  = {len(result.get('steps') or [])}")
        print()
        print("    回答节选：")
        print("    " + final[:400].replace("\n", "\n    "))
        print()

        if router.get("type") != "langchain":
            failures.append(f"路由结果异常: {router.get('type')}")
        if not docs:
            failures.append("未检索到任何文档")
        if not final:
            failures.append("未生成回答")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"端到端: {exc}")
        print(f"    ✗ {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()

    # ---------- 6) indexer 图 ----------
    print()
    print("【6】indexer 图（新栈下重新灌一次索引）")
    try:
        from index_graph.graph import graph as index_graph

        res = await index_graph.ainvoke(
            {"docs": []},
            RunnableConfig(
                configurable={
                    "embedding_model": "dashscope/text-embedding-v4",
                    "retriever_provider": "elastic-local",
                }
            ),
        )
        print(f"    ✓ indexer 执行完成，返回 docs = {res.get('docs')!r}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"indexer: {exc}")
        print(f"    ✗ {type(exc).__name__}: {exc}")

    # ---------- 汇总 ----------
    print()
    print("=" * 72)
    if failures:
        print(f"结论：✗ 升级后的栈存在 {len(failures)} 个问题：")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("结论：✓ 升级后的技术栈可以完整跑通本项目，未发现代码不兼容。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
