"""验证配置在整条链路上的传递行为（实测，不猜）。

要回答的问题：
  1. 不传 config 时，节点拿到什么？
  2. configurable 里的值如何到达节点？
  3. 父图调用子图时（不显式传 config），子图能继承到配置吗？
  4. 节点里能不能看到 framework 注入的其他键（thread_id 等）？

用法:
    .venv\\Scripts\\python.exe verify_config_flow.py
"""

import asyncio
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from typing import TypedDict  # noqa: E402

from langchain_core.runnables import RunnableConfig  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

from shared.configuration import BaseConfiguration  # noqa: E402


class S(TypedDict):
    log: list


def _show(label: str, config) -> None:
    if config is None:
        print(f"    {label}: config is None")
        return
    cfg = config.get("configurable") or {}
    print(f"    {label}: configurable 的键 = {sorted(cfg.keys())}")
    print(f"        embedding_model = {cfg.get('embedding_model')!r}")


async def main() -> None:
    print("=" * 72)
    print("实验 A：普通单图 —— 不传 config vs 传 config")
    print("=" * 72)

    async def node(state: S, *, config: RunnableConfig):
        _show("节点内看到的 config", config)
        resolved = BaseConfiguration.from_runnable_config(config)
        print(f"        解析后 embedding_model = {resolved.embedding_model!r}")
        print(f"        解析后 retriever_provider = {resolved.retriever_provider!r}")
        return {"log": []}

    b = StateGraph(S)
    b.add_node("n", node)
    b.add_edge(START, "n")
    b.add_edge("n", END)
    g = b.compile()

    print("  [A1] 不传 config：")
    await g.ainvoke({"log": []})
    print()
    print("  [A2] 传了 config：")
    await g.ainvoke(
        {"log": []},
        RunnableConfig(configurable={"embedding_model": "dashscope/text-embedding-v4"}),
    )

    print()
    print("=" * 72)
    print("实验 B：父图调用子图（不显式传 config）—— 配置会继承吗？")
    print("=" * 72)

    async def child_node(state: S, *, config: RunnableConfig):
        _show("子图节点内看到的 config", config)
        return {"log": []}

    cb = StateGraph(S)
    cb.add_node("child", child_node)
    cb.add_edge(START, "child")
    cb.add_edge("child", END)
    child = cb.compile()

    async def parent_node(state: S, *, config: RunnableConfig):
        _show("父图节点内看到的 config", config)
        print("        -> 现在调用子图，注意【不传 config】")
        await child.ainvoke({"log": []})
        return {"log": []}

    pb = StateGraph(S)
    pb.add_node("parent", parent_node)
    pb.add_edge(START, "parent")
    pb.add_edge("parent", END)
    parent = pb.compile()

    print("  [B1] 父图带配置运行：")
    await parent.ainvoke(
        {"log": []},
        RunnableConfig(configurable={"embedding_model": "dashscope/被父图设置的模型"}),
    )

    print()
    print("=" * 72)
    print("实验 C：框架自己会往 configurable 里放什么？")
    print("=" * 72)

    async def spy_node(state: S, *, config: RunnableConfig):
        cfg = config.get("configurable") or {}
        print(f"    全部 configurable 键: {sorted(cfg.keys())}")
        for k in sorted(cfg.keys()):
            v = cfg[k]
            print(f"        {k:<16} = {str(v)[:60]!r}")
        return {"log": []}

    sb = StateGraph(S)
    sb.add_node("spy", spy_node)
    sb.add_edge(START, "spy")
    sb.add_edge("spy", END)
    spy = sb.compile()

    await spy.ainvoke(
        {"log": []},
        RunnableConfig(
            configurable={"embedding_model": "dashscope/x", "my_custom_key": "自定义值"}
        ),
    )


if __name__ == "__main__":
    asyncio.run(main())
