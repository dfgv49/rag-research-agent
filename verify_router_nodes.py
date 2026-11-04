"""单独调用主图的四个「路由段」节点，观察它们的真实行为。

不依赖 Elasticsearch —— 这四个节点只调模型，不查向量库。

验证内容：
  1. analyze_and_route_query：三种提问各自被分类成什么？
  2. route_query：分类结果如何映射成「下一站」？遇到非法值会怎样？
  3. ask_for_more_info：追问时，模型能否说出「为什么需要更多信息」？
  4. respond_to_general_query：拒答时是否礼貌、是否引导用户澄清？
  5. 两个终点分支的实现是不是几乎一样？

用法:
    .venv\\Scripts\\python.exe verify_router_nodes.py
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

from dataclasses import dataclass, field  # noqa: E402

from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402

from retrieval_graph.graph import (  # noqa: E402
    analyze_and_route_query,
    ask_for_more_info,
    respond_to_general_query,
    route_query,
)


@dataclass
class FakeState:
    """手搓一个 state，只带这两个节点需要的字段。"""

    messages: list = field(default_factory=list)
    router: dict = field(default_factory=lambda: {"type": "general", "logic": ""})


CONFIG = RunnableConfig(
    configurable={
        "query_model": "dashscope/qwen-flash",
        "response_model": "dashscope/qwen-plus",
    }
)


async def main() -> None:
    print("=" * 74)
    print("路由段四个节点的单独调用实测")
    print("=" * 74)

    cases = [
        "LangGraph 里怎么定义节点？",
        "我的代码报错了",
        "今天天气怎么样",
    ]

    print()
    print("【1】analyze_and_route_query —— 分类 + 给出理由")
    print("-" * 74)
    for q in cases:
        state = FakeState(messages=[HumanMessage(content=q)])
        out = await analyze_and_route_query(state, config=CONFIG)
        router = out["router"]
        print(f"  提问：{q}")
        print(f"    -> type  = {router['type']!r}")
        print(f"    -> logic = {router['logic']!r}")
        print()

        print("    route_query 映射结果：", end=" ")
        state.router = router
        print(f"{route_query(state)!r}")
        print()

    print("【2】route_query —— 遇到非法 type 会怎样")
    print("-" * 74)
    bad = FakeState(messages=[])
    bad.router = {"type": "天气类问题", "logic": "模型自由发挥的结果"}
    try:
        route_query(bad)
        print("  没报错（不符合预期）")
    except Exception as e:
        print(f"  抛出 {type(e).__name__}: {e}")
    print("  -> 这就是为什么 Router.type 必须用 Literal 锁死取值")

    print()
    print("【3】ask_for_more_info —— 追问分支")
    print("-" * 74)
    state = FakeState(messages=[HumanMessage(content="我的代码报错了")])
    state.router = {"type": "more-info", "logic": "用户报告出错但没有提供错误信息"}
    out = await ask_for_more_info(state, config=CONFIG)
    print("  模型输出：")
    print("   ", str(out["messages"][0].content).replace("\n", "\n    "))

    print()
    print("【4】respond_to_general_query —— 拒答分支")
    print("-" * 74)
    state = FakeState(messages=[HumanMessage(content="今天天气怎么样")])
    state.router = {"type": "general", "logic": "用户询问天气，与 LangChain 无关"}
    out = await respond_to_general_query(state, config=CONFIG)
    print("  模型输出：")
    print("   ", str(out["messages"][0].content).replace("\n", "\n    "))

    print()
    print("=" * 74)
    print("小结")
    print("=" * 74)
    print("  - 四个节点全部只依赖模型，不依赖向量库")
    print("  - 追问与拒答两个分支结构相同，只差一段提示词")
    print("  - 它们的返回值都是 {'messages': [模型回复]}，追加进对话历史")


if __name__ == "__main__":
    asyncio.run(main())
