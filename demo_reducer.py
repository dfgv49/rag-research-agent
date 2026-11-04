"""验证 LangGraph 的 State / Reducer 机制，用数字和真实行为说明问题。

用法:
    .venv\\Scripts\\python.exe demo_reducer.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from typing import Annotated, TypedDict  # noqa: E402

from langgraph.graph import END, START, StateGraph  # noqa: E402

from shared.state import reduce_docs  # noqa: E402


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------- 实验一
def test_reduce_docs() -> None:
    section("实验一：项目里 documents 字段用的 reduce_docs 到底怎么合并")

    print("【1】传入字符串内容 —— 去重不生效（这是一个真实的坑）")
    first = reduce_docs([], ["文档A", "文档B"])
    print(f"  第 1 次写入: {len(first)} 篇")
    second = reduce_docs(first, ["文档A", "文档B"])
    print(f"  第 2 次写入同样内容: {len(second)} 篇  <-- 没有去重，变成 4 篇")
    print("  原因：字符串分支只做了『追加』，没有先检查 uuid 是否已存在。")

    print()
    print("【2】传入 dict / Document 形式 —— 去重正常生效")
    c = reduce_docs([], [{"page_content": "文档A"}, {"page_content": "文档B"}])
    print(f"  第 1 次写入: {len(c)} 篇")
    d = reduce_docs(c, [{"page_content": "文档A"}, {"page_content": "文档B"}])
    print(f"  第 2 次写入同样内容: {len(d)} 篇  <-- 正确去重，仍是 2 篇")
    print("  说明：dict 分支会先比对 uuid，重复的会被跳过。")

    print()
    print("【3】新内容追加而非覆盖")
    e = reduce_docs(c, [{"page_content": "文档C"}])
    print(f"  结果: {len(e)} 篇 -> {[x.page_content for x in e]}")

    print()
    print("【4】传入字符串 \"delete\" —— 清空指令")
    f = reduce_docs(e, "delete")
    print(f"  结果: {len(f)} 篇 -> {f}")
    print("  项目在 create_research_plan 里正是用这个机制清空上一轮的研究资料。")


# ---------------------------------------------------------------- 实验二
class DefaultState(TypedDict):
    """messages 用默认行为（直接覆盖）。"""

    messages: list[str]


class AppendState(TypedDict):
    """messages 使用 add 归约器（追加）。"""

    messages: Annotated[list[str], lambda left, right: (left or []) + (right or [])]


async def run_graph(state_cls, label: str) -> None:
    async def node_a(state):
        print(f"    node_a 收到: {state}")
        return {"messages": ["来自节点A"]}

    async def node_b(state):
        print(f"    node_b 收到: {state}")
        return {"messages": ["来自节点B"]}

    builder = StateGraph(state_cls)
    builder.add_node(node_a)
    builder.add_node(node_b)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", END)
    graph = builder.compile()

    result = await graph.ainvoke({"messages": ["用户提问"]})
    print(f"    -> {label} 最终 state: {result}")


async def test_reducer() -> None:
    section("实验二：同一个字段，加不加 Reducer 结果完全不同")
    print("两个节点都往 messages 写入，看看最终留下了什么。")
    print()
    print("【A】不加归约器时（新值覆盖旧值）")
    await run_graph(DefaultState, "覆盖")
    print()
    print("【B】加了归约器时（新值追加到旧值）")
    await run_graph(AppendState, "追加")
    print()
    print("结论：Reducer 决定『新数据如何写进旧数据』。")
    print("      项目的 messages 用的是 langgraph 内置的 add_messages，效果就是 B（追加）。")


# ---------------------------------------------------------------- 实验三
async def test_loop() -> None:
    section("实验三：循环边是怎么实现的（对照 check_finished）")

    class LoopState(TypedDict):
        steps: list[str]
        log: Annotated[list[str], lambda l, r: (l or []) + (r or [])]

    async def work(state: LoopState):
        current = state["steps"][0]
        print(f"    正在处理: {current}  (剩余 {len(state['steps'])} 步)")
        return {"steps": state["steps"][1:], "log": [f"完成 {current}"]}

    def more_left(state: LoopState):
        return "work" if state["steps"] else "done"

    async def done(state: LoopState):
        print(f"    全部完成，结果: {state['log']}")
        return {}

    builder = StateGraph(LoopState)
    builder.add_node("work", work)
    builder.add_node("done", done)
    builder.add_edge(START, "work")
    builder.add_conditional_edges("work", more_left, {"work": "work", "done": "done"})
    builder.add_edge("done", END)
    graph = builder.compile()

    print("初始 steps = ['第一步', '第二步', '第三步']")
    await graph.ainvoke({"steps": ["第一步", "第二步", "第三步"], "log": []})
    print()
    print("结论：work -> work 这条自循环边，就是项目里 conduct_research 的行为。")
    print("      每绕一圈就消化掉一个步骤，直到 steps 为空才跳出循环。")
    print("      这是 LangGraph 相对普通 DAG 编排引擎最本质的能力差异。")


async def main() -> None:
    test_reduce_docs()
    await test_reducer()
    await test_loop()
    print()
    print("=" * 70)
    print("全部演示完成。这些行为都来自真实的 LangGraph 运行时。")
    print("=" * 70)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
