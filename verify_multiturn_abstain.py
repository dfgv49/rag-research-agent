"""临时验证脚本：多轮指代改写（任务 A）与证据不足拒答（任务 B）。

不依赖 Elasticsearch / Docker：用 src/sample_docs.json 的 32 篇样例文档建内存向量库，
只把 shared.retrieval.make_retriever 在运行时替换成内存检索器（shard/retrieval.py 本身不改）。

覆盖：
  1. 改写生效：3 组「历史 + 带代词追问」直接调用 rewrite_query 节点，打印前后对比；
     1b. 子图端到端：同一句追问，带历史 vs 不带历史，生成的检索词差异。
  2. 拒答生效：知识库里没有答案的问题跑完整 retrieval_graph；
     2b. 资料稀薄时提示词里的显式约束（真实 LLM 回复 + 实际 system prompt 头部）；
     2c. 有资料时的正常作答（不回归）。
  3. 不回归：模块导入自检 + 直接调用 tests/unit_tests/test_configuration.py 里的用例
     （.venv 里没有 pytest，用标准库方式执行）。

用法:
    .\\.venv\\Scripts\\python.exe verify_multiturn_abstain.py
"""

import asyncio
import importlib
import json
import os
import sys
import warnings
from contextlib import contextmanager

warnings.filterwarnings("ignore")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402
from langchain_core.vectorstores import InMemoryVectorStore  # noqa: E402

import shared.retrieval as shared_retrieval  # noqa: E402
from retrieval_graph import prompts  # noqa: E402,F401
from retrieval_graph.configuration import AgentConfiguration  # noqa: E402
from retrieval_graph.researcher_graph.graph import graph as researcher_graph  # noqa: E402
from retrieval_graph.researcher_graph.graph import rewrite_query  # noqa: E402
from retrieval_graph.researcher_graph.state import ResearcherState  # noqa: E402
from retrieval_graph.state import AgentState  # noqa: E402
from shared.state import reduce_docs  # noqa: E402

# 注意：retrieval_graph/__init__.py 的 `from retrieval_graph.graph import graph` 把子模块属性
# `graph` 覆盖成了编译后的图对象，所以这里必须从 sys.modules 拿模块本身。
graph_module = importlib.import_module("retrieval_graph.graph")

# 该账号实测：qwen-turbo → 429，qwen-max → 403 free quota exhausted。
# 所以这里显式覆盖成可用模型（与 configuration.py 的新默认值一致）。
MODEL_CONFIG = RunnableConfig(
    configurable={
        "query_model": "dashscope/qwen-flash",
        "response_model": "dashscope/qwen-plus",
        "embedding_model": "dashscope/text-embedding-v4",
    }
)

# 检索打分阈值：把「不相关」的资料挡在门外，让「知识库里没有答案」这件事真实发生。
# 观测值（text-embedding-v4 + 32 篇样例文档，cosine 相似度）：
#   域内问题 0.58~0.69；「LangGraph 的 CEO 是谁？」最高 0.55；「今天天气怎么样」最高 0.20。
SCORE_THRESHOLD = 0.57

RESULTS: list[tuple[str, bool, str]] = []


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"\n[{'PASS' if ok else 'FAIL'}] {name}" + (f" —— {detail}" if detail else ""))


async def with_retry(label: str, factory, attempts: int = 4):
    """429/403/网络抖动时的降级重试，避免一次限流就让整份证据缺失。"""
    delay = 3
    for attempt in range(1, attempts + 1):
        try:
            return await factory()
        except (AttributeError, TypeError, NameError, KeyError):
            # 编程错误重试没有意义，直接抛出
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"  [retry {attempt}/{attempts}] {label} 失败：{type(exc).__name__}: {str(exc)[:150]}")
            if attempt == attempts:
                raise
            await asyncio.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# 内存向量库 + 运行时替换检索器（src/shared/retrieval.py 一行都不改）
# ---------------------------------------------------------------------------
def build_store() -> InMemoryVectorStore:
    with open(os.path.join(ROOT, "src", "sample_docs.json"), encoding="utf-8") as handle:
        docs = reduce_docs([], json.load(handle))
    encoder = shared_retrieval.make_text_encoder(
        AgentConfiguration.from_runnable_config({}).embedding_model
    )
    store = InMemoryVectorStore(embedding=encoder)
    store.add_documents(docs)
    print(f"内存向量库已建立：{len(docs)} 篇样例文档")
    return store


STORE = build_store()


class InMemoryRetriever:
    """最小检索器：ainvoke(query) -> list[Document]，可选相似度阈值过滤。"""

    def __init__(self, store: InMemoryVectorStore, k: int = 4, score_threshold: float | None = None):
        self.store = store
        self.k = k
        self.score_threshold = score_threshold
        self.last_scores: list[float] = []

    async def ainvoke(self, query, config=None):  # noqa: ANN001, ARG002
        pairs = self.store.similarity_search_with_score(query, k=self.k)
        self.last_scores = [round(float(score), 4) for _, score in pairs]
        return [
            doc
            for doc, score in pairs
            if self.score_threshold is None or float(score) >= self.score_threshold
        ]


@contextmanager
def patched_retriever(score_threshold: float | None = None, k: int = 4):
    retriever = InMemoryRetriever(STORE, k=k, score_threshold=score_threshold)

    @contextmanager
    def _fake_make_retriever(config):  # noqa: ARG001, ANN001
        yield retriever

    original = shared_retrieval.make_retriever
    shared_retrieval.make_retriever = _fake_make_retriever
    try:
        yield retriever
    finally:
        shared_retrieval.make_retriever = original


# ---------------------------------------------------------------------------
# 1. 改写生效
# ---------------------------------------------------------------------------
REWRITE_CASES = [
    {
        "label": "案例 1：历史聊「节点」，追问「那它和边有什么区别？」",
        "history": [
            HumanMessage("LangGraph 里的节点是怎么定义的？"),
            AIMessage(
                "在 LangGraph 里，节点一般就是一个 Python 函数（同步或异步），第一个位置参数是 state，"
                "第二个可选参数是 config。节点通过 add_node 注册到 StateGraph 上。"
            ),
        ],
        "question": "那它和边有什么区别？",
        "must_contain": ["节点", "边"],
    },
    {
        "label": "案例 2：历史聊「并行执行」，追问「那这个具体怎么写？」",
        "history": [
            HumanMessage("怎么把 LangGraph 里的多个节点并行执行？"),
            AIMessage(
                "可以在一个节点里返回多个 Send 对象来动态并行分发任务，也可以用 add_edge 声明"
                "多个出边让下游节点并行执行。"
            ),
        ],
        "question": "那这个具体怎么写？",
        "must_contain": ["Send"],
    },
    {
        "label": "案例 3：历史聊「条件边」，追问「那它什么时候触发？」",
        "history": [
            HumanMessage("LangGraph 的条件边是什么？"),
            AIMessage(
                "条件边用 add_conditional_edges 声明：它接收一个路由函数，路由函数读当前 state，"
                "返回下一个要执行的节点名，图就按返回值跳转。"
            ),
        ],
        "question": "那它什么时候触发？",
        "must_contain": ["条件边"],
    },
]


async def test_rewrite() -> None:
    section("1. 改写生效：直接调用 rewrite_query 节点（带历史）")
    all_ok = True
    for case in REWRITE_CASES:
        state = ResearcherState(question=case["question"], history=case["history"])
        out = await with_retry(
            f"rewrite_query({case['question']})",
            lambda s=state: rewrite_query(s, config=MODEL_CONFIG),
        )
        rewritten = out["rewritten_question"]
        print(f"\n{case['label']}")
        print(f"  原始问题 ：{case['question']}")
        print(f"  改写结果 ：{rewritten}")
        missing = [token for token in case["must_contain"] if token not in rewritten]
        ok = not missing
        all_ok = all_ok and ok
        if missing:
            print(f"  ✗ 缺少自包含信息：{missing}")
        await asyncio.sleep(1)
    record("1. 三个带代词的追问都被改写成自包含查询", all_ok)

    section("1. 边界：没有历史时不改写（跳过 LLM 调用）")
    out = await rewrite_query(
        ResearcherState(question="LangGraph 的节点怎么定义？", history=[]),
        config=MODEL_CONFIG,
    )
    print(f"  history=[] 时 rewritten_question = {out['rewritten_question']!r}")
    record(
        "1. 无历史 → 原样透传，省一次 LLM 调用",
        out["rewritten_question"] == "LangGraph 的节点怎么定义？",
    )


async def test_rewrite_end_to_end() -> None:
    section("1b. 子图端到端：同一句追问，带历史 vs 不带历史")
    question = "那它和边有什么区别？"
    history = REWRITE_CASES[0]["history"]

    with patched_retriever(score_threshold=None, k=4):
        without = await with_retry(
            "researcher(no history)",
            lambda: researcher_graph.ainvoke({"question": question}, MODEL_CONFIG),
        )
        with_history = await with_retry(
            "researcher(with history)",
            lambda: researcher_graph.ainvoke(
                {"question": question, "history": history}, MODEL_CONFIG
            ),
        )

    print(f"\n  追问：{question}")
    print(f"\n  [A] 不带历史（修复前行为）")
    print(f"      rewritten_question : {without.get('rewritten_question')!r}")
    for i, q in enumerate(without["queries"], 1):
        print(f"      检索词 {i}: {q}")
    print(f"      命中文档数: {len(without['documents'])}")
    print(f"\n  [B] 带历史（修复后行为）")
    print(f"      rewritten_question : {with_history.get('rewritten_question')!r}")
    for i, q in enumerate(with_history["queries"], 1):
        print(f"      检索词 {i}: {q}")
    print(f"      命中文档数: {len(with_history['documents'])}")

    rewritten = with_history.get("rewritten_question", "")
    ok = ("节点" in rewritten and "边" in rewritten)
    record(
        "1b. 子图在检索前先消解指代，改写结果进入 generate_queries",
        ok,
        f"带历史命中文档 {len(with_history['documents'])} 篇 / 不带历史 {len(without['documents'])} 篇",
    )


# ---------------------------------------------------------------------------
# 2. 拒答生效
# ---------------------------------------------------------------------------
UNKNOWN_QUESTION = "LangGraph 的 CEO 是谁？"


async def run_full_graph(question: str, score_threshold: float | None):
    with patched_retriever(score_threshold=score_threshold, k=4) as retriever:
        result = await with_retry(
            f"main_graph({question})",
            lambda: graph_module.graph.ainvoke(
                {"messages": [("user", question)]}, MODEL_CONFIG
            ),
        )
        return result, list(retriever.last_scores)


def print_answer(text: str) -> None:
    print("\n  ---")
    for line in str(text).splitlines():
        print(f"  {line}")
    print("  ---")


async def test_abstain_unknown_question() -> None:
    section("2a. 拒答生效：知识库里没有答案的问题，跑完整 retrieval_graph")
    result, scores = await run_full_graph(UNKNOWN_QUESTION, SCORE_THRESHOLD)
    answer = str(result["messages"][-1].content)
    docs = result.get("documents") or []

    print(f"\n  用户提问 : {UNKNOWN_QUESTION}")
    print(f"  路由结果 : {result['router']['type']}")
    print(f"  剩余计划 : {result.get('steps')}")
    print(f"  最后一次检索打分 : {scores}（阈值 {SCORE_THRESHOLD}）")
    print(f"  累计证据 : {len(docs)} 篇")
    print("\n  最终回复原文：")
    print_answer(answer)

    admitted = any(token in answer for token in ("不足", "无法确认", "没有", "不知道"))
    # 只把「真的断言了某个身份」当作编造；「CEO 是谁」这种提问式复述不算。
    fabricated = any(
        token in answer for token in ("担任 CEO", "CEO 叫", "CEO 为", "首席执行官叫", "创始人是")
    )
    record(
        "2a. 明确表示资料不足，没有编造「CEO 是谁」",
        admitted and not fabricated,
        f"证据 {len(docs)} 篇，走的是稀薄证据约束分支",
    )


async def test_abstain_no_evidence_branch() -> None:
    section("2b. 检索完全为空 → 走确定性拒答分支，不再调 LLM 硬答")
    # 用一个域内问题 + 极高阈值，让检索真实返回空列表，触发 respond 的空证据分支
    result, scores = await run_full_graph("LangGraph 里节点是怎么定义的？", 0.99)
    answer = str(result["messages"][-1].content)
    docs = result.get("documents") or []

    print(f"\n  用户提问 : LangGraph 里节点是怎么定义的？（域内问题，但阈值 0.99 全部过滤）")
    print(f"  路由结果 : {result['router']['type']}")
    print(f"  检索打分 : {scores}（阈值 0.99）")
    print(f"  检索文档 : {len(docs)} 篇")
    print("\n  最终回复原文：")
    print_answer(answer)

    record(
        "2b. 空证据 → 回复与 NO_EVIDENCE_REPLY 逐字一致（respond 未进入 LLM 分支）",
        len(docs) == 0 and answer == graph_module.NO_EVIDENCE_REPLY,
    )


async def test_abstain_thin_evidence() -> None:
    section("2c. 资料稀薄时：提示词前置告警 + 真实 LLM 回复")
    thin_docs = [
        Document(
            page_content="LangGraph 是一个用 Python 和 JavaScript 构建有状态、多参与者应用的库。",
            metadata={"uuid": "thin-1"},
        )
    ]
    state = AgentState(messages=[HumanMessage(UNKNOWN_QUESTION)], documents=thin_docs)

    captured: list[list] = []
    original_loader = graph_module.load_chat_model

    def recording_loader(name):  # noqa: ANN001, ANN202
        inner = original_loader(name)

        class _Recorder:
            async def ainvoke(self, messages, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
                captured.append(messages)
                return await inner.ainvoke(messages, *args, **kwargs)

        return _Recorder()

    graph_module.load_chat_model = recording_loader
    try:
        out = await with_retry(
            "respond(thin evidence)", lambda: graph_module.respond(state, config=MODEL_CONFIG)
        )
    finally:
        graph_module.load_chat_model = original_loader

    # respond 里的 messages 是 dict 风格（{"role": ...}），不是 BaseMessage
    first = captured[0][0]
    system_prompt = first["content"] if isinstance(first, dict) else first.content
    print(f"\n  证据：{len(thin_docs)} 篇 / {sum(len(d.page_content) for d in thin_docs)} 字"
          f"（阈值 {graph_module.LOW_EVIDENCE_DOC_COUNT} 篇 / {graph_module.LOW_EVIDENCE_CHAR_COUNT} 字）")
    print("  实际发出的 system prompt 头部：")
    print_answer(system_prompt[:200])
    answer = str(out["messages"][0].content)
    print("\n  LLM 回复原文：")
    print_answer(answer)

    warned = system_prompt.startswith("【本轮证据不足告警】")
    admitted = any(token in answer for token in ("不足", "无法确认", "不知道", "没有", "无法回答"))
    record(
        "2c. 稀薄证据 → 提示词前置告警，模型明确表示资料不足",
        warned and admitted,
    )


async def test_grounded_answer() -> None:
    section("2d. 不回归：有资料时照常带引用作答")
    question = "LangGraph 里节点是怎么定义的？"
    with patched_retriever(score_threshold=None, k=4) as retriever:
        result = await with_retry(
            "main_graph(grounded question)",
            lambda: graph_module.graph.ainvoke(
                {"messages": [("user", question)]}, MODEL_CONFIG
            ),
        )
        scores = list(retriever.last_scores)

    answer = str(result["messages"][-1].content)
    docs = result.get("documents") or []
    print(f"\n  提问：{question}")
    print(f"  路由：{result['router']['type']}  文档：{len(docs)} 篇  打分：{scores}")
    print("\n  回复原文（前 600 字）：")
    print_answer(answer[:600])
    record(
        "2d. 有资料时正常作答且带 [n] 引用",
        len(docs) > 0 and "[" in answer and answer != graph_module.NO_EVIDENCE_REPLY,
    )


# ---------------------------------------------------------------------------
# 3. 不回归：导入自检 + 单元测试
# ---------------------------------------------------------------------------
def test_no_regression() -> None:
    section("3. 不回归：模块导入自检 + tests/unit_tests 用例（pytest 未安装）")

    modules = [
        "shared.configuration",
        "shared.state",
        "shared.utils",
        "shared.retrieval",
        "retrieval_graph.prompts",
        "retrieval_graph.configuration",
        "retrieval_graph.state",
        "retrieval_graph.graph",
        "retrieval_graph.researcher_graph.state",
        "retrieval_graph.researcher_graph.graph",
    ]
    for name in modules:
        importlib.import_module(name)
        print(f"  import ok: {name}")

    # 每个配置字段都必须有 default，否则 Studio / from_runnable_config 会直接报错
    from dataclasses import MISSING, fields

    missing_defaults = [
        f.name
        for f in fields(AgentConfiguration)
        if f.default is MISSING and f.default_factory is MISSING  # type: ignore[misc]
    ]
    print(f"  配置字段缺 default 的：{missing_defaults}")
    record("3a. 全部配置字段都有 default", not missing_defaults, str(missing_defaults))

    cfg = AgentConfiguration.from_runnable_config({})
    print(f"  query_model={cfg.query_model}  response_model={cfg.response_model}")
    print(f"  query_rewrite_system_prompt 长度={len(cfg.query_rewrite_system_prompt)}")

    # 提示词里的字面大括号必须转义，否则 .format() 会抛 KeyError/IndexError
    formatted = cfg.response_system_prompt.format(context="<doc>测试</doc>")
    print(f"  response_system_prompt.format(context=...) 成功，长度={len(formatted)}")
    assert "{$" not in formatted

    from tests.unit_tests.test_configuration import test_configuration_empty

    test_configuration_empty()
    print("  tests/unit_tests/test_configuration.py::test_configuration_empty 通过")
    record("3b. 导入自检 + 单元测试用例全部通过", True)


# ---------------------------------------------------------------------------
async def main() -> None:
    await test_rewrite()
    await test_rewrite_end_to_end()
    await test_abstain_unknown_question()
    await test_abstain_no_evidence_branch()
    await test_abstain_thin_evidence()
    await test_grounded_answer()
    test_no_regression()

    section("汇总")
    failed = 0
    for name, ok, detail in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" —— {detail}" if detail else ""))
        failed += 0 if ok else 1
    print(f"\n  共 {len(RESULTS)} 项，失败 {failed} 项")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
