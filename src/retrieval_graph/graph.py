"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing & routing user queries, generating research plans to answer user questions,
conducting research, and formulating responses.
"""

import re
from typing import Any, Literal, Optional, TypedDict, cast

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from retrieval_graph.configuration import AgentConfiguration
from retrieval_graph.researcher_graph.graph import graph as researcher_graph
from retrieval_graph.state import AgentState, InputState, Router
from shared.utils import format_docs, load_chat_model

# How much conversation the researcher subgraph gets, to resolve pronouns in a
# follow-up question. 6 messages = 3 question/answer rounds.
#
# Why truncate at all: `state.messages` is append-only and grows with the session, so
# passing it whole would (a) make token cost grow linearly with conversation length on
# every research step, and (b) bury the rewrite instruction under earlier answers. A
# pronoun almost always points at the immediately preceding turn or two, so older
# content adds cost without adding signal. The per-message cap exists because a single
# long previous answer would otherwise dominate the prompt.
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CHARS_PER_MESSAGE = 800

# Evidence floor for the answer step. Below either threshold the context is treated as
# "probably too thin", and the response prompt gets an explicit insufficiency warning
# instead of letting the model improvise. Both numbers are heuristics tied to the
# default k=4 retrieval; they should be recalibrated once the eval harness lands.
LOW_EVIDENCE_DOC_COUNT = 3
LOW_EVIDENCE_CHAR_COUNT = 1200

# Answer used when retrieval came back completely empty. Written by hand (no model call)
# so the agent cannot "helpfully" invent an answer from thin air.
NO_EVIDENCE_REPLY = (
    "抱歉，我在现有知识库里没有检索到和您这个问题相关的资料，所以这次没法给出有依据的回答。"
    "与其给您一个听起来合理、实际却可能是编的结论，我更想先如实说明：这一点我确实查不到。"
    "如果方便，可以再补充一下您用的框架版本、涉及的类名或 API，或者完整的报错信息，我再帮您查一次。"
)


def _recent_history(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Return a small, bounded window of the conversation for the researcher subgraph.

    Keeps only dialog turns (drops system/tool messages), the last
    `MAX_HISTORY_MESSAGES` of them, each shortened to
    `MAX_HISTORY_CHARS_PER_MESSAGE` characters. See the constants above for why this
    bound exists.
    """
    dialog = [message for message in messages if message.type in ("human", "ai")]
    trimmed: list[AnyMessage] = []
    for message in dialog[-MAX_HISTORY_MESSAGES:]:
        content = message.content
        text = content if isinstance(content, str) else str(content)
        text = " ".join(text.split())
        if len(text) > MAX_HISTORY_CHARS_PER_MESSAGE:
            text = text[:MAX_HISTORY_CHARS_PER_MESSAGE] + "…（已截断）"
        # Rebuild rather than copy: pydantic's copy API differs between langchain-core
        # versions, and the subgraph only ever reads role + text from these messages.
        trimmed.append(type(message)(content=text))
    return trimmed


def _evidence_is_thin(documents: list[Document]) -> bool:
    """Whether the retrieved context looks too thin to answer from with confidence."""
    total_chars = sum(len(document.page_content or "") for document in documents)
    return len(documents) < LOW_EVIDENCE_DOC_COUNT or total_chars < LOW_EVIDENCE_CHAR_COUNT


# 答案里的引用标记形如 [1]、[2]。只认阿拉伯数字：提示词要求的就是这种写法。
CITATION_PATTERN = re.compile(r"\[(\d+)\]")

# 回链里 snippet 的长度。够用户在点进去之前判断"是不是这段"，又不至于把整个
# chunk 塞进接口响应里。
CITATION_SNIPPET_CHARS = 80


def _document_locator(document: Document) -> tuple[Optional[str], Optional[int]]:
    """从文档 metadata 里取出 (source_uuid, chunk_index)。

    分块入库的文档带 `source_uuid` + `chunk_index`；整篇入库（未分块）的老文档
    只有 `uuid`，此时把它当作文档身份返回，`chunk_index` 为 None——调用方据此
    就能区分"定位到某一块"和"只能定位到整篇"。

    Args:
        document: 命中的文档。

    Returns:
        tuple[Optional[str], Optional[int]]: 文档源 uuid 与块序号。
    """
    metadata = document.metadata or {}
    # 注意 `uuid` 只做兜底：分块后 chunk 自己的 uuid 是按下游规则重算的，
    # 真正稳定的文档身份在 `source_uuid` 上。
    source_uuid = metadata.get("source_uuid") or metadata.get("uuid")
    chunk_index = metadata.get("chunk_index")
    return (str(source_uuid) if source_uuid else None, int(chunk_index) if isinstance(chunk_index, int) else None)


def extract_citations(
    answer: str, documents: Optional[list[Document]]
) -> list[dict[str, Any]]:
    """把答案里的 `[n]` 解析成对具体文档（及其 chunk）的回链。

    序号按答案里**第一次出现**的顺序去重，`[n]` 对应 `documents[n-1]`
    （与提示词约定的 1-based 编号一致）。

    模型偶尔会编出一个不存在的序号（例如上下文只有 4 段却写了 `[9]`）。
    这种情况不会被静默忽略：返回项里 `resolved` 为 False，并附 `reason`
    说明越界原因，前端可以据此把这些标记标红。

    Args:
        answer: 模型生成的答案文本。
        documents: 本轮喂给模型的上下文文档，顺序即 `[1]`、`[2]`… 的顺序。

    Returns:
        list[dict[str, Any]]: 每个引用标记一项，形如
        ``{"marker": "[1]", "position": 1, "resolved": True,
        "source_uuid": "...", "chunk_index": 2, "snippet": "..."}``；
        越界时 ``resolved`` 为 False，`source_uuid`/`chunk_index`/`snippet`
        为 None，并带 ``reason``。

    Examples:
        >>> docs = [Document(page_content="甲"), Document(page_content="乙")]
        >>> extract_citations("先看甲[1]，再看乙[2]，以及不存在的[9]。", docs)[-1]
        {'marker': '[9]', 'position': 9, 'resolved': False, 'source_uuid': None,
         'chunk_index': None, 'snippet': None, 'reason': '...'}
    """
    context = list(documents or [])
    citations: list[dict[str, Any]] = []
    seen: set[str] = set()

    for match in CITATION_PATTERN.finditer(answer or ""):
        marker = match.group(0)
        if marker in seen:
            continue
        seen.add(marker)

        position = int(match.group(1))
        if 1 <= position <= len(context):
            document = context[position - 1]
            source_uuid, chunk_index = _document_locator(document)
            snippet = " ".join((document.page_content or "").split())
            if len(snippet) > CITATION_SNIPPET_CHARS:
                snippet = snippet[:CITATION_SNIPPET_CHARS] + "…"
            citations.append(
                {
                    "marker": marker,
                    "position": position,
                    "resolved": True,
                    "source_uuid": source_uuid,
                    "chunk_index": chunk_index,
                    "snippet": snippet,
                }
            )
        else:
            citations.append(
                {
                    "marker": marker,
                    "position": position,
                    "resolved": False,
                    "source_uuid": None,
                    "chunk_index": None,
                    "snippet": None,
                    # 明确说出"上下文只有几段"，便于定位是模型幻觉还是上下文丢失。
                    "reason": (
                        f"引用序号 {position} 越界：本轮上下文只有 {len(context)} 段资料。"
                        "该标记不对应任何真实文档，答案中的这部分内容缺少可追溯来源。"
                    ),
                }
            )

    return citations


def _answer_text(message: BaseMessage) -> str:
    """取出消息的纯文本（兼容 content 是分段列表的情况）。"""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") if isinstance(part, dict) else str(part)
        for part in content
    )


async def analyze_and_route_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Router]:
    """Analyze the user's query and determine the appropriate routing.

    This function uses a language model to classify the user's query and decide how to route it
    within the conversation flow.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type and logic).
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    messages = [
        {"role": "system", "content": configuration.router_system_prompt}
    ] + state.messages
    response = cast(
        Router, await model.with_structured_output(Router).ainvoke(messages)
    )
    return {"router": response}


def route_query(
    state: AgentState,
) -> Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]:
    """Determine the next step based on the query classification.

    Args:
        state (AgentState): The current state of the agent, including the router's classification.

    Returns:
        Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]: The next step to take.

    Raises:
        ValueError: If an unknown router type is encountered.
    """
    _type = state.router["type"]
    if _type == "langchain":
        return "create_research_plan"
    elif _type == "more-info":
        return "ask_for_more_info"
    elif _type == "general":
        return "respond_to_general_query"
    else:
        raise ValueError(f"Unknown router type {_type}")


async def ask_for_more_info(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response asking the user for more information.

    This node is called when the router determines that more information is needed from the user.

    Args:
        state (AgentState): The current state of the agent, including conversation history and router logic.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    system_prompt = configuration.more_info_system_prompt.format(
        logic=state.router["logic"]
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def respond_to_general_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response to a general query not related to LangChain.

    This node is called when the router classifies the query as a general question.

    Args:
        state (AgentState): The current state of the agent, including conversation history and router logic.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    system_prompt = configuration.general_system_prompt.format(
        logic=state.router["logic"]
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[str] | str]:
    """Create a step-by-step research plan for answering a LangChain-related query.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used to generate the plan.

    Returns:
        dict[str, list[str]]: A dictionary with a 'steps' key containing the list of research steps.
    """

    class Plan(TypedDict):
        """Generate research plan."""

        steps: list[str]

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model).with_structured_output(Plan)
    messages = [
        {"role": "system", "content": configuration.research_plan_system_prompt}
    ] + state.messages
    response = cast(Plan, await model.ainvoke(messages))
    return {"steps": response["steps"], "documents": "delete"}


async def conduct_research(state: AgentState) -> dict[str, Any]:
    """Execute the first step of the research plan.

    This function takes the first step from the research plan and uses it to conduct research.

    Args:
        state (AgentState): The current state of the agent, including the research plan steps.

    Returns:
        dict[str, list[str]]: A dictionary with 'documents' containing the research results and
                              'steps' containing the remaining research steps.

    Behavior:
        - Invokes the researcher_graph with the first step of the research plan.
        - Hands the subgraph a bounded window of the conversation (`_recent_history`) so
          the subgraph can turn a pronoun-laden step into a self-contained query.
        - Updates the state with the retrieved documents and removes the completed step.
    """
    history = _recent_history(state.messages)
    result = await researcher_graph.ainvoke(
        {"question": state.steps[0], "history": history}
    )
    return {"documents": result["documents"], "steps": state.steps[1:]}


def check_finished(state: AgentState) -> Literal["respond", "conduct_research"]:
    """Determine if the research process is complete or if more research is needed.

    This function checks if there are any remaining steps in the research plan:
        - If there are, route back to the `conduct_research` node
        - Otherwise, route to the `respond` node

    Args:
        state (AgentState): The current state of the agent, including the remaining research steps.

    Returns:
        Literal["respond", "conduct_research"]: The next step to take based on whether research is complete.
    """
    if len(state.steps or []) > 0:
        return "conduct_research"
    else:
        return "respond"


async def respond(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Any]:
    """Generate a final response to the user's query based on the conducted research.

    This function formulates a comprehensive answer using the conversation history and the documents retrieved by the researcher.

    Args:
        state (AgentState): The current state of the agent, including retrieved documents and conversation history.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, Any]: A dictionary with a 'messages' key containing the generated
        response, and a 'citations' key mapping every `[n]` in that response back to
        the source document / chunk it refers to.
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    documents = state.documents or []

    # No evidence at all -> answer without the model. Asking a model to answer from an
    # empty context is exactly the "confidently wrong" failure mode we want gone.
    if not documents:
        return {"messages": [AIMessage(content=NO_EVIDENCE_REPLY)], "citations": []}

    model = load_chat_model(configuration.response_model)
    context = format_docs(documents)
    prompt = configuration.response_system_prompt.format(context=context)
    if _evidence_is_thin(documents):
        # Prepended, not appended: the tail of a long prompt is the first thing a model
        # stops honouring, and this is the instruction we least want it to drop.
        total_chars = sum(len(document.page_content or "") for document in documents)
        prompt = (
            "【本轮证据不足告警】本次检索只拿到 "
            f"{len(documents)} 段资料、合计约 {total_chars} 字，很可能不足以回答问题。"
            "如果你无法仅凭这些资料回答用户，必须直接说明现有资料不足、无法确认，"
            "并指出缺什么信息；严禁用你自己的先验知识补足事实、结论或数字。\n\n"
        ) + prompt
    messages = [{"role": "system", "content": prompt}] + state.messages
    response = await model.ainvoke(messages)
    # 引用回链：把答案里的 [n] 映射回具体文档 / chunk，作为节点输出的一部分带给前端。
    citations = extract_citations(_answer_text(response), documents)
    return {"messages": [response], "citations": citations}


# Define the graph
builder = StateGraph(AgentState, input=InputState, config_schema=AgentConfiguration)
builder.add_node(analyze_and_route_query)
builder.add_node(ask_for_more_info)
builder.add_node(respond_to_general_query)
builder.add_node(conduct_research)
builder.add_node(create_research_plan)
builder.add_node(respond)

builder.add_edge(START, "analyze_and_route_query")
builder.add_conditional_edges("analyze_and_route_query", route_query)
builder.add_edge("create_research_plan", "conduct_research")
builder.add_conditional_edges("conduct_research", check_finished)
builder.add_edge("ask_for_more_info", END)
builder.add_edge("respond_to_general_query", END)
builder.add_edge("respond", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "RetrievalGraph"
