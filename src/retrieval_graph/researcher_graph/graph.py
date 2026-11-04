"""Researcher graph used in the conversational retrieval system as a subgraph.

This module defines the core structure and functionality of the researcher graph,
which is responsible for generating search queries and retrieving relevant documents.
"""

from typing import Any, TypedDict, cast

import asyncio

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from retrieval_graph.configuration import AgentConfiguration
from retrieval_graph.researcher_graph.state import (
    QueryState,
    ReplaceDocuments,
    ResearcherState,
)
from shared import hybrid, retrieval
from shared.utils import load_chat_model

# Defensive second truncation on the subgraph side: the main graph already trims the
# window, but the researcher can also be invoked directly (tools, scripts, tests) and
# must never push a whole session transcript into the rewrite prompt.
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CHARS_PER_MESSAGE = 600


def _message_text(message: AnyMessage) -> str:
    """Flatten one message into a single short line for the rewrite prompt."""
    content = message.content
    text = content if isinstance(content, str) else str(content)
    text = " ".join(text.split())
    if len(text) > MAX_HISTORY_CHARS_PER_MESSAGE:
        text = text[:MAX_HISTORY_CHARS_PER_MESSAGE] + "…（已截断）"
    return text


def _format_history(history: list[AnyMessage]) -> str:
    """Render a recent conversation window as plain text for the rewrite prompt."""
    lines = []
    for message in history[-MAX_HISTORY_MESSAGES:]:
        role = "用户" if message.type == "human" else "助手"
        lines.append(f"{role}：{_message_text(message)}")
    return "\n".join(lines)


async def rewrite_query(
    state: ResearcherState, *, config: RunnableConfig
) -> dict[str, str]:
    """Rewrite the research step into a self-contained query using the conversation.

    A follow-up like "那它和边有什么区别？" carries its meaning only through the previous
    turn. Sending it to the retriever verbatim matches nothing useful, so this node
    resolves pronouns with the recent history before any search query is generated.

    Args:
        state (ResearcherState): Current researcher state, including the step to research
            and the recent conversation window passed down by the main graph.
        config (RunnableConfig): Configuration with the model and the rewrite prompt.

    Returns:
        dict[str, str]: The `rewritten_question` key holding a self-contained query.
    """
    # No history means there is nothing to resolve: skip the extra model call entirely.
    if not state.history:
        return {"rewritten_question": state.question}

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    messages = [
        {"role": "system", "content": configuration.query_rewrite_system_prompt},
        {
            "role": "human",
            "content": (
                "<对话历史>\n"
                f"{_format_history(state.history)}\n"
                "</对话历史>\n\n"
                "<待改写的问题>\n"
                f"{state.question}\n"
                "</待改写的问题>"
            ),
        },
    ]
    response = await model.ainvoke(messages)
    content = response.content
    rewritten = (content if isinstance(content, str) else str(content)).strip()
    # Never let a failed rewrite turn into an empty search query.
    return {"rewritten_question": rewritten or state.question}


async def generate_queries(
    state: ResearcherState, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Generate search queries based on the (rewritten) research step.

    This function uses a language model to generate diverse search queries to help answer the question.

    Args:
        state (ResearcherState): The current state of the researcher, including the step to research
            and the self-contained query produced by `rewrite_query`.
        config (RunnableConfig): Configuration with the model used to generate queries.

    Returns:
        dict[str, list[str]]: A dictionary with a 'queries' key containing the list of generated search queries.
    """

    class Response(TypedDict):
        queries: list[str]

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model).with_structured_output(Response)
    # Prefer the pronoun-free rewrite; fall back to the raw step when rewrite was skipped.
    query = state.rewritten_question or state.question
    messages = [
        {"role": "system", "content": configuration.generate_queries_system_prompt},
        {"role": "human", "content": query},
    ]
    response = cast(Response, await model.ainvoke(messages))
    return {"queries": response["queries"]}


async def retrieve_documents(
    state: QueryState, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on a given query.

    This function uses a retriever to fetch relevant documents for a given query.

    Args:
        state (QueryState): The current state containing the query string.
        config (RunnableConfig): Configuration with the retriever used to fetch documents.

    Returns:
        dict[str, list[Document]]: A dictionary with a 'documents' key containing the list of retrieved documents.
    """
    with retrieval.make_retriever(config) as retriever:
        response = await retriever.ainvoke(state.query, config)
        return {"documents": response}


async def rerank_documents(
    state: ResearcherState, *, config: RunnableConfig
) -> dict[str, Any]:
    """把合并后的候选文档重排，留下最相关的前 `rerank_top_n` 篇。

    这个节点位于 `Send` 并行分支**汇合之后**，所以是「一次全局重排」：并行分支
    各自把文档写进 `documents`（按 uuid 去重合并），这里拿到的已经是合并后的候选
    集，一次 rerank 调用即可，延迟不随查询条数放大。

    失败一律降级：`hybrid.rerank` 不抛异常，失败时返回 `(None, 原因)`，本节点于是
    保持候选的原顺序（融合顺序）并只截断到 `rerank_top_n`，把原因写进
    `rerank_error`。重排只是排序优化，绝不能让整条问答链失败。

    Args:
        state (ResearcherState): 当前 researcher 状态，`documents` 已由并行分支合并。
        config (RunnableConfig): 配置（候选数 / top_n / 模型 / 超时 / 开关）。

    Returns:
        dict[str, Any]: 新的 `documents` 顺序，以及 `rerank_applied` / `rerank_error`。
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    documents = list(state.documents or [])
    if not documents:
        return {"rerank_applied": False, "rerank_error": "no documents to rerank"}

    top_n = max(configuration.rerank_top_n, 1)
    candidates = documents[: max(configuration.rerank_candidates, 1)]
    if not configuration.rerank_enabled:
        return {
            "documents": ReplaceDocuments(candidates[:top_n]),
            "rerank_applied": False,
            "rerank_error": "disabled by configuration",
        }

    # 用「自包含的改写问题」重排，而不是某一条扩写出来的检索词：候选是从多条
    # 检索词合并来的，重排要以研究步骤本身的语义为准。
    query = state.rewritten_question or state.question
    results, error = await asyncio.to_thread(
        hybrid.rerank,
        query,
        [document.page_content for document in candidates],
        model=configuration.rerank_model,
        top_n=top_n,
        timeout=configuration.rerank_timeout_s,
    )
    if results is None:
        return {
            "documents": ReplaceDocuments(candidates[:top_n]),
            "rerank_applied": False,
            "rerank_error": error or "unknown rerank failure",
        }
    return {
        "documents": ReplaceDocuments(
            [candidates[index] for index, _score in results]
        ),
        "rerank_applied": True,
        "rerank_error": "",
    }


def retrieve_in_parallel(state: ResearcherState) -> list[Send]:
    """Create parallel retrieval tasks for each generated query.

    This function prepares parallel document retrieval tasks for each query in the researcher's state.

    Args:
        state (ResearcherState): The current state of the researcher, including the generated queries.

    Returns:
        Literal["retrieve_documents"]: A list of Send objects, each representing a document retrieval task.

    Behavior:
        - Creates a Send object for each query in the state.
        - Each Send object targets the "retrieve_documents" node with the corresponding query.
    """
    return [
        Send("retrieve_documents", QueryState(query=query)) for query in state.queries
    ]


# Define the graph
builder = StateGraph(ResearcherState)
builder.add_node(rewrite_query)
builder.add_node(generate_queries)
builder.add_node(retrieve_documents)
builder.add_node("rerank", rerank_documents)
builder.add_edge(START, "rewrite_query")
builder.add_edge("rewrite_query", "generate_queries")
builder.add_conditional_edges(
    "generate_queries",
    retrieve_in_parallel,  # type: ignore
    path_map=["retrieve_documents"],
)
builder.add_edge("retrieve_documents", "rerank")
builder.add_edge("rerank", END)
# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "ResearcherGraph"
