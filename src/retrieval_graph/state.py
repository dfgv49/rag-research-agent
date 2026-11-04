"""State management for the retrieval graph.

This module defines the state structures used in the retrieval graph. It includes
definitions for agent state, input state, and router classification schema.
"""

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from shared.state import reduce_docs


# Optional, the InputState is a restricted version of the State that is used to
# define a narrower interface to the outside world vs. what is maintained
# internally.
@dataclass(kw_only=True)
class InputState:
    """Represents the input state for the agent.

    This class defines the structure of the input state, which includes
    the messages exchanged between the user and the agent. It serves as
    a restricted version of the full State, providing a narrower interface
    to the outside world compared to what is maintained internally.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    """Messages track the primary execution state of the agent.

    Typically accumulates a pattern of Human/AI/Human/AI messages; if
    you were to combine this template with a tool-calling ReAct agent pattern,
    it may look like this:

    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect
         information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    
        (... repeat steps 2 and 3 as needed ...)
    4. AIMessage without .tool_calls - agent responding in unstructured
        format to the user.

    5. HumanMessage - user responds with the next conversational turn.

        (... repeat steps 2-5 as needed ... )
    
    Merges two lists of messages, updating existing messages by ID.

    By default, this ensures the state is "append-only", unless the
    new message has the same ID as an existing message.

    Returns:
        A new list of messages with the messages from `right` merged into `left`.
        If a message in `right` has the same ID as a message in `left`, the
        message from `right` will replace the message from `left`."""


class Router(TypedDict):
    """Classify user query."""

    logic: str
    type: Literal["more-info", "langchain", "general"]


# This is the primary state of your agent, where you can store any information


@dataclass(kw_only=True)
class AgentState(InputState):
    """State of the retrieval graph / agent."""

    router: Router = field(default_factory=lambda: Router(type="general", logic=""))
    """The router's classification of the user's query."""
    steps: list[str] = field(default_factory=list)
    """A list of steps in the research plan."""
    documents: Annotated[list[Document], reduce_docs] = field(default_factory=list)
    """Populated by the retriever. This is a list of documents that the agent can reference."""
    citations: list[dict[str, Any]] = field(default_factory=list)
    """答案里 [n] 标记到具体文档的回链，由 `respond` 节点写入。

    每一项形如 ``{"marker": "[1]", "position": 1, "resolved": True,
    "source_uuid": ..., "chunk_index": 2, "snippet": "..."}``。模型编出不存在
    的序号时 ``resolved`` 为 False 并带 ``reason``，不会被静默丢掉。

    用 `field(default_factory=list)` 而不是 `Annotated[..., reducer]`：这是每次
    作答整批覆盖的派生结果，不需要跨节点累积。`AgentState` 是 `kw_only=True`，
    给了默认值才能在不改现有调用点的前提下加字段。
    """

    # Feel free to add additional attributes to your state as needed.
    # Common examples include retrieved documents, extracted entities, API connections, etc.
