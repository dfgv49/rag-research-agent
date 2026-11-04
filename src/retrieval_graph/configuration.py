"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from retrieval_graph import prompts
from shared.configuration import BaseConfiguration


@dataclass(kw_only=True)
class AgentConfiguration(BaseConfiguration):
    """The configuration for the agent."""

    # models

    # 默认模型说明（2026-02 实测该账号）：
    # - 原默认值 dashscope/qwen-turbo 连续返回 429 RateLimitError（"exceeded your current
    #   request limit"），原默认值 dashscope/qwen-max 返回 403 Free quota exhausted，
    #   两者都跑不通，会让本地评测与自测直接失败。
    # - 因此改用当前账号可用的 qwen-flash（路由/计划/查询改写，快且便宜）与
    #   qwen-plus（最终作答，质量更高）。仍然只写 provider/model 形式，可随时用
    #   RunnableConfig 的 configurable 覆盖。

    query_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="dashscope/qwen-flash",
        metadata={
            "description": "The language model used for processing and refining queries. Should be in the form: provider/model-name. Defaults to dashscope/qwen-flash because qwen-turbo returns HTTP 429 (rate limited) on this account."
        },
    )

    response_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="dashscope/qwen-plus",
        metadata={
            "description": "The language model used for generating responses. Should be in the form: provider/model-name. Defaults to dashscope/qwen-plus because qwen-max returns HTTP 403 (free quota exhausted) on this account."
        },
    )

    # prompts

    router_system_prompt: str = field(
        default=prompts.ROUTER_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for classifying user questions to route them to the correct node."
        },
    )

    more_info_system_prompt: str = field(
        default=prompts.MORE_INFO_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for asking for more information from the user."
        },
    )

    general_system_prompt: str = field(
        default=prompts.GENERAL_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for responding to general questions."
        },
    )

    research_plan_system_prompt: str = field(
        default=prompts.RESEARCH_PLAN_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for generating a research plan based on the user's question."
        },
    )

    generate_queries_system_prompt: str = field(
        default=prompts.GENERATE_QUERIES_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used by the researcher to generate queries based on a step in the research plan."
        },
    )

    query_rewrite_system_prompt: str = field(
        default=prompts.QUERY_REWRITE_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used by the researcher to rewrite a follow-up question into a self-contained search query, using the conversation history."
        },
    )

    response_system_prompt: str = field(
        default=prompts.RESPONSE_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for generating responses."},
    )
