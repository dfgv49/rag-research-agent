"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config


@dataclass(kw_only=True)
class BaseConfiguration:
    """Configuration class for indexing and retrieval operations.

    This class defines the parameters needed for configuring the indexing and
    retrieval processes, including embedding model selection, retriever provider choice, and search parameters.
    """

    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = field(
        default="dashscope/text-embedding-v4",
        metadata={
            "description": "Name of the embedding model to use. Must be a valid embedding model name."
        },
    )

    retriever_provider: Annotated[
        Literal["elastic-local", "elastic", "pinecone", "mongodb"],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = field(
        default="elastic-local",
        metadata={
            "description": "The vector store provider to use for retrieval. Options are 'elastic', 'pinecone', or 'mongodb'."
        },
    )

    search_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "Additional keyword arguments to pass to the search function of the retriever."
        },
    )

    # ------------------------------------------------------------------
    # 重排（cross-encoder，researcher 子图的可选节点）
    #
    # 本文件现在只保留**重排**相关字段（rerank_*）；混合检索那 9 个参数已移出
    # 主配置，下沉到实现处作为实验默认值（见下方说明）。
    #
    # rerank_* 字段由 `retrieval_graph/researcher_graph/graph.py` 的 rerank 节点读取；
    # 它们同样**不参与** `make_retriever` 的老路径，`retriever_provider` 取值集合没变，
    # 现有向量检索行为完全不受影响。
    #
    # ============================ 实测结论（重排默认关闭的依据） ============================
    # 语料：229 篇英文技术文档（src/sample_docs_large.json）；提问：中文。
    # 评测：60 条用例，配对 bootstrap（2000 次重采样，95% CI）。
    #
    # 1) 重排 vs 纯向量（同规模对照）：mrr −0.0550，95% CI 跨 0 → **无显著差异**；
    #    但每次检索多花 854~1241 ms（整篇文档候选）→ 无收益却付延迟，故默认关闭。
    # 2) 天花板分析：候选池可达召回 0.9500，重排后保留 0.9333（98.25%，60 条只丢 1 条）
    #    → 瓶颈在候选池，不在重排本身；要提高指标应先解决召回。
    # 3) 触发重开的条件：改用 **chunk 级重排**（候选 ~300 字符，落在 cross-encoder
    #    的有效窗口内）。当前整篇文档（最长 8000 字符）远超窗口，是重排失效的主因之一。
    #
    # 背景（混合检索的结论，仅供追溯；参数已不在本文件）：
    #   混合检索在本项目为统计显著负收益（mrr −0.14，p≈0.001）；RRF 稀疏权重消融显示
    #   最优权重为 0；受控实验（把 BM25 输入换成英译）使负收益消失，证实根因为跨语言
    #   分词失效。完整数据见 report_final.md 第 4.6 节。
    # ==================================================================================

    # 已移出主配置：hybrid_provider / hybrid_corpus_path / hybrid_dense_k / hybrid_sparse_k /
    # hybrid_rank_constant / hybrid_dense_weight / hybrid_sparse_weight / bm25_k1 / bm25_b
    #
    # 这 9 个字段原本只服务于「混合检索」这条实验路径，没有任何产品代码读取它们
    # （唯一读取方 make_hybrid_retriever 已随 ES 路径一并删除）。留在 BaseConfiguration
    # 里会让它们出现在 LangGraph Studio 的配置面板上，看起来像产品功能。
    #
    # 现在这些参数作为**实验默认值**下沉到实现处：
    #   - BM25 的 k1 / b            -> hybrid.BM25Index(k1=1.5, b=0.75)
    #   - RRF 的 rank_constant      -> hybrid.rrf_fuse(rank_constant=60)
    #   - dense_k / sparse_k / 权重 -> verify_hybrid_ab.py 的模块级常量
    #
    # 复现实验：python verify_hybrid_ab.py --corpus src/sample_docs_large.json \
    #             --testcases eval/testcases_synthetic.yaml --skip-chunk --no-latency

    rerank_enabled: bool = field(
        # 实测（229 篇 / 60 条，配对 bootstrap）：重排与纯向量在统计上**无显著差异**
        # （同规模对照 D−A′：mrr −0.0550，95% CI 跨 0），但每次检索要多花
        # 854~1241 ms（整篇文档候选）。没有收益却付出延迟 → 默认关闭。
        # 天花板分析：候选池可达召回 0.9500，重排后保留 0.9333（98.25%，只丢 1 条），
        # 说明瓶颈在候选池而非重排本身 —— 若要重开，先解决召回，再谈重排。
        # 触发重开的条件：改用 chunk 级重排（候选 ~300 字符，落在 cross-encoder 有效窗口内）。
        default=False,
        metadata={
            "description": "Whether the researcher subgraph reranks its merged candidates. Defaults to False: measured no statistically significant gain and +850~1240 ms latency on this corpus. A rerank failure always degrades to no rerank."
        },
    )

    rerank_model: str = field(
        default="gte-rerank-v2",
        metadata={"description": "DashScope/Bailian rerank model name."},
    )

    rerank_candidates: int = field(
        default=20,
        metadata={"description": "How many fused candidates are handed to the reranker."},
    )

    rerank_top_n: int = field(
        default=5,
        metadata={"description": "How many documents survive reranking."},
    )

    rerank_timeout_s: float = field(
        default=30.0, metadata={"description": "Per-call rerank timeout in seconds."}
    )

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        """Create an IndexConfiguration instance from a RunnableConfig object.

        Args:
            cls (Type[T]): The class itself.
            config (Optional[RunnableConfig]): The configuration object to use.

        Returns:
            T: An instance of IndexConfiguration with the specified configuration.
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


T = TypeVar("T", bound=BaseConfiguration)
