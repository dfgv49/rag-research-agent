"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for different
vector store backends, specifically Elasticsearch, Pinecone, and MongoDB.

It also holds the **hybrid** retrieval path (BM25 + vectors, fused with RRF, optionally
reranked). That path is additive: `make_retriever` and its behaviour are untouched, and
the hybrid retriever has an in-process backend so it works with no Elasticsearch, no
Docker and no extra dependency.
"""

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Optional, Sequence

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever

from shared import hybrid
from shared.configuration import BaseConfiguration
from shared.state import _generate_uuid

## Encoder constructors

# 阿里云百炼 text-embedding-v4 支持 2048 / 1536 / 1024 / 768 / 512 / 256 / 128 / 64，
# text-embedding-v3 支持 1024 / 768 / 512 / 256 / 128 / 64。
# 这里统一用 1024：官方默认值，效果与成本平衡最好。
DASHSCOPE_EMBEDDING_DIMENSIONS = int(os.getenv("DASHSCOPE_EMBEDDING_DIMENSIONS", "1024"))

# 百炼 embedding 接口对「单次请求的文本条数」有硬上限（不同模型上限不同：
# text-embedding-v3/v4 为 20，v1/v2 为 25）。langchain-openai 默认 chunk_size=1000，
# 一旦一次灌入超过 20 篇文档就会报 400 InvalidParameter，因此这里显式调小。
DASHSCOPE_EMBEDDING_CHUNK_SIZE = int(os.getenv("DASHSCOPE_EMBEDDING_CHUNK_SIZE", "20"))


def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder."""
    provider, model = model.split("/", maxsplit=1)
    match provider:
        case "dashscope":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                model=model,
                api_key=os.environ["DASHSCOPE_API_KEY"],
                base_url=os.getenv(
                    "DASHSCOPE_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
                dimensions=DASHSCOPE_EMBEDDING_DIMENSIONS,
                # 关键：关闭 tiktoken 预切分。langchain-openai 默认会先用 tiktoken
                # 计算 token 数，但 tiktoken 不认识 text-embedding-v4 这类模型名，
                # 会直接抛错。关掉它即可走普通文本接口。
                check_embedding_ctx_length=False,
                # 关键：限制单次请求的文本条数，避免触发百炼的 batch size 上限。
                chunk_size=DASHSCOPE_EMBEDDING_CHUNK_SIZE,
            )
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model)
        case "cohere":
            from langchain_cohere import CohereEmbeddings

            return CohereEmbeddings(model=model)  # type: ignore
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


## Retriever constructors


@contextmanager
def make_elastic_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific elastic index."""
    from langchain_elasticsearch import ElasticsearchStore

    connection_options = {}
    if configuration.retriever_provider == "elastic-local":
        connection_options = {
            "es_user": os.environ["ELASTICSEARCH_USER"],
            "es_password": os.environ["ELASTICSEARCH_PASSWORD"],
        }

    else:
        connection_options = {"es_api_key": os.environ["ELASTICSEARCH_API_KEY"]}

    vstore = ElasticsearchStore(
        **connection_options,  # type: ignore
        es_url=os.environ["ELASTICSEARCH_URL"],
        index_name="langchain_index",
        embedding=embedding_model,
    )

    yield vstore.as_retriever(search_kwargs=configuration.search_kwargs)


@contextmanager
def make_pinecone_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific pinecone index."""
    from langchain_pinecone import PineconeVectorStore

    vstore = PineconeVectorStore.from_existing_index(
        os.environ["PINECONE_INDEX_NAME"], embedding=embedding_model
    )
    yield vstore.as_retriever(search_kwargs=configuration.search_kwargs)


@contextmanager
def make_mongodb_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific MongoDB Atlas index & namespaces."""
    from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

    vstore = MongoDBAtlasVectorSearch.from_connection_string(
        os.environ["MONGODB_URI"],
        namespace="langgraph_retrieval_agent.default",
        embedding=embedding_model,
    )
    yield vstore.as_retriever(search_kwargs=configuration.search_kwargs)


@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Generator[VectorStoreRetriever, None, None]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    match configuration.retriever_provider:
        case "elastic" | "elastic-local":
            with make_elastic_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "pinecone":
            with make_pinecone_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "mongodb":
            with make_mongodb_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(BaseConfiguration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )


## Hybrid retrieval: BM25 + vectors -> RRF -> optional rerank

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def resolve_corpus_path(path: str) -> str:
    """把语料路径解析成绝对路径；相对路径以仓库根目录为基准。

    服务/脚本的工作目录不一定是仓库根（LangGraph 从 `src/` 起、评测脚本从根起），
    因此相对路径不能依赖 cwd。
    """
    if os.path.isabs(path):
        return path
    return os.path.join(_REPO_ROOT, path)


def load_corpus(path: str = "src/sample_docs.json") -> list[Document]:
    """读取语料，并按 `shared.state._generate_uuid` 的规则补上 uuid 元数据。

    uuid 必须与 `eval/testcases.yaml` 里的 `expected_doc_uuids` 用同一条规则
    （md5(page_content) -> uuid），否则评测会把「文档 ID 对不上」全判成 0 分。
    同 page_content 重复出现时只保留一份。

    Args:
        path: 语料路径（相对路径按仓库根解析）。

    Returns:
        list[Document]: 每个文档都带 `metadata["uuid"]`，顺序与文件一致。
    """
    resolved = resolve_corpus_path(path)
    with open(resolved, encoding="utf-8") as handle:
        raw = json.load(handle)

    documents: list[Document] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            content, metadata = item, {}
        else:
            content = item.get("page_content", "") or ""
            metadata = dict(item.get("metadata") or {})
        doc_id = metadata.get("uuid") or _generate_uuid(content)
        if not content or doc_id in seen:
            continue
        seen.add(doc_id)
        metadata["uuid"] = doc_id
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


@dataclass
class HybridHit:
    """混合检索的一路命中。

    Attributes:
        document: 命中的文档（带 uuid 元数据）。
        score: 该路的原始分数。`route == "fused"` 时固定为 0.0——RRF 只用名次，
            融合结果的可信信号是 `rank` 而不是分数，不要跨路比较 score。
        route: 来源，"dense" / "sparse" / "fused" / "rerank"。
        rank: 在该路结果里的名次，从 1 开始。
    """

    document: Document
    score: float
    route: str
    rank: int


@dataclass
class HybridResult:
    """一次混合检索的结果 + 观测数据（耗时 / 降级原因）。"""

    hits: list[HybridHit]
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def documents(self) -> list[Document]:
        """只取文档列表，顺序即最终排序。"""
        return [hit.document for hit in self.hits]

    def ids(self) -> list[str]:
        """只取 uuid 列表，顺序即最终排序（评测脚本直接用这个）。"""
        return [str(hit.document.metadata.get("uuid", "")) for hit in self.hits]


class InMemoryHybridRetriever:
    """进程内混合检索：向量路（余弦）+ 稀疏路（BM25）+ RRF（+ 可选重排）。

    没有 Elasticsearch 也能完整工作：向量由项目自己的 embedding 模型算出来存在
    内存里，稀疏路是 `shared.hybrid.BM25Index`。这让「混合检索到底有没有用」可以
    在 ES 不可用的环境下被真实量出来，而不是只能靠推断。

    Args:
        documents: 语料；缺 uuid 的会按 `_generate_uuid` 补上。
        embedding_model: 向量路用的 encoder（须实现 `langchain_core` Embeddings）。
        k1: BM25 词频饱和参数。
        b: BM25 文档长度归一化参数。
        dense_k: 向量路默认候选窗口。
        sparse_k: 稀疏路默认候选窗口。
        rank_constant: RRF 默认 rank constant。
        dense_weight: 向量路的 RRF 权重。
        sparse_weight: 稀疏路的 RRF 权重。
        rerank_model: 默认重排模型。
        rerank_candidates: 默认送进重排的候选数。
        rerank_top_n: 默认重排后保留数。
        rerank_timeout: 默认重排超时（秒）。

    Examples:
        >>> retriever = InMemoryHybridRetriever(load_corpus(), encoder)  # doctest: +SKIP
        >>> [d.metadata["uuid"] for d in retriever.search("q", k=4).documents]  # doctest: +SKIP
    """

    def __init__(
        self,
        documents: Sequence[Document],
        embedding_model: Embeddings,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        dense_k: int = 20,
        sparse_k: int = 20,
        rank_constant: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        rerank_model: str = hybrid.DEFAULT_RERANK_MODEL,
        rerank_candidates: int = 20,
        rerank_top_n: int = 5,
        rerank_timeout: float = 30.0,
    ) -> None:
        prepared: list[Document] = []
        seen: set[str] = set()
        for document in documents:
            metadata = dict(document.metadata or {})
            doc_id = metadata.get("uuid") or _generate_uuid(document.page_content)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            metadata["uuid"] = doc_id
            prepared.append(Document(page_content=document.page_content, metadata=metadata))
        self.documents = prepared
        self.embedding_model = embedding_model
        self._bm25 = hybrid.BM25Index(
            [document.page_content for document in prepared], k1=k1, b=b
        )
        self._matrix: Optional[np.ndarray] = None
        self._matrix_ms = 0.0
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.rank_constant = rank_constant
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rerank_model = rerank_model
        self.rerank_candidates = rerank_candidates
        self.rerank_top_n = rerank_top_n
        self.rerank_timeout = rerank_timeout

    # -- 向量路 ---------------------------------------------------------

    def _corpus_matrix(self) -> np.ndarray:
        """懒加载语料向量矩阵（已做 L2 归一化，余弦相似度 = 点积）。"""
        if self._matrix is None:
            texts = [document.page_content for document in self.documents]
            started = time.perf_counter()
            vectors = self.embedding_model.embed_documents(texts)
            matrix = np.asarray(vectors, dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            self._matrix = matrix / np.maximum(norms, 1e-9)
            self._matrix_ms = (time.perf_counter() - started) * 1000.0
        return self._matrix

    def dense_search(self, query: str, k: int) -> list[HybridHit]:
        """向量路召回，按余弦相似度降序。"""
        matrix = self._corpus_matrix()
        started = time.perf_counter()
        vector = np.asarray(
            self.embedding_model.embed_query(query), dtype=np.float32
        )
        vector = vector / max(float(np.linalg.norm(vector)), 1e-9)
        similarities = matrix @ vector
        order = np.argsort(-similarities)[: max(int(k), 0)]
        self._last_dense_ms = (time.perf_counter() - started) * 1000.0
        return [
            HybridHit(
                document=self.documents[int(index)],
                score=float(similarities[int(index)]),
                route="dense",
                rank=rank,
            )
            for rank, index in enumerate(order, start=1)
        ]

    # -- 稀疏路 ---------------------------------------------------------

    def sparse_search(self, query: str, k: int) -> list[HybridHit]:
        """BM25 稀疏路召回，按 BM25 分数降序（分数为 0 的不返回）。"""
        started = time.perf_counter()
        scored = self._bm25.search(query, k=k)
        self._last_sparse_ms = (time.perf_counter() - started) * 1000.0
        return [
            HybridHit(
                document=self.documents[index],
                score=score,
                route="sparse",
                rank=rank,
            )
            for rank, (index, score) in enumerate(scored, start=1)
        ]

    # -- 融合 + 重排 -----------------------------------------------------

    def search(
        self,
        query: str,
        *,
        dense_k: Optional[int] = None,
        sparse_k: Optional[int] = None,
        rank_constant: Optional[int] = None,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None,
        rerank: bool = False,
        rerank_candidates: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
        rerank_model: Optional[str] = None,
        rerank_timeout: Optional[float] = None,
        rerank_api_key: Optional[str] = None,
        k: Optional[int] = None,
    ) -> HybridResult:
        """混合检索：两路召回 -> RRF 融合 ->（可选）重排。

        Args:
            query: 查询串。
            dense_k: 向量路候选窗口，None 用实例默认值。
            sparse_k: 稀疏路候选窗口，None 用实例默认值。
            rank_constant: RRF 常数，None 用实例默认值。
            dense_weight: 向量路 RRF 权重。
            sparse_weight: 稀疏路 RRF 权重。
            rerank: 是否在融合之后再调一次重排。失败自动降级为「不重排，保持融合
                顺序取前 rerank_top_n 条」，绝不外抛。
            rerank_candidates: 送进重排的候选数。
            rerank_top_n: 重排后保留条数。
            rerank_model: 重排模型。
            rerank_timeout: 重排超时（秒）。
            rerank_api_key: 显式 API Key；None 时读环境变量。
            k: 最后再截断到 k 条；None 表示不截断。

        Returns:
            HybridResult: 有序命中 + 耗时/降级等观测数据。
        """
        dense_k = self.dense_k if dense_k is None else dense_k
        sparse_k = self.sparse_k if sparse_k is None else sparse_k
        rank_constant = (
            self.rank_constant if rank_constant is None else rank_constant
        )
        dense_weight = self.dense_weight if dense_weight is None else dense_weight
        sparse_weight = self.sparse_weight if sparse_weight is None else sparse_weight
        rerank_candidates = (
            self.rerank_candidates if rerank_candidates is None else rerank_candidates
        )
        rerank_top_n = self.rerank_top_n if rerank_top_n is None else rerank_top_n
        rerank_model = self.rerank_model if rerank_model is None else rerank_model
        rerank_timeout = (
            self.rerank_timeout if rerank_timeout is None else rerank_timeout
        )

        stats: dict[str, Any] = {}
        dense_hits = self.dense_search(query, dense_k)
        sparse_hits = self.sparse_search(query, sparse_k)
        stats["dense_hits"] = len(dense_hits)
        stats["sparse_hits"] = len(sparse_hits)
        stats["dense_ms"] = round(self._last_dense_ms, 1)
        stats["sparse_ms"] = round(self._last_sparse_ms, 1)

        started = time.perf_counter()
        fused_ids = hybrid.rrf_fuse(
            [
                [hit.document.metadata["uuid"] for hit in dense_hits],
                [hit.document.metadata["uuid"] for hit in sparse_hits],
            ],
            rank_constant=rank_constant,
            weights=[dense_weight, sparse_weight],
        )
        by_id = {
            hit.document.metadata["uuid"]: hit for hit in dense_hits + sparse_hits
        }
        hits = [
            HybridHit(document=by_id[doc_id].document, score=0.0, route="fused", rank=rank)
            for rank, doc_id in enumerate(fused_ids, start=1)
        ]
        stats["fuse_ms"] = round((time.perf_counter() - started) * 1000.0, 2)

        if rerank and hits:
            candidates = hits[: max(int(rerank_candidates), 1)]
            started = time.perf_counter()
            results, error = hybrid.rerank(
                query,
                [hit.document.page_content for hit in candidates],
                model=rerank_model,
                top_n=max(int(rerank_top_n), 1),
                timeout=rerank_timeout,
                api_key=rerank_api_key,
            )
            stats["rerank_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
            stats["rerank_candidates"] = len(candidates)
            if results is None:
                # 降级：不重排，保持融合顺序，只截断到 top_n。
                stats["rerank_applied"] = False
                stats["rerank_error"] = error
                hits = candidates[: max(int(rerank_top_n), 1)]
            else:
                stats["rerank_applied"] = True
                stats["rerank_error"] = None
                hits = [
                    HybridHit(
                        document=candidates[index].document,
                        score=score,
                        route="rerank",
                        rank=rank,
                    )
                    for rank, (index, score) in enumerate(results, start=1)
                ]

        if k is not None:
            hits = hits[: max(int(k), 0)]
        stats["returned"] = len(hits)
        return HybridResult(hits=hits, stats=stats)

# ---------------------------------------------------------------------------
# 已删除：ElasticHybridRetriever / make_elastic_hybrid_retriever / make_hybrid_retriever
#
# 决策依据（2026-10）：
#   1. ES 混合检索路径**从未连过真实集群**（本机 Docker/WSL 故障，9200 一直不通），
#      属于「写了但没验证」的代码；留着会让技术栈出现无法追溯的声明。
#   2. make_hybrid_retriever 无任何调用方 —— 实验脚本直接构造 InMemoryHybridRetriever。
#   3. 实测结论：本项目（中文提问 / 英文语料）下混合检索为统计显著负收益
#      （mrr −0.14，95% CI [−0.24, −0.06]，p≈0.001），且 RRF 稀疏路最优权重为 0。
#      详见 report_final.md 第 4.6 节的两个消融实验。
#
# 保留：InMemoryHybridRetriever 与 hybrid.py 的 BM25Index / rrf_fuse / rerank ——
#       它们是消融实验的引擎，由 verify_hybrid_ab.py 直接调用，与产品默认路径无关。
# ---------------------------------------------------------------------------
