"""进程内语义缓存。

一次查询要花掉的真正成本有两块：embedding 调用，以及（更贵的）整条检索链与
LLM 调用。用户重复问同一件事、或者换个说法再问一遍时，这两块成本都可以省掉。

本模块提供的是一个**不依赖任何外部服务**的语义缓存：

- 用 embedding 把 query 变成向量，和已缓存的 query 向量比余弦相似度；
- 相似度 ≥ 阈值（默认 0.95）即视为同一个问题，直接返回缓存结果；
- 未命中就让调用方走原流程，然后把「query 向量 + 结果」存进来；
- 有 LRU 上限（默认 128 条）与 TTL（默认 1 小时，按**单调时钟**计量）。

接口被抽象成 `CacheBackend` 协议，所以「检索」这一步可以被透明替换：
``cache.get(query)`` 命中就直接用，未命中再走真正的检索，最后 ``cache.set``。

为什么不用 Redis：本项目未安装 redis 客户端、也没有可连的服务端，更没有
Docker 可用。写一个「必须连 Redis 才能验证」的缓存，等于交付一个未经验证的模块。
因此这里做的是**可验证的进程内实现**，同时留出 `RedisCacheBackend` 骨架，
等真正部署到多进程/多实例环境时再切换（切换方法见该类 docstring）。

典型用法::

    from shared.cache import InProcessSemanticCache, wrap_retriever

    cache = InProcessSemanticCache()
    documents = cache.get_or_compute(query, lambda: retriever.invoke(query))
    print(cache.stats())
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Optional, Protocol, TypeVar, runtime_checkable

import numpy as np
from langchain_core.embeddings import Embeddings

T = TypeVar("T")

# 0.95 的取法：这是「语义等价」而不是「话题相关」的门槛。实测同义改写
# （换语序、加语气词）的余弦相似度通常 ≥ 0.97，而「同一主题的不同问题」
# 往往落在 0.90~0.95 之间——阈值取低了会把不同问题的答案张冠李戴。
DEFAULT_SIMILARITY_THRESHOLD = 0.95

# 128 条 ≈ 几十个会话的常见热点问题量级；再大就要考虑共享后端了。
DEFAULT_MAX_ENTRIES = 128

# 1 小时：文档没变的前提下，答案的有效期主要取决于模型版本与提示词是否变化。
DEFAULT_TTL_SECONDS = 3600.0


@dataclass
class CacheStats:
    """命中统计快照。"""

    lookups: int = 0
    """总查询次数。"""
    hits: int = 0
    """命中次数（含 exact 与 semantic）。"""
    exact_hits: int = 0
    """query 文本完全相同、连 embedding 都不用算的命中。"""
    semantic_hits: int = 0
    """靠向量相似度过线的命中。"""
    misses: int = 0
    """未命中次数。"""
    evictions: int = 0
    """因超过 LRU 上限被淘汰的条目数。"""
    expirations: int = 0
    """因 TTL 过期被清理的条目数。"""
    total_hit_latency_ms: float = 0.0
    """命中路径的累计耗时（毫秒），用于算平均命中耗时。"""
    entries: int = 0
    """当前缓存条目数。"""

    @property
    def hit_rate(self) -> float:
        """命中率 = 命中数 / 总查询数；没有查询时返回 0.0。"""
        return self.hits / self.lookups if self.lookups else 0.0

    @property
    def avg_hit_latency_ms(self) -> float:
        """平均命中耗时（毫秒）；没有命中时返回 0.0。"""
        return self.total_hit_latency_ms / self.hits if self.hits else 0.0

    def as_dict(self) -> dict[str, Any]:
        """导出为普通 dict（便于打日志 / 上报）。"""
        return {
            "lookups": self.lookups,
            "hits": self.hits,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "avg_hit_latency_ms": round(self.avg_hit_latency_ms, 3),
            "evictions": self.evictions,
            "expirations": self.expirations,
            "entries": self.entries,
        }


@runtime_checkable
class CacheBackend(Protocol):
    """缓存的抽象接口。

    只要实现下面四个方法，就能替换掉 `InProcessSemanticCache`——包括换成
    Redis、Memcached 或者带独立服务的向量缓存。调用方只依赖这个协议。
    """

    def get(self, query: str) -> Optional[Any]:
        """查询缓存；命中返回结果，未命中返回 None。"""
        ...

    def set(self, query: str, result: Any) -> None:
        """把 query 及其结果写入缓存。"""
        ...

    def stats(self) -> dict[str, Any]:
        """返回命中统计。"""
        ...

    def clear(self) -> None:
        """清空缓存。"""
        ...


@dataclass
class _Entry(Generic[T]):
    """一条缓存记录。"""

    query: str
    vector: np.ndarray
    result: T
    created_at: float
    expires_at: float
    last_used: float


class InProcessSemanticCache(Generic[T]):
    """进程内语义缓存（余弦相似度 + LRU + TTL）。

    Args:
        encoder: 文本向量化器。不传则用 `shared.retrieval.make_text_encoder`
            按 `embedding_model` 惰性创建（复用项目已有的 encoder 配置，
            不再新增依赖）。
        embedding_model: encoder 的 `provider/model` 名。
        similarity_threshold: 命中阈值，见 `DEFAULT_SIMILARITY_THRESHOLD`。
        max_entries: LRU 上限。
        ttl_seconds: 条目存活秒数，按单调时钟计。
        clock: 单调时钟函数，默认 `time.monotonic`；测试可注入假时钟。

    Examples:
        >>> cache: InProcessSemanticCache[int] = InProcessSemanticCache(
        ...     encoder=_StubEncoder(), max_entries=2
        ... )
        >>> cache.get("a")            # 未命中
        >>> cache.set("a", 1)
        >>> cache.get("a")            # 命中
        1
    """

    def __init__(
        self,
        encoder: Optional[Embeddings] = None,
        *,
        embedding_model: str = "dashscope/text-embedding-v4",
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._encoder = encoder
        self._embedding_model = embedding_model
        self.similarity_threshold = float(similarity_threshold)
        self.max_entries = int(max_entries)
        self.ttl_seconds = float(ttl_seconds)
        self._clock = clock
        # OrderedDict 本身就是 LRU 结构：新条目放末尾，淘汰从头部拿，
        # 命中时 move_to_end 刷新最近使用顺序。
        self._entries: "OrderedDict[str, _Entry[T]]" = OrderedDict()
        # query 文本完全相同就走这条快路：连 embedding 都不用算。
        self._stats = CacheStats()
        self._lock = threading.Lock()
        self._similarity_ms = 0.0

    # -- encoder --------------------------------------------------------

    @property
    def encoder(self) -> Embeddings:
        """惰性创建文本 encoder，避免 import 时就要求 API Key。"""
        if self._encoder is None:
            from shared.retrieval import make_text_encoder

            self._encoder = make_text_encoder(self._embedding_model)
        return self._encoder

    def _embed(self, query: str) -> np.ndarray:
        """把 query 变成 L2 归一化向量（归一化后余弦相似度 = 点积）。"""
        vector = np.asarray(self.encoder.embed_query(query), dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / max(norm, 1e-9)

    # -- 内部 ----------------------------------------------------------

    def _purge_expired(self, now: float) -> int:
        """清掉过期条目，返回清掉的数量。"""
        expired = [key for key, entry in self._entries.items() if entry.expires_at <= now]
        for key in expired:
            del self._entries[key]
        self._stats.expirations += len(expired)
        return len(expired)

    def _lookup_vector(self, vector: np.ndarray) -> Optional[_Entry[T]]:
        """在未过期条目里找相似度最高且过阈值的一条。"""
        started = time.perf_counter()
        best: Optional[_Entry[T]] = None
        best_score = -1.0
        for entry in self._entries.values():
            score = float(np.dot(entry.vector, vector))
            if score > best_score:
                best_score, best = score, entry
        self._similarity_ms = (time.perf_counter() - started) * 1000.0
        if best is not None and best_score >= self.similarity_threshold:
            return best
        return None

    def _store(self, query: str, vector: np.ndarray, result: T) -> None:
        """写入一条记录并维持 LRU 上限。"""
        now = self._clock()
        if query in self._entries:
            self._entries.move_to_end(query)
            self._entries[query] = _Entry(
                query=query,
                vector=vector,
                result=result,
                created_at=now,
                expires_at=now + self.ttl_seconds,
                last_used=now,
            )
            return
        self._entries[query] = _Entry(
            query=query,
            vector=vector,
            result=result,
            created_at=now,
            expires_at=now + self.ttl_seconds,
            last_used=now,
        )
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self._stats.evictions += 1

    def _note_hit(self, entry: _Entry[T], started: float, *, exact: bool) -> T:
        """记录一次命中并返回结果。"""
        self._entries.move_to_end(entry.query)
        entry.last_used = self._clock()
        self._stats.hits += 1
        if exact:
            self._stats.exact_hits += 1
        else:
            self._stats.semantic_hits += 1
        self._stats.total_hit_latency_ms += (time.perf_counter() - started) * 1000.0
        return entry.result

    # -- CacheBackend --------------------------------------------------

    def get(self, query: str) -> Optional[T]:
        """查询缓存。命中返回结果，未命中返回 None。"""
        with self._lock:
            started = time.perf_counter()
            self._stats.lookups += 1
            now = self._clock()
            self._purge_expired(now)

            exact = self._entries.get(query)
            if exact is not None:
                return self._note_hit(exact, started, exact=True)

            entry = self._lookup_vector(self._embed(query))
            if entry is not None:
                return self._note_hit(entry, started, exact=False)

            self._stats.misses += 1
            return None

    def set(self, query: str, result: T) -> None:
        """把 query 与结果写入缓存。"""
        with self._lock:
            now = self._clock()
            self._purge_expired(now)
            self._store(query, self._embed(query), result)

    def get_or_compute(self, query: str, compute: Callable[[], T]) -> T:
        """命中就用缓存，未命中就调 `compute()` 并把结果存起来。

        与先 `get` 再 `set` 的区别：这里只算一次 query 向量，不会为了写入再
        调一次 embedding。

        Args:
            query: 用户查询文本。
            compute: 未命中时执行的原始流程（例如真正的检索）。

        Returns:
            T: 缓存中的结果，或 `compute()` 的返回值。
        """
        with self._lock:
            started = time.perf_counter()
            self._stats.lookups += 1
            now = self._clock()
            self._purge_expired(now)

            exact = self._entries.get(query)
            if exact is not None:
                return self._note_hit(exact, started, exact=True)

            vector = self._embed(query)
            entry = self._lookup_vector(vector)
            if entry is not None:
                return self._note_hit(entry, started, exact=False)

            self._stats.misses += 1

        # 慢路径（真正的检索）放在锁外：它可能耗时几百毫秒甚至几秒，
        # 占着锁会让并发的 Send 检索全部串行化。
        result = compute()
        with self._lock:
            self._store(query, np.asarray(vector, dtype=np.float32), result)
        return result

    def stats(self) -> dict[str, Any]:
        """返回命中统计（命中率、平均命中耗时、淘汰/过期计数等）。"""
        with self._lock:
            self._purge_expired(self._clock())
            self._stats.entries = len(self._entries)
            snapshot = self._stats.as_dict()
            # 相似度比较本身的耗时，用于区分「快路命中」和「向量比较命中」。
            snapshot["similarity_ms"] = round(self._similarity_ms, 3)
        return snapshot

    def clear(self) -> None:
        """清空缓存（统计一并归零）。"""
        with self._lock:
            self._entries.clear()
            self._stats = CacheStats()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def wrap_retriever(cache: InProcessSemanticCache[T], retrieve: Callable[[str], T]) -> Callable[[str], T]:
    """把一个「query -> 结果」的函数包成带语义缓存的版本。

    这是「缓存直接替换检索这一步」的最短接法，不用改检索实现本身::

        cached = wrap_retriever(InProcessSemanticCache(), retriever.invoke)
        documents = cached("LangGraph 怎么做并行？")

    Args:
        cache: 语义缓存实例。
        retrieve: 原始检索函数。

    Returns:
        Callable[[str], T]: 带缓存语义的同签名函数。
    """

    def cached(query: str) -> T:
        return cache.get_or_compute(query, lambda: retrieve(query))

    return cached


class RedisCacheBackend(Generic[T]):
    """Redis 语义缓存后端 —— **骨架，未实现，也不在本环境验证**。

    本环境未安装 `redis` 客户端、没有可连的 Redis 服务端，也不允许装依赖，
    因此这里只留结构与切换说明；`get` / `set` 会抛出清晰的错误而不是静默失败。

    为什么需要它：`InProcessSemanticCache` 的缓存活在**单个进程**里。一旦用
    gunicorn / uvicorn 多 worker、或者 LangGraph 部署成多实例，缓存命中率会被
    进程数除尽，且各进程结果不一致。生产环境应换成共享后端。

    切换步骤::

        # 1. 装依赖（本环境禁止，部署环境才做）
        #    pip install redis            # 客户端
        #    pip install redisvl          # 语义缓存需要向量索引，可选但推荐
        #    或直接要求 Redis Stack / RediSearch 模块（提供向量相似度检索）

        # 2. 配置环境变量
        #    REDIS_URL=redis://localhost:6379/0
        #    REDIS_CACHE_INDEX=rag_semantic_cache   # RediSearch 索引名
        #    REDIS_CACHE_TTL=3600

        # 3. 把构造点换掉（调用方代码不用改）
        #    from shared.cache import RedisCacheBackend
        #    cache = RedisCacheBackend.from_env()

    实现要点（留给后续）：

    - 向量索引：用 `FT.CREATE` 建 HNSW/FLAT 索引，字段为 query 向量；
      `FT.SEARCH ... KNN k 1` 取最近邻，返回的 distance 转成余弦相似度后
      再和阈值比较——阈值语义必须与进程内实现一致，否则切换后命中率会漂移。
    - 序列化：结果通常是 `list[Document]`，用 pickle 或 json 存；json 更安全，
      但要注意 `Document.metadata` 里可能有非 JSON 类型。
    - TTL 交给 Redis 的 `EXPIRE`（秒级），LRU 交给 `maxmemory-policy allkeys-lru`，
      不要在这里自己维护——自己维护会和 Redis 的淘汰策略打架。
    - 失败必须降级：Redis 不可用时 `get` 返回 None（当作未命中），绝不阻断检索。

    Args:
        url: Redis 连接串，默认读 `REDIS_URL`。
        similarity_threshold: 命中阈值，语义同 `InProcessSemanticCache`。
        ttl_seconds: 条目存活秒数。
        index_name: RediSearch 索引名。
    """

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        index_name: str = "rag_semantic_cache",
    ) -> None:
        self.url = url
        self.similarity_threshold = float(similarity_threshold)
        self.ttl_seconds = float(ttl_seconds)
        self.index_name = index_name

    @classmethod
    def from_env(cls) -> "RedisCacheBackend[T]":
        """按环境变量构造（`REDIS_URL` / `REDIS_CACHE_INDEX` / `REDIS_CACHE_TTL`）。"""
        import os

        return cls(
            url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            index_name=os.getenv("REDIS_CACHE_INDEX", "rag_semantic_cache"),
            ttl_seconds=float(os.getenv("REDIS_CACHE_TTL", DEFAULT_TTL_SECONDS)),
        )

    @staticmethod
    def _require_redis() -> Any:
        """延迟导入 redis 客户端；未安装时给出可执行的错误信息。"""
        try:
            import redis  # type: ignore[import-not-found]
        except ImportError as error:  # pragma: no cover - 本环境必然走这里
            raise RuntimeError(
                "Redis 语义缓存未启用：当前环境未安装 `redis` 客户端。"
                "如需启用，请先 `pip install redis`（语义检索还需要 Redis Stack 或 "
                "redisvl 提供的向量索引），并设置 REDIS_URL；本项目的进程内实现"
                "（InProcessSemanticCache）无需任何依赖，可直接使用。"
            ) from error
        return redis

    def get(self, query: str) -> Optional[T]:
        """未实现：需要先按类 docstring 接上向量索引。"""
        self._require_redis()
        raise NotImplementedError(
            "RedisCacheBackend.get 尚未实现，见 shared/cache.py 的类 docstring。"
        )

    def set(self, query: str, result: T) -> None:
        """未实现：需要先按类 docstring 接上向量索引。"""
        self._require_redis()
        raise NotImplementedError(
            "RedisCacheBackend.set 尚未实现，见 shared/cache.py 的类 docstring。"
        )

    def stats(self) -> dict[str, Any]:
        """未实现：命中统计应由 Redis 侧计数器提供。"""
        self._require_redis()
        raise NotImplementedError("RedisCacheBackend.stats 尚未实现。")

    def clear(self) -> None:
        """未实现：应删除索引内全部键。"""
        self._require_redis()
        raise NotImplementedError("RedisCacheBackend.clear 尚未实现。")
