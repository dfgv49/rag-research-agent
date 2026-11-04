"""混合检索的「无服务」基础件：纯 Python BM25 + RRF 融合 + 百炼 rerank 客户端。

本模块刻意只依赖标准库，不引入任何新依赖（项目里没装 rank_bm25 / torch /
sentence-transformers），也不要求 Elasticsearch 或任何外部服务在跑，因此 BM25 与
RRF 可以在单测和离线脚本里直接调用。只有 `rerank()` 需要网络。

三个部件
--------
1. `BM25Index` —— 稀疏路（词法召回）。Okapi BM25，参数 k1 / b 有默认值。
2. `rrf_fuse()` —— 倒数排名融合，把多路「已排序的 id 列表」合成一路。
3. `rerank()` —— 百炼 gte-rerank-v2 客户端，失败一律降级（返回 `None` + 原因），
   绝不向上抛异常，因为重排只是「排序优化」，不该让整条问答链崩掉。

中文分词选型（为什么是「字 bigram + 英文标识符整体成词」）
------------------------------------------------
* 不引入分词器：jieba / pkuseg 都要新依赖，而项目明确禁止安装新依赖。
* 不用「单字」：中文单字的区分度太低，"的"、"是"、"图" 几乎出现在每篇文档里，
  IDF 拉不开差距，检索结果会被高频字主导。
* 用「字 bigram」（相邻两字成词元，单字串退化为单字）：这是无词典条件下最常用
  的近似做法——"检索" 会变成 ("检索")，"并行检索" → ("并行", "行检", "检索")。
  它不需要词典就能把连续汉字切成有局部语序信息的词元，对 BM25 这种只看 tf/idf
  的模型来说，bigram 的召回明显好于单字、又不需要外部分词器。代价是会产生
  "行检" 这种跨词边界的伪词元，带一点噪声。
* 英文/标识符按 `[a-z0-9_]+` 切分并小写：保留 `config_schema`、`add_messages`
  这类下划线标识符整体匹配，这对本项目（LangChain/LangGraph 文档 + 中文提问里的
  英文 API 名）是最有价值的一路信号。
* 不做中文→英文的同义词/术语表扩展：那属于 query expansion，不是分词问题，
  硬编码一份术语表会把评测集「调」进实现里。本题材（全英文语料 + 全中文提问）
  的正确结论写在 README/回报里：字 bigram 无法跨语言匹配，BM25 只在提问里带
  英文标识符时才有贡献。这是**已知局限，不是 bug**：
  - 局限 1：跨语言（中文提问 / 英文语料）完全召回不到，sparse 路为空；
  - 局限 2：bigram 无词形还原，`graph` 与 `graphs` 视为不同词元；
  - 局限 3：纯 Python 全量扫描，复杂度 O(N·|q|)，语料只有 32 篇时是微秒级，
    上万篇就必须换倒排索引（生产路径用 Elasticsearch 的 BM25）。
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from typing import Any, Hashable, Iterable, Optional, Sequence

__all__ = [
    "BM25Index",
    "DEFAULT_RERANK_MODEL",
    "DEFAULT_RERANK_URL",
    "rrf_fuse",
    "rerank",
    "tokenize",
]

#: 百炼文本重排的默认模型（本账号实测可用）。取不到会降级，不会抛错。
DEFAULT_RERANK_MODEL = "gte-rerank-v2"

#: 百炼原生（非 OpenAI 兼容）rerank 端点；实测可用。
DEFAULT_RERANK_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)

# CJK 统一表意文字（含扩展 A 与兼容区）。只用于「按字切 bigram」。
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
# 拉丁字母 / 数字 / 下划线串：`config_schema`、`add_messages` 整体保留。
_LATIN_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """把一段文本切成 BM25 用的词元序列。

    规则（详见模块 docstring 的选型说明）：
    - 连续汉字按「相邻两字」切 bigram；单字串退化为单字本身；
    - 拉丁字母/数字/下划线串按 `[a-z0-9_]+` 整体成词，并统一小写；
    - 标点、空白、其余符号一律丢弃。

    Args:
        text: 原文。允许为空字符串或 None（返回空列表）。

    Returns:
        list[str]: 词元列表，允许重复（BM25 需要词频）。
    """
    if not text:
        return []
    lowered = text.lower()
    tokens: list[str] = []
    for run in _CJK_RUN_RE.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    tokens.extend(_LATIN_TOKEN_RE.findall(lowered))
    return tokens


class BM25Index:
    """Okapi BM25 的内存实现：对一组文档建索引，按查询打分。

    只存词频与文档长度，不做任何持久化；语料很小（本项目 32 篇），打分是纯
    全量扫描，因此不需要倒排表。

    Attributes:
        k1: 词频饱和参数，默认 1.5。越大越奖励重复出现的词。
        b: 文档长度归一化参数，默认 0.75。0 表示完全不做长度归一。

    Examples:
        >>> idx = BM25Index(["graph state machine", "vector retriever store"])
        >>> [doc_id for doc_id, _ in idx.search("vector", k=1)]
        [1]
    """

    def __init__(
        self,
        documents: Sequence[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not 0.0 <= k1:
            raise ValueError("k1 must be >= 0")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be within [0, 1]")
        self.k1 = float(k1)
        self.b = float(b)
        self._term_freqs: list[Counter[str]] = [
            Counter(tokenize(doc or "")) for doc in documents
        ]
        self._lengths: list[int] = [sum(tf.values()) for tf in self._term_freqs]
        #: 平均文档长度；空语料时为 0，`_length_norm` 会跳过归一化。
        self.avgdl: float = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )
        doc_freq: Counter[str] = Counter()
        for tf in self._term_freqs:
            doc_freq.update(tf.keys())
        n_docs = len(self._term_freqs)
        # 概率型 IDF 的 +1 平滑，保证非负：语料里到处都有的词权重趋近 0。
        self._idf: dict[str, float] = {
            term: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def __len__(self) -> int:
        return len(self._term_freqs)

    def _length_norm(self, doc_index: int) -> float:
        if self.avgdl <= 0.0:
            return 1.0
        return self.k1 * (
            1.0 - self.b + self.b * self._lengths[doc_index] / self.avgdl
        )

    def score(self, query: str) -> list[float]:
        """对所有文档算一遍 BM25 分数（未排序，索引顺序）。

        Args:
            query: 查询串。

        Returns:
            list[float]: 与文档等长的分数列表，全部 >= 0。
        """
        terms = tokenize(query)
        scores = [0.0] * len(self._term_freqs)
        if not terms:
            return scores
        idf = self._idf
        k1 = self.k1
        for doc_index, tf in enumerate(self._term_freqs):
            norm = self._length_norm(doc_index)
            total = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                total += idf.get(term, 0.0) * (freq * (k1 + 1.0)) / (freq + norm)
            scores[doc_index] = total
        return scores

    def search(
        self, query: str, k: Optional[int] = None, *, min_score: float = 0.0
    ) -> list[tuple[int, float]]:
        """按 BM25 分数降序返回 `(文档下标, 分数)`。

        Args:
            query: 查询串。
            k: 最多返回几条；None 表示不限。
            min_score: 分数下限（默认 0，即丢掉完全不匹配的文档）。

        Returns:
            list[tuple[int, float]]: 降序排列的 `(doc_index, score)`；分数相同按
            文档下标升序，保证结果稳定可复现。
        """
        scored = [(i, s) for i, s in enumerate(self.score(query)) if s > min_score]
        scored.sort(key=lambda item: (-item[1], item[0]))
        if k is not None:
            if k <= 0:
                return []
            scored = scored[:k]
        return scored


def rrf_fuse(
    ranked_lists: Iterable[Sequence[Hashable]],
    *,
    rank_constant: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> list[Hashable]:
    """Reciprocal Rank Fusion：把多路已排序的 id 列表融合成一路。

    公式（每路 i、每个 id 在其列表中的名次 r 从 1 开始）::

        score(d) = Σ_i  w_i / (rank_constant + r_i(d))

    RRF 只用名次、不用原始分数，因此可以把「余弦相似度」和「BM25 分数」这种量纲
    完全不同的两路直接相加，不需要任何归一化——这是选它而不是加权分数和的理由。

    Args:
        ranked_lists: 多路「已排序的 id 列表」，每路按相关度从高到低。允许为空。
        rank_constant: 平滑常数，默认 60（原论文取值）。越大越削弱头部名次的优势。
        weights: 每路的权重，长度必须与 `ranked_lists` 一致；None 表示全 1。

    Returns:
        list[Hashable]: 融合后的 id 列表，按融合分数降序，每个 id 只出现一次。
        空输入（整体为空、或每路都为空）返回空列表。

    Raises:
        ValueError: `rank_constant <= 0`，或 `weights` 长度与路数不一致。

    Examples:
        >>> rrf_fuse([["a", "b"], ["b", "a"]])
        ['a', 'b']
        >>> rrf_fuse([])
        []
    """
    if rank_constant <= 0:
        raise ValueError("rank_constant must be > 0")
    lists = [list(ranked) for ranked in ranked_lists]
    if weights is None:
        weights = [1.0] * len(lists)
    else:
        weights = [float(w) for w in weights]
        if len(weights) != len(lists):
            raise ValueError(
                f"weights length {len(weights)} != number of lists {len(lists)}"
            )

    fused: dict[Hashable, float] = {}
    best_rank: dict[Hashable, int] = {}
    first_seen: dict[Hashable, int] = {}
    for route_index, (ranked, weight) in enumerate(zip(lists, weights)):
        seen_in_route: set[Hashable] = set()
        for rank, doc_id in enumerate(ranked, start=1):
            # 同一路里的重复 id 只按首次出现计一次，否则某一路刷屏就能刷分。
            if doc_id is None or doc_id in seen_in_route:
                continue
            seen_in_route.add(doc_id)
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (rank_constant + rank)
            if doc_id not in best_rank or rank < best_rank[doc_id]:
                best_rank[doc_id] = rank
            if doc_id not in first_seen:
                first_seen[doc_id] = route_index * 10_000 + rank
    # 同分时的次级键：先看它在某一路里的最好名次，再看首次出现的位置，保证稳定。
    return sorted(
        fused,
        key=lambda doc_id: (-fused[doc_id], best_rank[doc_id], first_seen[doc_id]),
    )


def _rerank_api_key(api_key: Optional[str]) -> Optional[str]:
    """取 API Key：显式入参优先，其次环境变量。取不到返回 None。"""
    key = (api_key or "").strip()
    if key:
        return key
    return (os.getenv("DASHSCOPE_API_KEY") or "").strip() or None


def rerank(
    query: str,
    documents: Sequence[str],
    *,
    model: str = DEFAULT_RERANK_MODEL,
    top_n: Optional[int] = None,
    timeout: float = 30.0,
    api_key: Optional[str] = None,
    url: str = DEFAULT_RERANK_URL,
) -> tuple[Optional[list[tuple[int, float]]], Optional[str]]:
    """调用百炼文本重排，返回按相关度降序的 `(下标, 分数)`。

    这是一个「绝不抛异常」的客户端：网络错误、HTTP 错误、超时、返回结构异常
    全部**降级**为 `(None, 原因字符串)`，由调用方决定是否退回原顺序。重排只是
    排序优化，任何失败都不值得让整条问答链失败。

    Args:
        query: 用户/研究步骤的查询串。
        documents: 待重排的文档正文列表。
        model: 重排模型名，默认 `gte-rerank-v2`。
        top_n: 只返回前 n 条；None 表示全部返回。
        timeout: 单次请求超时（秒），默认 30。
        api_key: 显式 API Key；None 时读环境变量 `DASHSCOPE_API_KEY`。
        url: rerank 端点，默认百炼原生端点。

    Returns:
        tuple: `(results, error)`
            - 成功：`results` 是 `[(原本次的文档下标, relevance_score), ...]`，
              按分数降序；`error` 为 None。
            - 失败：`results` 为 None，`error` 是给人看的原因（含 HTTP 码/异常）。

    Examples:
        >>> results, error = rerank("q", [])          # doctest: +SKIP
        >>> results, error = rerank("q", ["d"], api_key="")  # doctest: +SKIP
    """
    texts = [doc or "" for doc in documents]
    if not texts:
        return [], None
    key = _rerank_api_key(api_key)
    if key is None:
        return None, "DASHSCOPE_API_KEY 未设置，跳过重排"

    payload: dict[str, Any] = {
        "model": model,
        "input": {"query": query or "", "documents": texts},
        "parameters": {"return_documents": False},
    }
    if top_n is not None:
        payload["parameters"]["top_n"] = int(top_n)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:  # 4xx / 5xx：把响应体前 200 字带回来
        try:
            detail = exc.read().decode("utf-8", "replace")[:200]
        except Exception:  # noqa: BLE001 - 连错误体都读不到时不要二次抛错
            detail = ""
        return None, f"HTTP {exc.code}: {detail}"
    except TimeoutError:
        return None, f"timeout after {timeout}s"
    except Exception as exc:  # noqa: BLE001 - 网络/DNS/JSON 一律降级
        return None, f"{type(exc).__name__}: {exc}"

    results = _parse_rerank_body(body, size=len(texts))
    if results is None:
        return None, f"unexpected response shape: {str(body)[:200]}"
    results.sort(key=lambda item: (-item[1], item[0]))
    if top_n is not None:
        results = results[: max(int(top_n), 0)]
    return results, None


def _parse_rerank_body(
    body: Any, *, size: int
) -> Optional[list[tuple[int, float]]]:
    """校验并解析 rerank 响应；结构不对返回 None（调用方据此降级）。"""
    try:
        raw_results = body["output"]["results"]
    except (KeyError, TypeError):
        return None
    if not isinstance(raw_results, list):
        return None
    parsed: list[tuple[int, float]] = []
    seen: set[int] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            return None
        try:
            index = int(item["index"])
            score = float(item["relevance_score"])
        except (KeyError, TypeError, ValueError):
            return None
        if not 0 <= index < size:
            return None
        if index in seen:
            continue
        seen.add(index)
        parsed.append((index, score))
    return parsed
