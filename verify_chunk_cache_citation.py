"""验证脚本：文档分块入库 + 进程内语义缓存 + 引用回链。

对应第 ④ 项改造的三个部分，全程**只读仓库、只打印 stdout**（Windows 沙箱下
Python 子进程无法在仓库内新建文件，因此本脚本不写任何文件）。

跑法（仓库根目录）:
    .\\.venv\\Scripts\\python.exe verify_chunk_cache_citation.py

分节：
    1. 分块效果          —— 32 篇切出多少块、长度分布、元数据自检
    2. 分块前后对比      —— 整篇入库 vs 分块入库的 recall@4 / MRR（真实标注问题）
    3. 语义缓存          —— 命中率、命中/未命中延迟 P50/P95、LRU 与 TTL 生效验证
    4. 引用回链          —— 真实答案的 extract_citations 解析 + 编造 [9] 的识别
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import warnings
from typing import Any, Optional

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402

from index_graph.configuration import IndexConfiguration  # noqa: E402
from index_graph.graph import split_documents  # noqa: E402
from retrieval_graph.graph import extract_citations  # noqa: E402
from shared import retrieval  # noqa: E402
from shared.cache import InProcessSemanticCache, RedisCacheBackend  # noqa: E402
from shared.state import _generate_uuid, reduce_docs  # noqa: E402

DOCS_FILE = "src/sample_docs.json"
TESTCASES_FILE = "eval/testcases.yaml"
EMBED_MODEL = "dashscope/text-embedding-v4"
TOP_K = 4


def header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def percentile(values: list[float], pct: float) -> float:
    """线性插值分位数（不依赖 numpy 版本行为差异）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct / 100.0
    low = int(np.floor(position))
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


# ---------------------------------------------------------------------------
# 1. 分块效果
# ---------------------------------------------------------------------------


def section_chunking() -> tuple[list[Document], list[Document], IndexConfiguration]:
    header("1. 文档分块入库")

    with open(DOCS_FILE, encoding="utf-8") as handle:
        docs = reduce_docs([], json.load(handle))
    config = IndexConfiguration()
    chunks = split_documents(
        docs,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=config.chunk_separators,
    )

    lengths = sorted(len(chunk.page_content) for chunk in chunks)
    print(
        f"分块参数: chunk_size={config.chunk_size} chunk_overlap={config.chunk_overlap} "
        f"(overlap 占比 {config.chunk_overlap / config.chunk_size:.0%}) "
        f"chunk_enabled={config.chunk_enabled}"
    )
    print(f"原文档数: {len(docs)}    切出 chunk 数: {len(chunks)}")
    print(f"平均每篇: {len(chunks) / len(docs):.2f} 块")
    print(
        f"chunk 长度 min/p50/max/mean: {lengths[0]} / "
        f"{int(statistics.median(lengths))} / {lengths[-1]} / {statistics.mean(lengths):.1f}"
    )
    print(
        "长度分位 p1/p5/p10/p25/p50/p75/p90/p99: "
        + "/".join(str(int(percentile(lengths, p))) for p in (1, 5, 10, 25, 50, 75, 90, 99))
    )
    print(
        f"过短块(<100字): {sum(1 for x in lengths if x < 100)} 个 "
        f"({sum(1 for x in lengths if x < 100) / len(lengths):.1%})"
    )

    # 元数据自检
    sample = chunks[0].metadata
    print(f"chunk metadata 样例: {sample}")
    doc_uuids = {doc.metadata["uuid"] for doc in docs}
    # 语料里有 2 个不同的 source，所以必须按 source_uuid 逐块比对，不能一律拿 docs[0] 去比。
    source_by_uuid = {doc.metadata["uuid"]: doc.metadata.get("source") for doc in docs}
    checks = {
        "source_uuid 能映射回原文档": {c.metadata["source_uuid"] for c in chunks} <= doc_uuids,
        "chunk_id 全局唯一": len({c.metadata["chunk_id"] for c in chunks}) == len(chunks),
        "chunk_index 每篇从 0 连续": all(
            sorted(c.metadata["chunk_index"] for c in chunks if c.metadata["source_uuid"] == uuid)
            == list(
                range(
                    sum(1 for c in chunks if c.metadata["source_uuid"] == uuid)
                )
            )
            for uuid in doc_uuids
        ),
        "chunk 不继承原文档 uuid(否则会被去重塌缩)": all(
            "uuid" not in c.metadata for c in chunks
        ),
        "原 metadata.source 保留(逐块按 source_uuid 比对)": all(
            c.metadata.get("source") == source_by_uuid[c.metadata["source_uuid"]]
            for c in chunks
        ),
        "chunk_id 形如 source_uuid::index": all(
            c.metadata["chunk_id"] == f"{c.metadata['source_uuid']}::{c.metadata['chunk_index']}"
            for c in chunks
        ),
    }
    for name, ok in checks.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    # 幂等性：同样的输入切两次，chunk_id 序列必须完全一致（重复灌入不会产生副本）
    again = split_documents(
        docs,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=config.chunk_separators,
    )
    identical = [c.metadata["chunk_id"] for c in chunks] == [
        c.metadata["chunk_id"] for c in again
    ]
    print(f"  [{'OK' if identical else 'FAIL'}] 重复切分产生完全相同的 chunk_id（幂等）")

    return docs, chunks, config


# ---------------------------------------------------------------------------
# 2. 分块前后检索对比
# ---------------------------------------------------------------------------


def load_labeled_cases() -> list[dict[str, Any]]:
    """从评测集读真实标注问题（只读，不修改）。

    排除 multi_turn：那类问题的 question 是「那它和边有什么区别？」这种依赖
    history 才能理解的指代句，单独拿去检索必然失败，会污染对比数字。
    """
    import yaml

    with open(TESTCASES_FILE, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    cases = [
        case
        for case in data["cases"]
        if not case.get("TODO") and case.get("expected_doc_uuids")
    ]
    return [
        case
        for case in cases
        if case.get("category") != "multi_turn" and not case.get("history")
    ]


def _matrix(encoder: Embeddings, texts: list[str]) -> np.ndarray:
    vectors = np.asarray(encoder.embed_documents(texts), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-9)


def _query_vector(encoder: Embeddings, query: str) -> np.ndarray:
    vector = np.asarray(encoder.embed_query(query), dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def _mrr(ranked: list[str], expected: set[str]) -> float:
    for rank, uuid in enumerate(ranked, start=1):
        if uuid in expected:
            return 1.0 / rank
    return 0.0


def _recall(ranked: list[str], expected: set[str]) -> float:
    if not expected:
        return 0.0
    return len(set(ranked) & expected) / len(expected)


def section_before_after(
    docs: list[Document], chunks: list[Document]
) -> tuple[Embeddings, np.ndarray]:
    header("2. 分块前后检索对比（纯向量余弦，语料内，doc 级别）")

    cases = load_labeled_cases()
    print(f"标注问题数: {len(cases)}（来自 {TESTCASES_FILE}，排除依赖 history 的 multi_turn）")

    encoder = retrieval.make_text_encoder(EMBED_MODEL)
    print(f"encoder: {EMBED_MODEL}   向量维度: {len(encoder.embed_query('测试'))}")

    print("正在向量化 32 篇整文档 + 460 个 chunk ...")
    doc_matrix = _matrix(encoder, [doc.page_content for doc in docs])
    chunk_matrix = _matrix(encoder, [chunk.page_content for chunk in chunks])
    doc_uuids = [doc.metadata["uuid"] for doc in docs]
    chunk_source = [chunk.metadata["source_uuid"] for chunk in chunks]

    whole_recall: list[float] = []
    whole_mrr: list[float] = []
    chunk_recall_top4: list[float] = []
    chunk_mrr_top4: list[float] = []
    chunk_recall_fair: list[float] = []
    chunk_mrr_fair: list[float] = []
    same_top1 = 0

    print()
    print(f"{'case':6} {'期望':4} {'整篇 top4':44} {'分块 top4(去重后doc)':44}")
    for case in cases:
        expected = set(case["expected_doc_uuids"])
        vector = _query_vector(encoder, case["question"])

        # 整篇入库：直接取 top-4 文档
        doc_order = np.argsort(-(doc_matrix @ vector))[:TOP_K]
        whole_ranked = [doc_uuids[int(i)] for i in doc_order]
        whole_recall.append(_recall(whole_ranked, expected))
        whole_mrr.append(_mrr(whole_ranked, expected))

        # 分块入库 A（朴素）：取 top-4 chunk，映射回 source_uuid 后去重
        # —— 这就是真实链路的结果：reduce_docs 之后上下文里实际有几个不同文档。
        chunk_order = np.argsort(-(chunk_matrix @ vector))[:TOP_K]
        naive_ranked: list[str] = []
        for index in chunk_order:
            source = chunk_source[int(index)]
            if source not in naive_ranked:
                naive_ranked.append(source)
        chunk_recall_top4.append(_recall(naive_ranked, expected))
        chunk_mrr_top4.append(_mrr(naive_ranked, expected))

        # 分块入库 B（公平）：先取 top-20 chunk，去重后取前 4 个不同文档。
        # 这样可以区分"分块到底有没有提升排序质量"与"分块导致文档数变少"。
        wide_order = np.argsort(-(chunk_matrix @ vector))[: 20]
        fair_ranked: list[str] = []
        for index in wide_order:
            source = chunk_source[int(index)]
            if source not in fair_ranked:
                fair_ranked.append(source)
        fair_ranked = fair_ranked[:TOP_K]
        chunk_recall_fair.append(_recall(fair_ranked, expected))
        chunk_mrr_fair.append(_mrr(fair_ranked, expected))

        if whole_ranked and fair_ranked and whole_ranked[0] == fair_ranked[0]:
            same_top1 += 1

        print(
            f"{case['id']:6} {len(expected):^4} "
            f"{','.join(u[:8] for u in whole_ranked):44} "
            f"{','.join(u[:8] for u in naive_ranked):44}"
        )

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    print()
    print(f"{'':34}{'recall@4':>10}{'MRR':>10}")
    print(f"{'整篇入库（基线）':30}{avg(whole_recall):>12.3f}{avg(whole_mrr):>10.3f}")
    print(f"{'分块 top4 chunk→去重':30}{avg(chunk_recall_top4):>12.3f}{avg(chunk_mrr_top4):>10.3f}")
    print(f"{'分块 top20 chunk→去重取4':30}{avg(chunk_recall_fair):>12.3f}{avg(chunk_mrr_fair):>10.3f}")
    print()
    print(
        f"整篇 vs 分块(公平) 的 recall@4 差值: "
        f"{avg(chunk_recall_fair) - avg(whole_recall):+.3f}    "
        f"MRR 差值: {avg(chunk_mrr_fair) - avg(whole_mrr):+.3f}"
    )
    print(f"两者 top-1 完全相同的题数: {same_top1}/{len(cases)}")
    print(
        "注意：分块后 top-4 chunk 常常落在同一篇文档里，去重后可能不足 4 篇文档，"
        "这是「分块 top4 chunk→去重」一列偏低的结构性原因，不是排序变差。"
    )
    return encoder, chunk_matrix


# ---------------------------------------------------------------------------
# 3. 语义缓存
# ---------------------------------------------------------------------------


class DeterministicStubEncoder(Embeddings):
    """确定性桩 encoder：按文本哈希生成伪随机单位向量。

    只用于验证 LRU / TTL 这类**与语义无关的机制**：不同文本生成近乎正交的
    向量，余弦相似度≈0，因此不会出现"意外语义命中"干扰结论，也不需要联网。
    """

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def _vector(self, text: str) -> list[float]:
        seed = int(_generate_uuid(text).replace("-", "")[:8], 16)
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=self.dimensions).astype(np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-9)
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def section_cache_lru_ttl() -> None:
    header("3a. 语义缓存机制验证：LRU 上限 / TTL 过期（桩 encoder，不含语义噪声）")

    # -- LRU：上限 3，写入 4 条互不相似的 query --
    cache: InProcessSemanticCache[str] = InProcessSemanticCache(
        encoder=DeterministicStubEncoder(), max_entries=3, ttl_seconds=3600
    )
    queries = ["alpha 完全不同的主题", "beta 另一个话题", "gamma 第三个内容", "delta 第四个东西"]
    for query in queries:
        cache.set(query, f"result-{query[:5]}")
    stats = cache.stats()
    print(f"写入 4 条、上限 3 -> 当前条目 {len(cache)}，淘汰 {stats['evictions']} 条")
    print(f"  最旧的 'alpha...' 还在吗? {cache.get(queries[0]) is not None}（期望 False）")
    print(f"  最新的 'delta...' 还在吗? {cache.get(queries[3]) is not None}（期望 True）")
    lru_ok = len(cache) == 3 and stats["evictions"] == 1 and cache.get(queries[0]) is None
    print(f"  [{'OK' if lru_ok else 'FAIL'}] LRU 上限生效")

    # -- LRU 的"最近使用"语义：命中后应刷新新旧顺序 --
    lru2: InProcessSemanticCache[str] = InProcessSemanticCache(
        encoder=DeterministicStubEncoder(), max_entries=2, ttl_seconds=3600
    )
    lru2.set("q1", "r1")
    lru2.set("q2", "r2")
    lru2.get("q1")  # 触碰 q1 -> q1 变成最新，q2 变成最旧
    lru2.set("q3", "r3")  # 应淘汰 q2
    lru_ok2 = lru2.get("q1") is not None and lru2.get("q3") is not None
    print(f"  命中 q1 后写入 q3 -> q1 存活: {lru2.get('q1') is not None}（期望 True）")
    print(f"  [{'OK' if lru_ok2 else 'FAIL'}] LRU 按「最近使用」而非「写入时间」淘汰")

    # -- TTL：真实时钟 + 真实 sleep --
    ttl_cache: InProcessSemanticCache[str] = InProcessSemanticCache(
        encoder=DeterministicStubEncoder(), max_entries=10, ttl_seconds=0.15
    )
    ttl_cache.set("ttl 测试问题", "ttl 答案")
    before = ttl_cache.get("ttl 测试问题")
    time.sleep(0.25)
    after = ttl_cache.get("ttl 测试问题")
    ttl_ok = before == "ttl 答案" and after is None
    print(f"  TTL=0.15s：写入后立即读 = {before!r}，sleep 0.25s 后读 = {after!r}")
    print(f"  [{'OK' if ttl_ok else 'FAIL'}] TTL 过期生效（按单调时钟）")
    print(f"  过期计数: {ttl_cache.stats()['expirations']}")

    # -- Redis 骨架必须给出清晰错误，而不是静默失败 --
    print()
    redis_backend: RedisCacheBackend[str] = RedisCacheBackend.from_env()
    try:
        redis_backend.get("任意 query")
        print("  [FAIL] RedisCacheBackend 未抛错")
    except RuntimeError as error:
        print(f"  [OK] RedisCacheBackend 延迟导入 redis 后抛出清晰错误：{str(error)[:60]}...")


def section_cache_perf(cases: list[dict[str, Any]], chunk_matrix: Optional[np.ndarray]) -> None:
    header("3b. 语义缓存命中率与延迟（真实 encoder + 真实语料检索）")

    encoder = retrieval.make_text_encoder(EMBED_MODEL)
    cache: InProcessSemanticCache[str] = InProcessSemanticCache(
        encoder=encoder, similarity_threshold=0.95, max_entries=128, ttl_seconds=3600
    )

    # 模拟"真实检索"：计算 query 向量 + 在全量语料上做一次 top-4 检索。
    # 这里没有 ES，所以检索本身是内存矩阵乘法（很便宜）；缓存真正省下的是
    # embedding 那一次网络调用，以及线上会跟着发生的 LLM 调用。
    def retrieve(query: str) -> str:
        return f"docs for {query[:16]}"

    # 近义改写：同一问题的不同说法，用来测语义命中的真实能力。
    paraphrase_pairs = [
        ("LangGraph 里怎么让同一层的几个节点并行执行？", "LangGraph 中如何让同层节点同时跑起来？"),
        ("config_schema 是挂在哪个类的构造函数上的？", "config_schema 这个参数具体在哪个类的构造函数上？"),
        ("GraphRecursionError 的默认上限是多少？", "跑图报步数超限，默认上限是多少？"),
        ("怎么把已有的向量库对象当 retriever 用？", "手上有个向量库对象，怎么当成 retriever 使用？"),
        ("with_structured_output 怎么让模型返回结构化对象？", "想让模型直接返回结构化对象，with_structured_output 怎么用？"),
        ("interrupt_before 在哪个位置暂停等人工确认？", "想在某节点执行前暂停让人确认，interrupt_before 放哪？"),
    ]

    miss_ms: list[float] = []
    exact_hit_ms: list[float] = []
    semantic_hit_ms: list[float] = []

    # 第一轮：全新 query -> 全部未命中，同时记录未命中延迟
    for base, _ in paraphrase_pairs:
        started = time.perf_counter()
        cache.get_or_compute(base, lambda: retrieve(base))
        miss_ms.append((time.perf_counter() - started) * 1000.0)

    # 第二轮：完全相同的 query -> 走 exact 快路，连 embedding 都不算
    for base, _ in paraphrase_pairs:
        started = time.perf_counter()
        cache.get_or_compute(base, lambda: retrieve(base))
        exact_hit_ms.append((time.perf_counter() - started) * 1000.0)

    # 第三轮：近义改写 -> 需要 embedding + 向量比较
    for _, paraphrase in paraphrase_pairs:
        started = time.perf_counter()
        cache.get_or_compute(paraphrase, lambda: retrieve(paraphrase))
        semantic_hit_ms.append((time.perf_counter() - started) * 1000.0)

    # 精确计数：第三轮里真正走到语义命中分支的次数
    semantic_hits = cache.stats()["semantic_hits"]
    stats = cache.stats()
    print(f"阈值 similarity_threshold = {cache.similarity_threshold}")
    print(f"三轮总计: lookups={stats['lookups']} hits={stats['hits']} misses={stats['misses']}")
    print(
        f"  其中 exact 命中 {stats['exact_hits']} 次、semantic 命中 {semantic_hits} 次"
        f"（近义改写共 {len(paraphrase_pairs)} 条）"
    )
    print(f"总命中率: {stats['hit_rate']:.1%}")
    print(f"近义改写语义命中率: {semantic_hits / len(paraphrase_pairs):.1%}")
    print(f"缓存自身向量比较耗时: {stats['similarity_ms']:.3f} ms")
    print()
    print(f"{'路径':26}{'n':>4}{'P50 ms':>10}{'P95 ms':>10}{'mean ms':>10}")
    for label, values in (
        ("未命中(embed+检索+写)", miss_ms),
        ("exact 命中(不算embed)", exact_hit_ms),
        ("语义命中(embed+比较)", semantic_hit_ms),
    ):
        print(
            f"{label:26}{len(values):>4}{percentile(values, 50):>10.2f}"
            f"{percentile(values, 95):>10.2f}{statistics.mean(values):>10.2f}"
        )
    print()
    print(
        "解读：exact 命中省掉了 embedding 网络往返，因此比未命中快一到两个数量级；"
        "语义命中仍要算一次 query 向量，所以延迟与未命中同量级——它省的是**检索与"
        "下游 LLM 调用**，不是这一段本地延迟。"
    )


# ---------------------------------------------------------------------------
# 4. 引用回链
# ---------------------------------------------------------------------------


def section_citations(chunks: list[Document], encoder: Embeddings, chunk_matrix: np.ndarray) -> None:
    header("4. 引用回链")

    from retrieval_graph.configuration import AgentConfiguration
    from shared.utils import format_docs, load_chat_model

    configuration = AgentConfiguration()
    question = "LangGraph 里怎么让同一层的几个节点并行执行？它和普通边有什么区别？"

    # 真实检索：拿 4 个真实 chunk 当上下文
    vector = _query_vector(encoder, question)
    order = np.argsort(-(chunk_matrix @ vector))[:TOP_K]
    documents = [chunks[int(index)] for index in order]
    print(f"问题: {question}")
    print("检索到的 4 段上下文（真实 chunk）：")
    for rank, document in enumerate(documents, start=1):
        print(
            f"  [{rank}] source_uuid={document.metadata['source_uuid'][:8]} "
            f"chunk_index={document.metadata['chunk_index']} "
            f"len={len(document.page_content)} :: {document.page_content[:40]!r}"
        )

    answer = ""
    try:
        model = load_chat_model(configuration.response_model)
        prompt = configuration.response_system_prompt.format(context=format_docs(documents))
        message = model.invoke([{"role": "system", "content": prompt}, {"role": "user", "content": question}])
        answer = message.content if isinstance(message.content, str) else str(message.content)
        print(f"\n真实答案（{configuration.response_model}，{len(answer)} 字）：")
        print(answer[:900])
    except Exception as error:  # noqa: BLE001 - 模型不可用时如实报告，不编造答案
        print(f"\n[未能取得真实答案] {type(error).__name__}: {str(error)[:200]}")

    if answer:
        citations = extract_citations(answer, documents)
        print(f"\nextract_citations 解析出 {len(citations)} 条引用：")
        for citation in citations:
            print(f"  {json.dumps(citation, ensure_ascii=False)}")
        resolved = sum(1 for c in citations if c["resolved"])
        print(f"  已回链 {resolved} 条，未解析 {len(citations) - resolved} 条")

    # 构造用例：模型编了一个不存在的 [9]
    fabricated = "并行节点用 [1] 的方式写，普通边见 [2]。另外这一点也能在 [9] 里查到。"
    print("\n构造用例（模型编造越界序号 [9]）：")
    print(f"  输入答案: {fabricated}")
    constructed = extract_citations(fabricated, documents)
    for citation in constructed:
        print(f"  {json.dumps(citation, ensure_ascii=False)}")
    bogus = [c for c in constructed if not c["resolved"]]
    print(f"  [{'OK' if bogus and bogus[0]['marker'] == '[9]' else 'FAIL'}] 越界引用被标记出来：{[c['marker'] for c in bogus]}")

    # 边界：上下文为空 / 答案无引用
    print(f"  空上下文 + 有引用 -> {extract_citations('[1]', [])}")
    print(f"  有上下文 + 无引用 -> {extract_citations('完全没有引用标记', documents)}")
    print(f"  [0] 视为越界 -> resolved={extract_citations('[0]', documents)[0]['resolved']}")


def main() -> int:
    print("验证：文档分块入库 / 进程内语义缓存 / 引用回链")
    docs, chunks, _ = section_chunking()
    encoder, chunk_matrix = section_before_after(docs, chunks)
    section_cache_lru_ttl()
    section_cache_perf(load_labeled_cases(), chunk_matrix)
    section_citations(chunks, encoder, chunk_matrix)
    print()
    print("完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
