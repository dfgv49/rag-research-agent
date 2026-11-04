"""混合检索（BM25 + 向量，RRF）+ 重排的 A/B/C 对比测量脚本。

三套配置，把「混合检索」和「重排」的贡献分开量：

    A. baseline      纯向量，k=4  ← 项目当前行为
    B. hybrid        BM25 + 向量 → RRF 融合，候选 20，不重排
    C. hybrid+rerank 同 B，候选 20 → 真实 rerank → top_n=5

**两组语料各跑一遍**（两组指标口径完全相同）：

    组 1  doc 级：`--corpus` 指定的语料（默认 `src/sample_docs.json` 的 32 篇）整篇入库。
          小语料下指标会饱和（篇数少时做 k=4 检索判别力不足），保留它是为了和当前生产
          行为对账。
    组 2  chunk 级：把同一批文档切成 ~400 字 / overlap 50 的 chunk，
          检索单元是 chunk，但**指标仍在文档级计算**：一个 chunk 命中即视为它的
          `source_uuid`（= 原文 md5 uuid）命中。从远多于 k 的候选里取 k 条，指标才有
          区分度。组 2 顺带给出「整篇入库 vs 分块入库」的对比。

指标统一从 `eval/metrics.py` 只读 import（`context_precision` 就是那里的 Average
Precision 口径），保证与项目评测完全同源。

真实数据：`text-embedding-v4`（真实 API）+ `gte-rerank-v2`（真实 API）。
零新增依赖；输出只走 stdout，仓库里不写任何文件。

用法：
    .\\.venv\\Scripts\\python.exe verify_hybrid_ab.py                # 两组都跑
    .\\.venv\\Scripts\\python.exe verify_hybrid_ab.py --skip-doc     # 只跑 chunk 级
    .\\.venv\\Scripts\\python.exe verify_hybrid_ab.py --limit-cases 3 --no-latency   # 冒烟
    .\\.venv\\Scripts\\python.exe verify_hybrid_ab.py --corpus src/sample_docs_large.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Callable, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.join(REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import yaml
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from eval.metrics import context_precision, mrr, ndcg_at_k, recall_at_k
from shared import hybrid
from shared.retrieval import InMemoryHybridRetriever, load_corpus, make_text_encoder
from shared.state import _generate_uuid

#: 默认语料：与历史结果保持可比；用 --corpus 换成别的文件（相对路径按仓库根解析）。
DEFAULT_CORPUS = "src/sample_docs.json"

RECALL_KS = (1, 4, 8, 20)
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
DENSE_CANDIDATES = 20
RERANK_CANDIDATES = 20
RERANK_TOP_N = 5
LATENCY_SIZES = (10, 20)


# ----------------------------------------------------------------------------
# 环境与数据加载
# ----------------------------------------------------------------------------


def load_env(path: str = ".env") -> None:
    """把仓库 `.env` 读进 os.environ（不覆盖已存在的变量），不改文件。

    注意 `.env` 用的是 CR 换行，Python 文本模式的通用换行会把 `\\r` 也当行尾。
    """
    full = os.path.join(REPO_ROOT, path)
    if not os.path.exists(full):
        return
    with open(full, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def resolve_path(path: str) -> str:
    """把语料路径解析成绝对路径：相对路径以仓库根为基准（不依赖当前工作目录）。"""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def display_path(path: str) -> str:
    """打印用路径：仓库内显示相对路径，仓库外（如 $env:TEMP）显示绝对路径。"""
    absolute = os.path.abspath(resolve_path(path))
    try:
        relative = os.path.relpath(absolute, REPO_ROOT)
    except ValueError:
        # 不同盘符（仓库在 D:、TEMP 在 C:）算不出相对路径，直接给绝对路径
        return absolute.replace("\\", "/")
    return absolute.replace("\\", "/") if relative.startswith("..") else relative.replace("\\", "/")


def load_documents(corpus_path: str) -> list[Document]:
    """读取语料，校验「文件存在 / 是 JSON 数组 / 非空」，失败时中文提示退出。

    Args:
        corpus_path: 语料路径（相对路径按仓库根解析）。

    Returns:
        list[Document]: 带 uuid 元数据的整篇文档。

    Raises:
        SystemExit: 任一校验失败时直接退出，不抛裸栈。
    """
    resolved = resolve_path(corpus_path)
    if not os.path.isfile(resolved):
        raise SystemExit(
            f"语料文件不存在: {resolved}\n  提示：用 --corpus <路径> 指定语料（默认 {DEFAULT_CORPUS}）"
        )
    try:
        with open(resolved, encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"语料不是合法 JSON: {resolved}\n  {exc}") from exc
    except OSError as exc:
        raise SystemExit(f"语料读取失败: {resolved}\n  {exc}") from exc
    if not isinstance(raw, list):
        raise SystemExit(f"语料必须是 JSON 数组（每项是一篇文档），实际是 {type(raw).__name__}: {resolved}")
    if not raw:
        raise SystemExit(f"语料为空（0 篇），无法评测: {resolved}")

    documents = load_corpus(resolved)
    if not documents:
        raise SystemExit(f"语料解析后为 0 篇（内容为空或全部重复）: {resolved}")
    return documents


def load_cases(path: str = "eval/testcases.yaml") -> list[dict[str, Any]]:
    """读取评测集：只要 `TODO` 为 false 且带非空 `expected_doc_uuids` 的用例。

    Args:
        path: 测试集 YAML 路径；相对路径按仓库根解析，也支持绝对路径
            （例如 `--testcases "$env:TEMP\\cases_synthetic.yaml"`）。

    Raises:
        SystemExit: 文件不存在 / YAML 解析失败 / 缺 cases 字段时给出中文提示，
            而不是抛裸堆栈。
    """
    full_path = resolve_path(path)
    if not os.path.exists(full_path):
        raise SystemExit(
            f"测试集文件不存在：{full_path}\n"
            "  用 --testcases <路径> 指定其他测试集；"
            "默认是 eval/testcases.yaml（其 uuid 只对应 src/sample_docs.json）。"
        )
    try:
        with open(full_path, encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise SystemExit(f"测试集 YAML 解析失败：{full_path}\n  {exc}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise SystemExit(
            f"测试集格式不对：{full_path}\n"
            "  期望顶层是 mapping 且含 cases 列表（与 eval/testcases.yaml 同结构）。"
        )

    cases = []
    for case in payload["cases"]:
        if case.get("TODO", False):
            continue
        expected = [str(x) for x in (case.get("expected_doc_uuids") or []) if x]
        if not expected:
            continue
        cases.append(
            {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "history": case.get("history") or [],
                "expected": expected,
            }
        )
    return cases


def build_query(case: dict[str, Any]) -> str:
    """构造检索用的查询串（A/B/C 三配置完全一致）。

    多轮用例的 `question` 是纯指代（"那它和边有什么区别？"），单独送去检索必然
    全零；这里把**上一轮对话**截断后拼在前面，作为「不用 LLM 的朴素改写」。这样做是
    为了 (a) 确定性、(b) 零额外 LLM 调用、(c) 三个配置拿到完全相同的输入，差异只
    来自检索与重排本身。
    """
    history = case.get("history") or []
    if not history:
        return case["question"]
    last = " ".join(str(history[-1]).split())[:300]
    return f"{last}\n{case['question']}"


def chunk_corpus(
    documents: Sequence[Document],
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chunk_chars: int = 30,
) -> list[Document]:
    """把整篇文档切成 chunk，并在 metadata 里保留 `source_uuid`。

    切分用已安装的 `langchain_text_splitters.RecursiveCharacterTextSplitter`（不是
    新依赖）。分隔符按「先段落、再句子（中英文标点都列上）、最后硬切」的顺序，
    这样中文和英文混排的语料都不会被切在句子中间。

    切分器会把 `### Send¶` 这类小标题、甚至一个 `.` 切成独立片段（实测 444 个里
    有 9 个短于 30 字）。这些片段本身不是可检索的知识单元，反而会污染 BM25 的
    文档长度统计，因此短于 `min_chunk_chars` 的片段会被并回前一个 chunk。

    Args:
        documents: 原始整篇文档（须带 uuid 元数据）。
        chunk_size: 目标 chunk 长度（字符）。
        chunk_overlap: 相邻 chunk 的重叠字符数。
        min_chunk_chars: 低于该长度的片段并回前一个 chunk。

    Returns:
        list[Document]: chunk 列表，每个 chunk 的 metadata 含
        `uuid`（chunk 自身 ID）、`source_uuid`（原文档 uuid）、`chunk_index`。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        keep_separator=True,
        separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "，", ", ", " ", ""],
    )
    chunks: list[Document] = []
    for document in documents:
        source_uuid = document.metadata["uuid"]
        pieces: list[str] = []
        for text in splitter.split_text(document.page_content):
            if not text.strip():
                continue
            if pieces and len(text.strip()) < min_chunk_chars:
                pieces[-1] = f"{pieces[-1]}{text}"
            else:
                pieces.append(text)
        for index, text in enumerate(pieces):
            metadata = {
                "uuid": _generate_uuid(f"{source_uuid}|{index}|{text}"),
                "source_uuid": source_uuid,
                "chunk_index": index,
                "source": document.metadata.get("source", ""),
            }
            chunks.append(Document(page_content=text, metadata=metadata))
    return chunks


# ----------------------------------------------------------------------------
# 向量缓存（让三配置共用完全相同的向量，同时把请求量压到最低）
# ----------------------------------------------------------------------------


class CachedEmbeddings(Embeddings):
    """给任意 Embeddings 加一层进程内缓存，并在**算向量之前**按字符上限截断。

    这是**测量脚本专用**：同一段文本重复 embedding 结果不变，缓存它既不影响任何
    指标，又能让 A/B/C 三配置严格共用同一批向量，还省掉几十次 HTTP 往返。

    截断说明：`text-embedding-v4` 对单次输入长度有上限（约 33000），超了直接报
    `<400> InvalidParameter: Range of input length should be [1, 33000]`。
    大语料 `src/sample_docs_large.json` 最长 136352 字符，不截断根本建不了库。
    截断只作用于**送进 embedding 的文本**；缓存仍按原文做键，
    检索 / 重排用的 `document.page_content` 依旧是全文。
    """

    def __init__(self, inner: Embeddings, max_chars: int = 0) -> None:
        self.inner = inner
        self.max_chars = max_chars if max_chars and max_chars > 0 else 0
        self._docs: dict[str, list[float]] = {}
        self._queries: dict[str, list[float]] = {}
        self.document_calls = 0
        self.query_calls = 0
        #: 本次运行中有多少条文本在嵌入前被截断
        self.truncated_texts = 0

    def _clip(self, text: str) -> str:
        """按码位截断（Python str 切片不会切开中文 / emoji）。"""
        if self.max_chars and len(text) > self.max_chars:
            return text[: self.max_chars]
        return text

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        missing = [text for text in texts if text not in self._docs]
        if missing:
            self.document_calls += 1
            payload = []
            for text in missing:
                clipped = self._clip(text)
                if len(clipped) != len(text):
                    self.truncated_texts += 1
                payload.append(clipped)
            for text, vector in zip(missing, self.inner.embed_documents(payload)):
                self._docs[text] = vector
        return [self._docs[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if text not in self._queries:
            self.query_calls += 1
            self._queries[text] = self.inner.embed_query(self._clip(text))
        return self._queries[text]


# ----------------------------------------------------------------------------
# 三配置检索
# ----------------------------------------------------------------------------


def doc_ids(hits: Sequence[Any]) -> list[str]:
    """把命中映射成「文档级」ID 并保序去重（chunk 映射回 source_uuid）。"""
    ordered: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        metadata = hit.document.metadata
        doc_id = str(metadata.get("source_uuid") or metadata.get("uuid") or "")
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        ordered.append(doc_id)
    return ordered


def run_configs(
    retriever: InMemoryHybridRetriever,
    query: str,
    api_key: Optional[str],
    rerank_stats: dict[str, Any],
) -> dict[str, Any]:
    """对一条查询跑完 A / B / C，返回三份「文档级有序 ID 列表」。

    同时把 C 用来重排的候选文本原样带出来，供延迟测量复用（不再重复召回）。
    """
    # A. baseline：纯向量，k=4（= 项目当前行为）
    a_hits = retriever.dense_search(query, 4)

    # B. hybrid：两路召回 → RRF 融合（候选 20 路）
    b_result = retriever.search(
        query, dense_k=DENSE_CANDIDATES, sparse_k=DENSE_CANDIDATES
    )

    # C. hybrid + rerank：候选 20 → 真实重排 → top_n=5
    c_result = retriever.search(
        query,
        dense_k=DENSE_CANDIDATES,
        sparse_k=DENSE_CANDIDATES,
        rerank=True,
        rerank_candidates=RERANK_CANDIDATES,
        rerank_top_n=RERANK_TOP_N,
        rerank_api_key=api_key,
    )
    if c_result.stats.get("rerank_applied"):
        rerank_stats["applied"] += 1
    else:
        rerank_stats["degraded"] += 1
        error = str(c_result.stats.get("rerank_error") or "")
        rerank_stats["errors"][error] = rerank_stats["errors"].get(error, 0) + 1

    return {
        "A": doc_ids(a_hits),
        "B": doc_ids(b_result.hits),
        "C": doc_ids(c_result.hits),
        "sparse_hits": b_result.stats.get("sparse_hits", 0),
        "dense_hits": b_result.stats.get("dense_hits", 0),
        "fused_len": len(b_result.hits),
        "rerank_candidates": c_result.stats.get("rerank_candidates", 0),
        "b_texts": [hit.document.page_content for hit in b_result.hits],
    }


def score(ranked_ids: Sequence[str], expected: Sequence[str]) -> dict[str, Optional[float]]:
    """按 `eval/metrics.py` 的口径算一条用例的全部指标。"""
    metrics: dict[str, Optional[float]] = {}
    for k in RECALL_KS:
        metrics[f"recall@{k}"] = recall_at_k(list(ranked_ids), list(expected), k)
    metrics["mrr"] = mrr(list(ranked_ids), list(expected))
    metrics["ndcg@10"] = ndcg_at_k(list(ranked_ids), list(expected), 10)
    metrics["context_precision"] = context_precision(list(ranked_ids), list(expected))
    return metrics


METRIC_KEYS = [f"recall@{k}" for k in RECALL_KS] + [
    "mrr",
    "ndcg@10",
    "context_precision",
]


def mean_of(values: list[Optional[float]]) -> Optional[float]:
    """忽略 None 的算术平均（None = 该条用例指标未定义，不能当 0 分）。"""
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def aggregate(
    rows: Sequence[dict[str, Any]], configs: Sequence[str]
) -> list[dict[str, Optional[float]]]:
    """把逐用例结果按配置聚合成均值行。"""
    out = []
    for config in configs:
        row: dict[str, Optional[float]] = {"config": config}
        for metric in METRIC_KEYS:
            row[metric] = mean_of([r["metrics"][config][metric] for r in rows])
        out.append(row)
    return out


# ----------------------------------------------------------------------------
# 输出
# ----------------------------------------------------------------------------


def fmt(value: Optional[float], width: int = 12) -> str:
    if value is None:
        return "n/a".rjust(width)
    return f"{value:.4f}".rjust(width)


def print_table(
    title: str, rows: Sequence[dict[str, Any]], configs: Sequence[str]
) -> None:
    """打印「配置 × 指标」表。"""
    print(f"\n{title}")
    header = (
        "config".ljust(20)
        + "".join(m.rjust(17) for m in METRIC_KEYS)
    )
    print(header)
    print("-" * len(header))
    labels = {
        "A": "A vector k=4",
        "B": "B hybrid RRF k=20",
        "C": "C hybrid+rerank",
    }
    for row in rows:
        line = labels.get(row["config"], row["config"]).ljust(20)
        for metric in METRIC_KEYS:
            line += fmt(row[metric], 17)
        print(line)


def print_latency(title: str, samples: dict[int, list[float]], errors: int) -> None:
    """打印 rerank 延迟的 min / p50 / max。"""
    print(f"\n{title}")
    print("candidates     n      min_ms     p50_ms     max_ms    mean_ms   errors")
    print("-" * 76)
    for size in sorted(samples):
        values = sorted(samples[size])
        if not values:
            print(f"{size:<12} {0:>5}        n/a        n/a        n/a        n/a  {errors}")
            continue
        print(
            f"{size:<12} {len(values):>5} {values[0]:>10.1f} {statistics.median(values):>10.1f}"
            f" {values[-1]:>10.1f} {statistics.fmean(values):>10.1f} {errors:>7}"
        )


# ----------------------------------------------------------------------------
# 一组语料的完整评测
# ----------------------------------------------------------------------------


def evaluate(
    name: str,
    retriever: InMemoryHybridRetriever,
    cases: Sequence[dict[str, Any]],
    api_key: Optional[str],
    measure_latency: bool,
) -> dict[str, Any]:
    """在给定语料（doc 级或 chunk 级）上跑完 A/B/C 并打印全部表格。"""
    configs = ("A", "B", "C")
    rerank_stats: dict[str, Any] = {"applied": 0, "degraded": 0, "errors": {}}
    rows: list[dict[str, Any]] = []
    latency: dict[int, list[float]] = {size: [] for size in LATENCY_SIZES}
    latency_errors = 0

    for case in cases:
        query = build_query(case)
        outcome = run_configs(retriever, query, api_key, rerank_stats)
        entry = {
            "id": case["id"],
            "category": case["category"],
            "ranked": {config: outcome[config] for config in configs},
            "metrics": {
                config: score(outcome[config], case["expected"]) for config in configs
            },
            "sparse_hits": outcome["sparse_hits"],
            "dense_hits": outcome["dense_hits"],
            "fused_len": outcome["fused_len"],
        }
        rows.append(entry)

        if measure_latency:
            for size in LATENCY_SIZES:
                texts = outcome["b_texts"][:size]
                if len(texts) < size:
                    continue
                started = time.perf_counter()
                results, error = hybrid.rerank(
                    query,
                    texts,
                    model="gte-rerank-v2",
                    top_n=RERANK_TOP_N,
                    timeout=30,
                    api_key=api_key,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if results is None:
                    latency_errors += 1
                    print(f"  [warn] {case['id']} rerank@{size} 失败: {error}")
                else:
                    latency[size].append(elapsed_ms)

    print(f"\n{'=' * 78}\n[{name}] 语料规模={len(retriever.documents)}  用例数={len(cases)}\n{'=' * 78}")

    print_table(f"[{name}] 总体指标（n={len(rows)}）", aggregate(rows, configs), configs)

    categories = sorted({row["category"] for row in rows})
    for category in categories:
        subset = [row for row in rows if row["category"] == category]
        print_table(
            f"[{name}] category={category}（n={len(subset)}）",
            aggregate(subset, configs),
            configs,
        )

    # 逐用例明细：能看到「同一用例 A/B/C 到底差在哪」。
    print(f"\n[{name}] 逐用例明细（recall@4 / ndcg@10，括号里是 sparse 路命中数）")
    print("case           category         A_r@4   B_r@4   C_r@4  A_ndcg  B_ndcg  C_ndcg  sparse")
    print("-" * 92)
    for row in rows:
        def pick(config: str, metric: str) -> Optional[float]:
            return row["metrics"][config][metric]

        print(
            f"{row['id']:<14} {row['category']:<15}"
            f" {fmt(pick('A', 'recall@4'), 7)} {fmt(pick('B', 'recall@4'), 7)} {fmt(pick('C', 'recall@4'), 7)}"
            f" {fmt(pick('A', 'ndcg@10'), 7)} {fmt(pick('B', 'ndcg@10'), 7)} {fmt(pick('C', 'ndcg@10'), 7)}"
            f" {row['sparse_hits']:>7}"
        )

    sparse_cases = sum(1 for row in rows if row["sparse_hits"] > 0)
    print(
        f"\n[{name}] 稀疏路诊断: {sparse_cases}/{len(rows)} 条用例 BM25 有命中；"
        f"融合候选数中位数={statistics.median([row['fused_len'] for row in rows]):.0f}；"
        f"rerank 生效 {rerank_stats['applied']} 次 / 降级 {rerank_stats['degraded']} 次"
    )
    for error, count in rerank_stats["errors"].items():
        print(f"  [降级原因 x{count}] {error}")

    if measure_latency:
        print_latency(
            f"[{name}] rerank 延迟（真实 gte-rerank-v2，候选 10 / 20 篇）",
            latency,
            latency_errors,
        )

    # 两个方法学补充：先定位瓶颈在哪一环，再判断差值是否显著。
    print_ceiling(name, rows, cases)
    print_significance(name, rows)

    return {
        "name": name,
        "rows": rows,
        "overall": aggregate(rows, configs),
        "by_category": {c: aggregate([r for r in rows if r["category"] == c], configs) for c in categories},
        "latency": latency,
        "rerank_stats": rerank_stats,
        "sparse_cases": sparse_cases,
    }


def _eval_stats():
    """按需导入 eval/stats.py（避免顶层 import 影响 --help 的启动速度）。"""
    import sys as _sys

    eval_dir = os.path.join(REPO_ROOT, "eval")
    if eval_dir not in _sys.path:
        _sys.path.insert(0, eval_dir)
    import stats as _stats  # noqa: PLC0415

    return _stats


def print_ceiling(name: str, rows: Sequence[dict[str, Any]], cases: Sequence[dict[str, Any]]) -> None:
    """召回天花板分析：区分「池里没有」与「池里有但重排丢了」。

    为什么需要这一步（对齐 BEIR / MS MARCO 的分阶段评测）：
        只报最终 top-n 的指标，无法判断差在哪一环。正确做法是先算
        **候选池的可达召回**（Recall@候选数），再看最终输出保留了多少。
        池里没有正确文档 → 任何重排都不可能救回，这是硬上限。

    C 的候选池就是 B 的完整输出（融合后 ≤20 条），所以：
        ceiling = 在 B 的整个列表上算出的 recall
        achieved = 在 C 的最终列表上算出的 recall
        lost = ceiling − achieved > 0 表示「重排把池里已有的正确文档丢了」
    """
    by_id = {case["id"]: case for case in cases}
    ceilings: list[float] = []
    achieved: list[float] = []
    lost_cases: list[str] = []
    unreachable: list[str] = []

    for row in rows:
        case = by_id.get(row["id"])
        if not case:
            continue
        expected = list(case.get("expected") or [])
        if not expected:
            continue
        b_list = row["ranked"]["B"]
        c_list = row["ranked"]["C"]
        if not b_list:
            continue
        ceil = recall_at_k(b_list, expected, len(b_list))
        got = recall_at_k(c_list, expected, max(len(c_list), 1))
        if ceil is None or got is None:
            continue
        ceilings.append(ceil)
        achieved.append(got)
        if ceil == 0:
            unreachable.append(row["id"])
        elif got < ceil - 1e-9:
            lost_cases.append(row["id"])

    n = len(ceilings)
    if not n:
        return

    mean_ceil = sum(ceilings) / n
    mean_got = sum(achieved) / n
    print(f"\n[{name}] 召回天花板分析（C 的候选池 = B 的完整输出）")
    print("-" * 78)
    print(f"  候选池可达召回 ceiling(B 全列表) : {mean_ceil:.4f}   ← 任何重排的理论上限")
    print(f"  重排后实际保留   achieved(C 全列表): {mean_got:.4f}")
    print(f"  保留率 achieved/ceiling           : {(mean_got / mean_ceil):.2%}" if mean_ceil else "")
    print(f"  池外不可救（候选里根本没有正确文档）: {len(unreachable)} 条 / {n}  → 上限天然被压低")
    print(f"  池里有但被重排丢掉               : {len(lost_cases)} 条"
          + (f"  例: {', '.join(lost_cases[:5])}" if lost_cases else ""))


def print_significance(name: str, rows: Sequence[dict[str, Any]]) -> None:
    """配对 bootstrap：判断 A/B/C 之间的差值是否显著（而非随机波动）。

    为什么用配对：三组配置跑的是同一批用例，逐条做差能消掉「题目难度差异」，
    在同样样本量下比「两组独立比较」灵敏得多。
    """
    stats = _eval_stats()
    metrics = ("recall@4", "recall@8", "mrr", "ndcg@10", "context_precision")
    pairs = (("B", "A"), ("C", "B"), ("C", "A"))

    print(f"\n[{name}] 统计显著性（配对 bootstrap，2000 次重采样，95% CI）")
    print("-" * 78)
    for left, right in pairs:
        print(f"  {left} − {right}")
        for metric in metrics:
            a_vals = [row["metrics"][right][metric] for row in rows]
            b_vals = [row["metrics"][left][metric] for row in rows]
            print("    " + stats.paired_summary(a_vals, b_vals, metric=metric))


def delta(new: dict[str, Any], old: dict[str, Any], metric: str) -> Optional[float]:
    """两个聚合行在某个指标上的差值（任一为 None 则返回 None）。"""
    a = new.get(metric)
    b = old.get(metric)
    if a is None or b is None:
        return None
    return a - b


def print_conclusion(results: Sequence[dict[str, Any]]) -> None:
    """把两组结果汇成一句句可核对的三句话结论。"""
    print(f"\n{'=' * 78}\n结论\n{'=' * 78}")
    for result in results:
        overall = {row["config"]: row for row in result["overall"]}
        name = result["name"]
        print(f"\n[{name}]")
        for metric in ("recall@4", "recall@8", "mrr", "ndcg@10", "context_precision"):
            b_minus_a = delta(overall["B"], overall["A"], metric)
            c_minus_b = delta(overall["C"], overall["B"], metric)
            print(
                f"  {metric:<18} A={fmt(overall['A'][metric], 8)} B={fmt(overall['B'][metric], 8)}"
                f" C={fmt(overall['C'][metric], 8)}"
                f"  (B-A={('n/a' if b_minus_a is None else f'{b_minus_a:+.4f}')},"
                f" C-B={('n/a' if c_minus_b is None else f'{c_minus_b:+.4f}')})"
            )


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="混合检索（BM25+向量 / RRF）+ 重排的 A/B/C 对比测量",
    )
    parser.add_argument(
        "--corpus",
        default=DEFAULT_CORPUS,
        help=(
            f"语料文件路径（默认 {DEFAULT_CORPUS}；相对路径按仓库根解析）。"
            "换大语料时指过去，例如 --corpus src/sample_docs_large.json"
        ),
    )
    parser.add_argument(
        "--testcases",
        default="eval/testcases.yaml",
        help=(
            "测试集路径（默认 eval/testcases.yaml；相对路径按仓库根解析，也接受绝对路径）。"
            "注意 eval/testcases.yaml 的 expected_doc_uuids 只对应旧 32 篇语料 "
            "src/sample_docs.json；换大语料时必须同时换一份匹配的测试集，例如 "
            '--testcases "$env:TEMP\\cases_synthetic.yaml"'
        ),
    )
    parser.add_argument(
        "--embed-cache",
        default=None,
        metavar="PATH",
        help=(
            "启用磁盘级 embedding 缓存（推荐）。缓存 key 含模型名+维度，换模型不会误命中。"
            "229 篇语料首次约 8~10 分钟，之后重跑几乎免费。不传则每次重新计算。"
        ),
    )
    parser.add_argument(
        "--max-embed-chars",
        type=int,
        default=8000,
        dest="max_embed_chars",
        help=(
            "嵌入时单条文本的字符上限（默认 8000；传 0 表示不截断）。"
            "大语料最长 136352 字符，会超出 text-embedding-v4 的输入上限而报 400，"
            "用它才能建库；截断只作用于 embedding，检索用的正文仍是全文。"
        ),
    )
    parser.add_argument(
        "--limit-cases",
        type=int,
        default=0,
        dest="limit_cases",
        help="只跑前 N 条用例，用于先冒烟（默认 0 = 全跑）",
    )
    parser.add_argument(
        "--skip-doc",
        action="store_true",
        dest="skip_doc",
        help="只跑 chunk 级（跳过 doc 级）",
    )
    parser.add_argument(
        "--skip-chunk",
        action="store_true",
        dest="skip_chunk",
        help=(
            "只跑 doc 级（跳过 chunk 级）。chunk 级会把语料切成数千块，"
            "大语料下按当前 embedding 速率要跑很久；与 --skip-doc 同时给出时报错。"
        ),
    )
    parser.add_argument(
        "--no-latency",
        action="store_true",
        dest="no_latency",
        help="跳过 rerank 延迟测量（少发请求，冒烟更快）",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    load_env()
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    skip_doc = bool(args.skip_doc)
    skip_chunk = bool(getattr(args, "skip_chunk", False))
    no_latency = bool(args.no_latency)

    if skip_doc and skip_chunk:
        raise SystemExit(
            "--skip-doc 与 --skip-chunk 不能同时使用：\n"
            "  --skip-doc   只跑 chunk 级\n"
            "  --skip-chunk 只跑 doc 级\n"
            "  两个一起给等于两组都不跑，没有任何可比较的内容。"
        )

    print("=" * 78)
    print("混合检索（BM25+向量 / RRF）+ 重排 A/B/C 对比")
    print("=" * 78)
    print(f"embedding : dashscope/text-embedding-v4 (dim=1024)   key={'OK' if api_key else 'MISSING'}")
    print(f"rerank    : {hybrid.DEFAULT_RERANK_MODEL} @ {hybrid.DEFAULT_RERANK_URL}")
    print(f"rrf       : rank_constant=60, dense_weight=1.0, sparse_weight=1.0")
    print(f"configs   : A=vector k=4 | B=hybrid k={DENSE_CANDIDATES} | C=hybrid {RERANK_CANDIDATES}->rerank->{RERANK_TOP_N}")
    print(f"chunking  : size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")

    documents = load_documents(args.corpus)
    cases = load_cases(args.testcases)
    if args.limit_cases and args.limit_cases > 0:
        cases = cases[: args.limit_cases]
        print(f"limit     : 只跑前 {len(cases)} 条用例（--limit-cases {args.limit_cases}）")
    category_counts: dict[str, int] = {}
    for case in cases:
        category_counts[case["category"]] = category_counts.get(case["category"], 0) + 1
    print(f"corpus    : {len(documents)} 篇（doc 级）  <- {display_path(args.corpus)}")
    print(f"testcases : {len(cases)} 条可用（{' / '.join(f'{k}={v}' for k, v in sorted(category_counts.items()))}）")

    known = {document.metadata["uuid"] for document in documents}
    missing = [
        (case["id"], uuid)
        for case in cases
        for uuid in case["expected"]
        if uuid not in known
    ]
    print(f"标注校验  : expected_doc_uuids 全部能在语料里找到 —— "
          f"{'是' if not missing else f'否（{len(missing)} 条对不上，前 3 条 {missing[:3]}）'}")

    base_encoder = make_text_encoder("dashscope/text-embedding-v4")
    if args.embed_cache:
        import sys as _sys

        eval_dir = os.path.join(REPO_ROOT, "eval")
        if eval_dir not in _sys.path:
            _sys.path.insert(0, eval_dir)
        from embed_cache import DiskCachedEmbeddings  # noqa: PLC0415

        base_encoder = DiskCachedEmbeddings(base_encoder, cache_path=args.embed_cache)
        print(f"embedding 缓存: {args.embed_cache}")
    # 缓存包在截断内层：key 按「最终送入 embedding 的文本」算，与 run_eval 一致
    encoder = CachedEmbeddings(base_encoder, max_chars=args.max_embed_chars)
    results = []

    if not skip_doc:
        doc_retriever = InMemoryHybridRetriever(documents, encoder)
        started = time.perf_counter()
        doc_retriever._corpus_matrix()
        print(f"\n[doc 级] 语料向量化完成：{len(documents)} 篇 / {time.perf_counter() - started:.1f}s"
              f"（embed_documents 调用 {encoder.document_calls} 次）")
        cap_desc = f"上限 {encoder.max_chars} 字符" if encoder.max_chars else "未启用"
        print(f"  嵌入截断：{cap_desc}，{encoder.truncated_texts} 条文本被截断"
              f"（仅对嵌入生效，检索用的正文仍是全文）")
        results.append(
            evaluate(f"组1 doc级({len(documents)}篇)", doc_retriever, cases, api_key, not no_latency)
        )

    if skip_chunk:
        print("\n[chunk 级] 已跳过（--skip-chunk）：本次只跑组1 doc 级。")
    else:
        chunks = chunk_corpus(documents)
        chunk_retriever = InMemoryHybridRetriever(chunks, encoder)
        started = time.perf_counter()
        chunk_retriever._corpus_matrix()
        print(
            f"\n[chunk 级] 切分得到 {len(chunks)} 个 chunk；"
            f"向量化 {time.perf_counter() - started:.1f}s"
            f"（embed_documents 调用 {encoder.document_calls} 次）"
        )
        results.append(
            evaluate(f"组2 chunk级({len(chunks)}个)", chunk_retriever, cases, api_key, not no_latency)
        )

    # 分块前后对比：同一份 A 配置（纯向量 k=4）在两种语料上的差别。
    if len(results) == 2:
        print(f"\n{'=' * 78}\n分块前 vs 分块后（同一配置 A：纯向量 k=4）\n{'=' * 78}")
        doc_overall = {row["config"]: row for row in results[0]["overall"]}
        chunk_overall = {row["config"]: row for row in results[1]["overall"]}
        print(
            "metric".ljust(20)
            + f"整篇入库({len(documents)})".rjust(17)
            + f"分块入库({len(chunks)})".rjust(17)
            + "差值".rjust(17)
        )
        print("-" * 71)
        for metric in METRIC_KEYS:
            before = doc_overall["A"][metric]
            after = chunk_overall["A"][metric]
            diff = None if before is None or after is None else after - before
            print(
                metric.ljust(20)
                + fmt(before, 17)
                + fmt(after, 17)
                + ("n/a".rjust(17) if diff is None else f"{diff:+.4f}".rjust(17))
            )
        print(
            "说明：分块后 k=4 指的是 4 个 chunk，映射回文档后可能少于 4 篇；"
            "这就是分块检索的固有取舍（精度换粒度）。"
        )

    print_conclusion(results)
    print(
        "\n注：以上全部是本次实跑的真实数字；rerank / embedding 的失败会以"
        "「降级次数 + 原因」原样打印，不做任何修数。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
