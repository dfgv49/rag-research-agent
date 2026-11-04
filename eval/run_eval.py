"""一条命令跑完整评测，产出人读的 .md 与机读的 .json 基线报告。

用法（Windows / pwsh，工作目录 = 项目根目录）:

    # 全量：检索指标 + 生成侧 judge（约 3~5 分钟，消耗对话模型额度）
    .\\.venv\\Scripts\\python.exe eval\\run_eval.py --output-dir "$env:TEMP\\rag_eval_reports"

    # 只跑检索指标（不调用对话模型，快）
    .\\.venv\\Scripts\\python.exe eval\\run_eval.py --skip-judge --output-dir "$env:TEMP\\rag_eval_reports"

    # 开启查询改写，并在报告里给出带 history 用例的改写前后对比
    .\\.venv\\Scripts\\python.exe eval\\run_eval.py --rewrite --output-dir "$env:TEMP\\rag_eval_reports"

    # 冒烟
    .\\.venv\\Scripts\\python.exe eval\\run_eval.py --limit 6 --skip-judge

    # 换语料（默认 src/sample_docs.json，可用 --corpus 指向大语料）
    .\\.venv\\Scripts\\python.exe eval\\run_eval.py --corpus src\\sample_docs_large.json --skip-judge --output-dir "$env:TEMP\\rag_eval_reports"

--------------------------------------------------------------------------
为什么命令里带 --output-dir "$env:TEMP\\rag_eval_reports"
--------------------------------------------------------------------------
在某些 DSH/沙箱配置下，Python 子进程**无法在项目目录内新建文件或目录**
（`PermissionError` / `WinError 5`），但写系统临时目录正常。此时默认的
`eval/reports` 会写失败。把输出目录指到 `$env:TEMP\\rag_eval_reports` 即可绕开：
报告内容完全一样，只是落盘位置不同。没有这个限制的环境可以省略该参数。

--------------------------------------------------------------------------
为什么用内存向量库
--------------------------------------------------------------------------
项目默认走 Elasticsearch（config.retriever_provider = elastic-local），但评测环境里
Docker 未运行、ES 连不上。为保证评测可复现、不依赖外部服务，这里复用 eval_recall.py
的做法：用 langchain_core 的 InMemoryVectorStore 索引语料文件（默认 src/sample_docs.json，
可用 --corpus 换成别的语料，例如大语料 src/sample_docs_large.json）的整篇文档。
embedding 仍然走项目真实的 make_text_encoder（dashscope/text-embedding-v4，
dimensions=1024、chunk_size=20），文档切分与 uuid 走项目真实的 reduce_docs，
所以「检索算法」层面与线上一致，差别只在存储后端（见 README 已知限制）。

--------------------------------------------------------------------------
为什么生成侧不跑完整 graph
--------------------------------------------------------------------------
完整主图需要：路由分类 → 拆研究计划 → 扩写检索词 → Send 并行检索 → 合并 → 带引用作答。
它强依赖 ES 检索器，跑不通；而且失败时会混淆「规划/路由写错了」和「答案质量差」两件事，
无法归因。评测要的是「当前检索 + 作答链路在给定语料上的表现」，因此这里选一条简单可靠
的路径：直接把 top-k context + 问题交给 qwen 作答，再用 judge 打分。
这样检索指标与生成指标各自可归因，且整条链路无外部服务依赖。

--------------------------------------------------------------------------
与 src/ 的关系
--------------------------------------------------------------------------
本文件只 import src/shared/ 下的基础能力（embedding 工厂、配置、reduce_docs），
**刻意不 import src/retrieval_graph/**，以免与并行修改该模块的其他人互相干扰。
查询改写的提示词也写在本文件里。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from collections import Counter, OrderedDict
from datetime import datetime
from typing import Any, Optional, Sequence

# --------------------------------------------------------------------------- #
# 路径与依赖
# --------------------------------------------------------------------------- #
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EVAL_DIR)

# 让 `import metrics / judge` 在「直接 python eval/run_eval.py」和
# 「python -m eval.run_eval」两种执行方式下都成立。
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

# 让项目自己的 src/shared 可以被 import（与 eval_recall.py 一致）
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dotenv import load_dotenv  # noqa: E402

# 显式指定 .env 路径：不依赖当前工作目录，避免从别处运行时读不到 API Key。
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(ENV_PATH)

import metrics as M  # noqa: E402
from judge import (  # noqa: E402
    DEFAULT_ANSWER_MODELS,
    DEFAULT_JUDGE_MODELS,
    DEFAULT_REWRITE_MODELS,
    Judge,
    LLMChannel,
)

#: 默认语料。默认值必须保持不变，否则历史基线数字不可比；用 --corpus 换成别的文件。
DOCS_FILE = os.path.join(SRC_DIR, "sample_docs.json")
TESTCASES_FILE = os.path.join(EVAL_DIR, "testcases.yaml")
DEFAULT_OUTPUT_DIR = os.path.join(EVAL_DIR, "reports")


def display_path(path: str) -> str:
    """报告里显示路径：项目内的显示相对路径，项目外（如 $env:TEMP）显示绝对路径。"""
    absolute = os.path.abspath(path)
    try:
        relative = os.path.relpath(absolute, PROJECT_ROOT)
    except ValueError:
        # 不同盘符（工作区在 D:、TEMP 在 C:）算不出相对路径，直接给绝对路径
        return absolute.replace("\\", "/")
    if relative.startswith(".."):
        return absolute.replace("\\", "/")
    return relative.replace("\\", "/")


#: 检索深度：一次取这么多，再截断出 Recall@1/4/8/20
RETRIEVAL_DEPTH = max(M.DEFAULT_RECALL_KS)

#: 生成侧实际喂给模型几条 context（与项目默认 search_kwargs.k 对齐）
GENERATION_TOP_K = 4

#: 判定「拒答」的关键词（no_answer 类别用，启发式，不是模型判断）
REFUSAL_MARKERS = (
    "没有找到",
    "未找到",
    "没有提到",
    "未提及",
    "没有提及",
    "资料中没有",
    "文中没有",
    "语料没有",
    "不包含",
    "无法回答",
    "无法确定",
    "没有相关信息",
    "not found",
    "no information",
    "not mentioned",
    "does not mention",
    "cannot answer",
    "no relevant",
)


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
def load_testcases(path: str) -> list[dict[str, Any]]:
    """读取 testcases.yaml（pyyaml 已安装，因此用 YAML 而不是降级成 JSON）。"""
    import yaml

    with open(path, encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    cases = spec.get("cases") if isinstance(spec, dict) else None
    if not cases:
        raise SystemExit(f"测试集为空或格式不对: {path}")
    return cases


def load_corpus_payload(path: str) -> list[Any]:
    """读取语料 JSON，并校验「文件存在 / 是 JSON 数组 / 非空」。

    Args:
        path: 语料路径（相对路径按当前工作目录解析，与 open 的行为一致）。

    Returns:
        list: 语料原始数组。

    Raises:
        SystemExit: 任一校验失败时，带中文提示直接退出（不抛裸栈）。
    """
    if not os.path.isfile(path):
        raise SystemExit(
            f"语料文件不存在: {os.path.abspath(path)}\n"
            f"  提示：用 --corpus <路径> 指定语料（默认 {DOCS_FILE}）"
        )
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"语料不是合法 JSON: {os.path.abspath(path)}\n  {exc}") from exc
    except OSError as exc:
        raise SystemExit(f"语料读取失败: {os.path.abspath(path)}\n  {exc}") from exc
    if not isinstance(raw, list):
        raise SystemExit(
            f"语料必须是 JSON 数组（每项是一篇文档），实际是 {type(raw).__name__}: "
            f"{os.path.abspath(path)}"
        )
    if not raw:
        raise SystemExit(f"语料为空（0 篇），无法评测: {os.path.abspath(path)}")
    return raw


#: 嵌入时单篇文本的字符上限（--max-embed-chars 的默认值）。
#: 背景：text-embedding-v4 单次输入长度上限约 33000，超过会报
#: `<400> InvalidParameter: Range of input length should be [1, 33000]`。
#: 旧 32 篇语料最长仅 3996 字符，所以 8000 对历史基线完全没有影响（回归不变）；
#: 大语料（src/sample_docs_large.json，最长 136352 字符）必须靠它才能入库。
#: 传 0 表示不截断（大语料下会直接报 400）。
DEFAULT_MAX_EMBED_CHARS = 8000


def truncate_for_embedding(text: str, max_chars: int) -> str:
    """按字符码位截断文本（Python str 切片不会切开多字节中文 / emoji）。

    Args:
        text: 原始文本。
        max_chars: 上限字符数；<= 0 表示不截断。

    Returns:
        str: 截断后的文本。
    """
    if max_chars and max_chars > 0 and len(text) > max_chars:
        return text[:max_chars]
    return text


def make_truncating_encoder(inner, max_chars: int):
    """把 embedding 客户端包一层：**只在计算向量时**截断文本。

    为什么可以只截断嵌入而不动入库正文：
        InMemoryVectorStore.add_documents() 把 `[doc.page_content for doc in documents]`
        交给 `embed_documents()` 算向量，但真正存进 store 的 `text` 字段取的是
        `doc.page_content` 原文（见 langchain_core/vectorstores/in_memory.py 的
        add_documents）。所以在 encoder 这一层截断，只会影响**向量**，
        存进库里的正文和后续拼进 prompt 的 context **仍然是全文**。

    Args:
        inner: 真实的 embedding 客户端（例如 OpenAIEmbeddings）。
        max_chars: 嵌入时的字符上限；<= 0 时原样透传。

    Returns:
        Embeddings: 包装后的客户端。
    """
    from langchain_core.embeddings import Embeddings

    class TruncatingEmbeddings(Embeddings):
        """见 make_truncating_encoder 的说明。异步方法直接委托给 inner，保留原生批量。"""

        def embed_documents(self, texts):
            return inner.embed_documents([truncate_for_embedding(t, max_chars) for t in texts])

        def embed_query(self, text):
            return inner.embed_query(truncate_for_embedding(text, max_chars))

        async def aembed_documents(self, texts):
            return await inner.aembed_documents([truncate_for_embedding(t, max_chars) for t in texts])

        async def aembed_query(self, text):
            return await inner.aembed_query(truncate_for_embedding(text, max_chars))

    return TruncatingEmbeddings()


def build_vector_store(
    corpus_path: str = DOCS_FILE,
    max_embed_chars: int = 0,
    embed_cache_path: Optional[str] = None,
):
    """用给定语料文件建内存向量库（复用 eval_recall.py 的思路）。

    Args:
        corpus_path: 语料 JSON 路径（由 --corpus 决定，默认 src/sample_docs.json）。
        max_embed_chars: 嵌入时的单篇字符上限，<= 0 表示不截断。
        embed_cache_path: 磁盘 embedding 缓存路径；None 表示不启用缓存。

    Returns:
        tuple: (retriever, documents, stats)。
            retriever 的 k 固定为 RETRIEVAL_DEPTH；
            stats 记录语料长度、截断情况与缓存命中，用于控制台提示与报告。
    """
    from langchain_core.vectorstores import InMemoryVectorStore

    from shared import retrieval
    from shared.configuration import BaseConfiguration
    from shared.state import reduce_docs

    # 先校验语料，避免语料有问题时还去初始化 embedding 客户端
    raw = load_corpus_payload(corpus_path)

    config = BaseConfiguration.from_runnable_config({})
    encoder = retrieval.make_text_encoder(config.embedding_model)

    # 复用项目真实的归约器：uuid = md5(page_content) → 与线上完全一致
    documents = reduce_docs([], raw)
    if not documents:
        raise SystemExit(f"语料解析后为 0 篇（内容为空或全部重复）: {os.path.abspath(corpus_path)}")

    lengths = [len(doc.page_content) for doc in documents]
    corpus_max = max(lengths)
    truncation_enabled = bool(max_embed_chars and max_embed_chars > 0)
    truncated_docs = sum(1 for n in lengths if truncation_enabled and n > max_embed_chars)

    stats = {
        "corpus_size": len(documents),
        "corpus_max_chars": corpus_max,
        "corpus_min_chars": min(lengths),
        "max_embed_chars": max_embed_chars if truncation_enabled else 0,
        "truncation_enabled": truncation_enabled,
        "truncated_docs": truncated_docs,
        # 截断后实际送去算向量/存库的最大长度
        "effective_max_chars": min(corpus_max, max_embed_chars) if truncation_enabled else corpus_max,
    }

    # 缓存要包在**最内层**：先截断成最终送入 embedding 的文本，再按该文本算缓存 key。
    # 这样同一个前缀的文档能共用缓存，也能复用外部（如临时探针）已算好的向量。
    cache_obj = None
    if embed_cache_path:
        from embed_cache import DiskCachedEmbeddings

        cache_obj = DiskCachedEmbeddings(encoder, cache_path=embed_cache_path)
        encoder = cache_obj

    if truncation_enabled:
        encoder = make_truncating_encoder(encoder, max_embed_chars)

    store = InMemoryVectorStore(embedding=encoder)
    store.add_documents(documents)

    # 截断包装器没有 stats()，所以要留住内层缓存对象的引用
    if cache_obj is not None:
        stats["embed_cache"] = cache_obj.stats()

    retriever = store.as_retriever(search_kwargs={"k": RETRIEVAL_DEPTH})
    return retriever, documents, stats


# --------------------------------------------------------------------------- #
# 检索
# --------------------------------------------------------------------------- #
def build_query(case: dict[str, Any]) -> str:
    """把多轮历史拼进检索 query（= 关闭改写时的检索行为）。

    多轮用例的当前问句含指代（「那它和边有什么区别」），单独拿去检索几乎没有信号，
    真实系统也是把对话历史一起送进去的。因此这里把 history 和 question 顺序拼接。
    这个拼接结果就是 --rewrite 对比实验里的「改写前」基线。
    """
    history = case.get("history") or []
    if history:
        return "\n".join(list(history) + [case["question"]])
    return case["question"]


def doc_uuid(doc) -> str:
    """取出文档 uuid（reduce_docs 已写入 metadata.uuid）。"""
    return (doc.metadata or {}).get("uuid", "")


async def retrieve_ids(retriever, query: str) -> list[str]:
    """检索 top-RETRIEVAL_DEPTH 篇文档，返回 uuid 列表。"""
    docs = await retriever.ainvoke(query)
    return [doc_uuid(d) for d in docs if doc_uuid(d)]


def compute_retrieval_metrics(
    ranked_ids: Sequence[str], expected_ids: Sequence[str]
) -> dict[str, Optional[float]]:
    """算一条用例的全部检索指标。expected 为空时各指标为 None（未定义）。"""
    result: dict[str, Optional[float]] = {}
    for k in M.DEFAULT_RECALL_KS:
        result[f"recall@{k}"] = M.recall_at_k(ranked_ids, expected_ids, k)
    result["mrr"] = M.mrr(ranked_ids, expected_ids)
    result["ndcg@10"] = M.ndcg_at_k(ranked_ids, expected_ids, k=10)
    result["context_precision"] = M.context_precision(ranked_ids, expected_ids)
    return result


# --------------------------------------------------------------------------- #
# 查询改写（--rewrite）
# --------------------------------------------------------------------------- #
REWRITE_PROMPT = """你是一个检索查询改写器。用户正在进行多轮对话，当前问题是带代词或省略的追问，
单独拿去检索会因为没有指代对象而检不准。

请结合【对话历史】，把【当前问题】改写成一个**不依赖上下文、可以独立拿去检索**的完整问题。

要求：
1. 把「它」「这个」「那如果…」之类指代补全成历史里对应的真实对象；
2. 保留原问题的意图和关键术语（例如 config_schema、add_messages 这类专有名词要原样保留）；
3. 不要臆造历史里没有的新需求，不要把问题扩写成多个问题；
4. 只输出改写后的那一句问题本身，不要解释、不要加引号、不要加任何前缀。

【对话历史】
{history}

【当前问题】
{question}

【改写后的问题】"""


def _clean_rewrite(raw: str) -> str:
    """清洗模型返回的改写结果：取第一段非空行，去掉常见前缀和包裹引号。"""
    text = (raw or "").strip()
    if not text:
        return ""

    for line in text.splitlines():
        if line.strip():
            text = line.strip()
            break

    for prefix in (
        "改写后的问题：",
        "改写后的问题:",
        "改写后：",
        "改写后:",
        "改写：",
        "改写:",
        "问题：",
        "问题:",
        "Rewritten question:",
        "Query:",
    ):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()

    return text.strip("\"'“”「」`‘’").strip()


def rewrite_question(
    channel: LLMChannel, history: Sequence[str], question: str
) -> tuple[Optional[str], Optional[str]]:
    """把一个带指代的追问改写成自包含的检索 query。

    Args:
        channel: 用来调改写模型的通道。
        history: 历史轮次（按时间正序）。
        question: 当前问题。

    Returns:
        tuple: (rewritten_query, error)。失败时 rewritten_query 为 None、error 有值，
        调用方据此降级为使用原查询。
    """
    prompt = REWRITE_PROMPT.format(
        history="\n".join(f"- {h}" for h in history), question=question
    )
    try:
        raw = channel.invoke(prompt)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:300]}"

    cleaned = _clean_rewrite(raw)
    if not cleaned:
        return None, f"改写结果为空（模型原始输出: {raw[:120]!r}）"
    return cleaned, None


# --------------------------------------------------------------------------- #
# 生成
# --------------------------------------------------------------------------- #
GENERATION_PROMPT = """你是严谨的技术问答助手。请**只依据【参考资料】**回答问题。

要求：
1. 在关键结论后用 [1][2] 这样的编号标注它来自哪一段参考资料；
2. 如果参考资料不足以回答问题，必须明确说明「参考资料中没有找到相关信息」，不要编造；
3. 用中文回答，简洁准确，不要重复问题。

【参考资料】
{context}

【问题】
{question}

【回答】"""


def format_contexts_for_generation(docs: Sequence) -> str:
    """把文档列表格式化成带编号的参考资料。"""
    if not docs:
        return "（无）"
    blocks = []
    for index, doc in enumerate(docs, start=1):
        blocks.append(f"[{index}] {doc.page_content}")
    return "\n\n".join(blocks)


def generate_answer(
    channel: LLMChannel, question: str, docs: Sequence
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """用 top-k context 直接让模型作答。

    Returns:
        tuple: (answer, model_name, error)。
    """
    prompt = GENERATION_PROMPT.format(
        context=format_contexts_for_generation(docs),
        question=question,
    )
    try:
        answer = channel.invoke(prompt)
        return answer, channel.used_model, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {str(exc)[:300]}"


def detect_refusal(answer: Optional[str]) -> Optional[bool]:
    """启发式判断模型是否拒答（no_answer 类别专用）。

    Returns:
        bool: True 表示命中了拒答关键词。
        None: 没有答案（生成失败）时为 None。
    """
    if not answer:
        return None
    lowered = answer.lower()
    return any(marker.lower() in lowered for marker in REFUSAL_MARKERS)


# --------------------------------------------------------------------------- #
# 单条用例
# --------------------------------------------------------------------------- #
async def evaluate_case(
    case: dict[str, Any],
    retriever,
    judge: Optional[Judge],
    answer_channel: Optional[LLMChannel],
    rewrite_channel: Optional[LLMChannel],
    rewrite_enabled: bool,
) -> dict[str, Any]:
    """评一条用例：检索指标 +（可选）查询改写对比 +（可选）生成与 judge 打分。"""
    expected = list(case.get("expected_doc_uuids") or [])
    history = case.get("history") or []
    baseline_query = build_query(case)

    record: dict[str, Any] = {
        "id": case.get("id"),
        "category": case.get("category"),
        "question": case.get("question"),
        "history": history,
        "expected_doc_uuids": expected,
        "baseline_query": baseline_query,
        "rewritten_query": None,
        "rewrite_error": None,
        "errors": [],
    }

    # ---- 1) 基线检索（= 关闭改写时的行为，永远算，供前后对比） ----
    try:
        baseline_ids = await retrieve_ids(retriever, baseline_query)
    except Exception as exc:
        baseline_ids = []
        record["errors"].append(f"检索失败 {type(exc).__name__}: {str(exc)[:300]}")

    record["retrieval_baseline"] = compute_retrieval_metrics(baseline_ids, expected)
    record["retrieved_doc_uuids_baseline"] = baseline_ids

    # ---- 2) 查询改写（只对带 history 的用例生效） ----
    query_used = baseline_query
    if rewrite_enabled and history and rewrite_channel is not None:
        rewritten, rewrite_error = rewrite_question(rewrite_channel, history, case["question"])
        record["rewritten_query"] = rewritten
        record["rewrite_error"] = rewrite_error
        if rewritten:
            query_used = rewritten
        else:
            # 改写失败不崩：降级为使用原查询，并留下标记
            record["errors"].append(f"查询改写失败，已降级用原查询: {rewrite_error}")

    record["search_query"] = query_used

    # ---- 3) 用实际 query 检索；如果与基线相同就直接复用 ----
    if query_used == baseline_query:
        ranked_ids = baseline_ids
        record["retrieval"] = record["retrieval_baseline"]
    else:
        try:
            ranked_ids = await retrieve_ids(retriever, query_used)
        except Exception as exc:
            ranked_ids = []
            record["errors"].append(f"改写后检索失败 {type(exc).__name__}: {str(exc)[:300]}")
        record["retrieval"] = compute_retrieval_metrics(ranked_ids, expected)

    record["retrieved_doc_uuids"] = ranked_ids

    # ---- 4) 生成 + judge ----
    is_no_answer = case.get("category") == "no_answer"
    gen_docs = []
    if judge is not None and answer_channel is not None:
        try:
            docs = await retriever.ainvoke(query_used)
            gen_docs = docs[:GENERATION_TOP_K]
        except Exception as exc:
            record["errors"].append(f"取生成上下文失败 {type(exc).__name__}: {str(exc)[:200]}")

        answer, model_name, gen_error = generate_answer(answer_channel, case["question"], gen_docs)
        record["generation"] = {
            "answer": answer,
            "model": model_name,
            "context_doc_uuids": [doc_uuid(d) for d in gen_docs],
            "error": gen_error,
        }
        if gen_error:
            record["errors"].append(f"生成失败 {gen_error}")

        if is_no_answer:
            # no_answer 用例没有参考答案，answer_relevancy / context_recall 在这里没有意义；
            # 真正要测的是「有没有编造」，因此用定义明确的拒答检测，
            # 而不是让 judge 给一个解释不了的分数。详见 README。
            record["refused"] = detect_refusal(answer)
            record["judge"] = {}
        elif answer:
            contexts = [d.page_content for d in gen_docs]
            verdicts: dict[str, Any] = {}
            verdicts["faithfulness"] = judge.faithfulness(case["question"], answer, contexts).as_dict()
            verdicts["answer_relevancy"] = judge.answer_relevancy(case["question"], answer).as_dict()
            hint = case.get("expected_answer_hint")
            if hint:
                verdicts["context_recall"] = judge.context_recall(case["question"], contexts, hint).as_dict()
            record["judge"] = verdicts
            for name, verdict in verdicts.items():
                if verdict.get("judge_error"):
                    record["errors"].append(f"judge[{name}] {verdict['judge_error']}")
        else:
            record["judge"] = {}
    else:
        record["generation"] = None
        record["judge"] = {}
        if is_no_answer:
            record["refused"] = None

    return record


# --------------------------------------------------------------------------- #
# 聚合
# --------------------------------------------------------------------------- #
def mean_of(values: Sequence[Optional[float]]) -> Optional[float]:
    """忽略 None 求平均；全为 None 时返回 None。"""
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


RETRIEVAL_METRIC_KEYS = [f"recall@{k}" for k in M.DEFAULT_RECALL_KS] + [
    "mrr",
    "ndcg@10",
    "context_precision",
]
JUDGE_METRIC_KEYS = ["faithfulness", "answer_relevancy", "context_recall"]


def aggregate(records: Sequence[dict[str, Any]], source: str = "retrieval") -> dict[str, Any]:
    """把一批用例记录聚合成指标汇总。

    Args:
        records: 用例记录列表。
        source: "retrieval" 用实际 query 的结果；"retrieval_baseline" 用关闭改写的基线结果。
    """
    retrieval: dict[str, Any] = {}
    for key in RETRIEVAL_METRIC_KEYS:
        values = [(r.get(source) or {}).get(key) for r in records]
        retrieval[key] = {
            "value": mean_of(values),
            "n": len([v for v in values if v is not None]),
        }

    judge_agg: dict[str, Any] = {}
    for key in JUDGE_METRIC_KEYS:
        values = [(r.get("judge") or {}).get(key, {}).get("score") for r in records if r.get("judge")]
        judge_agg[key] = {
            "value": mean_of(values),
            "n": len([v for v in values if v is not None]),
            "missing": len([v for v in values if v is None]),
        }

    refusal_flags = [r.get("refused") for r in records if r.get("category") == "no_answer"]
    refusal_valid = [f for f in refusal_flags if f is not None]

    return {
        "n_cases": len(records),
        "retrieval": retrieval,
        "judge": judge_agg,
        "refusal": {
            "value": (sum(1 for f in refusal_valid if f) / len(refusal_valid)) if refusal_valid else None,
            "n": len(refusal_valid),
        },
    }


def build_rewrite_comparison(records: Sequence[dict[str, Any]], enabled: bool) -> dict[str, Any]:
    """构造「关闭改写 vs 开启改写」的对比结果（只看带 history 的用例）。"""
    subset = [r for r in records if r.get("history")]
    if not enabled or not subset:
        return {"enabled": enabled, "available": False, "n_cases": len(subset)}

    keys = ["recall@4", "mrr", "ndcg@10"]
    metrics: dict[str, Any] = {}
    for key in keys:
        base_values = [(r.get("retrieval_baseline") or {}).get(key) for r in subset]
        new_values = [(r.get("retrieval") or {}).get(key) for r in subset]
        metrics[key] = {
            "baseline": mean_of(base_values),
            "rewritten": mean_of(new_values),
            "n": len([v for v in new_values if v is not None]),
        }

    rows = []
    for record in subset:
        base = record.get("retrieval_baseline") or {}
        new = record.get("retrieval") or {}
        rows.append(
            {
                "id": record.get("id"),
                "question": record.get("question"),
                "baseline_query": record.get("baseline_query"),
                "rewritten_query": record.get("rewritten_query"),
                "rewrite_error": record.get("rewrite_error"),
                "baseline": {k: base.get(k) for k in keys},
                "rewritten": {k: new.get(k) for k in keys},
            }
        )

    return {
        "enabled": True,
        "available": True,
        "n_cases": len(subset),
        "n_rewritten_ok": len([r for r in subset if r.get("rewritten_query")]),
        "n_rewrite_failed": len([r for r in subset if r.get("rewrite_error")]),
        "metrics": metrics,
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# 报告渲染
# --------------------------------------------------------------------------- #
def fmt(value: Optional[float], digits: int = 4) -> str:
    """None 显示成 n/a，便于一眼看出「未评测」和「0 分」的区别。"""
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    """把报告字典渲染成 Markdown 文本。"""
    meta = report["meta"]
    overall = report["overall"]
    lines: list[str] = []

    lines.append("# RAG Research Agent 评测基线报告")
    lines.append("")
    lines.append(f"- 生成时间：`{meta['generated_at']}`")
    lines.append(f"- 语料：`{meta['corpus']}`（{meta['corpus_size']} 篇）")
    if meta.get("embed_truncation_enabled"):
        trunc_desc = (
            f"**启用**（上限 {meta.get('max_embed_chars')} 字符），"
            f"{meta.get('embed_truncated_docs')} 篇被截断，"
            f"嵌入实际最大 {meta.get('embed_effective_max_chars')} 字符"
        )
    else:
        trunc_desc = "未启用"
    lines.append(
        f"- 语料长度：最大 {meta.get('corpus_max_chars')} 字符；"
        f"嵌入截断：{trunc_desc}；**context 仍为全文（截断只作用于 embedding）**"
    )
    lines.append(f"- 测试集：`{meta['testcases_file']}`（{meta['total_cases']} 条）")
    lines.append(
        f"- 实际评测：**{meta['evaluated_cases']}** 条真实用例；"
        f"跳过 TODO 占位 **{meta['skipped_todo']}** 条"
    )
    lines.append(
        f"- 检索：内存向量库 InMemoryVectorStore（embedding = `{meta['embedding_model']}`，"
        f"depth = {meta['retrieval_depth']}）"
    )
    lines.append(f"- 查询改写（--rewrite）：{'**开启**' if meta['rewrite_enabled'] else '关闭'}")
    if meta.get("judge_enabled"):
        lines.append(
            f"- 生成：top-{meta['generation_top_k']} context 直接作答，"
            f"作答模型 `{meta.get('answer_model') or 'n/a'}`；"
            f"judge 模型 `{meta.get('judge_model') or 'n/a'}`"
        )
    else:
        lines.append("- 生成侧：**本次使用 --skip-judge，未运行生成与 judge**")
    lines.append("")

    # ---- 总表 ----
    lines.append("## 1. 总体指标")
    lines.append("")
    lines.append("| 指标 | 数值 | 参与用例数 | 说明 |")
    lines.append("|---|---:|---:|---|")
    notes = {
        "recall@1": "top-1 里命中期望文档的比例",
        "recall@4": "top-4 覆盖率（项目默认 k）",
        "recall@8": "top-8 覆盖率",
        "recall@20": f"top-20 覆盖率（语料 {meta['corpus_size']} 篇）",
        "mrr": "第一条正确结果的倒数排名均值",
        "ndcg@10": "带位置折扣的排序质量",
        "context_precision": "检索结果里有用文档的平均精度",
    }
    for key in RETRIEVAL_METRIC_KEYS:
        item = overall["retrieval"][key]
        lines.append(f"| {key} | {fmt(item['value'])} | {item['n']} | {notes.get(key, '')} |")

    if meta.get("judge_enabled"):
        judge_notes = {
            "faithfulness": "答案是否只依据 context（1.0 = 无编造）",
            "answer_relevancy": "答案是否切题",
            "context_recall": "context 是否覆盖参考答案要点",
        }
        for key in JUDGE_METRIC_KEYS:
            item = overall["judge"][key]
            suffix = f"，失败 {item['missing']} 条" if item["missing"] else ""
            lines.append(f"| {key} | {fmt(item['value'])} | {item['n']} | {judge_notes[key]}{suffix} |")
        refusal = overall["refusal"]
        if refusal["n"]:
            lines.append(
                f"| no_answer 拒答率 | {fmt(refusal['value'])} | {refusal['n']} | "
                "命中拒答关键词的比例（启发式） |"
            )
    lines.append("")

    # ---- 分类表 ----
    lines.append("## 2. 分类指标")
    lines.append("")
    columns = ["recall@4", "recall@8", "mrr", "ndcg@10", "context_precision"]
    if meta.get("judge_enabled"):
        columns = columns + JUDGE_METRIC_KEYS
    lines.append("| category | 用例数 | " + " | ".join(columns) + " |")
    lines.append("|---|---:|" + "---:|" * len(columns))
    for category, agg in report["per_category"].items():
        cells = [fmt(agg["retrieval"][c]["value"]) for c in columns[:5]]
        if meta.get("judge_enabled"):
            cells = cells + [fmt(agg["judge"][c]["value"]) for c in JUDGE_METRIC_KEYS]
        lines.append(f"| {category} | {agg['n_cases']} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- 查询改写对比 ----
    comparison = report.get("rewrite_comparison") or {}
    section = 3
    if comparison.get("enabled"):
        lines.append(f"## {section}. 查询改写前后对比（`--rewrite`）")
        lines.append("")
        if not comparison.get("available"):
            lines.append("测试集里没有带 `history` 的用例，无法对比。")
        else:
            lines.append(
                f"对象：带 `history` 的 **{comparison['n_cases']}** 条用例"
                f"（改写成功 {comparison['n_rewritten_ok']} 条，"
                f"失败降级 {comparison['n_rewrite_failed']} 条）。"
            )
            lines.append("")
            lines.append(
                "「改写前」= `history + question` 直接拼接去检索；"
                "「改写后」= 改写成的自包含 query。"
            )
            lines.append("")
            lines.append("| 指标 | 改写前 | 改写后 | 差值 | 用例数 |")
            lines.append("|---|---:|---:|---:|---:|")
            for key, item in comparison["metrics"].items():
                base_v = item["baseline"]
                new_v = item["rewritten"]
                delta = "n/a" if (base_v is None or new_v is None) else f"{new_v - base_v:+.4f}"
                lines.append(f"| {key} | {fmt(base_v)} | {fmt(new_v)} | {delta} | {item['n']} |")
            lines.append("")
            lines.append("### 3.1 每条用例的改写结果")
            lines.append("")
            lines.append("| id | 当前问题 | 改写前的检索 query | 改写后的 query | recall@4 前→后 | mrr 前→后 |")
            lines.append("|---|---|---|---|---|---|")
            for row in comparison["rows"]:
                question = (row["question"] or "").replace("|", "\\|")
                if len(question) > 30:
                    question = question[:30] + "…"
                baseline_query = (row["baseline_query"] or "").replace("\n", " ⏎ ").replace("|", "\\|")
                if len(baseline_query) > 46:
                    baseline_query = baseline_query[:46] + "…"
                if row["rewritten_query"]:
                    cell = row["rewritten_query"].replace("|", "\\|")
                    if len(cell) > 46:
                        cell = cell[:46] + "…"
                else:
                    cell = f"**改写失败**：{(row['rewrite_error'] or '')[:28]}"
                r_change = (
                    f"{fmt(row['baseline'].get('recall@4'), 2)} → "
                    f"{fmt(row['rewritten'].get('recall@4'), 2)}"
                )
                m_change = (
                    f"{fmt(row['baseline'].get('mrr'), 2)} → "
                    f"{fmt(row['rewritten'].get('mrr'), 2)}"
                )
                lines.append(
                    f"| {row['id']} | {question} | {baseline_query} | {cell} | {r_change} | {m_change} |"
                )
        lines.append("")
        section += 1

    lines.append(f"## {section}. 失败 / 异常用例明细")
    lines.append("")
    failures = report["failures"]
    if not failures:
        lines.append("本次没有命中判据内的失败用例。")
    else:
        lines.append("| id | category | question | recall@8 | mrr | ndcg@10 | 说明 |")
        lines.append("|---|---|---|---:|---:|---:|---|")
        for item in failures:
            question = (item["question"] or "").replace("|", "\\|")
            if len(question) > 42:
                question = question[:42] + "…"
            lines.append(
                f"| {item['id']} | {item['category']} | {question} | "
                f"{fmt(item['retrieval'].get('recall@8'))} | "
                f"{fmt(item['retrieval'].get('mrr'))} | "
                f"{fmt(item['retrieval'].get('ndcg@10'))} | {item['why']} |"
            )
    lines.append("")

    section += 1
    lines.append(f"## {section}. 跳过的用例与异常")
    lines.append("")
    lines.append("### TODO 占位条目（未评测）")
    lines.append("")
    per_cat = meta["skipped_by_category"]
    if per_cat:
        lines.append("| category | 跳过条数 |")
        lines.append("|---|---:|")
        for category, count in per_cat.items():
            lines.append(f"| {category} | {count} |")
        lines.append(f"| **合计** | **{meta['skipped_todo']}** |")
    else:
        lines.append("无。")
    lines.append("")
    lines.append("### judge / 生成 / 改写异常")
    lines.append("")
    error_counts = report["error_summary"]
    if error_counts:
        lines.append("| 异常类型 | 出现次数 |")
        lines.append("|---|---:|")
        for message, count in error_counts.items():
            lines.append(f"| {message} | {count} |")
    else:
        lines.append("无。")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "> 已知限制：内存向量库 ≠ Elasticsearch；judge 是 LLM 近似打分；"
        f"TODO 条目被跳过；语料 {meta['corpus_size']} 篇（小语料下指标波动大）。"
        "详见 `eval/README.md`。"
    )
    lines.append("")
    return "\n".join(lines)


def collect_failures(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """挑出需要人看的失败用例。

    判据（满足任一）：
      - 有期望文档但 MRR = 0（一个都没召回）
      - 有期望文档但 Recall@8 < 1.0
      - judge 任一指标低于 0.5
      - no_answer 用例没有拒答
      - 该用例记录了异常
    """
    failures: list[dict[str, Any]] = []
    for record in records:
        reasons: list[str] = []
        retrieval = record.get("retrieval") or {}
        expected = record.get("expected_doc_uuids") or []

        if expected:
            if retrieval.get("mrr") == 0.0:
                reasons.append("检索完全未命中")
            elif retrieval.get("recall@8") is not None and retrieval["recall@8"] < 1.0:
                reasons.append(f"Recall@8={fmt(retrieval['recall@8'], 2)} 未全中")

        for key in JUDGE_METRIC_KEYS:
            score = (record.get("judge") or {}).get(key, {}).get("score")
            if score is not None and score < 0.5:
                reasons.append(f"{key}={fmt(score, 2)} 偏低")

        if record.get("category") == "no_answer" and record.get("refused") is False:
            reasons.append("未拒答（可能编造）")

        if record.get("errors"):
            reasons.append("存在异常")

        if reasons:
            failures.append(
                {
                    "id": record["id"],
                    "category": record["category"],
                    "question": record["question"],
                    "retrieval": retrieval,
                    "why": "；".join(dict.fromkeys(reasons)),
                }
            )
    return failures


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
async def run(args: argparse.Namespace) -> int:
    """执行评测主流程。"""
    started = datetime.now()

    print("=" * 78)
    print("RAG Research Agent 评测基线")
    print("=" * 78)

    cases = load_testcases(args.testcases)
    print(f"测试集已加载：{len(cases)} 条  <- {args.testcases}")

    real_cases = [c for c in cases if not c.get("TODO")]
    todo_cases = [c for c in cases if c.get("TODO")]
    if args.limit:
        real_cases = real_cases[: args.limit]
    print(f"  真实用例 {len(real_cases)} 条 | TODO 占位跳过 {len(todo_cases)} 条")

    if not real_cases:
        raise SystemExit("没有可评测的用例（全部是 TODO 占位？）")

    print("\n正在构建内存向量库（dashscope/text-embedding-v4）...")
    embed_cache_path = None
    if not args.no_embed_cache and args.embed_cache is not None:
        from embed_cache import default_cache_path

        embed_cache_path = default_cache_path() if args.embed_cache == "" else args.embed_cache
        print(f"embedding 缓存：{embed_cache_path}")

    retriever, documents, embed_stats = build_vector_store(
        args.corpus, args.max_embed_chars, embed_cache_path
    )
    cache_stats = embed_stats.get("embed_cache")
    if cache_stats:
        print(
            f"  缓存命中：{cache_stats['hits']} / 请求 {cache_stats['misses']} 次"
            f"（命中率 {cache_stats['hit_rate']:.0%}，库内共 {cache_stats['cached']} 条）"
        )
    print(f"  向量库就绪：{len(documents)} 篇文档  <- {display_path(args.corpus)}")
    print(f"  语料长度：最长 {embed_stats['corpus_max_chars']} 字符，"
          f"最短 {embed_stats['corpus_min_chars']} 字符")
    if not embed_stats["truncation_enabled"]:
        print(f"  嵌入前截断：未启用（--max-embed-chars 0），"
              f"最长 {embed_stats['corpus_max_chars']} 字符将原样送入 embedding")
    elif embed_stats["truncated_docs"]:
        print(f"  嵌入前截断：{embed_stats['truncated_docs']} 篇超过 "
              f"{embed_stats['max_embed_chars']} 字符"
              f"（最长 {embed_stats['corpus_max_chars']}），仅对嵌入生效")
        print("  提醒：入库正文与拼给模型的 context 仍是全文，未截断。")
    else:
        print(f"  嵌入前截断：已启用（上限 {embed_stats['max_embed_chars']} 字符），"
              f"没有文档超限（最长 {embed_stats['corpus_max_chars']}）")

    def make_channel(explicit: Optional[str], env_var: str, defaults: Sequence[str]) -> LLMChannel:
        names = [explicit] if explicit else None
        return LLMChannel(model_names=names, env_var=env_var, defaults=defaults)

    answer_channel: Optional[LLMChannel] = None
    judge: Optional[Judge] = None
    if args.skip_judge:
        print("  生成侧：已跳过（--skip-judge，不发起任何对话模型调用）")
    else:
        answer_channel = make_channel(args.answer_model, "EVAL_ANSWER_MODEL", DEFAULT_ANSWER_MODELS)
        judge_channel = make_channel(args.judge_model, "EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODELS)
        judge = Judge(channel=judge_channel)
        print(f"  生成侧：启用；作答候选 {answer_channel.candidates}；judge 候选 {judge_channel.candidates}")

    rewrite_channel: Optional[LLMChannel] = None
    if args.rewrite:
        rewrite_channel = make_channel(args.rewrite_model, "EVAL_REWRITE_MODEL", DEFAULT_REWRITE_MODELS)
        print(f"  查询改写：开启；候选 {rewrite_channel.candidates}")
    else:
        print("  查询改写：关闭（加 --rewrite 开启）")

    records: list[dict[str, Any]] = []
    for index, case in enumerate(real_cases, start=1):
        print(f"\n[{index}/{len(real_cases)}] {case['id']} ({case['category']})")
        try:
            record = await evaluate_case(
                case, retriever, judge, answer_channel, rewrite_channel, bool(args.rewrite)
            )
        except Exception as exc:  # 单条用例失败不能拖垮整轮评测
            record = {
                "id": case.get("id"),
                "category": case.get("category"),
                "question": case.get("question"),
                "history": case.get("history") or [],
                "expected_doc_uuids": case.get("expected_doc_uuids") or [],
                "baseline_query": build_query(case),
                "rewritten_query": None,
                "rewrite_error": None,
                "search_query": build_query(case),
                "retrieved_doc_uuids": [],
                "retrieved_doc_uuids_baseline": [],
                "retrieval": {},
                "retrieval_baseline": {},
                "generation": None,
                "judge": {},
                "errors": [f"用例整体异常 {type(exc).__name__}: {str(exc)[:300]}"],
            }
            traceback.print_exc()

        records.append(record)

        retrieval = record.get("retrieval") or {}
        line = f"    Recall@4={fmt(retrieval.get('recall@4'))}  MRR={fmt(retrieval.get('mrr'))}"
        if record.get("rewritten_query"):
            line += f"  rewritten={record['rewritten_query'][:40]}"
        judge_scores = record.get("judge") or {}
        if judge_scores:
            parts = [f"{k}={fmt(v.get('score'), 2)}" for k, v in judge_scores.items()]
            line += "  judge: " + " ".join(parts)
        if record.get("category") == "no_answer":
            line += f"  refused={record.get('refused')}"
        print(line)

    overall = aggregate(records)
    per_category: "OrderedDict[str, Any]" = OrderedDict()
    for category in ["cross_section", "exact_term", "multi_turn", "no_answer", "permission"]:
        subset = [r for r in records if r["category"] == category]
        if subset:
            per_category[category] = aggregate(subset)

    comparison = build_rewrite_comparison(records, bool(args.rewrite))
    failures = collect_failures(records)

    error_counter: Counter[str] = Counter()
    for record in records:
        for message in record.get("errors") or []:
            error_counter[message[:160]] += 1

    judge_model = judge.channel.used_model if judge is not None else None
    if judge is not None and judge_model is None:
        judge_model = judge.channel.candidates[0] if judge.channel.candidates else None

    report = {
        "meta": {
            "generated_at": started.strftime("%Y-%m-%d %H:%M:%S"),
            "corpus": display_path(args.corpus),
            "corpus_size": len(documents),
            "testcases_file": display_path(args.testcases),
            "total_cases": len(cases),
            "evaluated_cases": len(records),
            "skipped_todo": len(todo_cases),
            "skipped_by_category": dict(Counter(c.get("category") for c in todo_cases)),
            "embedding_model": "dashscope/text-embedding-v4",
            "retrieval_depth": RETRIEVAL_DEPTH,
            "generation_top_k": GENERATION_TOP_K,
            # ---- 语料长度与嵌入截断 ----
            "corpus_max_chars": embed_stats["corpus_max_chars"],
            "corpus_min_chars": embed_stats["corpus_min_chars"],
            "max_embed_chars": embed_stats["max_embed_chars"],
            "embed_truncation_enabled": embed_stats["truncation_enabled"],
            "embed_truncated_docs": embed_stats["truncated_docs"],
            "embed_effective_max_chars": embed_stats["effective_max_chars"],
            # 截断只作用于 embedding：入库正文与拼给模型的 context 仍是全文
            "truncation_applies_to_context": False,
            "judge_enabled": judge is not None,
            "judge_model": judge_model,
            "answer_model": answer_channel.used_model if answer_channel is not None else None,
            "rewrite_enabled": bool(args.rewrite),
            "rewrite_model": rewrite_channel.used_model if rewrite_channel is not None else None,
            "answer_candidates": answer_channel.candidates if answer_channel else [],
            "judge_candidates": judge.channel.candidates if judge else [],
            "rewrite_candidates": rewrite_channel.candidates if rewrite_channel else [],
            "skip_judge_flag": bool(args.skip_judge),
        },
        "overall": overall,
        "per_category": per_category,
        "rewrite_comparison": comparison,
        "cases": records,
        "failures": failures,
        "error_summary": dict(error_counter),
    }

    # ---- 输出 ----
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        stamp = started.strftime("%Y%m%d_%H%M")
        suffix = "_rewrite" if args.rewrite else ""
        base = os.path.join(args.output_dir, f"baseline_{stamp}{suffix}")
        md_path = base + ".md"
        json_path = base + ".json"

        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write(render_markdown(report))
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    except OSError as exc:
        raise SystemExit(
            f"报告写入失败: {os.path.abspath(args.output_dir)}\n  {exc}\n"
            '  提示：沙箱环境请用 --output-dir 指到 $env:TEMP 下'
            '（例如 --output-dir "$env:TEMP\\rag_eval_reports"）'
        ) from exc

    # ---- 控制台汇总 ----
    print("\n" + "=" * 78)
    print("检索指标（总体）")
    print("=" * 78)
    for key in RETRIEVAL_METRIC_KEYS:
        item = overall["retrieval"][key]
        print(f"  {key:<18} {fmt(item['value']):<10} (n={item['n']})")

    if judge is not None:
        print("\n生成侧指标（总体）")
        print("=" * 78)
        for key in JUDGE_METRIC_KEYS:
            item = overall["judge"][key]
            print(f"  {key:<18} {fmt(item['value']):<10} (n={item['n']}, 失败={item['missing']})")
        if overall["refusal"]["n"]:
            print(
                f"  {'no_answer 拒答率':<16} {fmt(overall['refusal']['value']):<10} "
                f"(n={overall['refusal']['n']})"
            )
        print(f"  judge 实际使用模型: {judge_model}")

    if comparison.get("available"):
        print("\n查询改写前后对比（带 history 的用例）")
        print("=" * 78)
        print(
            f"  用例数 {comparison['n_cases']} | 改写成功 {comparison['n_rewritten_ok']} | "
            f"失败降级 {comparison['n_rewrite_failed']}"
        )
        for key, item in comparison["metrics"].items():
            base_v, new_v = item["baseline"], item["rewritten"]
            delta = "n/a" if (base_v is None or new_v is None) else f"{new_v - base_v:+.4f}"
            print(f"  {key:<12} 改写前 {fmt(base_v):<10} 改写后 {fmt(new_v):<10} 差值 {delta}")

    print(f"\n报告已写出：\n  {md_path}\n  {json_path}")
    print(f"耗时 {(datetime.now() - started).total_seconds():.1f}s")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="RAG Research Agent 评测基线（内存向量库，不依赖 ES）",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="只跑检索指标，跳过生成与 judge（不发起任何对话模型调用）",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="对带 history 的用例做查询改写，并在报告里给改写前后对比（默认关闭）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只跑前 N 条真实用例，用于冒烟测试（默认 0 = 全跑）",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"报告输出目录（默认 {DEFAULT_OUTPUT_DIR}；沙箱环境请指到 $env:TEMP 下）",
    )
    parser.add_argument(
        "--testcases",
        default=TESTCASES_FILE,
        help=f"测试集路径（默认 {TESTCASES_FILE}）",
    )
    parser.add_argument(
        "--corpus",
        default=DOCS_FILE,
        help=(
            f"语料文件路径（默认 {DOCS_FILE}）。"
            "换大语料时指过去，例如 --corpus src/sample_docs_large.json；"
            "默认值保持不变是为了历史基线可比。"
        ),
    )
    parser.add_argument(
        "--max-embed-chars",
        type=int,
        default=DEFAULT_MAX_EMBED_CHARS,
        help=(
            f"嵌入时单篇文本的字符上限（默认 {DEFAULT_MAX_EMBED_CHARS}；传 0 表示不截断）。"
            "大语料（如 src/sample_docs_large.json，最长 136352 字符）会超出 "
            "text-embedding-v4 的输入长度上限而报 400，用它才能入库；"
            "截断只作用于 embedding，入库正文与拼给模型的 context 仍是全文。"
        ),
    )
    parser.add_argument(
        "--embed-cache",
        default=None,
        metavar="PATH",
        help=(
            "启用磁盘级 embedding 缓存（推荐）。传路径，或传空字符串用默认 TEMP 路径。"
            "缓存按「送进 embedding 的最终文本」做 sha256 key，命中即不发请求；"
            "大语料首次约 8~10 分钟，之后重跑几乎免费。"
        ),
    )
    parser.add_argument(
        "--no-embed-cache",
        action="store_true",
        help="显式关闭 embedding 缓存（默认关闭；要省钱请用 --embed-cache）",
    )
    parser.add_argument(
        "--answer-model",
        default=None,
        help="作答模型，例如 dashscope/qwen-plus（默认 qwen-plus → qwen-flash 降级）",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="judge 模型，例如 dashscope/qwen-plus（默认 qwen-plus → qwen-flash 降级）",
    )
    parser.add_argument(
        "--rewrite-model",
        default=None,
        help="查询改写模型，例如 dashscope/qwen-flash（默认 qwen-flash → qwen-plus 降级）",
    )
    return parser.parse_args(argv)


def main() -> int:
    """同步入口。"""
    args = parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
