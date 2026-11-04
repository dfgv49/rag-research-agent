"""召回率 / 精确率对比实验（本地哈希向量，语料可放大到任意规模）。

为什么用本地哈希向量而不是百炼？
  为了能自由控制语料规模（本例放大到约 900 篇），并且结果可复现、零成本。
  向量质量不如真实 embedding，但所有策略用的是同一个向量空间，
  对比结论（哪种策略召回更好）依然有效。

对比策略：
  A. 单查询 k=4             （项目默认）
  B. 单查询 k=10
  C. 多查询 3条 k=4         （项目实际做法）
  D. 多查询 3条 k=4 + 重排
  E. 多查询 3条 k=8 + 重排

指标：
  Recall@k   目标文档是否出现在结果里
  Precision  结果里有多少是相关文档（看它挤掉了多少位置）
  MRR        目标文档排在第几位（越靠前越好）

用法:
    .venv\\Scripts\\python.exe eval_recall2.py
"""

import asyncio
import hashlib
import json
import math
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402

from shared.state import reduce_docs  # noqa: E402

DOCS_FILE = "src/sample_docs.json"
DIM = 256


def tokenize(text: str) -> list[str]:
    """英文单词 + 中文双字组（bigram）。

    为什么要加中文 bigram：测试问题大多是中文，而语料是英文。
    如果只取英文单词，中文问题会得到空向量，整个实验就失去意义。
    中文 bigram 能捕捉「节点」「条件边」这类词，且与英文文档里的
    node / conditional edge 恰好没有重叠 —— 这正好模拟了真实的
    「跨语言检索困难」场景。
    """
    text = (text or "").lower()
    tokens = re.findall(r"[a-zA-Z]{3,}", text)
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    tokens += ["".join(cjk[i:i + 2]) for i in range(len(cjk) - 1)]
    return tokens


class LocalHashEmbeddings(Embeddings):
    """极简本地向量：按词哈希 + 次线性词频加权 + L2 归一化。

    不是好向量，但它稳定、免费、可复现，适合做策略对比。
    """

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        counts: dict[str, int] = {}
        for tok in tokenize(text):
            counts[tok] = counts.get(tok, 0) + 1
        for tok, c in counts.items():
            h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
            vec[h % DIM] += 1.0 + math.log(c)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


# ---------------------------------------------------------------- 语料构造
TOPICS = {
    "langgraph": [
        "LangGraph is a library for building stateful multi-actor applications with LLMs",
        "In LangGraph, nodes are typically python functions where the first argument is the state",
        "You MUST compile your graph before you can use it and inspect the state schema",
        "Conditional edges call a routing function to decide the next node at runtime",
        "The Send API dispatches the same node multiple times in parallel with custom state",
        "Reducer functions define how a state key is updated when multiple nodes write it",
        "Checkpointers persist graph state so you can resume a run or replay a step",
        "Subgraphs let you nest a compiled graph as a node inside a parent graph",
        "Add a config schema to expose configurable parameters such as the model name",
        "Parallel execution runs sibling nodes in the same superstep and merges results",
    ],
    "langchain": [
        "LangChain provides standard extendable interfaces for chat models and vector stores",
        "Runnables expose input and output schemas to inspect the expected data shape",
        "A retriever accepts a query and returns the most relevant documents from a store",
        "Text splitters break long documents into chunks that fit the embedding window",
        "Prompt templates format variables into messages for the chat model to consume",
        "Output parsers convert raw model text into structured objects such as JSON",
    ],
}

# 干扰文档：与真实主题共享通用词（graph、state、run 等），但语义无关
DISTRACTOR_TEMPLATES = [
    "The {noun} of the {domain} graph is analyzed in terms of state and run metrics",
    "A method to manage {noun} state inside the {domain} system during a run",
    "How to configure {noun} parameters for the {domain} pipeline at scale",
    "Performance measurements of the {domain} {noun} under parallel load conditions",
    "Best practices for storing {noun} data and querying it efficiently in the {domain} store",
]
NOUNS = [
    "inventory", "invoice", "warehouse", "shipment", "recipe", "rainfall",
    "traffic", "patient", "ledger", "crop", "turbine", "bridge", "subway",
    "hotel", "flight", "menu", "salary", "tax", "migration", "particle",
]
DOMAINS = [
    "logistics", "retail", "agriculture", "healthcare", "banking", "transport",
    "hospitality", "meteorology", "biology", "manufacturing", "telecom", "energy",
]


def build_corpus():
    """真实文档 32 篇 + 干扰文档约 900 篇。"""
    with open(DOCS_FILE, encoding="utf-8") as f:
        real_docs = reduce_docs([], json.load(f))
    for i, d in enumerate(real_docs):
        d.metadata["uuid"] = f"real-{i:02d}"
        d.metadata["kind"] = "real"

    distractors = []
    idx = 0
    for noun in NOUNS:
        for domain in DOMAINS:
            for tmpl in DISTRACTOR_TEMPLATES:
                idx += 1
                text = tmpl.format(noun=noun, domain=domain)
                distractors.append(
                    Document(
                        page_content=text + f" (reference {idx})",
                        metadata={"uuid": f"dist-{idx:04d}", "kind": "distractor"},
                    )
                )
    # 再给真实主题加一些同域但不同主题的干扰
    for topic, docs in TOPICS.items():
        for i, text in enumerate(docs):
            distractors.append(
                Document(
                    page_content=text + " — related integration notes",
                    metadata={"uuid": f"near-{topic}-{i}", "kind": "near"},
                )
            )
    return real_docs, distractors


TEST_SET = [
    ("LangGraph 里节点是怎么定义的？", "real-01"),
    ("怎么编译一个图？必须 compile 吗？", "real-02"),
    ("条件边怎么决定下一个节点？", "real-03"),
    ("怎么让同一个节点并行执行多次？", "real-04"),
    ("状态被多个节点同时写入怎么办？", "real-05"),
    ("检查点（checkpoint）有什么作用？", "real-06"),
    ("子图怎么嵌套进主图？", "real-07"),
    ("怎么给图暴露可配置参数？", "real-08"),
    ("Runnable 的输入输出结构怎么看？", "real-10"),
    ("检索器是干什么的？", "real-11"),
]


def target_id(hit_docs, uuid):
    for d in hit_docs:
        if d.metadata.get("uuid") == uuid:
            return True
    return False


def rank_of(hit_docs, uuid):
    for i, d in enumerate(hit_docs, 1):
        if d.metadata.get("uuid") == uuid:
            return i
    return 0


def rerank(question: str, docs: list[Document], top_k: int) -> list[Document]:
    """模拟 rerank：按关键词重合度重排，取前 top_k。"""
    q_tokens = set(tokenize(question))
    scored = []
    for i, d in enumerate(docs):
        d_tokens = set(tokenize(d.page_content))
        overlap = len(q_tokens & d_tokens)
        # 用「重合词数」为主，「原始排名」为辅（稳定排序）
        scored.append((-overlap, i, d))
    scored.sort()
    return [d for _, _, d in scored[:top_k]]


async def main() -> None:
    import contextlib

    import langchain_core.embeddings as emb_mod
    from langchain_core.vectorstores import InMemoryVectorStore

    print("=" * 82)
    print("召回率 / 精确率对比实验（本地哈希向量，放大语料）")
    print("=" * 82)

    real_docs, distractors = build_corpus()
    all_docs = real_docs + distractors
    print(f"语料规模：真实文档 {len(real_docs)} 篇 + 干扰文档 {len(distractors)} 篇"
          f" = 共 {len(all_docs)} 篇")

    # 用本地哈希向量替换真实 embedding（只在这个脚本内生效）
    local = LocalHashEmbeddings()
    print("正在建索引（本地向量，无需联网）...")
    store = InMemoryVectorStore(embedding=local)
    store.add_documents(all_docs)
    print("索引完成")
    print()

    strategies = [
        ("A. 单查询 k=4", False, 4, False),
        ("B. 单查询 k=10", False, 10, False),
        ("C. 多查询3条 k=4", True, 4, False),
        ("D. 多查询3条 k=4+重排", True, 4, True),
        ("E. 多查询3条 k=8+重排", True, 8, True),
    ]

    stats = {n: {"hit": 0, "rr": 0.0, "prec": []} for n, _, _, _ in strategies}

    for question, uuid in TEST_SET:
        print(f"【问题】{question}   （目标文档 {uuid}）")
        base = await store.asimilarity_search(question, k=20)

        for name, multi, k, use_rerank in strategies:
            if multi:
                # 用「查询改写」模拟多查询：原问题的几种改写
                variants = [
                    question,
                    question + " langgraph node state graph",
                    question.replace("怎么", "如何").replace("吗", ""),
                ]
                merged: list[Document] = []
                for v in variants:
                    docs = await store.asimilarity_search(v, k=k * 2)
                    merged = reduce_docs(merged, docs)
                pool = merged
            else:
                pool = base

            if use_rerank:
                result = rerank(question, pool, k)
            else:
                result = pool[:k]

            ok = target_id(result, uuid)
            r = rank_of(result, uuid)
            # precision：结果里属于「真实文档」的比例（简化定义）
            relevant = sum(1 for d in result if d.metadata.get("kind") == "real")
            prec = relevant / len(result) if result else 0.0

            if ok:
                stats[name]["hit"] += 1
                stats[name]["rr"] += 1.0 / r
            stats[name]["prec"].append(prec)

            print(
                f"    {name:<22} 命中={'✓' if ok else '✗'} "
                f"排名={r if r else '-':<3} 返回={len(result):<3} "
                f"其中相关={relevant}"
            )
        print()

    n = len(TEST_SET)
    print("=" * 82)
    print("汇总")
    print("=" * 82)
    print(f"  测试问题数 = {n}，语料 {len(all_docs)} 篇")
    print()
    print(f"  {'策略':<24}{'Recall@k':>10}{'MRR':>8}{'平均相关比例':>14}")
    print("  " + "-" * 56)
    for name, _, _, _ in strategies:
        s = stats[name]
        recall = s["hit"] / n
        mrr = s["rr"] / n
        prec = sum(s["prec"]) / len(s["prec"])
        print(f"  {name:<24}{recall * 100:>9.1f}%{mrr:>8.3f}{prec:>13.2f}")

    print()
    print("  说明：")
    print("    Recall@k  目标文档是否出现在结果里（越高越好）")
    print("    MRR       目标文档排名的倒数平均（越接近 1 越靠前）")
    print("    平均相关比例  结果里真实文档占比（反映精确率，越高说明噪声越少）")


if __name__ == "__main__":
    asyncio.run(main())
