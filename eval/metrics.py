"""检索层评测指标：纯函数实现，零网络依赖、零第三方依赖。

本模块只做「给定一个按相关度排序的 ID 列表 + 一组期望命中的 ID，算出标量分数」，
不负责加载语料、不负责调模型，因此可以被任何单测直接 import 调用：

    from eval.metrics import recall_at_k, mrr, ndcg_at_k, context_precision

--------------------------------------------------------------------------
统一的约定
--------------------------------------------------------------------------
1. 参数一律用「已排序的检索结果 ID 列表」（ranked_ids / hit_ids）+ 「期望命中的
   ID 集合」（expected_ids）。ID 是文档 uuid 字符串，但本模块不关心它具体是什么，
   只要能哈希比较即可。
2. 结果列表按相关度从高到低排列，第 1 个元素是 top-1。
3. 重复 ID 按首次出现位置保留一次（保序去重），避免同一条文档被重复计分。
4. **边界约定（所有函数一致）**：
   - expected_ids 为空 → 返回 None。这是「未定义」而不是 0 分：
     没有标注正确答案时，任何分数都没有意义，调用方应当把它排除在聚合之外，
     而不是当成 0 分拉低均值。
   - expected_ids 非空但检索结果里一个都没命中 → 返回 0.0。
   - 检索结果为空列表但 expected_ids 非空 → 返回 0.0。
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

__all__ = [
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "context_precision",
    "DEFAULT_RECALL_KS",
]

#: run_eval.py 默认输出的 Recall@K 档位
DEFAULT_RECALL_KS = (1, 4, 8, 20)


def _unique_ids(ids: Optional[Iterable[str]]) -> list[str]:
    """保序去重，并丢掉 None / 空串。

    Args:
        ids: 任意可迭代的 ID 序列，允许为 None。

    Returns:
        list[str]: 去重后的 ID 列表，保持原始先后顺序。
    """
    if not ids:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in ids:
        if item is None or item == "":
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _expected_set(expected_ids: Optional[Iterable[str]]) -> set[str]:
    """把期望 ID 列表规范化成集合（去重、去空）。"""
    return {i for i in (expected_ids or []) if i is not None and i != ""}


def recall_at_k(
    hit_ids: Optional[Sequence[str]],
    expected_ids: Optional[Sequence[str]],
    k: int,
) -> Optional[float]:
    """Recall@K —— 前 K 个结果里命中了多少比例的正确文档。

    算法:
        Recall@K = |{top-K 结果} ∩ {期望 ID}| / |{期望 ID}|

        只看「覆盖率」，不看顺序：期望 3 篇、命中 2 篇就是 2/3 = 0.667，
        即使命中的那 2 篇排在第 1 和第 8 位也一样。
        当 |expected| <= K 时，Recall@K == 1.0 就等价于「全中」。

    Args:
        hit_ids: 按相关度降序排列的检索结果 ID 列表（完整列表，函数内部取前 k 个）。
        expected_ids: 期望命中的文档 ID 列表（标注的标准答案）。
        k: 截断位置，常用 1 / 4 / 8 / 20。必须为正整数。

    Returns:
        float: [0.0, 1.0] 的命中比例。
        None: 当 expected_ids 为空（无标准答案，指标未定义）。

    边界情况:
        - expected_ids 为 None 或空列表 → 返回 None。
        - k <= 0 → 返回 0.0（不取任何结果，命中数必然为 0）。
        - hit_ids 为 None 或空列表 → 返回 0.0。
        - hit_ids 里有重复 ID → 按首次出现计一次，不会重复加分。
        - |expected| <= k 且全部命中 → 返回 1.0。

    Examples:
        >>> recall_at_k(["a", "b", "c"], ["b", "z"], 1)
        0.0
        >>> recall_at_k(["a", "b", "c"], ["b", "z"], 4)
        0.5
        >>> recall_at_k(["a"], [], 4) is None
        True
    """
    expected = _expected_set(expected_ids)
    if not expected:
        return None
    if k <= 0:
        return 0.0
    top_k = _unique_ids(hit_ids)[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & expected) / len(expected)


def mrr(
    ranked_ids: Optional[Sequence[str]],
    expected_ids: Optional[Sequence[str]],
) -> Optional[float]:
    """MRR (Mean Reciprocal Rank) —— 第一条正确结果排得有多靠前的倒数。

    算法:
        RR = 1 / rank(第一条命中的正确文档)
        单个查询的 MRR 就是它的 RR；跨查询取算术平均才是真正意义上的 MRR，
        本函数算的是「单条查询的 RR」，聚合由调用方完成。

        只看第一条命中：top-1 命中得 1.0，top-4 命中得 0.25，top-10 命中得 0.1。
        它衡量「用户第一眼看到对的东西」的能力，比 Recall 更敏感于排序。

    Args:
        ranked_ids: 按相关度降序排列的检索结果 ID 列表（全量，不截断）。
        expected_ids: 期望命中的文档 ID 列表。

    Returns:
        float: [0.0, 1.0]，值为 1/rank。
        None: expected_ids 为空时（未定义）。

    边界情况:
        - expected_ids 为空 → 返回 None。
        - 一条都没命中 → 返回 0.0。
        - ranked_ids 为空 → 返回 0.0。
        - 重复 ID 只按首次出现计算排名。

    Examples:
        >>> mrr(["x", "b", "y"], ["b"])
        0.5
        >>> mrr(["b", "x"], ["b"])
        1.0
        >>> mrr(["x"], ["b"])
        0.0
        >>> mrr(["x"], []) is None
        True
    """
    expected = _expected_set(expected_ids)
    if not expected:
        return None
    for rank, doc_id in enumerate(_unique_ids(ranked_ids), start=1):
        if doc_id in expected:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_ids: Optional[Sequence[str]],
    expected_ids: Optional[Sequence[str]],
    k: int = 10,
) -> Optional[float]:
    """nDCG@K —— 打折累积增益的归一化版本，同时考虑「命中」和「排得多靠前」。

    算法（二值相关性：命中=1，未命中=0）:
        DCG@K  = Σ_{i=1..K} rel_i / log2(i + 1)
        IDCG@K = Σ_{i=1..min(|expected|, K)} 1 / log2(i + 1)   # 理想排序：正确文档全排最前
        nDCG@K = DCG@K / IDCG@K

        第 1 位的增益权重是 1/log2(2)=1.0，第 2 位 1/log2(3)≈0.63，
        第 4 位≈0.43，第 10 位≈0.29 —— 越靠后贡献越小。
        IDCG 用「期望文档数量」而非 K 做上限，因此期望只有 1 篇时，
        排在 top-1 就直接得 1.0。

    Args:
        ranked_ids: 按相关度降序排列的检索结果 ID 列表。
        expected_ids: 期望命中的文档 ID 列表。
        k: 截断位置，默认 10。

    Returns:
        float: [0.0, 1.0]，1.0 表示排序等同于理想排序。
        None: expected_ids 为空时（未定义）。

    边界情况:
        - expected_ids 为空 → 返回 None。
        - k <= 0 → 返回 0.0。
        - 一条都没命中 → DCG=0 → 返回 0.0。
        - 重复 ID 只按首次出现计算，不会重复贡献增益。

    Examples:
        >>> round(ndcg_at_k(["b", "x"], ["b"], k=10), 4)
        1.0
        >>> round(ndcg_at_k(["x", "b"], ["b"], k=10), 4)
        0.6309
        >>> ndcg_at_k(["x"], [], k=10) is None
        True
    """
    expected = _expected_set(expected_ids)
    if not expected:
        return None
    if k <= 0:
        return 0.0
    top_k = _unique_ids(ranked_ids)[:k]
    if not top_k:
        return 0.0

    dcg = 0.0
    for i, doc_id in enumerate(top_k, start=1):
        if doc_id in expected:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def context_precision(
    ranked_ids: Optional[Sequence[str]],
    expected_ids: Optional[Sequence[str]],
) -> Optional[float]:
    """Context Precision —— 检索回来的上下文里，有多大比例是真正有用的。

    算法（Average Precision，二值相关性）:
        对检索列表从头扫到尾，每遇到一个「命中」的文档，就记下此刻的
        Precision@position = 命中数 / 当前位置，最后对这些位置求平均：
            AP = (1 / 命中总数) · Σ_{i: 第 i 位命中} (命中数_i / i)

        它同时惩罚两件事：召回了多少无关文档（稀释分母），
        以及正确文档被无关文档挤到多后面。一条都没命中时直接是 0.0。

        注意本函数只在「已检索到的结果」范围内计算，不惩罚「没召回的那部分」
        （那是 Recall 的职责），因此它和 Recall 是互补的两个视角。

    Args:
        ranked_ids: 按相关度降序排列的检索结果 ID 列表。
        expected_ids: 期望命中的文档 ID 列表。

    Returns:
        float: [0.0, 1.0]。
        None: expected_ids 为空时（未定义）。

    边界情况:
        - expected_ids 为空 → 返回 None。
        - ranked_ids 为空 → 返回 0.0。
        - 一条都没命中 → 返回 0.0。
        - 命中全部且没有无关文档 → 返回 1.0。
        - 重复 ID 只按首次出现计算。

    Examples:
        >>> round(context_precision(["b", "x", "y"], ["b"]), 4)
        1.0
        >>> round(context_precision(["x", "b"], ["b"]), 4)
        0.5
        >>> context_precision(["x"], ["b"])
        0.0
        >>> context_precision(["x"], []) is None
        True
    """
    expected = _expected_set(expected_ids)
    if not expected:
        return None
    ranked = _unique_ids(ranked_ids)
    if not ranked:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for position, doc_id in enumerate(ranked, start=1):
        if doc_id in expected:
            hits += 1
            precision_sum += hits / position
    if hits == 0:
        return 0.0
    return precision_sum / hits
