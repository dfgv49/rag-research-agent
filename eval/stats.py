"""统计显著性工具：把「方向」升级成「可信」。

为什么需要
----------
评测集只有几十条用例时，`mrr` 差 0.05 完全可能是噪声。直接报「B 比 A 差 0.1187」
是不严谨的 —— 面试官一句「这会不会是随机波动」就能问住。

本模块提供两件事：

1. **置信区间**：对某个指标给出「真实值大概在什么范围」。
2. **配对显著性检验**：对「同一批用例上两个配置的差值」判断是否显著不为 0。

为什么用「配对」而不是「两组独立」
--------------------------------
两个配置跑的是**同一批用例**（第 i 条用例在两个配置下的表现天然相关）。
配对方法把「用例难度差异」消掉，只保留「配置差异」，统计效力高得多 ——
同样的样本量能检出更小的真实差异。

实现选择：**bootstrap（自助法）**
--------------------------------
不假设分布、不依赖 scipy，纯 numpy 即可。检索评测常用做法。

⚠️ 使用前提：`a` 与 `b` 必须**逐条对齐**（第 i 个元素是同一道题）。
   如果你的两个列表顺序不同，结果毫无意义。
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

__all__ = [
    "bootstrap_ci",
    "paired_bootstrap",
    "paired_summary",
    "format_ci",
]


def _clean(values: Sequence[Optional[float]]) -> list[float]:
    """去掉 None（指标未定义，例如 no_answer 类没有检索期望文档）。"""
    return [float(v) for v in values if v is not None]


def bootstrap_ci(
    values: Sequence[Optional[float]],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 20261022,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """对单个指标的均值做 bootstrap 置信区间。

    Args:
        values: 逐用例的指标值（可含 None，会被剔除）。
        n_boot: 重采样次数，默认 2000。
        alpha: 显著性水平，0.05 → 95% 置信区间。
        seed: 固定随机种子，保证结果可复现。

    Returns:
        tuple: ``(mean, ci_low, ci_high)``。无有效值时全为 None。

    Examples:
        >>> m, lo, hi = bootstrap_ci([0.0, 1.0, 1.0, 1.0], n_boot=200)
        >>> round(lo, 2) <= round(m, 2) <= round(hi, 2)
        True
    """
    data = _clean(values)
    if not data:
        return None, None, None

    n = len(data)
    mean = sum(data) / n
    if n == 1:
        return mean, mean, mean

    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [data[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()

    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot) - 1
    hi_idx = max(0, min(hi_idx, n_boot - 1))
    return mean, means[lo_idx], means[hi_idx]


def paired_bootstrap(
    a: Sequence[Optional[float]],
    b: Sequence[Optional[float]],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 20261022,
) -> dict[str, Optional[float]]:
    """配对 bootstrap：检验 ``mean(b) − mean(a)`` 是否显著不为 0。

    Args:
        a: 配置 A 的逐用例指标（基线）。
        b: 配置 B 的逐用例指标，**必须与 a 逐条对齐**。
        n_boot: 重采样次数。
        alpha: 显著性水平。
        seed: 固定种子。

    Returns:
        dict，含：
            - ``delta``      : mean(b) − mean(a)
            - ``ci_low`` / ``ci_high`` : 差值的 95% 置信区间
            - ``significant``: 置信区间是否不跨 0（等价于双侧 alpha 显著）
            - ``n``          : 参与配对的用例数
            - ``wins`` / ``losses`` / ``ties`` : 逐条看，B 比 A 好/差/平的条数
            - ``p_boot``     : 用 bootstrap 分布估计的双侧 p 值（近似）
    """
    pairs = [(float(x), float(y)) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return {
            "delta": None, "ci_low": None, "ci_high": None, "significant": None,
            "n": 0, "wins": 0, "losses": 0, "ties": 0, "p_boot": None,
        }

    diffs = [y - x for x, y in pairs]
    delta = sum(diffs) / n

    wins = sum(1 for d in diffs if d > 1e-9)
    losses = sum(1 for d in diffs if d < -1e-9)
    ties = n - wins - losses

    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(s) / n)
    boots.sort()

    lo_idx = max(0, int((alpha / 2) * n_boot))
    hi_idx = min(n_boot - 1, int((1 - alpha / 2) * n_boot) - 1)
    ci_low, ci_high = boots[lo_idx], boots[hi_idx]
    significant = not (ci_low <= 0.0 <= ci_high)

    # 以观测到的 delta 为枢轴，估计「差值 ≤ 0」的概率（双侧近似）
    if delta == 0:
        p_boot = 1.0
    else:
        if delta > 0:
            extreme = sum(1 for x in boots if x <= 0)
        else:
            extreme = sum(1 for x in boots if x >= 0)
        p_boot = min(1.0, 2 * extreme / n_boot)

    return {
        "delta": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant": significant,
        "n": n,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "p_boot": p_boot,
    }


def paired_summary(
    a: Sequence[Optional[float]],
    b: Sequence[Optional[float]],
    *,
    metric: str = "",
    n_boot: int = 2000,
    seed: int = 20261022,
) -> str:
    """把 ``paired_bootstrap`` 的结果渲染成一行可读文本。"""
    r = paired_bootstrap(a, b, n_boot=n_boot, seed=seed)
    if r["delta"] is None:
        return f"{metric}: 无可配对数据"
    flag = "显著" if r["significant"] else "**不显著**"
    sign = "+" if r["delta"] >= 0 else ""
    return (
        f"{metric}: {sign}{r['delta']:.4f} "
        f"[95% CI {r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
        f"p≈{r['p_boot']:.3f} → {flag} | "
        f"逐条 {r['wins']}胜/{r['losses']}负/{r['ties']}平 (n={r['n']})"
    )


def format_ci(values: Sequence[Optional[float]], n_boot: int = 2000, seed: int = 20261022) -> str:
    """把 ``bootstrap_ci`` 渲染成 ``0.9333 [0.8812, 0.9781]`` 这样的短文本。"""
    mean, lo, hi = bootstrap_ci(values, n_boot=n_boot, seed=seed)
    if mean is None:
        return "n/a"
    return f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"


if __name__ == "__main__":
    # 自检：跑几个可预期的例子，不入库、不联网
    print("=== 1) 明显差异应判显著 ===")
    a = [1.0] * 40 + [0.0] * 20          # 60 条，A 有 20 条完全失败
    b = [1.0] * 60                        # B 全对
    print(paired_summary(a, b, metric="demo_mrr"))

    print("\n=== 2) 微小差异应判不显著 ===")
    a = [1.0, 0.0] * 30
    b = [1.0, 0.0] * 29 + [0.0, 1.0]
    print(paired_summary(a, b, metric="tiny_diff"))

    print("\n=== 3) 单指标置信区间 ===")
    vals = [1.0] * 40 + [0.0] * 20
    print("mean [95% CI]:", format_ci(vals))

    print("\n=== 4) 空值处理 ===")
    print(paired_summary([None, None], [1.0, 0.0], metric="all_none"))
    print("ci:", format_ci([None]))
