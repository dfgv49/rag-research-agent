"""磁盘级 embedding 缓存：让重复评测几乎零成本。

为什么需要
----------
`text-embedding-v4` 在本项目账号上很慢（实测 229 篇文档约 475~603 秒），
而且**每次跑评测都要重新付费重新等**。结果就是：改一行评测代码 → 重付十分钟
和一笔额度 → 谁都不敢迭代。

这个模块把「文本 → 向量」按内容哈希缓存到磁盘，命中就不再发请求。

设计要点
--------
1. **按内容哈希做 key**（sha256），不按文档 id —— 语料改了但某篇没变，它仍然命中。
2. **文档与查询分命名空间**（`d:` / `q:`）。同一个模型下两者向量通常相同，但
   不同实现可能行为不同，分开存才不会串。
3. **向量存 float32**，比 JSON 小一半以上，且与 numpy 余弦运算精度一致。
4. **写入走原子替换**（先写临时文件再 `os.replace`），避免跑到一半崩掉把缓存写坏。
5. **缓存损坏不致命**：读不出来就当空缓存，重新算，不抛异常。

⚠️ 沙箱注意
-----------
Windows 沙箱下 Python 子进程**不能在工作区新建文件**，所以缓存文件默认落在
`$env:TEMP`。这是有意为之：缓存是中间产物，不该进仓库。

用法
----
```python
from embed_cache import DiskCachedEmbeddings

cached = DiskCachedEmbeddings(base_embeddings, cache_path=...)
vectors = cached.embed_documents(texts)   # 命中走缓存，未命中才调 API
print(cached.stats())                     # 看命中率
```

命令行自检（不消耗额度，只打印缓存内容）：

    .\\.venv\\Scripts\\python.exe eval\\embed_cache.py --stats
"""

from __future__ import annotations

import hashlib
import os
import pickle
import sys
from typing import Any, Iterable, Optional

import numpy as np
from langchain_core.embeddings import Embeddings

__all__ = ["DiskCachedEmbeddings", "default_cache_path", "derive_model_tag"]


def default_cache_path(name: str = "embed_cache.pkl") -> str:
    """缓存默认放在 TEMP（工作区不可写的沙箱环境下唯一可行位置）。"""
    return os.path.join(os.environ.get("TEMP") or ".", name)


def derive_model_tag(inner: Any) -> str:
    """从 Embeddings 实例推导「模型身份」，用于隔离不同模型的向量。

    为什么必须做：缓存 key 若只含文本哈希，**换 embedding 模型后会静默命中旧向量**
    ——不报错、不提示，得到一个看起来正常但完全错误的结果。这类 bug 极难排查，
    所以模型名与维度必须进 key。

    Args:
        inner: 被包装的 Embeddings 实例。

    Returns:
        str: 形如 ``text-embedding-v4|1024`` 的标签；推导不出时退化为类名。
    """
    parts: list[str] = []
    for attr in ("model", "model_name", "deployment", "model_id"):
        value = getattr(inner, attr, None)
        if isinstance(value, str) and value:
            parts.append(value)
            break
    dimensions = getattr(inner, "dimensions", None)
    if dimensions is None:
        dimensions = getattr(inner, "dim", None)
    if dimensions is not None:
        parts.append(str(dimensions))
    return "|".join(parts) if parts else type(inner).__name__


def _text_key(prefix: str, tag: str, text: str) -> str:
    """内容哈希 + 命名空间 + 模型身份。

    ⚠️ 模型身份（``tag``）必须参与哈希：否则换模型后会命中旧向量且不报错。
    """
    payload = f"{prefix}|{tag}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DiskCachedEmbeddings(Embeddings):
    """给任意 Embeddings 加一层磁盘缓存。

    Args:
        inner: 真正干活的 Embeddings 实例（例如 OpenAIEmbeddings）。
        cache_path: pickle 缓存文件路径。父目录不存在会自动创建。
        model_tag: 模型身份标签；默认从 ``inner`` 推导（模型名 + 维度）。
            不同标签的缓存互不命中 —— 这是「换模型不会误用旧向量」的保证。
        verbose: 是否打印首次加载信息。
    """

    def __init__(
        self,
        inner: Embeddings,
        cache_path: Optional[str] = None,
        model_tag: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        self.inner = inner
        self.cache_path = cache_path or default_cache_path()
        self.model_tag = model_tag or derive_model_tag(inner)
        self._data: dict[str, np.ndarray] = {}
        self._hits = 0
        self._misses = 0
        self._dirty = False
        self._load(verbose)

    # ---- 缓存文件读写 ----------------------------------------------------

    def _load(self, verbose: bool) -> None:
        if not os.path.exists(self.cache_path):
            if verbose:
                print(f"[embed-cache] 新建缓存：{self.cache_path}")
            return
        try:
            with open(self.cache_path, "rb") as handle:
                payload = pickle.load(handle)
            if isinstance(payload, dict):
                self._data = payload
                if verbose:
                    print(f"[embed-cache] 载入 {len(self._data)} 条向量：{self.cache_path}")
        except Exception as exc:  # 缓存坏了不能拖垮评测
            print(f"[embed-cache] 缓存读取失败，按空缓存处理：{type(exc).__name__}: {exc}")
            self._data = {}

    def save(self) -> None:
        """原子写盘。父目录不可创建时给出清晰提示而不是裸崩。"""
        if not self._dirty:
            return
        parent = os.path.dirname(self.cache_path)
        try:
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp = self.cache_path + ".tmp"
            with open(tmp, "wb") as handle:
                pickle.dump(self._data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, self.cache_path)
            self._dirty = False
        except OSError as exc:
            print(
                f"[embed-cache] 缓存写入失败（不影响本次结果）：{type(exc).__name__}: {exc}\n"
                f"[embed-cache] 提示：沙箱下请把缓存指到 $env:TEMP —— {self.cache_path}"
            )

    # ---- 统计 ------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "cached": len(self._data),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (self._hits / total) if total else 0.0,
            "path": self.cache_path,
            "model_tag": self.model_tag,
        }

    # ---- Embeddings 协议 -------------------------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量取向量；只对未命中的文本调 API。"""
        keys = [_text_key("d:", self.model_tag, t) for t in texts]

        missing: list[tuple[str, str]] = []
        seen: set[str] = set()
        for key, text in zip(keys, texts):
            if key in self._data:
                self._hits += 1
            elif key not in seen:
                seen.add(key)
                missing.append((key, text))
            # 同一批里重复文本也只请求一次

        if missing:
            vectors = self.inner.embed_documents([text for _, text in missing])
            for (key, _), vector in zip(missing, vectors):
                self._data[key] = np.asarray(vector, dtype=np.float32)
            self._misses += len(missing)
            self._dirty = True
            self.save()

        return [self._data[key].tolist() for key in keys]

    def embed_query(self, text: str) -> list[float]:
        """单条查询取向量（独立命名空间，避免与文档向量串用）。"""
        key = _text_key("q:", self.model_tag, text)
        if key in self._data:
            self._hits += 1
            return self._data[key].tolist()

        self._misses += 1
        vector = np.asarray(self.inner.embed_query(text), dtype=np.float32)
        self._data[key] = vector
        self._dirty = True
        self.save()
        return vector.tolist()

    # ---- 供外部批量灌入（例如从别的脚本迁移已有向量）---------------------

    def prime(self, texts: Iterable[str], vectors: Iterable[Any], *, namespace: str = "d:") -> int:
        """把已有向量直接写进缓存，不调 API。返回写入条数。"""
        count = 0
        for text, vector in zip(texts, vectors):
            self._data[_text_key(namespace, self.model_tag, text)] = np.asarray(vector, dtype=np.float32)
            count += 1
        if count:
            self._dirty = True
            self.save()
        return count


def main() -> int:
    """只读自检：打印缓存内容统计，不发起任何网络请求。"""
    import argparse

    parser = argparse.ArgumentParser(description="查看 embedding 磁盘缓存（只读，不调 API）")
    parser.add_argument("--cache", default=default_cache_path(), help="缓存文件路径")
    args = parser.parse_args()

    if not os.path.exists(args.cache):
        print(f"缓存不存在：{args.cache}")
        return 1

    with open(args.cache, "rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        print(f"缓存格式异常（应为 dict，实际 {type(data).__name__}）")
        return 1

    docs = sum(1 for k in data if k.startswith("d:"))
    queries = sum(1 for k in data if k.startswith("q:"))
    total_floats = sum(int(np.asarray(v).size) for v in data.values())
    print(f"缓存文件：{args.cache}")
    print(f"  总条数  ：{len(data)}（文档 {docs} / 查询 {queries}）")
    print(f"  向量维度：{int(np.asarray(next(iter(data.values()))).size)}")
    print(f"  占用估算：{total_floats * 4 / 1024 / 1024:.1f} MB（float32）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
