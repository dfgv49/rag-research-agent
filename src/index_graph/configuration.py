"""Define the configurable parameters for the index graph."""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.configuration import BaseConfiguration

# This file contains sample documents to index, based on the following LangChain and LangGraph documentation pages:
# - https://python.langchain.com/v0.3/docs/concepts/
# - https://langchain-ai.github.io/langgraph/concepts/low_level/
DEFAULT_DOCS_FILE = "src/sample_docs.json"

# ---------------------------------------------------------------------------
# 文档分块参数
#
# 为什么要分块：整篇入库时一篇文档只有一个向量，长文档会被「语义稀释」——向量
# 是全文 4000 字的平均，任何单一子主题都不再突出；命中时也只能整篇塞进上下文，
# 把真正有用的几段淹没在无关内容里。
#
# 384 字的取法：中文 RAG 的常用区间是 300~600。本语料 32 篇、p50 长度 3873 字，
# 384 字约切出 8~10 块，正好是「一段完整论述」的粒度；再小（<200）单块会失去
# 上下文、检索到的片段答不了问题，再大（>800）又会回到语义稀释的老问题。
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 384

# 重叠取 chunk_size 的 1/6（≈16%，落在建议的 10%~20% 区间）。中文句子长、标点
# 稀疏，硬性按长度切经常把一句话劈成两半，重叠保证跨块的句子至少有一块是完整的。
DEFAULT_CHUNK_OVERLAP = 64

# 分隔符优先级：先按空行（段落）切，再按换行（列表项 / 标题），然后才是中文句末
# 标点，最后退到空格与空串（单字符硬切）。顺序即「尽量在语义边界处断开」。
DEFAULT_CHUNK_SEPARATORS = ("\n\n", "\n", "。", "！", "？", "；", " ", "")


@dataclass(kw_only=True)
class IndexConfiguration(BaseConfiguration):
    """Configuration class for indexing and retrieval operations.

    This class defines the parameters needed for configuring the indexing and
    retrieval processes, including embedding model selection, retriever provider choice, and search parameters.
    """

    docs_file: str = field(
        default=DEFAULT_DOCS_FILE,
        metadata={
            "description": "Path to a JSON file containing default documents to index."
        },
    )

    # -- 分块 ------------------------------------------------------------

    chunk_enabled: bool = field(
        default=True,
        metadata={
            "description": "Whether to split documents into chunks before indexing. False reproduces the legacy whole-document behaviour (one vector per document)."
        },
    )

    chunk_size: int = field(
        default=DEFAULT_CHUNK_SIZE,
        metadata={
            "description": "Maximum characters per chunk. 300-600 suits Chinese prose; larger values re-introduce the semantic dilution that chunking exists to avoid."
        },
    )

    chunk_overlap: int = field(
        default=DEFAULT_CHUNK_OVERLAP,
        metadata={
            "description": "Characters of overlap between neighbouring chunks, so a sentence straddling a boundary survives intact somewhere. 10-20% of chunk_size."
        },
    )

    chunk_separators: list[str] = field(
        default_factory=lambda: list(DEFAULT_CHUNK_SEPARATORS),
        metadata={
            "description": "Separators tried in order when splitting: blank line, newline, then Chinese sentence terminators, space, and finally a hard character cut."
        },
    )
