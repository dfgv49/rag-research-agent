"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""

import json
from typing import Optional, Sequence

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from index_graph.configuration import IndexConfiguration
from index_graph.state import IndexState
from shared import retrieval
from shared.state import _generate_uuid, reduce_docs

# chunk_id 的连接符。要求：稳定、可复算、对 Elasticsearch 的 _id 合法
# （ES 的 _id 上限 512 字节，且不允许 '#'、'?'；'::' 安全）。
CHUNK_ID_SEPARATOR = "::"


def _recursive_split(
    text: str, chunk_size: int, separators: Sequence[str]
) -> list[str]:
    """按分隔符优先级把文本递归切到不超过 chunk_size 的片段。

    这是 `RecursiveCharacterTextSplitter` 的等价实现，用作 `langchain_text_splitters`
    未安装时的兜底（本项目不允许安装依赖，所以兜底必须自己写）。语义一致：从优先级
    最高的分隔符开始切，切出来的片段仍然超长就换下一个分隔符继续递归，最后才硬切。

    Args:
        text: 待切分文本。
        chunk_size: 单片段最大字符数。
        separators: 分隔符，按优先级从高到低排列；空串表示「按字符硬切」。

    Returns:
        list[str]: 片段列表。残留的分隔符会保留在片段末尾，避免丢标点。
    """
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    separator = ""
    remaining: Sequence[str] = ()
    for index, candidate in enumerate(separators):
        if candidate and candidate in text:
            separator = candidate
            remaining = separators[index + 1 :]
            break

    if not separator:
        # 没有可用分隔符（或只剩空串）：按字符硬切，保证一定能切小。
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(separator)
    pieces: list[str] = []
    for index, part in enumerate(parts):
        # 把分隔符接回前一个片段，这样句子标点不会被切走。
        piece = part if index == len(parts) - 1 else part + separator
        if not piece:
            continue
        if len(piece) > chunk_size:
            pieces.extend(_recursive_split(piece, chunk_size, remaining))
        else:
            pieces.append(piece)
    return pieces


def _merge_pieces(
    pieces: Sequence[str], chunk_size: int, chunk_overlap: int
) -> list[str]:
    """把细碎片段贪心地合并成接近 chunk_size 的块，并按 chunk_overlap 加重叠。

    Args:
        pieces: 已经切到不超过 chunk_size 的片段。
        chunk_size: 单块最大字符数。
        chunk_overlap: 相邻块之间重叠的字符数。

    Returns:
        list[str]: 去掉首尾空白后的块列表（空块会被丢弃）。
    """
    step = max(chunk_size - chunk_overlap, 1)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current)
            # 保留上一块的结尾作为重叠部分，避免跨块的句子被截断。
            current = current[-chunk_overlap:] if chunk_overlap > 0 else ""
        current += piece
        # 单个片段本身可能仍超过 chunk_size（例如没有标点的长串）。
        while len(current) > chunk_size:
            chunks.append(current[:chunk_size])
            current = current[step:]
    if current:
        chunks.append(current)
    return [stripped for stripped in (c.strip() for c in chunks) if stripped]


def _simple_split(text: str, chunk_size: int, chunk_overlap: int, separators: Sequence[str]) -> list[str]:
    """不依赖 langchain 的分块实现（兜底路径）。"""
    return _merge_pieces(
        _recursive_split(text, chunk_size, separators), chunk_size, chunk_overlap
    )


def _split_text(
    text: str, *, chunk_size: int, chunk_overlap: int, separators: Sequence[str]
) -> list[str]:
    """切分单篇文档正文，优先用 langchain 的 splitter。"""
    if not text or not text.strip():
        return []
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        return _simple_split(text, chunk_size, chunk_overlap, separators)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(separators),
        keep_separator="end",
        length_function=len,
    )
    return [piece for piece in splitter.split_text(text) if piece.strip()]


def split_documents(
    docs: Sequence[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
    separators: Sequence[str],
) -> list[Document]:
    """把整篇文档切成带溯源元数据的 chunk。

    每个 chunk 的 metadata 带上：

    - ``source_uuid``: 原文档的 uuid（``_generate_uuid(原始 page_content)``），
      用于把 chunk 映射回文档；引用回链靠它。
    - ``chunk_index``: 该 chunk 在原文档中的序号，从 0 开始。
    - ``chunk_id``: ``f"{source_uuid}::{chunk_index}"``，稳定的向量库主键。

    原文档 metadata 的其它字段（如 ``source``）原样保留。

    注意：这里**故意不写** ``metadata["uuid"]``。``shared.state.reduce_docs`` 与
    ``retrieval.InMemoryHybridRetriever`` 都按 ``metadata["uuid"]`` 去重，若 chunk
    沿用原文档的 uuid，同一篇文档的所有 chunk 会被判成重复而塌缩成一条——分块就白做了。
    留空则下游按 chunk 正文各自算 uuid，正好一块一个身份。

    Args:
        docs: 待切分的整篇文档。
        chunk_size: 单块最大字符数。
        chunk_overlap: 相邻块重叠字符数。
        separators: 分隔符优先级。

    Returns:
        list[Document]: 拼好的 chunk 列表，顺序是「文档顺序 → 块顺序」。
    """
    chunks: list[Document] = []
    for doc in docs:
        content = doc.page_content or ""
        source_uuid = doc.metadata.get("uuid") or _generate_uuid(content)
        base_metadata = {
            key: value for key, value in (doc.metadata or {}).items() if key != "uuid"
        }
        for chunk_index, text in enumerate(
            _split_text(
                content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
            )
        ):
            chunks.append(
                Document(
                    page_content=text,
                    metadata={
                        **base_metadata,
                        "source_uuid": source_uuid,
                        "chunk_index": chunk_index,
                        "chunk_id": f"{source_uuid}{CHUNK_ID_SEPARATOR}{chunk_index}",
                    },
                )
            )
    return chunks


async def index_docs(
    state: IndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
    """Asynchronously index documents in the given state using the configured retriever.

    This function takes the documents from the state, ensures they have a user ID,
    adds them to the retriever's index, and then signals for the documents to be
    deleted from the state.

    If docs are not provided in the state, they will be loaded
    from the configuration.docs_file JSON file.

    Args:
        state (IndexState): The current state containing documents and retriever.
        config (Optional[RunnableConfig]): Configuration for the indexing process.r
    """
    if not config:
        raise ValueError("Configuration required to run index_docs.")

    configuration = IndexConfiguration.from_runnable_config(config)
    docs = state.docs
    if not docs:
        # 显式指定 utf-8：Windows 上默认编码是 GBK，若文档中含有中文等非 ASCII
        # 字符，不指定编码会直接抛 UnicodeDecodeError。
        with open(configuration.docs_file, encoding="utf-8") as f:
            serialized_docs = json.load(f)
            docs = reduce_docs([], serialized_docs)

    # 用稳定 id 作为向量库主键，使重复索引变成幂等操作：同一份内容即使写入多次，
    # 也只会覆盖同一条记录，而不是产生副本。若不传 ids，写入方会为每次调用生成
    # 随机 id，导致重复灌入时文档成倍堆积，而这些副本会占满检索结果的前几名，
    # 把真正多样的文档挤出去。
    if configuration.chunk_enabled:
        # 分块后每条记录是一个 chunk，主键是 chunk_id（源 uuid + 块序号），
        # 因为源 uuid 与块序号都由内容决定，所以同样幂等。
        records = split_documents(
            docs,
            chunk_size=configuration.chunk_size,
            chunk_overlap=configuration.chunk_overlap,
            separators=configuration.chunk_separators,
        )
        ids = [record.metadata["chunk_id"] for record in records]
    else:
        # 旧行为：整篇一个向量。
        records = list(docs)
        ids = [
            doc.metadata.get("uuid") or _generate_uuid(doc.page_content)
            for doc in records
        ]

    with retrieval.make_retriever(config) as retriever:
        await retriever.aadd_documents(records, ids=ids)

    return {"docs": "delete"}


# Define the graph
builder = StateGraph(IndexState, config_schema=IndexConfiguration)
builder.add_node(index_docs)
builder.add_edge(START, "index_docs")
builder.add_edge("index_docs", END)
# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "IndexGraph"
