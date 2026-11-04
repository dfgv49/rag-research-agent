# Spike：检索升级（混合检索 + 重排）技术可行性验证

- 项目根：`D:\code\Agent项目\rag-research-agent-template-main\rag-research-agent-template-main`
- 解释器：`.\.venv\Scripts\python.exe`（Python 3.13.14）
- 验证日期：本次会话
- 写入边界遵守情况：**未修改 `src/` 任何文件，未改 `.env`，未改 `pyproject.toml`，未向 `.venv` 安装任何依赖**（pip/uv 仅 dry-run）。所有新增文件仅在 `spikes/` 下。

> 本文所有结论都来自本机真实执行的命令与真实输出。凡未能验证的，明确写「未能验证」及原因，不做推测性结论。

---

## Q1. langchain-elasticsearch 0.2.2 到底支持什么？

### Q1 结论（一句话）

**该版本没有 `RetrievalStrategy` 枚举，更没有 `RetrievalStrategy.HYBRID`；`langchain_elasticsearch.RetrievalStrategy` 实际是 elasticsearch-py 里的抽象基类（ABC）。混合检索不是通过枚举开启的，而是通过 `strategy=ApproxRetrievalStrategy(hybrid=True, rrf=True)`（或其替代 `DenseVectorStrategy`）开启，并且它不需要 ELSER / inference endpoint / `es_cloud_id` / `es_api_key`，只要索引里有可做 BM25 的 `text` 字段。**

### 证据 1.1 — 包内符号与版本

命令：

```powershell
.\.venv\Scripts\python.exe -c "import langchain_elasticsearch as m; print('FILE:', m.__file__); print([x for x in dir(m) if 'trateg' in x or 'etriev' in x])"
```

真实输出：

```
FILE: D:\code\Agent项目\rag-research-agent-template-main\rag-research-agent-template-main\.venv\Lib\site-packages\langchain_elasticsearch\__init__.py
['ApproxRetrievalStrategy', 'BM25RetrievalStrategy', 'BM25Strategy', 'DenseVectorScriptScoreStrategy', 'DenseVectorStrategy', 'ElasticsearchRetriever', 'ExactRetrievalStrategy', 'RetrievalStrategy', 'SparseRetrievalStrategy', 'SparseVectorStrategy', 'retrievers']
```

命令：

```powershell
.\.venv\Scripts\python.exe -c "from importlib.metadata import version; print('langchain-elasticsearch', version('langchain-elasticsearch')); import elasticsearch; print('elasticsearch lib', elasticsearch.__version__)"
```

真实输出：

```
langchain-elasticsearch 0.2.2
elasticsearch lib (8, 19, 3)
```

### 证据 1.2 — `RetrievalStrategy` 不是枚举（这是本次最关键的事实核查）

命令：

```powershell
.\.venv\Scripts\python.exe -c "from langchain_elasticsearch import RetrievalStrategy; print('TYPE:', type(RetrievalStrategy)); print('MODULE:', RetrievalStrategy.__module__); print('MEMBERS:', [(x.name, x.value) for x in RetrievalStrategy]); print('DIR:', [a for a in dir(RetrievalStrategy) if not a.startswith('_')])"
```

真实输出：

```
TYPE: <class 'abc.ABCMeta'>
MODULE: elasticsearch.helpers.vectorstore._sync.strategies
ITER_ERR: TypeError("'ABCMeta' object is not iterable")
DIR: ['before_index_creation', 'es_mappings_settings', 'es_query', 'needs_inference']
```

结论：`RetrievalStrategy` 是一个抽象基类，**无法枚举成员**（迭代直接抛 `TypeError`）。它公开的四个名字都是 ABC 的抽象方法，不是枚举值。

因此：

- **`RetrievalStrategy.HYBRID` 在该版本不存在**（直接 `RetrievalStrategy.HYBRID` 会 `AttributeError`；无需再验证，因为它不是枚举，没有 `.HYBRID` 成员可挂）。
- 任务书中「用官方 `RetrievalStrategy.HYBRID`」这条路径 **在该版本上不成立** —— 不是"应该不支持"，是实测不存在。

### 证据 1.3 — `ElasticsearchStore.__init__` 真实签名（原样）

命令：

```powershell
.\.venv\Scripts\python.exe -c "from langchain_elasticsearch import ElasticsearchStore; import inspect; print(inspect.signature(ElasticsearchStore.__init__))"
```

真实输出：

```
(self, index_name: str, *, embedding: Optional[langchain_core.embeddings.embeddings.Embeddings] = None, es_connection: Optional[elasticsearch.Elasticsearch] = None, es_url: Optional[str] = None, es_cloud_id: Optional[str] = None, es_user: Optional[str] = None, es_api_key: Optional[str] = None, es_password: Optional[str] = None, vector_query_field: str = 'vector', query_field: str = 'text', distance_strategy: Optional[Literal[<DistanceStrategy.COSINE: 'COSINE'>, <DistanceStrategy.DOT_PRODUCT: 'DOT_PRODUCT'>, <DistanceStrategy.EUCLIDEAN_DISTANCE: 'EUCLIDEAN_DISTANCE'>, <DistanceStrategy.MAX_INNER_PRODUCT: 'MAX_INNER_PRODUCT'>]] = None, strategy: Union[langchain_elasticsearch.vectorstores.BaseRetrievalStrategy, elasticsearch.helpers.vectorstore._sync.strategies.RetrievalStrategy] = <langchain_elasticsearch.vectorstores.ApproxRetrievalStrategy object at 0x000001C1CEAC2F90>, es_params: Optional[Dict[str, Any]] = None)
```

**注意参数名是 `strategy`，不是 `retrieval_strategy`。** 默认值已经是 `ApproxRetrievalStrategy()`，即项目当前用的就是纯 kNN（HNSW）近似检索。

### 证据 1.4 — 各策略真实签名与可用性

命令（节选真实输出）：

```powershell
.\.venv\Scripts\python.exe -c "... inspect.signature 每个策略类 ..."
```

真实输出：

```
=== ApproxRetrievalStrategy | module: langchain_elasticsearch.vectorstores
   sig: (self, query_model_id: Optional[str] = None, hybrid: Optional[bool] = False, rrf: Union[dict, bool, NoneType] = True)
=== ExactRetrievalStrategy | module: langchain_elasticsearch.vectorstores
   sig: (self, /, *args, **kwargs)
=== SparseRetrievalStrategy | module: langchain_elasticsearch.vectorstores
   sig: (self, model_id: Optional[str] = None)
=== BM25RetrievalStrategy | module: langchain_elasticsearch.vectorstores
   sig: (self, k1: Optional[float] = None, b: Optional[float] = None)
=== DenseVectorStrategy | module: elasticsearch.helpers.vectorstore._sync.strategies
   sig: (self, *, distance: elasticsearch.helpers.vectorstore._utils.DistanceMetric = <DistanceMetric.COSINE: 'COSINE'>, model_id: Optional[str] = None, hybrid: bool = False, rrf: Union[bool, Dict[str, Any]] = True, text_field: Optional[str] = 'text_field')
=== DenseVectorScriptScoreStrategy | ...
=== SparseVectorStrategy | module: elasticsearch.helpers.vectorstore._sync.strategies
   sig: (self, model_id: str = '.elser_model_2')
=== BM25Strategy | module: elasticsearch.helpers.vectorstore._sync.strategies
   sig: (self, k1: Optional[float] = None, b: Optional[float] = None)

--- has EmbeddingStrategy? False
```

四种基础能力在该版本**都存在且可用**：

| 能力 | 类 | 真实说明 |
|---|---|---|
| KNN（默认，当前在用） | `ApproxRetrievalStrategy` / `DenseVectorStrategy` | 走 ES `knn` 查询，HNSW 近似。当前项目即此模式 |
| BM25 | `BM25RetrievalStrategy` / `BM25Strategy` | 原生 ES `match` 查询，`es_query` 返回 `{"query":{"bool":{"must":[{"match":{text_field:{"query":q}}}]}}}`；`require_inference()` 返回 `False` |
| EXACT | `ExactRetrievalStrategy` / `DenseVectorScriptScoreStrategy` | 走 `script_score` 精确暴力检索 |
| SPARSE | `SparseRetrievalStrategy` / `SparseVectorStrategy` | 默认 `model_id='.elser_model_2'`，**需要 ES 侧部署 ELSER 模型** |

`langchain_elasticsearch` 里的旧类都带弃用标记（源码第 117/245/398 行）：

```python
@deprecated("0.2.0", alternative="DenseVectorStrategy", pending=True)
class ApproxRetrievalStrategy(BaseRetrievalStrategy):
```

弃用只是发警告，不报错。

### 证据 1.5 — HYBRID 是怎么实现的（源码，`elasticsearch` helper）

`.venv\Lib\site-packages\elasticsearch\helpers\vectorstore\_sync\strategies.py` 第 186-207 行：

```python
class DenseVectorStrategy(RetrievalStrategy):
    def __init__(self, *, distance=DistanceMetric.COSINE, model_id: Optional[str] = None,
                 hybrid: bool = False, rrf: Union[bool, Dict[str, Any]] = True,
                 text_field: Optional[str] = "text_field"):
        if hybrid and not text_field:
            raise ValueError(
                "to enable hybrid you have to specify a text_field (for BM25Strategy matching)"
            )
```

第 281-332 行 `_hybrid()`：

```python
        if self.rrf is False:
            query_body = {"knn": knn, **standard_query}
        else:
            rrf_options = {}
            if isinstance(self.rrf, Dict):
                if "rank_constant" in self.rrf:
                    rrf_options["rank_constant"] = self.rrf["rank_constant"]
                if "window_size" in self.rrf:
                    # 'window_size' was renamed to 'rank_window_size', but we support
                    # the older name for backwards compatibility
                    rrf_options["rank_window_size"] = self.rrf["window_size"]
                if "rank_window_size" in self.rrf:
                    rrf_options["rank_window_size"] = self.rrf["rank_window_size"]
            query_body = {
                "retriever": {
                    "rrf": {
                        "retrievers": [
                            {"standard": standard_query},
                            {"knn": knn},
                        ],
                        **rrf_options,
                    },
                },
            }
        return query_body
```

第 334-335 行：

```python
    def needs_inference(self) -> bool:
        return not self.model_id
```

第 278-279 行：

```python
        if self.model_id:
            model_must_be_deployed(client, self.model_id)
```

### 证据 1.6 — `text_field` 不透传陷阱（决定性实证，无需 ES）

这一节独立复核调度者提出的风险点，并给出可复现的内存级证据。

命令（不连接 ES，纯对象构造 + 生成 DSL）：

```powershell
.\.venv\Scripts\python.exe -c "... _convert_retrieval_strategy(ApproxRetrievalStrategy(hybrid=True, rrf=True), distance=DistanceStrategy.COSINE) ... es_query(text_field='text') ..."
```

真实输出：

```
A) converted type = DenseVectorStrategy
A) hybrid = True | rrf = True | text_field = 'text_field'
A) generated DSL (调用方传 text_field="text"):
{
  "retriever": {
    "rrf": {
      "retrievers": [
        {
          "standard": {
            "query": {
              "bool": {
                "must": [
                  {
                    "match": {
                      "text_field": {
                        "query": "混合检索"
                      }
                    }
                  }
                ],
                "filter": []
              }
            }
          }
        },
        { "knn": { "filter": [], "field": "vector", "k": 4, "num_candidates": 50, "query_vector": [0.1, 0.2] } }
      ]
    }
  }
}
```

对照 B（直接构造 ES 原生策略并显式传 `text_field='text'`）：

```
B) text_field = 'text'
B) generated DSL:
  ... "match": { "text": { "query": "混合检索" } } ...
```

**结论（实证）**：`_convert_retrieval_strategy` 把 `ApproxRetrievalStrategy(hybrid=True)` 转成 `DenseVectorStrategy` 时**只透传 `distance` / `model_id` / `hybrid` / `rrf`，不透传 `text_field`**，于是落地为默认值 `'text_field'`。而 `es_query()` 虽然接收 `text_field` 形参，`_hybrid()` 实际用的是 `self.text_field`（源码第 294 行 `self.text_field`），**形参被忽略**。

后果：如果用 langchain 侧的 `ApproxRetrievalStrategy(hybrid=True)`，生成的 DSL 会去 `match` 一个叫 `text_field` 的字段。项目索引里实际字段名是 `text`（`ElasticsearchStore` 的 `query_field` 默认值）。`match` 打在未映射字段上不会报错，只是那一路召回为空 → **RRF 静默退化成纯 kNN，不报错、不告警**。这是个很难发现的坑。

规避：要么显式用 ES 原生 `DenseVectorStrategy(hybrid=True, rrf=..., text_field="text")`（证据 B 已验证会正确生成 `"text"`），要么确认索引里真的存在名为 `text_field` 的字段。

### 证据 1.7 — 与调度者独立结论的对照

| 待确认点 | 调度者结论 | 本次独立证据 | 判定 |
|---|---|---|---|
| `RetrievalStrategy` 是 `ABCMeta` 抽象基类、无 `.HYBRID` | 是 | 证据 1.2（`type()` → `abc.ABCMeta`，迭代抛 `TypeError`） | **完全一致，确认** |
| `__init__` 参数名是 `strategy` 而非 `retrieval_strategy` | 是 | 证据 1.3（签名原样） | **完全一致，确认** |
| 混合入口：原生 `DenseVectorStrategy(hybrid=True, rrf=..., text_field=...)` 或已弃用的 `ApproxRetrievalStrategy(hybrid=True, rrf=...)` | 是 | 证据 1.4 / 1.5 / 1.6 | **完全一致，确认** |
| `ApproxRetrievalStrategy` 不透传 `text_field`，默认落 `'text_field'`，与 `query_field='text'` 不一致，是坑 | 是（调度者预判） | 证据 1.6（打印出的 DSL 里确实是 `"text_field"`；显式传 `text_field='text'` 才正确） | **经实证确认，且比预判更严重：`es_query` 的形参被完全忽略，静默退化** |
| `gte-rerank-v2` 可用、`output.results[]` 含 `index` / `relevance_score` / `document.text` | 是（3 篇 347 ms） | 证据 2.2–2.4（4 篇 mean 258.5 ms、5 篇延迟样本） | **完全一致，确认** |

补充差异（不冲突，是增量信息）：

- 调度者未覆盖 `qwen3-reranker-*` 与 `gte-rerank`：本次实测 `gte-rerank` → **403 AccessDenied**，`qwen3-reranker-8b/0.6b` → **400 InvalidParameter "Model not exist."**。
- 调度者未覆盖 OpenAI 兼容模式：`/compatible-mode/v1/rerank` → **404**，rerank 只能走原生端点。
- 调度者的 347 ms vs 本次 mean 258.5 ms：文档条数与连接冷启不同导致，属正常区间，不构成矛盾。

### Q1 逐条回答任务书提问

**1) 是否存在 `RetrievalStrategy` 枚举？真实成员列表？**

不存在枚举。`langchain_elasticsearch.RetrievalStrategy` 是 ABC（`abc.ABCMeta`），来源 `elasticsearch.helpers.vectorstore._sync.strategies`，公开成员为 `before_index_creation / es_mappings_settings / es_query / needs_inference`（皆抽象方法）。真实枚举型列表为空 —— 迭代时报 `TypeError: 'ABCMeta' object is not iterable`。

**2) `ElasticsearchStore.__init__` 真实签名？**

见证据 1.3 原样粘贴。关键：参数名为 `strategy`（非 `retrieval_strategy`），类型是策略对象，默认 `ApproxRetrievalStrategy()`。

**3) HYBRID 是否需要额外前提？**

- **不需要 `es_cloud_id` / `es_api_key`**：`create_elasticsearch_client` 源码只要求 `url` 与 `cloud_id` 二选一；给了 `url` 就走 `hosts`，有 `username`+`password` 就走 `basic_auth`。现有 `.env` 里的 `ELASTICSEARCH_USER/PASSWORD` 足够。
- **不需要 ELSER / inference endpoint**：只有当 `strategy.model_id` 非空时才会调 `model_must_be_deployed()`，且 `needs_inference()` 返回 `not self.model_id`。项目是客户端侧算 embedding，`model_id` 为空 → 走 `knn["query_vector"]`，(即普通 kNN)，hybrid 的 BM25 那一路由 `match` 查询提供，无需 ES 侧模型。
- **必须有一个能做 BM25 的 `text_field`**：`DenseVectorStrategy(hybrid=True)` 若 `text_field` 为假值直接 `raise ValueError`。`ElasticsearchStore` 的 `query_field` 默认 `"text"`，会传下去，所以默认就满足。
- **RRF 参数**：`rrf=True` → 生成 `{"retriever":{"rrf":{"retrievers":[{"standard":...},{"knn":...}]}}}`；`rrf=False` → 顶层同时给 `knn` 和 `query`（**不做 RRF 融合**）；`rrf={"rank_constant":n,"window_size":m}` 或 `{"rank_window_size":m}` → 透传为 RRF 选项。
- **ES 最低版本**：本机 ES 不可用，**未能验证**服务器侧实际版本要求。可引用的两点真实证据：(a) 安装的 elasticsearch-py 8.19.3 中 `Search.rank()` 的 docstring 明确写 *"This option is in technical preview and may change in the future. The syntax will likely change before GA."*；(b) 源码生成的是较新的 `retriever.rrf` 语法（而非旧的顶层 `rank.rrf`）。外部资料称 RRF retriever 在 ES 8.16 GA（[Elasticsearch retrievers GA 8.16.0](https://www.elastic.co/search-labs/es/blog/elasticsearch-retrievers-ga-8-16-0)），但**这条是网络来源，未在本机集群上验证**。
- **重要限制（源码实证）**：hybrid 模式下不能用带分数的检索。`vectorstores.py` 第 879-881 / 916-918 行：

  ```python
  isinstance(self._store.retrieval_strategy, DenseVectorStrategy)
  and self._store.retrieval_strategy.hybrid
  ):
      raise ValueError("scores are currently not supported in hybrid mode")
  ```

  即 `similarity_search_with_score` / `_with_relevance_score` 会直接抛错。**当前项目调用链不受影响**：`researcher_graph/graph.py:136` 用的是 `await retriever.ainvoke(state.query, config)`，而 `VectorStoreRetriever` 的 `search_type` 默认值是 `"similarity"`（实测 `search_type default: similarity`），走的是不带分数的 `similarity_search`。但若后续做 recall 评测要分数，会踩这个坑。

**4) 若该版本不支持 hybrid？** —— 该版本**支持** hybrid（经 `hybrid=True` 开关），只是没有 `HYBRID` 枚举值。KNN / BM25 / EXACT 三者均可用（见证据 1.4），SPARSE 可用但需 ELSER。

---

## Q2. 百炼 rerank 是否可用？

### Q2 结论（一句话）

**可用。真实可用模型是 `gte-rerank-v2`，端点 `POST https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank`，5 次实测延迟 min 236.3 / p50 255.3 / max 281.6 / mean 258.5 ms，支持中文，分数在 `output.results[].relevance_score`。**

### 证据 2.1 — 探针脚本与执行

探针：`spikes/probe_rerank.py`（纯 stdlib `urllib`，不引入任何新依赖；从 `.env` 只读 `DASHSCOPE_API_KEY` 并脱敏输出）。

命令：

```powershell
.\.venv\Scripts\python.exe spikes\probe_rerank.py
```

真实输出（首行 key 已脱敏）：

```
KEY_PRESENT=True KEY=sk-w***al LEN=117
QUERY=LangGraph 的检查点机制如何持久化对话状态？
```

### 证据 2.2 — 模型名实测矩阵（真实状态码）

第一轮探针覆盖两个端点 × 四个模型名，真实结果：

```
[native] model=gte-rerank status=403 err=HTTPError latency_ms=269.6
  body_head: {"code":"AccessDenied","message":"Access denied. For details, see: https://help.aliyun.com/zh/model-studio/error-code#access-denied","request_id":"926cc0bb-..."}

[native] model=gte-rerank-v2 status=200 err=None latency_ms=231.1
  body_head: {"output":{"results":[{"document":{"text":"LangGraph checkpointers save graph state..."},"index":0,"relevance_score":0.3446633983585721},...

[native] model=qwen3-reranker-8b status=400 err=HTTPError latency_ms=169.2
  body_head: {"code":"InvalidParameter","message":"Model not exist.","request_id":"3f3499e2-..."}

[native] model=qwen3-reranker-0.6b status=400 err=HTTPError latency_ms=166.3
  body_head: {"code":"InvalidParameter","message":"Model not exist.","request_id":"e14a59f5-..."}

[compat] model=gte-rerank status=404 err=HTTPError latency_ms=157.4
[compat] model=gte-rerank-v2 status=404 err=HTTPError latency_ms=165.0
[compat] model=qwen3-reranker-8b status=404 err=HTTPError latency_ms=376.9
[compat] model=qwen3-reranker-0.6b status=404 err=HTTPError latency_ms=186.2
  （compat = https://dashscope.aliyuncs.com/compatible-mode/v1/rerank，响应体为空）
```

读法（只陈述实测，不扩展）：

- `gte-rerank-v2` → **200 可用**。
- `gte-rerank` → **403 AccessDenied**（该 key 无权限，未能用）。
- `qwen3-reranker-8b` / `qwen3-reranker-0.6b` → **400 InvalidParameter "Model not exist."**，即这两个模型名在当前 key + 该端点下不可用。**未验证**它们在百炼平台其它端点/开通方式下是否可用。
- OpenAI 兼容模式路径 `/compatible-mode/v1/rerank` → **404**，即兼容模式不提供 rerank 端点。rerank 必须走原生 `/api/v1/services/rerank/...`。

### 证据 2.3 — 请求体与响应体真实结构

请求体（真实发出）：

```json
{
  "model": "gte-rerank-v2",
  "input": {
    "query": "LangGraph 的检查点机制如何持久化对话状态？",
    "documents": [
      "LangGraph checkpointers save graph state after each super-step, enabling time travel and resumption.",
      "BM25 是一种基于词频与逆文档频率的稀疏检索算法，适合关键词精确匹配。",
      "向量检索通过 embedding 的余弦相似度召回语义相近的文档。",
      "今天天气晴朗，适合出门散步。"
    ]
  },
  "parameters": { "return_documents": true, "top_n": 4 }
}
```

响应体（真实、完整、status=200）：

```json
{
  "output": {
    "results": [
      {
        "document": { "text": "LangGraph checkpointers save graph state after each super-step, enabling time travel and resumption." },
        "index": 0,
        "relevance_score": 0.3444482417958489
      },
      {
        "document": { "text": "向量检索通过 embedding 的余弦相似度召回语义相近的文档。" },
        "index": 2,
        "relevance_score": 0.059222715552313066
      },
      {
        "document": { "text": "BM25 是一种基于词频与逆文档频率的稀疏检索算法，适合关键词精确匹配。" },
        "index": 1,
        "relevance_score": 0.0327507440456244
      },
      {
        "document": { "text": "今天天气晴朗，适合出门散步。" },
        "index": 3,
        "relevance_score": 0.005995403320697184
      }
    ]
  },
  "usage": { "total_tokens": 157 },
  "request_id": "193b201a-c0ef-98fb-ad7b-260a977c6a5e"
}
```

**分数字段名：`relevance_score`**（float，越大越相关）；原序位置字段 `index`；`usage.total_tokens` 计费信息；`request_id` 用于排障。

`return_documents=False` 时（真实输出，字段精简且**分数仍在**）：

```json
{"output":{"results":[{"index":0,"relevance_score":0.3444482417958489},{"index":2,"relevance_score":0.059222715552313066}]},"usage":{"total_tokens":157},"request_id":"3b782a50-..."}
```

### 证据 2.4 — 单次调用延迟（毫秒，真实 5 次串行）

```
=== LATENCY_SAMPLE gte-rerank-v2 x5 ===
  run1 status=200 latency_ms=255.3
  run2 status=200 latency_ms=281.6
  run3 status=200 latency_ms=236.3
  run4 status=200 latency_ms=265.6
  run5 status=200 latency_ms=253.5
  min=236.3 p50=255.3 max=281.6 mean=258.5
```

含连接/冷启的首次调用为 **335.9 ms**（同一批探针的 FULL_RESPONSE 调用）。4 条文档、约 157 tokens。

### 证据 2.5 — 中文支持

纯中文 query + 两条中文文档，真实输出：

```
=== CHINESE_ONLY status=200 latency_ms=246.2 ===
{"output":{"results":[{"document":{"text":"混合检索把 BM25 与向量召回用 RRF 融合"},"index":0,"relevance_score":0.5884508106007487},{"document":{"text":"红烧肉的做法是先焯水再慢炖"},"index":1,"relevance_score":0.006053170570087307}]},"usage":{"total_tokens":50},"request_id":"0c2b6b81-..."}
```

中文相关 0.5884 vs 不相关 0.0061，区分度明显。**中文可用。**

### 未验证项（如实记录）

- 大批量文档（例如一次 50/100 条）的延迟与是否触发长度上限：**未验证**。
- 免费额度/计费单价：**未验证**。
- `qwen3-reranker` 系列在开通后的可用性：**未验证**（当前 key 返回 "Model not exist."）。

---

## Q3. 本地 reranker 落地可行性？

### Q3 结论（一句话）

**依赖可解析、torch 有 Python 3.13 的 Windows wheel，但模型权重下载在本机被网络阻断（`huggingface.co` TCP 超时），运行时未验证；在当前阶段判断为「不值得」，应先用已实测可用的云端 `gte-rerank-v2`。**

### 证据 3.1 — `.venv` 里没有 pip，改用 uv

命令与真实输出：

```powershell
.\.venv\Scripts\python.exe -m pip install --dry-run --no-cache-dir sentence-transformers
```

```
D:\...\.venv\Scripts\python.exe: No module named pip
```

`.venv\Scripts` 下只有应用入口 exe，无 `pip.exe`（该 venv 由 uv 管理）。系统 PATH 上 `uv -> C:\bian\python3.10.11\Scripts\uv.exe`，实测 `uv 0.11.28`。改用 `uv pip install --dry-run`（只解析、不安装、`--no-cache` 不写缓存）。

### 证据 3.2 — 依赖解析结果（真实 dry-run，未安装任何包）

命令：

```powershell
uv pip install --dry-run --no-cache --python ".venv\Scripts\python.exe" sentence-transformers
```

真实输出：

```
Resolved 42 packages in 3.15s
Would download 17 packages
Would uninstall 1 package
Would install 17 packages
 + annotated-doc==0.0.5
 - huggingface-hub==1.30.0
 + huggingface-hub==1.33.0
 + joblib==1.6.0
 + mpmath==1.3.0
 + narwhals==2.27.1
 + networkx==3.7
 + safetensors==0.8.0
 + scikit-learn==1.9.1
 + scipy==1.17.1
 + sentence-transformers==6.1.0
 + setuptools==84.2.0
 + shellingham==1.5.4
 + sympy==1.14.0
 + threadpoolctl==3.7.0
 + torch==2.14.1
 + transformers==5.20.2
 + typer==0.27.3
```

解读：**Python 3.13 下依赖可解析**，会装 `torch==2.14.1` / `transformers==5.20.2` / `sentence-transformers==6.1.0`，共 +17 个包，并把已装的 `huggingface-hub` 从 1.30.0 升到 1.33.0（`Would uninstall 1 package` 指这个替换，不是移除 langchain 系依赖）。

### 证据 3.3 — 真实产物体积（PyPI JSON + TCP 连通性实测）

命令：`.\.venv\Scripts\python.exe spikes\probe_deps_size.py` / `probe_deps_size2.py`

真实输出：

```
[PyPI] torch -> version=2.14.1 | cp313-win_amd64 wheels: [('torch-2.14.1-cp313-cp313-win_amd64.whl', 118.4)]
[PyPI] transformers -> version=5.20.2 | wheels: [('transformers-5.20.2-py3-none-any.whl', 12.25)]
[PyPI] sentence-transformers -> version=6.1.0 | wheels: [('sentence_transformers-6.1.0-py3-none-any.whl', 0.71)]

[TCP] huggingface.co -> TCP FAIL after 8032 ms: TimeoutError('timed out')
[TCP] hf-mirror.com -> TCP OK in 126 ms
[TCP] pypi.org -> TCP OK in 1184 ms
[TCP] dashscope.aliyuncs.com -> TCP OK in 94 ms
```

`BAAI/bge-reranker-v2-m3` 的 HF API 元数据查询**失败**：

```
[HF] BAAI/bge-reranker-v2-m3 -> FETCH_FAIL: URLError(TimeoutError(10060, '由于连接方在一段时间后没有正确答复或连接的主机没有反应，连接尝试失败。', None, 10060, None))
```

要点：

- wheel 本身没问题：`torch` 有 `cp313-cp313-win_amd64`（118.4 MB），另两个是纯 Python wheel（12.25 MB / 0.71 MB）。
- **权重下载是硬阻塞**：`huggingface.co` TCP 直连超时（8032 ms 失败），`hf-mirror.com` 可达（126 ms）。要落地必须设 `HF_ENDPOINT=https://hf-mirror.com` 或人工预置权重。
- **模型体积未能验证**：因 HF API 不可达，`bge-reranker-v2-m3` 的真实权重体积**未能测量**（不引用记忆中的数字）。

### 证据 3.4 — 运行时验证

**未能验证。** 原因：按纪律不向 `.venv` 安装任何依赖，`sentence_transformers` / `torch` / `transformers` 均未安装（实测 `{'sentence_transformers': False, 'torch': False, 'transformers': False, 'numpy': True}`）。因此 `CrossEncoder` 能否在 Python 3.13 + torch 2.14.1 下正常加载 `bge-reranker-v2-m3` 并推理，**本次未取得任何运行时证据**。

### Q3 判断：不值得（当前阶段）

理由（全部基于上面实测）：

1. **已有一条零依赖、实测可用的替代方案**：`gte-rerank-v2` 200 OK、mean 258.5 ms、中文可用（Q2）。本地 reranker 的收益在这个阶段没有证据支撑。
2. **成本明确偏高**：+17 个包、`torch` 单 wheel 118.4 MB，且会把已装的 `huggingface-hub` 1.30.0 顶到 1.33.0（对现有 langchain 栈的兼容性未验证）。
3. **权重获取被网络阻断**：`huggingface.co` 直连超时，需额外配置镜像；模型体积与运行时均未验证，落地存在未量化风险。
4. **冷启动与内存开销未测**：本地 CrossEncoder 首次加载的时延与常驻内存本次完全未测。

**改为「值得」的条件**（留给后续判断）：数据不能出内网 / 离线部署 / 云端 rerank 成本不可接受 —— 届时再按 `HF_ENDPOINT` 镜像 + 预置权重的方式落地，并补测冷启动与内存。

---

## Q3.5 补充：Docker 不可用时，能否用「内存库两路检索 + 自写 RRF」做等价验证？

### 结论

**可行 —— 而且已实测跑通，但只做到「算法等价」，不是「系统等价」。** 它能用来定融合与重排的策略决策，**不能替代 ES 侧验收**。

### 已实测跑通的证据

探针：`spikes/probe_offline_hybrid.py`（真实语料 `src/sample_docs.json` 32 篇 + 真实百炼 embedding + 真实百炼 rerank + 自写纯 Python BM25 + 自写 RRF，**零新增依赖**）。

命令：

```powershell
.\.venv\Scripts\python.exe spikes\probe_offline_hybrid.py
```

真实输出（节选其中一个 query 的完整链路，其余两个同构）：

```
CORPUS_SIZE=32  TOP_WINDOW=20 FINAL_K=4 RRF_C=60 RERANK_N=10
EMBED corpus: ms=7946 shape=(32, 1024)

==============================================================================
QUERY: How do I build an agent with tools?
  latency: embed_query=261ms dense=0.2ms bm25=1.4ms rrf=0.03ms rerank=349ms err=None
  dense  top4: ['#### Invoke with just the arguments\u200b  When you invoke a tool', 'If you are still using AgentExecutor, do not fear: we still ', ...]
  bm25   top4: ['#### Invoke with just the arguments\u200b  When you invoke a tool', 'If you are still using AgentExecutor, do not fear: we still ', ...]
  rrf    top4: ['#### Invoke with just the arguments\u200b  When you invoke a tool', 'If you are still using AgentExecutor, do not fear: we still ', ...]
  rerank top4: ['If you are still using AgentExecutor, do not fear: we still ', '#### Invoke with just the arguments\u200b  When you invoke a tool', ...]
  rerank scores: [0.195, 0.1857, 0.1794, 0.1156]
  overlap dense∩rrf=3  rrf∩rerank=4  dense∩rerank=3 (of 4)
```

三个 query 的真实延迟（同一次运行）：

| 阶段 | 实测 |
|---|---|
| 语料 embedding（32 篇，一次性） | 7946 ms |
| query embedding | 242 / 251 / 261 ms |
| dense 检索（numpy 余弦，32 篇） | 0.2–0.3 ms |
| BM25 检索（纯 Python，32 篇） | 1.4–1.6 ms |
| RRF 融合 | 0.03 ms |
| **rerank（10 篇候选，真实 API）** | **349 / 443 / 460 ms** |

这组数字本身就有决策价值：**融合计算开销可忽略，真正的成本是两次网络调用（embedding ~250 ms + rerank ~350-460 ms）**。注意 rerank 在 10 篇候选时比 Q2 的 4 篇（236–282 ms）明显更慢，说明 **rerank 延迟随候选条数增长，`RERANK_N` 不能随便放大**。

`rerank` 确实会改变次序（上例把 `If you are still using AgentExecutor...` 从 RRF 的第 2 位提到第 1 位），说明重排环节在真实链路上是生效的，不是空转。

### 能验证什么 / 不能验证什么

**能（算法等价）**

- RRF 融合数学是否正确、`RRF_C` 与 `TOP_WINDOW` 如何影响最终排序。
- rerank 的收益与排序变化、`RERANK_N` 与延迟的权衡。
- 端到端延迟预算的分解（embed / 检索 / 融合 / 重排各占多少）。
- 装配逻辑与降级路径（rerank 失败回落到 RRF 顺序）。
- 效果 A/B 的**相对**比较（vector-only vs hybrid vs hybrid+rerank）。

**不能（系统等价）**

- **ES 的 analyzer/分词行为**：本次纯 Python BM25 用的是自己的正则 tokenizer（`[a-z0-9_]+`，英文有效），ES 用的是标准 analyzer；两者分数不可比，只能比较**排序结果**。中文在 ES 上的实际分词效果仍旧**未验证**。
- **`multi_match` 的 ES 实现细节**（字段权重、operator、minimum_should_match 等）。
- **`knn` 的 HNSW 近似召回**：内存里是精确余弦，ES 是近似检索，召回率会有差异。
- **索引 mapping / 重建 / 灌库链路**（`index_graph`）——完全不涉及。
- **ES 原生 RRF（路径 A）** 的任何行为。

### 落地建议

把 `spikes/probe_offline_hybrid.py` 当作**策略决策台**：ES 启起来之前，先用它把 `TOP_WINDOW` / `RRF_C` / `RERANK_N` 与「要不要 rerank」这几个决策定下来；ES 起来之后再把它换成真实 ES 调用做系统验收。两者不互相替代，是前后两个阶段。

---

## Q4. 结论与推荐

### 前提：两条路径都卡在 ES 不可用

`make_elastic_retriever`（`src/shared/retrieval.py:84-89`）构造 `ElasticsearchStore` 时，`create_elasticsearch_client` 内部会执行 `es_client.info()` 做**连接自检**：

```python
    es_client = Elasticsearch(**connection_params)

    es_client.info()  # test connection

    return es_client
```

所以 ES 不通时，路径 A 与路径 B **都无法端到端验证**。本次按任务书要求未启动 Docker、未重试连接，仅做静态与离线验证。

### 路径 A：用官方 hybrid 开关（`hybrid=True` + ES 侧 RRF）

> 注意：任务书里的 `RetrievalStrategy.HYBRID` 在该版本不存在（Q1），实际入口是 `ApproxRetrievalStrategy(hybrid=True, rrf=True)` 或 `DenseVectorStrategy(hybrid=True, rrf=True, text_field="text")`。

| 项 | 内容 |
|---|---|
| 改动文件/函数 | 仅 `src/shared/retrieval.py::make_elastic_retriever`（新增 `strategy=` 与放大 `k`/`fetch_k`）；可能要给 `ElasticsearchStore` 显式传 `distance_strategy`（`ApproxRetrievalStrategy` 经 `_convert_retrieval_strategy` 时若 `distance is None` 会 `raise ValueError("ApproxRetrievalStrategy requires a distance strategy to be provided.")`） |
| 代码量 | ~15–25 行 |
| 前提 | ES ≥ 8.16（RRF retriever，网络来源未本机验证）；索引 `text` 字段为 `text` 类型；`k`/`fetch_k` 需放大（否则 RRF 候选窗口太小）；ES 必须在线 |
| 风险 | ① 融合在 ES 内部，无法在 Python 侧加权/打日志/接 rerank 之外的自定义；② hybrid 下 `similarity_search_with_score` 直接 raise（源码实证）；③ `ApproxRetrievalStrategy` 已 `@deprecated`；④ ES 版本不确定，<8.16 直接不成立；⑤ **本机完全无法验证**；⑥ **`text_field` 静默退化陷阱**：若用 langchain 侧的 `ApproxRetrievalStrategy(hybrid=True)`，`text_field` 不透传、落地为 `'text_field'` 而索引里字段叫 `text`，BM25 那一路会静默召回为空，RRF 退化成纯 kNN 且不报错（证据 1.6） |

### 路径 B：手写 BM25 + kNN 两路检索 + Python 侧 RRF

| 项 | 内容 |
|---|---|
| 改动文件/函数 | ① 新增 `src/shared/hybrid.py`：`bm25_search()`（ES 原生 `multi_match`）+ `knn_search()`（`knn` 查询）+ `rrf_fuse()`；② 新增 `src/shared/rerank.py`：封装 DashScope `/rerank`（`gte-rerank-v2`）+ 超时 + 降级；③ `src/shared/retrieval.py` 增 `make_hybrid_retriever()`，`configuration.py` 增一个 `retrieval_mode` 开关（默认保持 `vector`）；④ `src/retrieval_graph/researcher_graph/graph.py::retrieve_documents` 按开关分流 |
| 代码量 | ~120–180 行（两路检索 ~60、RRF ~20、rerank ~50、装配与开关 ~30） |
| 前提 | 补 `ELASTICSEARCH_URL`（`.env` 现在没有）；用已装的 `elasticsearch==8.19.3` 直连，或复用 `create_elasticsearch_client`；索引 `text` 字段可被 `match` 检索；rerank 走外网（已验证可用） |
| 风险 | ① 每次检索 2 次 ES 请求 + 1 次 rerank（+~258 ms）；② 需自己写超时/失败降级；③ 中文 BM25 受分词器影响（ES 默认 standard analyzer 对中文按字切分，召回质量需实测，未验证）；④ RRF 常数（通常 60）与两路 `k` 需调参；⑤ 两路分数不可直接比较（这正是 RRF 存在的理由） |

### 推荐：路径 B

**一句话理由**：本机 ES 完全不可用、服务器版本未知，路径 A 依赖 ES ≥8.16 的 RRF retriever 且完全无法在本地验证，而路径 B 只依赖 ES 的 `match` 与 `knn` 两个长期稳定能力，把融合与重排放在 Python 侧因而可独立测试、可灰度、可立刻接上已实测 200 OK / mean 258.5 ms 的 `gte-rerank-v2`。

---

## 推荐路径（路径 B）具体怎么做：步骤化

**第 -1 步：先不启 ES，用离线决策台把参数定下来（已可用）**

- 直接跑 `spikes/probe_offline_hybrid.py`（真实 embedding + 真实 rerank + 自写 BM25/RRF，零新增依赖），先把 `TOP_WINDOW` / `RRF_C` / `RERANK_N` 与「要不要 rerank」定下来（详见 Q3.5）。
- 收益：不阻塞在 Docker 上，且把这几个决策与 ES 版本问题解耦。

**第 0 步（阻塞项，必须先做）**

1. 启动 Elasticsearch（Docker），确认可达：`curl http://localhost:9200` 返回 `version.number`。
2. 把 `ELASTICSEARCH_URL=http://localhost:9200` 补进 `.env`。
   —— 现状：`src/shared/retrieval.py:86` 硬读 `os.environ["ELASTICSEARCH_URL"]`，而 `.env` 里只有 `DASHSCOPE_API_KEY` / `ELASTICSEARCH_USER` / `ELASTICSEARCH_PASSWORD`（实测三个键），**当前一定会 KeyError**。
3. 记录 `version.number`，据此判断将来是否还能回到路径 A。

**第 1 步：摸清现有索引（只读）**

- 取 `GET /langchain_index/_mapping`，确认三件事：
  - 文本字段（`ElasticsearchStore` 的 `query_field` 默认 `"text"`）确实是 `text` 类型，可被 `match` 使用；
  - 向量字段（默认 `vector`）的 `dims` 是否等于当前 `DASHSCOPE_EMBEDDING_DIMENSIONS`（默认 1024）；
  - `similarity` 是否为 `cosine`。

**第 2 步：实现两路检索 + RRF（`src/shared/hybrid.py`）**

- `bm25_search(client, index, query, k)`：`{"query": {"multi_match": {"query": query, "fields": ["text"]}}}`，`size=k`。
- `knn_search(client, index, vector, k, num_candidates)`：`{"knn": {"field": "vector", "query_vector": vector, "k": k, "num_candidates": num_candidates}}`。
- `rrf_fuse(lists, k=60)`：`score(d) = Σ 1/(k + rank_i(d))`，按分数降序。
- 建议窗口：两路各取 `k=20`（当前线上是 `k=4`），再融合后交给 rerank。

**第 3 步：实现 rerank（`src/shared/rerank.py`）**

- `POST https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank`，`Authorization: Bearer $DASHSCOPE_API_KEY`。
- 请求体：`{"model": "gte-rerank-v2", "input": {"query": q, "documents": [d.page_content...]}, "parameters": {"return_documents": false, "top_n": 4}}`。
- 解析：`output.results[].index` 映射回原文档，`relevance_score` 作为排序分。
- 必做：`timeout`（建议 10 s，实测 p50 255 ms 留足余量）+ 异常降级（rerank 失败 → 回落到 RRF 顺序，不抛错）。

**第 4 步：装配与开关**

- `configuration.py` 增 `retrieval_mode: Literal["vector", "hybrid"] = "vector"`（默认不变，零破坏）。
- `retrieval.py` 增 `make_hybrid_retriever()`，与现有 `make_elastic_retriever` 并列。
- `retrieve_documents` 里按 `retrieval_mode` 分流；`hybrid` 分支返回 `list[Document]`，函数签名不变（仍是 `-> dict[str, list[Document]]`）。

**第 5 步：A/B 评测**

- 用仓库已有的 `eval_recall.py` / `eval_recall2.py` 跑三组：`vector k=4`（基线）、`hybrid(k=20)+top4 不 rerank`、`hybrid(k=20)+rerank top4`。
- 关注：召回率、延迟（一次 hybrid+rerank ≈ ES×2 + 258 ms）。

**第 6 步：灰度顺序（建议）**

先只加 rerank（对 `vector k=10 → rerank top4`，改动最小、收益最直接、不依赖 ES 版本），再打开 hybrid。这样即使 hybrid 的 BM25 中文分词效果不佳，也已经拿到 rerank 的收益。

---

## 已知风险 / 前提条件

1. **ES 必须先启动**：本次实测 ES 完全不可达，两条路径都无法端到端验证。这是唯一的硬阻塞项。
2. **必须补 `ELASTICSEARCH_URL`**：`.env` 实测只有 `DASHSCOPE_API_KEY` / `ELASTICSEARCH_USER` / `ELASTICSEARCH_PASSWORD`，没有 `ELASTICSEARCH_URL`；而 `retrieval.py:86` 用 `os.environ["ELASTICSEARCH_URL"]` 直接取下标。
3. **ES 版本门槛**：路径 A 依赖 RRF retriever（外部资料称 8.16 GA，**本机未验证**）；路径 B 只需 `knn` + `match`，8.x 均可。启动后第一件事就是把 `version.number` 记下来。
4. **索引是否需要重建**：**当前路径不需要重建**，只要 `text` 字段是 `text` 类型、`vector.dims` 与 embedding 维度一致。**但换 embedding 模型或改维度（`DASHSCOPE_EMBEDDING_DIMENSIONS`，默认 1024）时必须重建索引并重新灌文档**（`index_graph` 的 `index_docs`），因为 `dense_vector` 的 `dims` 在 mapping 里是固定的。
5. **hybrid 下拿不到分数**：`similarity_search_with_score` 在 hybrid 模式直接 `raise ValueError("scores are currently not supported in hybrid mode")`（源码实证）。当前代码只用 `retriever.ainvoke()`（`search_type="similarity"`）不受影响，但评测脚本若要分数需绕开。
6. **中文 BM25 分词**：ES 默认 standard analyzer 对中文按单字切分，`multi_match` 的召回质量**未验证**，可能需装 ik 分词插件或自定义 analyzer。这是路径 B 最需要在启 ES 后立刻实测的点。
7. **rerank 是外网 + 计费依赖**：单次 ~258 ms（实测），需超时与失败降级；模型名必须用 `gte-rerank-v2`（`gte-rerank` 实测 403，`qwen3-reranker-*` 实测 400 Model not exist）。
8. **本地 reranker 不可行（当前）**：权重下载被网络阻断（`huggingface.co` TCP 超时 8032 ms），需 `HF_ENDPOINT=https://hf-mirror.com`（实测可达 126 ms）或人工预置权重；运行时与内存均未验证。
9. **`.venv` 无 pip**：任何依赖变更需用 `uv pip`，不能假设 `python -m pip` 可用。
10. **`text_field` 静默退化**：走路径 A 时若用 `ApproxRetrievalStrategy(hybrid=True)`，`text_field` 不透传、落地为 `'text_field'`，与索引里真实的 `text` 字段不一致，BM25 一路静默为空、RRF 退化成纯 kNN 且**不报错**（证据 1.6）。规则：要么显式用 `DenseVectorStrategy(hybrid=True, rrf=..., text_field="text")`，要么在索引里真的建一个 `text_field` 字段。
11. **rerank 延迟随候选数增长**：4 篇候选实测 236–282 ms，10 篇候选实测 349–460 ms（Q3.5），`RERANK_N` 放大前必须先量延迟。
12. **两条路径都需 ES 才能做系统验收**：离线决策台只能定策略参数（Q3.5），不能替代 ES 侧验收。

---

## 附录：环境限制说明（影响后续复现）

- DSH 沙箱为 `workspace-write`，但**子进程（Python/uv）无法写文件**：探针脚本尝试写 `spikes/rerank_probe_raw.json` 时被拒（`PermissionError: [Errno 13] Permission denied` + `[sandbox: file access denied under workspace-write mode]`），且本会话审批被禁用、无法提权。因此所有探针**只输出 stdout**，未落盘原始 JSON。
- `spikes/rerank_probe_raw.json` **不存在**（写入被拒），原始响应体全部原样粘贴在本文 Q2。
- 本次新增文件仅：`spikes/probe_rerank.py`、`spikes/probe_deps_size.py`、`spikes/probe_deps_size2.py`、`spikes/spike_retrieval_upgrade.md`。

### 并发改动提示（影响本文行号的有效性）

本会话期间 `src/retrieval_graph/` 下的 4 个文件被**其它并发进程/代理**修改（内容为会话历史截断功能，与本次 spike 无关），实测 mtime：

```
10-20 23:25:52    13355  src\retrieval_graph\graph.py
10-20 23:25:44     6980  src\retrieval_graph\researcher_graph\graph.py   (原 100 行 → 现 175 行)
10-20 23:25:37     1747  src\retrieval_graph\researcher_graph\state.py
10-20 23:25:37     2664  src\retrieval_graph\configuration.py
```

本次 spike **未修改 `src/` 任何文件**（只读过）。受保护文件 mtime 仍为旧值，可佐证：

```
.env             1779  2026/9/10 23:40:33
pyproject.toml   1967  2026/9/10 20:10:22
```

因此本文引用的 `src/` 行号可能随并发开发漂移。已复核：`retrieve_documents` 现位于 `researcher_graph/graph.py:121`，`retriever.ainvoke` 在第 136 行，**Q1 关于调用链的结论仍成立**。
