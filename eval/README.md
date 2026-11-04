# `eval/` —— RAG Research Agent 评测体系

一套可重复运行的分层评测基座：**检索层用纯函数算指标，生成层用自实现的 LLM-as-judge 打分**，
不依赖 Elasticsearch、不依赖 ragas、不安装任何新依赖。

---

## 1. 目录结构

```
eval/
├── testcases.yaml       # 分层测试集：5 类 × 12 条 = 60 条（24 真实 + 36 TODO 占位）
├── metrics.py           # 检索指标纯函数：recall_at_k / mrr / ndcg_at_k / context_precision
├── judge.py             # 自实现 LLM-as-judge：faithfulness / answer_relevancy / context_recall
├── run_eval.py          # 一条命令跑完整评测，产出 .md + .json 基线报告
├── README.md            # 本文件
└── reports/             # 报告输出目录（运行时自动创建）
    ├── baseline_<YYYYMMDD_HHMM>.md            # 人读
    ├── baseline_<YYYYMMDD_HHMM>.json          # 机读
    └── baseline_<YYYYMMDD_HHMM>_rewrite.md    # 开了 --rewrite 时的对比报告
```

| 文件 | 职责 | 是否联网 |
|---|---|---|
| `metrics.py` | 纯函数，输入输出都是列表/浮点数 | 否（只 import `math`） |
| `judge.py` | 调 dashscope 模型打分，内置降级 | 是（仅在打分时） |
| `run_eval.py` | 建向量库、跑用例、聚合、渲染报告 | 是（embedding + 对话） |
| `testcases.yaml` | 数据，无代码 | 否 |

---

## 2. 怎么跑（Windows / pwsh，工作目录 = 项目根目录）

> **重要：命令里都带了 `--output-dir "$env:TEMP\rag_eval_reports"`。**
> 在很多沙箱 / 受限权限的 DSH 环境里，Python 子进程**无法在项目目录内新建文件或目录**，
> 会直接抛 `PermissionError` / `WinError 5`（拒绝访问）。把输出目录指到系统临时目录就能绕开，
> 报告内容完全一样，只是落盘位置不同。
> 如果你的环境没有这个限制，可以省略该参数，报告会写到默认的 `eval/reports/`。

```powershell
# 全量：检索指标 + 生成侧 judge（约 3 分钟，消耗对话模型额度）
.\.venv\Scripts\python.exe eval\run_eval.py --output-dir "$env:TEMP\rag_eval_reports"

# 只跑检索指标：不发起任何对话模型调用，约 15 秒，不消耗对话额度
.\.venv\Scripts\python.exe eval\run_eval.py --skip-judge --output-dir "$env:TEMP\rag_eval_reports"

# 开启查询改写，并在报告里给出带 history 用例的改写前后对比（约 20 秒）
.\.venv\Scripts\python.exe eval\run_eval.py --rewrite --skip-judge --output-dir "$env:TEMP\rag_eval_reports"

# 改写 + 全量 judge
.\.venv\Scripts\python.exe eval\run_eval.py --rewrite --output-dir "$env:TEMP\rag_eval_reports"

# 冒烟：只跑前 6 条真实用例
.\.venv\Scripts\python.exe eval\run_eval.py --limit 6 --skip-judge
```

| 参数 | 作用 |
|---|---|
| `--skip-judge` | 只跑检索指标，跳过生成与 judge（**不发起任何对话模型调用**） |
| `--rewrite` | 对带 `history` 的用例做查询改写，并输出改写前后对比（默认关闭） |
| `--limit N` | 只跑前 N 条**真实**用例（TODO 不计入），用于冒烟 |
| `--output-dir DIR` | 报告输出目录，默认 `eval/reports`；沙箱环境请指到 `$env:TEMP` 下 |
| `--testcases PATH` | 指定测试集，默认 `eval/testcases.yaml` |
| `--corpus PATH` | 指定语料，默认 `src/sample_docs.json`；换大语料传 `src/sample_docs_large.json` |
| `--max-embed-chars N` | **嵌入时**单篇字符上限，默认 `8000`；传 `0` 表示不截断（见第 2.1 节） |
| `--answer-model M` | 覆盖作答模型，例如 `dashscope/qwen-plus` |
| `--judge-model M` | 覆盖 judge 模型 |
| `--rewrite-model M` | 覆盖改写模型，例如 `dashscope/qwen-flash` |

**前置条件**：项目根目录有 `.env`，且含 `DASHSCOPE_API_KEY`（可选 `DASHSCOPE_BASE_URL`）。
`run_eval.py` 用**绝对路径**显式加载 `<项目根>/.env`，所以从任何工作目录运行都能读到 Key。

### 2.1 大语料与嵌入截断（`--max-embed-chars`）

**问题**：`text-embedding-v4` 对单次输入的长度有上限（约 33000），超出直接报

```
openai.BadRequestError: Error code: 400
InvalidParameter: Range of input length should be [1, 33000]
```

旧语料 `src/sample_docs.json` 最长只有 3996 字符，所以从没暴露过。新语料
`src/sample_docs_large.json`（229 篇，最长 **136352** 字符）一跑就崩——
其中 **150 篇**超过 8000 字符、45 篇超过 33000 字符。

**修复**：新增 `--max-embed-chars`（默认 `8000`），在**计算嵌入向量之前**把每篇
`page_content` 按 `content[:N]` 截断（Python str 按码位切，不会切开中文 / emoji）。

**关键设计：截断只作用于 embedding，入库正文与 context 仍是全文。**
实现方式是把 embedding 客户端包一层（`make_truncating_encoder`），而不是截断文档本身：

```python
# langchain_core/vectorstores/in_memory.py::add_documents
texts = [doc.page_content for doc in documents]
vectors = self.embedding.embed_documents(texts)   # ← 只有这里拿到（被截断的）文本
...
self.store[doc_id_] = {..., "text": doc.page_content, ...}   # ← 存的仍是原文
```

所以向量由「前 8000 字符」算出，但 `InMemoryVectorStore` 里存的 `text`、
以及后续拼进 prompt 的 context 都是**完整原文**，没有因这个修复而丢失内容。
控制台会同时打印一行提示，例如：

```
  语料长度：最长 136352 字符，最短 794 字符
  嵌入前截断：150 篇超过 8000 字符（最长 136352），仅对嵌入生效
  提醒：入库正文与拼给模型的 context 仍是全文，未截断。
```

报告头部也会记一行：语料篇数、最大长度、是否启用截断、截断后送入嵌入的最大长度。

**默认值对历史基线零影响**：旧语料最长 3996 < 8000，不会有任何文档被截断，
因此七项指标与历史完全一致（已实测回归）。

```powershell
# 大语料 + 匹配的合成测试集
.\.venv\Scripts\python.exe eval\run_eval.py --corpus src\sample_docs_large.json `
    --testcases "$env:TEMP\cases_synthetic.yaml" --skip-judge `
    --output-dir "$env:TEMP\rag_eval_reports"

# 关掉截断（大语料下会重新报 400，仅用于确认上限行为）
.\.venv\Scripts\python.exe eval\run_eval.py --corpus src\sample_docs_large.json --max-embed-chars 0
```

> ⚠️ **换语料必须同时换测试集**：`eval/testcases.yaml` 里 24 条真实用例的
> `expected_doc_uuids` 全部指向旧 32 篇语料，在大语料上跑会得到一堆 0 分——
> 那不是模型退步，而是标注对不上。大语料请用与之匹配的合成测试集。

### 附：验证指标函数本身

```powershell
.\.venv\Scripts\python.exe -c "from eval.metrics import recall_at_k; print(recall_at_k(['a','b','c'], ['b'], 4))"
.\.venv\Scripts\python.exe -m doctest eval\metrics.py -v
```

---

## 3. `--rewrite`：查询改写模式

### 它解决什么问题

`multi_turn` 类用例的当前问题是**带指代的追问**，例如历史问「LangGraph 里节点是怎么定义的」，
当前问「**那它**和边有什么区别？」。这种问句单独拿去检索几乎没有信号。

`--rewrite` 做的事：把 `history + question` 交给改写模型（默认 `qwen-flash`），
改写成一句**自包含、可独立检索**的完整问题，再用它去检索。不带 `history` 的用例保持原样。

**为什么要做这个对比**：如果不做，改写逻辑有没有生效、有没有用，指标上完全看不出来。
开了 `--rewrite` 之后，报告里会多一节「查询改写前后对比」：

- 汇总表：带 `history` 用例的 `recall@4` / `mrr` / `ndcg@10` 在改写前后各是多少、差值多少；
- 逐条表：每条用例的**改写前后的 query 原文**、改写后的 `recall@4` / `mrr` 变化。

「改写前」的基线是 `history + question` 直接拼接去检索（见 `build_query()`），
「改写后」是模型改写出的自包含 query。两者用的是同一套 embedding 与检索逻辑，可直接对比。

JSON 里每条用例都有 `baseline_query` / `rewritten_query` / `rewrite_error` /
`retrieval_baseline` / `retrieval` 字段，`rewritten_query` 是**原样**保存的，便于人工检查改写质量。

**改写失败不崩**：API 报错、超时、返回空 → 降级为使用原查询，在结果里记 `rewrite_error`
和一条 `errors` 记录，该用例照常参与评测。

### 本次实测结论（重要）

在本语料上，改写**质量很好但检索指标没有提升**：

| 指标 | 改写前 | 改写后 | 差值 |
|---|---:|---:|---:|
| recall@4 | 0.8750 | 0.8750 | +0.0000 |
| mrr | 0.5417 | 0.5417 | +0.0000 |
| ndcg@10 | 0.6486 | 0.6439 | -0.0047 |

改写本身是正确的（「那它和边有什么区别？」→「LangGraph 里节点和边有什么区别？」），
但检索结果几乎没变。原因有二：

1. **基线本来就不弱**：`history + question` 拼接后，embedding 已经能理解指代，改写没有带来新信息；
2. **天花板效应**：4 条 multi_turn 里有 3 条改写前 `recall@4` 就已经是 1.0，没有提升空间。

这反而定位出了真正的瓶颈：multi_turn 的 `mrr` 只有 **0.5417** ——
正确答案通常排在**第 2~3 位**而不是第 1 位。要提高的是**排序**（rerank），不是查询改写。
详见 `eval/reports/baseline_20261020_2331_rewrite.md`。

---

## 4. 测试集怎么补

`eval/testcases.yaml` 现在有 **24 条真实可用** + **36 条 TODO 占位**。补一条只需三步：

**第一步：从语料里算 uuid。** uuid 规则是 `md5(page_content)` 转成标准 uuid 字符串，
即 `src/shared/state.py` 的 `_generate_uuid()`。别手写，跑一下：

```powershell
.\.venv\Scripts\python.exe -c "import json,sys; sys.path.insert(0,'src'); from shared.state import _generate_uuid; [print(_generate_uuid(d['page_content']), ' '.join(d['page_content'].split())[:70]) for d in json.load(open('src/sample_docs.json',encoding='utf-8'))]"
```

**第二步：照结构填一条。**

```yaml
  - id: et_09
    category: exact_term            # 五选一
    question: "……用户会怎么问……"
    history:                        # 只有 multi_turn 需要
      - "上一轮的问题"
    expected_doc_uuids:
      - <上一步算出来的 uuid>       # 只放「确实包含答案」的文档
    expected_answer_hint: "参考答案要点，用于 context_recall 打分"
    TODO: false                     # 改成 false 才会真正被评测
```

**第三步：核对。** 确认 `expected_doc_uuids` 里的每篇文档真的包含答案。
一条用例如果塞进「相关但不含答案」的文档，Recall / nDCG 会失真。

填写规则：

1. 问题写成真实用户口吻，**不要直接抄文档标题**。
2. `expected_doc_uuids` 只放包含答案的文档，通常 1~3 篇。
3. `no_answer` 类的 `expected_doc_uuids` 必须是 `[]`——它不参与检索指标，只统计拒答。
4. `permission` 类必须同时带 `user_id` 和 `allowed_user_ids`。
5. 多轮用例的 `history` 按时间正序。

> **当前 `permission` 类 12 条全是 TODO**：`src/sample_docs.json` 的 metadata 只有 `source`，
> 没有 owner 元数据，所以「命中该文档但应被过滤」这个断言今天无法成立。
> 要先给语料补 ACL 元数据再启用这一类，见第 7 节。

---

## 5. 指标定义

### 5.1 检索指标（`metrics.py`，纯函数）

约定：`expected_ids` 为空时所有检索指标返回 **`None`（未定义）**，而不是 0 分——
没有标注答案就没有意义，聚合时会被排除。检索结果里一个都没命中才返回 `0.0`。

| 指标 | 公式 | 含义 |
|---|---|---|
| **Recall@K** | `\|topK ∩ expected\| / \|expected\|` | top-K 覆盖了多少标准答案，不看顺序 |
| **MRR** | `1 / rank(第一条命中)` | 第一条正确结果排多靠前；top-1 得 1.0、top-4 得 0.25 |
| **nDCG@K** | `DCG@K / IDCG@K`，`DCG@K = Σ rel_i / log2(i+1)` | 同时考虑命中与位置，第 1 位权重 1.0、第 10 位约 0.29 |
| **Context Precision** | `AP = (1/命中数) · Σ_{命中的位置 i} (命中数_i / i)` | 检索结果里有用文档占多大比例、被无关文档稀释多少 |

跑评测时默认输出 `Recall@1 / @4 / @8 / @20`（`@4` 对应项目默认 `k`），
以及 `MRR`、`nDCG@10`、`ContextPrecision`。

### 5.2 生成侧指标（`judge.py`，LLM-as-judge）

三个指标都由 dashscope 模型按提示词打分，返回 `0.0~1.0` + 中文理由：

| 指标 | 评测什么 | 打分梯度 |
|---|---|---|
| `faithfulness` | 答案是否**只依据 context**，有没有编造 context 里没有的内容 | 1.0 全部有依据 / 0.4 一半无依据 / 0.0 基本编造 |
| `answer_relevancy` | 答案是否**切题**，有没有答非所问 | 1.0 直接完整 / 0.4 沾边没答到 / 0.0 完全跑题 |
| `context_recall` | 检索到的 context 是否**覆盖参考答案的关键信息** | 1.0 关键点全覆盖 / 0.4 只覆盖一小部分 / 0.0 基本无关 |

`context_recall` 需要 `expected_answer_hint` 当参考答案；用例没写 hint 时这一项跳过。

### 5.3 拒答率（`no_answer` 类别专用）

`no_answer` 用例没有参考答案，`answer_relevancy` / `context_recall` 对它没有意义，
所以这一类**不用 judge**，改用关键词启发式判断模型有没有拒答
（`没有找到` / `未提及` / `无法回答` / `not found` / `does not mention` …，见
`run_eval.py` 的 `REFUSAL_MARKERS`），报告里给出「拒答率」。

> 这是启发式，不是语义判断。模型用别的话术绕开关键词就会被判成「未拒答」。
> 想要更准就换成让 judge 判断「答案是否声称资料不足」。

### 5.4 为什么生成侧不跑完整 graph

完整主图是「路由分类 → 拆研究计划 → 扩写检索词 → `Send` 并行检索 → 合并去重 → 带引用作答」，
它**强依赖 Elasticsearch**，当前环境跑不通；而且一旦答案不好，无法区分是
「规划/路由写错了」还是「检索没召回」还是「作答环节编造」。

所以生成侧选了一条简单可靠的路径：**直接把 top-4 context + 问题交给 qwen 作答**，
再用 judge 打分。这样检索指标与生成指标各自可归因，整条链路也没有外部服务依赖。

---

## 6. 报告怎么看

每次运行产出两份同名文件（`--rewrite` 时文件名会带 `_rewrite` 后缀）：

- `baseline_<YYYYMMDD_HHMM>.md` —— 人读：总体指标表、分类指标表、改写对比（如启用）、
  失败用例明细、跳过与异常
- `baseline_<YYYYMMDD_HHMM>.json` —— 机读：`meta` / `overall` / `per_category` /
  `rewrite_comparison` / `cases` / `failures` / `error_summary`

**`n/a` 和 `0.0000` 是不同的东西**：

- `n/a` = 该指标未定义或未评测（例如 `no_answer` 用例的检索指标，因为没有期望文档）；
- `0.0000` = 评了，但确实是 0 分。

失败用例的判定判据（见 `collect_failures()`）：有期望文档但 top-8 未全中、
MRR=0（完全未命中）、judge 任一指标 < 0.5、`no_answer` 未拒答、或该用例记录了异常。

---

## 7. 已知限制

1. **内存向量库 ≠ Elasticsearch。**
   评测用 `InMemoryVectorStore` 把 32 篇文档全量放进内存做暴力相似度搜索；
   线上是 ES 的 HNSW 近似检索 + 分词 + 过滤。两者在**召回行为上会有差异**，
   尤其 ES 的 `search_kwargs` 过滤、分片、近似索引参数效应，评测里完全没有。
   embedding 模型与 `reduce_docs` 的 uuid 规则是**真实**的，所以「向量质量」层面有参考价值。
2. **judge 是 LLM 近似打分。**
   实测同一份测试集连跑两次，`faithfulness` 从 0.8000 变成 0.7842（单条用例 0.70→0.40），
   波动约 ±0.02 级别、单条用例可达 ±0.3。**不适合当精确阈值**，
   适合看趋势和抓明显问题（编造、跑题、context 没覆盖）。
   judge 失败（API 报错 / 超时 / JSON 解析失败）时该指标记为缺失并计入 `judge_error`，
   **不会让评测崩掉，也不会被伪造成 0 分或满分**。
3. **TODO 条目被跳过。**
   60 条里 36 条是占位。报告里 `实际评测` 只统计真实用例；
   分类表里 `permission` 类不会出现，因为它的 12 条全是 TODO。
4. **语料只有 32 篇：指标既波动大，又已经饱和。**
   - **波动大**：一条用例的期望文档通常 1~3 篇，命中与否对 `Recall@4` 的影响是几十个百分点。
     改动一条用例就可能让总分明显跳动，**不要用小数点后几位去做回归门禁**。
   - **已饱和（重要）**：`Recall@20` 恒为 1.0（32 篇里取 20 篇，几乎必然覆盖），
     说明检索指标在这份语料上已经触顶。实测连**纯代词追问**（例如「那它呢？」）
     在完全不改写的情况下 `recall@4` 就已经是 1.0。因此 `--rewrite` 这类**查询侧改动的收益
     在这份语料上无法被测量**——不是改写无效，而是这把尺子量不出来。
     要真正看出区分度，需要 chunk 级（约 150~300 个检索单元）的语料，让 top-k 存在真实取舍。
     同理，`Recall@8` 已到 0.9386、`Recall@20` 已到 1.0，**任何提升召回的手段都很难在这套测试集上体现**；
     目前唯一还有区分度的维度是**排序**（`MRR` 0.7544、`nDCG@10` 0.7654）。
5. **`no_answer` 的拒答判定是关键词启发式**，见 5.3。
6. **`permission` 类目前无法真正评测。**
   `src/sample_docs.json` 的 metadata 只有 `source`，没有 `owner_user_id` / `allowed_user_ids`，
   构造不出「命中该文档但应被过滤」的断言。要启用需要先给语料补 ACL 元数据，
   再在 `run_eval.py` 的检索后加一层按 `allowed_user_ids` 的过滤。
7. **多轮只测「检索能否找回」，不测真实多轮对话。**
   检索 query 是 `history + question`（或 `--rewrite` 后的自包含 query），
   模拟「历史一起送进检索」。生成侧仍然只看当前问题，没有真正走多轮上下文。
8. **`--limit N` 按测试集顺序取前 N 条真实用例**，不是随机抽样。
   想只测某一类，改测试集或用 `--testcases` 传一个子集文件。
9. **嵌入截断后，长文档只有开头部分参与向量检索（真实代价）。**
   `--max-embed-chars 8000` 让向量只由每篇的**前 8000 字符**算出。
   入库正文和 context 仍是全文（这点没有损失），但对大语料里那 150 篇超长文档来说，
   **正文后段的语义无法通过向量相似度命中**：关键词落在 8000 字符之后的问句会检不到该文档。
   本次新语料 `recall@20 = 0.9167`（而非旧语料的 1.0）里就含有这部分影响。
   要减轻这一点，正确做法是**先切块（chunk）再入库**，而不是把上限调大——
   调大只是把 400 错误往后推，且 embedding 成本与耗时都会上升。
10. **大语料 embedding 很慢（账号侧限流）。**
    229 篇 × 最多 8000 字符，实测整批约 **10 分钟**（12 个请求，单请求 5~12s，
    额外时间主要花在限流重试上）。`--skip-judge` 也省不掉这一步，因为建库就要算向量；
    想快只能减语料或先切块。**并发化风险大**（会加重 429），本仓库刻意没有做。

---

## 8. 模型选择与降级记录

本项目配置里的默认模型**当前不可用**，评测侧因此**显式指定**可用模型，不读
`src/retrieval_graph/configuration.py` 的默认值：

| 模型 | 实测结果 |
|---|---|
| `qwen-turbo` | **429 RateLimitError**（超出请求限额）→ 不可用 |
| `qwen-max` | **403 PermissionDenied**（免费额度耗尽）→ 不可用 |
| `qwen-plus` | 可用 → **作答 + judge 的默认首选** |
| `qwen-flash` | 可用 → 备选；也是**改写**的首选 |
| `qwen-long` | 可用（未用于默认路径） |
| `text-embedding-v4` | 可用，dim = 1024 |

默认降级链（可用 `--answer-model` / `--judge-model` / `--rewrite-model` 或环境变量
`EVAL_ANSWER_MODEL` / `EVAL_JUDGE_MODEL` / `EVAL_REWRITE_MODEL` 覆盖）：

- 作答：`qwen-plus` → `qwen-flash`
- judge：`qwen-plus` → `qwen-flash`
- 改写：`qwen-flash` → `qwen-plus`

每个模型最多尝试 2 次（1 次调用 + 1 次重试），重试之间做递增退避（1.5s、3s）。
某个模型不可用就粘性地切到下一个候选，报告 `meta` 里记录**实际使用**的模型名。

指定模型示例：

```powershell
.\.venv\Scripts\python.exe eval\run_eval.py --judge-model dashscope/qwen-plus --answer-model dashscope/qwen-plus
```

---

## 9. 其他降级与踩坑记录

| 情况 | 处理 |
|---|---|
| `pyyaml` 是否可用 | 已安装（6.0.3），因此测试集用 **YAML**，没有降级成 JSON |
| Elasticsearch 不可用 | 全程走 `InMemoryVectorStore`，不依赖 Docker / ES |
| Python 子进程无法写项目目录 | 用 `--output-dir "$env:TEMP\rag_eval_reports"` 绕开（见第 2 节） |
| `load_dotenv()` 读不到 Key | 用**绝对路径** `load_dotenv(<项目根>/.env)`，不依赖 cwd |
| 模型输出带 ```` ```json ```` 围栏 / 前后有废话 | `_extract_json()` 三级降级：整体解析 → 去围栏 → 正则抓第一个 `{...}` |
| 改写模型返回带引号 / 带前缀 | `_clean_rewrite()` 去前缀、去包裹引号、只取第一行 |
| 中文编码 | 所有文件 UTF-8；读写在 `open(..., encoding="utf-8")` 显式指定；控制台命令加 `-X utf8` |

---

## 10. 本次基线数字

完整报告见 `eval/reports/`：

- `baseline_20261020_2331.md` —— 主基线（judge 开启、改写关闭）
- `baseline_20261020_2331_rewrite.md` —— 改写前后对比（`--rewrite`、`--skip-judge`）

主基线（24 条真实用例，其中 19 条参与检索与生成指标）：

| 指标 | 数值 |
|---|---:|
| Recall@1 | 0.3684 |
| Recall@4 | 0.8070 |
| Recall@8 | 0.9386 |
| Recall@20 | 1.0000 |
| MRR | 0.7544 |
| nDCG@10 | 0.7654 |
| Context Precision | 0.6790 |
| faithfulness | 0.7842 |
| answer_relevancy | 0.9474 |
| context_recall | 0.7947 |
| no_answer 拒答率 | 1.0000（n=5） |

> 这些数字由 `run_eval.py` 直接生成，未经人工修改。
>
> **关于文件落盘位置**：本次生成基线的环境里，Python 子进程无权写项目目录，
> 所以 `.json` 落在了工作区内的可写目录 `eval_out/`（与 `eval/reports/` 里的 `.md` 同名同源）。
> 在没有该限制的环境重跑，`.md` 与 `.json` 会一起写到 `eval/reports/`。
> 归档在 `eval/reports/` 的两份 `.md` 就是 `run_eval.py` 的原样输出。
