"""从语料自动生成合成测试集（双轨方案的 A 轨）。

背景
----
语料从 32 篇扩到 200+ 篇之后，`eval/testcases.yaml` 里的 `expected_doc_uuids`
全部失效（它们指向旧语料的 md5）。人工重写一套要一天，会阻塞评测。

本脚本用 LLM 为**每条抽样语料**生成「用户会怎么问」的问题，期望文档就是那篇。
用途是**跑通流程 + 做 A/B/C 相对比较**，不是对外报绝对数字。

⚠️ 合成集的已知偏差（必须标注，不要拿去写简历）
------------------------------------------------
LLM 是「读过这篇文档」之后才写问题的，因此问题会与文档共享词汇 → 检索更易命中
→ **绝对 recall 被系统性高估**。
但三组配置（A/B/C）用同一套题，偏差相互抵消，**相对比较仍然有效**。

真正对外的数字请用人工集（见 `出题指南.md`）。

用法
----
1) 离线看抽样（不调 API，先确认抽样质量）：
   .\\.venv\\Scripts\\python.exe eval\\make_synthetic_cases.py --corpus src\\sample_docs_large.json --dry-run

2) 真正生成（需要可用额度）：
   .\\.venv\\Scripts\\python.exe eval\\make_synthetic_cases.py ^
       --corpus src\\sample_docs_large.json --n 60 --out "$env:TEMP\\cases_synthetic.yaml"

3) 指定模型 / 生成多轮用例比例：
   ... --model dashscope/qwen-plus --multi-turn-ratio 0.25

产物写到 `--out` 指定路径（**默认写 `$env:TEMP`，因为 Windows 沙箱下 Python
子进程无法在工作区新建文件**）。生成完再用编辑器把 YAML 搬进仓库。

额度不可用（如 400 Arrearage）时会**立即停下并打印清晰提示**，不会静默产出空测试集。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import uuid as uuidlib
from typing import Any, Optional

# ---- 与 shared.state._generate_uuid 保持一致的 uuid 规则 ---------------------

def doc_uuid(page_content: str) -> str:
    """page_content 的 md5 → uuid 字符串。"""
    return str(uuidlib.UUID(hashlib.md5(page_content.encode()).hexdigest()))


# ---- 语料加载 ---------------------------------------------------------------

def load_corpus(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        raise SystemExit(f"语料不存在：{path}")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise SystemExit(f"语料应为非空 JSON 数组：{path}")
    return data


def meta_of(doc: dict[str, Any], key: str, default: str = "") -> str:
    return str((doc.get("metadata") or {}).get(key, default) or default)


# ---- 抽样 -------------------------------------------------------------------

def sample_docs(
    docs: list[dict[str, Any]],
    n: int,
    seed: int,
    min_chars: int = 400,
) -> list[dict[str, Any]]:
    """确定性抽样。

    要求：
      - 只挑正文长度 >= min_chars 的文档（太短的内容出不了好问题）
      - 同一 section 内不重复抽太多（按 section 轮转，保证题目分散）
      - 固定随机种子，保证可复现
    """
    candidates = [d for d in docs if len(d.get("page_content") or "") >= min_chars]
    if not candidates:
        raise SystemExit(f"没有正文长度 >= {min_chars} 的文档，无法出题")

    # 按 section 分桶，轮转抽取，避免全部来自同一主题
    buckets: dict[str, list[dict[str, Any]]] = {}
    for doc in candidates:
        section = meta_of(doc, "section") or "unknown"
        buckets.setdefault(section, []).append(doc)

    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    picked: list[dict[str, Any]] = []
    keys = sorted(buckets)
    index = 0
    while len(picked) < n and any(buckets[k] for k in keys):
        key = keys[index % len(keys)]
        if buckets[key]:
            picked.append(buckets[key].pop())
        index += 1
    return picked


# ---- 模型调用 ---------------------------------------------------------------

class ModelUnavailable(RuntimeError):
    """额度/凭证不可用（如 400 Arrearage）—— 应立即停下，不要重试。"""


def build_llm(model: str, timeout: float = 60.0):
    """按 provider/model 构造 ChatOpenAI（只支持 dashscope 与本项目默认）。"""
    from langchain_openai import ChatOpenAI

    provider, _, name = model.partition("/")
    if provider != "dashscope":
        raise SystemExit(f"当前只支持 dashscope/*，收到：{model}")
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("环境变量 DASHSCOPE_API_KEY 缺失（应先加载 .env）")
    base_url = os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    return ChatOpenAI(
        model=name, api_key=api_key, base_url=base_url, temperature=0, timeout=timeout
    )


def is_arrearage(exc: Exception) -> bool:
    """识别账号欠费（与限流、额度耗尽区分开，因为处置完全不同）。"""
    text = str(exc)
    return "Arrearage" in text or "account is in good standing" in text


QUESTION_PROMPT = """你在为一个中文技术问答系统出测试题。

下面是一篇技术文档的内容。请写出**一个真实用户会怎么问**、并且这篇文档能回答的问题。

硬性要求：
1. 用中文，口语化，像工程师在提问，不要用「请解释」「简述」这类书面语开头
2. **绝对不要照抄文档标题**，也不要直接堆砌文档里的专有名词
3. 问题要指向文档里的**具体结论或做法**，而不是泛泛地问「X 是什么」
4. 长度 15~40 个汉字
5. 只输出问题本身，不要加编号、引号、解释

文档标题：{title}
文档内容（节选）：
{content}
"""

HISTORY_PROMPT = """你在为一个中文多轮对话的检索系统出测试题。

下面是一篇技术文档。请构造一组「两轮对话」：
- 第 1 轮：一个关于该文档主题的完整问题
- 第 2 轮：一个**带代词**的追问（必须含「它」「这个」「那两者」之类，单独看不懂在问什么，但结合第 1 轮就能明白）

要求：
1. 全中文，口语化
2. 第 2 轮必须是追问，且必须依赖第 1 轮才能理解
3. 只输出两行，第一行是问题1，第二行是问题2，不要编号和解释

文档标题：{title}
文档内容（节选）：
{content}
"""


def ask_llm(llm, prompt: str, retries: int = 1) -> str:
    """调模型；欠费直接抛 ModelUnavailable，其它错误重试一次。"""
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = llm.invoke(prompt)
            return str(response.content).strip()
        except Exception as exc:  # noqa: BLE001 - 需要区分欠费/其它
            if is_arrearage(exc):
                raise ModelUnavailable(
                    "账号欠费（400 Arrearage），所有模型调用都会失败。"
                    "请先到阿里云百炼控制台充值或关闭「仅使用免费额度」，再重跑本脚本。"
                ) from exc
            last = exc
            if attempt < retries:
                time.sleep(2.0)
    raise RuntimeError(f"模型调用失败：{type(last).__name__}: {str(last)[:200]}")


# ---- 主流程 -----------------------------------------------------------------

def make_cases(
    docs: list[dict[str, Any]],
    llm,
    multi_turn_ratio: float,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    total = len(docs)
    for position, doc in enumerate(docs, start=1):
        content = (doc.get("page_content") or "")[:2500]
        title = meta_of(doc, "title") or meta_of(doc, "source") or "(无标题)"
        uid = doc_uuid(doc.get("page_content") or "")
        want_multi = multi_turn_ratio > 0 and (position % max(int(1 / multi_turn_ratio), 2) == 0)

        try:
            if want_multi:
                raw = ask_llm(llm, HISTORY_PROMPT.format(title=title, content=content))
                lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                if len(lines) < 2:
                    raise RuntimeError(f"多轮用例格式异常：{raw[:120]!r}")
                first = lines[0].lstrip("0123456789.、) ").strip()
                second = lines[1].lstrip("0123456789.、) ").strip()
                cases.append({
                    "id": f"sy_mt_{position:03d}",
                    "category": "multi_turn",
                    "question": second,
                    "history": [first],
                    "expected_doc_uuids": [uid],
                    "expected_answer_hint": f"（合成）来自文档：{title}",
                    "TODO": False,
                })
            else:
                question = ask_llm(llm, QUESTION_PROMPT.format(title=title, content=content))
                question = question.strip().strip('"“”').lstrip("0123456789.、) ").strip()
                if len(question) < 6:
                    raise RuntimeError(f"问题过短：{question!r}")
                category = "exact_term" if any(
                    token in content for token in ("`", "def ", "class ")
                ) and position % 3 == 0 else "cross_section"
                cases.append({
                    "id": f"sy_{position:03d}",
                    "category": category,
                    "question": question,
                    "expected_doc_uuids": [uid],
                    "expected_answer_hint": f"（合成）来自文档：{title}",
                    "TODO": False,
                })
        except ModelUnavailable:
            raise  # 欠费立即上抛，别继续
        except Exception as exc:  # 单条失败不影响整体
            print(f"  [skip] doc#{position} 生成失败：{type(exc).__name__}: {str(exc)[:120]}")
        print(f"  [{position}/{total}] {cases[-1]['category'] if cases else '-'}  {cases[-1]['question'][:40] if cases else ''}")
    return cases


def dump_yaml(payload: dict[str, Any], path: str) -> None:
    import yaml

    try:
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
    except PermissionError as exc:
        raise SystemExit(
            f"无法写入 {path}（Windows 沙箱下 Python 子进程不能写工作区）。\n"
            f"请把 --out 指到 $env:TEMP 下，例如 "
            f'--out "$env:TEMP\\cases_synthetic.yaml"\n原始错误：{exc}'
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从语料生成合成测试集（双轨方案的 A 轨，A/B 相对比较用）",
    )
    parser.add_argument("--corpus", default="src/sample_docs_large.json",
                        help="语料 JSON（默认 src/sample_docs_large.json）")
    parser.add_argument("--n", type=int, default=60, help="生成多少条（默认 60）")
    parser.add_argument("--seed", type=int, default=20261020, help="抽样随机种子")
    parser.add_argument("--min-chars", type=int, default=400, help="只挑正文不少于该长度的文档")
    parser.add_argument("--multi-turn-ratio", type=float, default=0.2,
                        help="多轮用例占比，0 表示不生成（默认 0.2）")
    parser.add_argument("--model", default="dashscope/qwen-plus", help="出题模型")
    parser.add_argument("--out", default=os.path.join(
        os.environ.get("TEMP", "."), "cases_synthetic.yaml"), help="输出 YAML 路径")
    parser.add_argument("--dry-run", action="store_true", help="只抽样并打印，不调模型")
    args = parser.parse_args()

    docs = load_corpus(args.corpus)
    picked = sample_docs(docs, args.n, args.seed, args.min_chars)

    print(f"语料：{args.corpus}（{len(docs)} 篇）")
    print(f"抽样：{len(picked)} 篇（种子 {args.seed}，正文 >= {args.min_chars} 字）")
    buckets: dict[str, int] = {}
    for doc in picked:
        key = meta_of(doc, "section") or "unknown"
        buckets[key] = buckets.get(key, 0) + 1
    print("抽样主题分布：" + ", ".join(f"{k}={v}" for k, v in sorted(buckets.items())))

    if args.dry_run:
        print("\n[dry-run] 前 5 篇抽样（不调模型）：")
        for doc in picked[:5]:
            print(f"  uuid={doc_uuid(doc['page_content'])}")
            print(f"    标题：{meta_of(doc, 'title') or '(无)'}")
            print(f"    来源：{meta_of(doc, 'source') or '(无)'}")
            print(f"    正文：{' '.join(doc['page_content'].split())[:100]}…")
        print("\n抽样正常。去掉 --dry-run 即真正生成（会消耗对话额度）。")
        return 0

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.getcwd(), ".env"), encoding="utf-8")
    except Exception:
        pass

    llm = build_llm(args.model)
    print(f"\n开始生成（模型 {args.model}）……")
    try:
        cases = make_cases(picked, llm, args.multi_turn_ratio)
    except ModelUnavailable as exc:
        print(f"\n⛔ {exc}")
        print("已生成的部分不会落盘（避免产生残缺测试集）。")
        return 2

    if not cases:
        print("\n没有生成任何用例，不落盘。")
        return 1

    payload = {
        "meta": {
            "name": "synthetic",
            "corpus": args.corpus,
            "corpus_size": len(docs),
            "model": args.model,
            "seed": args.seed,
            "n_requested": args.n,
            "n_generated": len(cases),
            "warning": "合成集：问题由读过源文档的 LLM 生成，绝对 recall 偏高，仅用于 A/B 相对比较",
        },
        "cases": cases,
    }
    dump_yaml(payload, args.out)

    by_category: dict[str, int] = {}
    for case in cases:
        by_category[case["category"]] = by_category.get(case["category"], 0) + 1
    print(f"\n生成 {len(cases)} 条 → {args.out}")
    print("分类：" + ", ".join(f"{k}={v}" for k, v in sorted(by_category.items())))
    print("\n⚠️ 合成集绝对 recall 偏高，只用于 A/B 相对比较；对外数字请用人工集（出题指南.md）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
