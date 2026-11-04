"""阿里云百炼接入体检脚本。

按用户确认的方案验证三件事：
  1. 对话模型 qwen-flash（快模型，原文 claude-3-haiku 的替代）
  2. 对话模型 qwen-plus（强模型，原文 claude-3-5-sonnet 的替代）
  3. 向量模型 text-embedding-v4 + 1024 维（原文 text-embedding-3-small 的替代）

另外额外验证本 Agent 的关键能力：with_structured_output（结构化输出）。

用法：
    .venv\\Scripts\\python.exe test_bailian.py

前置条件：项目根目录存在 .env，且其中填好了 DASHSCOPE_API_KEY。
"""

# 模型说明：qwen-turbo 当前返回 429、qwen-max 返回 403（免费额度耗尽），故统一改用 qwen-flash / qwen-plus。

import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
API_KEY = os.environ.get("DASHSCOPE_API_KEY")

CHAT_FAST = "qwen-flash"
CHAT_STRONG = "qwen-plus"
EMBED_MODEL = "text-embedding-v4"
EMBED_DIM = 1024

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))
    flag = "PASS" if ok else "FAIL"
    print(f"[{flag}] {name}")
    print(f"       {detail}")


def main() -> int:
    print("=" * 68)
    print("阿里云百炼接入体检")
    print("=" * 68)
    print(f"base_url     : {BASE_URL}")
    print(f"api_key      : {'已读取 (len=%d)' % len(API_KEY) if API_KEY else '缺失'}")
    print(f"快模型       : {CHAT_FAST}")
    print(f"强模型       : {CHAT_STRONG}")
    print(f"向量模型     : {EMBED_MODEL} ({EMBED_DIM} 维)")
    print("-" * 68)

    if not API_KEY:
        record(
            "环境变量 DASHSCOPE_API_KEY",
            False,
            "未找到密钥。请复制 .env.example 为 .env 并填入 DASHSCOPE_API_KEY 后重跑。",
        )
        return 1

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    # ---------- 1) 快模型 ----------
    try:
        llm = ChatOpenAI(
            model=CHAT_FAST,
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,
        )
        reply = llm.invoke("用一句话说明什么是向量检索。")
        text = str(reply.content).strip()
        record(f"快模型 {CHAT_FAST} 对话", bool(text), text[:100] or "(空响应)")
    except Exception as exc:  # noqa: BLE001
        record(f"快模型 {CHAT_FAST} 对话", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    # ---------- 2) 强模型 ----------
    try:
        llm = ChatOpenAI(
            model=CHAT_STRONG,
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,
        )
        reply = llm.invoke("用一句话说明什么是 RAG。")
        text = str(reply.content).strip()
        record(f"强模型 {CHAT_STRONG} 对话", bool(text), text[:100] or "(空响应)")
    except Exception as exc:  # noqa: BLE001
        record(f"强模型 {CHAT_STRONG} 对话", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    # ---------- 3) 结构化输出（本 Agent 的核心依赖）----------
    try:
        from pydantic import BaseModel, Field

        class Router(BaseModel):
            """复刻项目里的 Router 结构。"""

            logic: str = Field(description="分类理由")
            type: str = Field(description="分类结果")

        llm = ChatOpenAI(
            model=CHAT_FAST,
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,
        )
        out = llm.with_structured_output(Router).invoke(
            "用户说：LangGraph 里怎么定义一个节点？"
        )
        ok = isinstance(out, Router)
        record(
            f"结构化输出 (with_structured_output / {CHAT_FAST})",
            ok,
            f"{type(out).__name__}: logic={getattr(out, 'logic', None)!r} type={getattr(out, 'type', None)!r}",
        )
    except Exception as exc:  # noqa: BLE001
        record(
            f"结构化输出 (with_structured_output / {CHAT_FAST})",
            False,
            f"{type(exc).__name__}: {exc}",
        )
        traceback.print_exc()

    # ---------- 4) 向量模型 ----------
    try:
        emb = OpenAIEmbeddings(
            model=EMBED_MODEL,
            api_key=API_KEY,
            base_url=BASE_URL,
            dimensions=EMBED_DIM,
            check_embedding_ctx_length=False,
        )
        vec = emb.embed_query("测试一下向量维度")
        got = len(vec)
        record(
            f"向量模型 {EMBED_MODEL}",
            got == EMBED_DIM,
            f"返回维度 = {got}（期望 {EMBED_DIM}）",
        )

        batch = emb.embed_documents(["第一段文本", "第二段文本"])
        record(
            "向量模型批量调用",
            len(batch) == 2 and all(len(v) == EMBED_DIM for v in batch),
            f"批量返回 {len(batch)} 条向量，各条维度 {[len(v) for v in batch]}",
        )
    except Exception as exc:  # noqa: BLE001
        record(f"向量模型 {EMBED_MODEL}", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    # ---------- 5) 项目自定义加载器的可用性 ----------
    try:
        from langchain_openai import OpenAIEmbeddings as _OA  # noqa: F401

        import inspect

        # 注意：这里必须用 inspect.signature(类)，不能用 类.__init__。
        # langchain_openai 0.1.x 的 OpenAIEmbeddings 是 pydantic v1 模型，
        # 它的 __init__ 被元类包装成 (*args, **data)，直接看 __init__ 什么都看不到。
        sig = inspect.signature(_OA)
        has_dim = "dimensions" in sig.parameters
        has_ctx = "check_embedding_ctx_length" in sig.parameters
        record(
            "本地 langchain-openai 版本兼容性",
            has_dim and has_ctx,
            f"dimensions={'有' if has_dim else '无'}, check_embedding_ctx_length={'有' if has_ctx else '无'}",
        )
    except Exception as exc:  # noqa: BLE001
        record("本地 langchain-openai 版本兼容性", False, f"{type(exc).__name__}: {exc}")

    # ---------- 汇总 ----------
    print("-" * 68)
    failed = [n for n, ok, _ in results if not ok]
    passed = [n for n, ok, _ in results if ok]
    print(f"通过 {len(passed)} 项，失败 {len(failed)} 项")
    if failed:
        print("失败项：")
        for n in failed:
            print(f"  - {n}")
        print()
        print("结论：百炼尚未完全接通，请先解决上面的失败项。")
        return 1
    print()
    print("结论：百炼已接通，可以继续跑 indexer / retrieval_graph。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
