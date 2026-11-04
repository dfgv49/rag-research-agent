"""直接测试当前提示词文件下的路由分类效果（不经服务器）。

用法:
    .venv\\Scripts\\python.exe test_router_prompt.py
"""

# 模型说明：qwen-turbo 当前返回 429、qwen-max 返回 403（免费额度耗尽），故统一改用 qwen-flash / qwen-plus。

import sys
import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from shared.utils import load_chat_model  # noqa: E402
from retrieval_graph.prompts import ROUTER_SYSTEM_PROMPT  # noqa: E402
from retrieval_graph.state import Router  # noqa: E402

QUESTIONS = [
    ("LangGraph 里怎么定义节点？", "langchain"),
    ("LangGraph 里 Send 是干什么用的？", "langchain"),
    ("langchain 的 AgentExecutor 报错了，怎么排查？", "langchain"),
    ("我的代码报错了", "more-info"),
    ("今天午饭吃什么好", "general"),
]


def main() -> int:
    print("=" * 70)
    print("路由提示词分类测试（直连模型，不经服务器）")
    print("=" * 70)
    print("当前提示词里关于 langchain 的规则片段：")
    for line in ROUTER_SYSTEM_PROMPT.splitlines():
        if "QUESTION ABOUT" in line or "never `more-info`" in line:
            print("   ", line.strip())
    print("-" * 70)

    model = load_chat_model("dashscope/qwen-flash").with_structured_output(Router)

    bad = 0
    for q, expect in QUESTIONS:
        out = model.invoke(
            [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ]
        )
        got = out["type"]
        flag = "OK " if got == expect else "BAD"
        if got != expect:
            bad += 1
        print(f"[{flag}] 期望={expect:<10} 实得={got:<10} | {q}")
        print(f"       理由: {out['logic'][:70]}")

    print("-" * 70)
    print(f"不符预期: {bad} / {len(QUESTIONS)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
