"""验证 prompts.py 里的两个细节：
  1. [${number}] 的转义 —— format 之后模型实际看到什么？
  2. {logic} 占位符被真实填入的效果

用法:
    .venv\\Scripts\\python.exe verify_prompts.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from retrieval_graph.prompts import MORE_INFO_SYSTEM_PROMPT, RESPONSE_SYSTEM_PROMPT


def main() -> None:
    print("=" * 70)
    print("prompts.py 细节验证")
    print("=" * 70)

    print()
    print("【1】引用标记的转义")
    print("-" * 70)
    marker = "[${{number}}]"
    print(f"  源码里写的是 : {marker}")
    filled = RESPONSE_SYSTEM_PROMPT.format(context="（这里是检索到的文档）")
    idx = filled.find("notation")
    print(f"  format 之后   : ...{filled[idx - 30:idx + 25]}...")
    print("  说明：双大括号 {{ }} 是 f-string / format 的转义写法，")
    print("        运行时会被还原成单个大括号，模型最终看到 [${number}]。")

    print()
    print("【2】{logic} 占位符填充效果")
    print("-" * 70)
    merged = MORE_INFO_SYSTEM_PROMPT.format(logic="用户报告出错但没有提供错误信息")
    start = merged.find("<logic>")
    print(merged[start:start + 110])

    print()
    print("【3】{context} 占位符填充效果（截取片段）")
    print("-" * 70)
    filled2 = RESPONSE_SYSTEM_PROMPT.format(context="<documents>\n<document>示例文档</document>\n</documents>")
    s = filled2.find("<context>")
    print(filled2[s:s + 120])


if __name__ == "__main__":
    main()
