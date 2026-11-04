"""检查 LangSmith 追踪配置是否就绪（不发出任何网络请求）。

用法:
    .venv\\Scripts\\python.exe check_langsmith_setup.py
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

REQUIRED = "LANGSMITH_API_KEY"
OPTIONAL = [
    "LANGSMITH_TRACING",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_WORKSPACE_ID",
]
LEGACY = ["LANGCHAIN_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT", "LANGCHAIN_ENDPOINT"]


def mask(value: str) -> str:
    if not value:
        return "(空)"
    if len(value) <= 10:
        return value[:3] + "****"
    return value[:8] + "****" + value[-4:]


def main() -> int:
    print("=" * 68)
    print("LangSmith 追踪配置检查")
    print("=" * 68)

    ok = True

    print("\n【必需】")
    value = os.environ.get(REQUIRED)
    if value:
        print(f"  ✓ {REQUIRED} = {mask(value)}")
    else:
        print(f"  ✗ {REQUIRED} 未设置 —— 这正是 Studio 提示 “API key is missing” 的原因")
        ok = False

    print("\n【可选】")
    for name in OPTIONAL:
        v = os.environ.get(name)
        print(f"  {name} = {mask(v) if v else '(未设置，将使用默认值)'}")

    print("\n【旧版变量名（不推荐，仅供参考）】")
    for name in LEGACY:
        v = os.environ.get(name)
        print(f"  {name} = {mask(v) if v else '(未设置)'}")

    print("\n【SDK 读取结果】")
    try:
        from langsmith import Client

        client = Client()  # 不发网络请求
        info = getattr(client, "info", None)
        print(f"  ✓ 客户端创建成功，api_url = {getattr(info, 'api_url', '(未知)')}")
        proj = os.environ.get("LANGSMITH_PROJECT") or "(默认 default)"
        print(f"  追踪会写入项目: {proj}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ 客户端创建失败: {type(exc).__name__}: {exc}")
        ok = False

    print("\n" + "=" * 68)
    if ok:
        print("配置看起来已就绪。请重启 langgraph dev 让 .env 生效。")
        return 0
    print("配置不完整，请按上面的提示补全 .env 后重启 langgraph dev。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
