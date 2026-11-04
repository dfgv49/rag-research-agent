@echo off
chcp 65001 >nul
setlocal

REM ============================================================
REM  start-studio.bat  -  一键启动 LangGraph Studio
REM  双击运行即可。按 Ctrl+C 停止服务。
REM ============================================================

REM 自动切到脚本所在目录（也就是项目根目录）
cd /d "%~dp0"

echo ============================================================
echo   启动 LangGraph Studio
echo ============================================================
echo.

REM --- 前置检查 1：虚拟环境 ---
if not exist ".venv\Scripts\langgraph.exe" (
    echo [错误] 找不到 .venv\Scripts\langgraph.exe
    echo.
    echo 请先确认：
    echo   1. 你是在项目根目录运行（这个目录里应该有 langgraph.json、pyproject.toml）
    echo   2. 依赖已经装好。如果没有，先执行：
    echo        uv sync
    echo        uv add --dev "langgraph-cli[inmem]"
    echo.
    pause
    exit /b 1
)

REM --- 前置检查 2：.env ---
if not exist ".env" (
    echo [错误] 找不到 .env 文件
    echo.
    echo 请先复制模板并填入你的百炼密钥：
    echo        Copy-Item .env.example .env
    echo   然后编辑 .env，把 DASHSCOPE_API_KEY 换成真实密钥。
    echo.
    pause
    exit /b 1
)

echo [1/2] 环境检查通过
echo       Python  : .venv\Scripts\python.exe
echo       配置文件: .env
echo.
echo [2/2] 正在启动服务，稍等几秒...
echo.
echo       启动成功后，浏览器打开下面这个地址即可进入 Studio：
echo.
echo         https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
echo.
echo       本机接口文档: http://127.0.0.1:2024/docs
echo.
echo       === 保持这个窗口开着，按 Ctrl+C 可以停止服务 ===
echo.

REM --allow-blocking 是必需的：langchain 内部有同步调用，
REM 在异步服务里会被判定为阻塞操作并导致运行失败。
.venv\Scripts\langgraph.exe dev --port 2024 --allow-blocking

echo.
echo 服务已停止。
pause
