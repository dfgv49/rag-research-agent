"""自实现的 LLM-as-judge：三个生成侧指标，不依赖 ragas。

提供三个打分函数，各返回 0.0~1.0 的分数 + 理由（JudgeResult）：
  - faithfulness      : 答案是否只依据给定 context，有没有编造 context 里没有的内容
  - answer_relevancy  : 答案是否切题
  - context_recall    : 检索到的 context 是否覆盖了参考答案的关键信息

--------------------------------------------------------------------------
关键设计
--------------------------------------------------------------------------
1. **走 OpenAI 兼容接口**：与 src/shared/utils.py 的 dashscope 分支完全一致
   （ChatOpenAI + base_url + api_key + temperature=0），只是这里额外加了超时。
2. **结构化输出自己解析**：不依赖 with_structured_output（它要靠工具调用/JSON mode，
   不同模型支持不一），而是提示词里强制模型输出 {"score": ..., "reason": ...}，
   再用 pydantic 校验。这样任何模型都能用，解析失败也能精确定位原因。
3. **失败绝不炸穿评测**：API 报错 / 超时 / JSON 解析失败 一律返回 score=None 的
   JudgeResult，并把原因记在 judge_error 里。调用方统计「缺失」而不是崩掉。
4. **只用实测可用的模型**：本项目配置里的默认模型（query_model=qwen-turbo、
   response_model=qwen-max）在当前账号下分别返回 429 限流与 403 免费额度耗尽，**都跑不通**，
   因此评测侧**不使用 src/ 里的默认模型**，而是显式指定实测可用的 qwen-plus / qwen-flash。
   候选模型按顺序降级，某个模型不可用就粘性地切到下一个，并把实际用到的模型名记进报告。
   可用环境变量 EVAL_JUDGE_MODEL / EVAL_ANSWER_MODEL 覆盖。
5. **超时 60s + 最多 1 次重试 + 429 退避**：由本模块控制重试（ChatOpenAI 的 max_retries 设为 0），
   重试之间 sleep 递增退避，这样每次失败都能被准确记录，而不是被底层静默重试掉。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from pydantic import BaseModel, Field, ValidationError

__all__ = [
    "Judge",
    "JudgeResult",
    "JudgeVerdict",
    "LLMChannel",
    "ModelUnavailable",
    "DEFAULT_ANSWER_MODELS",
    "DEFAULT_JUDGE_MODELS",
    "DEFAULT_REWRITE_MODELS",
    "JUDGE_TIMEOUT_SECONDS",
]

#: dashscope 兼容 OpenAI 的默认 base_url
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

#: 单次调用超时（秒）
JUDGE_TIMEOUT_SECONDS = 60.0

#: 单个模型最多尝试次数（= 1 次调用 + 1 次重试）
MAX_ATTEMPTS_PER_MODEL = 2

#: 失败重试之间的基础退避秒数（第 n 次重试 sleep n * BACKOFF）
RETRY_BACKOFF_SECONDS = 1.5

# ---- 模型选择 --------------------------------------------------------------
# 当前账号实测的可用性矩阵：
#   qwen-turbo -> 429 限流（不可用）
#   qwen-max   -> 403 免费额度耗尽（不可用）
#   qwen-plus  -> 可用（本基线的 judge 与作答实际使用它）
#   qwen-flash -> 可用（便宜、快，适合轻量判断）
#   qwen-long  -> 可用
# 因此评测侧**不使用 src/retrieval_graph/configuration.py 的默认模型**
# （query_model=qwen-turbo、response_model=qwen-max，两个都跑不通），
# 而是在这里显式指定可用模型，并允许用环境变量覆盖。

#: 作答（生成侧）候选模型
DEFAULT_ANSWER_MODELS = ("dashscope/qwen-plus", "dashscope/qwen-flash")

#: judge（打分）候选模型
DEFAULT_JUDGE_MODELS = ("dashscope/qwen-plus", "dashscope/qwen-flash")

#: 查询改写候选模型（轻量任务，优先用 flash）
DEFAULT_REWRITE_MODELS = ("dashscope/qwen-flash", "dashscope/qwen-plus")


class ModelUnavailable(RuntimeError):
    """所有候选模型都不可用时抛出。"""


class JudgeVerdict(BaseModel):
    """judge 期望模型返回的结构化结果。"""

    score: float = Field(..., ge=0.0, le=1.0, description="0.0~1.0 的分数")
    reason: str = Field(default="", description="简短理由")


@dataclass
class JudgeResult:
    """单个指标的打分结果。

    Attributes:
        metric: 指标名（faithfulness / answer_relevancy / context_recall）。
        score: 0.0~1.0；None 表示本次打分失败（见 judge_error）。
        reason: 模型给出的理由；失败时为空串。
        judge_error: 失败原因；成功时为 None。
        model: 实际使用的模型名；失败时为 None。
    """

    metric: str
    score: Optional[float] = None
    reason: str = ""
    judge_error: Optional[str] = None
    model: Optional[str] = None

    @property
    def ok(self) -> bool:
        """是否拿到了有效分数。"""
        return self.score is not None

    def as_dict(self) -> dict:
        """转成可直接 json.dump 的字典。"""
        return {
            "metric": self.metric,
            "score": None if self.score is None else round(self.score, 4),
            "reason": self.reason,
            "judge_error": self.judge_error,
            "model": self.model,
        }


def _candidate_models(
    explicit: Optional[Sequence[str]] = None,
    env_var: str = "EVAL_JUDGE_MODEL",
    defaults: Sequence[str] = DEFAULT_JUDGE_MODELS,
) -> list[str]:
    """确定候选模型列表（显式传入 > 环境变量 > 默认）。

    Args:
        explicit: 显式指定的模型列表，优先级最高。
        env_var: 环境变量名，用于覆盖默认值。
        defaults: 默认候选模型（按降级顺序）。
    """
    if explicit:
        return list(explicit)
    env = os.getenv(env_var, "").strip()
    if env:
        return [env]
    return list(defaults)


class LLMChannel:
    """在若干候选模型之间做最简降级的最小封装。

    只在真正调用时才初始化客户端（懒加载），因此 import 本模块不需要 API Key。
    一旦某个模型成功过就固定用它；它之后再失败才尝试下一个。
    """

    def __init__(
        self,
        model_names: Optional[Sequence[str]] = None,
        timeout: float = JUDGE_TIMEOUT_SECONDS,
        env_var: str = "EVAL_JUDGE_MODEL",
        defaults: Sequence[str] = DEFAULT_JUDGE_MODELS,
        retry_backoff: float = RETRY_BACKOFF_SECONDS,
    ) -> None:
        self._candidates = _candidate_models(model_names, env_var=env_var, defaults=defaults)
        self._timeout = timeout
        self._retry_backoff = retry_backoff
        self._index = 0
        self._clients: dict[str, object] = {}
        self.used_model: Optional[str] = None
        self.last_error: Optional[str] = None

    @property
    def candidates(self) -> list[str]:
        """候选模型名列表（按降级顺序）。"""
        return list(self._candidates)

    def _client(self, model_name: str):
        """构造 / 复用某个模型的客户端。"""
        if model_name in self._clients:
            return self._clients[model_name]

        from langchain_openai import ChatOpenAI

        provider, _, bare_model = model_name.partition("/")
        if provider != "dashscope":
            raise ValueError(f"本评测只支持 dashscope 路由，收到: {model_name!r}")

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ModelUnavailable(
                "环境变量 DASHSCOPE_API_KEY 未设置，无法调用 judge 模型。"
                "请确认 .env 存在且已被 load_dotenv 载入。"
            )

        client = ChatOpenAI(
            model=bare_model,
            api_key=api_key,
            base_url=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
            temperature=0,
            timeout=self._timeout,
            # 重试由本类自己控制，关掉底层静默重试，保证失败原因可被记录。
            max_retries=0,
        )
        self._clients[model_name] = client
        return client

    def invoke(self, prompt: str) -> str:
        """调用模型返回文本。所有候选都失败时抛出 ModelUnavailable。

        Args:
            prompt: 完整提示词。

        Returns:
            str: 模型返回的文本内容。

        Raises:
            ModelUnavailable: 所有候选模型均不可用，或未配置 API Key。
        """
        errors: list[str] = []

        while self._index < len(self._candidates):
            model_name = self._candidates[self._index]
            try:
                client = self._client(model_name)
            except ModelUnavailable:
                # 没有 Key 这种全局性问题，直接上抛，不再试别的模型。
                raise
            except Exception as exc:  # 构造失败（比如路由非法）
                errors.append(f"{model_name}: {type(exc).__name__}: {exc}")
                self._index += 1
                continue

            for attempt in range(1, MAX_ATTEMPTS_PER_MODEL + 1):
                try:
                    message = client.invoke(prompt)  # type: ignore[attr-defined]
                    text = message.content if hasattr(message, "content") else str(message)
                    self.used_model = model_name
                    self.last_error = None
                    return text if isinstance(text, str) else str(text)
                except Exception as exc:
                    detail = f"{type(exc).__name__}: {str(exc)[:300]}"
                    errors.append(f"{model_name} (attempt {attempt}/{MAX_ATTEMPTS_PER_MODEL}): {detail}")
                    self.last_error = detail
                    if attempt >= MAX_ATTEMPTS_PER_MODEL:
                        break
                    # 429 之类的限流往往是瞬时的：退避后再试一次，仍失败才换模型。
                    time.sleep(self._retry_backoff * attempt)
            self._index += 1

        raise ModelUnavailable("所有候选模型均不可用： " + " | ".join(errors[-4:]))


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)
_OBJECT_RE = re.compile(r"\{.*\}", re.S)


def _extract_json(text: str) -> dict:
    """从模型输出里抠出第一个 JSON 对象。

    依次尝试：整体解析 → 去掉 ``` 代码围栏后解析 → 正则抓第一个 {...} 解析。

    Args:
        text: 模型原始输出。

    Returns:
        dict: 解析出的字典。

    Raises:
        ValueError: 三种方式都解析不出 JSON 对象。
    """
    if not text or not text.strip():
        raise ValueError("模型返回空内容")

    candidates: list[str] = [text.strip()]

    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    obj = _OBJECT_RE.search(text)
    if obj:
        candidates.append(obj.group(0))

    last_error: Optional[Exception] = None
    for chunk in candidates:
        try:
            parsed = json.loads(chunk)
        except Exception as exc:
            last_error = exc
            continue
        if isinstance(parsed, dict):
            return parsed
        last_error = ValueError(f"JSON 顶层不是对象，而是 {type(parsed).__name__}")

    raise ValueError(f"无法从模型输出中解析 JSON: {last_error}; 原文前 200 字: {text[:200]!r}")


def _clip(text: str, limit: int) -> str:
    """截断过长文本，避免提示词无界膨胀。"""
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[已截断，原文共 {len(text)} 字]"


def _format_contexts(contexts: Iterable[str], per_doc: int = 2000, total: int = 24000) -> str:
    """把 context 列表拼成带编号的块，超长则截断。"""
    parts: list[str] = []
    used = 0
    for idx, ctx in enumerate(contexts, start=1):
        block = f"[{idx}] {_clip(ctx, per_doc)}"
        if used + len(block) > total:
            parts.append(f"...[其余 {max(0, len(list(parts)))} 段 context 因长度限制未展示]")
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts) if parts else "（无）"


_JSON_INSTRUCTION = (
    "你必须只输出一个 JSON 对象，不要输出任何解释性文字、不要用 Markdown 代码块包裹。\n"
    '格式严格为：{"score": <0.0 到 1.0 之间的浮点数>, "reason": "<不超过 80 字的中文理由>"}\n'
)


class Judge:
    """基于 LLM 的打分器。构造时不联网，首次打分时才按需初始化客户端。"""

    def __init__(
        self,
        model_names: Optional[Sequence[str]] = None,
        timeout: float = JUDGE_TIMEOUT_SECONDS,
        channel: Optional[LLMChannel] = None,
    ) -> None:
        self.channel = channel or LLMChannel(model_names=model_names, timeout=timeout)

    def _score(self, metric: str, prompt: str) -> JudgeResult:
        """通用打分流程：调模型 → 抽 JSON → 校验 → 夹到 [0,1]。失败返回 score=None。"""
        try:
            raw = self.channel.invoke(prompt)
        except ModelUnavailable as exc:
            return JudgeResult(metric=metric, judge_error=f"模型不可用: {exc}")
        except Exception as exc:  # 兜底，绝不外抛
            return JudgeResult(metric=metric, judge_error=f"调用异常 {type(exc).__name__}: {str(exc)[:300]}")

        try:
            payload = _extract_json(raw)
            verdict = JudgeVerdict(**payload)
        except (ValueError, ValidationError, TypeError) as exc:
            return JudgeResult(
                metric=metric,
                judge_error=f"结构化输出解析失败: {type(exc).__name__}: {str(exc)[:300]}",
                model=self.channel.used_model,
            )

        score = min(1.0, max(0.0, float(verdict.score)))
        return JudgeResult(
            metric=metric,
            score=score,
            reason=(verdict.reason or "").strip(),
            model=self.channel.used_model,
        )

    def faithfulness(
        self,
        question: str,
        answer: str,
        contexts: Sequence[str],
    ) -> JudgeResult:
        """答案是否只依据给定 context（有没有编造 context 里没有的内容）。

        Args:
            question: 用户问题。
            answer: 系统生成的答案。
            contexts: 检索到的上下文原文列表。

        Returns:
            JudgeResult: 1.0 表示每句话都能在 context 里找到依据；0.0 表示大量编造。
            失败时 score 为 None 且 judge_error 有值。
        """
        prompt = (
            "你是检索增强问答系统的评测员。请判断「答案」里的每一处事实陈述，"
            "是否都能在给定的「参考资料」中找到依据。\n\n"
            "打分标准：\n"
            "  1.0 = 全部有依据，没有任何编造；\n"
            "  0.7 = 主体有依据，只有少量无依据的补充；\n"
            "  0.4 = 一半左右无依据；\n"
            "  0.0 = 答案内容基本是编造的，或与参考资料矛盾。\n"
            "注意：参考资料之外但属于常识的表述不算编造，但实质性的技术细节必须有依据。\n\n"
            + _JSON_INSTRUCTION
            + "\n【用户问题】\n"
            + _clip(question, 2000)
            + "\n\n【参考资料】\n"
            + _format_contexts(contexts)
            + "\n\n【答案】\n"
            + _clip(answer, 4000)
        )
        return self._score("faithfulness", prompt)

    def answer_relevancy(
        self,
        question: str,
        answer: str,
    ) -> JudgeResult:
        """答案是否切题（有没有答非所问 / 跑题 / 空泛）。

        Args:
            question: 用户问题。
            answer: 系统生成的答案。

        Returns:
            JudgeResult: 1.0 表示直接、完整地回应了问题；0.0 表示完全答非所问。
        """
        prompt = (
            "你是检索增强问答系统的评测员。请判断「答案」有没有正面回答「用户问题」。\n\n"
            "打分标准：\n"
            "  1.0 = 直接、完整地回答了问题；\n"
            "  0.7 = 基本回答，但漏了部分要点或过于笼统；\n"
            "  0.4 = 沾边但没真正回答核心问题；\n"
            "  0.0 = 完全答非所问，或只是拒绝/空话。\n"
            "注意：只评价是否切题，不要因为答案简短就扣分。\n\n"
            + _JSON_INSTRUCTION
            + "\n【用户问题】\n"
            + _clip(question, 2000)
            + "\n\n【答案】\n"
            + _clip(answer, 4000)
        )
        return self._score("answer_relevancy", prompt)

    def context_recall(
        self,
        question: str,
        contexts: Sequence[str],
        reference_answer: str,
    ) -> JudgeResult:
        """检索到的 context 是否覆盖了参考答案的关键信息。

        Args:
            question: 用户问题。
            contexts: 检索到的上下文原文列表。
            reference_answer: 参考答案（本评测里用 testcase 的 expected_answer_hint）。

        Returns:
            JudgeResult: 1.0 表示参考答案里的关键点都能在 context 中找到；
            0.0 表示 context 完全没覆盖。
        """
        prompt = (
            "你是检索增强问答系统的评测员。请判断「参考资料」有没有覆盖「参考答案」里的关键信息。\n\n"
            "打分标准：\n"
            "  1.0 = 参考答案的关键点都能在参考资料里找到；\n"
            "  0.7 = 覆盖大部分关键点，缺 1 个次要点；\n"
            "  0.4 = 只覆盖了一小部分；\n"
            "  0.0 = 参考资料与参考答案基本无关。\n"
            "注意：只评价覆盖程度，不要评价参考答案本身写得好不好。\n\n"
            + _JSON_INSTRUCTION
            + "\n【用户问题】\n"
            + _clip(question, 2000)
            + "\n\n【参考答案】\n"
            + _clip(reference_answer, 2000)
            + "\n\n【参考资料】\n"
            + _format_contexts(contexts)
        )
        return self._score("context_recall", prompt)
