"""
FailureReflection 模块 - 失败后进行反思
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("control.reflection")


@dataclass
class ReflectionResult:
    """反思结果"""
    reflection_text: str
    root_cause: str | None = None
    suggested_correction: str | None = None
    should_retry: bool = True
    confidence: float = 0.5  # 0.0 - 1.0


@dataclass
class FailureRecord:
    """失败记录"""
    iteration: int
    tool_name: str | None
    error_message: str
    error_type: str
    timestamp: float
    context: str = ""


class FailureReflection:
    """失败反思模块

    专注于反思分析，帮助模型从失败中学习和恢复。
    """

    def __init__(self, config: "ReflectionConfig", llm_fn: Any | None = None):
        """初始化 FailureReflection。

        Args:
            config: ReflectionConfig 配置
            llm_fn: 异步 LLM 调用函数，签名为:
                    async def llm_fn(prompt: str, max_tokens: int, temperature: float) -> str
        """
        self.config = config
        self.llm_fn = llm_fn
        self.failure_history: list[FailureRecord] = []
        self.consecutive_failures = 0
        self.last_reflection_time = 0.0
        self._last_reflected_failure_count = 0

    def record_failure(
        self,
        iteration: int,
        tool_name: str | None,
        error_message: str,
        error_type: str = "unknown",
        context: str = "",
    ) -> None:
        """记录一次失败"""
        if not self.config.enabled:
            return

        record = FailureRecord(
            iteration=iteration,
            tool_name=tool_name,
            error_message=error_message,
            error_type=error_type,
            timestamp=time.time(),
            context=context,
        )
        self.failure_history.append(record)
        self.consecutive_failures += 1
        logger.debug(f"Recorded failure: {error_type} at iteration {iteration}")

    def record_success(self) -> None:
        """记录一次成功，重置连续失败计数"""
        self.consecutive_failures = 0

    def should_reflect(self) -> bool:
        """判断是否应该进行反思"""
        if not self.config.enabled:
            return False

        has_new_failure = len(self.failure_history) > self._last_reflected_failure_count
        if not has_new_failure:
            return False

        if self.config.trigger == "on_failure":
            return True
        elif self.config.trigger == "on_consecutive_failures":
            return self.consecutive_failures >= self.config.consecutive_failure_threshold
        elif self.config.trigger == "manual":
            return False  # 手动触发

        return False

    def _get_last_iteration(self) -> int:
        """获取最后一次失败的迭代号"""
        if not self.failure_history:
            return 0
        return self.failure_history[-1].iteration

    async def reflect(self, current_plan: str = "") -> ReflectionResult:
        """执行反思"""
        import time

        start_time = time.time()

        if self.llm_fn and self.config.enabled:
            result = await self._llm_reflect(current_plan)
        else:
            result = self._rule_based_reflect()

        result.reflection_text = result.reflection_text[:self.config.max_reflection_length]
        self.last_reflection_time = time.time() - start_time
        self._last_reflected_failure_count = len(self.failure_history)
        logger.info(f"Reflection completed in {self.last_reflection_time:.2f}s: {result.root_cause}")

        return result

    def _rule_based_reflect(self) -> ReflectionResult:
        """基于规则的简单反思"""
        if not self.failure_history:
            return ReflectionResult(reflection_text="No failures recorded.")

        last_failure = self.failure_history[-1]

        # 简单的根因分析
        error_type = last_failure.error_type.lower()
        if "timeout" in error_type:
            root_cause = "Tool execution timed out"
            correction = "Consider increasing timeout or simplifying the operation"
        elif "rate_limit" in error_type or "429" in error_type:
            root_cause = "API rate limit hit"
            correction = "Wait before retrying or reduce request frequency"
        elif "invalid" in error_type or "param" in error_type:
            root_cause = "Invalid parameters passed to tool"
            correction = "Review tool parameter requirements"
        elif "auth" in error_type or "permission" in error_type:
            root_cause = "Authentication or permission error"
            correction = "Check credentials and permissions"
        elif "not found" in error_type or "404" in error_type:
            root_cause = "Resource not found"
            correction = "Verify resource exists before attempting operation"
        else:
            root_cause = f"Unknown error: {last_failure.error_type}"
            correction = "Review error details and consider alternative approach"

        return ReflectionResult(
            reflection_text=f"Last failure: {last_failure.error_message}. Root cause: {root_cause}",
            root_cause=root_cause,
            suggested_correction=correction,
            should_retry=True,
            confidence=0.6,
        )

    async def _llm_reflect(self, current_plan: str) -> ReflectionResult:
        """使用 LLM 进行反思，引导模型从已知失败类别中诊断"""
        failure_summary = self._summarize_failures()

        prompt = f"""Analyze the following tool-call failures and classify the root cause.

FAILURE LOG:
{failure_summary}

CURRENT PLAN:
{current_plan or "(No plan available)"}

KNOWN FAILURE CATEGORIES (pick the best fit):
- A (Goal Misunderstanding): Misinterpreted task requirements, missed constraints, or fabricated assumptions.
- B (Action Instantiation): Wrong tool, missing/invalid parameters, or incorrect argument values.
- C (State Management): Lost track of intermediate results, overwrote working state, or failed to propagate facts.
- D (Verification Gap): Did not check whether the output matches task requirements before finishing.
- E (Strategy Selection): Chose an inefficient or incorrect approach, wrong tool sequence, or unnecessary steps.

Respond EXACTLY in this format (one line each):
Failure Category: [A/B/C/D/E]
Root Cause: [one sentence explaining why]
Correction: [specific actionable step to fix it]
Retry: [yes/no]"""

        try:
            response = await self.llm_fn(
                prompt=prompt,
                max_tokens=self.config.max_reflection_length,
                temperature=0.3,
            )

            text = response.strip() if response else ""
            return self._parse_structured_reflection(text)
        except Exception as e:
            logger.warning(f"LLM reflection failed: {e}")
            return self._rule_based_reflect()

    def _parse_structured_reflection(self, text: str) -> ReflectionResult:
        """解析结构化反思输出"""
        import re

        root_cause = None
        correction = None
        should_retry = True
        category = None

        for line in text.split("\n"):
            line_stripped = line.strip()
            lower = line_stripped.lower()
            # 匹配 "Failure Category: A" 或 "Category: B"
            if "failure category" in lower or (lower.startswith("category:") and category is None):
                m = re.search(r'[A-E]', line_stripped.upper())
                if m:
                    category = m.group()
            elif "root cause" in lower or "root_cause" in lower:
                val = line_stripped.split(":", 1)[-1].strip() if ":" in line_stripped else line_stripped
                if val:
                    root_cause = val
            elif "correction" in lower or "suggested" in lower:
                val = line_stripped.split(":", 1)[-1].strip() if ":" in line_stripped else line_stripped
                if val:
                    correction = val
            elif "retry" in lower:
                if "no" in lower or "false" in lower:
                    should_retry = False

        # 把 category 信息附到 root_cause 前面，方便下游使用
        if category and root_cause:
            root_cause = f"[{category}] {root_cause}"
        elif category:
            root_cause = f"[{category}] (unspecified root cause)"
        elif not root_cause:
            # 回退：取第一行非空内容
            for line in text.split("\n"):
                if line.strip():
                    root_cause = line.strip()
                    break

        return ReflectionResult(
            reflection_text=text,
            root_cause=root_cause,
            suggested_correction=correction,
            should_retry=should_retry,
            confidence=0.75 if category else 0.6,
        )

    def _summarize_failures(self) -> str:
        """总结失败历史"""
        if not self.failure_history:
            return "No failures"

        recent = self.failure_history[-5:]  # 最近 5 次
        parts = []
        for f in recent:
            parts.append(f"[Iter {f.iteration}] {f.tool_name}: {f.error_type} - {f.error_message[:100]}")

        return "\n".join(parts)

    def get_failure_stats(self) -> dict[str, Any]:
        """获取失败统计"""
        error_types = Counter(f.error_type for f in self.failure_history)
        return {
            "total_failures": len(self.failure_history),
            "consecutive_failures": self.consecutive_failures,
            "error_types": dict(error_types),
            "last_failure_time": self.failure_history[-1].timestamp if self.failure_history else None,
        }

    def clear(self) -> None:
        """清除失败历史"""
        self.failure_history = []
        self.consecutive_failures = 0
        self._last_reflected_failure_count = 0


# 前向引用
from .config import ReflectionConfig
