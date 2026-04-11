"""
OpenClawPro Harness — Agent 执行框架。

提供 NanoBotAgent 及其子模块（memory, control, collaboration, procedure），
封装 nanobot 核心引擎，供评测框架（ClawEvalKit）等外部项目调用。
"""

from .agent import NanoBotAgent, BaseAgent, AgentResult

__all__ = ["NanoBotAgent", "BaseAgent", "AgentResult"]
