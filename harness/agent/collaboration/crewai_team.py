"""
CrewAI 多 Agent 验证团队适配器 — SOTA 轨道 B (§6 Multi-Agent 层)

对应假设 H3 (D段): outcome verification 闭环断裂
验证方式: CrewAI executor+verifier crew 替代 T3 minimal dual-role，测 D 类 CRR

架构说明:
  - Executor agent: 执行工具调用，完成任务目标
  - Verifier agent: 根据任务目标检查 executor 的输出
  - Manager agent (可选): 协调 executor/verifier 的多轮交互
  - 与 T3 (minimal roles.py) 的区别: CrewAI 提供 role/backstory/goal DSL、
    内置任务分解、20 行起步；T3 是自研的简单消息 handoff

安装依赖:
  pip install crewai

快速验证:
  python3 -c "from OpenClawPro.harness.agent.collaboration.crewai_team import CrewAIVerifierHarness; print('OK')"
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agent.collaboration.crewai")


class CrewAIVerifierHarness:
    """CrewAI executor+verifier 双角色团队包装器。

    用法:
        harness = CrewAIVerifierHarness(base_agent, model="openai/gpt-4o-mini")
        result = harness.execute("分析这个 CSV 文件并生成报告")
        harness.cleanup()

    与 T3 minimal verifier 的对比 (§6 SOTA vs §5 minimal):
        T3 (minimal): 自研 roles.py，executor 完成后 verifier 单轮 review
        CrewAI (SOTA): 内置 role DSL + task dependencies + 可迭代 review
    """

    def __init__(
        self,
        base_agent,
        model: str = "openai/gpt-4o-mini",
        api_key: Optional[str] = None,
        max_rounds: int = 3,
        verbose: bool = False,
    ):
        """初始化 CrewAI 验证团队。

        Args:
            base_agent: 内层 BaseAgent 实例（提供任务上下文）
            model: CrewAI 使用的 LLM 模型 ID
            api_key: OpenAI 兼容 API key
            max_rounds: executor 和 verifier 最多交互轮次
            verbose: 是否打印 CrewAI 内部日志
        """
        self.base_agent = base_agent
        self.model = model
        self.max_rounds = max_rounds
        self.verbose = verbose
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._crew = None

    def _build_crew(self, task_description: str):
        """惰性构建 CrewAI crew（每次任务重新构建以避免状态污染）。

        Args:
            task_description: 任务描述，用于配置 executor/verifier 的 goal

        Returns:
            crewai.Crew 实例
        """
        try:
            from crewai import Agent, Crew, Task
            from crewai.llm import LLM
        except ImportError:
            raise ImportError(
                "CrewAI 未安装。请运行: pip install crewai\n"
                "文档: https://docs.crewai.com"
            )

        llm = LLM(model=self.model, api_key=self._api_key)

        executor = Agent(
            role="Task Executor",
            goal=f"Complete the following task accurately: {task_description[:200]}",
            backstory=(
                "You are an expert AI agent with strong tool-use skills. "
                "You execute tasks step by step, using available tools precisely."
            ),
            llm=llm,
            verbose=self.verbose,
            allow_delegation=False,
        )

        verifier = Agent(
            role="Task Verifier",
            goal=(
                "Critically review the executor's output for correctness, "
                "completeness, and adherence to the task requirements. "
                "Identify any failures in verification/recovery (Category D errors)."
            ),
            backstory=(
                "You are a meticulous quality reviewer. "
                "You check outputs against task goals, find logical errors, "
                "and request corrections when necessary."
            ),
            llm=llm,
            verbose=self.verbose,
            allow_delegation=False,
        )

        execution_task = Task(
            description=task_description,
            agent=executor,
            expected_output="A complete, accurate response to the task with all required deliverables.",
        )

        verification_task = Task(
            description=(
                f"Review the executor's output for task: '{task_description[:200]}'. "
                "Check: (1) Are all task requirements met? "
                "(2) Are there any errors, missing steps, or incorrect outputs? "
                "(3) Provide a PASS/FAIL verdict with specific feedback."
            ),
            agent=verifier,
            expected_output="Verification report: PASS or FAIL with detailed feedback.",
            context=[execution_task],
        )

        crew = Crew(
            agents=[executor, verifier],
            tasks=[execution_task, verification_task],
            verbose=self.verbose,
            max_rpm=10,
        )
        return crew

    def execute(self, prompt: str, session_id: Optional[str] = None):
        """执行 prompt，经由 executor→verifier 双角色流程。

        Falls back to base_agent 如果 CrewAI 不可用。

        Args:
            prompt: 用户任务描述
            session_id: 可选 session ID

        Returns:
            AgentResult（内容为 crew 最终输出）
        """
        try:
            crew = self._build_crew(prompt)
            crew_result = crew.kickoff(inputs={"task": prompt})
            # 将 crew 结果包装成 base_agent 同类型 AgentResult
            result = self.base_agent.execute.__class__
            # 直接从 base_agent 获取 AgentResult 类并构造
            from OpenClawPro.harness.agent.base import AgentResult
            return AgentResult(
                status="success",
                content=str(crew_result),
                transcript=[
                    {"type": "crew_output", "content": str(crew_result), "session_id": session_id}
                ],
                usage={},
            )
        except ImportError:
            logger.warning("[CrewAIVerifierHarness] CrewAI 不可用，降级到 base_agent")
            return self.base_agent.execute(prompt, session_id=session_id)
        except Exception as e:
            logger.error(f"[CrewAIVerifierHarness] crew 执行失败: {e}, 降级到 base_agent")
            return self.base_agent.execute(prompt, session_id=session_id)

    def cleanup(self) -> None:
        """清理 crew 实例。"""
        self._crew = None

    @staticmethod
    def install_guide() -> str:
        """返回安装说明。"""
        return """
=== CrewAI 安装指南 ===

1. 安装:
   pip install crewai

2. 快速验证:
   python3 -c "
   from crewai import Agent, Task, Crew
   print('CrewAI OK')
   "

3. 注意: CrewAI 默认使用 OpenAI 模型
   如需使用其他模型，设置 LLM() 时指定 model= 和 base_url=

4. 参考文档: https://docs.crewai.com
"""
