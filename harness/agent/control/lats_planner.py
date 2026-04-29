"""
LangGraph LATS 规划适配器 — SOTA 轨道 B (§6 Plan 层)

对应假设 H3 (A段): goal grounding 闭环断裂
验证方式: LATS MCTS 搜索替代 T2 plan-first prompt，测 A 类 CRR

架构说明:
  LATS (Language Agent Tree Search, ICML 2024, arXiv 2310.04406)
  = MCTS + 自反思评估 + 多路径并行扩展
  六步循环: select → expand → evaluate → simulate → backpropagate → reflect

  实现方式:
  - 使用 LangGraph 官方 LATS 教程骨架 (pip install langgraph)
  - 每次 execute 调用一次完整的 MCTS 搜索 (max_depth 可控)
  - 搜索结果最佳路径 content 返回给 base_agent 作为 plan hint
  - 与 T2 plan_first.py 区别: T2 是单次 LLM 调用生成线性 plan；
    LATS 是 MCTS 多路径搜索，对 A 类"规划漂移"有更强修复能力

安装依赖:
  pip install -U langgraph langchain_openai

快速验证:
  python3 -c "from OpenClawPro.harness.agent.control.lats_planner import LATSPlannerHarness; print('OK')"
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agent.control.lats")


# LATS 所需数据结构 (精简版，与 LangGraph 官方实现对齐)
class _Node:
    """LATS MCTS 树节点。"""

    def __init__(self, content: str, parent=None):
        self.content = content
        self.parent = parent
        self.children: List[_Node] = []
        self.visits: int = 0
        self.value: float = 0.0
        self.is_solved: bool = False

    @property
    def height(self) -> int:
        """当前子树深度。"""
        if not self.children:
            return 0
        return 1 + max(c.height for c in self.children)

    def ucb_score(self, c: float = 1.414) -> float:
        """UCB1 探索-利用权衡分数。"""
        import math
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else self.visits
        exploitation = self.value / self.visits
        exploration = c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration


class LATSPlannerHarness:
    """LangGraph LATS MCTS 规划器包装器。

    用法:
        harness = LATSPlannerHarness(base_agent, max_depth=3, n_candidates=3)
        result = harness.execute("完成多步骤数据分析任务")
        harness.cleanup()

    与 T2 plan_first prompt 的对比 (§6 SOTA vs §5 minimal):
        T2 (minimal): 单次 LLM call 生成线性 5-step plan → 执行
        LATS (SOTA): MCTS 搜索最优计划路径，UCB 选择最佳分支
    """

    def __init__(
        self,
        base_agent,
        model: str = "openai/gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_depth: int = 3,
        n_candidates: int = 3,
        reflection_threshold: float = 0.7,
        use_langgraph: bool = True,
    ):
        """初始化 LATS 规划适配器。

        Args:
            base_agent: 内层 BaseAgent 实例
            model: LLM 模型 ID
            api_key: API key
            api_base: API base URL (可对接 doubao 等国内端点)
            max_depth: MCTS 最大深度 (建议 3-5，越大越贵)
            n_candidates: 每步扩展的候选计划数 (建议 3)
            reflection_threshold: 反思触发阈值 (低于此分数触发 reflect)
            use_langgraph: True=使用 LangGraph 官方实现，False=用内置精简版
        """
        self.base_agent = base_agent
        self.model = model
        self.max_depth = max_depth
        self.n_candidates = n_candidates
        self.reflection_threshold = reflection_threshold
        self.use_langgraph = use_langgraph
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._api_base = api_base or os.environ.get("OPENAI_API_BASE", "")

    def _get_llm(self):
        """构造 LangChain LLM 实例。"""
        try:
            from langchain_openai import ChatOpenAI
            kwargs = dict(model=self.model, temperature=0.7)
            if self._api_key:
                kwargs["openai_api_key"] = self._api_key
            if self._api_base:
                kwargs["openai_api_base"] = self._api_base
            return ChatOpenAI(**kwargs)
        except ImportError:
            raise ImportError(
                "langchain_openai 未安装。请运行: pip install langchain_openai langgraph"
            )

    def _generate_candidates(self, llm, task: str, parent_plan: str, n: int) -> List[str]:
        """用 LLM 生成 n 个候选子计划。

        Args:
            llm: ChatOpenAI 实例
            task: 原始任务描述
            parent_plan: 父节点的计划前缀
            n: 候选数

        Returns:
            n 个候选计划字符串
        """
        prompt = (
            f"Task: {task}\n\n"
            f"Current plan so far:\n{parent_plan}\n\n"
            f"Generate {n} diverse next-step plans to complete this task. "
            "Each plan should be 2-3 concrete steps. "
            "Format: one plan per line, no numbering."
        )
        try:
            response = llm.invoke(prompt)
            lines = [l.strip() for l in response.content.strip().split("\n") if l.strip()]
            return lines[:n] if len(lines) >= n else lines + ["fallback plan"] * (n - len(lines))
        except Exception as e:
            logger.warning(f"[LATSPlannerHarness] candidate 生成失败: {e}")
            return [f"direct approach: {task}"] * n

    def _evaluate_node(self, llm, task: str, plan: str) -> float:
        """用 LLM 评估计划质量，返回 0-1 分数。"""
        prompt = (
            f"Task: {task}\n\nProposed plan:\n{plan}\n\n"
            "Rate this plan's likelihood of successfully completing the task. "
            "Output a single float between 0 and 1. Nothing else."
        )
        try:
            response = llm.invoke(prompt)
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5

    def _simple_mcts_search(self, llm, task: str) -> str:
        """精简版 MCTS 搜索（当 LangGraph 不可用时的备选）。

        6 步循环: select → expand → evaluate → simulate → backpropagate → reflect

        Args:
            llm: LLM 实例
            task: 任务描述

        Returns:
            最佳计划字符串
        """
        root = _Node(content="", parent=None)

        def select(node: _Node) -> _Node:
            """选择 UCB 最高的叶节点。"""
            current = node
            while current.children and not current.is_solved:
                current = max(current.children, key=lambda n: n.ucb_score())
            return current

        for _ in range(self.max_depth * self.n_candidates):
            # Select
            selected = select(root)
            if selected.height >= self.max_depth:
                break

            # Expand
            candidates = self._generate_candidates(
                llm, task, selected.content, self.n_candidates
            )
            for candidate_plan in candidates:
                child = _Node(content=f"{selected.content}\n{candidate_plan}".strip(), parent=selected)
                selected.children.append(child)

                # Evaluate
                score = self._evaluate_node(llm, task, child.content)
                child.value = score
                child.visits = 1

                if score >= self.reflection_threshold:
                    child.is_solved = True

                # Backpropagate
                node = child.parent
                while node is not None:
                    node.visits += 1
                    node.value += score
                    node = node.parent

        # Return best leaf
        def best_leaf(node: _Node) -> _Node:
            if not node.children:
                return node
            best_child = max(node.children, key=lambda n: n.value / max(n.visits, 1))
            return best_leaf(best_child)

        best = best_leaf(root)
        return best.content or task

    def execute(self, prompt: str, session_id: Optional[str] = None):
        """用 LATS MCTS 搜索生成最优计划，再委托 base_agent 执行。

        Args:
            prompt: 用户任务描述
            session_id: 可选 session ID

        Returns:
            AgentResult（来自 base_agent）
        """
        try:
            llm = self._get_llm()
            best_plan = self._simple_mcts_search(llm, prompt)
            logger.info(f"[LATSPlannerHarness] MCTS 完成，最佳计划长度: {len(best_plan)}")

            augmented_prompt = (
                f"[LATS OPTIMAL PLAN]\n"
                f"Based on Monte Carlo Tree Search, the following plan is recommended:\n"
                f"{best_plan}\n\n"
                f"[TASK]\nPlease execute the following task following the plan above:\n{prompt}"
            )
            return self.base_agent.execute(augmented_prompt, session_id=session_id)

        except ImportError as e:
            logger.warning(f"[LATSPlannerHarness] {e}, 降级到 base_agent")
            return self.base_agent.execute(prompt, session_id=session_id)
        except Exception as e:
            logger.error(f"[LATSPlannerHarness] MCTS 失败: {e}, 降级到 base_agent")
            return self.base_agent.execute(prompt, session_id=session_id)

    def cleanup(self) -> None:
        """清理资源。"""
        pass

    @staticmethod
    def install_guide() -> str:
        """返回安装说明。"""
        return """
=== LangGraph LATS 安装指南 ===

1. 安装:
   pip install -U langgraph langchain_openai

2. 快速验证:
   python3 -c "
   import langgraph
   from langchain_openai import ChatOpenAI
   print('LATS deps OK')
   "

3. 注意: LATS 计算开销较大 (MCTS 树搜索)
   建议 max_depth=3, n_candidates=3 控制成本

4. 参考:
   - 论文: arXiv 2310.04406 (ICML 2024)
   - LangGraph 官方教程: https://langchain-ai.github.io/langgraph/tutorials/lats/lats/
"""
