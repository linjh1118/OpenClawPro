"""
Letta (MemGPT) 长期记忆适配器 — SOTA 轨道 B (§6 Memory 层)

对应假设 H2: State Abstraction Failure
验证方式: Letta 持久 memory blocks 替代 T1 episodic buffer，测 C 类 CRR

架构说明:
  - Letta 维护三类 memory block: core memory / archival / recall
  - 每次 tool call 后 agent 可调 Letta 的 `archival_memory_insert` 将关键事实持久化
  - 每次新 tool call 前注入 core_memory 摘要进 system prompt
  - 与 T1 (store.py EpisodicStore) 的区别: Letta 跨 session 持久、有 PostgreSQL 后端、
    自动做 memory consolidation 而非纯 append

安装依赖:
  pip install letta

快速验证 (无 Postgres 的 SQLite 模式):
  python3 -c "from OpenClawPro.harness.agent.memory.letta_adapter import LettaMemoryHarness; print('OK')"
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agent.memory.letta")


class LettaMemoryHarness:
    """将 Letta stateful agent 包装成与现有 harness 兼容的记忆层。

    用法:
        harness = LettaMemoryHarness(base_agent, agent_id="exp88_test")
        result = harness.execute("帮我分析这个数据集")
        harness.cleanup()

    与 T1 EpisodicStore 的对比 (§6 SOTA 轨 vs §5 minimal 轨):
        T1 (minimal): 每轮 tool call 后 append → 同 session 内短期存储
        Letta (SOTA): 跨 session 持久、archival 自动合并、recall 语义搜索
    """

    def __init__(
        self,
        base_agent,
        agent_id: str = "exp88_letta_agent",
        letta_api_key: Optional[str] = None,
        letta_base_url: Optional[str] = None,
        model: str = "openai/gpt-4o",
        use_sqlite: bool = True,
    ):
        """初始化 Letta 记忆适配器。

        Args:
            base_agent: 内层 BaseAgent 实例 (提供 execute/execute_multi 接口)
            agent_id: Letta agent 唯一 ID，跨 session 复用同一 stateful agent
            letta_api_key: Letta Cloud API key (本地部署可为 None)
            letta_base_url: Letta server URL，默认 http://localhost:8283
            model: Letta 所用的 LLM 模型 ID
            use_sqlite: True=本地 SQLite 单机模式（快速开发用），False=PostgreSQL 生产模式
        """
        self.base_agent = base_agent
        self.agent_id = agent_id
        self.model = model
        self.use_sqlite = use_sqlite
        self._client = None
        self._letta_agent_state = None

        self._api_key = letta_api_key or os.environ.get("LETTA_API_KEY")
        self._base_url = letta_base_url or os.environ.get("LETTA_BASE_URL", "http://localhost:8283")

    def _get_client(self):
        """惰性初始化 Letta 客户端。"""
        if self._client is not None:
            return self._client
        try:
            from letta_client import Letta
        except ImportError:
            raise ImportError(
                "Letta 未安装。请运行: pip install letta-client\n"
                "文档: https://docs.letta.com"
            )
        if self._api_key:
            self._client = Letta(api_key=self._api_key, base_url=self._base_url)
        else:
            # 本地服务器模式 (需要先运行 `letta server` 或 `docker compose up`)
            self._client = Letta(base_url=self._base_url)
        logger.info(f"[LettaMemoryHarness] 连接 Letta server: {self._base_url}")
        return self._client

    def _get_or_create_agent(self):
        """获取或创建 Letta stateful agent。"""
        if self._letta_agent_state is not None:
            return self._letta_agent_state
        client = self._get_client()
        # 搜索已有同名 agent
        existing = [a for a in client.agents.list() if a.name == self.agent_id]
        if existing:
            self._letta_agent_state = existing[0]
            logger.info(f"[LettaMemoryHarness] 复用已有 Letta agent: {self.agent_id}")
        else:
            self._letta_agent_state = client.agents.create(
                name=self.agent_id,
                model=self.model,
                memory_blocks=[
                    {"label": "human", "value": "实验 exp88 ClawRecipe 研究任务"},
                    {"label": "persona", "value": "我是一个专注于记忆管理和状态跟踪的研究助手"},
                ],
            )
            logger.info(f"[LettaMemoryHarness] 创建新 Letta agent: {self.agent_id}")
        return self._letta_agent_state

    def get_core_memory_summary(self) -> str:
        """读取当前 core memory 内容，注入 base_agent 的 system prompt。

        Returns:
            core memory 的字符串摘要（constraints / derived_facts / pending_subgoals / artifact_paths）
        """
        try:
            client = self._get_client()
            agent_state = self._get_or_create_agent()
            # 从 memory blocks 中提取 core memory
            blocks = client.agents.core_memory.retrieve(agent_state.id)
            parts = []
            for block in blocks:
                if hasattr(block, "label") and hasattr(block, "value"):
                    parts.append(f"[{block.label}] {block.value}")
            return "\n".join(parts) if parts else ""
        except Exception as e:
            logger.warning(f"[LettaMemoryHarness] 读取 core memory 失败: {e}")
            return ""

    def insert_archival_memory(self, content: str) -> bool:
        """将新发现的事实插入 archival memory（跨 session 持久）。

        Args:
            content: 需要持久化的内容（如工具调用结果、新发现的约束）

        Returns:
            True=插入成功
        """
        try:
            client = self._get_client()
            agent_state = self._get_or_create_agent()
            client.agents.archival_memory.insert(agent_state.id, text=content)
            logger.debug(f"[LettaMemoryHarness] archival insert: {content[:80]}...")
            return True
        except Exception as e:
            logger.warning(f"[LettaMemoryHarness] archival insert 失败: {e}")
            return False

    def recall_memory(self, query: str, top_k: int = 5) -> List[str]:
        """语义搜索 recall memory，返回最相关的历史 facts。

        Args:
            query: 搜索 query
            top_k: 返回条数

        Returns:
            相关记忆列表
        """
        try:
            client = self._get_client()
            agent_state = self._get_or_create_agent()
            results = client.agents.archival_memory.list(
                agent_state.id,
                query=query,
                limit=top_k,
            )
            return [r.text for r in results if hasattr(r, "text")]
        except Exception as e:
            logger.warning(f"[LettaMemoryHarness] recall 失败: {e}")
            return []

    def execute(self, prompt: str, session_id: Optional[str] = None):
        """执行 prompt，注入 Letta core memory 后委托给 base_agent。

        Args:
            prompt: 用户任务描述
            session_id: 可选 session ID

        Returns:
            AgentResult（来自 base_agent）
        """
        # 1. 读取当前记忆状态注入 prompt
        memory_summary = self.get_core_memory_summary()
        if memory_summary:
            augmented_prompt = (
                f"[LETTA CORE MEMORY]\n{memory_summary}\n\n"
                f"[TASK]\n{prompt}"
            )
        else:
            augmented_prompt = prompt

        # 2. 执行 base_agent
        result = self.base_agent.execute(augmented_prompt, session_id=session_id)

        # 3. 将 tool call 结果保存到 archival memory
        if result.transcript:
            for entry in result.transcript:
                if isinstance(entry, dict) and entry.get("type") == "tool_result":
                    content = str(entry.get("content", ""))[:500]
                    if content:
                        self.insert_archival_memory(
                            f"[session:{session_id or 'default'}] tool_result: {content}"
                        )

        return result

    def cleanup(self) -> None:
        """清理本地缓存（Letta server 端 agent 仍持久存在）。"""
        self._client = None
        self._letta_agent_state = None
        logger.info(f"[LettaMemoryHarness] client cleared (agent '{self.agent_id}' persists in Letta server)")

    @staticmethod
    def install_guide() -> str:
        """返回安装说明（适合打印给队友）。"""
        return """
=== Letta (MemGPT) 安装指南 ===

1. 安装 Python SDK:
   pip install letta-client

2. 本地服务器 (快速开发, SQLite 模式):
   pip install letta
   letta server       # 默认端口 8283

3. 生产部署 (PostgreSQL):
   export LETTA_PG_URI="postgresql://user:pass@host:5432/letta"
   letta server --config config.yaml

4. 验证连接:
   python3 -c "
   from letta_client import Letta
   c = Letta(base_url='http://localhost:8283')
   print(c.agents.list())
   "

5. 参考文档: https://docs.letta.com
"""
