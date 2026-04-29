"""
ACE (Agentic Context Engineering) Skill Playbook 适配器 — SOTA 轨道 B (§6 Skills 层)

对应假设 H1: capability-activation gap (E类技能利用缺陷)
验证方式: ACE Generator/Reflector/Curator 三角色演化 playbook，替代 T4a 静态 procedure card，测 E1 CRR

架构说明 (arXiv 2510.04618, ICLR 2026):
  ACE 核心三角色:
  - Generator: 执行任务，生成推理轨迹
  - Reflector: 从成功/失败轨迹中蒸馏 concrete insights
  - Curator: 将 insights 增量更新进结构化 playbook (grow-and-refine)

  与 T4a DenseRetriever + SkillCard 的区别:
    T4a (minimal): 静态预编译 procedure card，检索后注入
    ACE (SOTA): playbook 随任务经验动态演化，防止 brevity bias 和 context collapse

安装依赖:
  git clone https://github.com/ace-agent/ace && cd ace && pip install -e .
  备选: pip install kayba-agentic-context-engine (第三方简化版)

快速验证:
  python3 -c "from OpenClawPro.harness.agent.procedure.ace_card import ACEPlaybookHarness; print('OK')"
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agent.procedure.ace")


class ACEPlaybook:
    """ACE grow-and-refine playbook 存储结构。

    一个 Playbook 就是一组结构化的 bullet（itemized bullets），
    每个 bullet 是一条可操作的 skill/procedure 规则。
    ACE 的关键设计: bullet 是增量添加/更新的，而不是整体重写（防 context collapse）。
    """

    def __init__(self, name: str, playbook_dir: Optional[Path] = None):
        self.name = name
        self.bullets: List[Dict[str, Any]] = []
        self._dir = playbook_dir or Path("/tmp/ace_playbooks")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{name}.json"
        self._load()

    def _load(self):
        """从磁盘加载 playbook。"""
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.bullets = data.get("bullets", [])
                logger.info(f"[ACEPlaybook] 加载 '{self.name}': {len(self.bullets)} bullets")
            except Exception as e:
                logger.warning(f"[ACEPlaybook] 加载失败: {e}, 使用空 playbook")
                self.bullets = []

    def _save(self):
        """持久化 playbook 到磁盘。"""
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump({"name": self.name, "bullets": self.bullets}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[ACEPlaybook] 保存失败: {e}")

    def add_bullet(self, content: str, source: str = "reflector") -> int:
        """增量添加一条 bullet（grow）。

        Args:
            content: bullet 内容（具体可操作的 insight）
            source: 来源标识（reflector / manual）

        Returns:
            新 bullet 的 index
        """
        bullet = {"id": len(self.bullets), "content": content, "source": source, "hits": 0}
        self.bullets.append(bullet)
        self._save()
        return bullet["id"]

    def update_bullet(self, idx: int, new_content: str) -> bool:
        """精细更新某条 bullet（refine，不重写整体）。

        Args:
            idx: bullet index
            new_content: 更新后内容

        Returns:
            True=更新成功
        """
        if 0 <= idx < len(self.bullets):
            self.bullets[idx]["content"] = new_content
            self._save()
            return True
        return False

    def retrieve_relevant(self, query: str, top_k: int = 5) -> List[str]:
        """检索最相关的 bullets（简单关键词匹配，生产可换 embedding 检索）。

        Args:
            query: 任务描述
            top_k: 返回条数

        Returns:
            相关 bullet 内容列表
        """
        if not self.bullets:
            return []
        query_lower = query.lower()
        scored = []
        for b in self.bullets:
            content_lower = b["content"].lower()
            # 简单共词分数
            words = set(query_lower.split())
            match_count = sum(1 for w in words if w in content_lower and len(w) > 3)
            scored.append((match_count, b["content"]))
        scored.sort(key=lambda x: -x[0])
        return [content for _, content in scored[:top_k] if _ > 0]

    def to_prompt_context(self, query: str, top_k: int = 5) -> str:
        """生成 ACE playbook 注入 prompt 的上下文字符串。"""
        relevant = self.retrieve_relevant(query, top_k)
        if not relevant:
            return ""
        bullets_text = "\n".join(f"- {b}" for b in relevant)
        return f"[ACE PLAYBOOK: {self.name}]\n{bullets_text}"


class ACEReflector:
    """ACE Reflector 角色: 从执行轨迹中蒸馏 insights。"""

    def __init__(self, llm_call_fn):
        """
        Args:
            llm_call_fn: 接受 prompt str 返回 str 的 LLM 调用函数
        """
        self.llm_call = llm_call_fn

    def reflect(self, task: str, transcript: List[Dict], success: bool) -> List[str]:
        """从执行轨迹中提取 concrete insights。

        Args:
            task: 原始任务描述
            transcript: 执行轨迹（AgentResult.transcript）
            success: 是否成功

        Returns:
            0-5 条具体可操作的 insight bullet
        """
        status = "successful" if success else "failed"
        transcript_text = json.dumps(transcript[:10], ensure_ascii=False)[:2000]

        prompt = (
            f"Task: {task}\n\n"
            f"Execution transcript ({status}):\n{transcript_text}\n\n"
            "As a Reflector in the ACE framework, extract 1-3 concrete, actionable insights "
            "that would help future agents perform this type of task better. "
            "Each insight should be specific and reusable (not task-specific). "
            "Format: one insight per line, starting with a verb (e.g., 'Always verify...', 'When X, use Y...')."
        )
        try:
            response = self.llm_call(prompt)
            insights = [l.strip() for l in response.strip().split("\n") if l.strip() and len(l.strip()) > 10]
            return insights[:5]
        except Exception as e:
            logger.warning(f"[ACEReflector] reflect 失败: {e}")
            return []


class ACEPlaybookHarness:
    """ACE Generator/Reflector/Curator 三角色 playbook 演化适配器。

    用法:
        harness = ACEPlaybookHarness(base_agent, playbook_name="exp88_skills")
        result = harness.execute("使用 Python 工具分析 CSV 文件")
        # 执行后自动更新 playbook
        harness.cleanup()

    与 T4a DenseRetriever+SkillCard 的对比 (§6 SOTA vs §5 minimal):
        T4a (minimal): 静态 procedure card + BERT 检索，card 不更新
        ACE (SOTA): playbook 随每次执行动态演化（Reflector 蒸馏 → Curator 增量更新）
    """

    def __init__(
        self,
        base_agent,
        playbook_name: str = "exp88_default",
        playbook_dir: Optional[Path] = None,
        llm_call_fn=None,
        enable_reflection: bool = True,
        top_k_bullets: int = 5,
    ):
        """初始化 ACE Playbook 适配器。

        Args:
            base_agent: 内层 BaseAgent 实例
            playbook_name: playbook 文件名（不同 task 域可用不同 playbook）
            playbook_dir: playbook 存储目录（默认 /tmp/ace_playbooks）
            llm_call_fn: LLM 调用函数 (str→str)，用于 Reflector 角色；
                         None=不启用自动 reflection（只用 playbook 注入）
            enable_reflection: 是否在每次 execute 后运行 Reflector 更新 playbook
            top_k_bullets: 注入 prompt 的 bullet 数量
        """
        self.base_agent = base_agent
        self.playbook = ACEPlaybook(playbook_name, playbook_dir)
        self.reflector = ACEReflector(llm_call_fn) if llm_call_fn else None
        self.enable_reflection = enable_reflection and (llm_call_fn is not None)
        self.top_k_bullets = top_k_bullets

    def seed_playbook(self, domain_hints: List[str]) -> None:
        """用领域 hint 初始化 playbook（冷启动时调用）。

        Args:
            domain_hints: 人工编写的初始 bullets（如 T4a 的 procedure card 内容）
        """
        for hint in domain_hints:
            self.playbook.add_bullet(hint, source="manual_seed")
        logger.info(f"[ACEPlaybookHarness] 冷启动: 添加 {len(domain_hints)} 条种子 bullets")

    def execute(self, prompt: str, session_id: Optional[str] = None):
        """Generator 角色: 注入 playbook 上下文 → 执行 base_agent → Reflector 更新 playbook。

        Args:
            prompt: 用户任务描述
            session_id: 可选 session ID

        Returns:
            AgentResult（来自 base_agent）
        """
        # Curator: 检索相关 bullets 注入 prompt
        playbook_context = self.playbook.to_prompt_context(prompt, self.top_k_bullets)
        if playbook_context:
            augmented_prompt = f"{playbook_context}\n\n[TASK]\n{prompt}"
        else:
            augmented_prompt = prompt

        # Generator: 执行
        result = self.base_agent.execute(augmented_prompt, session_id=session_id)

        # Reflector: 从轨迹中蒸馏 insights 并 grow-and-refine playbook
        if self.enable_reflection and self.reflector and result.transcript:
            success = result.status == "success"
            insights = self.reflector.reflect(prompt, result.transcript, success)
            for insight in insights:
                # Curator: 增量 grow（简单策略: 超过 20 条时找最相似的 bullet 更新而非新增）
                if len(self.playbook.bullets) >= 20:
                    # 找最相似 bullet 并 refine
                    similar = self.playbook.retrieve_relevant(insight, top_k=1)
                    if similar:
                        target_idx = next(
                            (i for i, b in enumerate(self.playbook.bullets) if b["content"] == similar[0]),
                            None,
                        )
                        if target_idx is not None:
                            self.playbook.update_bullet(target_idx, insight)
                            logger.debug(f"[ACEPlaybookHarness] refined bullet[{target_idx}]")
                            continue
                self.playbook.add_bullet(insight, source="reflector")
            logger.info(f"[ACEPlaybookHarness] Reflector 添加/更新 {len(insights)} 条 insights，总计 {len(self.playbook.bullets)} bullets")

        return result

    def cleanup(self) -> None:
        """playbook 已自动 persist，无需额外清理。"""
        pass

    def get_playbook_stats(self) -> Dict[str, Any]:
        """返回 playbook 统计信息（用于 §6 实验 log）。"""
        return {
            "name": self.playbook.name,
            "bullet_count": len(self.playbook.bullets),
            "reflection_enabled": self.enable_reflection,
            "top_k_bullets": self.top_k_bullets,
        }

    @staticmethod
    def install_guide() -> str:
        """返回安装说明。"""
        return """
=== ACE (Agentic Context Engineering) 安装指南 ===

1. 官方仓库:
   git clone https://github.com/ace-agent/ace
   cd ace && pip install -e .

2. 第三方简化版 (更易安装):
   pip install kayba-agentic-context-engine

3. 本地使用 ACEPlaybookHarness 不需要官方仓库：
   只需 base_agent + 一个 llm_call_fn 即可

4. 快速验证:
   python3 -c "
   from OpenClawPro.harness.agent.procedure.ace_card import ACEPlaybookHarness
   print('ACE harness OK, bullet count:', 0)
   "

5. 论文: arXiv 2510.04618 (ICLR 2026)
   GitHub: https://github.com/ace-agent/ace
"""
