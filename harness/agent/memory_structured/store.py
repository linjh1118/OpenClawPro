"""StructuredMemoryStore — Core structured state tracking for H2."""

import logging
import time
import uuid
from typing import Dict, List, Optional

from .config import StructuredMemoryConfig
from .slot import StateSlot
from .update_policy import (
    UpdatePolicy,
    SlotUpdateResult,
    parse_tool_result_for_constraints,
    parse_tool_result_for_derived_facts,
    detect_subgoal_completion,
    parse_tool_result_for_artifacts,
    extract_constraints_from_instruction,
    extract_pending_subgoals_from_instruction,
)


class StructuredMemoryStore:
    """Structured state tracker for H2 verification.

    Maintains four state slots that are actively UPDATED as the agent works:
    - constraints: task constraints extracted from tool results
    - derived_facts: facts derived from tool results (not directly observed)
    - pending_subgoals: subgoals pending completion
    - artifact_paths: paths to files created or modified

    Unlike EpisodicMemoryStore (T1) which passively stores tool results,
    this store actively TRANSFORMS tool results into decision-relevant
    structured representations.
    """

    _logger = logging.getLogger("memory.structured")

    def __init__(self, config: StructuredMemoryConfig):
        self.config = config
        self._iteration = 0
        self._enabled = config.enabled

        # State slots organized by type
        self._slots: Dict[str, List[StateSlot]] = {
            "constraints": [],
            "derived_facts": [],
            "pending_subgoals": [],
            "artifact_paths": [],
        }

        # Index for fast lookup
        self._slot_index: Dict[str, StateSlot] = {}

        # Tool result buffer — collected between LLM updates (avoids per-call LLM cost)
        self._tool_result_buffer: List[dict] = []
        self._last_llm_update_iteration: int = 0

    def reset(self) -> None:
        """Reset structured memory for new task."""
        self._slots = {k: [] for k in self._slots}
        self._slot_index.clear()
        self._iteration = 0
        self._tool_result_buffer.clear()
        self._last_llm_update_iteration = 0
        self._logger.debug("[StructuredMemoryStore] Reset for new task")

    def initialize_from_task(self, task_description: str, iteration: int = 0) -> None:
        """Parse task instruction (LLM-based) and populate initial structured state.

        先尝试用 LLM 提取结构化状态，失败时回退到正则提取。

        Args:
            task_description: 任务指令原文
            iteration: 初始 iteration（通常为 1）
        """
        if not self._enabled or not task_description:
            return

        # 优先尝试 LLM 提取
        if self.config.llm_model:
            try:
                self._llm_initialize_from_task(task_description, iteration)
                return
            except Exception as e:
                self._logger.warning(f"[StructuredMemoryStore] LLM extraction failed: {e}, falling back to regex")

        # 回退到正则提取
        self._regex_initialize_from_task(task_description, iteration)

    def _llm_initialize_from_task(self, task_description: str, iteration: int) -> None:
        """使用 LLM 从任务指令中提取结构化状态（同步版本）。

        在子线程中运行 asyncio.run()，避免阻塞主线程的 event loop。
        """
        import asyncio
        import threading
        t = threading.Thread(
            target=asyncio.run,
            args=(self._async_llm_initialize(task_description, iteration),),
            daemon=True,
        )
        t.start()
        t.join()

    async def _async_llm_initialize(self, task_description: str, iteration: int) -> None:
        """异步 LLM 提取实现。"""
        import litellm
        import os

        model = self.config.llm_model or "glm-4"
        api_url = self.config.llm_api_url
        api_key = self.config.llm_api_key or os.environ.get("GLM_API_KEY", "")

        # 路由判断（参考 nanobot.py 的逻辑）
        if api_url and "anthropic" in api_url.lower():
            if not model.startswith("anthropic/"):
                model = f"anthropic/{model}"

        system_prompt = """你是一个结构化信息提取助手。
从任务指令中提取以下结构化信息，返回 JSON 格式（只返回 JSON，不要其他内容）：

{
  "constraints": ["约束1", "约束2", ...],   # 必须满足的条件，每条尽量在20字以内
  "pending_subgoals": ["子目标1", "子目标2", ...],  # 待完成的步骤/动作，每条尽量在30字以内
  "entities": [{"name": "实体名", "type": "类型", "value": "值"}]  # 关键实体（如项目名、地点、金额）
}

约束示例: "exactly 10 lines", "must mention project name", "must be under 100KB"
子目标示例: "Read the report.txt file", "Write summary.txt with 10 lines", "Create key_points.txt with 4 sections"
实体示例: {"name": "budget", "type": "money", "value": "$47.3 million"}, {"name": "project", "type": "name", "value": "Meridian Solar"}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"任务指令:\n{task_description[:3000]}"},
        ]

        # Anthropic-compatible APIs (GLM, MiniMax) 需要 anthropic/ 前缀
        if not model.startswith("anthropic/") and not model.startswith("openai/"):
            model = f"anthropic/{model}"

        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.1,
        }
        if api_url:
            kwargs["api_base"] = api_url   # litellm 用 api_base，不是 api_url
        if api_key:
            kwargs["api_key"] = api_key

        try:
            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content if response.choices else ""
        except Exception as e:
            # litellm fallback: try direct call
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=api_key, base_url=api_url)
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1024,
                temperature=0.1,
            )
            content = resp.choices[0].message.content if resp.choices else ""

        # 解析 JSON
        import json
        try:
            # 尝试从 content 中提取 JSON
            json_str = content.strip()
            # 处理 markdown code block
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
            data = json.loads(json_str.strip())
        except (json.JSONDecodeError, Exception) as e:
            self._logger.warning(f"[StructuredMemoryStore] Failed to parse LLM response as JSON: {content[:200]}")
            raise

        # 填充 constraints
        for c in data.get("constraints", []):
            if isinstance(c, str) and c.strip():
                self._add_or_update_slot(
                    slot_type="constraints",
                    content=c.strip()[:200],
                    source_event="llm_task",
                    iteration=iteration,
                )

        # 填充 pending_subgoals
        for s in data.get("pending_subgoals", []):
            if isinstance(s, str) and s.strip():
                self._add_or_update_slot(
                    slot_type="pending_subgoals",
                    content=s.strip()[:200],
                    source_event="llm_task",
                    iteration=iteration,
                )

        # 填充 derived_facts (entities)
        for ent in data.get("entities", []):
            if isinstance(ent, dict) and ent.get("name"):
                fact = f"{ent.get('name')}: {ent.get('value', '')} ({ent.get('type', '')})"
                self._add_or_update_slot(
                    slot_type="derived_facts",
                    content=fact[:200],
                    source_event="llm_task",
                    derived=True,
                    iteration=iteration,
                )

        total = len(data.get("constraints", [])) + len(data.get("pending_subgoals", [])) + len(data.get("entities", []))
        self._logger.info(f"[StructuredMemoryStore] LLM initialized: {total} items")

    def _regex_initialize_from_task(self, task_description: str, iteration: int) -> None:
        """回退方案：用正则从任务指令中提取结构化状态。"""
        constraints = extract_constraints_from_instruction(task_description)
        subgoals = extract_pending_subgoals_from_instruction(task_description)

        for c in constraints:
            self._add_or_update_slot(
                slot_type="constraints",
                content=c,
                source_event="task_instruction",
                iteration=iteration,
            )
            self._logger.debug(f"[StructuredMemoryStore] constraint: {c[:80]}")

        for s in subgoals:
            self._add_or_update_slot(
                slot_type="pending_subgoals",
                content=s,
                source_event="task_instruction",
                iteration=iteration,
            )
            self._logger.debug(f"[StructuredMemoryStore] subgoal: {s[:80]}")

        if constraints or subgoals:
            self._logger.info(
                f"[StructuredMemoryStore] Regex fallback: "
                f"{len(constraints)} constraints, {len(subgoals)} subgoals"
            )

    def increment_iteration(self) -> None:
        """Increment the iteration counter."""
        self._iteration += 1

    def update_state(
        self,
        tool_name: str,
        tool_args: dict,
        tool_result: str,
    ) -> List[SlotUpdateResult]:
        """Buffer a tool result for later LLM processing (fast path).

        工具结果只暂存到 buffer，不直接解析。
        实际的 state update 由 flush_buffer() 在适当的时机调用 LLM 完成。

        Args:
            tool_name: Name of the tool that was executed
            tool_args: Tool arguments
            tool_result: Tool execution result string

        Returns:
            Empty list (actual updates are done by flush_buffer)
        """
        if not self._enabled:
            return []

        # 保留 artifact_paths 的同步解析（轻量、可靠）
        results = []
        if self.config.track_artifacts:
            artifact_paths = parse_tool_result_for_artifacts(
                tool_name, tool_args, tool_result
            )
            for path in artifact_paths:
                result = self._add_or_update_slot(
                    slot_type="artifact_paths",
                    content=path,
                    source_event=tool_name,
                    iteration=self._iteration,
                )
                results.append(result)

        # 工具结果压入 buffer，等 flush_buffer() 做 LLM 解析
        self._tool_result_buffer.append({
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result": tool_result[:20000] if tool_result else "",  # 截断避免过大
            "iteration": self._iteration,
        })

        # 超过 buffer 上限则丢弃最老的
        max_buf = self.config.llm_update_buffer_size
        if len(self._tool_result_buffer) > max_buf:
            self._tool_result_buffer.pop(0)

        return results

    def should_llm_update(self) -> bool:
        """判断是否应该触发一次 LLM 状态更新。

        触发条件（满足任一）：
          1. 距离上次 LLM 更新已过 llm_update_interval 个 iteration
          2. buffer 已满（达到 llm_update_buffer_size）
        """
        if not self._enabled or not self.config.llm_model:
            return False
        if not self._tool_result_buffer:
            return False
        interval = self.config.llm_update_interval
        if interval <= 0:
            return False
        since_last = self._iteration - self._last_llm_update_iteration
        buffer_full = len(self._tool_result_buffer) >= self.config.llm_update_buffer_size
        return since_last >= interval or buffer_full

    def flush_buffer(self) -> List[SlotUpdateResult]:
        """对 buffer 中的工具结果做 LLM 批量解析并更新 slots。

        调用频率由 should_llm_update() 控制。

        Returns:
            List of SlotUpdateResult from the LLM-driven updates
        """
        if not self._enabled or not self._tool_result_buffer:
            return []

        results = []

        # 如果没有 LLM 配置，回退到正则解析（批量）
        if not self.config.llm_model:
            results = self._regex_batch_update()
            self._tool_result_buffer.clear()
            self._last_llm_update_iteration = self._iteration
            return results

        # LLM 解析
        try:
            results = self._llm_batch_update_sync()
        except Exception as e:
            self._logger.warning(f"[StructuredMemoryStore] LLM update failed: {e}, falling back to regex")
            results = self._regex_batch_update()

        self._tool_result_buffer.clear()
        self._last_llm_update_iteration = self._iteration
        return results

    def _llm_batch_update_sync(self) -> List[SlotUpdateResult]:
        """同步封装：在子线程中运行异步 LLM 批量更新。"""
        import asyncio
        import threading
        result_holder = []

        def run_in_thread():
            result_holder.append(asyncio.run(self._async_llm_batch_update()))

        t = threading.Thread(target=run_in_thread, daemon=True)
        t.start()
        t.join()
        return result_holder[0] if result_holder else []

    async def _async_llm_batch_update(self) -> List[SlotUpdateResult]:
        """异步 LLM 批量更新：把所有 buffer 中的工具结果一起发给 LLM 解析。"""
        import litellm
        import os

        model = self.config.llm_model
        api_url = self.config.llm_api_url
        api_key = self.config.llm_api_key or os.environ.get("GLM_API_KEY", "")

        # 获取当前状态上下文（供 LLM 参考）
        current_constraints = [s.content for s in self.get_active_slots("constraints")]
        current_subgoals = [s.content for s in self.get_active_slots("pending_subgoals")]

        # 构建工具结果摘要
        tool_summary_lines = []
        for entry in self._tool_result_buffer[-8:]:  # 最多取最近 8 条
            tn = entry["tool_name"]
            tr = entry["tool_result"]
            # 截断每条结果
            tr_short = tr[:5000].replace("\n", "\\n") if tr else "(empty)"
            tool_summary_lines.append(f"[{tn}] {tr_short}")

        system_prompt = """你是一个结构化状态跟踪助手。
根据 Agent 最近的工具执行结果，更新结构化状态。

当前已有的状态（供你参考，避免重复添加）:
- constraints: {constraints}
- pending_subgoals: {subgoals}

工具执行摘要:
{tool_summary}

请以 JSON 格式返回状态更新:

{{
  "new_constraints": ["新发现的约束1", ...],     // 新增的约束
  "new_derived_facts": ["新推导的事实1", ...],  // 从工具结果中提炼的关键信息
  "completed_subgoals": ["完成的 subgoal 描述", ...],  // 哪些 pending subgoal 被完成了
  "state_summary": "一句话总结当前任务进展状态"       // 用于注入 agent 上下文的简短摘要
}}

规则:
- 只返回真正新的信息，不要重复已有状态
- derived_facts 应是决策相关的具体信息（如"预算: $47.3M" 而非 "read_file 返回了文件内容"）
- completed_subgoals 描述要与 pending_subgoals 中的条目对应
- 只返回 JSON，不要其他内容""".format(
            constraints=current_constraints[:5],
            subgoals=current_subgoals[:5],
            tool_summary="\n".join(tool_summary_lines),
        )

        # Anthropic-compatible APIs (GLM, MiniMax) 需要 anthropic/ 前缀
        if not model.startswith("anthropic/") and not model.startswith("openai/"):
            model = f"anthropic/{model}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "请分析以上工具结果，更新结构化状态。"},
        ]

        kwargs = {"model": model, "messages": messages, "max_tokens": 1024, "temperature": 0.1}
        if api_url:
            kwargs["api_base"] = api_url   # litellm 用 api_base，不是 api_url
        if api_key:
            kwargs["api_key"] = api_key

        try:
            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content if response.choices else ""
        except Exception:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=api_key, base_url=api_url)
            resp = await client.chat.completions.create(
                model=model, messages=messages, max_tokens=1024, temperature=0.1
            )
            content = resp.choices[0].message.content if resp.choices else ""

        # 解析 JSON
        import json
        json_str = content.strip()
        if json_str.startswith("```"):
            parts = json_str.split("```")
            for i, p in enumerate(parts):
                if i % 2 == 1:  # 代码块内容
                    if p.startswith("json"):
                        p = p[4:]
                    try:
                        data = json.loads(p.strip())
                        break
                    except json.JSONDecodeError:
                        continue
            else:
                raise ValueError(f"No valid JSON found in: {content[:200]}")
        else:
            data = json.loads(json_str)

        results = []

        # 新增 constraints
        for c in data.get("new_constraints", []):
            if isinstance(c, str) and c.strip():
                r = self._add_or_update_slot(
                    slot_type="constraints",
                    content=c.strip()[:200],
                    source_event="llm_batch",
                    iteration=self._iteration,
                )
                results.append(r)

        # 新增 derived_facts
        for f in data.get("new_derived_facts", []):
            if isinstance(f, str) and f.strip():
                r = self._add_or_update_slot(
                    slot_type="derived_facts",
                    content=f.strip()[:200],
                    source_event="llm_batch",
                    derived=True,
                    iteration=self._iteration,
                )
                results.append(r)

        # 标记完成的 subgoals
        for completed_desc in data.get("completed_subgoals", []):
            if not isinstance(completed_desc, str):
                continue
            desc_lower = completed_desc.lower()
            for slot in self.get_active_slots("pending_subgoals"):
                if any(kw in slot.content.lower() for kw in desc_lower.split()[:3]):
                    self._mark_complete(slot.slot_id)
                    results.append(SlotUpdateResult(updated=True, slot_id=slot.slot_id, action="completed"))
                    break

        self._logger.info(
            f"[StructuredMemoryStore] LLM batch update: "
            f"{len(data.get('new_constraints', []))} new constraints, "
            f"{len(data.get('new_derived_facts', []))} new facts, "
            f"{len(data.get('completed_subgoals', []))} completed"
        )
        return results

    def _regex_batch_update(self) -> List[SlotUpdateResult]:
        """回退方案：对 buffer 中的工具结果批量做正则解析。"""
        results = []
        for entry in self._tool_result_buffer:
            constraints = parse_tool_result_for_constraints(
                entry["tool_name"], entry["tool_args"], entry["tool_result"], entry["iteration"]
            )
            for c_text, source in constraints:
                r = self._add_or_update_slot(
                    slot_type="constraints", content=c_text,
                    source_event=source, iteration=entry["iteration"],
                )
                results.append(r)

            facts = parse_tool_result_for_derived_facts(
                entry["tool_name"], entry["tool_args"], entry["tool_result"], entry["iteration"]
            )
            for f_text, source in facts:
                r = self._add_or_update_slot(
                    slot_type="derived_facts", content=f_text,
                    source_event=source, derived=True, iteration=entry["iteration"],
                )
                results.append(r)

            # subgoal completion detection
            if self.config.detect_subgoal_completion:
                completed = detect_subgoal_completion(
                    entry["tool_name"], entry["tool_args"], entry["tool_result"],
                    self._slots["pending_subgoals"],
                )
                for sid in completed:
                    self._mark_complete(sid)
                    results.append(SlotUpdateResult(updated=True, slot_id=sid, action="completed"))

        self._logger.info(f"[StructuredMemoryStore] Regex batch update: {len(results)} results")
        return results

    def add_pending_subgoal(self, content: str, iteration: int) -> StateSlot:
        """Add a new pending subgoal to track."""
        return self._add_or_update_slot(
            slot_type="pending_subgoals",
            content=content,
            source_event="agent",
            iteration=iteration,
        )

    def _add_or_update_slot(
        self,
        slot_type: str,
        content: str,
        source_event: str,
        derived: bool = False,
        iteration: int = 0,
    ) -> SlotUpdateResult:
        """Add or update a slot of the given type."""
        max_items = getattr(self.config, f"max_{slot_type}", 20)
        slots = self._slots[slot_type]

        # Check if similar content already exists (dedup)
        existing_id = self._find_similar_slot(slot_type, content)
        if existing_id:
            slot = self._slot_index[existing_id]
            slot.touch(iteration, source_event)
            return SlotUpdateResult(updated=True, slot_id=slot.slot_id, action="updated")

        # Create new slot
        slot_id = str(uuid.uuid4())[:8]
        slot = StateSlot(
            slot_id=slot_id,
            slot_type=slot_type,
            content=content[:500],
            derived=derived,
            iteration_created=iteration,
            iteration_updated=iteration,
            source_event=source_event,
            version=1,
            active=True,
        )

        # Enforce capacity
        if len(slots) >= max_items:
            if self.config.keep_most_recent:
                old_slot = slots.pop(0)
                del self._slot_index[old_slot.slot_id]
            else:
                old_slot = min(slots, key=lambda s: s.iteration_updated)
                slots.remove(old_slot)
                del self._slot_index[old_slot.slot_id]

        slots.append(slot)
        self._slot_index[slot_id] = slot

        self._logger.debug(
            f"[StructuredMemoryStore] Created slot: {slot_type}/{slot_id}"
        )
        return SlotUpdateResult(updated=True, slot_id=slot_id, action="created")

    def _find_similar_slot(self, slot_type: str, content: str) -> Optional[str]:
        """Find an existing slot with similar content (for dedup)."""
        slots = self._slots.get(slot_type, [])
        content_lower = content.lower()

        for slot in slots:
            if not slot.active:
                continue
            if content_lower in slot.content.lower():
                return slot.slot_id
        return None

    def _mark_complete(self, slot_id: str) -> None:
        """Mark a pending subgoal as complete."""
        slot = self._slot_index.get(slot_id)
        if slot and slot.active:
            slot.active = False
            self._logger.debug(
                f"[StructuredMemoryStore] Marked complete: {slot_id}"
            )

    def get_active_slots(self, slot_type: str) -> List[StateSlot]:
        """Get all active slots of a given type."""
        return [s for s in self._slots.get(slot_type, []) if s.active]

    def retrieve(
        self, slot_types: Optional[List[str]] = None
    ) -> Dict[str, List[StateSlot]]:
        """Retrieve active slots by type.

        Args:
            slot_types: List of slot types to retrieve. If None, all types.

        Returns:
            Dict mapping slot_type -> list of active StateSlots
        """
        if slot_types is None:
            slot_types = list(self._slots.keys())

        return {st: self.get_active_slots(st) for st in slot_types}

    def format_for_prompt(
        self, slot_types: Optional[List[str]] = None
    ) -> str:
        """Format structured state as a readable context string.

        Args:
            slot_types: Which slot types to include. If None, all.

        Returns:
            Formatted state context string for prompt injection
        """
        if slot_types is None:
            slot_types = [
                "constraints",
                "derived_facts",
                "pending_subgoals",
                "artifact_paths",
            ]

        parts = ["[STRUCTURED STATE — Decision-Relevant Representations]"]

        for slot_type in slot_types:
            slots = self.get_active_slots(slot_type)
            if not slots:
                continue

            type_label = slot_type.replace("_", " ").title()
            parts.append(f"\n## {type_label}")

            for i, slot in enumerate(slots, 1):
                content = slot.content
                if len(content) > 200:
                    content = content[:200] + "..."

                derived_tag = " [DERIVED]" if slot.derived else ""
                parts.append(f"{i}. {content}{derived_tag}")

        parts.append("\n[END Structured State]")
        return "\n".join(parts)

    def get_summary(self) -> dict:
        """Get structured memory summary for logging/transcript."""
        return {
            "enabled": self._enabled,
            "iteration": self._iteration,
            "slot_counts": {st: len(self.get_active_slots(st)) for st in self._slots.keys()},
            "update_policy": self.config.update_policy.value,
        }

    @property
    def is_enabled(self) -> bool:
        return self._enabled
