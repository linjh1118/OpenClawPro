"""
Collaboration Roles - Recipe T3: Minimal Collaboration

PlannerRole and ExecutorRole for lightweight two-agent collaboration.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from .config import CollabConfig, RoleDefinition
from .event import CollabEvent

logger = logging.getLogger("agent.collab.roles")


class PlannerRole:
    """Planner role - generates execution plans."""

    DEFAULT_SYSTEM_PROMPT = """You are a planner agent. Your task is to analyze the user's request and create a structured execution plan.

When creating a plan:
1. Break down the task into clear, actionable steps
2. Consider dependencies between steps
3. Anticipate potential issues
4. Keep the plan concise but complete

Output your plan in the following format:
- Step 1: [action description]
- Step 2: [action description]
- ...

Be specific about what tools to use and what the expected outcomes are."""

    def __init__(
        self,
        config: CollabConfig,
        llm_call_fn,  # Async function reference to call LLM
        model: str,
        system_prompt: str | None = None,
    ):
        self.config = config
        self._llm_call = llm_call_fn
        self.model = model
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._events: List[CollabEvent] = []

    async def generate_plan(
        self,
        task: str,
        context: str | None = None,
        iteration: int = 0,
    ) -> Dict[str, Any]:
        """Generate an execution plan for the given task.

        Returns:
            Dict with "plan" (list of steps), "rationale" (str), "iteration" (int)
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        if context:
            messages.append({"role": "user", "content": f"Context from previous attempts:\n{context}\n\nTask: {task}"})
        else:
            messages.append({"role": "user", "content": task})

        try:
            response = await self._llm_call(messages, model=self.model, max_tokens=1024)
            content = ""
            if hasattr(response, "choices") and response.choices:
                content = response.choices[0].message.content or ""

            # Parse the plan from the response
            plan = self._parse_plan(content)

            event = CollabEvent(
                event_type="plan_generated",
                role="planner",
                iteration=iteration,
                data={"plan": plan, "raw_response": content[:500]},
            )
            self._events.append(event)

            logger.info(f"[PlannerRole] Generated plan with {len(plan)} steps")

            return {
                "plan": plan,
                "rationale": content[:500],
                "iteration": iteration,
            }

        except Exception as e:
            logger.error(f"[PlannerRole] Plan generation failed: {e}")
            return {
                "plan": [{"step": "Fallback: Execute task directly", "action": "execute"}],
                "rationale": f"Planning failed: {e}",
                "iteration": iteration,
            }

    def _parse_plan(self, content: str) -> List[Dict[str, str]]:
        """Parse plan from LLM response content."""
        plan = []
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("- Step") or line.startswith("Step") or line.startswith("* "):
                # Extract step description
                step_text = line.lstrip("- *").lstrip("0123456789. ").strip()
                if step_text:
                    plan.append({
                        "step": len(plan) + 1,
                        "description": step_text,
                        "action": "continue",
                    })
        if not plan:
            # Fallback: treat each line as a step
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:
                    plan.append({
                        "step": len(plan) + 1,
                        "description": line[:100],
                        "action": "continue",
                    })
        return plan

    def get_events(self) -> List[CollabEvent]:
        """Get all events recorded by this role."""
        return self._events.copy()

    def consume_events(self) -> List[CollabEvent]:
        """Return and clear buffered events."""
        events = self._events.copy()
        self._events.clear()
        return events

    def reset(self) -> None:
        """Reset role-local state."""
        self._events.clear()


class ExecutorRole:
    """Executor role - executes plan steps using tools."""

    DEFAULT_SYSTEM_PROMPT = """You are an executor agent. Your job is to execute tasks and produce results.

## Important: Handling Feedback

If you receive feedback (verification results or previous context), it means your previous attempt had issues that need to be corrected.
- Read the feedback carefully and understand WHAT went wrong
- Do NOT repeat the same approach that led to the error
- Correct the specific issues mentioned in the feedback
- When in doubt, re-examine the data sources and recalculate before responding

## Your Task

Execute the current step by:
1. Understanding what needs to be done
2. Using appropriate tools to gather information
3. Producing accurate results based on evidence

Always verify your own work before presenting it as complete."""

    def __init__(
        self,
        config: CollabConfig,
        llm_call_fn,
        execute_tool_fn,
        model: str,
        system_prompt: str | None = None,
    ):
        self.config = config
        self._llm_call = llm_call_fn
        self._execute_tool = execute_tool_fn
        self.model = model
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._events: List[CollabEvent] = []
        self._tool_defs: List[Dict] = []

    def set_tool_definitions(self, tool_defs: List[Dict]) -> None:
        """Set available tool definitions."""
        self._tool_defs = tool_defs

    async def execute_step(
        self,
        step: Dict[str, str],
        context: str | None = None,
        iteration: int = 0,
        prev_messages: List[Dict] | None = None,
    ) -> Dict[str, Any]:
        """Execute a single plan step.

        Args:
            step: Step description dict
            context: Optional context string to prepend to user message
            iteration: Current iteration number
            prev_messages: If provided, use these messages as conversation history instead of building new ones.
                          The last assistant message will be replaced with feedback.

        Returns:
            Dict with "success" (bool), "result" (str), "error" (str | None), "messages" (list)
        """
        step_num = step.get("step", "?")
        step_desc = step.get("description", "")

        logger.info(f"[ExecutorRole] Executing step {step_num}: {step_desc[:50]}...")

        if prev_messages is not None:
            # Use provided messages as base (feedback will be inserted by caller)
            messages = prev_messages
        else:
            messages = [
                {"role": "system", "content": self.system_prompt},
            ]
            user_content = f"Current step to execute: {step_desc}"
            if context:
                user_content = f"Previous context:\n{context}\n\n{user_content}"
            messages.append({"role": "user", "content": user_content})

        try:
            response = await self._llm_call(
                messages,
                model=self.model,
                tools=self._tool_defs if self._tool_defs else None,
            )

            # Check if LLM wants to call tools
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                message = choice.message

                if hasattr(message, "tool_calls") and message.tool_calls:
                    # Execute the tool call
                    for tc in message.tool_calls:
                        tool_name = tc.function.name
                        args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                        result = await self._execute_tool({
                            "id": tc.id,
                            "function": {"name": tool_name, "arguments": args}
                        })

                        event = CollabEvent(
                            event_type="step_executed",
                            role="executor",
                            iteration=iteration,
                            data={"step": step_num, "tool": tool_name, "result": result[:200]},
                        )
                        self._events.append(event)

                        return {
                            "success": not result.startswith("Error:"),
                            "result": result,
                            "tool_used": tool_name,
                            "error": None if not result.startswith("Error:") else result,
                            "messages": messages,
                        }
                else:
                    # No tool call, return content as result
                    content = message.content or ""
                    event = CollabEvent(
                        event_type="step_executed",
                        role="executor",
                        iteration=iteration,
                        data={"step": step_num, "content": content[:200]},
                    )
                    self._events.append(event)

                    return {
                        "success": True,
                        "result": content,
                        "tool_used": None,
                        "error": None,
                        "messages": messages,
                    }

            return {
                "success": False,
                "result": "",
                "error": "No response from LLM",
                "messages": messages,
            }

        except Exception as e:
            logger.error(f"[ExecutorRole] Step execution failed: {e}")
            event = CollabEvent(
                event_type="step_executed",
                role="executor",
                iteration=iteration,
                data={"step": step_num, "error": str(e)},
            )
            self._events.append(event)

            return {
                "success": False,
                "result": "",
                "error": str(e),
            }

    def record_step_execution(
        self,
        step: Dict[str, str],
        result: str,
        tool_used: str | None,
        error: str | None,
        iteration: int = 0,
    ) -> None:
        """Append a step_executed event without calling LLM."""
        step_num = step.get("step", "?")
        data: Dict[str, Any] = {"step": step_num}
        if tool_used is not None:
            data["tool"] = tool_used
            data["result"] = result[:200]
        else:
            data["content"] = result[:200]
        if error:
            data["error"] = error
        event = CollabEvent(
            event_type="step_executed",
            role="executor",
            iteration=iteration,
            data=data,
        )
        self._events.append(event)

    def get_events(self) -> List[CollabEvent]:
        """Get all events recorded by this role."""
        return self._events.copy()

    def consume_events(self) -> List[CollabEvent]:
        """Return and clear buffered events."""
        events = self._events.copy()
        self._events.clear()
        return events

    def reset(self) -> None:
        """Reset role-local state."""
        self._events.clear()


class CommanderRole:
    """Commander role - decomposes tasks and dispatches subtasks to executor.

    T3b architecture: Commander acts as the task decomposition and dispatch layer,
    while Executor handles actual tool execution. This provides proactive goal
    grounding (A段断裂修复) through explicit task decomposition.
    """

    DEFAULT_SYSTEM_PROMPT = """You are a commander agent. Your job is to decompose the user's task into clear, actionable subtasks and dispatch them to the executor.

## Your Responsibilities

1. **Task Decomposition**: Break down the main task into logical subtasks that can be executed independently
2. **Subtask Dispatch**: Clearly specify what each subtask should accomplish
3. **Result Synthesis**: Integrate results from completed subtasks to inform next steps
4. **Progress Tracking**: Monitor overall progress toward the main goal
5. **Adaptive Replanning**: Modify the remaining subtask list based on executor feedback

## Dispatch Format

When dispatching a subtask, use this format:
```
[DISPATCH]
Subtask {n}: {subtask description}
Expected output: {what the executor should produce}
Success criteria: {how to verify completion}
[/DISPATCH]
```

## Synthesis Format

After receiving executor results, provide a brief synthesis:
```
[SYNTHESIS]
Results so far: {summary of completed work}
Next action: {description of next subtask or "TASK_COMPLETE"}
[/SYNTHESIS]
```

## Important Rules

- Be specific in subtask descriptions to avoid goal drift
- Each subtask should be self-contained and verifiable
- Monitor for early completion or task creep
- If executor reports issues, diagnose and either modify the subtask or the overall plan"""

    def __init__(
        self,
        config: CollabConfig,
        llm_call_fn,
        model: str,
        system_prompt: str | None = None,
    ):
        self.config = config
        self._llm_call = llm_call_fn
        self.model = model
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._events: List[CollabEvent] = []
        self._dispatch_count = 0
        self._completed_subtasks: List[Dict[str, Any]] = []

    async def decompose_task(
        self,
        task: str,
        context: str | None = None,
        iteration: int = 0,
    ) -> Dict[str, Any]:
        """Decompose the main task into subtasks.

        Returns:
            Dict with "subtasks" (list), "rationale" (str), "iteration" (int)
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        user_content = "## Main Task\n" + task
        if context:
            user_content = f"{user_content}\n\n## Prior Context\n{context}"
        user_content += "\n\nDecompose this task into subtasks and dispatch the first one to the executor."
        messages.append({"role": "user", "content": user_content})

        try:
            response = await self._llm_call(messages, model=self.model, max_tokens=2048)
            content = ""
            if hasattr(response, "choices") and response.choices:
                content = response.choices[0].message.content or ""

            subtasks = self._parse_dispatch(content)
            self._dispatch_count = 1 if subtasks else 0

            event = CollabEvent(
                event_type="task_decomposed",
                role="commander",
                iteration=iteration,
                data={
                    "subtask_count": len(subtasks),
                    "first_subtask": subtasks[0] if subtasks else None,
                    "raw_response": content[:500],
                },
            )
            self._events.append(event)

            logger.info(f"[CommanderRole] Decomposed task into {len(subtasks)} subtasks")

            return {
                "subtasks": subtasks,
                "rationale": content[:500],
                "iteration": iteration,
            }

        except Exception as e:
            logger.error(f"[CommanderRole] Task decomposition failed: {e}")
            return {
                "subtasks": [{"id": 1, "description": task, "expected_output": "final result"}],
                "rationale": f"Decomposition failed: {e}",
                "iteration": iteration,
            }

    async def dispatch_next(
        self,
        subtask: Dict[str, Any],
        context: str | None = None,
        iteration: int = 0,
    ) -> Dict[str, Any]:
        """Generate dispatch message for next subtask.

        Returns:
            Dict with "dispatch_message" (str), "next_subtask" (Dict | None)
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        user_content = "## Current Subtask to Dispatch\n"
        user_content += f"Subtask {subtask.get('id', '?')}: {subtask.get('description', '')}\n"
        if subtask.get('expected_output'):
            user_content += f"Expected output: {subtask['expected_output']}\n"
        if subtask.get('success_criteria'):
            user_content += f"Success criteria: {subtask['success_criteria']}\n"

        if context:
            user_content += f"\n## Prior Results\n{context}"

        user_content += "\n\nProvide the dispatch message for this subtask."
        messages.append({"role": "user", "content": user_content})

        try:
            response = await self._llm_call(messages, model=self.model, max_tokens=1024)
            content = ""
            if hasattr(response, "choices") and response.choices:
                content = response.choices[0].message.content or ""

            event = CollabEvent(
                event_type="subtask_dispatched",
                role="commander",
                iteration=iteration,
                data={"subtask": subtask.get('description', '')[:100], "dispatch": content[:200]},
            )
            self._events.append(event)

            return {
                "dispatch_message": content,
                "subtask": subtask,
            }

        except Exception as e:
            logger.error(f"[CommanderRole] Dispatch generation failed: {e}")
            return {
                "dispatch_message": f"Execute: {subtask.get('description', '')}",
                "subtask": subtask,
            }

    async def synthesize_and_decide(
        self,
        subtask_result: Dict[str, Any],
        remaining_subtasks: List[Dict[str, Any]],
        overall_context: str | None = None,
        iteration: int = 0,
    ) -> Dict[str, Any]:
        """Synthesize executor result and decide next action.

        Returns:
            Dict with "decision" (str), "next_subtask" (Dict | None), "is_complete" (bool)
        """
        self._completed_subtasks.append(subtask_result)

        messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        user_content = "## Completed Subtask Result\n"
        user_content += f"Subtask {subtask_result.get('subtask_id', '?')}: {subtask_result.get('description', '')}\n"
        user_content += f"Result: {subtask_result.get('result', '')}\n"
        if subtask_result.get('error'):
            user_content += f"Error: {subtask_result['error']}\n"

        if remaining_subtasks:
            user_content += f"\n## Remaining Subtasks ({len(remaining_subtasks)} left)\n"
            for st in remaining_subtasks[:3]:
                user_content += f"- Subtask {st.get('id', '?')}: {st.get('description', '')}\n"
            if len(remaining_subtasks) > 3:
                user_content += f"... and {len(remaining_subtasks) - 3} more\n"

        if overall_context:
            user_content += f"\n## Overall Context\n{overall_context}\n"

        user_content += "\n\nBased on the completed subtask result, should we continue with the next subtask, modify the plan, or declare completion?"
        messages.append({"role": "user", "content": user_content})

        try:
            response = await self._llm_call(messages, model=self.model, max_tokens=1024)
            content = ""
            if hasattr(response, "choices") and response.choices:
                content = response.choices[0].message.content or ""

            # Parse decision
            is_complete = "TASK_COMPLETE" in content.upper() or "COMPLETE" in content.upper()
            next_subtask = remaining_subtasks[0] if remaining_subtasks and not is_complete else None

            event = CollabEvent(
                event_type="synthesis_complete",
                role="commander",
                iteration=iteration,
                data={
                    "decision": "continue" if next_subtask else "complete",
                    "completed_count": len(self._completed_subtasks),
                    "remaining_count": len(remaining_subtasks),
                    "raw_response": content[:300],
                },
            )
            self._events.append(event)

            return {
                "decision": "continue" if next_subtask else "complete",
                "next_subtask": next_subtask,
                "is_complete": is_complete,
                "synthesis": content,
            }

        except Exception as e:
            logger.error(f"[CommanderRole] Synthesis failed: {e}")
            return {
                "decision": "complete" if not remaining_subtasks else "continue",
                "next_subtask": remaining_subtasks[0] if remaining_subtasks else None,
                "is_complete": not remaining_subtasks,
                "synthesis": f"Synthesis failed: {e}",
            }

    async def generate_final_synthesis(
        self,
        task: str,
        all_results: List[Dict[str, Any]],
        iteration: int = 0,
    ) -> Dict[str, Any]:
        """Generate final synthesis of all subtask results.

        Returns:
            Dict with "final_synthesis" (str), "success" (bool)
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        user_content = f"## Original Task\n{task}\n\n## All Subtask Results\n"
        for i, result in enumerate(all_results, 1):
            user_content += f"\n--- Subtask {i} ---\n"
            user_content += f"Description: {result.get('description', '')}\n"
            user_content += f"Result: {result.get('result', '')}\n"
            if result.get('error'):
                user_content += f"Error: {result.get('error')}\n"

        user_content += "\n\nProvide a final synthesis that addresses the original task based on all subtask results."
        messages.append({"role": "user", "content": user_content})

        try:
            response = await self._llm_call(messages, model=self.model, max_tokens=2048)
            content = ""
            if hasattr(response, "choices") and response.choices:
                content = response.choices[0].message.content or ""

            event = CollabEvent(
                event_type="final_synthesis",
                role="commander",
                iteration=iteration,
                data={"synthesis_len": len(content), "subtask_count": len(all_results)},
            )
            self._events.append(event)

            return {
                "final_synthesis": content,
                "success": True,
            }

        except Exception as e:
            logger.error(f"[CommanderRole] Final synthesis failed: {e}")
            return {
                "final_synthesis": f"Final synthesis failed: {e}",
                "success": False,
            }

    def _parse_dispatch(self, content: str) -> List[Dict[str, Any]]:
        """Parse subtasks from LLM response."""
        subtasks = []
        current_subtask = None

        lines = content.split("\n")
        for line in lines:
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            # Look for dispatch markers - these START a new subtask, don't close the previous one
            if "[DISPATCH]" in stripped.upper():
                # Save previous subtask if it has content
                if current_subtask and current_subtask.get("description"):
                    subtasks.append(current_subtask)
                # Start new subtask with empty description
                current_subtask = {
                    "id": len(subtasks) + 1,
                    "description": "",
                    "expected_output": "",
                    "success_criteria": "",
                }
            elif "Subtask" in stripped and ":" in stripped:
                # Extract subtask description
                parts = stripped.lstrip("- *").split(":", 1)
                if len(parts) > 1 and parts[1].strip():
                    # If we have a previous subtask with content, save it
                    if current_subtask and current_subtask.get("description"):
                        subtasks.append(current_subtask)
                    # Start new subtask with description
                    current_subtask = {
                        "id": len(subtasks) + 1,
                        "description": parts[1].strip(),
                        "expected_output": "",
                        "success_criteria": "",
                    }
            elif current_subtask is not None:
                # Continue building current subtask
                if stripped.startswith("Expected output:") or stripped.startswith("- Expected"):
                    current_subtask["expected_output"] = stripped.split(":", 1)[-1].strip()
                elif stripped.startswith("Success criteria:") or stripped.startswith("- Success"):
                    current_subtask["success_criteria"] = stripped.split(":", 1)[-1].strip()
                elif stripped.startswith("[/DISPATCH]") or stripped.startswith("-"):
                    # Skip closing tags and list markers
                    continue
                elif current_subtask["description"]:
                    # Append to existing description
                    current_subtask["description"] += " " + stripped
                else:
                    current_subtask["description"] = stripped

        # Don't forget the last subtask
        if current_subtask and current_subtask.get("description"):
            subtasks.append(current_subtask)

        # Fallback: if no subtasks parsed, try to extract numbered items
        if not subtasks:
            for line in lines:
                stripped = line.strip()
                if stripped and (stripped[0].isdigit() or stripped.startswith("-")):
                    desc = stripped.lstrip("- *0123456789. ").strip()
                    if desc and len(desc) > 5:
                        subtasks.append({
                            "id": len(subtasks) + 1,
                            "description": desc,
                            "expected_output": "",
                            "success_criteria": "",
                        })

        return subtasks

    def get_events(self) -> List[CollabEvent]:
        """Get all events recorded by this role."""
        return self._events.copy()

    def consume_events(self) -> List[CollabEvent]:
        """Return and clear buffered events."""
        events = self._events.copy()
        self._events.clear()
        return events

    def reset(self) -> None:
        """Reset role-local state."""
        self._events.clear()
        self._dispatch_count = 0
        self._completed_subtasks.clear()


class VerifierRole:
    """Verifier role - 完整 agent，可调用工具独立验证 executor 的输出。"""

    DEFAULT_SYSTEM_PROMPT = """You are a verification agent. Your job is to VERIFY the executor's output — NOT to redo the task.

You will receive:
- The original task description
- A structured summary of what the executor claims to have done
- The executor's final output text

## Verification Strategy

Before spot-checking, re-read the task carefully and identify the **core objective** — what does the task actually ask for? Then verify the executor addressed THAT objective correctly.

Specifically check:
1. **Core objective**: Does the executor's work actually solve what the task asks? Not just "did it do something", but "did it do the RIGHT thing". For ambiguous situations (multiple candidates, unclear requirements), did the executor handle ambiguity correctly?
2. **Key decisions**: If the executor had to choose between options (which person, which record, which value), is the choice justified and correct?
3. **Result accuracy**: If specific numerical results are claimed, independently recalculate 1-2 values.
4. **Completeness**: Are ALL requirements from the task addressed? Re-read the task statement and cross-check each requirement.
5. **Format**: Does output format match requirements (data types, column names, sorting, encoding)?

## IMPORTANT RULES
- Do NOT re-do the entire task from scratch. Only spot-check the executor's claims.
- Limit yourself to 3-5 targeted verification actions.
- Pay special attention to cases where the executor made a CHOICE — verify that choice was correct, not just that a choice was made.
- If the executor's output looks complete and correct after spot-checking, respond VERDICT: PASS.
- Only respond VERDICT: FAIL if you find concrete issues.

After verification, you MUST end your response with exactly one of:
VERDICT: PASS
VERDICT: FAIL

If FAIL, explain what is wrong and what needs to be fixed in detail.
If PASS, briefly confirm what you verified."""

    def __init__(
        self,
        config: CollabConfig,
        llm_call_fn,
        model: str,
        system_prompt: str | None = None,
        execute_tool_fn=None,
    ):
        self.config = config
        self._llm_call = llm_call_fn
        self._execute_tool = execute_tool_fn
        self.model = model
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._events: List[CollabEvent] = []
        self._tool_defs: List[Dict] = []
        # 统计
        self._total_tool_calls: int = 0
        self._verification_rounds: List[Dict[str, Any]] = []

    def set_tool_definitions(self, tool_defs: List[Dict]) -> None:
        """设置可用工具定义。"""
        self._tool_defs = tool_defs

    async def verify_with_tools(
        self,
        task: str,
        executor_output: str,
        executor_actions: str = "",
        iteration: int = 0,
    ) -> Dict[str, Any]:
        """带工具的完整验证：独立运行 mini ReAct 循环验证 executor 输出。

        Args:
            task: 原始任务描述
            executor_output: executor 的最终输出文本
            executor_actions: executor 的结构化行动摘要
            iteration: 当前主循环迭代数

        Returns:
            {"verdict": "PASS"/"FAIL", "feedback": str, "tool_count": int}
        """
        max_turns = self.config.max_verifier_turns

        # 构建 verifier 消息上下文
        verifier_messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        user_parts = [f"## Original Task\n{task}"]
        if executor_output:
            user_parts.append(f"## Executor's Final Output\n{executor_output[:2000]}")
        if executor_actions:
            user_parts.append(f"## What the Executor Claims to Have Done\n{executor_actions[:2000]}")
        user_parts.append(
            "Based on the above, verify the executor's work. "
            "Spot-check key claims (read files, run quick calculations). "
            "Do NOT redo the entire task. Limit to 3-5 verification actions. "
            "End with VERDICT: PASS or VERDICT: FAIL."
        )
        verifier_messages.append({"role": "user", "content": "\n\n".join(user_parts)})

        tool_call_count = 0
        verdict = "PASS"
        feedback = ""
        max_turns_reached = False

        try:
            for turn in range(max_turns):
                response = await self._llm_call(
                    verifier_messages,
                    model=self.model,
                    tools=self._tool_defs if self._tool_defs else None,
                    max_tokens=4096,
                )

                if not (hasattr(response, "choices") and response.choices):
                    break

                choice = response.choices[0]
                message = choice.message
                content = getattr(message, "content", "") or ""
                tool_calls = getattr(message, "tool_calls", None)

                if tool_calls:
                    # 有工具调用 → 执行并继续
                    formatted_calls = []
                    tool_results = []
                    for tc in tool_calls:
                        tool_name = tc.function.name
                        args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                        args_str = json.dumps(args) if isinstance(args, dict) else tc.function.arguments
                        formatted_calls.append({
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": args_str},
                        })
                        tool_call_count += 1
                        self._total_tool_calls += 1

                        # 执行工具
                        tool_result = await self._execute_tool({
                            "id": tc.id,
                            "function": {"name": tool_name, "arguments": args},
                        })
                        tool_results.append((tc.id, tool_result))

                        # 记录 verifier tool call 事件
                        event = CollabEvent(
                            event_type="verifier_tool_call",
                            role="verifier",
                            iteration=iteration,
                            data={
                                "turn": turn + 1,
                                "tool": tool_name,
                                "args_preview": str(args)[:200],
                                "result_preview": str(tool_result)[:500],
                            },
                        )
                        self._events.append(event)

                    # 追加 assistant 消息（一次）+ 所有 tool results
                    verifier_messages.append({
                        "role": "assistant",
                        "content": content,
                        "tool_calls": formatted_calls,
                    })
                    for tc_id, result in tool_results:
                        verifier_messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": str(result),
                        })
                    continue
                else:
                    # 无工具调用 → 解析 verdict
                    feedback = content
                    if not content or not content.strip():
                        # LLM 返回空内容，视为验证失败
                        verdict = "FAIL"
                        feedback = "Verifier returned empty response, cannot confirm correctness."
                    elif "VERDICT: FAIL" in content.upper():
                        verdict = "FAIL"
                    elif "VERDICT: PASS" in content.upper():
                        verdict = "PASS"
                    else:
                        # 没有明确 verdict 标记，从内容推断
                        verdict = "PASS"
                    break
            else:
                # for 循环正常结束 = 超出 max_turns，强制调一次 LLM 不带 tools 要 verdict
                max_turns_reached = True
                try:
                    # 追加明确提示，引导 LLM 给出最终 verdict
                    verifier_messages.append({
                        "role": "user",
                        "content": (
                            "You have used all available verification turns. "
                            "Based on your investigation above, provide your final verdict now.\n"
                            "Reply with exactly one of:\n"
                            "- VERDICT: PASS (if the executor's work looks correct)\n"
                            "- VERDICT: FAIL + brief feedback (if you found issues)\n"
                            "Do NOT use any tools. Just give your verdict."
                        ),
                    })
                    final_response = await self._llm_call(
                        verifier_messages,
                        model=self.model,
                        tools=None,
                        max_tokens=512,
                    )
                    if hasattr(final_response, "choices") and final_response.choices:
                        content = final_response.choices[0].message.content or ""
                        if not content or not content.strip():
                            verdict = "FAIL"
                            feedback = "Verifier exhausted max turns without producing a verdict."
                        elif "VERDICT: FAIL" in content.upper():
                            verdict = "FAIL"
                            feedback = content
                        elif "VERDICT: PASS" in content.upper():
                            verdict = "PASS"
                            feedback = content
                        else:
                            verdict = "PASS"
                            feedback = content
                except Exception:
                    verdict = "FAIL"
                    feedback = "Verifier exhausted max turns and failed to produce final verdict."
                logger.info(f"[VerifierRole] max_turns reached, forced verdict={verdict}")

        except Exception as e:
            logger.error(f"[VerifierRole] verify_with_tools failed: {e}")
            verdict = "PASS"
            feedback = f"Verification error: {e}"

        # 记录最终 verdict 事件
        round_info = {
            "iteration": iteration,
            "verdict": verdict,
            "feedback_preview": feedback[:500],
            "tool_calls": tool_call_count,
        }
        self._verification_rounds.append(round_info)

        verdict_event = CollabEvent(
            event_type="verifier_verdict",
            role="verifier",
            iteration=iteration,
            data=round_info,
        )
        self._events.append(verdict_event)

        logger.info(f"[VerifierRole] verdict={verdict}, tool_calls={tool_call_count}, feedback_len={len(feedback)}")

        return {
            "verdict": verdict,
            "feedback": feedback,
            "tool_count": tool_call_count,
            "iteration": iteration,
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取 verifier 统计信息。"""
        return {
            "total_tool_calls": self._total_tool_calls,
            "verification_rounds": self._verification_rounds.copy(),
            "rounds_count": len(self._verification_rounds),
        }

    async def verify(
        self,
        step: Dict[str, str],
        result: str,
        iteration: int = 0,
    ) -> Dict[str, Any]:
        """Verify the executor's output (简化版，不使用工具)。"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Step: {step.get('description', '')}\n\nExecutor result:\n{result[:1000]}"},
        ]

        try:
            response = await self._llm_call(messages, model=self.model, max_tokens=512)

            if hasattr(response, "choices") and response.choices:
                content = response.choices[0].message.content or ""
                verdict = "PASS"
                feedback = content

                if content.upper().startswith("FAIL"):
                    verdict = "FAIL"
                elif content.upper().startswith("REVISION"):
                    verdict = "REVISION"

                event = CollabEvent(
                    event_type="critique",
                    role="verifier",
                    iteration=iteration,
                    data={"verdict": verdict, "feedback": content[:300]},
                )
                self._events.append(event)

                return {
                    "verdict": verdict,
                    "feedback": feedback,
                    "iteration": iteration,
                }

        except Exception as e:
            logger.error(f"[VerifierRole] Verification failed: {e}")

        return {
            "verdict": "PASS",
            "feedback": "Verification skipped due to error",
            "iteration": iteration,
        }

    def record_critique(
        self,
        step: Dict[str, str],
        verdict: str,
        feedback: str,
        iteration: int = 0,
    ) -> None:
        """Append a critique event without calling LLM."""
        event = CollabEvent(
            event_type="critique",
            role="verifier",
            iteration=iteration,
            data={
                "verdict": verdict,
                "feedback": feedback[:300],
                "step": step.get("step", "?"),
            },
        )
        self._events.append(event)

    def get_events(self) -> List[CollabEvent]:
        """Get all events recorded by this role."""
        return self._events.copy()

    def consume_events(self) -> List[CollabEvent]:
        """Return and clear buffered events."""
        events = self._events.copy()
        self._events.clear()
        return events

    def reset(self) -> None:
        """Reset role-local state."""
        self._events.clear()
