"""
Handoff Manager - Recipe T3: Minimal Collaboration

Manages role handoffs, queues, and context passing between planner and executor.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .config import CollabConfig, HandoffPolicy
from .event import CollabEvent
from .roles import PlannerRole, ExecutorRole, VerifierRole, CommanderRole

logger = logging.getLogger("agent.collab.handoff")


class HandoffManager:
    """Manages handoffs between collaboration roles.

    Flow (planner_executor mode):
        Task → Planner generates plan → Executor executes → [Critique on error] → Planner revises → Executor re-executes

    Flow (executor_verifier mode):
        Task → Executor executes → Verifier checks → [On fail] → Executor revises → Verifier checks again
    """

    def __init__(
        self,
        config: CollabConfig,
        planner: PlannerRole,
        executor: ExecutorRole,
        verifier: VerifierRole | None = None,
        commander: CommanderRole | None = None,
    ):
        self.config = config
        self.policy = config.handoff_policy
        self.planner = planner
        self.executor = executor
        self.verifier = verifier
        self.commander = commander

        self._handoff_count = 0
        self._events: List[CollabEvent] = []
        self._current_context: Dict[str, Any] = {}

    def reset(self) -> None:
        """Reset collaboration session state."""
        self._handoff_count = 0
        self._events.clear()
        self._current_context.clear()

    def register_events(self, events: List[CollabEvent]) -> None:
        """Register external collaboration events into the session log."""
        if events:
            self._events.extend(events)

    def record_handoff(self, from_role: str, to_role: str, reason: str, iteration: int = 0, **data) -> CollabEvent:
        """Record a role handoff event."""
        event = CollabEvent(
            event_type="handoff",
            role=from_role,
            iteration=iteration,
            data={
                "from": from_role,
                "to": to_role,
                "reason": reason,
                **data,
            },
        )
        self._events.append(event)
        return event

    def should_handoff(self, trigger: str) -> bool:
        """Check if a handoff should occur based on the policy."""
        if trigger == "always":
            return True
        if trigger == "manual":
            return False
        if self.policy.trigger == "always":
            return True
        if self.policy.trigger == "manual":
            return False
        return self.policy.trigger == trigger

    def can_handoff(self) -> bool:
        """Check if more handoffs are allowed."""
        return self._handoff_count < self.config.max_handoffs

    async def execute_planner_executor(
        self,
        task: str,
        initial_context: str | None = None,
    ) -> Dict[str, Any]:
        """Execute the planner-executor collaboration flow.

        Returns:
            Dict with "success", "result", "events", "iterations"
        """
        iteration = 0
        all_events: List[CollabEvent] = []
        final_result = ""

        # Generate initial plan
        plan_result = await self.planner.generate_plan(task, context=initial_context, iteration=iteration)
        all_events.extend(self.planner.get_events())
        iteration += 1

        if not plan_result.get("plan"):
            return {
                "success": False,
                "result": "Failed to generate plan",
                "events": all_events,
                "iterations": iteration,
            }

        # Execute plan steps
        context = initial_context or ""
        current_plan = plan_result["plan"]

        for step in current_plan:
            step_result = await self.executor.execute_step(step, context=context, iteration=iteration)
            all_events.extend(self.executor.get_events())
            iteration += 1

            is_error = step_result.get("error") is not None

            # Handle critique based on frequency setting
            should_critique = (
                (self.config.critique_frequency == "every_step") or
                (self.config.critique_frequency == "on_error" and is_error)
            )

            if should_critique and self.verifier:
                verify_result = await self.verifier.verify(step, step_result.get("result", ""), iteration=iteration)
                all_events.extend(self.verifier.get_events())
                iteration += 1

                if verify_result.get("verdict") in ("FAIL", "REVISION") and self.can_handoff():
                    # Trigger revision through planner
                    self._handoff_count += 1
                    feedback = verify_result.get('feedback', '')
                    revision_context = f"""Previous step FAILED verification and must be corrected.

Error: {step_result.get('error', 'Unknown error')}

Verification Feedback:
{feedback}

Instructions:
1. Analyze the feedback above to understand what went wrong
2. Revise your plan to address the specific issues
3. Ensure the new plan correctly handles the task requirements"""
                    new_plan_result = await self.planner.generate_plan(
                        task,
                        context=revision_context,
                        iteration=iteration,
                    )
                    all_events.extend(self.planner.get_events())
                    iteration += 1

                    if new_plan_result.get("plan"):
                        current_plan = new_plan_result["plan"]

                if step_result.get("result"):
                    context += f"\nStep {step['step']}: {step_result['result']}"

            elif step_result.get("result"):
                context += f"\nStep {step['step']}: {step_result['result']}"

            final_result = step_result.get("result", final_result)

        # Check if we need a final handoff
        if self.should_handoff("on_success") and self.can_handoff():
            logger.info(f"[HandoffManager] Final handoff triggered after {self._handoff_count} handoffs")

        return {
            "success": not any(e.data.get("error") for e in all_events if e.data.get("step")),
            "result": final_result,
            "events": all_events,
            "iterations": iteration,
            "handoff_count": self._handoff_count,
        }

    async def execute_executor_verifier(
        self,
        task: str,
        initial_context: str | None = None,
    ) -> Dict[str, Any]:
        """Execute the executor-verifier collaboration flow.

        Returns:
            Dict with "success", "result", "events", "iterations"
        """
        if not self.verifier:
            raise ValueError("VerifierRole is required for executor_verifier mode")

        iteration = 0
        all_events: List[CollabEvent] = []
        context = initial_context or ""
        prev_messages: List[Dict] | None = None

        # Execute initial attempt
        step = {"step": 1, "description": task, "action": "execute"}
        step_result = await self.executor.execute_step(step, context=context, iteration=iteration)
        all_events.extend(self.executor.get_events())
        iteration += 1

        final_result = step_result.get("result", "")
        prev_messages = step_result.get("messages", None)

        # Verify result
        max_verification_attempts = 3
        for attempt in range(max_verification_attempts):
            verify_result = await self.verifier.verify(step, final_result, iteration=iteration)
            all_events.extend(self.verifier.get_events())
            iteration += 1

            if verify_result.get("verdict") == "PASS":
                break

            if self.can_handoff() and attempt < max_verification_attempts - 1:
                self._handoff_count += 1
                feedback = verify_result.get('feedback', '')

                if prev_messages:
                    # Build new messages from previous conversation
                    # Remove last assistant message(s) and replace with feedback
                    new_messages = self._replace_last_with_feedback(prev_messages, feedback)
                    step_result = await self.executor.execute_step(
                        step, context=None, iteration=iteration, prev_messages=new_messages
                    )
                else:
                    # Fallback: use context-based approach
                    revision_context = f"""{context}

## IMPORTANT: CORRECTION REQUIRED

Your previous attempt FAILED verification. Please correct the issues below before re-executing:

{feedback}

Instructions:
1. Read the feedback above carefully
2. Identify what specifically went wrong
3. Re-examine the data sources to find the correct information
4. Produce a corrected result that addresses the feedback
5. Do NOT repeat the same incorrect approach"""
                    step_result = await self.executor.execute_step(
                        step, context=revision_context, iteration=iteration
                    )

                all_events.extend(self.executor.get_events())
                iteration += 1
                final_result = step_result.get("result", final_result)
                prev_messages = step_result.get("messages", None)
            else:
                break

        return {
            "success": verify_result.get("verdict") == "PASS",
            "result": final_result,
            "events": all_events,
            "iterations": iteration,
            "handoff_count": self._handoff_count,
        }

    def _replace_last_with_feedback(
        self, messages: List[Dict], feedback: str
    ) -> List[Dict]:
        """Replace the last assistant message with a user message containing feedback.

        This preserves the conversation history (system + user + assistant + tool results)
        but redirects the model to correct its error instead of continuing the wrong path.
        """
        new_messages = []
        found_first_assistant = False

        for msg in messages:
            role = msg.get("role", "")
            # Skip the first assistant message and all subsequent messages (tool results, final assistant, etc.)
            if role == "assistant":
                if not found_first_assistant:
                    # This is the first assistant message - skip it and mark that we found it
                    found_first_assistant = True
                    continue
                else:
                    # This is a subsequent assistant (or final) message - skip it
                    continue
            # Skip tool messages that came after the assistant's tool call
            if role == "tool":
                continue
            # Keep everything else (system, user)
            new_messages.append(msg)

        # Append feedback as user message
        new_messages.append({
            "role": "user",
            "content": f"""## IMPORTANT: CORRECTION REQUIRED

Your previous attempt FAILED verification. Please correct the issues below:

{feedback}

Instructions:
1. Read the feedback above carefully
2. Identify what specifically went wrong
3. Re-examine the data sources to find the correct information
4. Produce a corrected result that addresses the feedback
5. Do NOT repeat the same incorrect approach"""
        })

        return new_messages

    async def execute_commander_executor(
        self,
        task: str,
        initial_context: str | None = None,
    ) -> Dict[str, Any]:
        """Execute the commander-executor collaboration flow (T3b).

        Commander decomposes task into subtasks and dispatches them to Executor.
        This provides proactive goal grounding (A段断裂修复) through explicit task decomposition.

        Returns:
            Dict with "success", "result", "events", "iterations"
        """
        if not self.commander:
            raise ValueError("CommanderRole is required for commander_executor mode")

        iteration = 0
        all_events: List[CollabEvent] = []
        all_results: List[Dict[str, Any]] = []

        # Step 1: Commander decomposes the task
        decompose_result = await self.commander.decompose_task(task, context=initial_context, iteration=iteration)
        all_events.extend(self.commander.get_events())
        iteration += 1

        subtasks = decompose_result.get("subtasks", [])
        if not subtasks:
            return {
                "success": False,
                "result": "Commander failed to decompose task",
                "events": all_events,
                "iterations": iteration,
            }

        # Build context from completed subtasks
        def build_context():
            if not all_results:
                return initial_context
            ctx = initial_context or ""
            for r in all_results:
                ctx += f"\n[Subtask {r.get('subtask_id', '?')} completed]: {r.get('result', '')}"
            return ctx

        # Step 2: Execute subtasks iteratively
        while subtasks and self.can_handoff():
            current_subtask = subtasks[0]

            # Commander dispatches to executor
            dispatch_result = await self.commander.dispatch_next(current_subtask, context=build_context(), iteration=iteration)
            all_events.extend(self.commander.get_events())
            iteration += 1

            dispatch_message = dispatch_result.get("dispatch_message", current_subtask.get("description", ""))

            # Record handoff from commander to executor
            self.record_handoff("commander", "executor", "subtask_dispatch", iteration, subtask_id=current_subtask.get("id"))

            # Execute the subtask
            step = {
                "step": current_subtask.get("id", 1),
                "description": dispatch_message,
                "action": "continue",
            }
            exec_result = await self.executor.execute_step(step, context=build_context(), iteration=iteration)
            all_events.extend(self.executor.get_events())
            iteration += 1

            # Record result
            subtask_record = {
                "subtask_id": current_subtask.get("id", 1),
                "description": current_subtask.get("description", ""),
                "result": exec_result.get("result", ""),
                "error": exec_result.get("error"),
                "success": exec_result.get("success", False),
            }
            all_results.append(subtask_record)

            # Record handoff from executor back to commander
            self.record_handoff("executor", "commander", "subtask_complete", iteration, subtask_id=current_subtask.get("id"))

            # Remove completed subtask
            subtasks = subtasks[1:]

            # If we have more subtasks, ask commander to synthesize and decide
            if subtasks:
                self._handoff_count += 1
                decision_result = await self.commander.synthesize_and_decide(
                    subtask_record,
                    subtasks,
                    overall_context=build_context(),
                    iteration=iteration,
                )
                all_events.extend(self.commander.get_events())
                iteration += 1

                if decision_result.get("is_complete"):
                    break

                # Update remaining subtasks if commander modified plan
                # (current implementation keeps original order)

        # Step 3: Commander generates final synthesis
        final_result = await self.commander.generate_final_synthesis(task, all_results, iteration=iteration)
        all_events.extend(self.commander.get_events())
        iteration += 1

        return {
            "success": final_result.get("success", False),
            "result": final_result.get("final_synthesis", ""),
            "events": all_events,
            "iterations": iteration,
            "handoff_count": self._handoff_count,
            "subtask_count": len(all_results),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the collaboration session."""
        event_summary = {
            "total_events": len(self._events),
            "by_type": {},
            "by_role": {},
        }
        for event in self._events:
            event_summary["by_type"][event.event_type] = event_summary["by_type"].get(event.event_type, 0) + 1
            event_summary["by_role"][event.role] = event_summary["by_role"].get(event.role, 0) + 1
        return {
            "total_handoffs": self._handoff_count,
            "max_handoffs": self.config.max_handoffs,
            "mode": self.config.mode,
            "event_count": len(self._events),
            "events": event_summary,
        }

    def get_events(self) -> List[CollabEvent]:
        """Get all events recorded."""
        return self._events.copy()

    def consume_events(self) -> List[CollabEvent]:
        """Return and clear buffered events."""
        events = self._events.copy()
        self._events.clear()
        return events
