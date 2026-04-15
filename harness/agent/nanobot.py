"""
NanoBot Agent 适配器。

封装 nanobot 核心引擎，提供统一的 Agent 执行接口。
harness/ 与 nanobot/ 同在 OpenClawPro 仓库根目录下。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# nanobot/ 与 harness/ 同级，位于 OpenClawPro 仓库根目录下
# 需要将仓库根目录加入 sys.path 以便 import nanobot 模块
_REPO_ROOT = Path(__file__).parent.parent.parent  # harness/agent/nanobot.py → OpenClawPro/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import litellm
from litellm import acompletion
from .base import AgentResult, BaseAgent
from .memory import MemoryConfig, EpisodicMemoryStore
from .control import ControlConfig, PlanFirst, ReplanTrigger, FailureReflection, PreflightCheck, RetryPolicy
from .collaboration import CollabConfig, HandoffManager, PlannerRole, ExecutorRole, VerifierRole, get_collab_summary
from .procedure import ProceduralConfig, ProceduralStore, ProceduralTrigger, ProceduralExpander, get_procedure_summary
from nanobot.bus.queue import MessageBus
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, ListDirTool, EditFileTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool


class NanoBotAgent(BaseAgent):
    """NanoBot Agent 实现 - 使用 litellm 直接调用"""
    _logger = logging.getLogger("agent.nanobot")

    def __init__(
        self,
        model: str,
        api_url: str,
        api_key: str,
        workspace: Path,
        timeout: int = 300,
        memory_config: MemoryConfig | None = None,
        control_config: ControlConfig | None = None,
        collab_config: CollabConfig | None = None,
        procedural_config: ProceduralConfig | None = None,
        **kwargs,
    ):
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.workspace = workspace
        self.timeout = timeout
        self.kwargs = kwargs
        self.session_store_dir = Path(kwargs.get("session_store_dir") or (self.workspace / ".sessions"))
        self.session_store_dir.mkdir(parents=True, exist_ok=True)
        self.system_prompt = kwargs.get("system_prompt", "")

        # 初始化 memory store
        self._memory_config = memory_config or MemoryConfig(enabled=False)
        self._memory_store = EpisodicMemoryStore(self._memory_config)

        # 初始化 control 模块
        self._control_config = control_config or ControlConfig(enabled=False)
        self._init_control_modules()

        # 初始化 collaboration 模块 (T3)
        self._collab_config = collab_config or CollabConfig(enabled=False)
        self._init_collab_modules()

        # 初始化 procedural 模块 (T4)
        self._procedural_config = procedural_config or ProceduralConfig(enabled=False)
        self._init_procedural_modules()

        # 准备工作空间
        workspace.mkdir(parents=True, exist_ok=True)

        # 配置 litellm
        litellm.drop_params = True
        litellm.suppress_debug_info = True

        # 创建工具注册表
        self._tools = ToolRegistry()
        self._register_tools()

        # 状态跟踪
        self._usage = {}
        self._thread_local = threading.local()
        self._conversation_history: List[Dict] = []

        # Skills loading (T1 baseline skills)
        self._skills_summary: Optional[str] = None
        self._load_workspace_skills(workspace)

        # 打印初始化信息（API Key 脱敏）
        api_key_masked = (self.api_key[:4] + "****" + self.api_key[-4:]) if self.api_key and len(self.api_key) > 8 else "****"
        self._logger.info("🤖 =================== NanoBotAgent Initialized ===================")
        self._logger.info("  Model: %s", self.model)
        self._logger.info("  API URL: %s", self.api_url)
        self._logger.info("  API Key: %s", api_key_masked)
        self._logger.info("  Workspace: %s", self.workspace)
        self._logger.info("  Timeout: %ds", self.timeout)
        self._logger.info("=================================================================")

    @property
    def _transcript(self) -> List[Dict]:
        """Thread-local transcript storage."""
        if not hasattr(self._thread_local, 'transcript'):
            self._thread_local.transcript = []
        return self._thread_local.transcript

    @_transcript.setter
    def _transcript(self, value: List[Dict]) -> None:
        """Thread-local transcript storage setter for reset operations."""
        self._thread_local.transcript = value

    def _session_file(self, session_id: str) -> Path:
        safe_name = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in session_id)
        return self.session_store_dir / f"{safe_name}.json"

    @staticmethod
    def _as_text_item(content: str) -> Dict[str, Any]:
        return {
            "type": "text",
            "text": content,
        }

    @staticmethod
    def _normalize_transcript_tool_params(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Add compatibility aliases expected by some benchmark graders."""
        normalized = dict(args)
        if tool_name in {"read_file", "readFile"} and "path" in normalized and "files" not in normalized:
            normalized["files"] = [normalized["path"]]
        return normalized

    def _load_session_messages(self, session_id: str | None) -> List[Dict[str, Any]]:
        if not session_id:
            return []

        session_file = self._session_file(session_id)
        if not session_file.exists():
            base_messages: List[Dict[str, Any]] = []
            # 注入 skills summary 到 system prompt 之前
            if self._skills_summary:
                base_messages.append({"role": "system", "content": self._skills_summary})
            if self.system_prompt:
                base_messages.append({"role": "system", "content": self.system_prompt})
            return base_messages

        try:
            data = json.loads(session_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                # 如果 session 已有内容但没有 skills summary，注入它
                if self._skills_summary and not any(
                    "skills" in msg.get("content", "").lower()[:100] for msg in data if msg.get("role") == "system"
                ):
                    # 在第一个 system message 前插入 skills
                    has_system = any(msg.get("role") == "system" for msg in data)
                    if has_system:
                        # 找到第一个 system message 的位置
                        for i, msg in enumerate(data):
                            if msg.get("role") == "system":
                                data.insert(i, {"role": "system", "content": self._skills_summary})
                                break
                    else:
                        data.insert(0, {"role": "system", "content": self._skills_summary})
                return data
        except Exception:
            pass

        return []

    def _save_session_messages(self, session_id: str | None, messages: List[Dict[str, Any]]) -> None:
        if not session_id:
            return

        session_file = self._session_file(session_id)
        session_file.parent.mkdir(parents=True, exist_ok=True)
        session_file.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")

    def _register_tools(self) -> None:
        """注册工具"""
        disable = self.kwargs.get("disable_safety_guard", False)
        # 注册文件系统工具
        self._tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=self.workspace, disable_safety_guard=disable))
        self._tools.register(WriteFileTool(workspace=self.workspace, allowed_dir=self.workspace, disable_safety_guard=disable))
        self._tools.register(ListDirTool(workspace=self.workspace, allowed_dir=self.workspace, disable_safety_guard=disable))
        self._tools.register(EditFileTool(workspace=self.workspace, allowed_dir=self.workspace, disable_safety_guard=disable))

        # 注册 shell 工具
        self._tools.register(ExecTool(
            working_dir=str(self.workspace),
            restrict_to_workspace=True,
            disable_safety_guard=self.kwargs.get("disable_safety_guard", False),
        ))

        # 注册 web 工具
        self._tools.register(WebSearchTool())
        self._tools.register(WebFetchTool())

        # 不注册消息工具（不需要发送到其他 channel）

    def _load_workspace_skills(self, workspace: Path) -> None:
        """从 workspace 加载 skills 并构建 summary

        Args:
            workspace: 工作空间目录
        """
        try:
            from nanobot.agent.skills import SkillsLoader

            skills_loader = SkillsLoader(workspace)
            all_skills = skills_loader.list_skills(filter_unavailable=False)

            if all_skills:
                self._skills_summary = skills_loader.build_skills_summary()
                self._logger.info(f"Loaded {len(all_skills)} skills from workspace")
            else:
                self._skills_summary = None
                self._logger.debug("No skills found in workspace")
        except Exception as e:
            self._skills_summary = None
            self._logger.debug(f"Failed to load workspace skills: {e}")

    def _build_collab_platform_guidance(self) -> str:
        """Build platform-aware guidance for collaboration sub-roles."""
        if os.name == "nt":
            return (
                "Execution environment: Windows.\n"
                "- Do not assume POSIX shell syntax or GNU utilities are available.\n"
                "- Avoid commands like `mkdir -p`, `touch`, `rm`, `mv`, `ls`, `cat`, `grep`, `sed`, and `awk` unless you have verified a Windows-safe equivalent.\n"
                "- For shell-based text search on Windows, prefer `findstr` or `powershell -Command Select-String` instead of `grep`.\n"
                "- Prefer filesystem tools for file tasks: `write_file` creates parent directories automatically, `read_file` verifies contents, `list_dir` checks structure, and `edit_file` updates existing files.\n"
                "- Create files directly in the current task workspace unless the prompt explicitly asks for another top-level directory.\n"
            )
        return (
            "Execution environment: POSIX.\n"
            "- Prefer filesystem tools when they are simpler or more reliable than shell commands.\n"
            "- Create files directly in the current task workspace unless the prompt explicitly asks for another top-level directory.\n"
        )

    def _build_planner_system_prompt(self) -> str:
        base_prompt = PlannerRole.DEFAULT_SYSTEM_PROMPT.strip()
        return (
            f"{base_prompt}\n\n"
            "Additional execution guidance:\n"
            f"{self._build_collab_platform_guidance()}"
            "- When a task mainly asks for files or directories, plan around filesystem tools first instead of shell commands.\n"
            "- When a task asks for several facts from one document or dataset, prefer a compact extraction strategy (for example a focused script or targeted searches) over many repeated full-file reads.\n"
            "- Keep paths aligned with the user request and avoid inventing wrapper directories unless requested.\n"
        )

    def _build_executor_system_prompt(self) -> str:
        return (
            "You are an executor agent. Execute the current step precisely using the available tools.\n\n"
            "Execution guidance:\n"
            f"{self._build_collab_platform_guidance()}"
            "- Prefer `write_file` for creating new files, including nested paths such as `src/main.py`.\n"
            "- Use `list_dir` to verify directory structure and `read_file` to verify file contents.\n"
            "- Use `exec` only when shell execution is genuinely necessary.\n"
            "- For document/data extraction tasks, avoid repeating the same full-file read. Use a focused command or short script to gather the needed facts, then write the requested output artifact promptly.\n"
            "- Do not create extra top-level directories unless the task explicitly requests them.\n"
        )

    def _init_control_modules(self) -> None:
        """初始化 control 模块"""
        config = self._control_config

        # PlanFirst
        self._plan_first = PlanFirst(config.plan_first, self)

        # ReplanTrigger
        self._replan_trigger = ReplanTrigger(config.replan)

        # FailureReflection
        self._failure_reflection = FailureReflection(config.reflection, self)

        # PreflightCheck
        self._preflight_check = PreflightCheck(
            enabled=config.preflight_enabled,
            check_params=config.preflight_check_params,
            check_suitability=config.preflight_check_suitability,
        )

        # RetryPolicy
        self._retry_policy = RetryPolicy(config.retry)

        self._logger.info(f"Control modules initialized: enabled={config.enabled}")

    def _init_collab_modules(self) -> None:
        """初始化 collaboration 模块 (T3)"""
        config = self._collab_config
        self._planner_role = None
        self._executor_role = None
        self._verifier_role = None
        self._handoff_manager = None
        if not config.enabled:
            return

        # Create async LLM caller wrapper
        async def llm_call_fn(messages, model=None, tools=None, max_tokens=8192):
            return await self._call_llm(messages, model=model, tools=tools, max_tokens=max_tokens)

        # Create roles
        planner_model = config.planner_model or self.model
        verifier_model = config.verifier_model or self.model

        self._planner_role = PlannerRole(
            config=config,
            llm_call_fn=llm_call_fn,
            model=planner_model,
            system_prompt=self._build_planner_system_prompt(),
        )

        self._executor_role = ExecutorRole(
            config=config,
            llm_call_fn=llm_call_fn,
            execute_tool_fn=self._execute_tool,
            model=self.model,
            system_prompt=self._build_executor_system_prompt(),
        )

        if config.mode == "executor_verifier":
            self._verifier_role = VerifierRole(
                config=config,
                llm_call_fn=llm_call_fn,
                model=verifier_model,
            )
        else:
            self._verifier_role = None

        # Create handoff manager
        self._handoff_manager = HandoffManager(
            config=config,
            planner=self._planner_role,
            executor=self._executor_role,
            verifier=self._verifier_role,
        )

        self._logger.info(f"Collaboration modules initialized: mode={config.mode}, max_handoffs={config.max_handoffs}")

    def _init_procedural_modules(self) -> None:
        """初始化 procedural 模块 (T4)

        T4a: Program Support Cards via dense retrieval
        T4b: Skill Activation Prompts
        """
        config = self._procedural_config
        if not config.enabled:
            self._procedural_store = None
            self._procedural_trigger = None
            self._procedural_expander = None
            return

        self._procedural_store = ProceduralStore(config)
        self._procedural_trigger = ProceduralTrigger(config, self._procedural_store)
        self._procedural_expander = ProceduralExpander()

        ps_cfg = config.program_support
        sa_cfg = config.skill_activation
        cards_dir = ps_cfg.cards_dir or config.cards_dir

        self._logger.info(
            f"[T4] Procedural modules initialized: "
            f"T4a_program_support={ps_cfg.enabled}, "
            f"T4b_skill_activation={sa_cfg.enabled}, "
            f"cards_dir={cards_dir}, "
            f"card_count={self._procedural_store.get_card_count()}"
        )

    async def _call_llm(
        self,
        messages: List[Dict],
        model: str | None = None,
        tools: List[Dict] | None = None,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """调用 LLM"""
        try:
            requested_model = model or self.model
            is_openrouter = self.api_url and "openrouter" in self.api_url.lower()
            is_azure_proxy = self.api_url and "azure" in self.api_url.lower()
            is_anthropic_compat = self.api_url and "anthropic" in self.api_url.lower()

            # 打印 LLM 调用信息（API Key 脱敏）
            api_key_masked = (self.api_key[:4] + "****" + self.api_key[-4:]) if self.api_key and len(self.api_key) > 8 else "****"
            self._logger.info("📞 ==================== LLM Call ====================")
            self._logger.info("  Requested Model: %s", requested_model)
            self._logger.info("  API URL: %s", self.api_url)
            self._logger.info("  API Key: %s", api_key_masked)
            self._logger.info("  Max Tokens: %d", max_tokens)
            self._logger.info("  Tools: %s", len(tools) if tools else 0)
            self._logger.info("  Is OpenRouter: %s", is_openrouter)
            self._logger.info("  Is Azure Proxy: %s", is_azure_proxy)
            self._logger.info("  Is Anthropic Compat: %s", is_anthropic_compat)
            self._logger.info("===================================================")

            if is_openrouter:
                if not requested_model.startswith("openrouter/"):
                    requested_model = f"openrouter/{requested_model}"
            elif is_anthropic_compat:
                # Anthropic-compatible APIs (e.g. MiniMax, GLM)
                if not requested_model.startswith("anthropic/"):
                    requested_model = f"anthropic/{requested_model}"

            # For OpenAI-compatible APIs, add openai/ prefix for litellm routing
            if not is_openrouter and not is_azure_proxy and not is_anthropic_compat:
                if not requested_model.startswith("openai/"):
                    requested_model = f"openai/{requested_model}"

            # Azure proxy requires AzureOpenAI client
            if is_azure_proxy:
                from openai import AzureOpenAI

                # 转换 messages 格式
                def convert_msg(msg):
                    role = msg.get("role", "user")

                    # Azure tool result 消息
                    if role == "tool":
                        return {
                            "role": "tool",
                            "tool_call_id": msg.get("tool_call_id", ""),
                            "content": msg.get("content", ""),
                        }

                    # Assistant 消息可能包含 tool_calls (在 nanobot 中是 content blocks)
                    content = msg.get("content", "")
                    tool_calls = msg.get("tool_calls", [])

                    # 如果 content 是 list，处理 tool_call blocks
                    if isinstance(content, list):
                        text_parts = []
                        extracted_tool_calls = []
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "text":
                                    text_parts.append({"type": "text", "text": block.get("text", "")})
                                elif block.get("type") == "tool_call":
                                    extracted_tool_calls.append({
                                        "id": block.get("id", ""),
                                        "type": "function",
                                        "function": {
                                            "name": block.get("name", ""),
                                            "arguments": json.dumps(block.get("params", {})) if isinstance(block.get("params"), dict) else str(block.get("params", {}))
                                        }
                                    })
                            elif isinstance(block, str):
                                text_parts.append({"type": "text", "text": block})
                        content = "\n".join(p.get("text", "") for p in text_parts) if text_parts else (content if isinstance(content, str) else "")
                        if extracted_tool_calls:
                            tool_calls = extracted_tool_calls

                    # 构建 assistant 消息
                    result = {"role": role, "content": content}
                    if tool_calls:
                        result["tool_calls"] = tool_calls
                    return result

                azure_messages = [convert_msg(m) for m in messages]

                def call_sync():
                    client = AzureOpenAI(
                        api_key=self.api_key,
                        api_version="2024-02-01",
                        azure_endpoint=self.api_url.rstrip("/"),
                    )
                    # 转换 tools 格式
                    azure_tools = None
                    if tools:
                        azure_tools = [{"type": "function", "function": t["function"]} for t in tools]

                    resp = client.chat.completions.create(
                        model=requested_model,
                        messages=azure_messages,
                        max_tokens=max_tokens,
                        temperature=0.1,
                        tools=azure_tools,
                    )
                    return resp

                response = await asyncio.get_event_loop().run_in_executor(None, call_sync)
                self._logger.debug(f"[_call_llm] Azure response received, choices: {len(response.choices) if hasattr(response, 'choices') else 0}")

                if hasattr(response, "usage") and response.usage:
                    self._usage = {
                        "input_tokens": getattr(response.usage, "prompt_tokens", 0),
                        "output_tokens": getattr(response.usage, "completion_tokens", 0),
                        "total_tokens": getattr(response.usage, "total_tokens", 0),
                    }
                return response

            # For Anthropic-compatible APIs, litellm checks ANTHROPIC_API_KEY env var
            if is_anthropic_compat and self.api_key:
                os.environ["ANTHROPIC_API_KEY"] = self.api_key

            kwargs = {
                "model": requested_model,
                "messages": messages,
                "api_key": self.api_key,
                "api_base": self.api_url,
                "max_tokens": max_tokens,
                "temperature": 0.1,
            }

            if tools:
                kwargs["tools"] = tools

            # Retry on server errors (520, 500, InternalServerError)
            max_retries = 3
            for attempt in range(max_retries + 1):
                try:
                    response = await acompletion(**kwargs)
                    self._logger.debug(f"[_call_llm] response received, choices: {len(response.choices) if hasattr(response, 'choices') else 0}")

                    # 提取 usage
                    if hasattr(response, "usage") and response.usage:
                        self._usage = {
                            "input_tokens": getattr(response.usage, "prompt_tokens", 0),
                            "output_tokens": getattr(response.usage, "completion_tokens", 0),
                            "total_tokens": getattr(response.usage, "total_tokens", 0),
                        }

                    return response
                except Exception as retry_e:
                    err_str = str(retry_e).lower()
                    is_server_error = any(kw in err_str for kw in ["520", "500", "529", "overloaded", "internalservererror", "api_error"])
                    if is_server_error and attempt < max_retries:
                        delay = 5 * (2 ** attempt)
                        self._logger.warning(f"[_call_llm] Server error (attempt {attempt+1}/{max_retries+1}), retrying in {delay}s: {retry_e}")
                        await asyncio.sleep(delay)
                        continue
                    raise retry_e

        except Exception as e:
            raise Exception(f"LLM call failed: {e}")

    async def _execute_tool(self, tool_call: Dict) -> str:
        """执行工具调用"""
        tool_name = tool_call.get("function", {}).get("name", "")
        arguments = tool_call.get("function", {}).get("arguments", "{}")

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return f"Error: Invalid JSON arguments: {arguments}"

        # 查找工具
        tool = self._tools.get(tool_name)
        if not tool:
            return f"Error: Unknown tool: {tool_name}"

        try:
            result = await tool.execute(**arguments)
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {e}"

    def _is_tool_error(self, tool_result: str) -> bool:
        """Best-effort detection for tool execution failures."""
        normalized = (tool_result or "").strip().lower()
        return (
            normalized.startswith("error:")
            or normalized.startswith("error ")
            or normalized.startswith("error executing")
        )

    def _record_collab_events(self, events: List[Dict], register_manager: bool = True) -> None:
        """Persist collaboration events to transcript and session summary."""
        if not events:
            return
        if register_manager and self._handoff_manager:
            self._handoff_manager.register_events(events)
        for event in events:
            self._transcript.append({
                "type": "collab_event",
                **event.to_dict(),
            })

    def _consume_role_events(self) -> None:
        """Flush buffered collaboration role events into the transcript."""
        if self._planner_role:
            self._record_collab_events(self._planner_role.consume_events())
        if self._executor_role:
            self._record_collab_events(self._executor_role.consume_events())
        if self._verifier_role:
            self._record_collab_events(self._verifier_role.consume_events())

    def _build_plan_system_message(self, plan: List[Dict[str, Any]], revision: bool = False) -> Dict[str, str]:
        """Format a collaboration plan as a system message."""
        heading = "## Collaborative Plan Revision" if revision else "## Collaborative Plan"
        lines = [heading]
        for step in plan:
            lines.append(f"- Step {step.get('step', '?')}: {step.get('description', '')}")
        return {"role": "system", "content": "\n" + "\n".join(lines) + "\n"}

    async def _generate_collab_plan(
        self,
        current_task: str,
        messages: List[Dict],
        iteration: int,
        revision_reason: str | None = None,
    ) -> None:
        """Generate or revise a planner_executor plan and inject it into messages."""
        if not (self._collab_config.enabled and self._handoff_manager and self._planner_role):
            return
        if self._collab_config.mode != "planner_executor":
            return
        if revision_reason and not self._handoff_manager.can_handoff():
            return

        context_parts: List[str] = []
        if revision_reason:
            context_parts.append(revision_reason)
        if self._memory_store.is_enabled and self._collab_config.handoff_policy.include_memory:
            memory_context = self._memory_store.format_for_prompt()
            if memory_context:
                context_parts.append(memory_context)

        plan_result = await self._planner_role.generate_plan(
            current_task,
            context="\n\n".join(context_parts) if context_parts else None,
            iteration=iteration,
        )
        self._consume_role_events()

        plan = plan_result.get("plan") or []
        if not plan:
            return

        if revision_reason and self._handoff_manager.can_handoff():
            self._handoff_manager._handoff_count += 1
            handoff_event = self._handoff_manager.record_handoff(
                from_role="executor",
                to_role="planner",
                reason="tool_error",
                iteration=iteration,
                detail=revision_reason[:500],
            )
            self._record_collab_events([handoff_event], register_manager=False)

        messages[:] = [
            msg for msg in messages
            if not (msg.get("role") == "system" and "## Collaborative Plan" in msg.get("content", ""))
        ]
        messages.append(self._build_plan_system_message(plan, revision=bool(revision_reason)))

    def _extract_domain_from_task(self, task: str) -> str | None:
        """Extract domain from task description for card filtering.

        T4a: Program support cards are grouped by domain. This method
        tries to detect the domain from task keywords.

        Args:
            task: The task description

        Returns:
            Domain name if detected, None otherwise
        """
        domain_keywords = {
            "medical": ["pubmed", "clinical", "rct", "randomized", "biomedical", "medical", "healthcare"],
            "software": ["code", "debug", "refactor", "test", "deploy", "repository", "github"],
            "research": ["paper", "survey", "literature", "arxiv", "citation", "reference"],
            "data": ["dataset", "analysis", "visualization", "csv", "database", "query"],
        }
        task_lower = task.lower()
        for domain, keywords in domain_keywords.items():
            if any(kw in task_lower for kw in keywords):
                return domain
        return None

    async def _run_loop(self, messages: List[Dict], max_iterations: int = 50, max_output_tokens: int = 16384) -> str:
        """运行 agent loop"""
        tool_defs = self._tools.get_definitions()
        iteration = 0
        current_task = ""
        # 从 messages 中提取 task description（用于 control 模块）
        for msg in messages:
            if msg.get("role") == "user":
                current_task = msg.get("content", "")[:500] if len(msg.get("content", "")) > 500 else msg.get("content", "")
                break

        # Control: Plan-first - 在第一次迭代前生成计划
        if self._control_config.enabled and self._plan_first.config.enabled:
            if self._plan_first.should_generate_plan("task_start"):
                plan = await self._plan_first.generate_plan(current_task)
                plan_context = plan.to_context()
                if plan_context:
                    self._transcript.append({
                        "type": "control_event",
                        "event": "plan_first",
                        "plan": plan_context,
                    })

        while iteration < max_iterations:
            iteration += 1
            self._memory_store.increment_iteration()

            # Control: 检查是否需要重规划
            if self._control_config.enabled and self._replan_trigger.config.enabled:
                replan_decision = self._replan_trigger.should_replan(iteration)
                if replan_decision.should_replan:
                    self._logger.info(f"Replan triggered at iteration {iteration}: {replan_decision.reason}")
                    self._transcript.append({
                        "type": "control_event",
                        "event": "replan_triggered",
                        "reason": replan_decision.reason,
                        "signals": [{"type": s.signal_type, "desc": s.description} for s in replan_decision.signals],
                    })
                    # 生成新计划
                    if self._plan_first.config.enabled:
                        new_plan = await self._plan_first.generate_plan(current_task, context="Previous plan failed. ")
                        plan_context = new_plan.to_context()
                        if plan_context:
                            self._transcript.append({
                                "type": "control_event",
                                "event": "new_plan",
                                "plan": plan_context,
                            })
                    self._replan_trigger.confirm_replan()

            # 检索 memory 并注入到 messages
            if self._memory_store.is_enabled:
                memory_context = self._memory_store.format_for_prompt()
                if memory_context:
                    # 在 system message 后插入 memory context
                    memory_msg = {
                        "role": "system",
                        "content": f"\n\n{memory_context}"
                    }
                    # 找到 system 消息的位置并插入
                    system_idx = None
                    for i, msg in enumerate(messages):
                        if msg.get("role") == "system":
                            system_idx = i
                    if system_idx is not None:
                        messages.insert(system_idx + 1, memory_msg)
                    else:
                        messages.insert(0, memory_msg)

            # T4a: Program Support Cards - inject via dense retrieval at task start
            if self._procedural_config.enabled and self._procedural_trigger:
                self._procedural_trigger.increment_iteration()
                domain = self._extract_domain_from_task(current_task)

                # T4b: Skill Activation Prompt - inject at task start (iteration 1)
                if (
                    iteration == 1
                    and self._procedural_config.skill_activation.enabled
                    and self._procedural_config.skill_activation.inject_at_start
                ):
                    skill_act_prompt = self._procedural_expander.format_skill_activation(
                        self._procedural_config.skill_activation,
                        tool_definitions=tool_defs,
                    )
                    skill_act_msg = {"role": "system", "content": skill_act_prompt}
                    # Find insertion point: after system, before memory
                    insert_idx = 1
                    for i, msg in enumerate(messages):
                        if msg.get("role") == "system" and "memory" in msg.get("content", "").lower():
                            insert_idx = i
                            break
                    messages.insert(insert_idx, skill_act_msg)
                    self._logger.info("[T4b] Skill activation prompt injected at task start")

                # T4a: Retrieve program support cards via dense retrieval
                if self._procedural_config.program_support.enabled:
                    # Build context from recent messages
                    context_for_trigger = ""
                    for msg in reversed(messages[-10:]):
                        if msg.get("role") == "user":
                            context_for_trigger = msg.get("content", "")[:300]
                            break

                    # Dense retrieval (BERT bi-encoder) → top-k cards
                    retrieved = self._procedural_trigger.retrieve_cards(
                        current_task,
                        context=context_for_trigger,
                        domain=domain,
                    )
                    if retrieved:
                        cards = [card for card, score, _ in retrieved]
                        top_score = retrieved[0][1] if retrieved else 0.0
                        self._logger.info(
                            f"[T4a] Retrieved {len(cards)} program support cards "
                            f"(top score: {top_score:.4f})"
                        )
                        # Format and inject
                        procedure_context = self._procedural_expander.format_multiple(cards)
                        procedure_msg = {"role": "system", "content": procedure_context}
                        # Find insertion point: after skill activation prompt if present
                        insert_idx = 1
                        for i, msg in enumerate(messages):
                            if msg.get("role") == "system" and "Skill Activation" in msg.get("content", ""):
                                insert_idx = i + 1
                                break
                            if msg.get("role") == "system" and "memory" in msg.get("content", "").lower():
                                insert_idx = i + 1
                                break
                        messages.insert(insert_idx, procedure_msg)

            # T3: Collaboration - planner_executor uses planner for initial plan and bounded revisions.
            if self._collab_config.enabled and self._handoff_manager:
                if self._executor_role:
                    self._executor_role.set_tool_definitions(tool_defs)
                if iteration == 1:
                    await self._generate_collab_plan(current_task, messages, iteration=0)

            # 调用 LLM
            try:
                response = await self._call_llm(messages, tools=tool_defs, max_tokens=max_output_tokens)
            except Exception as e:
                self._logger.error(f"[_run_loop] _call_llm exception: {type(e).__name__}: {e}")
                raise  # Re-raise to let execute() handle status correctly

            # 获取响应内容
            content = ""
            tool_calls = []

            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                message = choice.message

                # 获取文本内容
                if hasattr(message, "content") and message.content:
                    content = message.content

                # 获取工具调用
                if hasattr(message, "tool_calls") and message.tool_calls:
                    tool_calls = [
                        {"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in message.tool_calls
                    ]

            # 添加 assistant 消息到历史（兼容 OpenAI 格式）
            assistant_msg = {"role": "assistant", "content": content}
            if tool_calls:
                # 格式化工具调用（添加 type 字段）
                formatted_calls = []
                for tc in tool_calls:
                    formatted_calls.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": tc["function"]
                    })
                assistant_msg["tool_calls"] = formatted_calls
            messages.append(assistant_msg)

            # 如果没有工具调用，返回
            if not tool_calls:
                # 空 content 重试一次（某些 API 兼容层偶发空回复）
                if not content:
                    print(f"[_run_loop] Empty response at iteration {iteration}, retrying once", flush=True)
                    # 撤回刚才添加的空 assistant 消息
                    messages.pop()
                    try:
                        retry_resp = await self._call_llm(messages, tools=tool_defs, max_tokens=max_output_tokens)
                        if hasattr(retry_resp, "choices") and retry_resp.choices:
                            rmsg = retry_resp.choices[0].message
                            if hasattr(rmsg, "content") and rmsg.content:
                                content = rmsg.content
                            if hasattr(rmsg, "tool_calls") and rmsg.tool_calls:
                                tool_calls = [
                                    {"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                    for tc in rmsg.tool_calls
                                ]
                            assistant_msg["content"] = content
                            if tool_calls:
                                formatted_calls = []
                                for tc in tool_calls:
                                    formatted_calls.append({
                                        "id": tc["id"],
                                        "type": "function",
                                        "function": tc["function"]
                                    })
                                assistant_msg["tool_calls"] = formatted_calls
                            messages.append(assistant_msg)
                    except Exception as e:
                        print(f"[_run_loop] Retry failed: {e}", flush=True)
                        messages.append(assistant_msg)

                    # 重试后如果有了 tool_calls，继续下一轮；否则 fall through
                    if tool_calls:
                        print(f"[_run_loop] Retry got tool_calls at iteration {iteration}", flush=True)
                        pass  # fall through to tool execution below
                    else:
                        print(f"[_run_loop] Retry still empty at iteration {iteration}, returning", flush=True)
                        # 添加 assistant 消息到 transcript
                        content_items = []
                        if content:
                            content_items.append(self._as_text_item(content))
                        self._transcript.append({
                            "type": "message",
                            "message": {
                                "role": "assistant",
                                "content": content_items,
                            }
                        })
                        return content

                # 正常非空回复，返回
                content_items = []
                if content:
                    content_items.append(self._as_text_item(content))
                self._transcript.append({
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": content_items,
                    }
                })
                return content

            # 执行工具调用
            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name", "")

                # 解析参数
                args_str = tool_call["function"]["arguments"]
                if isinstance(args_str, str):
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        args = {"raw": args_str}
                else:
                    args = args_str
                transcript_args = self._normalize_transcript_tool_params(tool_name, args)

                # Control: Preflight check
                if self._control_config.enabled and self._preflight_check.enabled:
                    preflight_result = self._preflight_check.check_tool_call(tool_name, args, current_task)
                    if not preflight_result.passed:
                        self._logger.warning(f"Preflight check failed for {tool_name}: {preflight_result.errors}")
                        self._transcript.append({
                            "type": "control_event",
                            "event": "preflight_failed",
                            "tool": tool_name,
                            "errors": preflight_result.errors,
                            "warnings": preflight_result.warnings,
                        })
                    elif preflight_result.warnings:
                        self._transcript.append({
                            "type": "control_event",
                            "event": "preflight_warning",
                            "tool": tool_name,
                            "warnings": preflight_result.warnings,
                        })

                # Control: Retry policy
                tool_result = ""
                if self._control_config.enabled and self._retry_policy.config.enabled:
                    success, result = await self._retry_policy.execute_with_retry(
                        tool_name,
                        self._execute_tool,
                        tool_call,
                    )
                    tool_result = result if success else result
                else:
                    tool_result = await self._execute_tool(tool_call)

                is_error = self._is_tool_error(tool_result)

                # Control: 记录错误和重试信号
                if is_error:
                    error_msg = tool_result
                    self._replan_trigger.record_error(error_msg, iteration, tool_name)
                    self._replan_trigger.record_action(f"{tool_name}({args.get('path', args.get('command', ''))})")

                    if (
                        self._collab_config.enabled
                        and self._handoff_manager
                        and self._collab_config.mode == "planner_executor"
                        and self._handoff_manager.can_handoff()
                    ):
                        revision_reason = (
                            f"Tool execution failed.\n"
                            f"Tool: {tool_name}\n"
                            f"Arguments: {json.dumps(args, ensure_ascii=False)}\n"
                            f"Error: {error_msg}"
                        )
                        await self._generate_collab_plan(
                            current_task,
                            messages,
                            iteration=iteration,
                            revision_reason=revision_reason,
                        )

                    if self._failure_reflection.config.enabled:
                        self._failure_reflection.record_failure(
                            iteration=iteration,
                            tool_name=tool_name,
                            error_message=error_msg,
                            error_type="execution_error",
                            context=current_task,
                        )

                # Control: 成功后重置失败计数
                if not is_error and self._failure_reflection.config.enabled:
                    self._failure_reflection.record_success()

                # Control: Failure reflection
                if self._control_config.enabled and self._failure_reflection.config.enabled:
                    if self._failure_reflection.should_reflect():
                        reflection = await self._failure_reflection.reflect()
                        self._transcript.append({
                            "type": "control_event",
                            "event": "failure_reflection",
                            "reflection": reflection.reflection_text,
                            "root_cause": reflection.root_cause,
                            "suggested_correction": reflection.suggested_correction,
                        })

                # 写入 memory (根据 write policy)
                event_type = "error" if is_error else "tool_result"
                if self._memory_store.should_write_event(event_type, tool_result):
                    self._memory_store.write(
                        content=tool_result,
                        source=event_type,
                        source_detail=tool_name,
                        memory_type="error" if is_error else "result",
                    )
                    self._logger.debug(f"[_run_loop] Wrote {event_type} to memory: {tool_name}")

                # 添加 assistant 消息到 transcript（格式兼容 pinchbench grading）
                # grading 代码期望: content = [{"type": "toolCall", "name": "...", "params": {...}}]
                content_items = []
                if content:
                    content_items.append(self._as_text_item(content))
                content_items.append({
                    "type": "toolCall",
                    "name": tool_call["function"]["name"],
                    "params": transcript_args
                })
                self._transcript.append({
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": content_items,
                    }
                })
                # 添加 tool 结果到 transcript
                self._transcript.append({
                    "type": "message",
                    "message": {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": tool_result,
                    }
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result,
                })

                # T4b: Check for unexpected tool results and re-trigger skill activation
                if (
                    self._procedural_config.enabled
                    and self._procedural_config.skill_activation.enabled
                    and self._procedural_trigger
                ):
                    should_retrigger = self._procedural_trigger.check_unexpected_result(
                        tool_name,
                        tool_result,
                    )
                    if should_retrigger and self._procedural_trigger.is_unexpected_triggered():
                        retrigger_prompt = self._procedural_expander.format_skill_activation_retrigger(
                            tool_name=tool_name,
                            result_summary=str(tool_result)[:200],
                        )
                        retrigger_msg = {"role": "system", "content": retrigger_prompt}
                        messages.append(retrigger_msg)
                        self._logger.info(f"[T4b] Skill activation re-triggered: unexpected result from {tool_name}")
                        # Clear the triggered flag so it doesn't re-fire on the same iteration
                        self._procedural_trigger.clear_unexpected_triggered()

        return content if content else "Max iterations reached"

    def execute(self, prompt: str, session_id: str | None = None, workspace: Path | None = None, **kwargs) -> AgentResult:
        """执行单个 prompt"""
        start_time = time.time()
        error_msg = ""
        self._usage = {}
        self._transcript = []

        # 重置 memory store
        self._memory_store.reset()

        # 重置 control 模块状态
        if self._control_config.enabled:
            self._replan_trigger.reset()
            self._failure_reflection.clear()
            self._retry_policy.reset()
            self._preflight_check.clear_history()
            self._plan_first.clear()

        # 重置 collaboration 模块状态 (T3)
        if self._collab_config.enabled and self._handoff_manager:
            self._handoff_manager.reset()
            if self._planner_role:
                self._planner_role.reset()
            if self._executor_role:
                self._executor_role.reset()
            if self._verifier_role:
                self._verifier_role.reset()

        # 重置 procedural 模块状态 (T4)
        if self._procedural_config.enabled and self._procedural_trigger:
            self._procedural_trigger.reset()

        self._logger.debug(f"[execute] prompt length: {len(prompt)}, session_id: {session_id}, workspace: {workspace}")

        # 使用传入的 workspace 或默认的
        if workspace is not None:
            workspace.mkdir(parents=True, exist_ok=True)
            current_workspace = workspace
        else:
            current_workspace = self.workspace

        # 如果 workspace 发生变化，重新加载 skills 并更新 session store 路径
        if current_workspace != self.workspace:
            self._load_workspace_skills(current_workspace)
            self.workspace = current_workspace
            self.session_store_dir = self.workspace / ".sessions"
            self.session_store_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.session_store_dir = current_workspace / ".sessions"
            self.session_store_dir.mkdir(parents=True, exist_ok=True)

        # 更新工具的 workspace
        for tool in self._tools._tools.values():
            if hasattr(tool, 'workspace'):
                tool.workspace = current_workspace
            if hasattr(tool, '_workspace'):
                tool._workspace = current_workspace
            if hasattr(tool, 'working_dir'):
                tool.working_dir = str(current_workspace)

        try:
            # 构建消息，支持基于 session_id 的跨调用上下文持久化
            messages = self._load_session_messages(session_id)
            messages.append({"role": "user", "content": prompt})

            # 记录 user message 到 transcript
            self._transcript.append({
                "type": "message",
                "message": {
                    "role": "user",
                    "content": [prompt],
                }
            })

            # 使用 asyncio 运行
            max_iters = kwargs.get("max_iterations", 50)
            max_output_tokens = kwargs.get("max_output_tokens", 16384)
            content = asyncio.run(self._run_loop(messages, max_iterations=max_iters, max_output_tokens=max_output_tokens))
            self._logger.debug(f"[execute] _run_loop returned, content length: {len(content) if content else 0}, transcript entries: {len(self._transcript)}")
            self._save_session_messages(session_id, messages)

        except Exception as e:
            self._logger.error(f"[execute] Exception caught: {type(e).__name__}: {e}")
            content = ""
            error_msg = str(e)
            self._transcript = []  # 清空 transcript

        execution_time = time.time() - start_time

        # 确定状态
        status = "success"
        iteration_exhausted = False
        if content == "Max iterations reached":
            status = "max_iterations_exceeded"
            iteration_exhausted = True
        if error_msg:
            status = "error"
            if "timed out" in error_msg.lower():
                status = "timeout"

        # 记录 memory summary 到 transcript
        if self._memory_store.is_enabled and self._transcript:
            memory_summary = self._memory_store.get_summary()
            self._transcript.append({
                "type": "memory_event",
                "event": "memory_summary",
                "summary": memory_summary,
            })
            self._logger.debug(f"[execute] Memory summary: {memory_summary}")

        # 记录 control summary 到 transcript
        if self._control_config.enabled and self._transcript:
            control_summary = {
                "replan_stats": self._replan_trigger.get_replan_stats(),
                "failure_stats": self._failure_reflection.get_failure_stats(),
                "retry_stats": self._retry_policy.get_retry_stats(),
                "preflight_stats": self._preflight_check.get_check_stats(),
            }
            self._transcript.append({
                "type": "control_event",
                "event": "control_summary",
                "summary": control_summary,
            })
            self._logger.debug(f"[execute] Control summary: {control_summary}")

        # 记录 collaboration summary 到 transcript (T3)
        if self._collab_config.enabled and self._transcript:
            collab_summary = None
            if self._handoff_manager:
                collab_summary = self._handoff_manager.get_summary()
            if collab_summary is None:
                collab_summary = {"enabled": True, "mode": self._collab_config.mode}
            self._transcript.append({
                "type": "collab_event",
                "event": "collab_summary",
                "summary": collab_summary,
            })
            self._logger.debug(f"[execute] Collaboration summary: {collab_summary}")

        # 记录 procedural summary 到 transcript (T4)
        if self._procedural_config.enabled and self._transcript:
            proc_summary = None
            if self._procedural_trigger and self._procedural_expander:
                proc_summary = get_procedure_summary(
                    self._procedural_trigger.get_events(),
                    self._procedural_expander.get_events(),
                )
            if proc_summary is None:
                proc_summary = {"enabled": True, "cards_dir": self._procedural_config.cards_dir}
            self._transcript.append({
                "type": "procedural_event",
                "event": "procedural_summary",
                "summary": proc_summary,
            })
            self._logger.debug(f"[execute] Procedural summary: {proc_summary}")

        return AgentResult(
            status=status,
            content=content,
            transcript=self._transcript,
            usage=self._usage,
            workspace=str(current_workspace),
            execution_time=execution_time,
            error=error_msg,
            iteration_exhausted=iteration_exhausted,
        )

    def execute_multi(
        self,
        prompts: List[str | Dict[str, Any]],
        session_id: str | None = None,
        workspace: Path | None = None,
    ) -> List[AgentResult]:
        """执行多轮对话"""
        results = []
        current_session_id = session_id
        for index, prompt_entry in enumerate(prompts):
            prompt_text = prompt_entry
            if isinstance(prompt_entry, dict):
                prompt_text = prompt_entry.get("prompt", "")
                entry_session_id = prompt_entry.get("id") or f"turn_{index}"
                if prompt_entry.get("new_session"):
                    current_session_id = f"{session_id}_{entry_session_id}" if session_id else entry_session_id
                elif current_session_id is None:
                    current_session_id = f"{session_id}_{entry_session_id}" if session_id else entry_session_id

            if not isinstance(prompt_text, str):
                prompt_text = str(prompt_text)

            result = self.execute(prompt_text, current_session_id, workspace=workspace)
            results.append(result)

        return results

    def cleanup(self) -> None:
        """清理资源"""
        pass


# ============================================================================
# Harbor Architecture: Agent runs on host, commands execute in container
# ============================================================================

class DockerExecTool(ExecTool):
    """ExecTool that runs commands inside a Docker container via docker exec.

    This is used in the Harbor architecture where:
    - NanoBotAgent runs on the host machine (Python 3.11)
    - Commands are executed inside a container via `docker exec`
    - Workspace is mounted from host to container at the same path
    """

    def __init__(
        self,
        container_name: str,
        mount_point: str = "/workspace",
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        restrict_to_workspace: bool = False,
        path_append: str = "",
        disable_safety_guard: bool = False,
    ):
        # Parent init but we override execute()
        super().__init__(
            timeout=600,  # Override default 60s timeout for docker exec
            working_dir=working_dir,
            deny_patterns=deny_patterns,
            allow_patterns=allow_patterns,
            restrict_to_workspace=restrict_to_workspace,
            path_append=path_append,
            disable_safety_guard=disable_safety_guard,
        )
        self.container_name = container_name
        self.mount_point = mount_point

    @property
    def parameters(self) -> dict:
        """Override: hide working_dir from LLM — it's always mount_point."""
        base = super().parameters
        props = dict(base.get("properties", {}))
        props.pop("working_dir", None)
        return {**base, "properties": props}

    async def execute(
        self, command: str, working_dir: str | None = None,
        timeout: int | None = None, **kwargs,
    ) -> str:
        """Execute command inside Docker container via docker exec."""
        # Always use mount_point as working dir inside container
        # Ignore working_dir from LLM — it may be a host path that doesn't exist in container
        container_cwd = self.mount_point

        guard_error = self._guard_command(command, container_cwd)
        if guard_error:
            return guard_error

        effective_timeout = min(timeout or self.timeout, self._MAX_TIMEOUT)

        # Build docker exec command
        # Use bash -c to run the command, with -w to set working directory
        # IMPORTANT: command must be properly quoted for bash -c
        import shlex
        docker_cmd = [
            "docker", "exec",
            "-w", container_cwd,
            self.container_name,
            "bash", "-c", shlex.quote(command)
        ]

        # DEBUG: Print command
        cmd_str = " ".join(docker_cmd)

        try:
            process = await asyncio.create_subprocess_shell(
                cmd_str,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Don't set cwd on host - the command runs in container via docker exec
                # The container working directory is already set via -w flag
                env=kwargs.get("env"),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                finally:
                    if sys.platform != "win32":
                        try:
                            os.waitpid(process.pid, os.WNOHANG)
                        except (ProcessLookupError, ChildProcessError):
                            pass
                return f"Error: Command timed out after {effective_timeout} seconds"

            output_parts = []

            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))

            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\n{stderr_text}")

            output_parts.append(f"\nExit code: {process.returncode}")

            result = "\n".join(output_parts) if output_parts else "(no output)"

            # Head + tail truncation
            max_len = self._MAX_OUTPUT
            display_result = result
            if len(result) > max_len:
                half = max_len // 2
                display_result = (
                    result[:half]
                    + f"\n\n... ({len(result) - max_len:,} chars truncated) ...\n\n"
                    + result[-half:]
                )

            # DEBUG: Print result
            return result

        except Exception as e:
            return f"Error executing command: {str(e)}"


class HarborNanoBotAgent(NanoBotAgent):
    """NanoBotAgent variant for Harbor architecture.

    In Harbor architecture:
    - Agent runs on the host machine (with Python 3.11)
    - File operations happen on the host filesystem (workspace mounted to container)
    - Command execution happens inside the container via docker exec

    Use this when task containers have base images that don't include Python 3.11.
    """

    def __init__(
        self,
        container_name: str,
        mount_point: str = "/workspace",
        **kwargs,
    ):
        self.container_name = container_name
        self.mount_point = mount_point
        # Store workspace path for later mapping
        self._host_workspace = kwargs.get("workspace")
        super().__init__(**kwargs)

    def _load_workspace_skills(self, workspace: Path) -> None:
        """Load skills and map location paths to container paths.

        In Harbor architecture, SkillsLoader reads from host workspace but
        agent's read_file operates inside container. We need to replace
        host paths with container paths in the skills summary.
        """
        super()._load_workspace_skills(workspace)

        # Map host workspace path to container mount_point in skills summary
        if self._skills_summary and self._host_workspace:
            host_path = str(self._host_workspace)
            # Replace host path with container mount_point in <location> tags
            self._skills_summary = self._skills_summary.replace(
                f"<location>{host_path}",
                f"<location>{self.mount_point}"
            )

    def _register_tools(self) -> None:
        """Register tools with DockerExecTool for command execution in container.

        In Harbor architecture, all file operations happen inside the container
        via docker exec, giving the agent a consistent view of the filesystem.
        """
        from nanobot.agent.tools.docker_filesystem import (
            DockerReadFileTool,
            DockerWriteFileTool,
            DockerListDirTool,
            DockerEditFileTool,
        )

        # Register Docker filesystem tools (execute inside container via docker exec)
        self._tools.register(DockerReadFileTool(
            container_name=self.container_name,
            mount_point=self.mount_point,
            allowed_dir=self.mount_point,
        ))
        self._tools.register(DockerWriteFileTool(
            container_name=self.container_name,
            mount_point=self.mount_point,
            allowed_dir=self.mount_point,
        ))
        self._tools.register(DockerListDirTool(
            container_name=self.container_name,
            mount_point=self.mount_point,
            allowed_dir=self.mount_point,
        ))
        self._tools.register(DockerEditFileTool(
            container_name=self.container_name,
            mount_point=self.mount_point,
            allowed_dir=self.mount_point,
        ))

        # Register shell tool that executes inside container
        self._tools.register(DockerExecTool(
            container_name=self.container_name,
            mount_point=self.mount_point,
            working_dir=self.mount_point,
            restrict_to_workspace=True,
            disable_safety_guard=self.kwargs.get("disable_safety_guard", False),
        ))
