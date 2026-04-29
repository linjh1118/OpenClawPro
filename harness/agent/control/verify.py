"""
SelfVerify 模块 - 任务完成后自检输出

在 agent 返回最终结果前，注入验证 prompt 让模型自查输出是否符合任务要求。
捕获"工具成功但语义错误"类问题（如文件命名错误、字段缺失等）。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("control.verify")


@dataclass
class VerifyConfig:
    """SelfVerify 配置"""
    enabled: bool = False
    # 最大验证轮次（防止无限循环）
    max_rounds: int = 1
    # 验证 prompt 最大 token
    max_tokens: int = 512


class SelfVerify:
    """自检验证模块

    在 agent 完成所有工具调用后，构建验证 prompt 注入消息流，
    让模型对照原始任务要求检查自己的输出。
    """

    # 验证 prompt 模板
    VERIFY_PROMPT = """## Output Verification Required

Before you finalize your answer, verify your output against the original task requirements.

### Original Task (key output requirements):
{task_requirements}

### Verification Checklist:
1. **Input integrity**: Confirm you read and used the ACTUAL input files from the workspace (NOT fabricated/mock data). Use read_file to spot-check key values in input files.
2. **File naming**: Check that ALL output files follow the exact naming pattern specified in the task.
3. **File structure**: Open each output file and verify it contains all required fields with correct types.
4. **Content correctness**: Verify key content matches what was derived from the real input data.
5. **Completeness**: Ensure ALL required output files have been created — no missing files.

Use list_dir and read_file to inspect your actual output files. If you find ANY issues, fix them immediately with the available tools. Only give your final text response when everything is correct."""

    def __init__(self, config: VerifyConfig):
        self.config = config
        self._rounds_done = 0
        self._issues_found = 0
        self._issues_fixed = 0

    @property
    def can_verify(self) -> bool:
        """是否还可以执行验证"""
        return self._rounds_done < self.config.max_rounds

    def build_verify_prompt(self, task_description: str) -> str:
        """构建验证 prompt

        从任务描述中提取关键的输出要求部分。
        """
        requirements = self._extract_requirements(task_description)
        return self.VERIFY_PROMPT.format(task_requirements=requirements)

    def _extract_requirements(self, task: str) -> str:
        """从任务描述中提取输出要求部分

        优先提取 ## Output / ## Requirements / ## Output 部分的内容，
        因为这些是最关键的验证依据。如果提取不到，返回完整任务描述的截断版本。
        """
        if not task:
            return "(No task description available)"

        # 尝试提取 ## Output 部分（包含文件命名等关键要求）
        output_match = re.search(
            r'(##\s*Output.*?)(?=##\s|$)',
            task, re.DOTALL | re.IGNORECASE
        )
        if output_match:
            output_section = output_match.group(1).strip()
            # 如果 Output 部分较短，也包含 Requirements 部分
            if len(output_section) < 200:
                req_match = re.search(
                    r'(##\s*Requirements.*?)(?=##\s|$)',
                    task, re.DOTALL | re.IGNORECASE
                )
                if req_match:
                    output_section = req_match.group(1).strip() + "\n\n" + output_section
            return output_section

        # 尝试提取 ## Requirements 部分
        req_match = re.search(
            r'(##\s*Requirements.*?)(?=##\s|$)',
            task, re.DOTALL | re.IGNORECASE
        )
        if req_match:
            return req_match.group(1).strip()

        # 都没有，返回截断的任务描述
        return task[:1500] if len(task) > 1500 else task

    def record_round(self) -> None:
        """记录一次验证轮次"""
        self._rounds_done += 1

    def record_issue_found(self) -> None:
        """记录发现的问题"""
        self._issues_found += 1

    def record_issue_fixed(self) -> None:
        """记录修复的问题"""
        self._issues_fixed += 1

    def get_verify_stats(self) -> dict[str, Any]:
        """获取验证统计"""
        return {
            "rounds_done": self._rounds_done,
            "issues_found": self._issues_found,
            "issues_fixed": self._issues_fixed,
        }

    def reset(self) -> None:
        """重置状态"""
        self._rounds_done = 0
        self._issues_found = 0
        self._issues_fixed = 0
