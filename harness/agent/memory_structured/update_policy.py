"""UpdatePolicy enum and update logic for structured state tracking."""

import re
from enum import Enum
from typing import List, Tuple, Optional


class UpdatePolicy(Enum):
    """How to update structured state slots."""
    ACTIVE = "active"  # Transform and update slots (H2 approach)
    PASSIVE = "passive"  # Store tool results as-is (T1 approach baseline)
    HYBRID = "hybrid"  # Active update + passive backup


class SlotUpdateResult:
    """Result of an update operation."""

    def __init__(self, updated: bool, slot_id: str = "", action: str = ""):
        self.updated = updated
        self.slot_id = slot_id
        self.action = action  # "created" | "updated" | "completed" | "removed" | "noop"


# -------------------------------------------------------------------
# 任务初始化：从任务指令中提取初始 constraints 和 pending_subgoals
# -------------------------------------------------------------------

def extract_constraints_from_instruction(task_text: str) -> List[str]:
    """从任务指令中提取所有约束条件。

    识别的约束模式:
      - "exactly N lines", "no more than N lines", "at least N items"
      - "must mention X", "must contain X", "must include X"
      - "must be X" (格式要求)
      - 数字+单位要求: "$47.3 million", "125 MW", "Q4 2027"
    """
    constraints = []
    text = task_text

    # 1. 行数/数量要求
    line_patterns = [
        (r"exactly\s+(\d+)\s+lines?", "exactly {n} lines"),
        (r"no\s+more\s+than\s+(\d+)\s+lines?", "no more than {n} lines"),
        (r"at\s+least\s+(\d+)\s+items?", "at least {n} items"),
        (r"minimum\s+(\d+)\s+items?", "minimum {n} items"),
        (r"(\d+)\s+lines?\s+long", "{n} lines long"),
    ]
    for pattern, template in line_patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            constraints.append(template.format(n=m.group(1)))

    # 2. "must mention/include/contain X" 模式
    must_patterns = [
        r"must\s+(?:mention|include|contain)\s+([^.,\n]+?)(?:\.|$)",
        r"must\s+be\s+([^.,\n]+?)(?:\.|$)",
        r"required\s+to\s+([^.,\n]+?)(?:\.|$)",
    ]
    for pattern in must_patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            val = m.group(1).strip()
            # 去掉末尾的标点和闭括号（处理 $47.3 million) 这种情况）
            while val and val[-1] in ".,;:)" and len(val) > 3:
                val = val[:-1]
            val = val.strip(".,;:\"'")
            if len(val) > 3 and len(val) < 200:
                constraints.append(f"must: {val}")

    # 3. 特定数值约束 (金额、容量、日期)
    numeric_constraints = [
        (r"\$[\d.]+\s*million", "has budget figure"),
        (r"\$[\d,]+", "has dollar amount"),
        (r"(\d+)\s*MW", "has capacity in MW"),
        (r"Q[1-4]\s+\d{4}", "has quarterly deadline"),
        (r"\d{4}", "has year reference"),
    ]
    found_specific = set()
    for pattern, label in numeric_constraints:
        if re.search(pattern, text):
            if label not in found_specific:
                found_specific.add(label)
                constraints.append(label)

    # 4. Section 结构要求 (如 "STAKEHOLDERS:", "RISKS:", "MILESTONES:", "METRICS:")
    section_pattern = r"^([A-Z][A-Z\s]+):\s*$"
    for m in re.finditer(section_pattern, text, re.MULTILINE):
        constraints.append(f"section_required: {m.group(1).strip()}")

    # 5. 去重
    seen = set()
    deduped = []
    for c in constraints:
        c_lower = c.lower()
        if c_lower not in seen:
            seen.add(c_lower)
            deduped.append(c)

    return deduped


def extract_pending_subgoals_from_instruction(task_text: str) -> List[str]:
    """从任务指令中提取待完成的子目标。

    识别的模式:
      - "Step N: ..."
      - "N. ..." (numbered steps)
      - "Write X" / "Create X" / "Generate X"
      - Markdown 列表项
    """
    subgoals = []
    text = task_text

    # 1. "Step N: ..." 模式
    for m in re.finditer(r"(?i)step\s+(\d+)[\s:–-]+([^\n]+)", text):
        subgoal = m.group(2).strip()
        if len(subgoal) > 5:
            subgoals.append(subgoal[:200])

    # 2. "N. ..." 或 "- ..." 列表项（取较长的条目，通常是有内容的步骤）
    list_pattern = r"(?i)^[\s]*[\-\*]?\s*([A-Z][^\n]{20,200})$"
    for m in re.finditer(list_pattern, text, re.MULTILINE):
        item = m.group(1).strip()
        # 过滤掉太像标题或太短的
        if item and not item.isupper():
            subgoals.append(item)

    # 3. "Write/Create/Generate X" 模式
    action_patterns = [
        r"(?i)(?:write|create|generate|produce|output)\s+[`\"]?([^\n`\"]{5,150})[`\"]?(?:\s+containing|\s+with|\s+include|\s+that|\s+which|\.|$)",
        r"(?i)(?:write|create|generate)\s+(?:to\s+)?[`\"]?(/[^\n`\"]{5,100})[`\"]?",
    ]
    for pattern in action_patterns:
        for m in re.finditer(pattern, text):
            val = m.group(1).strip()
            if len(val) > 3:
                subgoals.append(val[:200])

    # 去重 + 过滤约束性语句（must/should/cannot 开头的列表项不是 subgoal）
    constraint_leaders = ["must", "should", "cannot", "mustn't", "don't", "do not", "required"]
    seen = set()
    deduped = []
    for s in subgoals:
        s_lower = s.lower()
        # 过滤约束性语句（must/should 开头的不是 subgoal，是 constraint）
        if any(s_lower.startswith(lead) for lead in constraint_leaders):
            continue
        if s_lower not in seen and len(s) > 5:
            seen.add(s_lower)
            deduped.append(s)

    return deduped


# -------------------------------------------------------------------
# 从工具结果中解析 constraints
# -------------------------------------------------------------------

def parse_tool_result_for_constraints(
    tool_name: str,
    tool_args: dict,
    tool_result: str,
    iteration: int,
) -> List[Tuple[str, str]]:
    """Parse tool result for constraint information.

    从工具结果中提取以下类型的约束:
      - 验证/检查失败 => 隐含约束: "X 必须满足 Y"
      - 文件内容中的显式约束声明
      - 错误信息中的格式/值要求
    """
    constraints = []
    result_lower = tool_result.lower()

    # 1. 验证类工具失败 → 提取具体约束
    if any(kw in tool_name.lower() for kw in ["validate", "check", "test", "verify"]):
        if "error" in result_lower or "failed" in result_lower or "fail" in result_lower:
            # 提取第一个错误行作为约束
            for line in tool_result.split("\n"):
                if "error" in line.lower() or "failed" in line.lower() or "assert" in line.lower():
                    clean = line.strip()[:300]
                    if clean:
                        constraints.append((f"validation_error: {clean}", tool_name))
                        break

    # 2. 读取文件时，检测内容中的显式约束语句
    if tool_name == "read_file":
        for line in tool_result.split("\n"):
            line_stripped = line.strip()
            # 只检查较短的完整句子（像是规则/约束）
            if 20 < len(line_stripped) < 200 and not line_stripped.startswith(" "):
                lower_line = line_stripped.lower()
                if any(kw in lower_line for kw in ["must ", "require", "should ", "cannot ", "mustn't "]):
                    constraints.append((f"content_constraint: {line_stripped[:200]}", "read_file"))

    # 3. exec 命令失败 → 从错误中提取约束
    if tool_name == "exec":
        if "error" in result_lower[:50] or "traceback" in result_lower:
            for line in tool_result.split("\n"):
                if re.search(r"(?:error|exception|traceback):", line.lower()):
                    constraints.append((f"exec_error: {line.strip()[:200]}", tool_name))
                    break

    return constraints


# -------------------------------------------------------------------
# 从工具结果中解析 derived_facts（决策相关事实）
# -------------------------------------------------------------------

def parse_tool_result_for_derived_facts(
    tool_name: str,
    tool_args: dict,
    tool_result: str,
    iteration: int,
) -> List[Tuple[str, str]]:
    """Parse tool result for decision-relevant derived facts.

    关键转变: 不再存储原始内容，而是存储:
      - 实体+属性: "project_name: Meridian Solar Energy Project"
      - 数值+单位+上下文: "budget: $47.3 million USD"
      - 关系: "project located_in Clearwater Valley Nevada"
      - 状态: "report contains 521 lines"
    """
    facts = []
    if not tool_result or len(tool_result) < 5:
        return facts

    # 1. read_file: 提取关键实体和数值
    if tool_name == "read_file":
        path = tool_args.get("path", tool_args.get("file_path", "unknown"))
        content = tool_result

        # 提取项目/系统名称 (通常是 Title Case 或大写词组)
        title_matches = re.findall(r"(?:Project|Platform|System|Report)[:]?\s+([A-Z][A-Za-z\s\d]+?)(?:\n|Version|\d|\.|$)", content)
        for name in title_matches[:3]:
            name = name.strip()
            if 3 < len(name) < 80:
                facts.append((f"entity: {name}", "derived"))

        # 提取关键数值: 金额、容量、人数、百分比
        numeric_patterns = [
            (r"\$([\d.]+)\s*million", lambda v: f"budget: ${v} million"),
            (r"(\d+)\s*MW", lambda v: f"capacity: {v} MW"),
            (r"(\d+)\s*MW/", lambda v: f"capacity: {v} MW"),
            (r"(\d+\.?\d*)\s*%", lambda v: f"percentage: {v}%"),
            (r"(\d+)\s*nodes?", lambda v: f"node_count: {v}"),
            (r"(\d+\.?\d*)\s*petabytes?", lambda v: f"storage: {v} PB"),
            (r"(\d+\.?\d*)\s*TB", lambda v: f"storage: {v} TB"),
        ]
        found_numeric = set()
        for pattern, formatter in numeric_patterns:
            for m in re.finditer(pattern, content[:2000], re.IGNORECASE):
                val = m.group(0)
                if val not in found_numeric:
                    found_numeric.add(val)
                    formatted = formatter(m.group(1))
                    facts.append((formatted, "derived"))

        # 提取位置信息
        location_patterns = [
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2}|[A-Z][a-z]+)",  # City, State
            r"in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",  # in Clearwater Valley
        ]
        for pattern in location_patterns:
            for m in re.finditer(pattern, content[:1500]):
                loc = m.group(0).strip()
                if 5 < len(loc) < 60:
                    facts.append((f"location: {loc}", "derived"))
                    break

        # 提取日期/时间线
        date_patterns = [
            r"Q[1-4]\s+\d{4}",
            r"\d{4}-\d{2}-\d{2}",
            r"(?:by|before|target|deadline)[:]\s*([^\n,]{5,30})",
        ]
        for pattern in date_patterns:
            for m in re.finditer(pattern, content[:1500], re.IGNORECASE):
                date_str = m.group(0).strip()
                facts.append((f"deadline: {date_str}", "derived"))
                break

        # 文件行数
        line_count = len(content.split("\n"))
        if line_count > 10:
            facts.append((f"document_line_count: {line_count}", "derived"))

    # 2. glob: 从匹配结果提取目录结构信息
    elif tool_name == "glob":
        if tool_result and "No files" not in tool_result and "found" not in tool_result.lower():
            files = [f.strip() for f in tool_result.split("\n") if f.strip()]
            if files:
                exts = set()
                for f in files:
                    if "." in f:
                        exts.add(f.rsplit(".", 1)[-1])
                if exts:
                    facts.append((f"file_types: {', '.join(sorted(exts))}", "derived"))
                facts.append((f"file_count: {len(files)}", "derived"))

    # 3. exec: 从命令输出提取结构化事实
    elif tool_name == "exec":
        if "error" not in tool_result.lower()[:20]:
            # 提取文件大小
            size_match = re.search(r"(\d+)\s+(?:bytes?|KB|MB|GB)", tool_result)
            if size_match:
                facts.append((f"size: {size_match.group(0)}", "derived"))
            # 提取行数统计
            line_match = re.search(r"(\d+)\s+(?:lines?|records?)", tool_result, re.IGNORECASE)
            if line_match:
                facts.append((f"count: {line_match.group(0)}", "derived"))
            # 提取布尔状态
            if "success" in tool_result.lower():
                facts.append(("status: success", "derived"))

    # 4. list_dir: 提取目录结构
    elif tool_name == "list_dir":
        if tool_result:
            lines = [l.strip() for l in tool_result.split("\n") if l.strip()]
            if lines:
                facts.append((f"dir_contents_count: {len(lines)}", "derived"))
                # 提取子目录名
                dirs = [l.rstrip("/") for l in lines if l.endswith("/")]
                if dirs:
                    facts.append((f"subdirs: {', '.join(dirs[:5])}", "derived"))

    return facts


# -------------------------------------------------------------------
# 检测子目标完成
# -------------------------------------------------------------------

def detect_subgoal_completion(
    tool_name: str,
    tool_args: dict,
    tool_result: str,
    pending_subgoals: list,
) -> List[str]:
    """Detect which pending subgoals are completed by this tool result.

    匹配策略:
      1. 文件写入类工具 → 检查 pending subgoal 中是否提及该文件路径
      2. 测试/验证类工具 → 检查结果中的 pass/success 信号
      3. 关键词重叠匹配
    """
    if not pending_subgoals:
        return []

    completed = []

    # 从 tool_args 提取目标文件路径
    target_paths = []
    if tool_name in ("write_file", "writeFile", "edit_file", "editFile"):
        path = tool_args.get("path", "")
        if path:
            target_paths.append(path)
            # 也添加不带目录的文件名
            target_paths.append(Path(path).name if "/" in path else path)

    # 从 tool_result 提取成功信号
    result_lower = tool_result.lower()
    success_keywords = ["success", "completed", "done", "created", "written", "saved", "passed"]
    has_success = any(kw in result_lower for kw in success_keywords)
    has_failure = any(kw in result_lower for kw in ["error", "failed", "fail", "exception"])

    for subgoal in pending_subgoals:
        if not subgoal.active:
            continue

        subgoal_text = subgoal.content.lower()
        score = 0

        # 路径匹配
        for tp in target_paths:
            if tp.lower() in subgoal_text or subgoal_text in tp.lower():
                score += 3

        # 关键词匹配
        if has_success and any(kw in subgoal_text for kw in ["write", "create", "generate", "file"]):
            score += 2
        if has_success and any(kw in subgoal_text for kw in ["test", "check", "verify"]):
            score += 2

        # 文件扩展名匹配
        for tp in target_paths:
            if "." in tp:
                ext = tp.rsplit(".", 1)[-1]
                if ext in subgoal_text:
                    score += 2

        if score >= 3:
            completed.append(subgoal.slot_id)

    return completed


# -------------------------------------------------------------------
# 从工具结果中解析 artifact_paths
# -------------------------------------------------------------------

def parse_tool_result_for_artifacts(
    tool_name: str, tool_args: dict, tool_result: str
) -> List[str]:
    """Parse tool result for artifact paths created or modified.

    相比之前版本，增强了 exec 结果中的路径提取。
    """
    paths = []

    if tool_name in ("write_file", "writeFile"):
        path = tool_args.get("path", "")
        if path:
            paths.append(path)

    elif tool_name in ("edit_file", "editFile"):
        path = tool_args.get("path", "")
        if path:
            paths.append(path)

    elif tool_name == "exec":
        # 从 exec 输出提取文件路径
        path_patterns = [
            r"/tmp/[a-zA-Z0-9_./\-]+",
            r"/root/[a-zA-Z0-9_./\-]+",
            r"[a-zA-Z0-9_./\\-]+\.(py|json|yaml|yml|md|txt|csv|log)",
        ]
        found = set()
        for pattern in path_patterns:
            for m in re.finditer(pattern, tool_result):
                p = m.group(0)
                if p not in found and len(p) > 5:
                    found.add(p)
                    paths.append(p)
        # 限制数量
        paths = paths[:10]

    return paths


# -------------------------------------------------------------------
# 辅助
# -------------------------------------------------------------------

from pathlib import Path
