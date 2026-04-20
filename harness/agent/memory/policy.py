"""
Write and Retrieval policy enums for memory management.
"""

from enum import Enum


class WritePolicy(Enum):
    """When to write to episodic memory."""

    ALWAYS = "always"  # Write everything
    TOOL_RESULT_OR_ERROR = "tool_result_or_error"  # Default: write tool results and errors
    TOOL_RESULT = "tool_result"  # Only tool results
    ERROR = "error"  # Only errors
    LONG_CONTENT = "long_content"  # Only content exceeding threshold
    NEVER = "never"  # Never write


class RetrievalPolicy(Enum):
    """How to retrieve memories."""

    RECENT = "recent"  # Most recent first (default)
    FREQUENCY = "frequency"  # Most accessed first
    HYBRID = "hybrid"  # Combination of recent and frequency
    SEMANTIC = "semantic"  # Semantic similarity via embedding (requires query)


def should_write_to_memory(
    policy: WritePolicy,
    event_type: str,
    content: str = "",
    long_content_threshold: int = 500,
) -> bool:
    """Determine if an event should be written to memory.

    Args:
        policy: The write policy to apply
        event_type: Type of event ("tool_result", "error", "user_prompt")
        content: Content of the event
        long_content_threshold: Min length for LONG_CONTENT policy

    Returns:
        True if should write to memory
    """
    # Use policy.value comparisons to avoid enum class identity issues
    # (policy may come from a different import path than WritePolicy enum members)
    pv = policy.value
    if pv == "always":
        return True
    elif pv == "never":
        return False
    elif pv == "tool_result_or_error":
        return event_type in ("tool_result", "error")
    elif pv == "tool_result":
        return event_type == "tool_result"
    elif pv == "error":
        return event_type == "error"
    elif pv == "long_content":
        return len(content) >= long_content_threshold
    return False
