"""Docker-based file system tools for Harbor architecture.

All file operations are executed inside the Docker container via docker exec,
ensuring the agent has a consistent view of the filesystem.
"""

import asyncio
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


class _DockerFsTool(Tool):
    """Base class for Docker filesystem tools."""

    def __init__(
        self,
        container_name: str,
        mount_point: str = "/workspace",
        allowed_dir: str | None = None,
    ):
        self._container_name = container_name
        self._mount_point = mount_point
        self._allowed_dir = allowed_dir or mount_point

    def _resolve_path(self, path: str) -> str:
        """Resolve path to absolute path inside container."""
        if not path.startswith("/"):
            path = f"{self._mount_point}/{path}"
        return path

    def _is_allowed(self, path: str) -> bool:
        """Check if path is under allowed directory."""
        resolved = self._resolve_path(path)
        return resolved.startswith(self._allowed_dir)

    async def _docker_exec(self, cmd: list[str], timeout: int = 60) -> tuple[int, str, str]:
        """Execute command inside container via docker exec."""
        docker_cmd = [
            "docker", "exec",
            "-w", self._mount_point,
            self._container_name,
        ] + cmd

        process = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
            return process.returncode, stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")
        except asyncio.TimeoutError:
            process.kill()
            return -1, "", "Command timed out"


class DockerReadFileTool(_DockerFsTool):
    """Read file contents from inside Docker container."""

    _MAX_CHARS = 128_000
    _DEFAULT_LIMIT = 2000

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file inside the container."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to read"},
                "offset": {"type": "integer", "description": "Line number to start from (1-indexed)", "minimum": 1},
                "limit": {"type": "integer", "description": "Maximum lines to read", "minimum": 1},
            },
            "required": ["path"],
        }

    async def execute(self, path: str | None = None, offset: int = 1, limit: int | None = None, **kwargs: Any) -> Any:
        try:
            if not path:
                return "Error: Unknown path"

            if not self._is_allowed(path):
                return f"Error: Path {path} is outside allowed directory"

            container_path = self._resolve_path(path)

            # Check if file exists
            rc, _, _ = await self._docker_exec(["test", "-f", container_path])
            if rc != 0:
                return f"Error: File not found: {path}"

            # Read file content
            if offset > 1 or limit:
                # Use sed for pagination
                limit = limit or self._DEFAULT_LIMIT
                rc, stdout, stderr = await self._docker_exec(
                    ["sed", "-n", f"{offset},{offset + limit - 1}p", container_path]
                )
            else:
                rc, stdout, stderr = await self._docker_exec(["cat", container_path])

            if rc != 0:
                return f"Error reading file: {stderr}"

            content = stdout[:self._MAX_CHARS]
            lines = content.splitlines()

            # Add line numbers
            numbered = "\n".join(
                f"{i + offset:3}| {line}" for i, line in enumerate(lines)
            )

            if len(stdout) > self._MAX_CHARS:
                numbered += f"\n... ({len(stdout) - self._MAX_CHARS} chars truncated)"

            return numbered or "(empty file)"

        except Exception as e:
            return f"Error reading file: {e}"


class DockerWriteFileTool(_DockerFsTool):
    """Write file contents inside Docker container."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file inside the container."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to write"},
                "content": {"type": "string", "description": "The content to write"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str | None = None, content: str | None = None, **kwargs: Any) -> str:
        try:
            if not path:
                return "Error: Unknown path"
            if content is None:
                return "Error: Unknown content"

            if not self._is_allowed(path):
                return f"Error: Path {path} is outside allowed directory"

            container_path = self._resolve_path(path)

            # Create parent directory
            parent = Path(container_path).parent
            await self._docker_exec(["mkdir", "-p", str(parent)])

            # Write file using heredoc
            # Escape single quotes in content
            escaped_content = content.replace("'", "'\"'\"'")
            rc, stdout, stderr = await self._docker_exec(
                ["sh", "-c", f"cat > '{container_path}' << 'EOF'\n{escaped_content}\nEOF"]
            )

            if rc != 0:
                return f"Error writing file: {stderr}"

            return f"Successfully wrote {len(content)} bytes to {path}"

        except Exception as e:
            return f"Error writing file: {e}"


class DockerListDirTool(_DockerFsTool):
    """List directory contents inside Docker container."""

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List files and directories inside the container."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The directory path to list"},
            },
            "required": ["path"],
        }

    async def execute(self, path: str | None = None, **kwargs: Any) -> str:
        try:
            if not path:
                path = "."

            if not self._is_allowed(path):
                return f"Error: Path {path} is outside allowed directory"

            container_path = self._resolve_path(path)

            # Check if directory exists
            rc, _, _ = await self._docker_exec(["test", "-d", container_path])
            if rc != 0:
                return f"Error: Directory not found: {path}"

            # List directory
            rc, stdout, stderr = await self._docker_exec(
                ["ls", "-la", container_path]
            )

            if rc != 0:
                return f"Error listing directory: {stderr}"

            return stdout or "(empty directory)"

        except Exception as e:
            return f"Error listing directory: {e}"


class DockerEditFileTool(_DockerFsTool):
    """Edit file contents inside Docker container."""

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return "Edit a file by replacing text inside the container."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to edit"},
                "old_string": {"type": "string", "description": "The text to replace"},
                "new_string": {"type": "string", "description": "The replacement text"},
            },
            "required": ["path", "old_string", "new_string"],
        }

    async def execute(self, path: str | None = None, old_string: str | None = None,
                      new_string: str | None = None, **kwargs: Any) -> str:
        try:
            if not path or old_string is None or new_string is None:
                return "Error: Missing required parameters"

            if not self._is_allowed(path):
                return f"Error: Path {path} is outside allowed directory"

            container_path = self._resolve_path(path)

            # Read current content
            rc, content, _ = await self._docker_exec(["cat", container_path])
            if rc != 0:
                return f"Error: File not found: {path}"

            # Perform replacement
            if old_string not in content:
                return f"Error: old_string not found in file"

            new_content = content.replace(old_string, new_string, 1)

            # Write back
            escaped_content = new_content.replace("'", "'\"'\"'")
            rc, _, stderr = await self._docker_exec(
                ["sh", "-c", f"cat > '{container_path}' << 'EOF'\n{escaped_content}\nEOF"]
            )

            if rc != 0:
                return f"Error writing file: {stderr}"

            return f"Successfully edited {path}"

        except Exception as e:
            return f"Error editing file: {e}"
