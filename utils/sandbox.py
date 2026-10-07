"""
Sandbox Mode for Company AGI - Secure command execution.

Provides:
- Filesystem restrictions (allowed/blocked paths)
- Network isolation (optional)
- Command filtering (blocked patterns)
- Resource limits (timeout, memory)
- Audit logging
"""

import asyncio
import os
import re
import shlex
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .shell_safety import find_path_escape, find_shell_operator


class SandboxMode(Enum):
    """Sandbox restriction levels."""
    DISABLED = "disabled"  # No restrictions
    PERMISSIVE = "permissive"  # Block dangerous commands only
    STANDARD = "standard"  # Filesystem + network restrictions
    STRICT = "strict"  # Maximum isolation


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""
    mode: SandboxMode = SandboxMode.STANDARD

    # Filesystem restrictions
    allowed_paths: List[str] = field(default_factory=list)  # Glob patterns
    blocked_paths: List[str] = field(default_factory=lambda: [
        "/etc/*",
        "/var/*",
        "/usr/*",
        "/bin/*",
        "/sbin/*",
        "/System/*",
        "/Library/*",
        "~/.ssh/*",
        "~/.aws/*",
        "~/.config/*",
        "**/.git/config",
        "**/.env",
        "**/*.pem",
        "**/*.key",
        "**/secrets/*",
        "**/credentials/*",
    ])

    # Command restrictions
    blocked_commands: List[str] = field(default_factory=lambda: [
        "rm -rf /",
        "rm -rf /*",
        "sudo rm",
        "mkfs",
        "dd if=",
        ":(){ :|:& };:",  # Fork bomb
        "> /dev/sda",
        "chmod 777 /",
        "curl | sh",
        "wget | sh",
        "eval $(",
    ])
    blocked_command_patterns: List[str] = field(default_factory=lambda: [
        r"sudo\s+.*",
        r"su\s+-\s+root",
        r"chmod\s+[0-7]{3}\s+/(?!home|tmp|var/tmp)",
        r"chown\s+root",
        r">\s*/etc/",
        r"rm\s+-[rf]+\s+/[^/\s]",
    ])
    allowed_commands: Optional[List[str]] = None  # If set, only these are allowed

    # Network restrictions
    allow_network: bool = True
    blocked_hosts: List[str] = field(default_factory=list)
    allowed_hosts: Optional[List[str]] = None

    # Resource limits
    timeout: int = 120  # seconds
    max_output_size: int = 100000  # bytes

    # Working directory
    working_dir: Optional[str] = None
    use_temp_dir: bool = False

    # Logging
    log_commands: bool = True
    log_file: Optional[str] = None


@dataclass
class SandboxResult:
    """Result from sandboxed command execution."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    blocked: bool = False
    block_reason: Optional[str] = None
    command: str = ""
    duration_ms: Optional[float] = None


class CommandValidator:
    """Validates commands against sandbox rules."""

    def __init__(self, config: SandboxConfig):
        self.config = config
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        self._blocked_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.config.blocked_command_patterns
        ]

    def validate(self, command: str) -> tuple[bool, Optional[str]]:
        """
        Validate a command.
        Returns (is_valid, block_reason).
        """
        if self.config.mode == SandboxMode.DISABLED:
            return True, None

        # Check blocked commands (exact match)
        for blocked in self.config.blocked_commands:
            if blocked in command:
                return False, f"Command contains blocked pattern: {blocked}"

        # Check blocked patterns (regex)
        for pattern in self._blocked_patterns:
            if pattern.search(command):
                return False, f"Command matches blocked pattern: {pattern.pattern}"

        # Shell chaining, pipes, redirection and substitution run a second
        # command that is never inspected below — reject them whenever a
        # filter (allowlist or standard/strict mode) is expected to apply.
        if (
            self.config.allowed_commands is not None
            or self.config.mode in (SandboxMode.STANDARD, SandboxMode.STRICT)
        ):
            operator = find_shell_operator(command)
            if operator:
                return False, f"Blocked {operator}"

        # Check allowed commands (whitelist mode)
        if self.config.allowed_commands is not None:
            cmd_parts = shlex.split(command)
            if cmd_parts:
                base_cmd = cmd_parts[0]
                if base_cmd not in self.config.allowed_commands:
                    return False, f"Command not in allowlist: {base_cmd}"
                # Block inline code execution — LLM can smuggle arbitrary code
                # through `python -c "..."` which passes the allowlist.
                interpreters_inline = {"python", "python3", "node", "npx", "ruby", "perl", "bash", "sh"}
                if base_cmd in interpreters_inline and len(cmd_parts) >= 2:
                    if cmd_parts[1] in ("-c", "-e", "--eval", "-r"):
                        return False, (
                            f"Blocked inline code execution: {base_cmd} {cmd_parts[1]}. "
                            "Write code to a file and execute it instead."
                        )

        # Check for file path access
        is_valid, reason = self._validate_file_paths(command)
        if not is_valid:
            return False, reason

        # Paths (including relative escapes) must stay inside the sandbox
        # working directory when one is configured.
        if self.config.working_dir:
            escape = find_path_escape(command, Path(self.config.working_dir))
            if escape:
                return False, f"Blocked {escape}"

        return True, None

    def _validate_file_paths(self, command: str) -> tuple[bool, Optional[str]]:
        """Check file paths in command against restrictions."""
        import fnmatch

        # Extract potential file paths from command
        paths = self._extract_paths(command)

        for path in paths:
            # Expand user home
            expanded_path = os.path.expanduser(path)

            # Check blocked paths
            for blocked in self.config.blocked_paths:
                blocked_expanded = os.path.expanduser(blocked)
                if fnmatch.fnmatch(expanded_path, blocked_expanded):
                    return False, f"Access to path blocked: {path}"

            # Check allowed paths (if specified)
            if self.config.allowed_paths:
                allowed = False
                for allow_pattern in self.config.allowed_paths:
                    allow_expanded = os.path.expanduser(allow_pattern)
                    if fnmatch.fnmatch(expanded_path, allow_expanded):
                        allowed = True
                        break
                if not allowed:
                    return False, f"Path not in allowlist: {path}"

        return True, None

    def _extract_paths(self, command: str) -> List[str]:
        """Extract potential file paths from a command."""
        paths = []

        # Common path patterns
        path_patterns = [
            r'(?:^|\s)(/[^\s]+)',  # Absolute paths
            r'(?:^|\s)(~/[^\s]+)',  # Home-relative paths
            r'(?:^|\s)(\./[^\s]+)',  # Current-dir relative
            r'(?:^|\s)(\.\.?/[^\s]+)',  # Parent-dir relative
        ]

        for pattern in path_patterns:
            matches = re.findall(pattern, command)
            paths.extend(matches)

        return paths


class SandboxExecutor:
    """Executes commands in a sandboxed environment."""

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self.validator = CommandValidator(self.config)
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None

    async def execute(self, command: str) -> SandboxResult:
        """Execute a command in the sandbox."""
        import time
        start_time = time.time()

        # Validate command
        is_valid, block_reason = self.validator.validate(command)
        if not is_valid:
            return SandboxResult(
                success=False,
                blocked=True,
                block_reason=block_reason,
                command=command,
            )

        # Prepare working directory
        cwd = self._get_working_dir()

        # Prepare environment
        env = self._prepare_environment()

        try:
            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return SandboxResult(
                    success=False,
                    stderr=f"Command timed out after {self.config.timeout}s",
                    exit_code=-1,
                    command=command,
                    duration_ms=(time.time() - start_time) * 1000,
                )

            # Truncate output if too large
            stdout_str = stdout.decode(errors="replace")[:self.config.max_output_size]
            stderr_str = stderr.decode(errors="replace")[:self.config.max_output_size]

            duration_ms = (time.time() - start_time) * 1000

            # Log if enabled
            if self.config.log_commands:
                self._log_execution(command, process.returncode, duration_ms)

            return SandboxResult(
                success=process.returncode == 0,
                stdout=stdout_str,
                stderr=stderr_str,
                exit_code=process.returncode or 0,
                command=command,
                duration_ms=duration_ms,
            )

        except Exception as e:
            return SandboxResult(
                success=False,
                stderr=str(e),
                exit_code=-1,
                command=command,
                duration_ms=(time.time() - start_time) * 1000,
            )

    def _get_working_dir(self) -> str:
        """Get working directory for execution."""
        if self.config.use_temp_dir:
            if self._temp_dir is None:
                self._temp_dir = tempfile.TemporaryDirectory(prefix="sandbox_")
            return self._temp_dir.name

        if self.config.working_dir:
            return self.config.working_dir

        return os.getcwd()

    def _prepare_environment(self) -> Dict[str, str]:
        """Prepare environment variables for sandboxed execution."""
        env = os.environ.copy()

        if self.config.mode == SandboxMode.STRICT:
            # Minimal environment
            env = {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": os.environ.get("HOME", "/tmp"),
                "LANG": "en_US.UTF-8",
                "TERM": "xterm-256color",
            }

        # Block network if configured
        if not self.config.allow_network:
            # This is a hint - true network isolation requires OS-level support
            env["no_proxy"] = "*"
            env["NO_PROXY"] = "*"

        return env

    def _log_execution(self, command: str, exit_code: Optional[int], duration_ms: float) -> None:
        """Log command execution."""
        import json
        from datetime import datetime

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "exit_code": exit_code,
            "duration_ms": duration_ms,
            "mode": self.config.mode.value,
        }

        if self.config.log_file:
            log_path = Path(self.config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

    def cleanup(self) -> None:
        """Clean up temporary resources."""
        if self._temp_dir:
            self._temp_dir.cleanup()
            self._temp_dir = None


class SandboxedBashTool:
    """
    Bash tool with sandbox protection.
    Drop-in replacement for BashTool with security controls.
    """

    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
        workspace_root: str = ".",
    ):
        if config is None:
            config = SandboxConfig()

        # Add workspace to allowed paths
        workspace = Path(workspace_root).resolve()
        config.allowed_paths = config.allowed_paths or []
        config.allowed_paths.append(f"{workspace}/**")
        config.working_dir = str(workspace)

        self.config = config
        self.executor = SandboxExecutor(config)
        self.workspace_root = workspace

    async def execute(
        self,
        command: str,
        timeout: Optional[int] = None,
        safe_mode: bool = True,
    ) -> Dict[str, Any]:
        """Execute a command in the sandbox."""
        # Store original settings
        original_timeout = self.config.timeout
        original_mode = self.config.mode

        # Override timeout if specified
        if timeout:
            self.config.timeout = timeout

        # If safe_mode is disabled, use permissive mode
        if not safe_mode:
            self.config.mode = SandboxMode.PERMISSIVE

        try:
            result = await self.executor.execute(command)

            return {
                "success": result.success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "blocked": result.blocked,
                "block_reason": result.block_reason,
                "duration_ms": result.duration_ms,
            }

        finally:
            # Restore original settings
            self.config.timeout = original_timeout
            self.config.mode = original_mode

    def validate_command(self, command: str) -> tuple[bool, Optional[str]]:
        """Validate a command without executing it."""
        return self.executor.validator.validate(command)

    def cleanup(self) -> None:
        """Clean up resources."""
        self.executor.cleanup()


# Convenience functions for creating common sandbox configurations

def create_readonly_sandbox(workspace_root: str = ".") -> SandboxConfig:
    """Create a read-only sandbox configuration."""
    return SandboxConfig(
        mode=SandboxMode.STRICT,
        allowed_paths=[f"{Path(workspace_root).resolve()}/**"],
        blocked_commands=[
            "rm", "rmdir", "mv", "cp", "touch", "mkdir",
            "chmod", "chown", "ln", "> ", ">> ",
        ],
        blocked_command_patterns=[
            r">\s+",  # Redirections
            r">>\s+",
            r"tee\s+",
        ],
        allow_network=False,
    )


def create_development_sandbox(workspace_root: str = ".") -> SandboxConfig:
    """Create a development-friendly sandbox configuration."""
    workspace = Path(workspace_root).resolve()
    return SandboxConfig(
        mode=SandboxMode.STANDARD,
        allowed_paths=[
            f"{workspace}/**",
            "/tmp/**",
            "/var/tmp/**",
        ],
        blocked_paths=[
            "~/.ssh/*",
            "~/.aws/*",
            "**/.git/config",
            "**/.env",
        ],
        allow_network=True,
        timeout=300,  # 5 minutes for dev tasks
    )


def create_ci_sandbox() -> SandboxConfig:
    """Create a CI/CD-appropriate sandbox configuration."""
    return SandboxConfig(
        mode=SandboxMode.STRICT,
        allowed_paths=[
            os.getcwd() + "/**",
            "/tmp/**",
        ],
        blocked_commands=[
            "sudo",
            "su",
            "passwd",
            "ssh",
            "scp",
        ],
        allow_network=True,  # Needed for package installs
        timeout=600,  # 10 minutes for CI tasks
        use_temp_dir=False,
    )


# Global sandbox instance
_global_sandbox: Optional[SandboxExecutor] = None


def get_sandbox(config: Optional[SandboxConfig] = None) -> SandboxExecutor:
    """Get or create the global sandbox executor."""
    global _global_sandbox
    if _global_sandbox is None:
        _global_sandbox = SandboxExecutor(config or SandboxConfig())
    return _global_sandbox


def reset_sandbox() -> None:
    """Reset the global sandbox."""
    global _global_sandbox
    if _global_sandbox:
        _global_sandbox.cleanup()
    _global_sandbox = None
