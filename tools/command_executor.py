"""Command executor tool for running shell commands safely."""

import subprocess
import shlex
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio

from utils.shell_safety import find_path_escape, find_shell_operator, is_within_workspace

# Inline-code flag spellings accepted by the interpreters in
# ``CommandExecutor._INTERPRETERS_INLINE``.
_INLINE_CODE_FLAGS = ("-c", "-e", "--eval", "-r")


def _find_inline_code_flag(args: List[str]) -> Optional[str]:
    """Return the inline-code flag found in ``args``, or ``None``.

    ``args`` must be the interpreter's own arguments. Scanning stops at
    the ``-m <module>`` boundary: everything after ``-m`` belongs to the
    module's argument vector, not to the interpreter, so
    ``python3 -m pytest -c pytest.ini`` and ``python3 -m pip install -e .``
    are module flags, not inline-code flags. A code flag appearing
    before ``-m`` is still rejected (``python3 -c '<code>' -m pytest``).

    Within the interpreter's own arguments, the code-flag family is
    rejected wherever it appears (``python3 -O -c ...``,
    ``python3 -X importtime -c ...``), in ``--flag=value`` spelling
    (``node --eval=<code>``), and attached to its payload
    (``python3 -c"<code>"`` leaves ``-c<code>`` in a single token
    after ``shlex.split``; likewise ``perl -e'<code>'``).
    """
    for arg in args:
        if arg == "-m":
            # Module boundary: the rest of the argv is the module's.
            return None
        if arg in _INLINE_CODE_FLAGS:
            return arg
        if arg.startswith("--eval="):
            return "--eval=..."
        if arg.startswith("-c") and len(arg) > 2:
            return "-c..."
        if arg.startswith("-e") and len(arg) > 2:
            return "-e..."
    return None


@dataclass
class CommandResult:
    """Result of a command execution."""

    command: str
    return_code: int
    stdout: str
    stderr: str
    execution_time: float
    timestamp: datetime
    working_dir: str


class CommandExecutor:
    """
    Safe command execution tool for agents.
    Provides controlled access to shell commands with safety restrictions.
    """

    # Commands that are blocked for safety
    BLOCKED_COMMANDS = {
        "rm -rf /",
        "rm -rf ~",
        "mkfs",
        "dd if=",
        ":(){:|:&};:",  # Fork bomb
        "chmod -R 777 /",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
    }

    # Commands that are allowed (whitelist approach for production)
    SAFE_COMMANDS = {
        "ls", "cat", "grep", "find", "head", "tail", "wc",
        "pwd", "echo", "date", "whoami", "uname",
        "python", "python3", "pip", "pip3",
        "node", "npm", "npx",
        "git", "curl", "wget",
        "mkdir", "touch", "cp", "mv",
        "pytest", "mypy", "black", "flake8", "eslint",
    }

    def __init__(
        self,
        workspace_root: str = ".",
        timeout: int = 60,
        safe_mode: bool = True
    ):
        """
        Initialize command executor.

        Args:
            workspace_root: Root directory for command execution
            timeout: Default timeout in seconds
            safe_mode: If True, only allow whitelisted commands
        """
        self.workspace_root = Path(workspace_root).resolve()
        self.timeout = timeout
        self.safe_mode = safe_mode
        self.command_history: List[CommandResult] = []

    # Interpreters that accept inline code via -c/-e flags — these bypass
    # the allowlist because the LLM controls the code string, not the base command.
    _INTERPRETERS_INLINE = {"python", "python3", "node", "npx", "ruby", "perl", "bash", "sh"}

    def _is_command_safe(self, command: str) -> tuple[bool, str]:
        """
        Check if a command is safe to execute.

        Returns:
            Tuple of (is_safe, reason)
        """
        # Check blocked patterns
        for blocked in self.BLOCKED_COMMANDS:
            if blocked in command:
                return False, f"Blocked dangerous command pattern: {blocked}"

        if self.safe_mode:
            # Extract the base command
            try:
                parts = shlex.split(command)
            except ValueError as e:
                return False, f"Unparseable command: {e}"
            if not parts:
                return False, "Empty command"

            base_cmd = parts[0]
            # Handle paths to commands
            if "/" in base_cmd:
                base_cmd = os.path.basename(base_cmd)

            if base_cmd not in self.SAFE_COMMANDS:
                return False, f"Command '{base_cmd}' not in allowed list. Safe mode is enabled."

            # Block inline code execution — LLM can smuggle arbitrary code through
            # `python -c "import os; os.system(...)"` which passes the allowlist.
            # The flag is rejected wherever it appears in the interpreter's own
            # arguments (not only at parts[1]), in `--flag=value` spelling, and
            # when attached to its payload as a single token. The scan stops at
            # the `-m <module>` boundary: after `-m` the arguments belong to the
            # module (`python3 -m pytest -c pytest.ini`), not to python.
            if base_cmd in self._INTERPRETERS_INLINE:
                inline_flag = _find_inline_code_flag(parts[1:])
                if inline_flag:
                    return False, (
                        f"Blocked inline code execution: {base_cmd} {inline_flag}. "
                        "Write code to a file and execute it instead."
                    )

            # Shell chaining, pipes, redirection and substitution execute
            # commands that never touch the allowlist.
            operator = find_shell_operator(command)
            if operator:
                return False, (
                    f"Blocked {operator}. "
                    "Run a single command per call; chaining bypasses the allowlist."
                )

            # Arguments must not point outside the configured workspace.
            escape = find_path_escape(command, self.workspace_root)
            if escape:
                return False, f"Blocked {escape}."

        return True, "OK"

    def execute(
        self,
        command: str,
        working_dir: Optional[str] = None,
        timeout: Optional[int] = None,
        env: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Execute a shell command.

        Args:
            command: Command to execute
            working_dir: Working directory (defaults to workspace root)
            timeout: Timeout in seconds (defaults to instance timeout)
            env: Additional environment variables

        Returns:
            Dict with execution results
        """
        # Safety check
        is_safe, reason = self._is_command_safe(command)
        if not is_safe:
            return {
                "success": False,
                "error": reason,
                "command": command
            }

        # Resolve working directory — it must stay inside the workspace
        cwd = self.workspace_root
        if working_dir:
            cwd = Path(working_dir)
            if not cwd.is_absolute():
                cwd = self.workspace_root / cwd
            cwd = cwd.resolve()
            if not is_within_workspace(cwd, self.workspace_root):
                return {
                    "success": False,
                    "error": f"Working directory outside workspace: {working_dir}",
                    "command": command
                }

        if not cwd.exists():
            return {
                "success": False,
                "error": f"Working directory does not exist: {cwd}",
                "command": command
            }

        # Prepare environment
        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)

        # Execute command
        timeout = timeout or self.timeout
        start_time = datetime.now()

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=exec_env
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            cmd_result = CommandResult(
                command=command,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=execution_time,
                timestamp=start_time,
                working_dir=str(cwd)
            )
            self.command_history.append(cmd_result)

            return {
                "success": result.returncode == 0,
                "command": command,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": execution_time,
                "working_dir": str(cwd)
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds",
                "command": command
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "command": command
            }

    async def execute_async(
        self,
        command: str,
        working_dir: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute a command asynchronously.

        Args:
            command: Command to execute
            working_dir: Working directory
            timeout: Timeout in seconds
        """
        # Safety check
        is_safe, reason = self._is_command_safe(command)
        if not is_safe:
            return {"success": False, "error": reason, "command": command}

        cwd = self.workspace_root
        if working_dir:
            cwd = Path(working_dir)
            if not cwd.is_absolute():
                cwd = self.workspace_root / cwd
            cwd = cwd.resolve()
            if not is_within_workspace(cwd, self.workspace_root):
                return {
                    "success": False,
                    "error": f"Working directory outside workspace: {working_dir}",
                    "command": command
                }

        timeout = timeout or self.timeout
        start_time = datetime.now()

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                "success": process.returncode == 0,
                "command": command,
                "return_code": process.returncode,
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else "",
                "execution_time": execution_time,
                "working_dir": str(cwd)
            }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds",
                "command": command
            }
        except Exception as e:
            return {"success": False, "error": str(e), "command": command}

    def execute_python(
        self,
        script: str,
        script_args: List[str] = None,
        working_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a Python script or code.

        Args:
            script: Python file path or inline code
            script_args: Arguments to pass to the script
            working_dir: Working directory
        """
        script_path = Path(script)
        if script_path.exists() and script_path.suffix == ".py":
            # Execute as file
            args = " ".join(script_args or [])
            command = f"python3 {script} {args}"
        else:
            # Execute as inline code
            escaped_script = script.replace('"', '\\"')
            command = f'python3 -c "{escaped_script}"'

        return self.execute(command, working_dir=working_dir)

    def run_tests(
        self,
        test_path: str = ".",
        framework: str = "pytest",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run tests using a specified framework.

        Args:
            test_path: Path to test file or directory
            framework: Test framework (pytest, unittest)
            verbose: Enable verbose output
        """
        if framework == "pytest":
            cmd = f"pytest {'-v' if verbose else ''} {test_path}"
        elif framework == "unittest":
            cmd = f"python3 -m unittest {'discover' if Path(test_path).is_dir() else ''} {test_path}"
        else:
            return {"success": False, "error": f"Unknown test framework: {framework}"}

        return self.execute(cmd)

    def install_package(self, package: str, dev: bool = False) -> Dict[str, Any]:
        """
        Install a Python package using pip.

        Args:
            package: Package name (optionally with version)
            dev: Install as dev dependency
        """
        cmd = f"pip3 install {package}"
        return self.execute(cmd)

    def git_operation(self, operation: str, *args) -> Dict[str, Any]:
        """
        Execute a git operation.

        Args:
            operation: Git operation (status, add, commit, push, pull, etc.)
            args: Additional arguments
        """
        valid_operations = {
            "status", "add", "commit", "push", "pull", "fetch",
            "branch", "checkout", "merge", "log", "diff", "clone", "init"
        }

        if operation not in valid_operations:
            return {"success": False, "error": f"Invalid git operation: {operation}"}

        args_str = " ".join(str(a) for a in args)
        cmd = f"git {operation} {args_str}"

        return self.execute(cmd)

    def get_command_history(self) -> List[Dict[str, Any]]:
        """Get history of executed commands."""
        return [
            {
                "command": c.command,
                "return_code": c.return_code,
                "execution_time": c.execution_time,
                "timestamp": c.timestamp.isoformat(),
                "working_dir": c.working_dir
            }
            for c in self.command_history
        ]

    def clear_history(self) -> None:
        """Clear command history."""
        self.command_history.clear()
