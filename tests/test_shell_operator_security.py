"""Shell chaining / substitution / workspace escape tests.

Covers all three command execution paths: ``CommandExecutor``, ``BashTool``
and the sandbox ``CommandValidator`` (SandboxValidator). Simple approved
commands must keep working.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from tools.command_executor import CommandExecutor
from tools.unified_tools import BashTool, TaskTool
from utils.sandbox import CommandValidator, SandboxConfig, SandboxMode

CHAINING = [
    "ls ; ls -la",
    "ls && ls -la",
    "ls || ls -la",
    "ls | wc -l",
    "ls\nls",
]

REDIRECTION = [
    "echo hi > out.txt",
    "echo hi >> out.txt",
    "cat file < in.txt",
    "grep x file 2>&1",
]

SUBSTITUTION = [
    "ls $(ls)",
    "ls `ls`",
]

APPROVED = [
    "ls -la",
    "cat file.txt",
    "grep -r pattern .",
    "find . -name '*.py'",
    "python script.py",
    "python -m pytest",
    "pip install requests",
    "git status",
    'git commit -m "fix: handle a; b"',
    "grep -E 'x|y' file",
    "head -n 5 file",
]

PATH_ESCAPES = [
    "cat /etc/hosts",
    "cat ../outside/secret.txt",
    "ls /",
    "grep x --file=/etc/passwd",
]


# ============================================================
#  CommandExecutor
# ============================================================


@pytest.fixture
def executor(tmp_path):
    return CommandExecutor(workspace_root=str(tmp_path), safe_mode=True)


class TestCommandExecutorShellHardening:
    @pytest.mark.parametrize("command", CHAINING + REDIRECTION + SUBSTITUTION)
    def test_chaining_substitution_redirection_blocked(self, executor, command):
        is_safe, reason = executor._is_command_safe(command)
        assert not is_safe, command
        assert "Blocked" in reason

    @pytest.mark.parametrize("command", APPROVED)
    def test_simple_approved_commands_still_allowed(self, executor, command):
        is_safe, reason = executor._is_command_safe(command)
        assert is_safe, f"{command}: {reason}"

    @pytest.mark.parametrize("command", PATH_ESCAPES)
    def test_workspace_escapes_blocked(self, executor, command):
        is_safe, reason = executor._is_command_safe(command)
        assert not is_safe, command
        assert "path outside workspace" in reason

    def test_unparseable_command_is_rejected_not_raised(self, executor):
        is_safe, reason = executor._is_command_safe('echo "unclosed')
        assert not is_safe
        assert "Unparseable" in reason

    def test_execute_rejects_working_dir_outside_workspace(self, executor, tmp_path):
        outside = tmp_path.parent
        result = executor.execute("pwd", working_dir=str(outside))
        assert result["success"] is False
        assert "outside workspace" in result["error"]

    def test_execute_rejects_relative_working_dir_escape(self, executor):
        result = executor.execute("pwd", working_dir="..")
        assert result["success"] is False
        assert "outside workspace" in result["error"]

    def test_execute_runs_allowed_command_in_workspace(self, executor, tmp_path):
        result = executor.execute("pwd")
        assert result["success"] is True, result
        assert tmp_path.name in result["stdout"]

    def test_execute_refuses_chained_command(self, executor):
        result = executor.execute("ls ; ls -la")
        assert result["success"] is False
        assert "Blocked" in result["error"]

    def test_async_execute_refuses_chained_command(self, executor):
        result = asyncio_run(executor.execute_async("ls | wc -l"))
        assert result["success"] is False
        assert "Blocked" in result["error"]


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


# ============================================================
#  BashTool
# ============================================================


@pytest.fixture
def bash_tool(tmp_path):
    return BashTool(workspace_root=str(tmp_path), safe_mode=True)


class TestBashToolShellHardening:
    @pytest.mark.parametrize("command", CHAINING + REDIRECTION + SUBSTITUTION)
    def test_chaining_substitution_redirection_blocked(self, bash_tool, command):
        result = bash_tool.execute(command)
        assert not result.success, command
        assert "Blocked" in result.error

    @pytest.mark.parametrize("command", ["ls -la", "echo hi", "pwd", "cat file.txt"])
    def test_simple_approved_commands_still_allowed(self, bash_tool, command):
        # Only harmless commands are executed for real here — the full
        # allowlist is asserted through CommandExecutor._is_command_safe.
        result = bash_tool.execute(command)
        # May legitimately fail (file not found etc.) but must not be blocked
        # by the safety layer.
        assert "Blocked" not in (result.error or ""), f"{command}: {result.error}"

    @pytest.mark.parametrize("command", PATH_ESCAPES)
    def test_workspace_escapes_blocked(self, bash_tool, command):
        result = bash_tool.execute(command)
        assert not result.success, command
        assert "path outside workspace" in result.error

    @pytest.mark.parametrize(
        "command",
        [
            'python3 "-c" "print(1)"',
            "python3 '-c' 'print(1)'",
            '"python3" -c "print(1)"',
            'node "-e" "process.exit(0)"',
            'python "-c" "print(1)"',
            'node "--eval" "1"',
        ],
    )
    def test_quoted_inline_code_flag_blocked(self, bash_tool, command):
        """Quoted flags must not bypass the inline-code check (shlex, not split)."""
        result = bash_tool.execute(command)
        assert not result.success, command
        assert "Blocked inline code execution" in result.error

    @pytest.mark.parametrize(
        "command",
        [
            'echo "hello world"',
            "grep -E 'x|y' file.txt",
            "find . -name '*.py'",
            'git commit -m "fix: handle a; b"',
        ],
    )
    def test_quoted_arguments_still_allowed(self, bash_tool, command):
        # Quoting is stripped for the safety checks only — legitimate
        # quoted arguments must keep working.
        result = bash_tool.execute(command)
        assert "Blocked" not in (result.error or ""), f"{command}: {result.error}"

    def test_malformed_quoting_rejected_not_raised(self, bash_tool):
        result = bash_tool.execute('echo "unclosed')
        assert not result.success
        assert "Unparseable" in result.error

    def test_cwd_outside_workspace_rejected(self, bash_tool):
        result = bash_tool.execute("pwd", cwd="/etc")
        assert not result.success
        assert "outside workspace" in result.error

    def test_allowed_command_runs_in_workspace(self, bash_tool, tmp_path):
        result = bash_tool.execute("pwd")
        assert result.success, result.error
        assert tmp_path.name in result.data["stdout"]


# ============================================================
#  Background tasks (TaskTool)
# ============================================================


@pytest.fixture
def task_tool(tmp_path):
    return TaskTool(workspace_root=str(tmp_path))


class TestTaskToolShellHardening:
    @pytest.mark.parametrize("command", CHAINING + REDIRECTION + SUBSTITUTION)
    def test_chaining_substitution_redirection_blocked(self, task_tool, command):
        result = task_tool.run_background(command)
        assert not result.success, command
        assert "Blocked" in result.error

    def test_cwd_outside_workspace_rejected(self, task_tool):
        result = task_tool.run_background("pwd", cwd="/etc")
        assert not result.success
        assert "outside workspace" in result.error

    def test_simple_command_accepted(self, task_tool):
        result = task_tool.run_background("pwd")
        assert result.success, result.error
        assert result.data["task_id"] in task_tool.tasks


# ============================================================
#  Sandbox validator
# ============================================================


@pytest.fixture
def sandbox_validator(tmp_path):
    config = SandboxConfig(
        mode=SandboxMode.STANDARD,
        allowed_commands=[
            "ls", "cat", "echo", "grep", "find", "head",
            "python", "python3", "node", "pip", "git",
        ],
        working_dir=str(tmp_path),
    )
    return CommandValidator(config)


class TestSandboxValidatorShellHardening:
    @pytest.mark.parametrize("command", CHAINING + REDIRECTION + SUBSTITUTION)
    def test_chaining_substitution_redirection_blocked(self, sandbox_validator, command):
        is_valid, reason = sandbox_validator.validate(command)
        assert not is_valid, command
        assert reason and "Blocked" in reason

    @pytest.mark.parametrize("command", ["ls -la", "cat file.txt", 'echo "a; b"', "grep -E 'x|y' f"])
    def test_simple_allowed_commands_pass(self, sandbox_validator, command):
        is_valid, reason = sandbox_validator.validate(command)
        assert is_valid, reason

    def test_command_not_in_allowlist_rejected(self, sandbox_validator):
        is_valid, reason = sandbox_validator.validate("curl example.com")
        assert not is_valid
        assert "allowlist" in reason

    def test_inline_code_rejected_when_interpreter_allowlisted(self, sandbox_validator):
        is_valid, reason = sandbox_validator.validate("python3 -c 'print(1)'")
        assert not is_valid
        assert "inline code execution" in reason

    def test_absolute_path_outside_workspace_rejected(self, sandbox_validator):
        is_valid, reason = sandbox_validator.validate("cat /etc/hosts")
        assert not is_valid
        assert "blocked" in reason.lower()

    def test_relative_escape_rejected(self, sandbox_validator):
        is_valid, reason = sandbox_validator.validate("cat ../secret.txt")
        assert not is_valid
        assert "workspace" in reason

    def test_disabled_mode_still_validates_nothing(self, tmp_path):
        config = SandboxConfig(mode=SandboxMode.DISABLED)
        validator = CommandValidator(config)
        assert validator.validate("anything ; goes") == (True, None)


# ============================================================
#  Inline-code flag bypasses (AC1/AC2) and README paths (AC5)
# ============================================================

# The four bypass forms: the code-flag family must be rejected wherever it
# appears in the argument vector (not only at parts[1]), in `--flag=value`
# spelling, and attached to its payload as a single token after shlex.split.
BYPASS_INLINE = [
    "python3 -O -c 'print(1)'",
    "python3 -X importtime -c 'print(1)'",
    "node --eval=process.exit(0)",
    'python3 -c"print(1)"',
]

# Regression: ordinary commands must keep passing every gate.
ALLOWED_COMMANDS = [
    "python script.py",
    "python -m pytest",
    "pip install requests",
    "ls -la",
    "cat file.txt",
    "grep -r pattern .",
    'git commit -m "a; b"',
    "grep -E 'x|y'",
]

README = Path(__file__).resolve().parent.parent / "README.md"


class TestInlineCodeFlagBypasses:
    """Each verified bypass form must be reported unsafe by CommandExecutor,
    BashTool and the sandbox CommandValidator while safe mode / an allowlist
    applies, and by TaskTool.run_background (which has no allowlist)."""

    @pytest.mark.parametrize("command", BYPASS_INLINE)
    def test_bypass_blocked_by_all_entry_points(
        self, command, executor, bash_tool, sandbox_validator
    ):
        is_safe, reason = executor._is_command_safe(command)
        assert not is_safe, f"CommandExecutor allowed: {command}"
        assert "inline code execution" in reason.lower()

        result = bash_tool.execute(command)
        assert not result.success, f"BashTool allowed: {command}"
        assert "Blocked inline code execution" in (result.error or "")

        is_valid, reason = sandbox_validator.validate(command)
        assert not is_valid, f"CommandValidator allowed: {command}"
        assert reason and "inline code execution" in reason

    @pytest.mark.parametrize("command", BYPASS_INLINE)
    def test_bypass_blocked_in_task_tool(self, command, task_tool):
        result = task_tool.run_background(command)
        assert not result.success, f"TaskTool allowed: {command}"
        assert "Blocked inline code execution" in (result.error or "")


class TestAllowedCommandsRegression:
    """AC2: legitimate commands must keep passing every gate."""

    @pytest.mark.parametrize("command", ALLOWED_COMMANDS)
    def test_allowed_by_all_entry_points(
        self, command, executor, bash_tool, sandbox_validator, monkeypatch
    ):
        is_safe, reason = executor._is_command_safe(command)
        assert is_safe, f"CommandExecutor rejected {command}: {reason}"

        is_valid, reason = sandbox_validator.validate(command)
        assert is_valid, f"CommandValidator rejected {command}: {reason}"

        # Stub the subprocess so the test never runs pip/git/pytest for real —
        # only the safety layer is under test here.
        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        monkeypatch.setattr(
            "tools.unified_tools.subprocess.run",
            lambda *args, **kwargs: _Completed(),
        )
        result = bash_tool.execute(command)
        assert result.success, f"BashTool rejected {command}: {result.error}"
        assert "Blocked" not in (result.error or "")


def _readme_module_paths(text: str) -> set:
    """Module paths referenced by README.md.

    Sources: every directory-qualified ``*.py`` path in the file, every
    backticked code span (``orchestrator/workflow.py``, ``python main.py``),
    and the Project Structure tree, where bare entries such as
    ``base_agent.py`` resolve against the top-level folder they hang from.
    """
    paths = set()

    # Directory-qualified module paths, anywhere in the file.
    paths.update(re.findall(r"[\w-]+/[\w./-]+\.py", text))

    # Backticked code spans (inline spans only — fenced blocks have none).
    for span in re.findall(r"`[^`\n]+`", text):
        paths.update(re.findall(r"[\w./-]+\.py", span))

    # Project Structure tree block.
    marker = text.find("## Project Structure")
    if marker == -1:
        return paths
    block = text[marker:]
    fence = block.find("```")
    if fence == -1:
        return paths
    block = block[fence + 3:]
    end = block.find("```")
    if end != -1:
        block = block[:end]

    top = None
    for line in block.splitlines():
        match = re.match(r"^((?:│   )*)(?:├──|└──) (.+)$", line)
        if not match:
            continue
        depth = len(match.group(1)) // 4
        entry = match.group(2).split("#", 1)[0].strip()
        if depth == 0:
            if "/" in entry:
                top = entry.rstrip("/")
            else:
                paths.update(re.findall(r"[\w./-]+\.py", entry))
        elif top:
            paths.update(
                f"{top}/{name}" for name in re.findall(r"[\w.-]+\.py", entry)
            )
    return paths


class TestReadmeModuleReferences:
    """AC5: README.md must not reference a module path that does not exist."""

    def test_readme_module_paths_exist(self):
        text = README.read_text(encoding="utf-8")
        missing = sorted(
            path
            for path in _readme_module_paths(text)
            if not (README.parent / path).exists()
        )
        assert not missing, f"README references missing modules: {missing}"

    def test_readme_has_no_stale_usage_tracker_reference(self):
        # README used to document a usage tracker under utils/ — that module
        # (utils/usage_tracker.py) never existed; the real module is
        # utils/cost_tracker.py.
        text = README.read_text(encoding="utf-8")
        assert "usage_tracker" not in text
        assert "usage tracker" not in text.lower()
