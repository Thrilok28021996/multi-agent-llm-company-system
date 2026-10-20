"""Tests for command executor security — verifying the inline code execution bypass is blocked."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from tools.command_executor import CommandExecutor


@pytest.fixture
def executor():
    return CommandExecutor(workspace_root=".", safe_mode=True)


@pytest.fixture
def unsafe_executor():
    return CommandExecutor(workspace_root=".", safe_mode=False)


class TestInlineCodeExecutionBlocked:
    """The critical security fix: LLM cannot smuggle arbitrary code through
    `python -c "..."` which passes the allowlist."""

    @pytest.mark.parametrize("interpreter", ["python", "python3", "node", "npx"])
    def test_python_c_flag_blocked(self, executor, interpreter):
        """Interpreter -c 'code' must be blocked (inline code execution)."""
        is_safe, reason = executor._is_command_safe(f"{interpreter} -c 'import os'")
        assert not is_safe
        assert "inline code execution" in reason.lower()

    @pytest.mark.parametrize("interpreter", ["ruby", "perl", "bash", "sh"])
    def test_interpreter_not_in_allowlist(self, executor, interpreter):
        """Interpreters not in SAFE_COMMANDS are blocked entirely — even more secure."""
        is_safe, reason = executor._is_command_safe(f"{interpreter} -c 'code'")
        assert not is_safe
        assert "not in allowed list" in reason.lower()

    @pytest.mark.parametrize("flag", ["-c", "-e", "--eval", "-r"])
    def test各种_flags_blocked(self, executor, flag):
        """All inline-code flags must be blocked."""
        is_safe, reason = executor._is_command_safe(f"python {flag} 'print(1)'")
        assert not is_safe
        assert "inline code execution" in reason.lower()

    def test_python_script_file_allowed(self, executor):
        """python script.py (file execution) must still be allowed."""
        is_safe, reason = executor._is_command_safe("python script.py")
        assert is_safe
        assert reason == "OK"

    def test_python_module_flag_allowed(self, executor):
        """python -m pytest must still be allowed (not inline code)."""
        is_safe, reason = executor._is_command_safe("python -m pytest")
        assert is_safe

    def test_pip_install_allowed(self, executor):
        """pip install must still be allowed."""
        is_safe, reason = executor._is_command_safe("pip install requests")
        assert is_safe

    def test_ls_cat_grep_allowed(self, executor):
        """Basic read-only commands must still work."""
        for cmd in ["ls -la", "cat file.txt", "grep -r pattern .", "find . -name '*.py'"]:
            is_safe, reason = executor._is_command_safe(cmd)
            assert is_safe, f"Expected safe: {cmd}"

    def test_rm_rf_blocked(self, executor):
        """Dangerous commands must still be blocked."""
        is_safe, reason = executor._is_command_safe("rm -rf /")
        assert not is_safe
        assert "blocked" in reason.lower()

    def test_unsafe_mode_allows_inline(self, unsafe_executor):
        """With safe_mode=False, inline code should be allowed."""
        is_safe, reason = unsafe_executor._is_command_safe("python -c 'print(1)'")
        assert is_safe

    @pytest.mark.parametrize(
        "command",
        [
            "python3 -O -c 'print(1)'",
            "python3 -X importtime -c 'print(1)'",
            "node --eval=process.exit(0)",
            'python3 -c"print(1)"',
        ],
    )
    def test_flag_bypass_forms_blocked(self, executor, command):
        """The code-flag family is rejected wherever it appears: after other
        flags (`-O`, `-X importtime`), in `--flag=value` spelling, and
        attached to its payload as one shlex token."""
        is_safe, reason = executor._is_command_safe(command)
        assert not is_safe, command
        assert "inline code execution" in reason.lower()

    def test_node_e_flag_blocked(self, executor):
        """node -e 'require(...)' must be blocked."""
        is_safe, reason = executor._is_command_safe("node -e 'require(\"child_process\").exec(\"ls\")'")
        assert not is_safe
        assert "inline code execution" in reason.lower()
