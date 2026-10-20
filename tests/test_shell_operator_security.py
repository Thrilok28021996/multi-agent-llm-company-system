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


# ============================================================
#  F1 regression: the code-flag scan stops at the -m boundary
# ============================================================

# Once the command is `python3 -m <tool> ...`, every later argument belongs
# to <tool>: pytest's -c/-r and pip's -e/-c are the tool's own flags, not
# python's inline-code flags. All of these were allowed at base 63c859d and
# must be allowed at every gate again.
MODULE_FLAG_ALLOWED = [
    "python3 -m pytest -c pytest.ini",
    "python3 -m pytest -r a",
    "python3 -m pip install -c constraints.txt",
    "python3 -m pip install -e .",
    'python3 -m pip install -e ".[dev]"',
    "python -m pytest",
    "python3 -m http.server",
    "python3 -m pytest -q",
    "python3 -m pytest -v",
    # Same hand-off spelled the way python's getopt also accepts it.
    "python3 -mpytest -c pytest.ini",
]

# A code flag BEFORE the -m boundary is still the interpreter's own flag
# (python has not handed over to a module yet) — these stay blocked.
CODE_FLAG_BEFORE_M = [
    "python3 -c 'print(1)' -m pytest",
    "python3 -O -c 'print(1)' -m pytest",
]


class TestModuleFlagBoundary:
    """F1: ``python3 -m <tool> ...`` must not be read as inline code.

    The ``-m`` boundary is applied identically at all four gates:
    CommandExecutor._is_command_safe, BashTool.execute,
    TaskTool.run_background and the sandbox CommandValidator.validate.
    """

    @pytest.mark.parametrize("command", MODULE_FLAG_ALLOWED)
    def test_module_flag_command_allowed_by_all_entry_points(
        self, command, executor, bash_tool, sandbox_validator, task_tool, monkeypatch
    ):
        # Gate 1 — CommandExecutor
        is_safe, reason = executor._is_command_safe(command)
        assert is_safe, f"CommandExecutor rejected {command}: {reason}"

        # Gate 2 — sandbox CommandValidator
        is_valid, reason = sandbox_validator.validate(command)
        assert is_valid, f"CommandValidator rejected {command}: {reason}"

        # Gate 3 — BashTool (subprocess stubbed: only the safety layer runs)
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

        # Gate 4 — TaskTool (background runner stubbed: nothing is spawned,
        # only the gate that rejects/accepts the command is under test)
        monkeypatch.setattr(
            TaskTool,
            "_execute",
            lambda self, task, work_dir, timeout: None,
        )
        result = task_tool.run_background(command)
        assert result.success, f"TaskTool rejected {command}: {result.error}"
        assert "Blocked" not in (result.error or "")

    @pytest.mark.parametrize("command", CODE_FLAG_BEFORE_M)
    def test_code_flag_before_m_blocked_by_all_entry_points(
        self, command, executor, bash_tool, sandbox_validator, task_tool
    ):
        # Gate 1 — CommandExecutor
        is_safe, reason = executor._is_command_safe(command)
        assert not is_safe, f"CommandExecutor allowed: {command}"
        assert "inline code execution" in reason.lower()

        # Gate 2 — sandbox CommandValidator
        is_valid, reason = sandbox_validator.validate(command)
        assert not is_valid, f"CommandValidator allowed: {command}"
        assert reason and "inline code execution" in reason

        # Gate 3 — BashTool
        result = bash_tool.execute(command)
        assert not result.success, f"BashTool allowed: {command}"
        assert "Blocked inline code execution" in (result.error or "")

        # Gate 4 — TaskTool (rejected before anything is spawned)
        result = task_tool.run_background(command)
        assert not result.success, f"TaskTool allowed: {command}"
        assert "Blocked inline code execution" in (result.error or "")


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


# ============================================================
#  D1 (fix cycle 2): the -m boundary must respect python's
#  value-taking options, from ONE shared implementation
# ============================================================

# CPython parses `-X`, `-W` and `--check-hash-based-pycs` with getopt
# semantics: a BARE `-X`/`-W` consumes the following token as its value
# (verified live on CPython 3.12.13 — `python3 -W -m -c 'print(1)'`
# reports "-m" as an invalid -W action and then RUNS the payload). The old
# scan ended at any token starting with "-m", so the swallowed "-m" opened
# the module boundary and "-c" was never reached. Each of these runs a
# `-c` payload in python's own argv and must be BLOCKED at all four gates.
D1_VALUE_SMUGGLING_BLOCKED = [
    "python3 -X -m -c 'print(1)'",
    "python3 -W -m -c 'print(1)'",
    "python3 -X -mpytest -c 'print(1)'",
    # Attached/equals spellings consume nothing, so `-c` is python's own flag.
    "python3 -X=importtime -c 'print(1)'",
    "python3 -W action=mod -c 'print(1)'",
]

# The F1 cases that must NOT regress: after a real `-m` module hand-off the
# remaining flags belong to the module (pytest's -c config file, pip's -e
# extras), so they stay ALLOWED at all four gates.
D1_MODULE_HANDOFF_ALLOWED = [
    "python3 -m pytest -c pytest.ini",
    "python3 -m pip install -e .",
    'python3 -m pip install -e ".[dev]"',
    "python3 -mpytest -c pytest.ini",
]


class TestD1ValueTakingOptionSmuggling:
    """D1: `-X`/`-W` swallow a following `-m`, which must not open the
    module boundary — all four gates must block the payload."""

    @pytest.mark.parametrize("command", D1_VALUE_SMUGGLING_BLOCKED)
    def test_blocked_by_all_entry_points(
        self, command, executor, bash_tool, sandbox_validator, task_tool
    ):
        # Gate 1 — CommandExecutor
        is_safe, reason = executor._is_command_safe(command)
        assert not is_safe, f"CommandExecutor allowed: {command}"
        assert "inline code execution" in reason.lower()

        # Gate 2 — sandbox CommandValidator
        is_valid, reason = sandbox_validator.validate(command)
        assert not is_valid, f"CommandValidator allowed: {command}"
        assert reason and "inline code execution" in reason

        # Gate 3 — BashTool (rejected before any subprocess runs)
        result = bash_tool.execute(command)
        assert not result.success, f"BashTool allowed: {command}"
        assert "Blocked inline code execution" in (result.error or "")

        # Gate 4 — TaskTool (rejected before anything is spawned)
        result = task_tool.run_background(command)
        assert not result.success, f"TaskTool allowed: {command}"
        assert "Blocked inline code execution" in (result.error or "")

    @pytest.mark.parametrize("command", D1_MODULE_HANDOFF_ALLOWED)
    def test_module_handoff_still_allowed_by_all_entry_points(
        self, command, executor, bash_tool, sandbox_validator, task_tool, monkeypatch
    ):
        # Gate 1 — CommandExecutor
        is_safe, reason = executor._is_command_safe(command)
        assert is_safe, f"CommandExecutor rejected {command}: {reason}"

        # Gate 2 — sandbox CommandValidator
        is_valid, reason = sandbox_validator.validate(command)
        assert is_valid, f"CommandValidator rejected {command}: {reason}"

        # Gate 3 — BashTool (subprocess stubbed: only the safety layer runs)
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

        # Gate 4 — TaskTool (background runner stubbed: only the gate runs)
        monkeypatch.setattr(
            TaskTool,
            "_execute",
            lambda self, task, work_dir, timeout: None,
        )
        result = task_tool.run_background(command)
        assert result.success, f"TaskTool rejected {command}: {result.error}"
        assert "Blocked" not in (result.error or "")


# ============================================================
#  [A] (fix cycle 3): getopt-style clustered short options
# ============================================================

# Reviewer finding [A] (2026-10-11, pre-existing at base 63c859d): CPython
# parses short-option clusters letter-by-letter, so the code flag need not
# be the token's FIRST character. Live-verified on CPython 3.12.13
# (/Volumes/personal/conda_envs/assi/bin/python3.12): every form below
# EXECUTES a `-c` payload — `-Oc`/`-vOc`/`-Bc`/`-Oc<code>` all print 1,
# and `-cO` raises NameError on the executed code "O" — so each must be
# BLOCKED at all four gates.
CLUSTER_CODE_FLAG_BLOCKED = [
    "python3 -Oc 'print(1)'",
    "python3 -vOc 'print(1)'",
    "python3 -Bc 'print(1)'",
    "python3 -Oc'print(1)'",
    # `c` in flag position after another plain flag letter; live CPython
    # executes the code that follows inside the same cluster token.
    "python3 -cO",
]

# Live-CPython-backed ALLOW cases (every entry proven non-executing against
# the same interpreter). `-m` inside a cluster is the module hand-off
# (`python3 -Om pytest --version` prints "pytest 9.0.3" live), and a
# value-taking `X`/`W` inside a cluster consumes the REST OF THE TOKEN as
# its value (getopt value-nesting): `python3 -OXc foo` opens `foo` as a
# SCRIPT file live ("can't open file .../foo", never code execution), so
# the `c` after `X` must not be read as a code flag.
CLUSTER_MODULE_OR_VALUE_ALLOWED = [
    "python3 -Om pytest --version",
    # module args after a cluster hand-off are never scanned as python flags
    "python3 -Om pytest -c pytest.ini",
    "python3 -Om pytest -r a",
    # getopt value-nesting: c is inside -X's value, foo is a script name
    "python3 -OXc foo",
    "python3 -XOc foo",
    "python3 -Xc foo",
    "python3 -Wc foo",
    "python3 -WOc foo",
    # bare -W swallows the next token (-Om) as its value (D1 behaviour;
    # live: "Invalid -W option ignored ... '-Om'" then pytest is opened as
    # a script file) — no inline code executes, so this stays allowed.
    "python3 -W -Om pytest",
]


class TestClusteredShortOptionBypass:
    """[A]: a code flag inside a getopt short-option cluster must be
    rejected, while a cluster module hand-off and getopt value-nesting
    must not over-block. Applied identically at all four gates."""

    @pytest.mark.parametrize("command", CLUSTER_CODE_FLAG_BLOCKED)
    def test_cluster_code_flag_blocked_by_all_entry_points(
        self, command, executor, bash_tool, sandbox_validator, task_tool
    ):
        # Gate 1 — CommandExecutor
        is_safe, reason = executor._is_command_safe(command)
        assert not is_safe, f"CommandExecutor allowed: {command}"
        assert "inline code execution" in reason.lower()

        # Gate 2 — sandbox CommandValidator
        is_valid, reason = sandbox_validator.validate(command)
        assert not is_valid, f"CommandValidator allowed: {command}"
        assert reason and "inline code execution" in reason

        # Gate 3 — BashTool (rejected before any subprocess runs)
        result = bash_tool.execute(command)
        assert not result.success, f"BashTool allowed: {command}"
        assert "Blocked inline code execution" in (result.error or "")

        # Gate 4 — TaskTool (rejected before anything is spawned)
        result = task_tool.run_background(command)
        assert not result.success, f"TaskTool allowed: {command}"
        assert "Blocked inline code execution" in (result.error or "")

    @pytest.mark.parametrize("command", CLUSTER_MODULE_OR_VALUE_ALLOWED)
    def test_cluster_module_and_value_nesting_allowed_by_all_entry_points(
        self, command, executor, bash_tool, sandbox_validator, task_tool, monkeypatch
    ):
        # Gate 1 — CommandExecutor
        is_safe, reason = executor._is_command_safe(command)
        assert is_safe, f"CommandExecutor rejected {command}: {reason}"

        # Gate 2 — sandbox CommandValidator
        is_valid, reason = sandbox_validator.validate(command)
        assert is_valid, f"CommandValidator rejected {command}: {reason}"

        # Gate 3 — BashTool (subprocess stubbed: only the safety layer runs)
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

        # Gate 4 — TaskTool (background runner stubbed: only the gate runs)
        monkeypatch.setattr(
            TaskTool,
            "_execute",
            lambda self, task, work_dir, timeout: None,
        )
        result = task_tool.run_background(command)
        assert result.success, f"TaskTool rejected {command}: {result.error}"
        assert "Blocked" not in (result.error or "")


class TestInlineCodeScanHasSingleSourceOfTruth:
    """The dedupe acceptance criterion: exactly ONE definition of the scan
    in the whole repository, and every guarded command path resolves to it.
    Both checks are structural — a future hand-copied re-implementation
    fails this suite even if its behaviour is identical."""

    def test_exactly_one_definition_in_the_repository(self):
        # The marker is assembled at RUNTIME from two parts: adjacent string
        # literals are constant-folded at compile time, which would leak the
        # pattern into this file's .pyc and add noise to the repo-wide grep
        # acceptance check for the scan's definition (whose output must stay
        # exactly one line, in source and in bytecode).
        scan_keyword, scan_name = "def", "_find_inline_code_flag"
        definition_marker = scan_keyword + " " + scan_name
        repo_root = Path(__file__).resolve().parent.parent
        definitions = []
        for path in sorted(repo_root.rglob("*.py")):
            if "__pycache__" in path.parts or ".git" in path.parts:
                continue
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if definition_marker in line:
                    definitions.append(f"{path.relative_to(repo_root)}:{lineno}")
        assert len(definitions) == 1, (
            "expected exactly one definition of the inline-code scan, found "
            f"{len(definitions)}: {definitions}"
        )
        assert definitions[0].startswith("utils/shell_safety.py:"), definitions[0]

    def test_all_four_gates_resolve_to_the_shared_function(self):
        """Import identity: CommandExecutor, BashTool, TaskTool and the
        sandbox CommandValidator all use utils.shell_safety's function."""
        import tools.command_executor as command_executor
        import tools.unified_tools as unified_tools
        import utils.sandbox as sandbox
        import utils.shell_safety as shell_safety

        shared = shell_safety._find_inline_code_flag
        assert shared.__module__ == "utils.shell_safety"

        # Gate modules resolve the name from their module namespace —
        # identical object, not an equal copy.
        gates = {
            "CommandExecutor": command_executor,
            "BashTool/TaskTool": unified_tools,
            "CommandValidator": sandbox,
        }
        for gate, module in gates.items():
            assert getattr(module, "_find_inline_code_flag", None) is shared, (
                f"{gate} does not use utils.shell_safety's "
                f"_find_inline_code_flag (a re-copy has been introduced)"
            )

    def test_command_validator_gate_calls_the_shared_binding(self, monkeypatch):
        """Behavioural proof for the fourth gate: CommandValidator.validate
        resolves the scan from its module namespace (where the shared
        function was imported). Swap that binding for a canary and show the
        validator actually routes through it — combined with the identity
        check above, the gate provably calls utils.shell_safety's function."""
        import utils.sandbox as sandbox
        import utils.shell_safety as shell_safety

        # The binding the gate reads IS the shared function.
        assert sandbox._find_inline_code_flag is shell_safety._find_inline_code_flag

        seen = []

        def _canary(args, base_cmd):
            seen.append((list(args), base_cmd))
            return None

        monkeypatch.setattr(sandbox, "_find_inline_code_flag", _canary)

        config = SandboxConfig(
            mode=SandboxMode.STANDARD,
            allowed_commands=["python3"],
            working_dir=str(Path.cwd()),
        )
        validator = CommandValidator(config)
        is_valid, reason = validator.validate("python3 -m pytest -c pytest.ini")
        assert is_valid, reason
        assert seen == [(["-m", "pytest", "-c", "pytest.ini"], "python3")]
