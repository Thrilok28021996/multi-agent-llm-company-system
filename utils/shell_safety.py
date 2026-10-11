"""Quote-aware helpers for validating shell commands.

Shared by ``tools.command_executor.CommandExecutor``, ``tools.unified_tools.BashTool``
and the sandbox ``CommandValidator`` so every command execution path applies the
same rules:

* Shell chaining, pipes, redirection, backgrounding and command substitution
  are rejected — they execute commands that never touch the allowlist.
* Path arguments must stay inside the configured workspace, so a allowlisted
  command cannot be pointed at files outside the project.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

# Characters that chain, pipe, background or redirect commands. Any of them
# lets the shell run a second command that was never allowlisted.
_SHELL_OPERATORS = ";|&<>"


def find_shell_operator(command: str) -> str | None:
    """Return a description of the first unsafe shell construct, or ``None``.

    The scan is quote aware: ``git commit -m "a; b"`` and ``grep -E 'x|y'``
    stay legal, while ``ls ; rm -rf /`` or ``ls | sh`` are rejected. Backticks
    and ``$(...)`` are rejected even inside double quotes because the shell
    evaluates them before the command ever runs.
    """
    index = 0
    length = len(command)
    in_single = False
    in_double = False

    while index < length:
        char = command[index]

        if in_single:
            # Single quotes are literal in POSIX shells.
            if char == "'":
                in_single = False
            index += 1
            continue

        if in_double:
            if char == "\\":
                index += 2
                continue
            if char == '"':
                in_double = False
                index += 1
                continue
            if char == "`":
                return "command substitution (backtick)"
            if command.startswith("$(", index):
                return "command substitution ($(...))"
            index += 1
            continue

        if char == "'":
            in_single = True
            index += 1
            continue
        if char == '"':
            in_double = True
            index += 1
            continue
        if char == "\\":
            index += 2
            continue
        if char in _SHELL_OPERATORS:
            return f"shell operator ({char})"
        if char in "\r\n":
            return "shell operator (newline)"
        if char == "`":
            return "command substitution (backtick)"
        if command.startswith("$(", index):
            return "command substitution ($(...))"
        index += 1

    return None


# Inline-code flag spellings accepted by the allowlisted interpreters
# (``-c``/``-e``/``-r`` payloads for the interpreter family, ``--eval=``
# in node's ``--flag=value`` spelling).
_INLINE_CODE_FLAGS = ("-c", "-e", "--eval", "-r")

# python options that take their value from the FOLLOWING argv token when
# spelled as a bare option (``-X importtime``, ``-W ignore``,
# ``--check-hash-based-pycs never``). With an attached value
# (``-Ximporttime``, ``-Xopt=value``, ``-Wall``) CPython reads the value
# from the same token and consumes nothing after it — verified live against
# CPython 3.12: ``python3 -W -m -c 'print(1)'`` reports the ``-m`` token as
# an invalid -W action (it was consumed as -W's value) and then runs the
# ``-c`` payload, while ``python3 -Wall -c 'print(1)'`` runs the payload
# directly.
_PYTHON_VALUE_TAKING_OPTIONS = ("-X", "-W", "--check-hash-based-pycs")


def _find_inline_code_flag(args: list[str], base_cmd: str) -> str | None:
    """Return the inline-code flag found in ``args``, or ``None``.

    ``args`` must be the interpreter's own arguments and ``base_cmd`` the
    interpreter they were parsed for. This is the SINGLE implementation of
    the scan, shared by every guarded command path (``CommandExecutor``,
    ``BashTool``, ``TaskTool`` and the sandbox ``CommandValidator``).

    The code-flag family is rejected wherever it appears in the
    interpreter's own argv: exact (``-c``, ``-e``, ``--eval``, ``-r``),
    in ``--flag=value`` spelling (``node --eval=<code>``), and attached to
    its payload (``python3 -c"<code>"`` / ``perl -e'<code>'`` leave
    ``-c<code>`` / ``-e<code>`` as a single token after ``shlex.split``).

    For python-family interpreters (``base_cmd`` starting with ``python``)
    two rules model how CPython itself parses its own argv:

    * ``-X``, ``-W`` and ``--check-hash-based-pycs`` take their value from
      the FOLLOWING token when spelled as a bare option, so that token is
      skipped: in ``python3 -X -m -c 'print(1)'`` the ``-m`` is the value
      of ``-X`` (python then runs the ``-c`` payload — verified live), never
      the module boundary, and the later ``-c`` is still rejected.
      Attached spellings (``-Ximporttime``, ``-Xopt=value``, ``-Wall``)
      carry the value in the same token and consume nothing after it.

    * ``-m`` in flag position — the exact ``-m`` token or the attached
      ``-m<module>`` form python's getopt accepts, after any value token
      above has been skipped — ends the scan: everything after it belongs
      to the module's argument vector, not to python's
      (``python3 -m pytest -c pytest.ini`` and ``python3 -m pip install
      -e .`` run the module, not python's own flags). A code flag before
      the boundary is still python's own and stays rejected
      (``python3 -c '<code>' -m pytest``).

    The boundary is python-specific: ``-m`` is not a module hand-off for
    the other interpreters in the allowlisted set (for ``bash``/``sh`` it
    selects job-control mode), so their arguments are always scanned in
    full.
    """
    is_python_family = base_cmd.startswith("python")
    index = 0
    while index < len(args):
        arg = args[index]

        if is_python_family and arg in _PYTHON_VALUE_TAKING_OPTIONS:
            # The next token is this option's VALUE, not a python flag:
            # skip it so a value such as `-m` or `-c` is never mistaken for
            # the module boundary or a code flag (D1).
            index += 2
            continue

        if is_python_family and arg.startswith("-m"):
            # Module hand-off: the rest of the argv is the module's.
            return None

        if arg in _INLINE_CODE_FLAGS:
            return arg
        if arg.startswith("--eval="):
            return "--eval=..."
        if arg.startswith("-c") and len(arg) > 2:
            return "-c..."
        if arg.startswith("-e") and len(arg) > 2:
            return "-e..."
        index += 1

    return None


def _is_path_token(token: str) -> bool:
    """True when a token looks like a filesystem path worth validating."""
    if not token or "://" in token:
        return False
    if token == "..":
        return True
    return (
        token.startswith(("/", "./", "../", "~/"))
        or "/../" in token
        or token.endswith("/..")
    )


def find_path_escape(command: str, workspace_root: Path) -> str | None:
    """Return a description of the first path argument outside ``workspace_root``.

    Relative paths are resolved against the workspace root (the default working
    directory), ``~`` is expanded, and ``--flag=/path`` style arguments are
    inspected too. Returns ``None`` when every path stays inside the workspace.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        return "unparseable command (unbalanced quotes)"

    root = Path(workspace_root).resolve()
    for token in tokens:
        candidates = [token]
        if token.startswith("-") and "=" in token:
            candidates = [token.split("=", 1)[1]]

        for candidate in candidates:
            if not _is_path_token(candidate):
                continue
            path = (
                Path(os.path.expanduser(candidate))
                if candidate.startswith("~")
                else Path(candidate)
            )
            if not path.is_absolute():
                path = root / path
            resolved = path.resolve()
            if resolved != root and root not in resolved.parents:
                return f"path outside workspace: {candidate}"

    return None


def is_within_workspace(path: Path, workspace_root: Path) -> bool:
    """True when ``path`` resolves inside (or to) ``workspace_root``."""
    resolved = Path(path).resolve()
    root = Path(workspace_root).resolve()
    return resolved == root or root in resolved.parents
