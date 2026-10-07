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
