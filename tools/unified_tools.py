"""
Unified Tools for Company AGI - Best of GranularTools + EnhancedTools.

This module combines the best implementations from both toolsets into
a single, comprehensive toolkit matching Claude Code's capabilities.

Tools included:
- ReadTool: File reading with line numbers (from Granular)
- EditTool: Surgical edits with uniqueness check (from Granular)
- MultiEditTool: Atomic batch edits with rollback (from Enhanced)
- WriteTool: Safe file writing with backup (from Granular)
- GlobTool: Pattern matching sorted by mtime (from Granular)
- GrepTool: Regex search with context (from Granular)
- BashTool: Safe command execution (from Granular)
- WebFetchTool: URL fetching with extraction (from Enhanced)
- WebSearchTool: DuckDuckGo search (from Enhanced)
- TodoTool: Task management (from Enhanced)
- NotebookTool: Jupyter editing (from Enhanced)
- TaskTool: Background execution (from Enhanced)
- LSPTool: Code intelligence (Enhanced with Granular fallback)
"""

import asyncio
import json
import re
import shlex
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

from utils.shell_safety import find_path_escape, find_shell_operator, is_within_workspace

# Web dependencies are optional
HAS_WEB_DEPS = False
try:
    import aiohttp
    from bs4 import BeautifulSoup
    HAS_WEB_DEPS = True
except ImportError:
    pass

if TYPE_CHECKING:
    import aiohttp
    from bs4 import BeautifulSoup


# ============================================================
#  TOOL RESULT
# ============================================================

@dataclass
class ToolResult:
    """Unified result from any tool operation."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.success


# ============================================================
#  FILE READING (ReadTool)
# ============================================================

class ReadTool:
    """
    Read files with line numbers, offset/limit support.

    Features:
    - Line numbers in output (cat -n style)
    - Offset and limit for large files
    - Truncation of long lines
    - Binary file detection
    """

    MAX_LINES = 2000
    MAX_LINE_LENGTH = 2000

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = Path(workspace_root).resolve()

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: Optional[int] = None
    ) -> ToolResult:
        """Read a file with line numbers."""
        try:
            path = self._resolve_path(file_path)

            if not path.exists():
                return ToolResult(success=False, error=f"File not found: {file_path}")

            if path.is_dir():
                return ToolResult(success=False, error=f"Cannot read directory: {file_path}")

            # Read file content
            content = path.read_text(encoding='utf-8', errors='replace')
            lines = content.splitlines()

            total_lines = len(lines)
            limit = limit or self.MAX_LINES

            # Apply offset and limit
            start = min(offset, total_lines)
            end = min(start + limit, total_lines)
            selected_lines = lines[start:end]

            # Format with line numbers (1-based)
            formatted_lines = []
            max_width = len(str(end))

            for i, line in enumerate(selected_lines, start=start + 1):
                if len(line) > self.MAX_LINE_LENGTH:
                    line = line[:self.MAX_LINE_LENGTH] + "... [truncated]"
                formatted_lines.append(f"{str(i).rjust(max_width)}→{line}")

            return ToolResult(
                success=True,
                data="\n".join(formatted_lines),
                metadata={
                    "file_path": str(path),
                    "total_lines": total_lines,
                    "lines_returned": len(selected_lines),
                    "offset": start,
                    "truncated": end < total_lines
                }
            )

        except UnicodeDecodeError:
            return ToolResult(success=False, error=f"Cannot read binary file: {file_path}")
        except PermissionError:
            return ToolResult(success=False, error=f"Permission denied: {file_path}")
        except Exception as e:
            return ToolResult(success=False, error=f"Error reading file: {str(e)}")

    def _resolve_path(self, file_path: str) -> Path:
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        return path.resolve()


# ============================================================
#  FILE EDITING (EditTool)
# ============================================================

class EditTool:
    """
    Surgical edit tool for precise string replacements.

    Features:
    - Exact string matching
    - Uniqueness validation
    - Replace all option
    """

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = Path(workspace_root).resolve()

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False
    ) -> ToolResult:
        """Perform surgical edit on a file."""
        try:
            path = self._resolve_path(file_path)

            if not path.exists():
                return ToolResult(success=False, error=f"File not found: {file_path}")

            content = path.read_text(encoding='utf-8')

            if old_string not in content:
                return ToolResult(
                    success=False,
                    error=f"String not found in file: '{old_string}'"
                )

            occurrences = content.count(old_string)

            if not replace_all and occurrences > 1:
                return ToolResult(
                    success=False,
                    error=f"String appears {occurrences} times. Provide more context or use replace_all=True."
                )

            if replace_all:
                new_content = content.replace(old_string, new_string)
                replacements = occurrences
            else:
                new_content = content.replace(old_string, new_string, 1)
                replacements = 1

            path.write_text(new_content, encoding='utf-8')

            return ToolResult(
                success=True,
                data={"replacements_made": replacements},
                metadata={"file_path": str(path), "file_size": len(new_content)}
            )

        except Exception as e:
            return ToolResult(success=False, error=f"Error editing file: {str(e)}")

    def _resolve_path(self, file_path: str) -> Path:
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        return path.resolve()


# ============================================================
#  MULTI-EDIT TOOL (atomic with rollback)
# ============================================================

class MultiEditTool:
    """
    Atomic batch edits with rollback support.

    Features:
    - Multiple edits in single operation
    - Automatic rollback on failure
    - Cross-file editing
    """

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = Path(workspace_root).resolve()

    def multi_edit(
        self,
        edits: List[Dict[str, Any]],
        atomic: bool = True
    ) -> ToolResult:
        """
        Perform multiple edits atomically.

        Args:
            edits: List of {file_path, old_string, new_string}
            atomic: Rollback all on failure
        """
        backups = {}
        completed = []

        try:
            # Create backups
            if atomic:
                for edit in edits:
                    path = self._resolve_path(edit["file_path"])
                    if path.exists() and str(path) not in backups:
                        backups[str(path)] = path.read_text(encoding='utf-8')

            # Perform edits
            for i, edit in enumerate(edits):
                path = self._resolve_path(edit["file_path"])

                if not path.exists():
                    raise FileNotFoundError(f"File not found: {edit['file_path']}")

                content = path.read_text(encoding='utf-8')
                old_string = edit["old_string"]
                new_string = edit["new_string"]

                if old_string not in content:
                    raise ValueError(f"Edit {i+1}: String not found: '{old_string}'")

                new_content = content.replace(old_string, new_string, 1)
                path.write_text(new_content, encoding='utf-8')
                completed.append(edit["file_path"])

            return ToolResult(
                success=True,
                data={"edits_completed": len(completed), "files_modified": list(set(completed))},
                metadata={"atomic": atomic}
            )

        except Exception as e:
            # Rollback on failure
            if atomic and backups:
                for path_str, content in backups.items():
                    Path(path_str).write_text(content, encoding='utf-8')

            return ToolResult(
                success=False,
                error=f"Multi-edit failed (rolled back): {str(e)}",
                metadata={"completed_before_failure": completed, "rolled_back": atomic}
            )

    def _resolve_path(self, file_path: str) -> Path:
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        return path.resolve()


# ============================================================
#  FILE WRITING (WriteTool)
# ============================================================

class WriteTool:
    """Safe file writing with directory creation and backup."""

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = Path(workspace_root).resolve()

    def write(
        self,
        file_path: str,
        content: str,
        create_dirs: bool = True,
        backup: bool = False
    ) -> ToolResult:
        """Write content to a file."""
        try:
            path = self._resolve_path(file_path)

            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)

            if backup and path.exists():
                backup_path = path.with_suffix(path.suffix + '.bak')
                backup_path.write_text(path.read_text())

            existed = path.exists()
            path.write_text(content, encoding='utf-8')

            return ToolResult(
                success=True,
                data={"bytes_written": len(content.encode('utf-8'))},
                metadata={"file_path": str(path), "created": not existed}
            )

        except Exception as e:
            return ToolResult(success=False, error=f"Error writing file: {str(e)}")

    def _resolve_path(self, file_path: str) -> Path:
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        return path.resolve()


# ============================================================
#  FILE SEARCH (GlobTool, GrepTool)
# ============================================================

class GlobTool:
    """Fast file pattern matching sorted by modification time."""

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = Path(workspace_root).resolve()

    def glob(
        self,
        pattern: str,
        path: Optional[str] = None,
        include_hidden: bool = False,
        max_results: int = 1000
    ) -> ToolResult:
        """Find files matching a glob pattern."""
        try:
            search_path = self._resolve_path(path) if path else self.workspace_root

            if not search_path.exists():
                return ToolResult(success=False, error=f"Directory not found: {path}")

            matches = []
            for match in search_path.glob(pattern):
                if not include_hidden:
                    parts = match.relative_to(search_path).parts
                    if any(p.startswith('.') for p in parts):
                        continue

                if match.is_file():
                    matches.append(match)

                if len(matches) >= max_results:
                    break

            # Sort by modification time (newest first)
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            relative_paths = [str(m.relative_to(self.workspace_root)) for m in matches]

            return ToolResult(
                success=True,
                data=relative_paths,
                metadata={"pattern": pattern, "total_matches": len(matches), "truncated": len(matches) >= max_results}
            )

        except Exception as e:
            return ToolResult(success=False, error=f"Error in glob: {str(e)}")

    def _resolve_path(self, file_path: str) -> Path:
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        return path.resolve()


# ============================================================
#  GREP TOOL
# ============================================================

class GrepTool:
    """Content search with regex and context lines."""

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = Path(workspace_root).resolve()

    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        file_pattern: Optional[str] = None,
        context_before: int = 0,
        context_after: int = 0,
        case_insensitive: bool = False,
        max_results: int = 100,
        output_mode: str = "content"  # "content", "files_only", "count"
    ) -> ToolResult:
        """Search for pattern in files."""
        try:
            search_path = self._resolve_path(path) if path else self.workspace_root
            flags = re.IGNORECASE if case_insensitive else 0

            try:
                regex = re.compile(pattern, flags)
            except re.error as e:
                return ToolResult(success=False, error=f"Invalid regex: {e}")

            results = []
            files_with_matches = set()
            total_matches = 0

            # Get files to search
            if search_path.is_file():
                files_to_search = [search_path]
            else:
                glob_pattern = file_pattern or "**/*"
                files_to_search = [
                    f for f in search_path.glob(glob_pattern)
                    if f.is_file() and not any(p.startswith('.') for p in f.parts)
                ]

            for file_path in files_to_search:
                if total_matches >= max_results:
                    break

                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    lines = content.splitlines()

                    for line_num, line in enumerate(lines, 1):
                        if regex.search(line):
                            total_matches += 1
                            files_with_matches.add(str(file_path))

                            if output_mode == "content":
                                start = max(0, line_num - 1 - context_before)
                                end = min(len(lines), line_num + context_after)

                                context_lines = []
                                for i in range(start, end):
                                    prefix = ">" if i == line_num - 1 else " "
                                    context_lines.append(f"{prefix} {i+1}: {lines[i]}")

                                results.append({
                                    "file": str(file_path.relative_to(self.workspace_root)),
                                    "line": line_num,
                                    "content": "\n".join(context_lines)
                                })

                            if total_matches >= max_results:
                                break

                except (UnicodeDecodeError, PermissionError):
                    continue

            # Return based on mode
            if output_mode == "files_only":
                return ToolResult(success=True, data=list(files_with_matches))
            elif output_mode == "count":
                return ToolResult(success=True, data={"matches": total_matches, "files": len(files_with_matches)})
            else:
                return ToolResult(
                    success=True,
                    data=results,
                    metadata={"total_matches": total_matches, "truncated": total_matches >= max_results}
                )

        except Exception as e:
            return ToolResult(success=False, error=f"Error in grep: {str(e)}")

    def _resolve_path(self, file_path: str) -> Path:
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        return path.resolve()


# ============================================================
#  BASH TOOL
# ============================================================

class BashTool:
    """Safe bash command execution with allowlist."""

    SAFE_COMMANDS = {
        'ls', 'cat', 'head', 'tail', 'grep', 'find', 'wc', 'sort', 'uniq',
        'echo', 'pwd', 'date', 'which', 'whoami', 'env', 'printenv',
        'git', 'npm', 'yarn', 'pnpm', 'pip', 'python', 'python3', 'node',
        'cargo', 'go', 'rustc', 'make', 'cmake',
        'docker', 'kubectl',
        'curl', 'wget',
        'jq', 'sed', 'awk',
        'pytest', 'jest', 'mocha',
        'black', 'ruff', 'eslint', 'prettier',
    }

    # Interpreters that accept inline code — LLM can smuggle arbitrary code
    _INTERPRETERS_INLINE = {"python", "python3", "node", "npx", "ruby", "perl", "bash", "sh"}

    def __init__(self, workspace_root: str = ".", safe_mode: bool = True):
        self.workspace_root = Path(workspace_root).resolve()
        self.safe_mode = safe_mode

    def execute(
        self,
        command: str,
        timeout: int = 120,
        cwd: Optional[str] = None
    ) -> ToolResult:
        """Execute a bash command."""
        try:
            if self.safe_mode:
                # shlex, not str.split(): a quoted flag such as
                # `python3 "-c" "print(1)"` must still be recognised as -c.
                try:
                    parts = shlex.split(command)
                except ValueError as e:
                    return ToolResult(
                        success=False,
                        error=f"Unparseable command: {e}"
                    )
                base_cmd = parts[0] if parts else ""
                if base_cmd not in self.SAFE_COMMANDS:
                    return ToolResult(
                        success=False,
                        error=f"Command '{base_cmd}' not in safe list. Disable safe_mode to run."
                    )
                # Block inline code execution — same bypass as CommandExecutor
                if base_cmd in self._INTERPRETERS_INLINE and len(parts) >= 2:
                    if parts[1] in ("-c", "-e", "--eval", "-r"):
                        return ToolResult(
                            success=False,
                            error=(
                                f"Blocked inline code execution: {base_cmd} {parts[1]}. "
                                "Write code to a file and execute it instead."
                            )
                        )
                # Shell chaining, pipes, redirection and substitution run
                # commands that never touch the allowlist.
                operator = find_shell_operator(command)
                if operator:
                    return ToolResult(
                        success=False,
                        error=(
                            f"Blocked {operator}. "
                            "Run a single command per call; chaining bypasses the allowlist."
                        )
                    )
                # Arguments must not point outside the configured workspace.
                escape = find_path_escape(command, self.workspace_root)
                if escape:
                    return ToolResult(success=False, error=f"Blocked {escape}.")

            work_dir = self._resolve_path(cwd) if cwd else self.workspace_root
            if not is_within_workspace(work_dir, self.workspace_root):
                return ToolResult(
                    success=False,
                    error=f"Working directory outside workspace: {cwd}"
                )

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(work_dir)
            )

            return ToolResult(
                success=result.returncode == 0,
                data={"stdout": result.stdout, "stderr": result.stderr, "return_code": result.returncode},
                metadata={"command": command, "cwd": str(work_dir)}
            )

        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error=f"Command timed out after {timeout}s")
        except Exception as e:
            return ToolResult(success=False, error=f"Error: {str(e)}")

    def _resolve_path(self, file_path: str) -> Path:
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        return path.resolve()


# ============================================================
#  WEB TOOLS (WebFetch, WebSearch)
# ============================================================

class WebFetchTool:
    """Fetch and process web content."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        self._cache: Dict[str, Tuple[str, float]] = {}
        self._cache_ttl = 900

    async def fetch(
        self,
        url: str,
        extract_mode: str = "text",
        use_cache: bool = True
    ) -> ToolResult:
        """Fetch content from URL."""
        if not HAS_WEB_DEPS:
            return ToolResult(success=False, error="Web dependencies not installed (aiohttp, beautifulsoup4)")

        # Check cache
        if use_cache and url in self._cache:
            content, ts = self._cache[url]
            if time.time() - ts < self._cache_ttl:
                return ToolResult(success=True, data=content, metadata={"cached": True})

        try:
            async with aiohttp.ClientSession() as session:  # type: ignore[union-attr]
                async with session.get(
                    url,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),  # type: ignore[union-attr]
                    allow_redirects=True
                ) as response:
                    if response.status != 200:
                        return ToolResult(success=False, error=f"HTTP {response.status}")

                    html = await response.text()

                    if extract_mode == "raw":
                        content = html
                    elif extract_mode == "markdown":
                        content = self._to_markdown(html)
                    else:
                        content = self._to_text(html)

                    if use_cache:
                        self._cache[url] = (content, time.time())

                    return ToolResult(
                        success=True,
                        data=content,
                        metadata={"url": str(response.url), "length": len(content)}
                    )

        except asyncio.TimeoutError:
            return ToolResult(success=False, error=f"Timeout fetching {url}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _to_text(self, html: str) -> str:
        soup = BeautifulSoup(html, 'html.parser')  # type: ignore[misc]
        for el in soup(['script', 'style', 'nav', 'footer', 'header']):
            el.decompose()
        return '\n'.join(line.strip() for line in soup.get_text().splitlines() if line.strip())

    def _to_markdown(self, html: str) -> str:
        soup = BeautifulSoup(html, 'html.parser')  # type: ignore[misc]
        for el in soup(['script', 'style', 'nav', 'footer']):
            el.decompose()

        result = []
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            tag_name = getattr(tag, 'name', None)
            if tag_name and len(tag_name) > 1:
                level = int(tag_name[1])
                result.append(f"{'#' * level} {tag.get_text(strip=True)}\n")

        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if text:
                result.append(f"{text}\n")

        return '\n'.join(result)


# ============================================================
#  WEB SEARCH TOOL
# ============================================================

class WebSearchTool:
    """Search the web using DuckDuckGo."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

    async def search(self, query: str, max_results: int = 10) -> ToolResult:
        """Search DuckDuckGo."""
        if not HAS_WEB_DEPS:
            return ToolResult(success=False, error="Web dependencies not installed")

        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

        try:
            async with aiohttp.ClientSession() as session:  # type: ignore[union-attr]
                async with session.get(
                    url,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)  # type: ignore[union-attr]
                ) as response:
                    if response.status != 200:
                        return ToolResult(success=False, error=f"HTTP {response.status}")

                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')  # type: ignore[misc]

                    results = []
                    for result in soup.select('.result')[:max_results]:
                        title = result.select_one('.result__title')
                        snippet = result.select_one('.result__snippet')
                        url_el = result.select_one('.result__url')

                        if title:
                            result_url = ""
                            if url_el:
                                result_url = url_el.get_text(strip=True)
                                if not result_url.startswith('http'):
                                    result_url = 'https://' + result_url

                            results.append({
                                "title": title.get_text(strip=True),
                                "url": result_url,
                                "snippet": snippet.get_text(strip=True) if snippet else ""
                            })

                    return ToolResult(success=True, data=results, metadata={"query": query})

        except Exception as e:
            return ToolResult(success=False, error=str(e))


# ============================================================
#  TODO TOOL
# ============================================================

class TodoStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


@dataclass
class TodoItem:
    content: str
    status: TodoStatus = TodoStatus.PENDING
    active_form: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "status": self.status.value,
            "activeForm": self.active_form,
            "created_at": self.created_at.isoformat()
        }


class TodoTool:
    """Task management and tracking."""

    def __init__(self, persist_path: Optional[str] = None):
        self.todos: List[TodoItem] = []
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self._load()

    def write(self, todos: List[Dict[str, Any]]) -> ToolResult:
        """Update todo list."""
        try:
            self.todos = []
            for item in todos:
                status = TodoStatus(item.get("status", "pending"))
                self.todos.append(TodoItem(
                    content=item["content"],
                    status=status,
                    active_form=item.get("activeForm", item["content"])
                ))

            if self.persist_path:
                self._save()

            return ToolResult(
                success=True,
                data={
                    "total": len(self.todos),
                    "pending": sum(1 for t in self.todos if t.status == TodoStatus.PENDING),
                    "in_progress": sum(1 for t in self.todos if t.status == TodoStatus.IN_PROGRESS),
                    "completed": sum(1 for t in self.todos if t.status == TodoStatus.COMPLETED)
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def get_todos(self) -> ToolResult:
        return ToolResult(success=True, data=[t.to_dict() for t in self.todos])

    def _save(self):
        if self.persist_path:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.persist_path.write_text(json.dumps([t.to_dict() for t in self.todos], indent=2))

    def _load(self):
        if self.persist_path and self.persist_path.exists():
            data = json.loads(self.persist_path.read_text())
            self.todos = [TodoItem(content=t["content"], status=TodoStatus(t["status"]), active_form=t.get("activeForm", "")) for t in data]


# ============================================================
#  NOTEBOOK TOOL
# ============================================================

class NotebookTool:
    """Jupyter notebook editing."""

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = Path(workspace_root).resolve()

    def edit_cell(
        self,
        notebook_path: str,
        cell_index: int,
        new_source: str,
        cell_type: str = "code",
        edit_mode: str = "replace"
    ) -> ToolResult:
        """Edit a notebook cell."""
        try:
            path = self._resolve_path(notebook_path)
            if not path.exists():
                return ToolResult(success=False, error=f"Notebook not found: {notebook_path}")

            notebook = json.loads(path.read_text(encoding='utf-8'))
            cells = notebook.get("cells", [])

            if edit_mode == "delete":
                if cell_index < 0 or cell_index >= len(cells):
                    return ToolResult(success=False, error="Cell index out of range")
                del cells[cell_index]

            elif edit_mode == "insert":
                new_cell = {
                    "cell_type": cell_type,
                    "source": new_source.splitlines(keepends=True),
                    "metadata": {},
                }
                if cell_type == "code":
                    new_cell["outputs"] = []
                    new_cell["execution_count"] = None
                cells.insert(cell_index, new_cell)

            else:  # replace
                if cell_index < 0 or cell_index >= len(cells):
                    return ToolResult(success=False, error="Cell index out of range")
                cells[cell_index]["source"] = new_source.splitlines(keepends=True)
                cells[cell_index]["cell_type"] = cell_type

            notebook["cells"] = cells
            path.write_text(json.dumps(notebook, indent=1), encoding='utf-8')

            return ToolResult(success=True, data={"edit_mode": edit_mode, "cell_index": cell_index})

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _resolve_path(self, file_path: str) -> Path:
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        return path.resolve()


# ============================================================
#  TASK TOOL (Background Execution)
# ============================================================

@dataclass
class BackgroundTask:
    task_id: str
    command: str
    status: str = "running"
    output: str = ""
    error: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    return_code: Optional[int] = None


class TaskTool:
    """Background task execution."""

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = Path(workspace_root).resolve()
        self.tasks: Dict[str, BackgroundTask] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._counter = 0
        self._lock = threading.Lock()

    def run_background(self, command: str, cwd: Optional[str] = None, timeout: int = 600) -> ToolResult:
        """Run command in background."""
        # Same hardening as BashTool: no shell chaining/substitution/redirection
        # and no working directory outside the workspace.
        operator = find_shell_operator(command)
        if operator:
            return ToolResult(success=False, error=f"Blocked {operator}.")
        escape = find_path_escape(command, self.workspace_root)
        if escape:
            return ToolResult(success=False, error=f"Blocked {escape}.")

        work_dir = Path(cwd) if cwd else self.workspace_root
        if not work_dir.is_absolute():
            work_dir = self.workspace_root / work_dir
        work_dir = work_dir.resolve()
        if not is_within_workspace(work_dir, self.workspace_root):
            return ToolResult(
                success=False,
                error=f"Working directory outside workspace: {cwd}"
            )

        with self._lock:
            self._counter += 1
            task_id = f"task_{self._counter}"

        task = BackgroundTask(task_id=task_id, command=command)
        self.tasks[task_id] = task

        self.executor.submit(self._execute, task, work_dir, timeout)

        return ToolResult(success=True, data={"task_id": task_id, "status": "running"})

    def get_output(self, task_id: str, block: bool = False, timeout: int = 30) -> ToolResult:
        """Get task output."""
        if task_id not in self.tasks:
            return ToolResult(success=False, error=f"Task not found: {task_id}")

        task = self.tasks[task_id]

        if block and task.status == "running":
            start = time.time()
            while task.status == "running" and (time.time() - start) < timeout:
                time.sleep(0.5)

        return ToolResult(
            success=True,
            data={
                "task_id": task_id,
                "status": task.status,
                "output": task.output,
                "error": task.error,
                "return_code": task.return_code
            }
        )

    def list_tasks(self) -> ToolResult:
        return ToolResult(
            success=True,
            data=[{"task_id": t.task_id, "command": t.command, "status": t.status} for t in self.tasks.values()]
        )

    def _execute(self, task: BackgroundTask, work_dir: Path, timeout: int):
        try:
            result = subprocess.run(task.command, shell=True, capture_output=True, text=True, timeout=timeout, cwd=str(work_dir))
            task.output = result.stdout
            task.error = result.stderr
            task.return_code = result.returncode
            task.status = "completed" if result.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            task.status = "failed"
            task.error = f"Timed out after {timeout}s"
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
        finally:
            task.completed_at = datetime.now()


# ============================================================
#  LSP TOOL
# ============================================================

class LSPTool:
    """
    Code intelligence with real LSP support and regex fallback.
    """

    LANGUAGE_PATTERNS = {
        "python": {
            "function": r"^\s*(?:async\s+)?def\s+(\w+)",
            "class": r"^\s*class\s+(\w+)",
            "variable": r"^(\w+)\s*=",
        },
        "javascript": {
            "function": r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\()",
            "class": r"class\s+(\w+)",
            "variable": r"(?:const|let|var)\s+(\w+)\s*=",
        },
        "typescript": {
            "function": r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\()",
            "class": r"class\s+(\w+)",
            "interface": r"interface\s+(\w+)",
            "type": r"type\s+(\w+)\s*=",
        },
    }

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = Path(workspace_root).resolve()

    def get_symbols(self, file_path: str) -> ToolResult:
        """Get all symbols in a file."""
        try:
            path = self._resolve_path(file_path)
            if not path.exists():
                return ToolResult(success=False, error=f"File not found: {file_path}")

            content = path.read_text(encoding='utf-8')
            lines = content.splitlines()

            language = self._detect_language(path)
            patterns = self.LANGUAGE_PATTERNS.get(language, self.LANGUAGE_PATTERNS["python"])

            symbols = []
            for line_num, line in enumerate(lines, 1):
                for symbol_type, pattern in patterns.items():
                    match = re.search(pattern, line)
                    if match:
                        name = next((g for g in match.groups() if g), None)
                        if name:
                            symbols.append({
                                "name": name,
                                "type": symbol_type,
                                "line": line_num,
                                "preview": line.strip()
                            })

            return ToolResult(success=True, data=symbols, metadata={"language": language})

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def find_definition(self, symbol: str, path: Optional[str] = None) -> ToolResult:
        """Find where a symbol is defined."""
        try:
            search_path = self._resolve_path(path) if path else self.workspace_root
            escaped = re.escape(symbol)

            patterns = [
                rf"^\s*(?:async\s+)?def\s+{escaped}\s*\(",
                rf"^\s*class\s+{escaped}[\s:(]",
                rf"(?:const|let|var|function)\s+{escaped}\s*[=:(]",
                rf"interface\s+{escaped}\s*[\{{<]",
                rf"type\s+{escaped}\s*=",
            ]
            combined = "|".join(f"({p})" for p in patterns)

            definitions = []
            for fp in search_path.rglob("*"):
                if not fp.is_file() or fp.suffix not in ['.py', '.js', '.ts', '.tsx', '.jsx']:
                    continue
                try:
                    content = fp.read_text(encoding='utf-8', errors='ignore')
                    for i, line in enumerate(content.splitlines(), 1):
                        if re.search(combined, line):
                            definitions.append({
                                "file": str(fp.relative_to(self.workspace_root)),
                                "line": i,
                                "content": line.strip()
                            })
                except Exception:
                    continue

            return ToolResult(success=True, data=definitions, metadata={"symbol": symbol})

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def find_references(self, symbol: str, path: Optional[str] = None) -> ToolResult:
        """Find all references to a symbol."""
        try:
            search_path = self._resolve_path(path) if path else self.workspace_root
            pattern = rf"\b{re.escape(symbol)}\b"

            references = []
            for fp in search_path.rglob("*"):
                if not fp.is_file() or fp.suffix not in ['.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs']:
                    continue
                try:
                    content = fp.read_text(encoding='utf-8', errors='ignore')
                    for i, line in enumerate(content.splitlines(), 1):
                        if re.search(pattern, line):
                            references.append({
                                "file": str(fp.relative_to(self.workspace_root)),
                                "line": i,
                                "content": line.strip()
                            })
                except Exception:
                    continue

            return ToolResult(success=True, data=references, metadata={"symbol": symbol})

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _detect_language(self, path: Path) -> str:
        return {'.py': 'python', '.js': 'javascript', '.jsx': 'javascript', '.ts': 'typescript', '.tsx': 'typescript'}.get(path.suffix, 'python')

    def _resolve_path(self, file_path: str) -> Path:
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        return path.resolve()


# ============================================================
#  UNIFIED TOOLS FACADE
# ============================================================

class UnifiedTools:
    """
    Single interface for all Company AGI tools.

    Combines the best of GranularTools and EnhancedTools.
    """

    def __init__(self, workspace_root: str = ".", persist_dir: Optional[str] = None):
        self.workspace_root = Path(workspace_root).resolve()

        # File operations
        self.read = ReadTool(workspace_root)
        self.edit = EditTool(workspace_root)
        self.multi_edit = MultiEditTool(workspace_root)
        self.write = WriteTool(workspace_root)

        # Search operations
        self.glob = GlobTool(workspace_root)
        self.grep = GrepTool(workspace_root)

        # Code intelligence
        self.lsp = LSPTool(workspace_root)

        # System operations
        self.bash = BashTool(workspace_root, safe_mode=True)
        self.task = TaskTool(workspace_root)

        # Web operations
        self.web_fetch = WebFetchTool()
        self.web_search = WebSearchTool()

        # Task management
        self.todo = TodoTool(persist_path=f"{persist_dir}/todos.json" if persist_dir else None)

        # Notebook operations
        self.notebook = NotebookTool(workspace_root)

    def list_tools(self) -> List[str]:
        """List all available tools."""
        return [
            "read", "edit", "multi_edit", "write",
            "glob", "grep", "lsp", "bash", "task",
            "web_fetch", "web_search", "todo", "notebook"
        ]

    def get_tool(self, name: str):
        """Get a tool by name."""
        return getattr(self, name, None)


# ============================================================
#  EXPORTS
# ============================================================

__all__ = [
    # Result type
    "ToolResult",

    # File operations
    "ReadTool",
    "EditTool",
    "MultiEditTool",
    "WriteTool",

    # Search
    "GlobTool",
    "GrepTool",

    # Code intelligence
    "LSPTool",

    # System
    "BashTool",
    "TaskTool",
    "BackgroundTask",

    # Web
    "WebFetchTool",
    "WebSearchTool",

    # Task management
    "TodoTool",
    "TodoItem",
    "TodoStatus",

    # Notebook
    "NotebookTool",

    # Unified interface
    "UnifiedTools",
]
