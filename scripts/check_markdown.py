#!/usr/bin/env python3
"""
Comprehensive Markdown Checker

Validates markdown files for:
1. Broken internal links (file references)
2. Code block syntax issues
3. Inconsistencies with actual codebase
4. Outdated examples
5. Formatting issues
"""

import re
import sys
from pathlib import Path
from typing import List


class MarkdownChecker:
    """Check markdown files for various issues."""

    def __init__(self, root_dir: str = "."):
        self.root = Path(root_dir)
        self.md_files = [
            f for f in self.root.glob("**/*.md")
            if not any(part.startswith(".") or part in ["venv", "build", "dist", "node_modules"]
                      for part in f.parts if part != ".pytest_cache")
        ]
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def check_internal_links(self, file_path: Path, content: str):
        """Check for broken internal file/section links."""
        # Match markdown links: [text](path) or [text](path#section)
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        links = re.findall(link_pattern, content)

        for text, link in links:
            # Skip external links
            if link.startswith(('http://', 'https://', 'mailto:', '#')):
                continue

            # Handle section links
            if '#' in link:
                file_part, section = link.split('#', 1)
                if not file_part:  # Just a section reference like (#section)
                    # Check if section exists in current file
                    section_id = section.lower().replace(' ', '-')
                    if f'#{section_id}' not in content.lower() and f'id="{section}"' not in content:
                        # Look for heading
                        heading_pattern = rf'^#+\s+{re.escape(section)}'
                        if not re.search(heading_pattern, content, re.MULTILINE | re.IGNORECASE):
                            self.warnings.append(
                                f"⚠️  {file_path}: Section link '#{section}' may be broken"
                            )
                    continue
                link = file_part

            # Resolve relative paths
            if link:
                target = (file_path.parent / link).resolve()
                if not target.exists():
                    self.errors.append(
                        f"❌ {file_path}: Broken link to '{link}'"
                    )

    def check_code_blocks(self, file_path: Path, content: str):
        """Check code blocks for syntax and consistency."""
        # Skip files documenting fixes (they show broken code intentionally)
        if any(name in file_path.name.lower() for name in ['fix', 'visual', 'summary', 'report']):
            return
        
        # Find all code blocks
        code_block_pattern = r'```(\w+)?\n(.*?)```'
        blocks = re.findall(code_block_pattern, content, re.DOTALL)

        for lang, code in blocks:
            code = code.strip()
            if not code:
                continue

            # Check Python code blocks for syntax
            if lang in ('python', 'py', 'python3'):
                try:
                    compile(code, '<string>', 'exec')
                except SyntaxError as e:
                    self.warnings.append(
                        f"⚠️  {file_path}: Python code block has syntax error: {e}"
                    )

            # Check for common issues
            if 'TODO' in code or 'FIXME' in code:
                self.info.append(
                    f"ℹ️  {file_path}: Code block contains TODO/FIXME"
                )

    def check_file_references(self, file_path: Path, content: str):
        """Check references to actual files in the codebase."""
        # Skip files that are documenting fixes/changes (they reference old issues)
        if any(name in file_path.name.lower() for name in ['fix', 'visual', 'summary', 'report']):
            return
        
        # Look for file path references like `path/to/file.py`
        file_ref_pattern = r'`([a-zA-Z_][a-zA-Z0-9_/]*\.py(?::\d+)?)`'
        refs = re.findall(file_ref_pattern, content)

        for ref in refs:
            # Skip generic placeholders
            if 'path/to' in ref or 'example' in ref.lower():
                continue
            
            # Strip line numbers if present
            file_ref = ref.split(':')[0]
            target = self.root / file_ref
            if not target.exists():
                self.warnings.append(
                    f"⚠️  {file_path}: Reference to non-existent file '{file_ref}'"
                )

    def check_command_examples(self, file_path: Path, content: str):
        """Check shell command examples for common issues."""
        # Find bash/shell code blocks
        bash_pattern = r'```(?:bash|sh|shell)\n(.*?)```'
        commands = re.findall(bash_pattern, content, re.DOTALL)

        for cmd_block in commands:
            lines = cmd_block.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Check for potentially dangerous commands
                if any(danger in line for danger in ['rm -rf /', 'dd if=', 'mkfs', ':(){:|:&};:']):
                    self.errors.append(
                        f"❌ {file_path}: Dangerous command in example: {line[:50]}"
                    )

                # Check for common mistakes
                if 'sudo pip' in line:
                    self.warnings.append(
                        f"⚠️  {file_path}: 'sudo pip' is discouraged, use virtual environments"
                    )

    def check_agent_references(self, file_path: Path, content: str):
        """Check that agent names match the codebase."""
        # Skip files that are documenting fixes (they show before/after)
        if any(name in file_path.name.lower() for name in ['fix', 'visual', 'summary', 'report']):
            return
        
        # Expected agents (from our fixes)
        expected_agents = {
            "CEO", "CTO", "ProductManager", "Researcher",
            "Developer", "QAEngineer", "DevOpsEngineer",
            "DataAnalyst", "SecurityEngineer"
        }

        # Look for potential agent references
        for agent in expected_agents:
            # Check for old/wrong names
            if agent == "DevOpsEngineer":
                # Ignore if it's in quotes (showing old code) or in code blocks
                if re.search(r'\b(?:DevOps|Dev-Ops|Dev\s+Ops)\b(?!\s*Engineer)', content):
                    # Check if it's not in a code block or quoted
                    lines = content.split('\n')
                    in_code_block = False
                    for line in lines:
                        if line.strip().startswith('```'):
                            in_code_block = not in_code_block
                        if not in_code_block and 'DevOps' in line and 'DevOpsEngineer' not in line:
                            # Skip if:
                            # - It's quoted (showing old code)
                            # - It's "DevOps Engineer" (human-readable title)
                            # - It's in a list/table context (shorthand is ok)
                            if ('"DevOps"' in line or "'DevOps'" in line or '`DevOps`' in line or
                                'DevOps Engineer' in line or 'DevOps,' in line or '| DevOps' in line):
                                continue
                            self.warnings.append(
                                f"⚠️  {file_path}: Found 'DevOps' without 'Engineer' - should be 'DevOpsEngineer'"
                            )
                            break

    def check_formatting(self, file_path: Path, content: str):
        """Check for formatting issues."""
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # Check for trailing whitespace
            if line.rstrip() != line and line.strip():  # Ignore blank lines
                self.info.append(
                    f"ℹ️  {file_path}:{i}: Trailing whitespace"
                )

            # Check for mixed list markers (but skip in diff blocks)
            if i > 1:
                prev_line = lines[i-2]
                if re.match(r'^\s*[-*+]\s', prev_line) and re.match(r'^\s*[-*+]\s', line):
                    prev_marker = re.match(r'^\s*([-*+])\s', prev_line).group(1)
                    curr_marker = re.match(r'^\s*([-*+])\s', line).group(1)
                    # Allow mixing - and + for diff notation
                    if prev_marker != curr_marker and not (set([prev_marker, curr_marker]) == {'-', '+'}):
                        self.warnings.append(
                            f"⚠️  {file_path}:{i}: Inconsistent list markers (mixing {prev_marker} and {curr_marker})"
                        )

        # Check for missing blank line before headers (common issue)
        header_pattern = r'^#{1,6}\s+.+'
        for i, line in enumerate(lines):
            if re.match(header_pattern, line):
                if i > 0 and lines[i-1].strip() and not re.match(header_pattern, lines[i-1]):
                    self.info.append(
                        f"ℹ️  {file_path}:{i+1}: Consider adding blank line before header"
                    )

    def check_consistency_with_code(self, file_path: Path, content: str):
        """Check that code examples match actual implementation."""
        # Skip files documenting fixes/changes
        if any(name in file_path.name.lower() for name in ['fix', 'visual', 'summary', 'report']):
            return
        
        # Check for model names
        if 'qwen3-8b' in content.lower() or 'ministral-8b' in content.lower():
            if 'config' not in str(file_path).lower():
                self.warnings.append(
                    f"⚠️  {file_path}: Contains old model names (qwen3-8b or ministral-8b)"
                )

        # Check for phase names
        if '"opportunity"' in content and 'opportunity_evaluation' not in content:
            if 'fix' not in str(file_path).lower() and 'visual' not in str(file_path).lower():
                self.warnings.append(
                    f"⚠️  {file_path}: May reference old phase name 'opportunity' instead of 'opportunity_evaluation'"
                )

        if '"design"' in content and 'technical_design' not in content:
            if 'fix' not in str(file_path).lower() and 'visual' not in str(file_path).lower():
                self.warnings.append(
                    f"⚠️  {file_path}: May reference old phase name 'design' instead of 'technical_design'"
                )

    def check_file(self, file_path: Path):
        """Run all checks on a single file."""
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            self.errors.append(f"❌ Cannot read {file_path}: {e}")
            return

        self.check_internal_links(file_path, content)
        self.check_code_blocks(file_path, content)
        self.check_file_references(file_path, content)
        self.check_command_examples(file_path, content)
        self.check_agent_references(file_path, content)
        self.check_formatting(file_path, content)
        self.check_consistency_with_code(file_path, content)

    def report(self):
        """Generate comprehensive report."""
        print("\n" + "="*70)
        print("  MARKDOWN CHECK REPORT")
        print("="*70 + "\n")

        if self.errors:
            print("❌ ERRORS:")
            for error in self.errors:
                print(f"   {error}")
            print()

        if self.warnings:
            print("⚠️  WARNINGS:")
            for warning in self.warnings[:30]:  # Limit output
                print(f"   {warning}")
            if len(self.warnings) > 30:
                print(f"   ... and {len(self.warnings) - 30} more warnings")
            print()

        if self.info:
            print("ℹ️  INFO (first 20):")
            for info in self.info[:20]:
                print(f"   {info}")
            if len(self.info) > 20:
                print(f"   ... and {len(self.info) - 20} more info items")
            print()

        # Summary
        print("="*70)
        print(f"📊 Files checked: {len(self.md_files)}")
        print(f"📊 Errors: {len(self.errors)}")
        print(f"📊 Warnings: {len(self.warnings)}")
        print(f"📊 Info: {len(self.info)}")
        print("="*70 + "\n")

        return len(self.errors) == 0

    def run(self):
        """Run all checks."""
        print(f"🔍 Scanning {len(self.md_files)} markdown files...")
        for file_path in self.md_files:
            self.check_file(file_path)
        return self.report()


def main():
    checker = MarkdownChecker()
    success = checker.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
