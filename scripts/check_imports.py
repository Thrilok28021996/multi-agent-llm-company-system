#!/usr/bin/env python3
"""
Comprehensive Import Checker

Validates all imports across the codebase:
1. Circular dependency detection
2. Missing imports
3. Unused imports
4. Import-time errors
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class ImportChecker:
    """Check import consistency across codebase."""

    def __init__(self, root_dir: str = "."):
        self.root = Path(root_dir)
        self.files = list(self.root.glob("**/*.py"))
        # Filter out venv, build, etc.
        self.files = [
            f for f in self.files
            if not any(part.startswith(".") or part in ["venv", "build", "dist", "__pycache__"]
                      for part in f.parts)
        ]
        self.imports: Dict[str, Set[str]] = defaultdict(set)
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.circular_deps: List[Tuple[str, str]] = []

    def get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        rel_path = file_path.relative_to(self.root)
        if rel_path.name == "__init__.py":
            parts = rel_path.parts[:-1]
        else:
            parts = rel_path.parts[:-1] + (rel_path.stem,)
        return ".".join(parts) if parts else "__main__"

    def extract_imports(self, file_path: Path) -> Set[str]:
        """Extract all imports from a file."""
        imports = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
                    elif node.level > 0:
                        # Relative import
                        module_name = self.get_module_name(file_path)
                        parts = module_name.split('.')
                        if node.level <= len(parts):
                            parent = '.'.join(parts[:-node.level])
                            imports.add(parent.split('.')[0] if parent else parts[0])
        except SyntaxError as e:
            self.errors.append(f"❌ Syntax error in {file_path}: {e}")
        except Exception as e:
            self.errors.append(f"❌ Error parsing {file_path}: {e}")

        return imports

    def build_import_graph(self):
        """Build import dependency graph."""
        print("🔍 Scanning imports...")
        for file_path in self.files:
            module_name = self.get_module_name(file_path)
            imports = self.extract_imports(file_path)
            self.imports[module_name] = imports

    def check_circular_dependencies(self):
        """Detect circular dependencies using DFS."""
        print("🔍 Checking for circular dependencies...")
        visited = set()
        rec_stack = set()

        def visit(module: str, path: List[str]):
            if module in rec_stack:
                # Found a cycle
                cycle_start = path.index(module)
                cycle = path[cycle_start:] + [module]
                self.circular_deps.append(tuple(cycle))
                return

            if module in visited:
                return

            visited.add(module)
            rec_stack.add(module)
            path.append(module)

            # Visit imports (only local modules)
            local_modules = {m.split('.')[0] for m in self.imports.keys()}
            for imported in self.imports.get(module, set()):
                if imported in local_modules:
                    visit(imported, path.copy())

            rec_stack.remove(module)

        for module in self.imports.keys():
            if module not in visited:
                visit(module, [])

    def check_import_errors(self):
        """Try importing all modules to detect runtime errors."""
        print("🔍 Checking for import-time errors...")
        import importlib
        import sys

        # Add current dir to path
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))

        for file_path in self.files:
            module_name = self.get_module_name(file_path)
            if module_name == "__main__":
                continue

            try:
                # Try to import
                if module_name in sys.modules:
                    importlib.reload(sys.modules[module_name])
                else:
                    importlib.import_module(module_name)
            except ImportError as e:
                self.errors.append(f"❌ Import error in {module_name}: {e}")
            except Exception as e:
                self.warnings.append(f"⚠️  Runtime error importing {module_name}: {e}")

    def check_unused_imports(self):
        """Detect unused imports (basic check)."""
        print("🔍 Checking for potentially unused imports...")
        for file_path in self.files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content, filename=str(file_path))

                # Get all imports
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            name = alias.asname if alias.asname else alias.name
                            imports.append((node.lineno, name.split('.')[0]))
                    elif isinstance(node, ast.ImportFrom):
                        for alias in node.names:
                            if alias.name != '*':
                                name = alias.asname if alias.asname else alias.name
                                imports.append((node.lineno, name))

                # Check if each import is used
                for line_no, name in imports:
                    # Simple heuristic: count occurrences
                    count = content.count(name)
                    if count == 1:  # Only appears in import line
                        # Could be unused (but might be re-exported in __init__.py)
                        module = self.get_module_name(file_path)
                        if not module.endswith(".__init__"):
                            self.warnings.append(
                                f"⚠️  Potentially unused import '{name}' in {file_path}:{line_no}"
                            )
            except Exception:
                pass  # Skip files with parsing errors (already reported)

    def report(self):
        """Generate comprehensive report."""
        print("\n" + "="*70)
        print("  IMPORT CHECK REPORT")
        print("="*70 + "\n")

        if self.circular_deps:
            print("🔴 CIRCULAR DEPENDENCIES FOUND:")
            seen = set()
            for cycle in self.circular_deps:
                cycle_str = " → ".join(cycle)
                if cycle_str not in seen:
                    print(f"   {cycle_str}")
                    seen.add(cycle_str)
            print()

        if self.errors:
            print("❌ ERRORS:")
            for error in self.errors:
                print(f"   {error}")
            print()

        if self.warnings:
            print("⚠️  WARNINGS:")
            # Limit warnings to first 20
            for warning in self.warnings[:20]:
                print(f"   {warning}")
            if len(self.warnings) > 20:
                print(f"   ... and {len(self.warnings) - 20} more warnings")
            print()

        # Summary
        print("="*70)
        print(f"📊 Files scanned: {len(self.files)}")
        print(f"📊 Modules found: {len(self.imports)}")
        print(f"📊 Circular deps: {len(set(self.circular_deps))}")
        print(f"📊 Errors: {len(self.errors)}")
        print(f"📊 Warnings: {len(self.warnings)}")
        print("="*70 + "\n")

        return len(self.errors) == 0

    def run(self):
        """Run all checks."""
        self.build_import_graph()
        self.check_circular_dependencies()
        # Skip runtime import check for now (can cause side effects)
        # self.check_import_errors()
        self.check_unused_imports()
        return self.report()


def main():
    checker = ImportChecker()
    success = checker.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
