#!/usr/bin/env python3
"""
Project Naming Consistency Checker

Validates that the project name is consistent across all files and that
pyproject.toml is correctly configured.
"""

import re
import sys
from pathlib import Path
from typing import List


class ProjectNamingChecker:
    """Check project naming consistency."""

    def __init__(self, root_dir: str = "."):
        self.root = Path(root_dir)
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        
        # Expected canonical names
        self.canonical_names = {
            "project_title": "Autonomous Company Orchestrator",
            "short_name": "Company AGI",  # For UI/docs
            "package_name": "multi-agent-llm-company-system",  # For pyproject.toml
            "repo_name": "autonomous-company-orchestrator",  # For GitHub
            "env_prefix": "COMPANY_AGI",  # For environment variables
            "logger_name": "company_agi",  # For logging
        }

    def check_pyproject_toml(self):
        """Validate pyproject.toml configuration."""
        print("🔍 Checking pyproject.toml...")
        
        pyproject = self.root / "pyproject.toml"
        if not pyproject.exists():
            self.errors.append("❌ pyproject.toml not found")
            return
        
        content = pyproject.read_text()
        
        # Check project name
        name_match = re.search(r'name\s*=\s*"([^"]+)"', content)
        if name_match:
            name = name_match.group(1)
            if name != self.canonical_names["package_name"]:
                self.warnings.append(
                    f"⚠️  pyproject.toml: package name is '{name}', expected '{self.canonical_names['package_name']}'"
                )
        else:
            self.errors.append("❌ pyproject.toml: no project name found")
        
        # Check dependencies vs requirements.txt
        requirements_file = self.root / "requirements.txt"
        if requirements_file.exists():
            req_deps = set()
            for line in requirements_file.read_text().split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name (before >=, ==, etc.)
                    pkg = re.split(r'[><=!]', line)[0].strip()
                    req_deps.add(pkg.lower())
            
            # Extract pyproject.toml dependencies
            pyproject_deps = set()
            deps_match = re.search(r'dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if deps_match:
                for dep in re.findall(r'"([^"]+)"', deps_match.group(1)):
                    pkg = re.split(r'[><=!]', dep)[0].strip()
                    pyproject_deps.add(pkg.lower())
            
            # Compare
            only_req = req_deps - pyproject_deps
            only_pyproject = pyproject_deps - req_deps
            
            if only_req:
                self.warnings.append(
                    f"⚠️  Dependencies in requirements.txt but not pyproject.toml: {only_req}"
                )
            if only_pyproject:
                self.warnings.append(
                    f"⚠️  Dependencies in pyproject.toml but not requirements.txt: {only_pyproject}"
                )
        
        # Check for [project.optional-dependencies] for dev tools
        if 'pytest' in content and '[project.optional-dependencies]' not in content:
            self.info.append(
                "ℹ️  pyproject.toml: pytest should be in [project.optional-dependencies] dev section, not dependencies"
            )

    def check_readme(self):
        """Check README.md for project name consistency."""
        print("🔍 Checking README.md...")
        
        readme = self.root / "README.md"
        if not readme.exists():
            self.errors.append("❌ README.md not found")
            return
        
        content = readme.read_text()
        
        # Check title (first h1)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            title = title_match.group(1)
            if title != self.canonical_names["project_title"]:
                self.info.append(
                    f"ℹ️  README.md: Title is '{title}', canonical is '{self.canonical_names['project_title']}'"
                )
        
        # Check git clone URL
        git_url_pattern = r'git clone.*?github\.com/[^/]+/([^\s]+)'
        git_match = re.search(git_url_pattern, content)
        if git_match:
            repo = git_match.group(1).replace('.git', '')
            if repo != self.canonical_names["repo_name"]:
                self.warnings.append(
                    f"⚠️  README.md: GitHub repo is '{repo}', expected '{self.canonical_names['repo_name']}'"
                )

    def check_usage_guide(self):
        """Check USAGE_GUIDE.md for project name consistency."""
        print("🔍 Checking USAGE_GUIDE.md...")
        
        usage = self.root / "USAGE_GUIDE.md"
        if not usage.exists():
            self.info.append("ℹ️  USAGE_GUIDE.md not found")
            return
        
        content = usage.read_text()
        
        # Check title
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            title = title_match.group(1)
            # Should be "{short_name} - Usage Guide"
            expected = f"{self.canonical_names['short_name']} - Usage Guide"
            if title != expected:
                self.info.append(
                    f"ℹ️  USAGE_GUIDE.md: Title is '{title}', expected '{expected}'"
                )
        
        # Check for references to old project names
        if 'backend/src/main.py' in content:
            self.warnings.append(
                "⚠️  USAGE_GUIDE.md: References non-existent 'backend/src/main.py'"
            )

    def check_env_vars(self):
        """Check environment variable naming consistency."""
        print("🔍 Checking environment variables...")
        
        # Check .env and .env.example
        for env_file in ['.env', '.env.example']:
            path = self.root / env_file
            if not path.exists():
                continue
            
            content = path.read_text()
            
            # Find all env var definitions
            env_vars = re.findall(r'^([A-Z_][A-Z0-9_]*)\s*=', content, re.MULTILINE)
            env_vars += re.findall(r'^#\s*([A-Z_][A-Z0-9_]*)\s*=', content, re.MULTILINE)
            
            for var in env_vars:
                if var.startswith('COMPANY_AGI_') or var.startswith('MODEL_'):
                    continue  # Expected prefixes
                elif 'COMPANY' in var or 'AGI' in var:
                    self.warnings.append(
                        f"⚠️  {env_file}: Env var '{var}' doesn't use standard prefix 'COMPANY_AGI_'"
                    )
        
        # Check Python files for env var usage
        for py_file in self.root.glob('**/*.py'):
            if any(part.startswith('.') for part in py_file.parts):
                continue
            
            try:
                content = py_file.read_text()
                # Find os.getenv calls
                env_calls = re.findall(r'os\.getenv\(["\']([^"\']+)["\']\)', content)
                
                for var in env_calls:
                    if 'COMPANY' in var or 'AGI' in var:
                        if not var.startswith('COMPANY_AGI_'):
                            self.warnings.append(
                                f"⚠️  {py_file}: Env var '{var}' should use 'COMPANY_AGI_' prefix"
                            )
            except:
                pass

    def check_logger_names(self):
        """Check logger naming consistency."""
        print("🔍 Checking logger names...")
        
        for py_file in self.root.glob('**/*.py'):
            if any(part.startswith('.') for part in py_file.parts):
                continue
            
            try:
                content = py_file.read_text()
                # Find logger names
                logger_calls = re.findall(r'getLogger\(["\']([^"\']+)["\']\)', content)
                
                for logger_name in logger_calls:
                    if logger_name == self.canonical_names["logger_name"]:
                        continue
                    if logger_name.startswith(self.canonical_names["logger_name"] + "."):
                        continue  # Subloggers are ok
                    if logger_name != "__name__":
                        self.info.append(
                            f"ℹ️  {py_file}: Logger '{logger_name}' doesn't use canonical '{self.canonical_names['logger_name']}'"
                        )
            except:
                pass

    def check_import_names(self):
        """Check that imports use correct package structure."""
        print("🔍 Checking import statements...")
        
        # The project should use relative imports or imports from top-level modules
        # Not "multi-agent-llm-company-system" since that's just the package name
        
        for py_file in self.root.glob('**/*.py'):
            if py_file.name == 'check_project_naming.py' or any(part.startswith('.') for part in py_file.parts):
                continue
            
            try:
                content = py_file.read_text()
                # Check for wrong absolute imports
                if 'from multi-agent-llm-company-system' in content:
                    self.errors.append(
                        f"❌ {py_file}: Importing from 'multi-agent-llm-company-system' (use relative imports)"
                    )
            except:
                pass

    def report(self):
        """Generate comprehensive report."""
        print("\n" + "="*70)
        print("  PROJECT NAMING CONSISTENCY REPORT")
        print("="*70 + "\n")
        
        print("📋 Canonical Names:")
        for key, value in self.canonical_names.items():
            print(f"   {key}: {value}")
        print()
        
        if self.errors:
            print("❌ ERRORS:")
            for error in self.errors:
                print(f"   {error}")
            print()
        
        if self.warnings:
            print("⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"   {warning}")
            print()
        
        if self.info:
            print("ℹ️  INFO:")
            for info in self.info[:20]:
                print(f"   {info}")
            if len(self.info) > 20:
                print(f"   ... and {len(self.info) - 20} more info items")
            print()
        
        # Summary
        print("="*70)
        print(f"📊 Errors: {len(self.errors)}")
        print(f"📊 Warnings: {len(self.warnings)}")
        print(f"📊 Info: {len(self.info)}")
        print("="*70 + "\n")
        
        return len(self.errors) == 0

    def run(self):
        """Run all checks."""
        self.check_pyproject_toml()
        self.check_readme()
        self.check_usage_guide()
        self.check_env_vars()
        self.check_logger_names()
        self.check_import_names()
        return self.report()


def main():
    checker = ProjectNamingChecker()
    success = checker.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
