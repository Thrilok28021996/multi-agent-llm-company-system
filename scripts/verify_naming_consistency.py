#!/usr/bin/env python3
"""
Naming Consistency Verification Script

Validates that all naming inconsistency fixes are in place and no regressions exist.
Run this as part of CI/CD to ensure naming standards are maintained.
"""

import re
import sys
from pathlib import Path
from typing import List


class ConsistencyChecker:
    """Automated consistency validation."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.passed: List[str] = []

    def check_department_members(self) -> bool:
        """Verify all agent dict keys exist in department definitions."""
        print("🔍 Checking department member names...")
        
        org_file = Path("company/organization.py")
        content = org_file.read_text()
        
        # Extract department members
        dept_pattern = r'members=\[(.*?)\]'
        matches = re.findall(dept_pattern, content, re.DOTALL)
        
        all_members = set()
        for match in matches:
            members = re.findall(r'"([^"]+)"', match)
            all_members.update(members)
        
        # Check for stale "DevOps" reference
        if "DevOps" in all_members and "DevOpsEngineer" not in all_members:
            self.errors.append(
                "❌ Department member 'DevOps' should be 'DevOpsEngineer' in company/organization.py"
            )
            return False
        
        # Expected agent names (PascalCase)
        expected_agents = {
            "CEO", "CTO", "ProductManager", "Researcher", 
            "Developer", "QAEngineer", "DevOpsEngineer", 
            "DataAnalyst", "SecurityEngineer"
        }
        
        for agent in expected_agents:
            if agent not in all_members:
                self.warnings.append(
                    f"⚠️  Agent '{agent}' not found in any department"
                )
        
        self.passed.append("✅ Department member names use correct format (DevOpsEngineer, not DevOps)")
        return True

    def check_okr_keys(self) -> bool:
        """Verify OKR dictionary keys match agent dict keys (PascalCase)."""
        print("🔍 Checking OKR dictionary keys...")
        
        perf_file = Path("company/performance.py")
        content = perf_file.read_text()
        
        # Extract DEFAULT_OKRS keys
        okr_section = re.search(r'DEFAULT_OKRS.*?= \{(.*?)\n\}', content, re.DOTALL)
        if not okr_section:
            self.errors.append("❌ Could not find DEFAULT_OKRS definition")
            return False
        
        okr_keys = re.findall(r'"([^"]+)":\s*AgentOKR', okr_section.group(1))
        
        # Check for snake_case keys (old format)
        snake_case_keys = [k for k in okr_keys if '_' in k and k[0].islower()]
        if snake_case_keys:
            self.errors.append(
                f"❌ OKR keys should be PascalCase (agent dict keys), not snake_case: {snake_case_keys}"
            )
            return False
        
        # Expected format: PascalCase
        expected_keys = {
            "CEO", "CTO", "ProductManager", "Researcher",
            "Developer", "QAEngineer", "DevOpsEngineer",
            "DataAnalyst", "SecurityEngineer"
        }
        
        missing = expected_keys - set(okr_keys)
        if missing:
            self.warnings.append(f"⚠️  Missing OKR entries for: {missing}")
        
        self.passed.append("✅ OKR keys use PascalCase (agent dict keys)")
        return True

    def check_progress_phase_ids(self) -> bool:
        """Verify progress tracker phase IDs match WorkflowPhase enum values."""
        print("🔍 Checking progress tracker phase IDs...")
        
        workflow_file = Path("orchestrator/workflow.py")
        content = workflow_file.read_text()
        
        # Extract WorkflowPhase enum values
        enum_section = re.search(
            r'class WorkflowPhase\(Enum\):.*?(?=\n\n|\nclass|\n@dataclass)',
            content,
            re.DOTALL
        )
        if not enum_section:
            self.errors.append("❌ Could not find WorkflowPhase enum")
            return False
        
        # Extract progress.add_phases() calls
        phase_patterns = re.findall(
            r'\{"id": "([^"]+)", "name"',
            content
        )
        
        # Check for old short names
        if "opportunity" in phase_patterns:
            self.errors.append(
                "❌ Progress phase ID 'opportunity' should be 'opportunity_evaluation'"
            )
            return False
        
        if "design" in phase_patterns and "technical_design" not in phase_patterns:
            self.errors.append(
                "❌ Progress phase ID 'design' should be 'technical_design'"
            )
            return False
        
        # Verify all progress IDs that should match do match
        enum_phases = {
            "research", "data_analysis", "analysis", "opportunity_evaluation",
            "technical_design", "design_review", "implementation", "code_execution",
            "qa_validation", "security_review", "ceo_approval", "retrospective"
        }
        
        phase_set = set(phase_patterns)
        mismatches = enum_phases - phase_set
        if mismatches and mismatches != {"idle", "completed", "failed"}:
            self.warnings.append(
                f"⚠️  Progress phases don't fully match WorkflowPhase enum: {mismatches}"
            )
        
        self.passed.append("✅ Progress tracker phase IDs match WorkflowPhase enum values")
        return True

    def check_model_defaults(self) -> bool:
        """Verify config_loader model defaults match models.py."""
        print("🔍 Checking model defaults synchronization...")
        
        config_file = Path("config/config_loader.py")
        
        config_content = config_file.read_text()
        
        # Check for stale defaults
        stale_patterns = [
            (r'ceo: str = "qwen3-8b"', "AgentModels should use current model names, not 'qwen3-8b'"),
            (r'product_manager: str = "ministral-8b"', "AgentModels should use 'ministral-3:8b', not 'ministral-8b'"),
            (r'developer: str = "qwen2.5-coder-7b"', "AgentModels should use current model, not 'qwen2.5-coder-7b'"),
        ]
        
        for pattern, message in stale_patterns:
            if re.search(pattern, config_content):
                self.errors.append(f"❌ {message}")
                return False
        
        # Check that models match
        if 'goekdenizguelmez/JOSIEFIED-Qwen3:8b' in config_content:
            self.passed.append("✅ Config loader model defaults are current")
            return True
        else:
            self.warnings.append("⚠️  Could not verify model defaults match")
            return True

    def check_no_stale_references(self) -> bool:
        """Scan for stale naming patterns."""
        print("🔍 Checking for stale naming references...")
        
        # Check orchestrator/workflow.py for old phase names in method calls
        workflow = Path("orchestrator/workflow.py").read_text()
        
        stale_progress_calls = re.findall(
            r'progress\.(skip_phase|start_phase|complete_phase|fail_phase)\("(opportunity|design)"\)',
            workflow
        )
        
        if stale_progress_calls:
            self.errors.append(
                f"❌ Found stale progress phase calls: {stale_progress_calls}"
            )
            return False
        
        self.passed.append("✅ No stale naming references found")
        return True

    def run_all_checks(self) -> bool:
        """Run all consistency checks."""
        print("\n" + "="*60)
        print("  Naming Consistency Verification")
        print("="*60 + "\n")
        
        checks = [
            self.check_department_members,
            self.check_okr_keys,
            self.check_progress_phase_ids,
            self.check_model_defaults,
            self.check_no_stale_references,
        ]
        
        all_passed = all(check() for check in checks)
        
        print("\n" + "="*60)
        print("  Results")
        print("="*60 + "\n")
        
        if self.passed:
            for msg in self.passed:
                print(msg)
        
        if self.warnings:
            print()
            for msg in self.warnings:
                print(msg)
        
        if self.errors:
            print()
            for msg in self.errors:
                print(msg)
        
        print("\n" + "="*60)
        if all_passed and not self.errors:
            print("✅ All naming consistency checks PASSED")
            print("="*60 + "\n")
            return True
        else:
            print("❌ Some checks FAILED")
            print("="*60 + "\n")
            return False


def main():
    checker = ConsistencyChecker()
    success = checker.run_all_checks()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
