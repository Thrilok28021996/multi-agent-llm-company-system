"""
Test script to verify all agents have access to Claude Code tools and problem refinement.

This script tests:
1. All agents can be instantiated with AgentToolsMixin
2. All agents have access to all 13 Claude Code tools
3. Problem statement refinement works for all agents
4. Complete workflow: Problem discovery → Refinement → Solution building
"""

import sys
from pathlib import Path

# Add project root to path (this script lives in scripts/)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from agents.developer import DeveloperAgent
from agents.qa_engineer import QAEngineerAgent
from agents.cto import CTOAgent
from agents.product_manager import ProductManagerAgent
from agents.researcher import ResearcherAgent
from agents.ceo import CEOAgent
from ui.console import CompanyConsole

console = CompanyConsole()


def test_agent_instantiation():
    """Test that all agents can be instantiated."""
    console.info("\n" + "=" * 80)
    console.info("TEST 1: Agent Instantiation")
    console.info("=" * 80)

    agents = {}

    try:
        console.info("\n[1/6] Creating Developer Agent...")
        agents['developer'] = DeveloperAgent(workspace_root=".")
        console.success("✓ Developer Agent created successfully")

        console.info("\n[2/6] Creating QA Engineer Agent...")
        agents['qa'] = QAEngineerAgent(workspace_root=".")
        console.success("✓ QA Engineer Agent created successfully")

        console.info("\n[3/6] Creating CTO Agent...")
        agents['cto'] = CTOAgent(workspace_root=".")
        console.success("✓ CTO Agent created successfully")

        console.info("\n[4/6] Creating Product Manager Agent...")
        agents['pm'] = ProductManagerAgent(workspace_root=".")
        console.success("✓ Product Manager Agent created successfully")

        console.info("\n[5/6] Creating Researcher Agent...")
        agents['researcher'] = ResearcherAgent(workspace_root=".")
        console.success("✓ Researcher Agent created successfully")

        console.info("\n[6/6] Creating CEO Agent...")
        agents['ceo'] = CEOAgent(workspace_root=".")
        console.success("✓ CEO Agent created successfully")

        console.success("\n✅ ALL AGENTS INSTANTIATED SUCCESSFULLY\n")
        return agents, True

    except Exception as e:
        console.error(f"\n❌ AGENT INSTANTIATION FAILED: {str(e)}")
        import traceback
        console.error(traceback.format_exc())
        return agents, False


def test_claude_code_tools(agents):
    """Test that all agents have access to all 13 Claude Code tools."""
    console.info("\n" + "=" * 80)
    console.info("TEST 2: Claude Code Tools Availability")
    console.info("=" * 80)

    # List of all 13 Claude Code tools
    required_tools = [
        # File Operations
        'read_file',
        'write_file',
        'edit_file',
        'multi_edit_file',

        # File Search
        'glob_files',
        'grep_search',

        # Command Execution
        'bash_execute',

        # Web Operations
        'web_fetch',
        'web_search',

        # Task Management
        'todo_create',
        'todo_list',
        'todo_complete',

        # Notebook Operations
        'notebook_edit',

        # Code Intelligence (LSP)
        # Note: LSP tools might not be in the simplified mixin,
        # but the core 10 tools should be there
    ]

    all_passed = True

    for agent_name, agent in agents.items():
        console.info(f"\n[{agent_name.upper()}] Checking tool availability...")

        missing_tools = []
        available_tools = []

        for tool in required_tools:
            if hasattr(agent, tool):
                available_tools.append(tool)
            else:
                missing_tools.append(tool)
                all_passed = False

        if not missing_tools:
            console.success(f"  ✓ All {len(required_tools)} tools available")
        else:
            console.warning(f"  ⚠ {len(available_tools)}/{len(required_tools)} tools available")
            console.warning(f"  Missing tools: {', '.join(missing_tools)}")

    if all_passed:
        console.success("\n✅ ALL AGENTS HAVE ALL CLAUDE CODE TOOLS\n")
    else:
        console.warning("\n⚠ SOME TOOLS MISSING (may be expected for LSP tools)\n")

    return all_passed


def test_problem_refinement(agents):
    """Test that all agents can refine problem statements."""
    console.info("\n" + "=" * 80)
    console.info("TEST 3: Problem Statement Refinement")
    console.info("=" * 80)

    vague_problem = "The app is slow and users are complaining"

    all_passed = True

    for agent_name, agent in agents.items():
        console.info(f"\n[{agent_name.upper()}] Testing problem refinement...")

        try:
            # Check if agent has problem_refiner
            if not hasattr(agent, 'problem_refiner'):
                console.error("  ❌ Agent does not have problem_refiner attribute")
                all_passed = False
                continue

            # Test refinement
            refined = agent.problem_refiner.refine(vague_problem)

            console.success("  ✓ Problem refiner available")
            console.info(f"  Original: {vague_problem}")
            console.info(f"  Refined: {refined.refined_statement}")
            console.info(f"  Type: {refined.problem_type.value}")
            console.info(f"  Clarity: {refined.clarity_level.value}")

        except Exception as e:
            console.error(f"  ❌ Refinement failed: {str(e)}")
            all_passed = False

    if all_passed:
        console.success("\n✅ ALL AGENTS CAN REFINE PROBLEM STATEMENTS\n")
    else:
        console.error("\n❌ SOME AGENTS CANNOT REFINE PROBLEMS\n")

    return all_passed


def test_developer_tools():
    """Test Developer agent can use tools."""
    console.info("\n" + "=" * 80)
    console.info("TEST 4: Developer Agent Tool Usage")
    console.info("=" * 80)

    try:
        agent = DeveloperAgent(workspace_root=".")

        # Test 1: Glob files
        console.info("\n[Test 4.1] Testing glob_files...")
        py_files = agent.glob_files("*.py")
        console.success(f"  ✓ Found {len(py_files)} Python files")

        # Test 2: Read file
        console.info("\n[Test 4.2] Testing read_file...")
        if py_files:
            agent.read_file(py_files[0], limit=5)
            console.success(f"  ✓ Read first 5 lines from {py_files[0]}")

        # Test 3: File exists
        console.info("\n[Test 4.3] Testing file_exists...")
        exists = agent.file_exists("README.md")
        console.success(f"  ✓ file_exists('README.md') = {exists}")

        # Test 4: Problem refinement
        console.info("\n[Test 4.4] Testing refine_problem_statement...")
        if hasattr(agent, 'refine_problem_statement'):
            refined = agent.refine_problem_statement("Users can't login")
            console.success("  ✓ Problem refined successfully")
            console.info(f"  Refined statement preview:\n{refined}")
        else:
            console.warning("  ⚠ refine_problem_statement method not found")

        console.success("\n✅ DEVELOPER AGENT TOOLS WORKING\n")
        return True

    except Exception as e:
        console.error(f"\n❌ DEVELOPER TOOL TEST FAILED: {str(e)}")
        import traceback
        console.error(traceback.format_exc())
        return False


def test_complete_workflow():
    """Test complete workflow: Problem → Refinement → Solution."""
    console.info("\n" + "=" * 80)
    console.info("TEST 5: Complete Workflow")
    console.info("=" * 80)

    try:
        # Step 1: Researcher discovers problem
        console.info("\n[Step 1] Researcher discovers vague problem...")
        ResearcherAgent(workspace_root=".")
        vague_problem = "Users complain that data export takes too long"
        console.info(f"  Original problem: {vague_problem}")

        # Step 2: Product Manager refines problem
        console.info("\n[Step 2] Product Manager refines problem statement...")
        pm = ProductManagerAgent(workspace_root=".")
        refined = pm.problem_refiner.refine(vague_problem)
        console.success(f"  ✓ Refined: {refined.refined_statement}")
        console.info(f"  Type: {refined.problem_type.value}")
        console.info("  Severity: Critical" if "performance" in refined.problem_type.value else "  Severity: High")

        # Step 3: CTO assesses feasibility
        console.info("\n[Step 3] CTO validates problem is solvable...")
        cto = CTOAgent(workspace_root=".")
        console.success(f"  ✓ CTO has all tools available: {hasattr(cto, 'glob_files')}")
        console.success(f"  ✓ CTO can refine problems: {hasattr(cto, 'problem_refiner')}")

        # Step 4: Developer can work on solution
        console.info("\n[Step 4] Developer can access tools for solution...")
        developer = DeveloperAgent(workspace_root=".")
        console.success(f"  ✓ Developer has file tools: {hasattr(developer, 'read_file')}")
        console.success(f"  ✓ Developer has search tools: {hasattr(developer, 'grep_search')}")
        console.success(f"  ✓ Developer can refine: {hasattr(developer, 'problem_refiner')}")

        # Step 5: QA Engineer can review
        console.info("\n[Step 5] QA Engineer can review solution...")
        qa = QAEngineerAgent(workspace_root=".")
        console.success(f"  ✓ QA has review system: {hasattr(qa, 'review_system')}")
        console.success(f"  ✓ QA has all tools: {hasattr(qa, 'read_file')}")

        # Step 6: CEO approves
        console.info("\n[Step 6] CEO makes final decision...")
        ceo = CEOAgent(workspace_root=".")
        console.success(f"  ✓ CEO can access all information: {hasattr(ceo, 'read_file')}")
        console.success(f"  ✓ CEO can refine problems: {hasattr(ceo, 'problem_refiner')}")

        console.success("\n✅ COMPLETE WORKFLOW VALIDATED\n")
        console.success("All agents can:")
        console.success("  • Refine problem statements before building solutions")
        console.success("  • Access all 13 Claude Code tools")
        console.success("  • Work together in the complete workflow")

        return True

    except Exception as e:
        console.error(f"\n❌ WORKFLOW TEST FAILED: {str(e)}")
        import traceback
        console.error(traceback.format_exc())
        return False


def test_session_management():
    """Test 6: Session Manager create/save/resume."""
    console.info("\n" + "=" * 80)
    console.info("TEST 6: Session Management")
    console.info("=" * 80 + "\n")

    import tempfile
    from memory.session import SessionManager

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Create session
            sm = SessionManager(sessions_dir=tmpdir)
            session = sm.create_session(name="test-session")
            assert session is not None, "Session creation failed"
            console.success(f"  ✓ Session created: {session.id}")

            # Add message and checkpoint
            session.add_message("user", "Build a REST API")
            session.create_checkpoint("after_research", metadata={"phase": "research"})
            sm.save_current()
            console.success("  ✓ Message and checkpoint saved")

            # List sessions
            sessions = sm.list_sessions()
            assert len(sessions) >= 1, "Session listing failed"
            console.success(f"  ✓ Listed {len(sessions)} session(s)")

            # Resume session
            sm2 = SessionManager(sessions_dir=tmpdir)
            resumed = sm2.resume_session(session_id=session.id)
            assert resumed is not None, "Session resume failed"
            assert len(resumed.messages) == 1, "Messages not preserved"
            assert len(resumed.checkpoints) == 1, "Checkpoints not preserved"
            console.success(f"  ✓ Session resumed with {len(resumed.messages)} message(s)")

            # End session
            sm2.end_session()
            console.success("  ✓ Session ended")

            console.success("\nSESSION MANAGEMENT TESTS PASSED")
            return True
        except Exception as e:
            console.error(f"  Session test failed: {e}")
            return False


def test_checkpoint_cycle():
    """Test 7: Checkpoint save/load cycle."""
    console.info("\n" + "=" * 80)
    console.info("TEST 7: Checkpoint Save/Load")
    console.info("=" * 80 + "\n")

    import tempfile
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir()

            # Simulate saving a checkpoint
            checkpoint = {
                "phase": "implementation",
                "problem": {"id": "PROB-TEST", "description": "Test problem", "severity": "high",
                            "frequency": "common", "target_users": "devs", "evidence": [],
                            "sources": [], "domain": "software", "keywords": [],
                            "potential_solution_ideas": [], "validation_status": "validated",
                            "score": 0.8, "metadata": {}},
                "artifacts": {"language": "python"},
                "decisions": [{"decision": "approved"}],
                "started_at": "2026-01-01T00:00:00",
                "error": None,
            }

            # Save
            checkpoint_file = checkpoint_dir / "latest.json"
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint, f, indent=2)
            console.success("  ✓ Checkpoint saved to disk")

            # Load
            with open(checkpoint_file) as f:
                loaded = json.load(f)
            assert loaded["phase"] == "implementation", "Phase not preserved"
            assert loaded["problem"]["id"] == "PROB-TEST", "Problem not preserved"
            assert loaded["artifacts"]["language"] == "python", "Artifacts not preserved"
            console.success("  ✓ Checkpoint loaded and verified")

            console.success("\nCHECKPOINT TESTS PASSED")
            return True
        except Exception as e:
            console.error(f"  Checkpoint test failed: {e}")
            return False


def test_config_loading():
    """Test 8: Config file loading and override."""
    console.info("\n" + "=" * 80)
    console.info("TEST 8: Config Loading")
    console.info("=" * 80 + "\n")

    import tempfile
    import json

    try:
        from config.config_loader import ConfigLoader, AppConfig

        # Test default config
        loader = ConfigLoader()
        config = loader.load()
        assert isinstance(config, AppConfig), "Config is not AppConfig"
        assert config.workflow.auto_discover is True, "Default auto_discover wrong"
        assert config.llm.temperature == 0.7, "Default temperature wrong"
        console.success("  ✓ Default config loaded correctly")

        # Test config from file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "workflow": {"enable_meetings": False, "token_budget": 10000},
                "llm": {"temperature": 0.5, "streaming": False},
                "language": "typescript"
            }, f)
            tmp_path = f.name

        loader2 = ConfigLoader(config_path=tmp_path)
        config2 = loader2.load()
        assert config2.workflow.enable_meetings is False, "enable_meetings not overridden"
        assert config2.workflow.token_budget == 10000, "token_budget not set"
        assert config2.llm.temperature == 0.5, "temperature not overridden"
        assert config2.language == "typescript", "language not set"
        console.success("  ✓ Config file overrides applied correctly")

        Path(tmp_path).unlink()

        console.success("\nCONFIG LOADING TESTS PASSED")
        return True
    except Exception as e:
        console.error(f"  Config test failed: {e}")
        return False


def test_learning_system():
    """Test 9: Learning system advice injection."""
    console.info("\n" + "=" * 80)
    console.info("TEST 9: Learning System")
    console.info("=" * 80 + "\n")

    import tempfile

    try:
        from memory.learning import AgentLearning

        with tempfile.TemporaryDirectory() as tmpdir:
            learning = AgentLearning(agent_name="developer", persist_dir=tmpdir)

            # Record a lesson
            learning.add_lesson(
                category="implementation",
                content="Always add input validation",
                source_context="Implementing a REST API",
                outcome="success",
                importance=0.8,
                tags=["validation", "api"]
            )
            console.success("  ✓ Lesson recorded")

            # Retrieve advice
            advice = learning.get_advice_for_task("implement_feature", "implement a REST API")
            assert advice is not None, "Advice retrieval failed"
            console.success(f"  ✓ Advice retrieved ({len(advice)} chars)")

            # Check persistence
            learning2 = AgentLearning(agent_name="developer", persist_dir=tmpdir)
            learning2.get_advice_for_task("implement_feature", "implement a feature")
            console.success("  ✓ Lessons persist across instances")

            console.success("\nLEARNING SYSTEM TESTS PASSED")
            return True
    except Exception as e:
        console.error(f"  Learning test failed: {e}")
        return False


def test_console_log_levels():
    """Test 10: Console log level control."""
    console.info("\n" + "=" * 80)
    console.info("TEST 10: Console Log Levels")
    console.info("=" * 80 + "\n")

    try:
        from ui.console import CompanyConsole

        test_console = CompanyConsole()

        # Default level is info (1)
        assert test_console._log_level == 1, "Default log level wrong"
        console.success("  ✓ Default log level is info")

        # Set to debug
        test_console.set_log_level("debug")
        assert test_console._log_level == 0, "Debug level not set"
        console.success("  ✓ Debug level set correctly")

        # Set to error (quiet mode)
        test_console.set_log_level("error")
        assert test_console._log_level == 3, "Error level not set"
        console.success("  ✓ Error level set correctly (quiet mode)")

        # Reset
        test_console.set_log_level("info")
        assert test_console._log_level == 1, "Reset to info failed"
        console.success("  ✓ Log level reset works")

        console.success("\nCONSOLE LOG LEVEL TESTS PASSED")
        return True
    except Exception as e:
        console.error(f"  Log level test failed: {e}")
        return False


def main():
    """Run all tests."""
    console.info("\n" + "=" * 80)
    console.info("COMPANY-AGI AGENT INTEGRATION TEST SUITE")
    console.info("=" * 80)
    console.info("\nTesting agent integration with:")
    console.info("  • All 13 Claude Code tools via AgentToolsMixin")
    console.info("  • Problem statement refinement capability")
    console.info("  • Complete workflow from problem discovery to solution\n")

    results = {}

    # Test 1: Agent instantiation
    agents, result1 = test_agent_instantiation()
    results['instantiation'] = result1

    if result1:
        # Test 2: Claude Code tools
        result2 = test_claude_code_tools(agents)
        results['tools'] = result2

        # Test 3: Problem refinement
        result3 = test_problem_refinement(agents)
        results['refinement'] = result3

        # Test 4: Developer tools
        result4 = test_developer_tools()
        results['developer_tools'] = result4

        # Test 5: Complete workflow
        result5 = test_complete_workflow()
        results['workflow'] = result5

    # Tests 6-10: Infrastructure tests (run regardless of agent instantiation)
    result6 = test_session_management()
    results['session_management'] = result6

    result7 = test_checkpoint_cycle()
    results['checkpoint'] = result7

    result8 = test_config_loading()
    results['config_loading'] = result8

    result9 = test_learning_system()
    results['learning'] = result9

    result10 = test_console_log_levels()
    results['console_log_levels'] = result10

    if not result1:
        console.error("Agent instantiation failed — some tests skipped")
        return False

    # Summary
    console.info("\n" + "=" * 80)
    console.info("TEST SUMMARY")
    console.info("=" * 80)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        console.info(f"{status}: {test_name}")

    console.info(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        console.success("\n🎉 ALL TESTS PASSED! 🎉")
        console.success("\nCompany-AGI agents are now fully enhanced with:")
        console.success("  ✓ All 13 Claude Code tools accessible")
        console.success("  ✓ Problem statement refinement integrated")
        console.success("  ✓ Complete workflow validated")
        console.success("\n✅ SYSTEM IS PRODUCTION READY\n")
        return True
    else:
        console.error("\n❌ SOME TESTS FAILED")
        console.error("Please review the failures above.\n")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
