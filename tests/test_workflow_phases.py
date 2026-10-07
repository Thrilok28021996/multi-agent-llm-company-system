"""Mocked regression tests for the phase functions extracted from workflow.py.

No LLM calls: every agent/trackers collaborator on the workflow object is a
mock, so these tests only exercise the extracted phase entry points, the
quality gates, interactive mode, the security-review retry and delivery
reporting.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from orchestrator.phase_types import WorkflowPhase
from orchestrator.phases import analysis as analysis_phase
from orchestrator.phases import approval as approval_phase
from orchestrator.phases import delivery as delivery_phase
from orchestrator.phases import helpers as helpers_phase
from orchestrator.phases import quality as quality_phase
from orchestrator.phases import research as research_phase

PHASE_MODULES = (
    analysis_phase,
    approval_phase,
    delivery_phase,
    helpers_phase,
    quality_phase,
    research_phase,
)


@pytest.fixture(autouse=True)
def quiet_console(monkeypatch):
    """Silence (and expose) the module-level console used by the phases."""
    mocks = {}
    for module in PHASE_MODULES:
        console_mock = MagicMock()
        monkeypatch.setattr(module, "console", console_mock)
        mocks[module.__name__] = console_mock
    return mocks


def make_workflow(tmp_path=None):
    """Build a fake workflow object with mocked collaborators."""
    root = Path(tmp_path) if tmp_path else Path.cwd()
    wf = MagicMock()
    wf.state = SimpleNamespace(
        phase=WorkflowPhase.IDLE,
        current_problem=None,
        artifacts={},
        decisions=[],
        started_at=None,
    )
    wf.output_dir = str(root)
    wf.workspace_root = str(root)
    wf.plan_manager = MagicMock()
    wf.plan_manager.current_plan = None
    wf.structured_logger = MagicMock()
    wf.enable_learning = False
    wf.enable_meetings = False
    wf._interactive = False
    wf._filter_project_files = lambda files: list(files)
    wf._drain_messages_for_agent = lambda name: ""
    return wf


def make_problem():
    return SimpleNamespace(
        id="P1",
        description="Build a thing that does the thing",
        evidence=["evidence one", "evidence two"],
        score=0.8,
        severity="high",
        target_users="developers",
        discovered_at=None,
        to_dict=lambda: {"id": "P1", "description": "Build a thing that does the thing"},
    )


def task_result(output=None, success=True, confidence=0.9):
    return SimpleNamespace(
        output=output if output is not None else {},
        success=success,
        confidence=confidence,
        execution_time=0.1,
        artifacts={},
    )


# ============================================================
#  Extracted phase entry points
# ============================================================


def test_phase_entry_points_are_exported_and_callable():
    from orchestrator import phases

    assert phases.__all__
    for name in phases.__all__:
        assert callable(getattr(phases, name)), name


@pytest.mark.asyncio
async def test_analysis_entry_point_runs_with_mocks():
    wf = make_workflow()
    wf.state.current_problem = make_problem()
    wf.researcher.execute_task = AsyncMock(return_value=task_result({"status": "validated"}))
    wf.product_manager.execute_task = AsyncMock(return_value=task_result({"status": "analyzed"}))

    await analysis_phase._run_analysis_phase(wf)

    assert wf.state.phase == WorkflowPhase.ANALYSIS
    assert wf.state.artifacts["problem_validation"] == {"status": "validated"}
    assert wf.state.artifacts["problem_analysis"] == {"status": "analyzed"}
    wf.researcher.execute_task.assert_awaited_once()
    wf.product_manager.execute_task.assert_awaited_once()


@pytest.mark.asyncio
async def test_analysis_entry_point_without_problem_returns_early():
    wf = make_workflow()

    await analysis_phase._run_analysis_phase(wf)

    assert wf.state.phase == WorkflowPhase.ANALYSIS
    assert wf.state.artifacts == {}
    assert analysis_phase.console.error.called


@pytest.mark.asyncio
async def test_research_entry_point_runs_with_mocks():
    wf = make_workflow()
    wf.research_sources.get_reddit_subreddits = MagicMock(return_value=["testing"])
    wf.problem_discoverer.discover_from_reddit = AsyncMock(return_value=[])
    wf.problem_discoverer.discover_from_hacker_news = AsyncMock(return_value=[])
    wf.problem_discoverer.discover_from_stackoverflow = AsyncMock(return_value=[])
    wf.problem_discoverer.discover_from_github_issues = AsyncMock(return_value=[])
    wf.problem_discoverer.filter_problems_with_solutions = AsyncMock(side_effect=lambda p: p)
    wf.shared_memory.get_by_type = MagicMock(return_value=[])
    wf._discover_from_web_search = AsyncMock(return_value=[])

    await research_phase._run_research_phase(wf)

    assert wf.state.phase == WorkflowPhase.RESEARCH
    assert wf.state.artifacts["discovered_problems"] == []
    wf.problem_discoverer.discover_from_reddit.assert_awaited_once()


@pytest.mark.asyncio
async def test_ceo_approval_entry_point_returns_decision():
    wf = make_workflow()
    wf.state.current_problem = make_problem()
    wf.ceo.execute_task = AsyncMock(return_value=task_result({"decision": "approved"}, confidence=0.95))

    approved = await approval_phase._run_ceo_approval(wf)

    assert approved is True
    assert wf.state.phase == WorkflowPhase.CEO_APPROVAL
    assert wf.state.artifacts["final_approval"] == {"decision": "approved"}
    assert wf.state.decisions[-1]["decision"] == "approved"


@pytest.mark.asyncio
async def test_ceo_approval_rejects_rejection():
    wf = make_workflow()
    wf.state.current_problem = make_problem()
    wf.ceo.execute_task = AsyncMock(return_value=task_result({"decision": "rejected"}, confidence=0.4))

    approved = await approval_phase._run_ceo_approval(wf)

    assert approved is False


@pytest.mark.asyncio
async def test_qa_entry_point_runs_with_mocks():
    wf = make_workflow()
    wf.state.current_problem = make_problem()
    wf.trust_tracker.get_review_threshold = MagicMock(return_value="standard")
    wf.qa_engineer = MagicMock()
    wf.qa_engineer._last_fp_score = 1.0
    wf.qa_engineer.execute_task = AsyncMock(
        side_effect=[
            task_result({"criterion_results": [{"status": "PASS"}]}),
            task_result({"verdict": "pass"}),
            task_result({"summary": "all good"}),
        ]
    )

    passed = await quality_phase._run_qa_phase(wf)

    assert passed is True
    assert wf.state.phase == WorkflowPhase.QA_VALIDATION
    assert wf.state.artifacts["qa_validation"] == {"verdict": "pass"}
    assert wf.state.artifacts["qa_report"] == {"summary": "all good"}
    assert wf.state.artifacts["test_plan"] == {"criterion_results": [{"status": "PASS"}]}


# ============================================================
#  Quality gates
# ============================================================


def test_evaluate_gate_research_needs_minimum_problems():
    wf = make_workflow()
    wf.state.artifacts["discovered_problems"] = [{"score": 0.7}]

    passed, reason = quality_phase._evaluate_gate(wf, "research")

    assert not passed
    assert "need >=" in reason


def test_evaluate_gate_research_passes_with_enough_problems():
    wf = make_workflow()
    wf.state.artifacts["discovered_problems"] = [{"score": 0.7}, {"score": 0.5}]

    passed, _ = quality_phase._evaluate_gate(wf, "research")

    assert passed


def test_evaluate_gate_unknown_phase_passes():
    wf = make_workflow()
    assert quality_phase._evaluate_gate(wf, "something_new") == (True, "")


@pytest.mark.asyncio
async def test_check_quality_gate_retries_then_passes():
    wf = make_workflow()
    attempts = iter([(False, "not enough problems"), (True, "")])
    wf._evaluate_gate = MagicMock(side_effect=lambda phase: next(attempts))
    wf._retry_gate = AsyncMock(return_value=True)

    passed = await quality_phase._check_quality_gate(wf, "research")

    assert passed is True
    assert wf._retry_gate.await_count == 1
    assert "degraded_phases" not in wf.state.artifacts


@pytest.mark.asyncio
async def test_check_quality_gate_degrades_after_exhausting_retries():
    wf = make_workflow()
    wf._evaluate_gate = MagicMock(return_value=(False, "still broken"))
    wf._retry_gate = AsyncMock(return_value=True)

    passed = await quality_phase._check_quality_gate(wf, "research")

    assert passed is False
    # Two retries across three attempts, then degrade rather than hang.
    assert wf._retry_gate.await_count == 2
    assert wf.state.artifacts["degraded_phases"] == ["research"]


@pytest.mark.asyncio
async def test_check_quality_gate_stops_when_retry_is_refused():
    wf = make_workflow()
    wf._evaluate_gate = MagicMock(return_value=(False, "broken"))
    wf._retry_gate = AsyncMock(return_value=False)

    passed = await quality_phase._check_quality_gate(wf, "research")

    assert passed is False
    assert wf._retry_gate.await_count == 1
    assert wf.state.artifacts["degraded_phases"] == ["research"]


# ============================================================
#  Interactive mode
# ============================================================


def test_interactive_gate_skips_when_not_interactive():
    """Regression: the extracted helper used `self` instead of `workflow`."""
    wf = SimpleNamespace()

    assert helpers_phase._interactive_gate(wf, "Research") is True


def test_interactive_gate_prompts_and_continues(monkeypatch):
    wf = SimpleNamespace(_interactive=True)
    asker = MagicMock()
    asker.ask.return_value = SimpleNamespace(skipped=False, timed_out=False, answer="y")
    monkeypatch.setattr("ui.interactive.AskUserQuestion", MagicMock(return_value=asker))

    assert helpers_phase._interactive_gate(wf, "Research") is True
    asker.ask.assert_called_once()


def test_interactive_gate_aborts_when_user_declines(monkeypatch):
    wf = SimpleNamespace(_interactive=True)
    asker = MagicMock()
    asker.ask.return_value = SimpleNamespace(skipped=False, timed_out=False, answer="")
    monkeypatch.setattr("ui.interactive.AskUserQuestion", MagicMock(return_value=asker))

    assert helpers_phase._interactive_gate(wf, "Implementation") is False
    assert helpers_phase.console.warning.called


def test_interactive_gate_aborts_when_prompt_is_skipped(monkeypatch):
    wf = SimpleNamespace(_interactive=True)
    asker = MagicMock()
    asker.ask.return_value = SimpleNamespace(skipped=True, timed_out=False, answer=None)
    monkeypatch.setattr("ui.interactive.AskUserQuestion", MagicMock(return_value=asker))

    assert helpers_phase._interactive_gate(wf, "Delivery") is False


# ============================================================
#  Security review retry (snapshot must be rebuilt after the fix)
# ============================================================


async def _run_security_review_with_fix(tmp_path, second_verdict="safe"):
    project = tmp_path / "proj"
    project.mkdir()
    source = project / "app.py"
    source.write_text("password = 'VULNERABLE'\n")

    wf = make_workflow(tmp_path)
    wf.state.current_problem = make_problem()
    wf.state.artifacts["project_name"] = str(project)
    wf.state.artifacts["implementation"] = {"files": ["app.py"]}

    reviews = []

    async def security_review(task):
        reviews.append(task)
        if len(reviews) == 1:
            return task_result(
                {"verdict": "critical_vulnerabilities", "critical_count": 1, "review": "hardcoded password"},
                success=False,
            )
        return task_result({"verdict": second_verdict, "critical_count": 0}, success=True)

    async def developer_fix(task):
        # The developer writes the fix to disk...
        source.write_text("password = os.environ['PASSWORD']\n")
        return task_result({"fixed": True})

    wf.security_engineer.execute_task = AsyncMock(side_effect=security_review)
    wf.developer.execute_task = AsyncMock(side_effect=developer_fix)

    result = await quality_phase._run_security_review_phase(wf)
    return result, reviews, wf


@pytest.mark.asyncio
async def test_security_review_rebuilds_snapshot_after_fix(tmp_path):
    result, reviews, wf = await _run_security_review_with_fix(tmp_path)

    assert result is True
    assert len(reviews) == 2
    # First pass saw the vulnerable code...
    assert "VULNERABLE" in reviews[0]["files"]["app.py"]
    # ...the retry must see the post-fix snapshot, not the stale one.
    assert "VULNERABLE" not in reviews[1]["files"]["app.py"]
    assert "os.environ" in reviews[1]["files"]["app.py"]
    assert reviews[0]["files"] != reviews[1]["files"]
    assert wf.state.artifacts["security_review"] == {"verdict": "safe", "critical_count": 0}


@pytest.mark.asyncio
async def test_security_review_fails_when_issues_remain(tmp_path):
    result, reviews, _ = await _run_security_review_with_fix(
        tmp_path, second_verdict="critical_vulnerabilities"
    )

    assert result is False
    assert len(reviews) == 2


@pytest.mark.asyncio
async def test_security_review_passes_without_fix_round_trip(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "app.py").write_text("safe = True\n")

    wf = make_workflow(tmp_path)
    wf.state.current_problem = make_problem()
    wf.state.artifacts["project_name"] = str(project)
    wf.security_engineer.execute_task = AsyncMock(
        return_value=task_result({"verdict": "safe", "critical_count": 0})
    )

    result = await quality_phase._run_security_review_phase(wf)

    assert result is True
    wf.security_engineer.execute_task.assert_awaited_once()
    wf.developer.execute_task.assert_not_called()


# ============================================================
#  Delivery: return codes must be checked
# ============================================================


async def _run_delivery(tmp_path, install_returncode, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "requirements.txt").write_text("requests\n")

    wf = make_workflow(tmp_path)
    wf.state.current_problem = make_problem()
    wf.state.artifacts["project_name"] = str(project)
    wf.developer.execute_task = AsyncMock(return_value=task_result({}))

    # Formatting and git init are out of interest here.
    fake_formatter = MagicMock()
    fake_formatter.format_directory.return_value = SimpleNamespace(files_formatted=[], errors=[])
    monkeypatch.setattr("tools.code_formatter.CodeFormatter", lambda **kwargs: fake_formatter)
    monkeypatch.setattr(
        delivery_phase,
        "GitTools",
        lambda repo_path: SimpleNamespace(is_repo=lambda: True),
    )

    def fake_run(cmd, **kwargs):
        joined = " ".join(str(c) for c in cmd)
        if "pip install" in joined:
            return SimpleNamespace(
                returncode=install_returncode,
                stdout="",
                stderr="ERROR: could not find a version" if install_returncode else "",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    delivery = await delivery_phase._run_delivery_phase(wf)
    return delivery, wf


@pytest.mark.asyncio
async def test_delivery_reports_failed_dependency_install(tmp_path, monkeypatch):
    delivery, wf = await _run_delivery(tmp_path, install_returncode=1, monkeypatch=monkeypatch)

    assert delivery["warnings"], "failed install must be reported"
    assert any("Dependency installation failed" in w for w in delivery["warnings"])

    success_messages = [str(c) for c in delivery_phase.console.success.call_args_list]
    assert not any("Python dependencies installed" in m for m in success_messages)
    assert delivery_phase.console.warning.called
    assert wf.state.artifacts["delivery"] is delivery


@pytest.mark.asyncio
async def test_delivery_reports_successful_dependency_install(tmp_path, monkeypatch):
    delivery, _ = await _run_delivery(tmp_path, install_returncode=0, monkeypatch=monkeypatch)

    assert delivery["warnings"] == []
    success_messages = [str(c) for c in delivery_phase.console.success.call_args_list]
    assert any("Python dependencies installed" in m for m in success_messages)
