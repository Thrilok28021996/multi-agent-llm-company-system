"""Phase: quality

Extracted from orchestrator/workflow.py.
Standalone functions — first param is the workflow instance.
"""

from pathlib import Path

from collaboration.meeting import MeetingType, quick_meeting
from orchestrator.phase_types import (
    CONFIDENCE_CRITICAL,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    WorkflowPhase,
)
from ui.console import console


async def _run_qa_phase(workflow) -> bool:
    """QA Engineer validates the solution with a review meeting."""
    workflow.state.phase = WorkflowPhase.QA_VALIDATION
    workflow.structured_logger.info("QA validation phase started")

    # Track plan step (step_3 = QA Validation)
    if workflow.plan_manager.current_plan:
        workflow.plan_manager.start_step("step_3")

    problem = workflow.state.current_problem
    if not problem:
        console.error("No problem for QA phase")
        return False

    implementation = workflow.state.artifacts.get("implementation", {})
    requirements = workflow.state.artifacts.get("requirements", {})

    # Create test plan
    test_plan_task = {
        "type": "create_test_plan",
        "feature": {
            "name": problem.description,
            "description": problem.description
        },
        "requirements": str(requirements)
    }
    test_plan_result = await workflow.qa_engineer.execute_task(test_plan_task)
    workflow.state.artifacts["test_plan"] = test_plan_result.output

    # Hold a review meeting
    review_participants = [
        {
            "name": "QAEngineer",
            "role": "Quality Assurance",
            "model": workflow.model_config.get_model_name("qa_engineer")
        },
        {
            "name": "Developer",
            "role": "Implementation",
            "model": workflow.model_config.get_model_name("developer")
        },
        {
            "name": "CTO",
            "role": "Technical Review",
            "model": workflow.model_config.get_model_name("cto")
        }
    ]

    execution_results = workflow.state.artifacts.get("execution_results", {})
    execution_summary = execution_results.get("summary", "Not run")

    dev_confidence = workflow.state.artifacts.get("developer_confidence")
    confidence_note = (
        f"\nDeveloper Self-Confidence: {dev_confidence:.0%} — "
        "FOCUS testing on areas the developer flagged as uncertain."
        if dev_confidence is not None and dev_confidence < 0.75 else ""
    )

    cto_peer_review = workflow.state.artifacts.get("cto_peer_review", "")
    cto_peer_note = f"\nCTO Peer Review Issues: {cto_peer_review[:400]}" if cto_peer_review else ""
    review_context = f"""
Problem Solved: {problem.description}
Requirements: {str(requirements)}
Implementation Summary: {str(implementation)}
Test Plan: {str(test_plan_result.output)}
Code Execution Results: {execution_summary}{confidence_note}{cto_peer_note}
"""
    if workflow.enable_meetings:
        review_outcome = await quick_meeting(
            topic="Code Review and Quality Assessment",
            participants=review_participants,
            meeting_type=MeetingType.REVIEW,
            context=review_context
        )

        workflow.state.artifacts["qa_review_meeting"] = {
            "decision": review_outcome.decision,
            "votes": review_outcome.votes,
            "action_items": review_outcome.action_items
        }

        # Store meeting in shared memory
        workflow.shared_memory.store_meeting(
            topic="Code Review and Quality Assessment",
            participants=["QAEngineer", "Developer", "CTO"],
            summary=review_outcome.decision,
            action_items=review_outcome.action_items
        )
        review_feedback = review_outcome.decision
    else:
        console.info("Meetings disabled — skipping QA review meeting")
        review_feedback = "Review meeting skipped (meetings disabled)"
        review_outcome = None
        workflow.state.artifacts["qa_review_meeting"] = {"decision": review_feedback, "meetings_disabled": True}

    # Determine review depth based on trust between QA and Developer
    review_depth = workflow.trust_tracker.get_review_threshold("QAEngineer", "Developer")
    depth_instructions = {
        "light": "Quick review — focus on critical functionality only. The developer has a strong track record.",
        "standard": "Standard review — check functionality and code quality.",
        "thorough": "Deep review — check edge cases, error handling, and logic flow. Previous work had quality issues.",
    }
    depth_note = depth_instructions.get(review_depth, depth_instructions["standard"])
    console.info(f"Review depth: {review_depth}")

    # Part 4b: Include root cause context for QA verification
    root_cause = workflow.state.artifacts.get("root_cause", "")
    qa_requirements_str = str(requirements)
    if root_cause:
        qa_requirements_str += f"\n\nVerify the implementation addresses this root cause: {root_cause}"

    # Validate solution
    dev_confidence_val = workflow.state.artifacts.get("developer_confidence")
    validation_task = {
        "type": "validate_solution",
        "solution": {
            "description": problem.description,
            "implementation": str(implementation)
        },
        "requirements": qa_requirements_str,
        "requirements_doc": workflow.state.artifacts.get("requirements_doc", {}),
        "problem": problem.to_dict(),
        "review_feedback": review_feedback,
        "review_depth": review_depth,
        "review_depth_instruction": depth_note,
        "execution_results": workflow.state.artifacts.get("execution_results", {}),
        "developer_confidence": dev_confidence_val,
        "cto_peer_review": cto_peer_review,
        "context_messages": workflow._drain_messages_for_agent("QAEngineer")
    }
    validation_result = await workflow.qa_engineer.execute_task(validation_task)
    # Wire first_principles_score from QA onto result
    validation_result.first_principles_score = getattr(workflow.qa_engineer, '_last_fp_score', 1.0)
    workflow.state.artifacts["qa_fp_score"] = validation_result.first_principles_score
    workflow.state.artifacts["qa_validation"] = validation_result.output
    # Store typed QAReport if produced
    if isinstance(validation_result.output, dict) and "qa_report" in validation_result.output:
        workflow.state.artifacts["qa_report_typed"] = validation_result.output["qa_report"]
        qa_report_dict = validation_result.output["qa_report"]
        passed = len([r for r in qa_report_dict.get("criterion_results", []) if r.get("status") == "PASS"])
        total = len(qa_report_dict.get("criterion_results", []))
        if total:
            console.info(f"QAReport: {passed}/{total} acceptance criteria passed")

    # Generate QA report (execution_results already fetched above)
    report_task = {
        "type": "generate_qa_report",
        "project": problem.id,
        "test_results": test_plan_result.output,
        "issues": review_outcome.action_items if review_outcome else [],
        "execution_results": execution_results
    }
    report_result = await workflow.qa_engineer.execute_task(report_task)
    workflow.state.artifacts["qa_report"] = report_result.output

    # Handle case where output might be a string instead of dict
    qa_output = validation_result.output if isinstance(validation_result.output, dict) else {"response": str(validation_result.output)}
    verdict = qa_output.get("verdict", "fail")
    if workflow.enable_meetings and review_outcome:
        approvals = sum(1 for v in review_outcome.votes.values() if v == "approve")
        console.info(f"Review Meeting Approvals: {approvals}/{len(review_outcome.votes)}")
    console.info(f"QA Verdict: {verdict.upper()}")

    # Track performance: QA task and Developer's work quality
    qa_passed = verdict in ["pass", "pass_with_issues"]
    workflow.performance_tracker.record_task(
        "QAEngineer", success=True,
        response_time_ms=validation_result.execution_time * 1000,
        tokens=0
    )
    workflow.performance_tracker.record_approval("Developer", approved=qa_passed)
    if not qa_passed:
        workflow.performance_tracker.record_rework("Developer")

    # Update trust: QA reviewed Developer's work
    # High-confidence reviews are a stronger signal for trust
    qa_confidence = validation_result.confidence
    if qa_confidence >= CONFIDENCE_HIGH:
        workflow.trust_tracker.update_trust("QAEngineer", "Developer", positive=qa_passed)
    else:
        workflow.trust_tracker.update_trust("QAEngineer", "Developer", positive=qa_passed)

    # Update experience
    if hasattr(workflow.qa_engineer, '_experience'):
        workflow.qa_engineer._experience.add_experience("validate_solution", True)
    if hasattr(workflow.developer, '_experience'):
        workflow.developer._experience.add_experience("implementation", qa_passed)

    # Graduated confidence response (OrgChart integration)
    if qa_confidence < CONFIDENCE_CRITICAL:
        supervisor = workflow.org_chart.get_supervisor("QAEngineer")
        console.warning(f"QA confidence CRITICAL ({qa_confidence:.2f}) — supervisor {supervisor} will re-review")
        workflow.performance_tracker.record_task("QAEngineer", success=False, response_time_ms=0, tokens=0)
    elif qa_confidence < CONFIDENCE_LOW:
        supervisor = workflow.org_chart.get_supervisor("QAEngineer")
        if supervisor:
            console.info(f"QA confidence low ({qa_confidence:.2f}) — supervisor {supervisor} notified")
    elif qa_confidence >= CONFIDENCE_HIGH:
        console.info(f"QA high-confidence verdict ({qa_confidence:.2f})")

    # Reflect on QA validation (learning)
    if workflow.enable_learning:
        workflow.qa_engineer.reflect(
            "validate_solution",
            str(validation_result.output),
            qa_passed
        )

    # Update plan step
    if workflow.plan_manager.current_plan:
        if qa_passed:
            workflow.plan_manager.complete_step("step_3", output=f"QA verdict: {verdict}")
        else:
            workflow.plan_manager.fail_step("step_3", error=f"QA verdict: {verdict}")

    return qa_passed


def _collect_review_files(workflow, project_path: Path) -> dict:
    """Snapshot the project's Python files for the security review."""
    files = {}
    if not project_path.exists():
        return files
    py_files = workflow._filter_project_files(list(project_path.rglob("*.py")))
    for f in py_files[:10]:  # Limit to 10 files for token efficiency
        try:
            files[str(f.relative_to(project_path))] = f.read_text()
        except Exception:
            pass
    return files


def _build_security_review_task(workflow, project_dir) -> dict:
    """Build the security review task from the *current* code and files.

    Called again after a developer security fix so the Security Engineer
    re-reviews the updated snapshot instead of the stale pre-fix one.
    """
    implementation = workflow.state.artifacts.get("implementation", {})
    project_path = Path(project_dir)
    return {
        "type": "security_review",
        "code": str(implementation),
        "files": _collect_review_files(workflow, project_path),
        "file_path": project_dir,
        "context": f"Problem: {workflow.state.current_problem.description if workflow.state.current_problem else 'N/A'}",
    }


# Verdicts SecurityEngineer._parse_security_verdict can emit for a
# security_review task. Anything else — a failed TaskResult, a non-dict
# output, or a missing/unknown verdict — must fail the phase rather than
# silently pass.
_SECURITY_VERDICT_CRITICAL = "critical_vulnerabilities"
_SECURITY_VERDICTS_PASS = frozenset({"secure", "has_issues"})


def _classify_security_review(review_result) -> str:
    """Classify a SecurityEngineer review as 'critical', 'pass' or 'failed'.

    Fails closed: an unsuccessful TaskResult or a missing/unrecognized
    verdict is reported as 'failed', never as a pass.
    """
    if not getattr(review_result, "success", False):
        return "failed"
    output = review_result.output if isinstance(review_result.output, dict) else {}
    verdict = output.get("verdict")
    if verdict == _SECURITY_VERDICT_CRITICAL:
        return "critical"
    if verdict in _SECURITY_VERDICTS_PASS:
        return "pass"
    return "failed"


async def _run_security_review_phase(workflow) -> bool:
    """Security Engineer reviews the solution code for vulnerabilities."""
    workflow.state.phase = WorkflowPhase.SECURITY_REVIEW
    workflow.structured_logger.info("Security review phase started")

    project_dir = workflow.state.artifacts.get("project_name", workflow.output_dir)

    # Security review
    review_task = _build_security_review_task(workflow, project_dir)
    console.agent_action("SecurityEngineer", "Security Review", f"Reviewing {len(review_task['files'])} files...")
    review_result = await workflow.security_engineer.execute_task(review_task)
    workflow.state.artifacts["security_review"] = review_result.output
    # Track SecurityEngineer experience and pool performance
    if hasattr(workflow.security_engineer, '_experience'):
        workflow.security_engineer._experience.add_experience("security_review", review_result.success)
    workflow.agent_pool.record_task_outcome("SecurityEngineer", review_result.success)
    workflow.performance_tracker.record_task(
        "SecurityEngineer", success=review_result.success,
        response_time_ms=review_result.execution_time * 1000, tokens=0
    )
    # Reflect on Security review (learning)
    if workflow.enable_learning:
        workflow.security_engineer.reflect(
            "security_review",
            str(review_result.output),
            review_result.success
        )

    # Parse verdict — an unsuccessful TaskResult or a missing/unknown
    # verdict must fail the phase, not pass it.
    output = review_result.output if isinstance(review_result.output, dict) else {}
    verdict = output.get("verdict", "unknown")
    critical_count = output.get("critical_count", 0)

    console.info(f"Security verdict: {str(verdict).upper()}, {critical_count} critical vulnerabilities")

    review_state = _classify_security_review(review_result)

    if review_state == "critical":
        console.error("Critical security vulnerabilities found! Must be fixed before deployment.")
        # Feed security issues to Developer for fixing
        project_dir = workflow.state.artifacts.get("project_name", workflow.output_dir)
        sec_fix_task = {
            "type": "fix_bug",
            "bug_description": f"Critical security vulnerabilities found. Fix these: {output.get('review', '')}",
            "error_message": "",
            "qa_report": "",
            "original_requirements": str(workflow.state.artifacts.get("requirements", "")),
            "output_dir": project_dir
        }
        fix_result = await workflow.developer.execute_task(sec_fix_task)
        workflow.state.artifacts["security_fix"] = fix_result.output
        console.success("Developer applied security fixes")

        # Re-run security review on a rebuilt snapshot — the developer has
        # just changed files on disk, so the pre-fix snapshot is stale.
        review_task = _build_security_review_task(workflow, project_dir)
        review_result = await workflow.security_engineer.execute_task(review_task)
        workflow.state.artifacts["security_review"] = review_result.output
        if _classify_security_review(review_result) == "pass":
            console.success("Security issues resolved after fix")
            return True
        console.error("Security re-review did not pass after the fix — failing closed.")
        return False

    if review_state != "pass":
        console.error(
            f"Security review unusable (success={getattr(review_result, 'success', False)}, "
            f"verdict={verdict!r}) — failing closed."
        )
        return False

    return True


async def _check_quality_gate(workflow, phase: str) -> bool:
    """Check quality gate for a phase. Returns True if passed, False if failed.

    On failure, retries up to 2 times, then proceeds with a degradation flag.
    """
    max_retries = 2

    for attempt in range(max_retries + 1):
        passed, reason = workflow._evaluate_gate(phase)
        if passed:
            return True

        if attempt < max_retries:
            console.warning(f"Quality gate '{phase}' failed (attempt {attempt + 1}/{max_retries + 1}): {reason}")
            retry_ok = await workflow._retry_gate(phase, reason)
            if not retry_ok:
                break
        else:
            console.warning(f"Quality gate '{phase}' failed after {max_retries + 1} attempts: {reason}")

    # Proceed with degradation flag
    workflow.state.artifacts.setdefault("degraded_phases", []).append(phase)
    console.warning(f"Proceeding with degraded quality for phase: {phase}")
    return False


def _evaluate_gate(workflow, phase: str) -> tuple:
    """Evaluate a quality gate. Returns (passed: bool, reason: str)."""
    if phase == "research":
        problems = workflow.state.artifacts.get("discovered_problems", [])
        user_provided = workflow.state.artifacts.get("user_provided_problem", False)
        min_required = 1 if user_provided else 2
        if len(problems) < min_required:
            return False, f"Only {len(problems)} problems found (need >= {min_required})"
        max_score = max((p.get("score", 0) if isinstance(p, dict) else getattr(p, "score", 0)) for p in problems)
        if not user_provided and max_score < 0.3:
            return False, f"Max problem score {max_score:.2f} < 0.3"
        return True, ""

    elif phase == "data_analysis":
        verdicts = workflow.state.artifacts.get("cross_validation_verdicts", {})
        summary = verdicts.get("summary", {}) if isinstance(verdicts, dict) else {}
        confirmed = summary.get("confirmed", 0) + summary.get("partially_confirmed", 0)
        if confirmed < 1:
            return False, "No findings confirmed or partially confirmed by DataAnalyst"
        return True, ""

    elif phase == "tech_design":
        root_cause = workflow.state.artifacts.get("root_cause", "")
        if not root_cause or len(root_cause.strip()) < 5:
            return False, "Root cause not extracted from CTO design"
        return True, ""

    elif phase == "implementation":
        impl = workflow.state.artifacts.get("implementation", {})
        if isinstance(impl, dict):
            files = impl.get("files_written", [])
        else:
            files = []
        code_artifacts = workflow.state.artifacts.get("project", {})
        if isinstance(code_artifacts, dict):
            files = files or code_artifacts.get("files_created", [])
        if not files:
            return False, "No code files in implementation artifacts"
        return True, ""

    elif phase == "qa":
        qa_output = workflow.state.artifacts.get("qa_validation", {})
        if isinstance(qa_output, dict):
            verdict = qa_output.get("verdict", "uncertain")
        else:
            verdict = "uncertain"
        if verdict == "uncertain":
            return False, "QA verdict is uncertain"
        return True, ""

    return True, ""


async def _retry_gate(workflow, phase: str, reason: str) -> bool:
    """Attempt to fix a quality gate failure. Returns True if retry was attempted."""
    if phase == "research":
        # Re-run with expanded keywords
        console.info("Re-running research with expanded keywords...")
        await workflow._discover_from_web_search()
        return True
    elif phase == "data_analysis":
        # Warn and proceed - can't easily re-run data analysis
        console.warning(f"Data analysis gate: {reason} — proceeding with caution")
        return False
    elif phase == "tech_design":
        # CTO retry
        console.info("Re-running CTO design to extract root cause...")
        problem = workflow.state.current_problem
        if problem:
            retry_task = {
                "type": "design_architecture",
                "problem": problem.to_dict(),
                "requirements": str(workflow.state.artifacts.get("requirements", "")),
                "context_messages": [],
            }
            result = await workflow.cto.execute_task(retry_task)
            workflow.state.artifacts["architecture"] = result.output
            # Re-extract root cause
            from utils.output_parser import StructuredOutputParser
            parser = StructuredOutputParser()
            design_text = result.artifacts.get("architecture_doc", str(result.output))
            parsed = parser.parse_cto_design(design_text)
            rc = parsed.get("root_cause", "")
            if rc:
                workflow.state.artifacts["root_cause"] = rc
            return True
        return False
    elif phase == "implementation":
        # Developer retry
        console.info("Re-running implementation...")
        return True  # The main loop handles this
    elif phase == "qa":
        # QA retry with clearer instructions
        console.info("Re-running QA with clearer instructions...")
        return True
    return False
