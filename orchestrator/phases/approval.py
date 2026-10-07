"""Phase: approval

Extracted from orchestrator/workflow.py.
Standalone functions — first param is the workflow instance.
"""

from orchestrator.phase_types import (
    CONFIDENCE_CRITICAL,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    WorkflowPhase,
)
from ui.console import console


async def _run_ceo_approval(workflow) -> bool:
    """CEO gives final approval."""
    workflow.state.phase = WorkflowPhase.CEO_APPROVAL
    workflow.structured_logger.info("CEO approval phase started")

    # Track plan step (step_4 = CEO Approval)
    if workflow.plan_manager.current_plan:
        workflow.plan_manager.start_step("step_4")

    problem = workflow.state.current_problem
    if not problem:
        console.error("No problem for CEO approval")
        return False

    execution_results = workflow.state.artifacts.get("execution_results", {})
    execution_summary = execution_results.get("summary", "Not run")
    # Include DevOps verdict in CEO context
    devops_validation = workflow.state.artifacts.get("devops_validation", "Not run")
    devops_summary = str(devops_validation)[:500] if devops_validation else "Not run"
    # Include Security review findings
    security_review = workflow.state.artifacts.get("security_review", None)
    security_summary = str(security_review)[:500] if security_review else "Not run"

    impl_fp = workflow.state.artifacts.get("impl_fp_score", 1.0)
    qa_fp = workflow.state.artifacts.get("qa_fp_score", 1.0)
    quality_score = execution_results.get("quality_score", "N/A")
    quality_details = execution_results.get("quality_details", {})
    quality_str = (
        f"Quality={quality_score:.0%} "
        f"(TODOs={quality_details.get('todo_count',0)}, "
        f"long_funcs={quality_details.get('long_functions',0)}, "
        f"hardcoded={quality_details.get('hardcoded_values',0)})"
        if isinstance(quality_score, float) else "Quality=N/A"
    )
    approval_task = {
        "type": "approve_solution",
        "solution": {
            "description": problem.description,
            "implementation": str(workflow.state.artifacts.get("implementation", {}))
        },
        "qa_report": str(workflow.state.artifacts.get("qa_report", {})),
        "security_report": security_summary,
        "original_problem": problem.to_dict(),
        "execution_summary": (
            f"{execution_summary}\nDevOps Validation: {devops_summary}\n"
            f"Security Review: {security_summary}\n"
            f"First-Principles Scores: Developer={impl_fp:.0%}, QA={qa_fp:.0%}\n"
            f"Code {quality_str}"
        ),
        "root_cause": workflow.state.artifacts.get("root_cause", ""),
    }
    result = await workflow.ceo.execute_task(approval_task)

    # Handle case where output might be a string instead of dict
    output = result.output if isinstance(result.output, dict) else {"response": str(result.output)}
    decision = output.get("decision", "rejected")
    reasoning = output.get("reasoning", "")
    workflow.state.decisions.append({
        "phase": "final_approval",
        "decision": decision,
        "reasoning": reasoning
    })

    workflow.state.artifacts["final_approval"] = result.output
    console.info(f"Final Approval: {decision.upper()}")

    # Track performance: CEO task and Developer approval
    approved = decision == "approved"
    workflow.performance_tracker.record_task(
        "CEO", success=True,
        response_time_ms=result.execution_time * 1000,
        tokens=0
    )
    workflow.performance_tracker.record_approval("Developer", approved=approved)
    if not approved:
        workflow.performance_tracker.record_rework("Developer")
    # Track CEO experience and pool performance
    if hasattr(workflow.ceo, '_experience'):
        workflow.ceo._experience.add_experience("approve_solution", approved)
    workflow.agent_pool.record_task_outcome("CEO", approved)
    # Reflect on CEO approval decision (learning)
    if workflow.enable_learning:
        workflow.ceo.reflect(
            "approve_solution",
            str(result.output),
            approved
        )

    # Update trust: CEO reviewed Developer's work
    workflow.trust_tracker.update_trust("CEO", "Developer", positive=approved)

    # Graduated confidence response (OrgChart integration)
    ceo_confidence = result.confidence
    if ceo_confidence < CONFIDENCE_CRITICAL:
        console.warning(f"CEO confidence CRITICAL ({ceo_confidence:.2f}) — decision may be unreliable")
    elif ceo_confidence < CONFIDENCE_LOW and not approved:
        console.info(f"CEO confidence low ({ceo_confidence:.2f}) — reviewing rejection reasoning")
    elif ceo_confidence >= CONFIDENCE_HIGH and approved:
        console.info(f"CEO high-confidence approval ({ceo_confidence:.2f})")

    # Update plan step
    if workflow.plan_manager.current_plan:
        if approved:
            workflow.plan_manager.complete_step("step_4", output=f"CEO: {decision}")
        else:
            workflow.plan_manager.fail_step("step_4", error=f"CEO: {decision} - {reasoning}")

    return approved


