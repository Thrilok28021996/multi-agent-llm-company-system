"""Phase: analysis

Extracted from orchestrator/workflow.py.
Standalone functions — first param is the workflow instance.
"""

from orchestrator.phase_types import WorkflowPhase
from orchestrator.task_manager import TaskPriority
from ui.console import console


async def _run_analysis_phase(workflow) -> None:
    """Run the analysis phase with Product Manager and Researcher."""
    workflow.state.phase = WorkflowPhase.ANALYSIS
    workflow.structured_logger.info("Analysis phase started")

    problem = workflow.state.current_problem
    if not problem:
        console.error("No problem to analyze")
        return

    # Researcher validates problem (tracked by TaskManager)
    console.agent_action("Researcher", "Validating Problem", problem.description)
    tm_task = workflow.task_manager.create_task(
        task_type="validate_problem",
        description=f"Validate problem: {problem.description}",
        assigned_to="Researcher",
        priority=TaskPriority.HIGH,
    )
    workflow.task_manager.start_task(tm_task.id)
    validation_task = {
        "type": "validate_problem",
        "problem": problem.to_dict(),
        "evidence": "\n".join(problem.evidence),
        "context_messages": workflow._drain_messages_for_agent("Researcher")
    }
    validation_result = await workflow.researcher.execute_task(validation_task)
    workflow.task_manager.complete_task(tm_task.id, {"status": "done"})
    workflow.state.artifacts["problem_validation"] = validation_result.output
    # Handle case where output might be a string instead of dict
    validation_status = validation_result.output.get('status', 'analyzing') if isinstance(validation_result.output, dict) else 'analyzing'
    console.agent_thinking("Researcher", f"Validation: {validation_status}...")
    # Track Researcher experience and pool performance
    if hasattr(workflow.researcher, '_experience'):
        workflow.researcher._experience.add_experience("validate_problem", validation_result.success)
    workflow.agent_pool.record_task_outcome("Researcher", validation_result.success)
    workflow.performance_tracker.record_task(
        "Researcher", success=validation_result.success,
        response_time_ms=validation_result.execution_time * 1000, tokens=0
    )
    # Researcher shares findings with DataAnalyst via message bus
    workflow.message_bus.share_finding(
        "Researcher", "DataAnalyst",
        f"Problem validation result: {validation_status}. Evidence summary: {str(validation_result.output)[:300]}"
    )

    # Product Manager analyzes problem (tracked by TaskManager)
    console.agent_action("ProductManager", "Analyzing Market Opportunity", "Evaluating problem severity and market fit...")
    pm_task = workflow.task_manager.create_task(
        task_type="analyze_problem",
        description=f"Analyze market opportunity: {problem.description}",
        assigned_to="ProductManager",
        priority=TaskPriority.HIGH,
    )
    workflow.task_manager.start_task(pm_task.id)
    analysis_task = {
        "type": "analyze_problem",
        "problem": problem.to_dict(),
        "research_data": str(validation_result.output)
    }
    analysis_result = await workflow.product_manager.execute_task(analysis_task)
    workflow.state.artifacts["problem_analysis"] = analysis_result.output
    workflow.task_manager.complete_task(pm_task.id, {"status": "done"})
    # Track ProductManager experience and pool performance
    if hasattr(workflow.product_manager, '_experience'):
        workflow.product_manager._experience.add_experience("analyze_problem", analysis_result.success)
    workflow.agent_pool.record_task_outcome("ProductManager", analysis_result.success)
    workflow.performance_tracker.record_task(
        "ProductManager", success=analysis_result.success,
        response_time_ms=analysis_result.execution_time * 1000, tokens=0
    )
    # Reflect on PM problem analysis (learning)
    if workflow.enable_learning:
        workflow.product_manager.reflect(
            "analyze_problem",
            str(analysis_result.output),
            analysis_result.success
        )

    # Reflect on validation (learning)
    if workflow.enable_learning:
        workflow.researcher.reflect(
            "validate_problem",
            str(validation_result.output),
            validation_result.success
        )

    console.info(f"Problem: {problem.description}")
    final_status = validation_result.output.get('status', 'done') if isinstance(validation_result.output, dict) else 'done'
    console.success(f"Analysis complete - Validation: {final_status}")


