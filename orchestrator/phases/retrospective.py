"""Phase: retrospective

Extracted from orchestrator/workflow.py.
Standalone functions — first param is the workflow instance.
"""

from datetime import datetime
from pathlib import Path

from collaboration.meeting import MeetingType, quick_meeting
from orchestrator.phase_types import WorkflowPhase
from ui.console import console


async def _run_retrospective_phase(workflow) -> None:
    """All agents reflect on the workflow run - what worked, what didn't, what to improve."""
    workflow.state.phase = WorkflowPhase.RETROSPECTIVE
    workflow.structured_logger.info("Retrospective phase started")

    if not workflow.enable_meetings:
        console.info("Meetings disabled — skipping retrospective")
        workflow.state.artifacts["retrospective"] = {"decision": "skipped", "meetings_disabled": True}
        return

    # Gather workflow summary for context
    status = workflow.state.phase.value
    duration = None
    if workflow.state.started_at:
        duration = (datetime.now() - workflow.state.started_at).total_seconds()

    duration_str = f"{duration:.0f}s" if duration else "N/A"
    retro_context = f"""
Workflow Status: {status}
Duration: {duration_str}
Problem: {workflow.state.current_problem.description if workflow.state.current_problem else 'N/A'}
Decisions Made: {len(workflow.state.decisions)}
Phases Completed: {', '.join(k for k, v in workflow.state.artifacts.items() if v)}
Error: {workflow.state.error or 'None'}
"""

    retro_participants = [
        {"name": "CEO", "role": "Strategic Leadership", "model": workflow.model_config.get_model_name("ceo")},
        {"name": "CTO", "role": "Technical Leadership", "model": workflow.model_config.get_model_name("cto")},
        {"name": "ProductManager", "role": "Product Strategy", "model": workflow.model_config.get_model_name("product_manager")},
        {"name": "Developer", "role": "Implementation", "model": workflow.model_config.get_model_name("developer")},
    ]

    retro_outcome = await quick_meeting(
        topic="Workflow Retrospective: What worked, what didn't, what to improve?",
        participants=retro_participants,
        meeting_type=MeetingType.RETROSPECTIVE,
        context=retro_context
    )

    workflow.state.artifacts["retrospective"] = {
        "decision": retro_outcome.decision,
        "key_insights": retro_outcome.key_insights,
        "action_items": retro_outcome.action_items
    }

    # Store learnings in shared memory
    workflow.shared_memory.store_meeting(
        topic="Workflow Retrospective",
        participants=[p["name"] for p in retro_participants],
        summary=retro_outcome.decision,
        action_items=retro_outcome.action_items
    )

    # Activate learning system analysis for each agent
    from memory.shared_memory import MemoryType
    for agent_name, agent in workflow.agents.items():
        if hasattr(agent, 'learning'):
            perf_analysis = agent.learning.analyze_performance()
            if perf_analysis.get("status") != "no_data":
                suggestions = agent.learning.get_improvement_suggestions()
                if suggestions:
                    summary = agent.learning.export_learning_summary()
                    console.info(
                        f"Learning: {agent_name} — "
                        f"{perf_analysis.get('total_lessons', 0)} lessons, "
                        f"{perf_analysis.get('success_rate', 0):.0%} success"
                    )
                    workflow.shared_memory.store(
                        content=f"Learning for {agent_name}: {summary}",
                        memory_type=MemoryType.INSIGHT,
                        created_by=agent_name,
                        tags=["learning", "retrospective", agent_name],
                        importance=0.6,
                    )

    console.success(f"Retrospective complete: {len(retro_outcome.key_insights)} insights, {len(retro_outcome.action_items)} action items")

    # HR review: fire underperformers, log hiring needs
    underperformers = workflow.agent_pool.get_underperformers(threshold=0.3, min_tasks=3)
    for under in underperformers:
        workflow.agent_pool.fire_agent(
            under.name,
            reason=(
                f"Performance score {under.performance_score:.0%} "
                f"({under.tasks_failed} failures / {under.tasks_completed + under.tasks_failed} tasks) "
                "below 30% threshold after retrospective review"
            )
        )
        # Log hiring need to backfill the role
        workflow.agent_pool.hire_agent(
            under.role,
            reason=f"Backfill after {under.name} termination — maintaining team capacity"
        )
    if underperformers:
        console.info(
            f"HR action: {len(underperformers)} agent(s) terminated and backfilled"
        )

    # Persist action items as next-run constraints in strategy memory
    action_items = retro_outcome.action_items if retro_outcome.action_items else []
    if action_items:
        constraints_text = "RETROSPECTIVE CONSTRAINTS (from last run): " + "; ".join(str(a) for a in action_items[:5])
        problem_key = (
            workflow.state.current_problem.description[:80]
            if workflow.state.current_problem else "general"
        )
        workflow.escalation_manager.strategy_memory.record_attempt(
            problem=problem_key,
            approach="retrospective_constraints",
            outcome="action_items",
            details=constraints_text,
            success=True
        )
        console.info(f"Stored {len(action_items)} retrospective action items as next-run constraints")

    # Write structured session summary JSON to disk for cross-run learning
    import json as _json
    summary_path = Path(workflow.data_dir) / "session_summary.json"
    session_summary = {
        "timestamp": datetime.now().isoformat(),
        "problem": workflow.state.current_problem.description if workflow.state.current_problem else "N/A",
        "duration_s": duration,
        "phases_completed": list(workflow.state.artifacts.keys()),
        "decisions": workflow.state.decisions[-20:],  # last 20 decisions
        "action_items": action_items,
        "key_insights": retro_outcome.key_insights if retro_outcome.key_insights else [],
        "error": workflow.state.error,
    }
    try:
        summary_path.write_text(_json.dumps(session_summary, indent=2, default=str))
        console.info(f"Session summary saved → {summary_path}")
    except Exception as _e:
        console.warning(f"Could not save session summary: {_e}")
