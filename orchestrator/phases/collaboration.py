"""Phase: collaboration

Extracted from orchestrator/workflow.py.
Standalone functions — first param is the workflow instance.
"""

from typing import Any, Dict, List, Optional


async def _team_brainstorm(workflow) -> Optional[Dict[str, Any]]:
    """All agents brainstorm solutions for a stuck problem.

    Uses a brainstorm meeting with CTO, PM, and Developer to
    generate fresh ideas, then picks the most promising one.
    """
    from collaboration.meeting import AgentMeeting
    from collaboration.meeting import MeetingType as MT
    problem = workflow.state.current_problem
    if not problem:
        return None

    strategy_context = workflow.escalation_manager.strategy_memory.summarize_for_next_attempt()

    meeting = AgentMeeting(MT.BRAINSTORM, f"Brainstorm solutions for: {problem.description}")
    meeting.add_participant("CTO", "cto", workflow.model_config.get_model_name("cto"))
    meeting.add_participant("ProductManager", "product_manager", workflow.model_config.get_model_name("product_manager"))
    meeting.add_participant("Developer", "developer", workflow.model_config.get_model_name("developer"))

    context = (
        f"Problem: {problem.description}\n"
        f"Previous attempts have failed.\n{strategy_context}\n"
        f"We need a DIFFERENT approach. Think creatively."
    )
    outcome = await meeting.run(context)

    # Record in strategy memory
    workflow.escalation_manager.strategy_memory.record_attempt(
        approach="team_brainstorm", round=len(workflow.escalation_manager.history),
        outcome="brainstorm_complete",
        feedback=str(outcome.key_insights[:3]) if outcome.key_insights else ""
    )

    # Use brainstorm ideas to inform next design
    if outcome.key_insights:
        brainstorm_ideas = "\n".join(outcome.key_insights[:5])
        redesign_task = {
            "type": "design_architecture",
            "problem": problem.to_dict(),
            "requirements": str(workflow.state.artifacts.get("requirements", "")),
            "constraint": f"Use these brainstormed ideas as inspiration:\n{brainstorm_ideas}"
        }
        redesign_result = await workflow.cto.execute_task(redesign_task)
        workflow.state.artifacts["architecture"] = redesign_result.output
        return {"ideas": outcome.key_insights, "new_architecture": redesign_result.output}

    return None


async def _decompose_problem(workflow) -> Optional[List[Dict]]:
    """Break an unsolvable problem into smaller sub-problems.

    Uses PM to identify independently-solvable pieces.
    """
    problem = workflow.state.current_problem
    if not problem:
        return None

    strategy_context = workflow.escalation_manager.strategy_memory.summarize_for_next_attempt()

    decompose_task = {
        "type": "define_requirements",
        "problem": problem.to_dict(),
        "target_users": problem.target_users,
        "constraint": (
            f"The full problem has proven too complex to solve in one attempt.\n"
            f"{strategy_context}\n\n"
            f"Break it into 2-3 INDEPENDENT sub-problems that can each be solved separately.\n"
            f"Each sub-problem should be self-contained and testable on its own.\n"
            f"Focus on the MINIMUM viable solution for each sub-problem."
        )
    }
    decompose_result = await workflow.product_manager.execute_task(decompose_task)

    # Record in strategy memory
    workflow.escalation_manager.strategy_memory.record_attempt(
        approach="decompose_problem",
        round=len(workflow.escalation_manager.history),
        outcome="decomposed",
        feedback=str(decompose_result.output) if decompose_result.output else ""
    )

    # Use decomposed requirements for next implementation
    workflow.state.artifacts["requirements"] = decompose_result.output
    await workflow._run_technical_design_phase()

    return [{"decomposed_requirements": decompose_result.output}]


async def _pivot_approach(workflow) -> None:
    """Try a completely different technical approach.

    CTO proposes alternative architectures/tech stacks.
    Strategy memory ensures we don't repeat failed approaches.
    """
    problem = workflow.state.current_problem
    if not problem:
        return

    strategy_context = workflow.escalation_manager.strategy_memory.summarize_for_next_attempt()
    tried_archs = workflow.escalation_manager.strategy_memory.tried_architectures

    pivot_task = {
        "type": "design_architecture",
        "problem": problem.to_dict(),
        "requirements": str(workflow.state.artifacts.get("requirements", "")),
        "constraint": (
            f"ALL previous approaches have failed. You MUST propose a COMPLETELY "
            f"different technical approach.\n\n"
            f"{strategy_context}\n\n"
            f"Previously tried architectures: {', '.join(tried_archs) if tried_archs else 'None recorded'}\n\n"
            f"Consider: different language, different framework, simpler architecture, "
            f"different data structures, or a fundamentally different algorithm.\n"
            f"Do NOT reuse any approach that has already been tried."
        )
    }
    pivot_result = await workflow.cto.execute_task(pivot_task)
    workflow.state.artifacts["architecture"] = pivot_result.output

    # Record in strategy memory
    arch_desc = str(pivot_result.output) if pivot_result.output else "unknown"
    workflow.escalation_manager.strategy_memory.record_architecture(arch_desc)
    workflow.escalation_manager.strategy_memory.record_attempt(
        approach="pivot_approach",
        round=len(workflow.escalation_manager.history),
        outcome="pivoted",
        feedback=arch_desc
    )


