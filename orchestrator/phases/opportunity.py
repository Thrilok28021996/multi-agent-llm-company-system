"""Phase: opportunity

Extracted from orchestrator/workflow.py.
Standalone functions — first param is the workflow instance.
"""

from collaboration.meeting import MeetingType, quick_meeting
from orchestrator.phase_types import WorkflowPhase
from ui.console import console


async def _run_opportunity_evaluation(workflow) -> bool:
    """CEO evaluates the opportunity with a decision meeting."""
    workflow.state.phase = WorkflowPhase.OPPORTUNITY_EVALUATION

    problem = workflow.state.current_problem
    if not problem:
        console.error("No problem to evaluate")
        return False
    analysis = workflow.state.artifacts.get("problem_analysis", {})

    # Hold a decision meeting with key stakeholders
    meeting_participants = [
        {
            "name": "CEO",
            "role": "Strategic Decision Maker",
            "model": workflow.model_config.get_model_name("ceo")
        },
        {
            "name": "CTO",
            "role": "Technical Advisor",
            "model": workflow.model_config.get_model_name("cto")
        },
        {
            "name": "ProductManager",
            "role": "Product Strategy",
            "model": workflow.model_config.get_model_name("product_manager")
        }
    ]

    # Gather research artifacts to give decision-makers full picture
    bias_flags = workflow.state.artifacts.get("bias_flags", [])
    bias_summary = "; ".join(
        f"{b.get('type','?')} ({b.get('severity','?')})" for b in bias_flags
    ) if bias_flags else "None detected"
    credibility_raw = workflow.state.artifacts.get("credibility_analysis", {})
    credibility_summary = (
        str(credibility_raw)[:300] if credibility_raw else "Not available"
    )
    counter_evidence = workflow.state.artifacts.get("counter_evidence", [])
    counter_summary = "; ".join(
        ce.get("description", "")[:80] for ce in counter_evidence if ce.get("counter_evidence")
    ) or "None found"
    opposing = workflow.state.artifacts.get("opposing_viewpoints", [])
    opposing_summary = "; ".join(
        f"{o.get('title','')}: {o.get('snippet','')[:60]}" for o in opposing[:3]
    ) or "None found"
    top_score = workflow.state.artifacts.get("discovered_problems", [{}])[0].get("score", "N/A") if workflow.state.artifacts.get("discovered_problems") else "N/A"
    freshness = workflow.state.artifacts.get("discovered_problems", [{}])[0].get("freshness_score", "N/A") if workflow.state.artifacts.get("discovered_problems") else "N/A"

    user_provided = workflow.state.artifacts.get("user_provided_problem", False)
    if user_provided:
        meeting_context = f"""
Problem: {problem.description}
Source: USER-PROVIDED REQUIREMENT (direct instruction, not research discovery)
Target Users: {problem.target_users}
Potential Solutions: {', '.join(problem.potential_solution_ideas) if problem.potential_solution_ideas else 'None specified'}

NOTE: This is a direct user requirement. Treat as APPROVED scope — evaluate technical feasibility only, not market validity.
"""
    else:
        meeting_context = f"""
Problem: {problem.description}
Severity: {problem.severity}
Target Users: {problem.target_users}
Composite Score: {top_score}
Data Freshness: {freshness}
Market Analysis: {str(analysis)}
Potential Solutions: {', '.join(problem.potential_solution_ideas) if problem.potential_solution_ideas else 'None specified'}

--- RESEARCH QUALITY FLAGS ---
Bias Detected: {bias_summary}
Credibility Analysis: {credibility_summary}
Counter-Evidence (reasons this may NOT be a real problem): {counter_summary}
Opposing Viewpoints: {opposing_summary}
"""
    if workflow.enable_meetings:
        meeting_outcome = await quick_meeting(
            topic=f"Should we pursue: {problem.description}?",
            participants=meeting_participants,
            meeting_type=MeetingType.DECISION,
            context=meeting_context
        )

        workflow.state.artifacts["opportunity_meeting"] = {
            "decision": meeting_outcome.decision,
            "consensus_reached": meeting_outcome.consensus_reached,
            "votes": meeting_outcome.votes,
            "key_insights": meeting_outcome.key_insights
        }
        meeting_decision = meeting_outcome.decision
    else:
        console.info("Meetings disabled — skipping team meeting, CEO decides alone")
        meeting_decision = "Proceed (meetings disabled)"
        workflow.state.artifacts["opportunity_meeting"] = {"decision": meeting_decision, "meetings_disabled": True}

    # CEO still makes final call based on meeting
    eval_task = {
        "type": "evaluate_opportunity",
        "problem": problem.to_dict(),
        "market_analysis": str(analysis),
        "meeting_outcome": meeting_decision,
        "user_provided_requirement": user_provided,
        **({"bias_flags": bias_summary, "credibility": credibility_summary,
            "counter_evidence": counter_summary, "opposing_viewpoints": opposing_summary,
            "freshness_score": freshness} if not user_provided else {}),
    }
    result = await workflow.ceo.execute_task(eval_task)

    # Handle case where output might be a string instead of dict
    output = result.output if isinstance(result.output, dict) else {"response": str(result.output)}
    decision = output.get("decision", "rejected")
    reasoning = output.get("reasoning", "")
    meeting_consensus = meeting_outcome.consensus_reached if workflow.enable_meetings else None
    workflow.state.decisions.append({
        "phase": "opportunity_evaluation",
        "decision": decision,
        "reasoning": reasoning,
        "meeting_consensus": meeting_consensus
    })

    workflow.state.artifacts["ceo_evaluation"] = result.output
    if workflow.enable_meetings:
        console.info(f"Meeting Consensus: {meeting_consensus}")
    console.agent_decision("CEO", decision.upper(), reasoning if reasoning else "")

    approved = decision == "approved"

    # Devil's Advocate gate: challenge approved decisions
    if workflow.enable_meetings and approved:
        console.info("Running Devil's Advocate review...")
        da_participants = [
            {"name": "ProductManager", "role": "Advocate", "model": workflow.model_config.get_model_name("product_manager")},
            {"name": "SecurityEngineer", "role": "Critic", "model": workflow.model_config.get_model_name("security_engineer")},
            {"name": "CEO", "role": "Judge", "model": workflow.model_config.get_model_name("ceo")},
        ]
        da_outcome = await quick_meeting(
            topic=f"Devil's Advocate: {problem.description}",
            participants=da_participants,
            meeting_type=MeetingType.DEVILS_ADVOCATE,
            context=f"Initial evaluation: APPROVED\nReasoning: {reasoning}",
        )
        workflow.state.artifacts["devils_advocate"] = {
            "decision": da_outcome.decision,
            "votes": da_outcome.votes,
            "key_insights": da_outcome.key_insights,
        }
        if da_outcome.decision and "reject" in da_outcome.decision.lower():
            console.warning("Devil's advocate overturned initial approval")
            approved = False

    return approved
