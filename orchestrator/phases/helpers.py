"""Phase: helpers

Extracted from orchestrator/workflow.py.
Standalone functions — first param is the workflow instance.
"""

from datetime import datetime

from ui.console import console


def _check_token_budget(workflow, agent_name: str) -> int:
    """Return adjusted max_tokens based on remaining agent budget."""
    session = workflow.cost_tracker.get_current_session()
    if not session or not getattr(workflow.cost_tracker, '_budget_tokens', None):
        return 4096  # No budget set
    budget_pct = workflow._TOKEN_BUDGETS.get(agent_name, 0.10)
    agent_budget = int(workflow.cost_tracker._budget_tokens * budget_pct)
    agent_summary = session.agents.get(agent_name)
    used = agent_summary.total_tokens if agent_summary else 0
    remaining = agent_budget - used
    if remaining < 512:
        console.warning(f"{agent_name} near token budget ({used}/{agent_budget})")
        return 512  # Minimum viable response
    return min(4096, remaining)


def _get_time_context(workflow) -> str:
    """Get time budget context for agent prompts."""
    if not workflow.max_workflow_minutes or not workflow.state.started_at:
        return ""
    elapsed = (datetime.now() - workflow.state.started_at).total_seconds() / 60
    remaining = workflow.max_workflow_minutes - elapsed
    if remaining < 5:
        return "[TIME CRITICAL: <5 min remaining. Simplest viable solution only.]"
    elif remaining < 15:
        return f"[Time remaining: {remaining:.0f}min. Prioritize core functionality.]"
    return ""


def _detect_failure_owner(workflow, feedback: str) -> str:
    """Determine which agent 'owns' the failure based on feedback content."""
    feedback_lower = feedback.lower()
    if any(w in feedback_lower for w in ["requirement", "scope", "feature missing", "spec"]):
        return "ProductManager"
    if any(w in feedback_lower for w in ["architecture", "design", "structure", "pattern"]):
        return "CTO"
    if any(w in feedback_lower for w in ["security", "vulnerability", "injection", "secret"]):
        return "SecurityEngineer"
    return "Developer"


def _adjust_agent_behavior(workflow) -> None:
    """Adjust agent parameters based on KPI performance from previous runs.

    Called at workflow start. Uses PerformanceTracker data to:
    - Lower temperature for underperformers (make more conservative)
    - Increase temperature for high performers (allow more creativity)
    - Add supervision note for agents with low scores
    - Log adjustments for transparency
    """
    all_kpis = workflow.performance_tracker.get_all_kpis()
    if not all_kpis:
        return  # No historical data yet

    for agent_name, kpis in all_kpis.items():
        score = kpis.calculate_score()
        agent = workflow.agents.get(agent_name)
        if not agent:
            continue

        if score < 40:
            # Underperformer: lower temperature, use reconfiguration recommendation
            agent.config.temperature = max(0.1, agent.config.temperature - 0.15)
            from company.organization import AgentManager
            recommendation = AgentManager.recommend_reconfiguration(agent_name, score)
            agent._supervision_note = (
                f"Performance score: {score:.0f}/100. {recommendation}"
            )
            console.info(f"Reconfiguration for {agent_name}: {recommendation}")
            workflow.structured_logger.info(
                f"Adjusted {agent_name}: temp lowered to {agent.config.temperature:.2f} (score={score:.0f})"
            )
        elif score > 80:
            # High performer: slight temp boost for creativity
            agent.config.temperature = min(1.0, agent.config.temperature + 0.05)
            workflow.structured_logger.info(
                f"Adjusted {agent_name}: temp raised to {agent.config.temperature:.2f} (score={score:.0f})"
            )


def _drain_messages_for_agent(workflow, agent_name: str) -> str:
    """Drain pending messages and return summary for agent context."""
    pending = workflow.message_bus.peek_messages(agent_name)
    if pending == 0:
        return ""
    msgs = workflow.message_bus.get_messages(agent_name, limit=5)
    if not msgs:
        return ""
    lines = []
    for msg in msgs:
        sender = getattr(msg, 'sender', 'unknown')
        content = getattr(msg, 'content', str(msg))
        lines.append(f"From {sender}: {content}")
    return "Pending team messages:\n" + "\n".join(lines)


def _load_previous_learnings(workflow) -> None:
    """Load retrospective action items from shared memory into agent context."""
    from memory.shared_memory import MemoryType
    retro_memories = workflow.shared_memory.get_by_type(MemoryType.MEETING)
    # Filter for retrospectives
    retro_mems = [m for m in retro_memories if "retrospective" in m.content.lower()]
    if not retro_mems:
        return
    # Take most recent 3
    retro_mems.sort(key=lambda m: m.created_at, reverse=True)
    lessons = []
    for mem in retro_mems[:3]:
        items = mem.metadata.get("action_items", [])
        lessons.extend(items[:3])
    if not lessons:
        return
    lessons_text = "\n".join(f"- {l}" for l in lessons[:8])
    for agent in workflow.agents.values():
        agent._retrospective_lessons = lessons_text
    console.info(f"Loaded {len(lessons[:8])} lessons from past retrospectives")

    # Also load learning system insights from past runs
    insight_memories = workflow.shared_memory.get_by_type(MemoryType.INSIGHT)
    learning_insights = [m for m in insight_memories if "learning" in (m.tags or [])]
    if learning_insights:
        learning_insights.sort(key=lambda m: m.created_at, reverse=True)
        for mem in learning_insights[:3]:
            agent_name = mem.created_by
            if agent_name in workflow.agents:
                existing = getattr(workflow.agents[agent_name], '_retrospective_lessons', '') or ''
                workflow.agents[agent_name]._retrospective_lessons = (
                    existing + f"\n{mem.content}"
                )
        console.info(f"Loaded {min(3, len(learning_insights))} learning insights from past runs")


def _filter_project_files(workflow, files: list) -> list:
    """Filter out files inside ignored directories."""
    return [
        f for f in files
        if not any(
            part in workflow._IGNORED_DIRS
            or part.endswith(workflow._IGNORED_SUFFIXES)
            for part in f.parts
        )
    ]


def _interactive_gate(workflow, phase_name: str, details: str = "") -> bool:
    """Pause for user approval in interactive mode. Returns True to continue, False to abort."""
    if not getattr(workflow, '_interactive', False):
        return True

    from ui.interactive import AskUserQuestion, QuestionType

    asker = AskUserQuestion(color_enabled=True)
    prompt = f"Phase complete: {phase_name}"
    if details:
        prompt += f"\n{details}"
    prompt += "\nContinue to next phase?"

    result = asker.ask(
        question=prompt,
        question_type=QuestionType.YES_NO,
        header="Approval",
        default="y",
    )

    if result.skipped or result.timed_out:
        return False

    if not result.answer:
        console.warning("Workflow aborted by user")
        return False

    return True
