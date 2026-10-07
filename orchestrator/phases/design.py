"""Phase: design

Extracted from orchestrator/workflow.py.
Standalone functions — first param is the workflow instance.
"""

from collaboration.meeting import MeetingType, quick_meeting
from orchestrator.phase_types import WorkflowPhase
from ui.console import console


async def _run_technical_design_phase(workflow) -> None:
    """CTO designs the technical architecture."""
    workflow.state.phase = WorkflowPhase.TECHNICAL_DESIGN
    workflow.structured_logger.info("Technical design phase started")

    problem = workflow.state.current_problem
    if not problem:
        console.error("No problem for technical design")
        return
    _analysis = workflow.state.artifacts.get("problem_analysis", {})  # Available for future use

    # Refine the problem statement before feeding to PM and CTO
    try:
        from research.problem_statement_refiner import ProblemStatementRefiner
        _refiner = ProblemStatementRefiner()
        _refined = _refiner.refine(problem.description)
        refined_desc = _refiner.format_refined_statement(_refined)
        workflow.state.artifacts["refined_problem"] = refined_desc
        console.info(
            f"Problem refined: type={_refined.problem_type.value}, "
            f"actionable={_refined.is_actionable}, confidence={_refined.confidence:.0%}"
        )
        if not _refined.is_actionable:
            for note in _refined.refinement_notes:
                console.warning(f"Problem refinement note: {note}")
        # Augment the problem dict passed to PM with refined description
        problem_dict_refined = problem.to_dict()
        problem_dict_refined["refined_description"] = refined_desc
    except Exception as _re:
        console.warning(f"Problem statement refiner failed: {_re}")
        problem_dict_refined = problem.to_dict()

    # Define requirements
    console.agent_action("ProductManager", "Defining Requirements", "Creating detailed specifications...")
    problem_analysis = workflow.state.artifacts.get("problem_analysis", {})
    req_task = {
        "type": "define_requirements",
        "problem": problem_dict_refined,
        "target_users": problem.target_users,
        "market_analysis": str(problem_analysis)[:600] if problem_analysis else ""
    }
    req_result = await workflow.product_manager.execute_task(req_task)
    workflow.state.artifacts["requirements"] = req_result.output
    # Store structured RequirementsDoc alongside raw output
    req_doc_dict = req_result.artifacts.get("requirements_doc") or (
        req_result.output.get("requirements_doc") if isinstance(req_result.output, dict) else None
    )
    if req_doc_dict:
        workflow.state.artifacts["requirements_doc"] = req_doc_dict
        ac_count = len(req_doc_dict.get("acceptance_criteria", []))
        console.info(f"RequirementsDoc: {ac_count} acceptance criteria extracted")
    console.agent_thinking("ProductManager", "Requirements defined")
    # Track PM experience for requirements definition
    if hasattr(workflow.product_manager, '_experience'):
        workflow.product_manager._experience.add_experience("define_requirements", req_result.success)
    workflow.agent_pool.record_task_outcome("ProductManager", req_result.success)

    # Technical design
    console.agent_action("CTO", "Designing Architecture", "Creating technical blueprint...")
    # Handle case where output might be a string instead of dict
    req_output = req_result.output if isinstance(req_result.output, dict) else {"response": str(req_result.output)}
    design_task = {
        "type": "design_architecture",
        "problem": problem_dict_refined,
        "requirements": str(req_output.get("requirements", "")),
        "requirements_doc": workflow.state.artifacts.get("requirements_doc", {}),
        "market_analysis": str(problem_analysis)[:400] if problem_analysis else "",
        "context_messages": workflow._drain_messages_for_agent("CTO")
    }
    # Debate: for complex tasks, run CEO vs CTO debate before finalizing architecture
    design_prompt = (
        f"Architecture for: {problem.description}\n\n"
        f"Requirements: {str(req_output.get('requirements', ''))[:400]}"
    )
    compute = workflow.cto._get_compute_config(design_prompt)
    if compute["consistency_samples"] >= 5:  # complex task — run debate
        try:
            from collaboration.debate import DebateOrchestrator
            debate = DebateOrchestrator(max_rounds=2)
            outcome = await debate.run_debate(
                agent_a=workflow.ceo,
                agent_b=workflow.cto,
                topic=f"Architecture for: {workflow.state.current_problem.description}",
                synthesizer=workflow.cto,
            )
            workflow.state.artifacts["architecture_debate"] = outcome.final_position
            console.info("Architecture debate complete — skipping HyperTree to avoid redundant LLM calls")
            # Inject debate outcome into the CTO's final design task
            design_task["requirements"] = (
                design_task["requirements"]
                + f"\n\nARCHITECTURE DEBATE OUTCOME:\n{outcome.final_position[:1000]}"
            )
            # HyperTree is skipped when Debate runs — both fire on complex tasks
            # and together produce 10+ LLM calls. Debate already covers the same ground.
        except Exception as _debate_err:
            console.warning(f"Architecture debate skipped: {_debate_err}")

    design_result = await workflow.cto.execute_task(design_task)
    workflow.state.artifacts["architecture"] = design_result.output
    # Store structured ArchitectureNote + Stories
    arch_note_dict = design_result.artifacts.get("architecture_note") or (
        design_result.output.get("architecture_note") if isinstance(design_result.output, dict) else None
    )
    if arch_note_dict:
        workflow.state.artifacts["architecture_note"] = arch_note_dict
        stories = arch_note_dict.get("stories", [])
        workflow.state.artifacts["stories"] = stories
        adr_count = len(arch_note_dict.get("architecture_decisions", []))
        console.info(f"ArchitectureNote: {adr_count} ADRs, {len(stories)} stories sharded")
    console.agent_thinking("CTO", "Architecture design complete")
    # Track CTO experience and pool performance
    if hasattr(workflow.cto, '_experience'):
        workflow.cto._experience.add_experience("design_architecture", design_result.success)
    workflow.agent_pool.record_task_outcome("CTO", design_result.success)
    workflow.performance_tracker.record_task(
        "CTO", success=design_result.success,
        response_time_ms=design_result.execution_time * 1000, tokens=0
    )
    # CTO shares architecture context with Developer via message bus
    arch_summary = str(design_result.output)[:300] if design_result.output else "Architecture designed"
    workflow.message_bus.share_finding("CTO", "Developer", f"Architecture ready: {arch_summary}")

    # Extract root cause from CTO's structured output
    from utils.output_parser import StructuredOutputParser
    design_parser = StructuredOutputParser()
    design_text = design_result.artifacts.get("architecture_doc", "")
    if not design_text and isinstance(design_result.output, dict):
        design_text = design_result.output.get("architecture", "")
    parsed_design = design_parser.parse_cto_design(str(design_text))
    root_cause = parsed_design.get("root_cause", "")
    if root_cause:
        workflow.state.artifacts["root_cause"] = root_cause
        console.info(f"Root cause identified: {root_cause}")
    else:
        console.warning("CTO did not provide a structured ROOT_CAUSE")
    # Score root cause depth — ensures deep first-principles thinking
    rcd_score = design_parser.score_root_cause_depth(root_cause or design_text[:500])
    workflow.state.artifacts["root_cause_depth"] = rcd_score
    console.info(f"Root cause analysis depth: {rcd_score}/4 (1=symptom, 2=proximate, 3=root, 4=systemic)")
    if rcd_score < 2:
        console.warning("Shallow root cause analysis — CTO will be prompted for deeper analysis on next redesign")
        workflow.state.artifacts["needs_deeper_root_cause"] = True

    # Reflect on CTO architecture design (learning)
    if workflow.enable_learning:
        workflow.cto.reflect(
            "design_architecture",
            str(design_result.output),
            design_result.success
        )

    # Feasibility assessment
    console.agent_action("CTO", "Assessing Feasibility", "Evaluating technical constraints...")
    design_output = design_result.output if isinstance(design_result.output, dict) else {"response": str(design_result.output)}
    feasibility_task = {
        "type": "assess_feasibility",
        "solution": design_output.get("architecture", ""),
        "constraints": [],
        "timeline": "MVP"
    }
    feasibility_result = await workflow.cto.execute_task(feasibility_task)
    workflow.state.artifacts["feasibility"] = feasibility_result.output

    # Handle case where output might be a string instead of dict
    feas_output = feasibility_result.output if isinstance(feasibility_result.output, dict) else {"response": str(feasibility_result.output)}
    feasibility = feas_output.get('feasibility', 'unknown')
    console.success(f"Feasibility Assessment: {feasibility}")

    # Create execution plan for remaining phases via PlanManager
    exec_plan = workflow.plan_manager.create_plan(
        name=f"solution-{problem.id}",
        description=f"Execution plan for: {problem.description}",
        tags=["auto-generated"],
    )
    exec_plan.add_step("Implementation", "Implement the solution based on architecture")
    exec_plan.add_step("Code Execution", "Run and verify generated code")
    exec_plan.add_step("QA Validation", "Validate solution quality")
    exec_plan.add_step("CEO Approval", "Get final approval from CEO")
    exec_plan.add_step("Delivery", "Package and deliver the solution")
    workflow.plan_manager.approve_plan(exec_plan, approved_by="CTO")
    workflow.plan_manager.start_execution(exec_plan)
    workflow.state.artifacts["execution_plan"] = exec_plan.to_dict()
    console.info(f"Execution plan created: {exec_plan.name} ({len(exec_plan.steps)} steps)")


async def _run_design_review_phase(workflow) -> None:
    """Quick meeting: CTO + Developer + QA review architecture before coding."""
    workflow.state.phase = WorkflowPhase.DESIGN_REVIEW
    workflow.structured_logger.info("Design review phase started")

    architecture = workflow.state.artifacts.get("architecture", {})
    requirements = workflow.state.artifacts.get("requirements", {})
    problem = workflow.state.current_problem

    if not workflow.enable_meetings:
        console.info("Meetings disabled — skipping design review")
        workflow.state.artifacts["design_review"] = {"decision": "skipped", "meetings_disabled": True}
        return

    review_participants = [
        {
            "name": "CTO",
            "role": "Architecture Author",
            "model": workflow.model_config.get_model_name("cto")
        },
        {
            "name": "Developer",
            "role": "Implementation Feasibility",
            "model": workflow.model_config.get_model_name("developer")
        },
        {
            "name": "QAEngineer",
            "role": "Testability Review",
            "model": workflow.model_config.get_model_name("qa_engineer")
        }
    ]

    review_context = f"""
Problem: {problem.description if problem else 'N/A'}
Architecture: {str(architecture)}
Requirements: {str(requirements)}

Review this architecture BEFORE implementation begins. Focus on:
1. Is the design implementable with available resources?
2. Are there obvious design flaws or missing components?
3. Is the design testable? Can QA validate it effectively?
4. Are there simpler alternatives that achieve the same goal?
"""
    review_outcome = await quick_meeting(
        topic="Design Review: Architecture Feasibility Check",
        participants=review_participants,
        meeting_type=MeetingType.REVIEW,
        context=review_context
    )

    workflow.state.artifacts["design_review"] = {
        "decision": review_outcome.decision,
        "votes": review_outcome.votes,
        "action_items": review_outcome.action_items
    }

    if review_outcome.decision == "needs_revision":
        console.warning("Design review flagged issues. CTO revising architecture...")
        # CTO revises based on feedback
        revise_task = {
            "type": "design_architecture",
            "problem": problem.to_dict() if problem else {},
            "requirements": str(requirements),
            "constraint": f"Design review feedback: {'; '.join(review_outcome.action_items[:3])}. Address these concerns."
        }
        revised = await workflow.cto.execute_task(revise_task)
        workflow.state.artifacts["architecture"] = revised.output
        console.success("Architecture revised based on design review feedback")
    else:
        console.success("Design review passed")
