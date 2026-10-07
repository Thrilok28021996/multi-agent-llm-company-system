"""Phase: implementation

Extracted from orchestrator/workflow.py.
Standalone functions — first param is the workflow instance.
"""

from pathlib import Path

from orchestrator.phase_types import WorkflowPhase
from ui.console import console


async def _run_implementation_phase(workflow) -> None:
    """Developer implements the solution."""
    workflow.state.phase = WorkflowPhase.IMPLEMENTATION
    workflow.structured_logger.info("Implementation phase started")

    # Track plan step
    if workflow.plan_manager.current_plan:
        workflow.plan_manager.start_step("step_1")

    problem = workflow.state.current_problem
    if not problem:
        console.error("No problem for implementation")
        return

    architecture = workflow.state.artifacts.get("architecture", {})
    requirements = workflow.state.artifacts.get("requirements", {})

    # Get language from problem metadata (default to python)
    language = problem.metadata.get("language", "python")
    workflow.state.artifacts["language"] = language
    console.info(f"Target language: {language}")

    # Determine project location
    # "Direct output" mode: output_dir is "." or matches workspace_root
    # In this mode, files go directly into the target directory (like Claude Code)

    is_direct_output = (
        workflow.output_dir == "."
        or Path(workflow.output_dir).resolve() == Path(workflow.workspace_root).resolve()
    )

    if is_direct_output:
        # Direct output mode - like Claude Code, files go into workspace_root
        project_name = str(Path(workflow.workspace_root).resolve())
        console.info(f"Generating files in: {project_name} (Claude Code style)")
    else:
        # Isolated project mode - create a subfolder per solution
        project_name = f"{workflow.output_dir}/solution_{problem.id}"
        console.info(f"Output folder: {project_name}/")

    # Store project_name so QA feedback loop can reference it
    workflow.state.artifacts["project_name"] = project_name

    # Create project
    console.agent_action("Developer", "Creating Project", f"Setting up {language} project structure...")
    create_task = {
        "type": "create_project",
        "name": project_name,
        "project_type": language,  # Use the target language
        "description": problem.description,
        "features": problem.potential_solution_ideas,
        "output_dir": project_name
    }
    create_result = await workflow.developer.execute_task(create_task)
    workflow.state.artifacts["project"] = create_result.output

    # Handle case where output might be a string instead of dict
    create_output = create_result.output if isinstance(create_result.output, dict) else {"response": str(create_result.output)}
    files_created = create_output.get('files_created', [])
    console.success(f"Project created with {len(files_created)} files")

    # Implement feature
    console.agent_action("Developer", "Implementing Features", "Writing code based on specifications...")
    # Handle case where architecture might be a string instead of dict
    arch_data = architecture if isinstance(architecture, dict) else {"response": str(architecture)}
    time_context = workflow._get_time_context()
    spec_text = str(requirements)
    if time_context:
        spec_text = f"{time_context}\n{spec_text}"
    impl_task = {
        "type": "implement_feature",
        "specification": spec_text,
        "architecture": str(arch_data.get("architecture", "")),
        "file_structure": {},
        "language": language,
        "output_dir": project_name,
        "stories": workflow.state.artifacts.get("stories", []),
        "requirements_doc": workflow.state.artifacts.get("requirements_doc", {}),
        "context_messages": workflow._drain_messages_for_agent("Developer")
    }
    impl_result = await workflow.developer.execute_task(impl_task)
    # Wire first_principles_score from agent onto TaskResult
    impl_result.first_principles_score = getattr(workflow.developer, '_last_fp_score', 1.0)
    workflow.state.artifacts["implementation"] = impl_result.output
    workflow.state.artifacts["impl_fp_score"] = impl_result.first_principles_score

    # RAG: index this implementation for future retrieval
    if workflow.state.artifacts.get("implementation"):
        impl_summary = str(workflow.state.artifacts["implementation"])[:2000]
        workflow.rag_store.add(
            content=f"Problem: {workflow.state.current_problem.description}\n\nSolution:\n{impl_summary}",
            tags=[workflow.state.current_problem.domain],
            source="solution"
        )
    if impl_result.first_principles_score < 0.6:
        console.warning(
            f"Developer first-principles compliance LOW ({impl_result.first_principles_score:.0%}). "
            "Solution may be a surface-level fix rather than addressing root cause."
        )

    # Parse developer confidence score from response text
    import re as _re_conf
    _impl_text = str(impl_result.output)
    _conf_match = _re_conf.search(r'CONFIDENCE:\s*([0-9]\.[0-9]+)', _impl_text, _re_conf.IGNORECASE)
    dev_confidence = float(_conf_match.group(1)) if _conf_match else None
    if dev_confidence is not None:
        workflow.state.artifacts["developer_confidence"] = dev_confidence
        if dev_confidence < 0.6:
            console.warning(
                f"Developer self-reported CONFIDENCE={dev_confidence:.0%}. "
                "QA should focus extra attention on flagged uncertain areas."
            )
        else:
            console.info(f"Developer confidence: {dev_confidence:.0%}")

    # Handle case where output might be a string instead of dict
    impl_output = impl_result.output if isinstance(impl_result.output, dict) else {"response": str(impl_result.output)}
    impl_files = impl_output.get('files_written', [])
    console.success(f"Implementation complete - {len(impl_files)} files written")

    # Reflect on implementation (learning)
    if workflow.enable_learning:
        workflow.developer.reflect(
            "implement_feature",
            str(impl_result.output),
            impl_result.success
        )

    # Complete plan step
    if workflow.plan_manager.current_plan:
        workflow.plan_manager.complete_step("step_1", output="Implementation complete")


