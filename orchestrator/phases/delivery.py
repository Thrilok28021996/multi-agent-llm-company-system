"""Phase: delivery

Extracted from orchestrator/workflow.py.
Standalone functions — first param is the workflow instance.
"""

from pathlib import Path
from typing import Any, Dict

from tools.git_tools import GitTools
from ui.console import console


async def _run_delivery_phase(workflow) -> Dict[str, Any]:
    """Package the approved solution for delivery."""
    workflow.structured_logger.info("Delivery phase started")

    # Track plan step (step_5 = Delivery)
    if workflow.plan_manager.current_plan:
        workflow.plan_manager.start_step("step_5")

    project_dir = workflow.state.artifacts.get("project_name", workflow.output_dir)
    project_path = Path(project_dir)
    problem = workflow.state.current_problem

    # Generate README via Developer
    readme_task = {
        "type": "implement_feature",
        "specification": (
            f"Create a README.md file for this project.\n"
            f"Project: {problem.description}\n"
            f"Include: Overview, Installation, Usage, Architecture.\n"
            f"Keep it concise and practical."
        ),
        "architecture": str(workflow.state.artifacts.get("architecture", "")),
        "file_structure": {},
        "language": workflow.state.artifacts.get("language", "python"),
        "output_dir": project_dir
    }
    await workflow.developer.execute_task(readme_task)

    # Generate Dockerfile for easy deployment
    language = workflow.state.artifacts.get("language", "python")
    dockerfile_task = {
        "type": "implement_feature",
        "specification": (
            f"Create a Dockerfile for this {language} project.\n"
            f"Project: {problem.description}\n"
            f"Requirements:\n"
            f"- Use a slim/alpine base image for {language}\n"
            f"- Install dependencies (requirements.txt or package.json if present)\n"
            f"- Set the correct WORKDIR and ENTRYPOINT\n"
            f"- Add a .dockerignore file excluding .git, __pycache__, node_modules, venv, .env\n"
            f"- Keep it simple and production-ready\n"
            f"Output ONLY the Dockerfile and .dockerignore files."
        ),
        "architecture": "",
        "file_structure": {},
        "language": language,
        "output_dir": project_dir
    }
    try:
        await workflow.developer.execute_task(dockerfile_task)
        console.success("Dockerfile generated for easy deployment")
    except Exception as e:
        console.warning(f"Dockerfile generation skipped: {e}")

    # Format generated code before delivery
    from tools.code_formatter import CodeFormatter
    try:
        formatter = CodeFormatter(timeout=30)
        format_result = formatter.format_directory(
            project_dir,
            recursive=True,
            exclude_patterns=["node_modules", "__pycache__", ".git", "venv", ".venv", ".deps", "dist", "build"]
        )
        if format_result.files_formatted:
            console.success(f"Formatted {len(format_result.files_formatted)} files")
        if format_result.errors:
            for err in format_result.errors[:3]:
                console.warning(f"Format warning: {err}")
    except Exception as e:
        console.warning(f"Code formatting skipped: {e}")

    # Track failures so the delivery summary never reports false success
    warnings = []

    # Initialize git repository for the delivered project
    try:
        project_git = GitTools(repo_path=project_dir)
        if not project_git.is_repo():
            import subprocess as _sp
            init_result = _sp.run(
                ["git", "init"], cwd=project_dir,
                capture_output=True, text=True, timeout=10, check=False
            )
            if init_result.returncode != 0:
                reason = (
                    f"git init failed (exit {init_result.returncode}): "
                    f"{(init_result.stderr or '').strip()}"
                )
                warnings.append(reason)
                console.warning(reason)
            else:
                stage_result = project_git.stage_all()
                commit_result = project_git.commit(
                    message=f"feat: initial solution for {problem.description}",
                    add_co_author=True,
                )
                if stage_result.success and commit_result.success:
                    console.success("Git repository initialized with initial commit")
                else:
                    reason = f"Git commit failed: {commit_result.error or stage_result.error}"
                    warnings.append(reason)
                    console.warning(reason)
        else:
            console.info("Git repository already exists, skipping init")
    except Exception as git_err:
        reason = f"Git init skipped: {git_err}"
        warnings.append(reason)
        console.warning(reason)

    # Install dependencies for the delivered project
    import subprocess as _sub
    try:
        req_file = project_path / "requirements.txt"
        pkg_file = project_path / "package.json"
        if req_file.exists():
            console.info("Installing Python dependencies...")
            install_result = _sub.run(
                ["python3", "-m", "pip", "install", "-q", "-r", str(req_file)],
                capture_output=True, text=True, timeout=120, cwd=project_dir,
                check=False
            )
            if install_result.returncode == 0:
                console.success("Python dependencies installed")
            else:
                reason = (
                    f"Dependency installation failed (exit {install_result.returncode}): "
                    f"{(install_result.stderr or install_result.stdout or '').strip()[-400:]}"
                )
                warnings.append(reason)
                console.warning(reason)
        elif pkg_file.exists():
            console.info("Installing Node.js dependencies...")
            npm_cmd = "npm" if _sub.run(["which", "npm"], capture_output=True).returncode == 0 else None
            if npm_cmd:
                install_result = _sub.run(
                    ["npm", "install", "--silent"],
                    capture_output=True, text=True, timeout=120, cwd=project_dir,
                    check=False
                )
                if install_result.returncode == 0:
                    console.success("Node.js dependencies installed")
                else:
                    reason = (
                        f"Node.js dependency installation failed (exit {install_result.returncode}): "
                        f"{(install_result.stderr or install_result.stdout or '').strip()[-400:]}"
                    )
                    warnings.append(reason)
                    console.warning(reason)
            else:
                console.warning("npm not found, skipping dependency install")
    except Exception as dep_err:
        reason = f"Dependency installation skipped: {dep_err}"
        warnings.append(reason)
        console.warning(reason)

    # Build delivery summary
    delivery = {
        "project_dir": project_dir,
        "problem": problem.description,
        "language": workflow.state.artifacts.get("language", "python"),
        "files": [str(f.relative_to(project_path))
                  for f in workflow._filter_project_files(list(project_path.rglob("*")))
                  if f.is_file()] if project_path.exists() else [],
        "execution_status": workflow.state.artifacts.get("execution_results", {}).get("summary", "Not run"),
        "warnings": warnings,
    }

    workflow.state.artifacts["delivery"] = delivery

    # Complete plan step
    if workflow.plan_manager.current_plan:
        workflow.plan_manager.complete_step("step_5", output=f"Delivered {len(delivery.get('files', []))} files")

    return delivery
