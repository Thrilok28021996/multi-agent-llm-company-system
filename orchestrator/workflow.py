"""Company workflow orchestrator - Manages the end-to-end problem-solving workflow."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents import (CEOAgent, CTOAgent, DataAnalystAgent, DeveloperAgent,
                    DevOpsEngineerAgent, ProductManagerAgent, QAEngineerAgent,
                    ResearcherAgent, SecurityEngineerAgent)
from collaboration.meeting import MeetingType, quick_meeting
from config.models import ModelConfig
from memory.session import SessionManager, SessionState
from memory.shared_memory import SharedMemory
from orchestrator.escalation import EscalationAction, EscalationManager
from orchestrator.message_bus import MessageBus
# Phase enum + confidence thresholds live in orchestrator/phase_types.py so the
# extracted phase modules can import them without a circular import. They are
# re-exported here for backwards compatibility.
from orchestrator import phase_types as _phase_types
from orchestrator.plan_mode import PlanManager
from orchestrator.task_manager import TaskManager
from research.cross_validator import CrossValidator
from research.problem_discoverer import (DiscoveredProblem, ProblemDiscoverer,
                                         ProblemFrequency, ProblemSeverity)
from research.sources import ResearchSources
from tools import UnifiedTools
from tools.git_tools import GitTools
from ui.console import console
from utils.cost_tracker import CostTracker
from utils.error_recovery import ErrorRecoverySystem, RetryStrategy
from utils.health_check import HealthChecker, HealthStatus
from utils.memory_monitor import MemoryMonitor
from utils.progress_tracker import ProgressTracker
from utils.structured_logging import LogLevel, get_structured_logger
from orchestrator.phases.research import _run_research_phase, _discover_from_web_search
from orchestrator.phases.data_analysis import _run_data_analysis_phase
from orchestrator.phases.analysis import _run_analysis_phase
from orchestrator.phases.opportunity import _run_opportunity_evaluation
from orchestrator.phases.design import _run_technical_design_phase, _run_design_review_phase
from orchestrator.phases.implementation import _run_implementation_phase
from orchestrator.phases.quality import _run_qa_phase, _run_security_review_phase, _check_quality_gate, _evaluate_gate, _retry_gate
from orchestrator.phases.approval import _run_ceo_approval
from orchestrator.phases.delivery import _run_delivery_phase
from orchestrator.phases.retrospective import _run_retrospective_phase
from orchestrator.phases.collaboration import _team_brainstorm, _decompose_problem, _pivot_approach
from orchestrator.phases.helpers import _check_token_budget, _get_time_context, _detect_failure_owner, _adjust_agent_behavior, _drain_messages_for_agent, _load_previous_learnings, _filter_project_files, _interactive_gate

# NOTE: CONFIDENCE_* thresholds and WorkflowPhase are defined in
# orchestrator/phase_types.py (imported above) so the extracted phase modules
# can use them without a circular import.
WorkflowPhase = _phase_types.WorkflowPhase
CONFIDENCE_CRITICAL = _phase_types.CONFIDENCE_CRITICAL
CONFIDENCE_LOW = _phase_types.CONFIDENCE_LOW
CONFIDENCE_MEDIUM = _phase_types.CONFIDENCE_MEDIUM
CONFIDENCE_HIGH = _phase_types.CONFIDENCE_HIGH


@dataclass
class WorkflowState:
    """Current state of the workflow."""

    phase: WorkflowPhase = WorkflowPhase.IDLE
    current_problem: Optional[DiscoveredProblem] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


# ============================================================
#  INITIALIZATION
# ============================================================


class CompanyWorkflow:
    """
    Orchestrates the full company workflow from problem discovery to solution delivery.
    """

    def __init__(
        self,
        workspace_root: str = ".",
        model_config: ModelConfig = None,
        memory_persist_dir: str = None,
        output_dir: str = "output/solutions",
        data_dir: Optional[str] = None,
        run_code_execution: bool = True,
        llm_config: Optional[Dict[str, Any]] = None,
        enable_meetings: bool = True,
        enable_learning: bool = True,
        enable_escalation: bool = True,
        enable_security_review: bool = True,
        enable_retrospective: bool = True,
        max_workflow_minutes: int = 0,
        force_stop: bool = False,
        scaffold_mode: bool = False
    ):
        self.workspace_root = workspace_root
        self.model_config = model_config or ModelConfig()
        self.memory_persist_dir = memory_persist_dir
        self.output_dir = output_dir
        self.run_code_execution = run_code_execution
        self.llm_config = llm_config or {}
        self.enable_meetings = enable_meetings
        self.enable_learning = enable_learning
        self.enable_escalation = enable_escalation
        self.enable_security_review = enable_security_review
        self.enable_retrospective = enable_retrospective
        self.max_workflow_minutes = max_workflow_minutes
        self.force_stop = force_stop
        self.scaffold_mode = scaffold_mode
        # data_dir is for internal Company-AGI data (logs, reports)
        # Defaults to workspace_root for backwards compatibility
        self.data_dir = data_dir or workspace_root

        # Initialize agents
        self._init_agents()

        # Apply LLM config overrides to agents (user config file settings)
        if self.llm_config:
            for agent in self.agents.values():
                if "temperature" in self.llm_config:
                    agent.config.temperature = self.llm_config["temperature"]
                if "max_tokens" in self.llm_config:
                    agent.config.max_tokens = self.llm_config["max_tokens"]
                if self.llm_config.get("streaming"):
                    agent._streaming_enabled = True
                    agent._stream_callback = console.create_stream_callback(agent.name)

        # Initialize infrastructure - logs go to data_dir, not workspace_root
        self.message_bus = MessageBus(log_dir=f"{self.data_dir}/logs")
        self.shared_memory = SharedMemory(persist_dir=memory_persist_dir or f"{self.data_dir}/memory")
        self.problem_discoverer = ProblemDiscoverer(
            model=self.model_config.get_model_name("researcher")
        )
        self.research_sources = ResearchSources()

        # Initialize unified tools (Claude Code style)
        self.tools = UnifiedTools(
            workspace_root=workspace_root,
            persist_dir=memory_persist_dir
        )

        # Register agents with message bus
        self._register_agents()

        # Initialize progress tracker
        self.progress = ProgressTracker(
            metrics_file=Path(self.data_dir) / "metrics" / "phase_metrics.json"
        )
        self.progress.add_phases([
            {"id": "research", "name": "Research"},
            {"id": "data_analysis", "name": "Data Analysis"},
            {"id": "analysis", "name": "Analysis"},
            {"id": "opportunity_evaluation", "name": "Opportunity Eval"},
            {"id": "technical_design", "name": "Technical Design"},
            {"id": "design_review", "name": "Design Review"},
            {"id": "implementation", "name": "Implementation"},
            {"id": "code_execution", "name": "Code Execution"},
            {"id": "qa_validation", "name": "QA Validation"},
            {"id": "security_review", "name": "Security Review"},
            {"id": "ceo_approval", "name": "CEO Approval"},
            {"id": "delivery", "name": "Delivery"},
            {"id": "retrospective", "name": "Retrospective"},
        ])

        # Initialize error recovery
        self.error_recovery = ErrorRecoverySystem(
            max_retries=2,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=2.0,
            enable_logging=True
        )

        # Initialize health checker
        self.health_checker = HealthChecker()

        # Initialize cost tracker and set as global singleton so agents can use it
        import utils.cost_tracker as _cost_tracker_mod
        self.cost_tracker = CostTracker(
            history_file=Path(self.data_dir) / "costs" / "cost_history.json"
        )
        _cost_tracker_mod._cost_tracker = self.cost_tracker

        # Task manager for tracking agent work
        self.task_manager = TaskManager()

        # Plan manager for structured execution plans
        self.plan_manager = PlanManager(plans_dir=f"{self.data_dir}/plans")

        # Structured logger with JSON output for tracing
        self.structured_logger = get_structured_logger(
            name="company_agi",
            level=LogLevel.INFO,
            json_output=Path(self.data_dir) / "logs" / "structured.jsonl"
        )

        # Memory monitor with warning callbacks and auto-compaction
        def _compact_agent_memory(agent_name: str) -> None:
            console.warning(f"[Memory] Agent {agent_name} memory high — compacting")
            agent = self.agents.get(agent_name)
            if agent:
                agent.memory.compact()

        # Memory monitor disabled — its background thread + psutil polling
        # waste RAM better reserved for Ollama LLM inference.
        self.memory_monitor = MemoryMonitor(
            on_warning=lambda w: console.warning(f"[Memory] {w.message}"),
            on_compact_needed=_compact_agent_memory,
        )
        self.memory_monitor.disabled = True

        # Git tools for version control in delivery
        self.git_tools = GitTools(repo_path=workspace_root)

        # Session manager for session persistence / resume
        self.session_manager = SessionManager(
            sessions_dir=f"{self.data_dir}/sessions"
        )
        # Clean up old sessions on startup
        try:
            removed = self.session_manager.cleanup_old_sessions()
            if removed > 0:
                console.debug(f"Cleaned up {removed} old session(s)")
        except Exception as e:
            console.warning(f"Session cleanup failed (non-critical): {e}")

        # Hooks manager for lifecycle events
        from utils.hooks import HooksManager
        self.hooks_manager = HooksManager()

        # Escalation manager for graduated recovery
        self.escalation_manager = EscalationManager()
        strat_path = Path(self.data_dir) / "escalation" / "strategy_memory.json"
        self.escalation_manager.load_strategy_memory(strat_path)

        # Cross-validator for research deduplication
        self.cross_validator = CrossValidator()

        # Performance tracker — records task outcomes and computes KPIs
        # Load from disk so KPIs persist across runs
        from company.performance import PerformanceTracker
        perf_path = Path(self.data_dir) / "performance" / "kpis.json"
        self.performance_tracker = PerformanceTracker.load(perf_path)

        # Org chart — defines reporting lines and escalation paths
        from company.organization import OrgChart
        self.org_chart = OrgChart.default()

        # Trust tracker — inter-agent trust based on review outcomes
        # Load from disk so trust persists across runs
        from company.trust import TrustTracker
        trust_path = Path(self.data_dir) / "trust" / "trust_scores.json"
        self.trust_tracker = TrustTracker.load(trust_path)

        # Sprint manager for team alignment and progress tracking
        from company.sprint import SprintManager
        self.sprint_manager = SprintManager()
        self._current_sprint_id: str = ""

        # Agent personality and experience


        from agents.personality import (DEFAULT_PERSONALITIES, AgentExperience,
                                        AgentPersonality)
        self._experience_dir = Path(self.data_dir) / "experience"
        for agent_name, agent in self.agents.items():
            role_key = agent.role.value
            # Apply personality
            personality = DEFAULT_PERSONALITIES.get(role_key, AgentPersonality())
            agent._personality = personality
            # Adjust temperature based on personality
            agent.config.temperature = personality.adjust_temperature(agent.config.temperature)
            # Load experience from disk
            exp_path = self._experience_dir / f"{agent_name}.json"
            agent._experience = AgentExperience.load(exp_path)

        # Company culture — inject values into all agent prompts
        from company.culture import COMPANY_CULTURE
        culture_text = COMPANY_CULTURE.get_prompt_injection()
        for agent in self.agents.values():
            agent.system_prompt += culture_text

        # Agent pool for headcount tracking
        from company.hiring import AgentPool
        self.agent_pool = AgentPool()
        for agent_name, agent in self.agents.items():
            self.agent_pool.register(agent_name, agent.role.value)

        # Meeting managers
        from company.meetings import StandupManager, IncidentResponseManager
        self.standup_manager = StandupManager()
        self.incident_manager = IncidentResponseManager()

        # Project backlog for multi-problem management
        from company.backlog import ProjectBacklog
        self.backlog = ProjectBacklog()

        # RAG store for injecting relevant past solutions into agent prompts
        from memory.rag_store import RAGStore
        self.rag_store = RAGStore(persist_dir=str(Path(self.output_dir) / "rag"))
        for agent in self.agents.values():
            agent.set_rag_store(self.rag_store)

        # Workflow state
        self.state = WorkflowState()
        self.workflow_history: List[WorkflowState] = []

    def _init_agents(self) -> None:
        """Initialize all company agents."""
        # Use get_model_name() which accepts string role names
        ceo_model = self.model_config.get_model_name("ceo")
        cto_model = self.model_config.get_model_name("cto")
        pm_model = self.model_config.get_model_name("product_manager")
        researcher_model = self.model_config.get_model_name("researcher")
        dev_model = self.model_config.get_model_name("developer")
        qa_model = self.model_config.get_model_name("qa_engineer")
        devops_model = self.model_config.get_model_name("devops_engineer")
        data_analyst_model = self.model_config.get_model_name("data_analyst")
        security_model = self.model_config.get_model_name("security_engineer")

        self.ceo = CEOAgent(
            model=ceo_model,
            workspace_root=self.workspace_root,
            memory_persist_dir=self.memory_persist_dir
        )
        self.cto = CTOAgent(
            model=cto_model,
            workspace_root=self.workspace_root,
            memory_persist_dir=self.memory_persist_dir
        )
        self.product_manager = ProductManagerAgent(
            model=pm_model,
            workspace_root=self.workspace_root,
            memory_persist_dir=self.memory_persist_dir
        )
        self.researcher = ResearcherAgent(
            model=researcher_model,
            workspace_root=self.workspace_root,
            memory_persist_dir=self.memory_persist_dir
        )
        self.developer = DeveloperAgent(
            model=dev_model,
            workspace_root=self.workspace_root,
            memory_persist_dir=self.memory_persist_dir
        )
        self.qa_engineer = QAEngineerAgent(
            model=qa_model,
            workspace_root=self.workspace_root,
            memory_persist_dir=self.memory_persist_dir
        )
        self.devops_engineer = DevOpsEngineerAgent(
            model=devops_model,
            workspace_root=self.workspace_root,
            memory_persist_dir=self.memory_persist_dir
        )
        self.data_analyst = DataAnalystAgent(
            model=data_analyst_model,
            workspace_root=self.workspace_root,
            memory_persist_dir=self.memory_persist_dir
        )
        self.security_engineer = SecurityEngineerAgent(
            model=security_model,
            workspace_root=self.workspace_root,
            memory_persist_dir=self.memory_persist_dir
        )

        self.agents = {
            "CEO": self.ceo,
            "CTO": self.cto,
            "ProductManager": self.product_manager,
            "Researcher": self.researcher,
            "Developer": self.developer,
            "QAEngineer": self.qa_engineer,
            "DevOpsEngineer": self.devops_engineer,
            "DataAnalyst": self.data_analyst,
            "SecurityEngineer": self.security_engineer
        }

        # Enable ReAct tools for all agents that need to call tools in their loops
        for agent in [
            self.developer,
            self.researcher,
            self.cto,
            self.qa_engineer,
            self.security_engineer,
            self.devops_engineer,
            self.data_analyst,
        ]:
            if hasattr(agent, 'enable_react_tools'):
                agent.enable_react_tools()

    def _register_agents(self) -> None:
        """Register all agents with the message bus."""
        for name, agent in self.agents.items():
            self.message_bus.register_agent(name, agent)

        # Wire shared memory into all agents for institutional knowledge
        for agent in self.agents.values():
            agent._shared_memory = self.shared_memory

        # Set up topic subscriptions
        self.message_bus.subscribe("CEO", "decisions")
        self.message_bus.subscribe("CEO", "approvals")
        self.message_bus.subscribe("CTO", "technical")
        self.message_bus.subscribe("ProductManager", "product")
        self.message_bus.subscribe("Developer", "implementation")
        self.message_bus.subscribe("QAEngineer", "testing")
        self.message_bus.subscribe("DevOpsEngineer", "deployment")
        self.message_bus.subscribe("DataAnalyst", "research")
        self.message_bus.subscribe("SecurityEngineer", "security")

    # ============================================================
    #  MAIN WORKFLOW ENTRY POINT
    # ============================================================

    async def run_full_workflow(
        self,
        auto_discover: bool = True,
        problem: DiscoveredProblem = None,
        max_approval_rounds: int = 0,
        resume: bool = False,
        interactive: bool = False,
        session_id: Optional[str] = None,
        dry_run: bool = False,
        scaffold_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete company workflow.

        Args:
            auto_discover: If True, discover problems automatically
            problem: If provided, work on this specific problem
            max_approval_rounds: Max QA+CEO retry rounds (0 = unlimited, retries until CEO approves)
            resume: If True, attempt to resume from last checkpoint
            interactive: If True, pause at key decision points for user approval
            session_id: If provided, resume this specific session
            dry_run: If True, skip code execution, testing, and delivery

        Returns:
            Workflow result with all artifacts
        """
        self._interactive = interactive
        self.state = WorkflowState(started_at=datetime.now())
        self.progress.start()

        # Create or resume a session
        if session_id:
            session = self.session_manager.resume_session(session_id=session_id)
            if session:
                console.info(f"Resumed session: {session.id}")
            else:
                console.warning(f"Session {session_id} not found, creating new session")
                session = self.session_manager.create_session(
                    name=problem.description if problem else "auto-discover"
                )
        elif resume:
            session = self.session_manager.resume_session()
            if session:
                console.info(f"Resumed most recent session: {session.id}")
            else:
                session = self.session_manager.create_session(
                    name=problem.description if problem else "auto-discover"
                )
        else:
            session = self.session_manager.create_session(
                name=problem.description if problem else "auto-discover"
            )

        # Set language early from problem metadata so all phases can access it
        if problem and problem.metadata.get("language"):
            self.state.artifacts["language"] = problem.metadata["language"]

        # Run pre-flight health checks
        console.info("Running pre-flight health checks...")
        health_results = self.health_checker.run_all_checks()
        overall_health = self.health_checker.get_overall_status()
        if overall_health == HealthStatus.UNHEALTHY:
            unhealthy = [name for name, r in health_results.items() if r.status == HealthStatus.UNHEALTHY]
            console.error(f"System unhealthy: {', '.join(unhealthy)}. Workflow may fail.")
        elif overall_health == HealthStatus.DEGRADED:
            console.warning("System health degraded. Proceeding with caution.")
        else:
            console.success("All health checks passed")

        # Start background health monitoring (daemon thread, stops with process)
        self.health_checker.start_background_checks()

        # Start cost tracking session
        cost_session_id = self.cost_tracker.start_session()
        console.info(f"Cost tracking session: {cost_session_id}")

        # Start structured logging context for this workflow run
        wf_log_ctx = self.structured_logger.new_context(workflow_id=cost_session_id)
        self.structured_logger.context = wf_log_ctx
        self.structured_logger.info("Workflow started")

        # Start memory monitoring
        self.memory_monitor.start_monitoring()

        # Trigger session start hook
        from utils.hooks import HookContext, HookEvent
        await self.hooks_manager.session_start(session.id)

        # Adjust agent behavior based on past performance
        self._adjust_agent_behavior()

        # Load lessons from past retrospectives into agent prompts
        self._load_previous_learnings()

        # Run 1:1 performance reviews for underperformers (before main work)
        await self._run_performance_reviews()

        # Attempt to resume from checkpoint
        resumed_phase = None
        if resume and self._load_checkpoint():
            resumed_phase = self.state.phase
            console.info(f"Resumed from checkpoint at phase: {resumed_phase.value}")

        try:
            # Load previous session summary and inject action items into relevant agents
            try:
                import json as _json
                _summary_path = Path(self.data_dir) / "session_summary.json"
                if _summary_path.exists():
                    _prev = _json.loads(_summary_path.read_text())
                    _prev_items = _prev.get("action_items", [])
                    _prev_insights = _prev.get("key_insights", [])
                    if _prev_items or _prev_insights:
                        _prev_note = "\n\n[PREVIOUS SESSION LESSONS]\n"
                        if _prev_items:
                            _prev_note += "Action items from last run:\n" + "\n".join(f"- {a}" for a in _prev_items[:5]) + "\n"
                        if _prev_insights:
                            _prev_note += "Key insights from last run:\n" + "\n".join(f"- {i}" for i in _prev_insights[:3]) + "\n"
                        _prev_note += "Apply these learnings to avoid repeating past mistakes.\n"
                        for _agent in [self.cto, self.developer, self.product_manager]:
                            if _prev_note not in _agent.system_prompt:
                                _agent.system_prompt += _prev_note
                        console.info(f"Previous session lessons injected ({len(_prev_items)} action items, {len(_prev_insights)} insights)")
            except Exception as _sle:
                console.warning(f"Session summary load failed: {_sle}")

            # Inject cross-run strategy memory into CTO + Developer at start of run
            try:
                cross_run_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()
                if cross_run_context and len(cross_run_context.strip()) > 20:
                    cross_run_injection = (
                        f"\n\n[CROSS-RUN STRATEGY MEMORY]\n{cross_run_context}\n"
                        "These lessons were learned from previous runs on similar problems. "
                        "Do NOT repeat approaches that previously failed.\n"
                    )
                    for agent in [self.cto, self.developer]:
                        if cross_run_injection not in agent.system_prompt:
                            agent.system_prompt += cross_run_injection
                    console.info("Cross-run strategy memory injected into CTO and Developer prompts")
            except Exception as _sme:
                console.warning(f"Cross-run strategy memory injection failed: {_sme}")

            # CEO morning brief — sets priorities and constraints for the whole run
            await self._run_ceo_morning_brief(problem)

            # Morning standup for team alignment
            problem_name = problem.description if problem else "Auto-Discovery"
            await self._run_morning_standup(problem_name)

            # Phase 1: Research
            if resumed_phase and resumed_phase.value not in ("idle", "research"):
                self.progress.skip_phase("research")
            else:
                self.progress.start_phase("research")
                if auto_discover and not problem:
                    console.section("PHASE 1: RESEARCH")
                    console.info("Discovering problems from web sources...")
                    research_result = await self.error_recovery.retry_async(
                        self._run_research_phase,
                        "research_phase"
                    )
                    if not research_result.success:
                        console.error(f"Research phase failed after retries: {research_result.error}")
                elif problem:
                    self.state.current_problem = problem
                    self.state.artifacts["discovered_problems"] = [problem.to_dict()]
                    self.state.artifacts["user_provided_problem"] = True
                    console.info("Using provided problem/requirement")
                    console.show_problem(problem.to_dict())
                self.progress.complete_phase("research")
                if self._current_sprint_id:
                    try:
                        self.sprint_manager.complete_task(self._current_sprint_id, "research")
                    except ValueError:
                        pass
                self._save_checkpoint()

            if not self.state.current_problem:
                # Part 2b: Expand search scope instead of immediate FAILED
                console.warning("No problems found — expanding search scope")
                expanded = False
                for expand_attempt in range(2):
                    console.info(f"Expanded search attempt {expand_attempt + 1}/2...")
                    # Re-run research with broader web search
                    try:
                        web_problems = await self._discover_from_web_search()
                        if web_problems:
                            top = max(web_problems, key=lambda p: p.score)
                            self.state.current_problem = top
                            self.state.artifacts["discovered_problems"] = [p.to_dict() for p in web_problems]
                            console.success(f"Found problem on expanded search: {top.description[:60]}")
                            expanded = True
                            break
                    except Exception as e:
                        console.warning(f"Expanded search failed: {e}")

                if not expanded:
                    self.state.phase = WorkflowPhase.FAILED
                    self.state.error = "No problem discovered (even after expanded search)"
                    self.progress.fail_phase("research", "No problem discovered")
                    console.error("No problem discovered even after expanded search!")
                    self.health_checker.stop_background_checks()
                    self.memory_monitor.stop_monitoring()
                    self.cost_tracker.end_session()
                    result = self._get_workflow_result()
                    self._persist_result(result)
                    return result

            # Store discovered problem in shared memory
            if self.state.current_problem:
                self.shared_memory.store_problem(
                    description=self.state.current_problem.description,
                    source="auto_discovery" if not problem else "user_input",
                    severity=self.state.current_problem.severity.value if isinstance(self.state.current_problem.severity, ProblemSeverity) else str(self.state.current_problem.severity),
                    domain=self.state.current_problem.domain,
                    created_by="researcher"
                )

            # Publish to message bus
            self.message_bus.publish(
                topic="product",
                sender="Researcher",
                content=f"Discovered problem: {self.state.current_problem.description}",
                priority=3
            )

            # Quality gate: Research
            await self._check_quality_gate("research")

            # Phase 1b: Data Analysis (cross-validate, deduplicate, detect bias)
            # Skipped in scaffold_mode — not meaningful when spec is user-provided
            if scaffold_mode or self.scaffold_mode:
                self.progress.skip_phase("data_analysis")
                console.info("Scaffold mode: skipping DATA ANALYSIS phase")
            elif resumed_phase and resumed_phase.value not in ("idle", "research", "data_analysis"):
                self.progress.skip_phase("data_analysis")
            else:
                self.progress.start_phase("data_analysis")
                console.section("PHASE 1b: DATA ANALYSIS")
                console.info("DataAnalyst cross-validating and deduplicating research findings...")
                await self._run_data_analysis_phase()

                # Part 3c: Feed DataAnalyst cross-validation results back into problem scoring
                cross_val_output = self.state.artifacts.get("cross_validation_verdicts", {})
                discovered = self.state.artifacts.get("discovered_problems", [])
                if cross_val_output and discovered and self.state.current_problem:
                    # Use parsed structured verdicts from DataAnalyst LLM cross-validation
                    from utils.output_parser import StructuredOutputParser
                    parser = StructuredOutputParser()
                    parsed_verdicts = cross_val_output if isinstance(cross_val_output, dict) else parser.parse_data_analyst_verdict(str(cross_val_output))
                    findings_list = parsed_verdicts.get("findings", [])

                    # Map parsed findings back to problems by index
                    validation_results = {}
                    for i, p_dict in enumerate(discovered):
                        pid = p_dict.get("id", "")
                        if i < len(findings_list):
                            finding = findings_list[i]
                            validation_results[pid] = {
                                "verdict": finding.get("verdict", "UNCONFIRMED"),
                                "confidence": finding.get("confidence", 0.5),
                                "evidence_for": finding.get("evidence_for", ""),
                                "evidence_against": finding.get("evidence_against", ""),
                            }

                    if validation_results:
                        # Reconstruct DiscoveredProblem objects for adjustment
                        problem_objects = []
                        for p_dict in discovered:
                            try:
                                prob = DiscoveredProblem(
                                    id=p_dict.get("id", ""),
                                    description=p_dict.get("description", ""),
                                    severity=ProblemSeverity(p_dict.get("severity", "medium")),
                                    frequency=ProblemFrequency(p_dict.get("frequency", "common")),
                                    score=p_dict.get("score", 0.0),
                                    sources=p_dict.get("sources", []),
                                    evidence=p_dict.get("evidence", []),
                                    keywords=p_dict.get("keywords", []),
                                    domain=p_dict.get("domain", "general"),
                                    target_users=p_dict.get("target_users", ""),
                                )
                                problem_objects.append(prob)
                            except Exception as e:
                                console.warning(f"Skipping malformed problem dict during validation: {e}")
                                continue

                        adjusted = self.problem_discoverer.apply_validation_adjustments(
                            problem_objects, validation_results
                        )
                        self.state.artifacts["discovered_problems"] = [p.to_dict() for p in adjusted]

                        # Update current problem if it was adjusted
                        if adjusted:
                            new_top = max(adjusted, key=lambda p: p.score)
                            if new_top.id != self.state.current_problem.id:
                                console.info(f"Top problem changed after validation feedback: {new_top.description[:60]}")
                                self.state.current_problem = new_top

                        console.info(f"Applied validation adjustments to {len(validation_results)} problems")

                # Quality gate: Data Analysis
                await self._check_quality_gate("data_analysis")

                self.progress.complete_phase("data_analysis")
                self._save_checkpoint()

            # Phase 2: Analysis
            if resumed_phase and resumed_phase.value not in ("idle", "research", "analysis"):
                self.progress.skip_phase("analysis")
            else:
                self.progress.start_phase("analysis")
                console.section("PHASE 2: ANALYSIS")
                console.info("Researcher and Product Manager analyzing the problem...")
                analysis_result = await self.error_recovery.retry_async(
                    self._run_analysis_phase,
                    "analysis_phase"
                )
                if not analysis_result.success:
                    console.error(f"Analysis phase failed after retries: {analysis_result.error}")
                self.progress.complete_phase("analysis")
                self._save_checkpoint()

            # Phase 3: Opportunity Evaluation (CEO Decision)
            # Skipped in scaffold_mode — user has already defined the spec
            if scaffold_mode or self.scaffold_mode:
                self.progress.skip_phase("opportunity_evaluation")
                console.info("Scaffold mode: skipping OPPORTUNITY EVALUATION phase (spec is user-provided)")
            elif resumed_phase and resumed_phase.value not in ("idle", "research", "analysis", "opportunity_evaluation"):
                self.progress.skip_phase("opportunity_evaluation")
            else:
                self.progress.start_phase("opportunity_evaluation")
                console.section("PHASE 3: OPPORTUNITY EVALUATION")
                console.info("CEO evaluating the opportunity with team meeting...")
                if not await self._run_opportunity_evaluation():
                    # Pivot: PM narrows scope and re-evaluate once
                    console.warning("Opportunity rejected by CEO. PM narrowing scope...")
                    narrow_task = {
                        "type": "define_requirements",
                        "problem": self.state.current_problem.to_dict(),
                        "target_users": self.state.current_problem.target_users,
                        "constraint": "Narrow the scope to the smallest viable subset that addresses the core pain point. Reduce complexity significantly."
                    }
                    narrow_result = await self.product_manager.execute_task(narrow_task)
                    self.state.artifacts["narrowed_requirements"] = narrow_result.output
                    # Update problem description with narrowed scope
                    if isinstance(narrow_result.output, dict):
                        narrowed_desc = narrow_result.output.get("requirements", "")
                        if narrowed_desc:
                            self.state.current_problem.description = f"[Narrowed] {self.state.current_problem.description}"
                            self.state.current_problem.potential_solution_ideas = [narrowed_desc]

                    # Re-evaluate with narrowed scope
                    console.info("Re-evaluating narrowed opportunity...")
                    if not await self._run_opportunity_evaluation():
                        # Part 2a: Try next problem from discovered list instead of FAILED
                        discovered = self.state.artifacts.get("discovered_problems", [])
                        pivoted = False
                        if len(discovered) > 1:
                            # Remove the rejected problem and try the next best
                            current_id = self.state.current_problem.id if self.state.current_problem else None
                            remaining = [p for p in discovered if p.get("id") != current_id]
                            if remaining:
                                best = max(remaining, key=lambda p: p.get("score", 0))
                                console.info(f"CEO rejected — pivoting to next problem: {best.get('description', '')[:60]}")
                                # Reconstruct DiscoveredProblem from dict
                                self.state.current_problem = DiscoveredProblem(
                                    id=best.get("id", "PIVOT-001"),
                                    description=best.get("description", ""),
                                    severity=ProblemSeverity(best.get("severity", "medium")),
                                    frequency=ProblemFrequency(best.get("frequency", "common")),
                                    target_users=best.get("target_users", ""),
                                    evidence=best.get("evidence", []),
                                    sources=best.get("sources", []),
                                    domain=best.get("domain", "general"),
                                    keywords=best.get("keywords", []),
                                    score=best.get("score", 0.0),
                                )
                                self.state.artifacts["discovered_problems"] = remaining
                                # Re-run opportunity evaluation with new problem
                                if await self._run_opportunity_evaluation():
                                    pivoted = True

                        if not pivoted:
                            self.state.phase = WorkflowPhase.FAILED
                            self.state.error = "Opportunity rejected by CEO (all problems exhausted)"
                            self.progress.fail_phase("opportunity_evaluation", "Rejected by CEO after pivot")
                            console.error("All problems rejected by CEO")
                            self.health_checker.stop_background_checks()
                            self.memory_monitor.stop_monitoring()
                            self.cost_tracker.end_session()
                            result = self._get_workflow_result()
                            self._persist_result(result)
                            return result
                console.success("Opportunity approved!")

                # Store decision in shared memory
                problem = self.state.current_problem
                last_decision = self.state.decisions[-1] if self.state.decisions else {}
                self.shared_memory.store_decision(
                    decision=f"Approved: {problem.description}",
                    reasoning=last_decision.get("reasoning", ""),
                    made_by="CEO",
                    participants=["CEO", "CTO", "ProductManager"]
                )

                # Announce approval via message bus
                self.message_bus.announce(
                    sender="CEO",
                    announcement=f"Approved project: {problem.description}"
                )

                self.progress.complete_phase("opportunity_evaluation")
                self._save_checkpoint()

                if not self._interactive_gate("Opportunity Evaluation", f"Problem: {problem.description}"):
                    self.state.phase = WorkflowPhase.FAILED
                    self.state.error = "Aborted by user"
                    result = self._get_workflow_result()
                    self._persist_result(result)
                    return result

            # Phase 4: Technical Design
            if resumed_phase and resumed_phase.value not in ("idle", "research", "analysis", "opportunity_evaluation", "technical_design"):
                self.progress.skip_phase("technical_design")
            else:
                self.progress.start_phase("technical_design")
                console.section("PHASE 4: TECHNICAL DESIGN")
                console.info("CTO designing technical architecture...")
                design_result = await self.error_recovery.retry_async(
                    self._run_technical_design_phase,
                    "technical_design_phase"
                )
                if not design_result.success:
                    console.error(f"Technical design phase failed after retries: {design_result.error}")
                console.success("Technical design complete")

                # Quality gate: Tech Design (root cause must be extracted)
                await self._check_quality_gate("tech_design")

                self.progress.complete_phase("technical_design")
                if self._current_sprint_id:
                    try:
                        self.sprint_manager.complete_task(self._current_sprint_id, "design")
                    except ValueError:
                        pass
                self._save_checkpoint()

                if not self._interactive_gate("Technical Design", "Architecture and requirements defined. Ready to implement."):
                    self.state.phase = WorkflowPhase.FAILED
                    self.state.error = "Aborted by user"
                    result = self._get_workflow_result()
                    self._persist_result(result)
                    return result

            # Phase 4b: Design Review (CTO + Developer + QA review architecture before coding)
            if resumed_phase and resumed_phase.value not in ("idle", "research", "data_analysis", "analysis", "opportunity_evaluation", "technical_design", "design_review"):
                self.progress.skip_phase("design_review")
            else:
                self.progress.start_phase("design_review")
                console.section("PHASE 4b: DESIGN REVIEW")
                console.info("CTO, Developer, and QA reviewing architecture before implementation...")
                await self._run_design_review_phase()
                self.progress.complete_phase("design_review")
                self._save_checkpoint()

            # Phase 5: Implementation
            if resumed_phase and resumed_phase.value not in ("idle", "research", "analysis", "opportunity_evaluation", "technical_design", "implementation"):
                self.progress.skip_phase("implementation")
            else:
                self.progress.start_phase("implementation")
                console.section("PHASE 5: IMPLEMENTATION")
                console.info("Developer implementing the solution...")
                impl_recovery = await self.error_recovery.retry_async(
                    self._run_implementation_phase,
                    "implementation_phase"
                )
                if not impl_recovery.success:
                    console.error(f"Implementation phase failed after retries: {impl_recovery.error}")
                console.success("Implementation complete")
                if self._current_sprint_id:
                    try:
                        self.sprint_manager.complete_task(self._current_sprint_id, "implement")
                    except ValueError:
                        pass

                # Notify message bus
                self.message_bus.publish(
                    topic="testing",
                    sender="Developer",
                    content="Implementation complete. Ready for QA review.",
                    priority=3
                )

                self.progress.complete_phase("implementation")
                self._save_checkpoint()

                # MoA: CEO and CTO independently review the implementation
                try:
                    if hasattr(self, 'ceo') and hasattr(self, 'cto'):
                        from collaboration.moa_aggregator import MoAReviewAggregator
                        _impl_artifact = self.state.artifacts.get("implementation", {})
                        if isinstance(_impl_artifact, dict):
                            impl_code = str(_impl_artifact.get("code", "") or _impl_artifact.get("implementation", ""))[:3000]
                        else:
                            impl_code = str(_impl_artifact)[:3000]
                        if impl_code:
                            ceo_review = await self.ceo.generate_response_async(
                                f"Review this implementation for business requirements alignment and quality:\n\n{impl_code[:2000]}\n\n"
                                f"Problem being solved: {self.state.current_problem.description}\n\n"
                                "Identify top 3 concerns. Be specific."
                            )
                            cto_review = await self.cto.generate_response_async(
                                f"Review this implementation for technical correctness and architecture:\n\n{impl_code[:2000]}\n\n"
                                "Identify top 3 technical issues. Be specific."
                            )
                            aggregator = MoAReviewAggregator(synthesizer_agent=self.qa_engineer)
                            moa_review = await aggregator.aggregate_reviews(
                                artifact=impl_code,
                                reviews={"CEO": ceo_review, "CTO": cto_review},
                                context=f"Problem: {self.state.current_problem.description}"
                            )
                            self.state.artifacts["moa_review"] = moa_review
                            console.info("MoA review complete — synthesized CEO + CTO feedback")

                            # Critic ensemble pass on the implementation
                            try:
                                from collaboration.critic_ensemble import CriticEnsemble
                                critic_ensemble = CriticEnsemble(personas=["skeptic", "pragmatist", "user"])
                                critique_result = await critic_ensemble.critique(
                                    agent=self.qa_engineer,
                                    artifact=impl_code,
                                    task_context=self.state.current_problem.description,
                                    synthesize=True,
                                )
                                self.state.artifacts["critic_ensemble_review"] = critique_result.get("synthesis", "")
                                console.info(f"Critic ensemble review complete — verdict: {critique_result.get('verdict', 'N/A')}")
                            except Exception as _critic_err:
                                console.warning(f"Critic ensemble review skipped: {_critic_err}")
                except Exception as _moa_err:
                    console.warning(f"MoA review skipped: {_moa_err}")

                if not self._interactive_gate("Implementation", "Code generated. Proceeding to testing and QA."):
                    self.state.phase = WorkflowPhase.FAILED
                    self.state.error = "Aborted by user"
                    result = self._get_workflow_result()
                    self._persist_result(result)
                    return result

            # Part 4b: Extract root cause statement from developer output for QA context
            impl_output = self.state.artifacts.get("implementation", {})
            root_cause_statement = ""
            if isinstance(impl_output, dict):
                impl_text = impl_output.get("implementation", "")
            else:
                impl_text = str(impl_output)
            # Look for ROOT CAUSE: line in developer output
            import re as _re
            _rc_match = _re.search(r'ROOT\s*CAUSE:\s*(.+?)(?:\n|$)', impl_text, _re.IGNORECASE)
            if _rc_match:
                root_cause_statement = _rc_match.group(1).strip()
                console.info(f"Root cause identified: {root_cause_statement[:80]}")
                self.state.artifacts["root_cause"] = root_cause_statement

            # Phases 6-8: QA + Security Validation + CEO Approval loop
            # Uses EscalationManager for graduated recovery when enabled
            max_qa_retries = 2
            approval_round = 0
            dev_fix_count = 0
            failure_history: List[str] = []
            # Never stop unless force_stop is set — unlimited rounds
            max_rounds = max_approval_rounds if (max_approval_rounds > 0 and getattr(self, 'force_stop', False)) else 999

            while True:
                approval_round += 1

                # Check token budget limit — allow up to 3 extensions before partial delivery
                if self.cost_tracker.is_over_budget():
                    budget_extensions = getattr(self, '_budget_extensions', 0)
                    self._budget_extensions = budget_extensions + 1
                    pct = max(0.1, 0.3 - (budget_extensions * 0.05))  # Decreasing but never zero
                    console.warning(f"Token budget reached (extension {budget_extensions + 1}). Extending by {int(pct*100)}% and compacting agent memories...")
                    self.cost_tracker.extend_budget(pct)
                    for agent in [self.developer, self.cto, self.product_manager, self.researcher, self.qa_engineer]:
                        if hasattr(agent, 'memory'):
                            agent.memory.compact()
                    # Emergency PM rescope on first hit
                    if budget_extensions == 0 and self.enable_escalation and hasattr(self, 'product_manager'):
                        console.info("Emergency PM rescope to reduce complexity...")
                        rescope_task = {
                            "type": "define_requirements",
                            "problem": self.state.current_problem.to_dict() if self.state.current_problem else {},
                            "target_users": getattr(self.state.current_problem, 'target_users', ''),
                            "constraint": "Budget nearly exhausted. Reduce to absolute minimum viable scope."
                        }
                        rescope_result = await self.product_manager.execute_task(rescope_task)
                        self.state.artifacts["requirements"] = rescope_result.output

                # Check time-based soft stop — allow up to 3 extensions before partial delivery
                if self.max_workflow_minutes > 0 and self.state.started_at:
                    elapsed_minutes = (datetime.now() - self.state.started_at).total_seconds() / 60
                    if elapsed_minutes >= self.max_workflow_minutes:
                        time_extensions = getattr(self, '_time_extensions', 0)
                        self._time_extensions = time_extensions + 1
                        pct = max(0.2, 0.5 - (time_extensions * 0.1))  # Decreasing but never zero
                        extended = int(self.max_workflow_minutes * pct)
                        self.max_workflow_minutes += extended
                        console.warning(f"Time limit reached (extension {time_extensions + 1}). Extending by {extended} minutes. Compacting memories...")
                        for agent in [self.developer, self.cto, self.product_manager, self.researcher, self.qa_engineer]:
                            if hasattr(agent, 'memory'):
                                agent.memory.compact()

                round_label = f"{approval_round}/{max_rounds}"

                # Inject post-mortem context when starting a new escalation cycle
                if approval_round > 1 and (approval_round - 1) % 11 == 0:
                    post_mortem = self.escalation_manager.generate_post_mortem(
                        workflow_state={"phase": self.state.phase.value},
                        failure_history=failure_history,
                    )
                    pm_context = (
                        f"POST-MORTEM AFTER {post_mortem['total_rounds']} ROUNDS:\n"
                        f"Pattern: {post_mortem['failure_pattern']}\n"
                        f"Recommendations: {'; '.join(post_mortem['recommendations'][:3])}\n"
                    )
                    self.state.artifacts["post_mortem_context"] = pm_context
                    console.info(f"Post-mortem injected for cycle-back (round {approval_round})")

                console.section(f"APPROVAL ROUND {round_label}")

                # === PHASE 5b: CODE EXECUTION CHECK ===
                if self.run_code_execution:
                    self.progress.start_phase("code_execution")
                    console.section(f"PHASE 5b: CODE EXECUTION CHECK (Round {round_label})")
                    console.info("Running generated code to check for errors...")
                    execution_results = await self._run_code_execution()
                    self.state.artifacts["execution_results"] = execution_results

                    if not execution_results["success"]:
                        console.error(f"Code execution failed: {execution_results['summary']}")
                        project_dir = self.state.artifacts.get("project_name", self.output_dir)
                        exec_fix_task = {
                            "type": "fix_bug",
                            "bug_description": f"Code execution failed with errors. Fix these runtime errors:\n{execution_results['summary']}",
                            "error_message": "\n".join(execution_results["runtime_errors"] + execution_results["syntax_errors"]),
                            "qa_report": execution_results.get("test_output", ""),
                            "original_requirements": str(self.state.artifacts.get("requirements", "")),
                            "output_dir": project_dir
                        }
                        fix_result = await self.developer.execute_task(exec_fix_task)
                        self.state.artifacts["execution_fix"] = fix_result.output
                        dev_fix_count += 1
                        console.success("Developer applied execution fixes")

                        execution_results = await self._run_code_execution()
                        self.state.artifacts["execution_results"] = execution_results
                        if execution_results["success"]:
                            console.success("Code now passes execution checks")
                        else:
                            console.warning(f"Code still has execution errors: {execution_results['summary']}")
                    else:
                        console.success(f"Code execution check passed: {execution_results['summary']}")
                    self.progress.complete_phase("code_execution")
                else:
                    console.info("Code execution check skipped (disabled in config)")
                    self.progress.skip_phase("code_execution")

                self._save_checkpoint()

                # Phase 5.5: CTO peer review before QA on round >= 3
                if approval_round >= 3:
                    console.section(f"PHASE 5.5: CTO PEER REVIEW (Round {round_label})")
                    console.info("CTO reviewing implementation before QA (escalation catch)...")
                    impl_files = self.state.artifacts.get("implementation", {}).get("files_written", [])
                    impl_code = self.state.artifacts.get("implementation", {}).get("code", "")
                    cto_review_task = {
                        "type": "review_code",
                        "code": impl_code[:3000] if impl_code else "(see files: " + ", ".join(impl_files[:5]) + ")",
                        "file_path": ", ".join(impl_files[:5]) if impl_files else "unknown",
                        "context": (
                            f"Round {approval_round} peer review before QA. "
                            f"Problem: {self.state.current_problem.description if self.state.current_problem else 'unknown'}. "
                            f"Architecture: {self.state.artifacts.get('technical_design', {}).get('architecture', '')[:300]}"
                        ),
                    }
                    try:
                        cto_review_result = await self.cto.execute_task(cto_review_task)
                        cto_peer_review = cto_review_result.output.get("review", cto_review_result.output.get("response", str(cto_review_result.output)))
                        self.state.artifacts["cto_peer_review"] = cto_peer_review
                        console.info(f"CTO peer review complete. Issues noted: {str(cto_peer_review)[:200]}")
                        # Notify Developer of CTO findings
                        self.message_bus.notify_issue(
                            from_agent="CTO",
                            to_agent="Developer",
                            issue=f"CTO peer-review (round {approval_round}): {str(cto_peer_review)[:500]}"
                        )
                    except Exception as _e:
                        console.warning(f"CTO peer review failed: {_e}")
                        self.state.artifacts["cto_peer_review"] = ""

                # Phase 6: QA Validation with inner feedback loop
                qa_passed = False
                self.progress.start_phase("qa_validation")
                for qa_attempt in range(1, max_qa_retries + 1):
                    console.section(f"PHASE 6: QA VALIDATION (Round {round_label}, Attempt {qa_attempt}/{max_qa_retries})")
                    console.info("QA Engineer validating the solution...")
                    qa_passed = await self._run_qa_phase()

                    if qa_passed:
                        break

                    if qa_attempt < max_qa_retries:
                        console.warning(f"QA failed on attempt {qa_attempt}. Sending feedback to Developer...")
                        qa_feedback = self.state.artifacts.get("qa_validation", {})
                        qa_report = self.state.artifacts.get("qa_report", {})

                        console.section(f"PHASE 6a: DEVELOPER FIX (QA Iteration {qa_attempt})")
                        console.info("Developer fixing issues found by QA...")
                        project_dir = self.state.artifacts.get("project_name", self.output_dir)
                        execution_results = self.state.artifacts.get("execution_results", {})
                        # Parse QA result for root cause feedback
                        qa_root_cause_note = ""
                        if isinstance(qa_feedback, dict):
                            validation_text = qa_feedback.get("validation", "")
                            from utils.output_parser import StructuredOutputParser
                            qa_parser = StructuredOutputParser()
                            qa_parsed = qa_parser.parse_qa_result(str(validation_text))
                            if qa_parsed.get("root_cause_addressed") is False:
                                qa_root_cause_note = (
                                    f"\nROOT CAUSE NOT ADDRESSED: {qa_parsed.get('root_cause_explanation', 'QA found solution treats symptoms, not root cause.')}"
                                    f"\nStated root cause: {self.state.artifacts.get('root_cause', 'unknown')}"
                                )

                        fix_task = {
                            "type": "fix_bug",
                            "bug_description": f"QA validation failed. Issues: {str(qa_feedback)}{qa_root_cause_note}",
                            "error_message": execution_results.get("summary", ""),
                            "qa_report": str(qa_report),
                            "original_requirements": str(self.state.artifacts.get("requirements", "")),
                            "output_dir": project_dir
                        }
                        fix_result = await self.developer.execute_task(fix_task)
                        self.state.artifacts["fix_iteration"] = fix_result.output
                        dev_fix_count += 1
                        console.success(f"Developer applied fixes (QA iteration {qa_attempt})")

                        if self.run_code_execution:
                            execution_results = await self._run_code_execution()
                            self.state.artifacts["execution_results"] = execution_results

                if qa_passed:
                    self.progress.complete_phase("qa_validation")
                    if self._current_sprint_id:
                        try:
                            self.sprint_manager.complete_task(self._current_sprint_id, "qa")
                        except ValueError:
                            pass
                else:
                    self.progress.fail_phase("qa_validation", "QA validation failed")

                # Phase 6b: Security Review (runs alongside QA results, both read-only)
                security_passed = True
                if self.enable_security_review and qa_passed:
                    self.progress.start_phase("security_review")
                    console.section(f"PHASE 6b: SECURITY REVIEW (Round {round_label})")
                    console.info("Security Engineer reviewing the solution...")
                    security_passed = await self._run_security_review_phase()
                    if security_passed:
                        self.progress.complete_phase("security_review")
                    else:
                        self.progress.fail_phase("security_review", "Security issues found")
                elif not self.enable_security_review:
                    self.progress.skip_phase("security_review")

                if not qa_passed or not security_passed:
                    # Collect failure feedback for escalation
                    qa_feedback_str = str(self.state.artifacts.get("qa_validation", ""))
                    if not security_passed:
                        qa_feedback_str += " | " + str(self.state.artifacts.get("security_review", ""))
                    failure_history.append(qa_feedback_str)
                    # QA notifies Developer of issues via message bus
                    self.message_bus.notify_issue(
                        "QAEngineer", "Developer",
                        f"Validation failed (round {approval_round}): {qa_feedback_str[:400]}",
                        "high"
                    )
                    # Incident response meeting on persistent failures (round >= 5)
                    if approval_round >= 5:
                        incident_prompt = self.incident_manager.generate_incident_prompt(
                            failure_description=qa_feedback_str[:300],
                            failure_history=failure_history,
                            round_number=approval_round
                        )
                        console.section(f"INCIDENT RESPONSE (Round {approval_round})")
                        console.warning(incident_prompt[:600])
                        self.state.artifacts["incident_analysis"] = incident_prompt

                    # OrgChart-aware failure routing
                    failing_agent = self._detect_failure_owner(qa_feedback_str)
                    supervisor = self.org_chart.get_supervisor(failing_agent)
                    if supervisor:
                        console.info(f"Escalation: {failing_agent}'s area flagged -> supervisor {supervisor} notified")
                        # Actually notify supervisor via message bus (OrgChart routing)
                        severity = "high" if approval_round >= 3 else "medium"
                        self.message_bus.notify_issue(
                            failing_agent, supervisor,
                            f"Round {approval_round} quality failure in {failing_agent}'s area: {qa_feedback_str[:300]}",
                            severity=severity
                        )

                    # Log trust score for the failing agent
                    trust = self.trust_tracker.get_trust("qa_engineer", failing_agent)
                    if trust < 0.3:
                        console.warning(f"Low trust in {failing_agent} ({trust:.2f}). Suggesting earlier escalation.")

                    if self.enable_escalation:
                        # Use EscalationManager to decide next action
                        qa_verdict = "fail" if not qa_passed else "security_fail"
                        action = self.escalation_manager.should_escalate(
                            round=approval_round,
                            qa_verdict=qa_verdict,
                            dev_fix_count=dev_fix_count,
                            failure_history=failure_history
                        )
                        console.info(f"Escalation decision: {action.value}")

                        if action == EscalationAction.DEVELOPER_FIX:
                            console.warning(f"QA/Security failed in round {approval_round}. Developer reworking solution...")
                            project_dir = self.state.artifacts.get("project_name", self.output_dir)
                            qa_feedback = self.state.artifacts.get("qa_validation", {})
                            qa_report = self.state.artifacts.get("qa_report", {})
                            execution_results = self.state.artifacts.get("execution_results", {})
                            strategy_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()
                            incident_context = self.state.artifacts.get("incident_analysis", "")
                            dev_messages = self._drain_messages_for_agent("Developer")
                            rework_task = {
                                "type": "fix_bug",
                                "bug_description": f"QA/Security validation failed. Issues: {str(qa_feedback)}",
                                "error_message": execution_results.get("summary", ""),
                                "qa_report": str(qa_report),
                                "original_requirements": str(self.state.artifacts.get("requirements", "")),
                                "output_dir": project_dir,
                                "strategy_context": f"{strategy_context}\n{incident_context}\n{dev_messages}"[:1200] if (incident_context or dev_messages) else strategy_context
                            }
                            rework_result = await self.developer.execute_task(rework_task)
                            self.state.artifacts[f"rework_round_{approval_round}"] = rework_result.output
                            self.escalation_manager.strategy_memory.record_attempt(
                                approach=action.value, round=approval_round,
                                outcome="attempted", feedback=failure_history[-1] if failure_history else ""
                            )
                            dev_fix_count += 1
                            console.success(f"Developer reworked solution (round {approval_round})")
                            continue

                        elif action == EscalationAction.CTO_REDESIGN:
                            console.warning("Escalating to CTO for architecture redesign...")
                            strategy_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()
                            needs_deeper = self.state.artifacts.get("needs_deeper_root_cause", False)
                            deeper_suffix = "\nWARNING: Previous root cause analysis was too shallow. Apply 5-WHY analysis: ask WHY 5 times before stating the root cause." if needs_deeper else ""
                            redesign_task = {
                                "type": "design_architecture",
                                "problem": self.state.current_problem.to_dict(),
                                "requirements": str(self.state.artifacts.get("requirements", "")),
                                "constraint": f"Previous design failed QA {approval_round} times. Failure history: {failure_history[-1]}. Redesign the technical approach from scratch.\n{strategy_context}{deeper_suffix}"
                            }
                            redesign_result = await self.cto.execute_task(redesign_task)
                            self.state.artifacts["architecture"] = redesign_result.output
                            self.escalation_manager.strategy_memory.record_attempt(
                                approach=action.value, round=approval_round,
                                outcome="attempted", feedback=failure_history[-1] if failure_history else ""
                            )
                            console.success("CTO redesigned architecture")
                            # Re-implement with new architecture
                            console.info("Developer re-implementing with new architecture...")
                            await self._run_implementation_phase()
                            console.success("Re-implementation complete")
                            continue

                        elif action == EscalationAction.PM_RESCOPE:
                            console.warning("Escalating to PM for requirements rescoping...")
                            strategy_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()
                            rescope_task = {
                                "type": "define_requirements",
                                "problem": self.state.current_problem.to_dict(),
                                "target_users": self.state.current_problem.target_users,
                                "constraint": f"Previous implementation failed after {approval_round} rounds. Reduce scope to the minimum viable subset. Drop non-essential features.\n{strategy_context}"
                            }
                            rescope_result = await self.product_manager.execute_task(rescope_task)
                            self.state.artifacts["requirements"] = rescope_result.output
                            self.escalation_manager.strategy_memory.record_attempt(
                                approach=action.value, round=approval_round,
                                outcome="attempted", feedback=failure_history[-1] if failure_history else ""
                            )
                            console.success("PM rescoped requirements")
                            # Redesign and re-implement with reduced scope
                            await self._run_technical_design_phase()
                            await self._run_implementation_phase()
                            console.success("Re-implementation with reduced scope complete")
                            continue

                        elif action == EscalationAction.TEAM_BRAINSTORM:
                            console.warning("Escalating to team brainstorm — all agents contribute solutions...")
                            brainstorm_result = await self._team_brainstorm()
                            if brainstorm_result:
                                self.state.artifacts["brainstorm_solutions"] = brainstorm_result
                                await self._run_implementation_phase()
                                console.success("Re-implementation after brainstorm complete")
                            continue

                        elif action == EscalationAction.DECOMPOSE_PROBLEM:
                            console.warning("Decomposing problem into sub-problems...")
                            sub_problems = await self._decompose_problem()
                            if sub_problems:
                                self.state.artifacts["sub_problems"] = sub_problems
                                await self._run_implementation_phase()
                                console.success("Re-implementation after decomposition complete")
                            continue

                        elif action == EscalationAction.PIVOT_APPROACH:
                            console.warning("Pivoting to a completely different approach...")
                            await self._pivot_approach()
                            await self._run_implementation_phase()
                            console.success("Re-implementation after pivot complete")
                            continue

                        elif action == EscalationAction.SIMPLIFY:
                            console.warning("Simplifying to absolute minimum viable solution...")
                            strategy_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()
                            simplify_task = {
                                "type": "define_requirements",
                                "problem": self.state.current_problem.to_dict(),
                                "target_users": self.state.current_problem.target_users,
                                "constraint": f"SIMPLIFY: Reduce to the absolute minimum. 1 file, 1 function, core feature ONLY. Cut everything else.\n{strategy_context}"
                            }
                            simplify_result = await self.product_manager.execute_task(simplify_task)
                            self.state.artifacts["requirements"] = simplify_result.output
                            self.escalation_manager.strategy_memory.record_attempt(
                                approach=action.value, round=approval_round,
                                outcome="attempted", feedback=failure_history[-1] if failure_history else ""
                            )
                            await self._run_technical_design_phase()
                            await self._run_implementation_phase()
                            console.success("Re-implementation with simplified scope complete")
                            continue

                        elif action == EscalationAction.ALTERNATIVE_STACK:
                            console.warning("Trying alternative technology stack...")
                            strategy_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()
                            alt_stack_task = {
                                "type": "design_architecture",
                                "problem": self.state.current_problem.to_dict(),
                                "requirements": str(self.state.artifacts.get("requirements", "")),
                                "constraint": f"ALTERNATIVE STACK: The current tech stack has failed repeatedly. Try a completely different language/framework.\n{strategy_context}"
                            }
                            alt_stack_result = await self.cto.execute_task(alt_stack_task)
                            self.state.artifacts["architecture"] = alt_stack_result.output
                            self.escalation_manager.strategy_memory.record_attempt(
                                approach=action.value, round=approval_round,
                                outcome="attempted", feedback=failure_history[-1] if failure_history else ""
                            )
                            await self._run_implementation_phase()
                            console.success("Re-implementation with alternative stack complete")
                            continue

                        elif action == EscalationAction.FRESH_START:
                            console.warning("FRESH START: Discarding all previous work and re-analyzing from scratch...")
                            self.escalation_manager.strategy_memory.record_attempt(
                                approach=action.value, round=approval_round,
                                outcome="attempted", feedback="Fresh start triggered"
                            )
                            # Re-run analysis from problem understanding
                            await self._run_analysis_phase()
                            await self._run_technical_design_phase()
                            await self._run_implementation_phase()
                            console.success("Re-implementation after fresh start complete")
                            continue

                    else:
                        # Escalation disabled: use legacy max_approval_rounds behavior
                        has_limit = max_approval_rounds > 0
                        at_limit = has_limit and approval_round >= max_approval_rounds
                        if not at_limit:
                            console.warning(f"QA/Security failed in round {approval_round}. Developer reworking solution...")
                            project_dir = self.state.artifacts.get("project_name", self.output_dir)
                            qa_feedback = self.state.artifacts.get("qa_validation", {})
                            qa_report = self.state.artifacts.get("qa_report", {})
                            execution_results = self.state.artifacts.get("execution_results", {})
                            rework_task = {
                                "type": "fix_bug",
                                "bug_description": f"Validation failed after {max_qa_retries} attempts. Issues: {str(qa_feedback)}",
                                "error_message": execution_results.get("summary", ""),
                                "qa_report": str(qa_report),
                                "original_requirements": str(self.state.artifacts.get("requirements", "")),
                                "output_dir": project_dir
                            }
                            rework_result = await self.developer.execute_task(rework_task)
                            self.state.artifacts[f"rework_round_{approval_round}"] = rework_result.output
                            dev_fix_count += 1
                            console.success(f"Developer reworked solution (round {approval_round})")
                            continue
                        else:
                            if getattr(self, 'force_stop', False):
                                console.warning(f"Validation failed after {approval_round} rounds. Force-stop enabled — delivering best available solution.")
                                self.state.phase = WorkflowPhase.COMPLETED
                                self.state.artifacts["partial_delivery"] = True
                                self.state.artifacts["delivery_note"] = f"Solution delivered after {approval_round} rounds. QA issues remain — manual review recommended."
                                break
                            else:
                                # Never stop: rework and continue
                                console.warning(f"QA/Security failed in round {approval_round} (at previous limit). Reworking — never giving up...")
                                project_dir = self.state.artifacts.get("project_name", self.output_dir)
                                qa_feedback = self.state.artifacts.get("qa_validation", {})
                                qa_report = self.state.artifacts.get("qa_report", {})
                                execution_results = self.state.artifacts.get("execution_results", {})
                                rework_task = {
                                    "type": "fix_bug",
                                    "bug_description": f"Validation failed after {approval_round} rounds. Issues: {str(qa_feedback)}",
                                    "error_message": execution_results.get("summary", ""),
                                    "qa_report": str(qa_report),
                                    "original_requirements": str(self.state.artifacts.get("requirements", "")),
                                    "output_dir": project_dir
                                }
                                rework_result = await self.developer.execute_task(rework_task)
                                self.state.artifacts[f"rework_round_{approval_round}"] = rework_result.output
                                dev_fix_count += 1
                                console.success(f"Developer reworked solution (round {approval_round})")
                                continue

                # Publish QA verdict via message bus
                qa_output = self.state.artifacts.get("qa_validation", {})
                verdict = qa_output.get("verdict", "unknown") if isinstance(qa_output, dict) else "unknown"
                self.message_bus.publish(
                    topic="decisions",
                    sender="QAEngineer",
                    content=f"QA verdict: {verdict.upper()}",
                    priority=4
                )

                self._save_checkpoint()

                # Phase 6c: DevOps Validation (after QA+Security pass, before CEO)
                if qa_passed and security_passed:
                    console.section(f"PHASE 6c: DEVOPS VALIDATION (Round {round_label})")
                    console.info("DevOps Engineer validating deployability...")
                    try:
                        devops_task = {
                            "type": "validate_deployment",
                            "description": f"Validate deployment readiness for: {self.state.current_problem.description if self.state.current_problem else 'solution'}",
                            "project_dir": self.state.artifacts.get("project_name", self.output_dir),
                            "files": list(self.state.artifacts.get("generated_files", {}).keys()) if isinstance(self.state.artifacts.get("generated_files"), dict) else [],
                            "requirements": str(self.state.artifacts.get("requirements", "")),
                        }
                        devops_result = await self.devops_engineer.execute_task(devops_task)
                        self.state.artifacts["devops_validation"] = devops_result.output
                        # Track DevOpsEngineer experience and pool performance
                        if hasattr(self.devops_engineer, '_experience'):
                            self.devops_engineer._experience.add_experience("validate_deployment", devops_result.success)
                        self.agent_pool.record_task_outcome("DevOpsEngineer", devops_result.success)
                        self.performance_tracker.record_task(
                            "DevOpsEngineer", success=devops_result.success,
                            response_time_ms=devops_result.execution_time * 1000, tokens=0
                        )
                        # Reflect on DevOps validation (learning)
                        if self.enable_learning:
                            self.devops_engineer.reflect(
                                "validate_deployment",
                                str(devops_result.output),
                                devops_result.success
                            )
                        devops_verdict = str(devops_result.output).upper() if isinstance(devops_result.output, str) else str(devops_result.output.get("verdict", "")).upper() if isinstance(devops_result.output, dict) else ""
                        if "NOT_DEPLOYABLE" in devops_verdict:
                            console.warning("DevOps: NOT_DEPLOYABLE — Developer must fix before CEO review")
                            # DevOps notifies Developer of deployment issues via message bus
                            self.message_bus.notify_issue(
                                "DevOpsEngineer", "Developer",
                                f"NOT_DEPLOYABLE: {str(devops_result.output)[:400]}",
                                "high"
                            )
                            # Send back to developer for deployment fixes
                            deploy_fix_task = {
                                "type": "fix_bug",
                                "bug_description": f"DevOps validation failed: {str(devops_result.output)}",
                                "error_message": "Fix deployment issues: missing files, unpinned deps, or broken entry point",
                                "original_requirements": str(self.state.artifacts.get("requirements", "")),
                                "output_dir": self.state.artifacts.get("project_name", self.output_dir),
                            }
                            deploy_fix = await self.developer.execute_task(deploy_fix_task)
                            self.state.artifacts["devops_fix"] = deploy_fix.output
                            dev_fix_count += 1
                            console.success("Developer applied DevOps fixes")
                            continue  # Re-run QA loop
                        elif "NEEDS_FIXES" in devops_verdict:
                            console.warning("DevOps: NEEDS_FIXES — proceeding with warnings")
                            # DevOps notifies Developer of deployment warnings via message bus
                            self.message_bus.notify_issue(
                                "DevOpsEngineer", "Developer",
                                f"NEEDS_FIXES (non-blocking): {str(devops_result.output)[:300]}",
                                "medium"
                            )
                        else:
                            console.success("DevOps: DEPLOYABLE")
                    except Exception as devops_err:
                        console.warning(f"DevOps validation error (non-blocking): {devops_err}")

                # Phase 7: CEO Final Approval
                self.progress.start_phase("ceo_approval")
                console.section(f"PHASE 7: CEO APPROVAL (Round {round_label})")
                console.info("Requesting CEO final approval...")
                if await self._run_ceo_approval():
                    self.state.phase = WorkflowPhase.COMPLETED
                    console.success("Solution approved by CEO!")
                    self.progress.complete_phase("ceo_approval")
                    if self._current_sprint_id:
                        try:
                            self.sprint_manager.complete_task(self._current_sprint_id, "approval")
                        except ValueError:
                            pass

                    # Store solution in shared memory
                    problem = self.state.current_problem
                    self.shared_memory.store_solution(
                        problem_id=problem.id,
                        solution_description=f"Solution approved: {problem.description}",
                        implementation_path=str(self.state.artifacts.get("project_name", "")),
                        created_by="developer"
                    )

                    self._save_checkpoint()

                    # Delivery phase (skip in dry-run mode)
                    if dry_run:
                        console.info("Dry-run mode: skipping delivery phase")
                        self.progress.skip_phase("delivery")
                    else:
                        self.progress.start_phase("delivery")
                        console.section("PHASE 8: DELIVERY")
                        console.info("Packaging solution for delivery...")
                        delivery = await self._run_delivery_phase()
                        console.success(f"Delivered: {len(delivery.get('files', []))} files in {delivery['project_dir']}")
                        self.progress.complete_phase("delivery")
                        self._save_checkpoint()
                    break

                self.progress.fail_phase("ceo_approval", "Rejected by CEO")

                # CEO rejected - collect feedback for escalation
                rejection = self.state.artifacts.get("final_approval", {})
                rejection_reasoning = rejection.get("reasoning", str(rejection)) if isinstance(rejection, dict) else str(rejection)
                failure_history.append(f"CEO rejection: {rejection_reasoning}")
                # CEO notifies Developer of rejection via message bus
                self.message_bus.notify_issue(
                    "CEO", "Developer",
                    f"CEO rejected (round {approval_round}): {rejection_reasoning[:400]}",
                    "high"
                )

                if self.enable_escalation:
                    action = self.escalation_manager.should_escalate(
                        round=approval_round,
                        qa_verdict="ceo_rejected",
                        dev_fix_count=dev_fix_count,
                        failure_history=failure_history
                    )
                    console.info(f"Escalation decision: {action.value}")

                    if action == EscalationAction.DEVELOPER_FIX:
                        console.warning(f"CEO rejected in round {approval_round}. Developer addressing feedback...")
                        project_dir = self.state.artifacts.get("project_name", self.output_dir)
                        execution_results = self.state.artifacts.get("execution_results", {})
                        strategy_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()
                        dev_messages = self._drain_messages_for_agent("Developer")
                        ceo_fix_task = {
                            "type": "fix_bug",
                            "bug_description": f"CEO rejected the solution. Feedback: {str(rejection_reasoning)}",
                            "error_message": execution_results.get("summary", ""),
                            "qa_report": str(self.state.artifacts.get("qa_report", "")),
                            "original_requirements": str(self.state.artifacts.get("requirements", "")),
                            "output_dir": project_dir,
                            "strategy_context": f"{strategy_context}\n{dev_messages}"[:1000] if dev_messages else strategy_context
                        }
                        ceo_fix_result = await self.developer.execute_task(ceo_fix_task)
                        self.state.artifacts[f"ceo_fix_round_{approval_round}"] = ceo_fix_result.output
                        self.escalation_manager.strategy_memory.record_attempt(
                            approach=action.value, round=approval_round,
                            outcome="attempted", feedback=failure_history[-1] if failure_history else ""
                        )
                        dev_fix_count += 1
                        console.success(f"Developer applied CEO feedback fixes (round {approval_round})")
                        continue
                    elif action == EscalationAction.CTO_REDESIGN:
                        console.warning("CEO rejected. Escalating to CTO for architecture redesign...")
                        strategy_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()
                        redesign_task = {
                            "type": "design_architecture",
                            "problem": self.state.current_problem.to_dict(),
                            "requirements": str(self.state.artifacts.get("requirements", "")),
                            "constraint": f"CEO rejected after {approval_round} rounds. Feedback: {rejection_reasoning}. Redesign approach.\n{strategy_context}"
                        }
                        redesign_result = await self.cto.execute_task(redesign_task)
                        self.state.artifacts["architecture"] = redesign_result.output
                        self.escalation_manager.strategy_memory.record_attempt(
                            approach=action.value, round=approval_round,
                            outcome="attempted", feedback=failure_history[-1] if failure_history else ""
                        )
                        await self._run_implementation_phase()
                        console.success("Re-implementation after CTO redesign complete")
                        continue
                    elif action == EscalationAction.PM_RESCOPE:
                        console.warning("CEO rejected. Escalating to PM for scope reduction...")
                        strategy_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()
                        rescope_task = {
                            "type": "define_requirements",
                            "problem": self.state.current_problem.to_dict(),
                            "target_users": self.state.current_problem.target_users,
                            "constraint": f"CEO rejected after {approval_round} rounds. Reduce scope dramatically.\n{strategy_context}"
                        }
                        rescope_result = await self.product_manager.execute_task(rescope_task)
                        self.state.artifacts["requirements"] = rescope_result.output
                        self.escalation_manager.strategy_memory.record_attempt(
                            approach=action.value, round=approval_round,
                            outcome="attempted", feedback=failure_history[-1] if failure_history else ""
                        )
                        await self._run_technical_design_phase()
                        await self._run_implementation_phase()
                        console.success("Re-implementation with reduced scope complete")
                        continue
                    elif action == EscalationAction.TEAM_BRAINSTORM:
                        console.warning("CEO rejected. Escalating to team brainstorm...")
                        brainstorm_result = await self._team_brainstorm()
                        if brainstorm_result:
                            self.state.artifacts["brainstorm_solutions"] = brainstorm_result
                            await self._run_implementation_phase()
                            console.success("Re-implementation after brainstorm complete")
                        continue
                    elif action == EscalationAction.DECOMPOSE_PROBLEM:
                        console.warning("CEO rejected. Decomposing problem into sub-problems...")
                        sub_problems = await self._decompose_problem()
                        if sub_problems:
                            self.state.artifacts["sub_problems"] = sub_problems
                            await self._run_implementation_phase()
                            console.success("Re-implementation after decomposition complete")
                        continue
                    elif action == EscalationAction.PIVOT_APPROACH:
                        console.warning("CEO rejected. Pivoting to a different approach...")
                        await self._pivot_approach()
                        await self._run_implementation_phase()
                        console.success("Re-implementation after pivot complete")
                        continue
                    elif action == EscalationAction.SIMPLIFY:
                        console.warning("CEO rejected. Simplifying to absolute minimum...")
                        strategy_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()
                        simplify_task = {
                            "type": "define_requirements",
                            "problem": self.state.current_problem.to_dict(),
                            "target_users": self.state.current_problem.target_users,
                            "constraint": f"CEO rejected. SIMPLIFY: 1 file, 1 function, core feature ONLY.\n{strategy_context}"
                        }
                        simplify_result = await self.product_manager.execute_task(simplify_task)
                        self.state.artifacts["requirements"] = simplify_result.output
                        self.escalation_manager.strategy_memory.record_attempt(
                            approach=action.value, round=approval_round,
                            outcome="attempted", feedback=failure_history[-1] if failure_history else ""
                        )
                        await self._run_technical_design_phase()
                        await self._run_implementation_phase()
                        console.success("Re-implementation with simplified scope complete")
                        continue
                    elif action == EscalationAction.ALTERNATIVE_STACK:
                        console.warning("CEO rejected. Trying alternative technology stack...")
                        strategy_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()
                        alt_stack_task = {
                            "type": "design_architecture",
                            "problem": self.state.current_problem.to_dict(),
                            "requirements": str(self.state.artifacts.get("requirements", "")),
                            "constraint": f"CEO rejected. ALTERNATIVE STACK: Try a completely different approach.\n{strategy_context}"
                        }
                        alt_stack_result = await self.cto.execute_task(alt_stack_task)
                        self.state.artifacts["architecture"] = alt_stack_result.output
                        self.escalation_manager.strategy_memory.record_attempt(
                            approach=action.value, round=approval_round,
                            outcome="attempted", feedback=failure_history[-1] if failure_history else ""
                        )
                        await self._run_implementation_phase()
                        console.success("Re-implementation with alternative stack complete")
                        continue
                    elif action == EscalationAction.FRESH_START:
                        console.warning("CEO rejected. FRESH START: Re-analyzing from scratch...")
                        self.escalation_manager.strategy_memory.record_attempt(
                            approach=action.value, round=approval_round,
                            outcome="attempted", feedback="Fresh start triggered after CEO rejection"
                        )
                        await self._run_analysis_phase()
                        await self._run_technical_design_phase()
                        await self._run_implementation_phase()
                        console.success("Re-implementation after fresh start complete")
                        continue
                else:
                    # Escalation disabled: use legacy behavior
                    has_limit = max_approval_rounds > 0
                    at_limit = has_limit and approval_round >= max_approval_rounds
                    if not at_limit:
                        console.warning(f"CEO rejected in round {approval_round}. Sending feedback to Developer...")
                        project_dir = self.state.artifacts.get("project_name", self.output_dir)
                        execution_results = self.state.artifacts.get("execution_results", {})
                        ceo_fix_task = {
                            "type": "fix_bug",
                            "bug_description": f"CEO rejected the solution. Feedback: {str(rejection_reasoning)}",
                            "error_message": execution_results.get("summary", ""),
                            "qa_report": str(self.state.artifacts.get("qa_report", "")),
                            "original_requirements": str(self.state.artifacts.get("requirements", "")),
                            "output_dir": project_dir
                        }
                        ceo_fix_result = await self.developer.execute_task(ceo_fix_task)
                        self.state.artifacts[f"ceo_fix_round_{approval_round}"] = ceo_fix_result.output
                        dev_fix_count += 1
                        console.success(f"Developer applied CEO feedback fixes (round {approval_round})")
                    else:
                        if getattr(self, 'force_stop', False):
                            console.warning(f"CEO rejected after {approval_round} rounds. Force-stop enabled — delivering best available solution.")
                            self.state.phase = WorkflowPhase.COMPLETED
                            self.state.artifacts["partial_delivery"] = True
                            self.state.artifacts["delivery_note"] = f"Solution delivered after {approval_round} CEO rejections. Last feedback: {rejection_reasoning}"
                            break
                        else:
                            # Never stop: rework and continue
                            console.warning(f"CEO rejected in round {approval_round} (at previous limit). Reworking — never giving up...")
                            project_dir = self.state.artifacts.get("project_name", self.output_dir)
                            execution_results = self.state.artifacts.get("execution_results", {})
                            ceo_fix_task = {
                                "type": "fix_bug",
                                "bug_description": f"CEO rejected the solution. Feedback: {str(rejection_reasoning)}",
                                "error_message": execution_results.get("summary", ""),
                                "qa_report": str(self.state.artifacts.get("qa_report", "")),
                                "original_requirements": str(self.state.artifacts.get("requirements", "")),
                                "output_dir": project_dir
                            }
                            ceo_fix_result = await self.developer.execute_task(ceo_fix_task)
                            self.state.artifacts[f"ceo_fix_round_{approval_round}"] = ceo_fix_result.output
                            dev_fix_count += 1
                            console.success(f"Developer applied CEO feedback fixes (round {approval_round})")
                            continue

            self.state.completed_at = datetime.now()

            # Phase 9: Retrospective (runs after delivery or failure)
            if self.enable_retrospective:
                self.progress.start_phase("retrospective")
                console.section("PHASE 9: RETROSPECTIVE")
                console.info("All agents reflecting on the workflow run...")
                await self._run_retrospective_phase()
                self.progress.complete_phase("retrospective")
            else:
                self.progress.skip_phase("retrospective")

            if self.state.phase == WorkflowPhase.COMPLETED:
                self._clear_checkpoint()

        except KeyboardInterrupt:
            self.state.phase = WorkflowPhase.FAILED
            self.state.error = "Interrupted by user (Ctrl+C)"
            self.state.completed_at = datetime.now()
            console.warning("\nWorkflow interrupted by user (Ctrl+C)")
            self._save_checkpoint()

        except Exception as e:
            import traceback
            self.state.error = str(e)
            self.state.completed_at = datetime.now()
            console.error(f"Workflow error: {str(e)}")
            console.error(traceback.format_exc())
            self._save_checkpoint()

            # Part 2c: Persistence loop — retry from research on unexpected errors
            pivot_count = getattr(self, '_pivot_count', 0)
            max_pivots = 3
            if pivot_count < max_pivots and self.state.phase != WorkflowPhase.COMPLETED:
                self._pivot_count = pivot_count + 1
                console.warning(f"Workflow error on attempt {pivot_count + 1}/{max_pivots} — resetting to research phase")
                self.state.phase = WorkflowPhase.RESEARCH
                self.state.error = None
                self.state.completed_at = None
                # Recursive retry — re-run the entire workflow
                try:
                    return await self.run_full_workflow(
                        auto_discover=True,
                        problem=None,
                        max_approval_rounds=max_approval_rounds,
                        resume=False,
                        interactive=interactive,
                        dry_run=dry_run,
                    )
                except Exception as retry_err:
                    console.error(f"Retry also failed: {retry_err}")

            self.state.phase = WorkflowPhase.FAILED
            # Trigger error hook
            try:
                from utils.hooks import HookContext, HookEvent
                await self.hooks_manager.execute(HookContext(
                    event=HookEvent.ERROR,
                    metadata={"error": str(e)}
                ))
            except Exception as hook_err:
                console.warning(f"Error hook failed: {hook_err}")

        finally:
            # Trigger session end hook
            try:
                session_id = self.session_manager.current_session.id if self.session_manager.current_session else ""
                await self.hooks_manager.session_end(session_id)
            except Exception as hook_err:
                console.warning(f"Session end hook failed: {hook_err}")

            # Cleanup: always stop background services and persist results
            self.health_checker.stop_background_checks()
            self.memory_monitor.stop_monitoring()

            # End session with appropriate state
            if self.state.phase == WorkflowPhase.COMPLETED:
                self.session_manager.end_session(SessionState.COMPLETED)
            elif self.state.phase == WorkflowPhase.FAILED:
                self.session_manager.end_session(SessionState.FAILED)
            else:
                self.session_manager.end_session(SessionState.PAUSED)

            self.structured_logger.info(f"Workflow ended with status: {self.state.phase.value}")
            self.state.artifacts["task_stats"] = self.task_manager.get_task_stats()

            cost_session = self.cost_tracker.end_session()
            if cost_session:
                self.state.artifacts["cost_summary"] = cost_session.to_dict()

            result = self._get_workflow_result()
            try:
                self._persist_result(result)
            except Exception as persist_err:
                console.warning(f"Could not persist workflow result: {persist_err}")

            if self.state.phase == WorkflowPhase.COMPLETED:
                self._print_completion_summary(result)

            return result

    # ============================================================
    #  QUALITY GATES
    # ============================================================




    # ============================================================
    #  PHASE 1: RESEARCH
    # ============================================================


    # ============================================================
    #  PHASE 3: ANALYSIS (PM + RESEARCHER)
    # ============================================================



    # ============================================================
    #  PHASE 4: OPPORTUNITY EVALUATION (CEO)
    # ============================================================


    # ============================================================
    #  PHASE 5: TECHNICAL DESIGN (CTO)
    # ============================================================


    # ============================================================
    #  PHASE 7: IMPLEMENTATION (DEVELOPER)
    # ============================================================


    # ============================================================
    #  PHASE 9-10: QA + SECURITY VALIDATION + CEO APPROVAL LOOP
    # ============================================================



    # ============================================================
    #  ESCALATION STRATEGIES
    # ============================================================




    # ============================================================
    #  HELPER METHODS
    # ============================================================

    # Token budget allocation per agent (fraction of total budget)
    _TOKEN_BUDGETS = {
        "CEO": 0.10, "CTO": 0.15, "ProductManager": 0.10,
        "Researcher": 0.15, "Developer": 0.25, "QAEngineer": 0.10,
        "DevOpsEngineer": 0.05, "DataAnalyst": 0.05, "SecurityEngineer": 0.05,
    }






    async def _run_ceo_morning_brief(self, problem) -> None:
        """CEO sets 2-3 priorities and constraints for the run.

        The brief is injected into all agent system prompts so the team
        operates under shared constraints throughout the workflow.
        """
        prob_desc = problem.description if problem else "auto-discover a problem"
        brief_prompt = (
            f"We are about to work on: {prob_desc}\n\n"
            "As CEO, write a 3-bullet morning brief for the team. Each bullet is one constraint or priority:\n"
            "- Bullet 1: primary user/outcome focus (what matters most to the end user)\n"
            "- Bullet 2: technical constraint (e.g., prefer stdlib, single file, no external services)\n"
            "- Bullet 3: quality bar (e.g., must work on first run, zero external dependencies)\n\n"
            "Keep each bullet ≤ 15 words. No fluff."
        )
        try:
            compute = self.ceo._get_compute_config(brief_prompt)
            if compute["consistency_samples"] >= 5:  # complex
                brief = await self.ceo.generate_with_tot(
                    brief_prompt,
                    evaluation_criteria="strategic clarity, risk identification, and actionability"
                )
            else:
                brief = await self.ceo.generate_response_async(brief_prompt, use_first_principles=False)
            brief = console._clean_llm_output(brief)
            self.state.artifacts["ceo_morning_brief"] = brief
            console.info(f"CEO Morning Brief:\n{brief}")
            # Inject brief into all agent system prompts for this run
            brief_injection = f"\n\n[CEO MORNING BRIEF]\n{brief}\n"
            for agent in self.agents.values():
                if brief_injection not in agent.system_prompt:
                    agent.system_prompt += brief_injection
        except Exception as e:
            console.warning(f"CEO morning brief skipped: {e}")

    async def _run_morning_standup(self, problem_name: str) -> None:
        """Create sprint and run standup meeting for team alignment."""
        from company.sprint import SprintTask

        sprint_tasks = [
            SprintTask(id="research", name="Research & Discovery", assigned_to="Researcher"),
            SprintTask(id="design", name="Technical Design", assigned_to="CTO"),
            SprintTask(id="implement", name="Implementation", assigned_to="Developer"),
            SprintTask(id="qa", name="QA Validation", assigned_to="QAEngineer"),
            SprintTask(id="approval", name="CEO Approval", assigned_to="CEO"),
        ]
        sprint = self.sprint_manager.create_sprint(
            name=f"Sprint: {problem_name}",
            goal=problem_name,
            tasks=sprint_tasks,
        )
        self.sprint_manager.start_sprint(sprint.id)
        self._current_sprint_id = sprint.id

        if not self.enable_meetings:
            return

        standup_context = f"Sprint goal: {problem_name}\n"
        for agent in self.agents.values():
            if hasattr(agent, '_retrospective_lessons') and agent._retrospective_lessons:
                standup_context += f"Lessons from last run:\n{agent._retrospective_lessons}\n"
                break

        standup_participants = [
            {"name": "CEO", "role": "Leadership", "model": self.model_config.get_model_name("ceo")},
            {"name": "CTO", "role": "Technical Lead", "model": self.model_config.get_model_name("cto")},
            {"name": "Developer", "role": "Implementation", "model": self.model_config.get_model_name("developer")},
        ]
        console.section("MORNING STANDUP")
        standup_outcome = await quick_meeting(
            topic=f"Sprint Kickoff: {problem_name}",
            participants=standup_participants,
            meeting_type=MeetingType.STANDUP,
            context=standup_context,
        )
        self.state.artifacts["standup"] = {
            "action_items": standup_outcome.action_items,
            "insights": standup_outcome.key_insights,
        }
        # Record structured standup minutes via StandupManager
        standup_reports = {}
        for i, insight in enumerate(standup_outcome.key_insights or []):
            participant = standup_participants[i % len(standup_participants)]
            standup_reports[participant["name"]] = insight
        standup_minutes = self.standup_manager.record_standup(
            attendees=[p["name"] for p in standup_participants],
            phase="sprint_kickoff",
            reports=standup_reports,
        )
        if standup_minutes.blockers:
            console.warning(f"Standup blockers identified: {len(standup_minutes.blockers)}")
            for blocker in standup_minutes.blockers:
                console.warning(f"  BLOCKER: {blocker[:200]}")


    async def _run_performance_reviews(self) -> None:
        """Run 1:1 meetings for underperforming agents."""
        if not self.enable_meetings:
            return
        underperformers = self.performance_tracker.get_underperformers(threshold=40.0)
        if not underperformers:
            return

        for agent_name in underperformers[:2]:  # Max 2 reviews per run
            supervisor_name = self.org_chart.get_supervisor(agent_name)
            if not supervisor_name or supervisor_name not in self.agents:
                continue

            agent = self.agents[agent_name]
            supervisor = self.agents[supervisor_name]
            kpis = self.performance_tracker.get_kpis(agent_name)

            context = (
                f"Agent: {agent_name}\n"
                f"Performance Score: {kpis.calculate_score():.0f}/100\n"
                f"Approval Rate: {kpis.approval_rate:.0%}\n"
                f"Rework Rate: {kpis.rework_rate:.0%}\n"
            )

            console.info(f"1:1 Review: {supervisor_name} -> {agent_name} (score: {kpis.calculate_score():.0f})")
            review_outcome = await quick_meeting(
                topic=f"Performance Review: {agent_name}",
                participants=[
                    {"name": supervisor_name, "role": "Supervisor", "model": supervisor.model},
                    {"name": agent_name, "role": "Report", "model": agent.model},
                ],
                meeting_type=MeetingType.ONE_ON_ONE,
                context=context,
            )
            if review_outcome.action_items:
                from company.organization import AgentManager
                reconfig = AgentManager.recommend_reconfiguration(agent_name, kpis.calculate_score())
                agent._supervision_note = (
                    f"Feedback from {supervisor_name}: "
                    + "; ".join(review_outcome.action_items[:3])
                    + f"\nSystem recommendation: {reconfig}"
                )

    # ============================================================
    #  DELIVERY + RETROSPECTIVE
    # ============================================================


    # ============================================================
    #  PHASE 2: DATA ANALYSIS
    # ============================================================


    # ============================================================
    #  PHASE 6: DESIGN REVIEW
    # ============================================================




    def _save_checkpoint(self) -> None:
        """Save workflow state checkpoint after each phase."""
        checkpoint_dir = Path(self.data_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "phase": self.state.phase.value,
            "problem": self.state.current_problem.to_dict() if self.state.current_problem else None,
            "artifacts": {},
            "decisions": self.state.decisions,
            "started_at": self.state.started_at.isoformat() if self.state.started_at else None,
            "error": self.state.error,
        }

        # Filter artifacts to only JSON-serializable values
        for key, value in self.state.artifacts.items():
            try:
                json.dumps(value)
                checkpoint["artifacts"][key] = value
            except (TypeError, ValueError):
                checkpoint["artifacts"][key] = str(value)

        checkpoint_file = checkpoint_dir / "latest.json"
        try:
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint, f, indent=2)
            # Also save per-session checkpoint so older checkpoints aren't lost
            if self.session_manager.current_session:
                run_file = checkpoint_dir / f"run_{self.session_manager.current_session.id}.json"
                with open(run_file, "w") as f:
                    json.dump(checkpoint, f, indent=2)
        except OSError as e:
            console.warning(f"Could not save checkpoint: {e}")

        # Also save session state with a checkpoint marker
        if self.session_manager.current_session:
            session = self.session_manager.current_session
            session.workflow_state = checkpoint
            session.create_checkpoint(
                name=self.state.phase.value,
                metadata={"phase": self.state.phase.value}
            )
            self.session_manager.save_current()

    def _load_checkpoint(self) -> bool:
        """Load workflow from last checkpoint. Returns True if loaded."""
        checkpoint_file = Path(self.data_dir) / "checkpoints" / "latest.json"
        if not checkpoint_file.exists():
            return False

        try:
            with open(checkpoint_file) as f:
                checkpoint = json.load(f)

            self.state.phase = WorkflowPhase(checkpoint["phase"])
            self.state.artifacts = checkpoint.get("artifacts", {})
            self.state.decisions = checkpoint.get("decisions", [])

            if checkpoint.get("started_at"):
                self.state.started_at = datetime.fromisoformat(checkpoint["started_at"])

            if checkpoint.get("problem"):
                prob_data = checkpoint["problem"]
                self.state.current_problem = DiscoveredProblem(
                    id=prob_data["id"],
                    description=prob_data["description"],
                    severity=ProblemSeverity(prob_data["severity"]),
                    frequency=ProblemFrequency(prob_data.get("frequency", "common")),
                    target_users=prob_data.get("target_users", ""),
                    evidence=prob_data.get("evidence", []),
                    sources=prob_data.get("sources", []),
                    domain=prob_data.get("domain", "software"),
                    keywords=prob_data.get("keywords", []),
                    potential_solution_ideas=prob_data.get("potential_solution_ideas", []),
                    validation_status=prob_data.get("validation_status", "unvalidated"),
                    score=prob_data.get("score", 0.0),
                    metadata=prob_data.get("metadata", {}),
                )

            self.state.error = checkpoint.get("error")
            return True
        except Exception as e:
            console.warning(f"Could not load checkpoint: {e}")
            return False

    def _clear_checkpoint(self) -> None:
        """Clear checkpoint after successful completion."""
        checkpoint_file = Path(self.data_dir) / "checkpoints" / "latest.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()

    # ============================================================
    #  RESULT PERSISTENCE
    # ============================================================

    def _persist_result(self, result: Dict[str, Any]) -> None:
        """Persist workflow result to history database (JSON file)."""
        results_file = Path(self.data_dir) / "results" / "workflow_history.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing results
        history = []
        if results_file.exists():
            try:
                with open(results_file) as f:
                    history = json.load(f)
            except Exception as e:
                console.warning(f"Could not load workflow history: {e}")

        # Append new result (with serializable data only)
        entry = {
            "id": len(history) + 1,
            "status": result.get("status"),
            "problem": result.get("problem", {}).get("description", "") if result.get("problem") else "",
            "domain": result.get("problem", {}).get("domain", "") if result.get("problem") else "",
            "language": result.get("artifacts", {}).get("language", "python"),
            "started_at": result.get("started_at"),
            "completed_at": result.get("completed_at"),
            "duration_seconds": result.get("duration_seconds"),
            "decisions": result.get("decisions", []),
            "delivery_dir": result.get("artifacts", {}).get("delivery", {}).get("project_dir", ""),
        }
        history.append(entry)

        try:
            with open(results_file, "w") as f:
                json.dump(history, f, indent=2)
        except OSError as e:
            console.warning(f"Could not write workflow history: {e}")

    def _get_workflow_result(self) -> Dict[str, Any]:
        """Get the complete workflow result."""
        duration = None
        if self.state.started_at and self.state.completed_at:
            duration = (self.state.completed_at - self.state.started_at).total_seconds()

        # Collect KPIs for all agents
        all_kpis = {}
        for agent_name in self.agents:
            kpis = self.performance_tracker.get_kpis(agent_name)
            if kpis.tasks_completed + kpis.tasks_failed > 0:
                all_kpis[agent_name] = {
                    "score": kpis.calculate_score(),
                    "tasks_completed": kpis.tasks_completed,
                    "tasks_failed": kpis.tasks_failed,
                    "approval_rate": round(kpis.approval_rate, 2),
                    "rework_rate": round(kpis.rework_rate, 2),
                }

        # Persist agent experience to disk
        for agent_name, agent in self.agents.items():
            if hasattr(agent, '_experience'):
                exp_path = self._experience_dir / f"{agent_name}.json"
                agent._experience.save(exp_path)

        # Persist performance, trust, and strategy trackers to disk
        perf_path = Path(self.data_dir) / "performance" / "kpis.json"
        self.performance_tracker.save(perf_path)
        trust_path = Path(self.data_dir) / "trust" / "trust_scores.json"
        self.trust_tracker.save(trust_path)
        strat_path = Path(self.data_dir) / "escalation" / "strategy_memory.json"
        self.escalation_manager.save_strategy_memory(strat_path)

        # Log KPIs
        if all_kpis:
            console.info("Agent KPI Summary:")
            for name, kpi_data in all_kpis.items():
                console.info(f"  {name}: score={kpi_data['score']:.1f}, "
                             f"approval={kpi_data['approval_rate']:.0%}")

        return {
            "status": self.state.phase.value,
            "problem": self.state.current_problem.to_dict() if self.state.current_problem else None,
            "decisions": self.state.decisions,
            "artifacts": self.state.artifacts,
            "error": self.state.error,
            "started_at": self.state.started_at.isoformat() if self.state.started_at else None,
            "completed_at": self.state.completed_at.isoformat() if self.state.completed_at else None,
            "duration_seconds": duration,
            "agent_kpis": all_kpis,
            "trust_scores": self.trust_tracker.get_all_scores(),
        }

    # ============================================================
    #  INTERACTIVE APPROVAL
    # ============================================================


    # ============================================================
    #  COMPLETION SUMMARY
    # ============================================================

    def _print_scaffold_summary(self, result: Dict[str, Any]) -> None:
        """Print the scaffold delivery summary with TODO counts and next steps."""
        delivery = self.state.artifacts.get("delivery", {})
        project_dir = delivery.get("project_dir", "")
        files = delivery.get("files", [])

        # Count # TODO occurrences in output directory
        todo_count = 0
        if project_dir and Path(project_dir).exists():
            for p in Path(project_dir).rglob("*"):
                if p.is_file() and p.suffix in (".py", ".js", ".ts", ".md"):
                    try:
                        todo_count += p.read_text(encoding="utf-8", errors="ignore").count("# TODO")
                    except Exception:
                        pass

        border = "\u2501" * 40
        console.info(border)
        console.info(" SCAFFOLD GENERATED \u2014 HUMAN REVIEW NEEDED")
        console.info(border)
        console.info(f"Files generated: {len(files)}")
        console.info(f"TODOs to complete: {todo_count}")
        console.info("Next steps:")
        console.info(f"  1. Review generated files in {project_dir or '(output dir)'}")
        console.info("  2. Complete all # TODO sections")
        console.info("  3. Run tests and fix failures")
        console.info("  4. Remove draft headers when production-ready")
        console.info(border)

    def _print_completion_summary(self, result: Dict[str, Any]) -> None:
        """Print a clear summary after successful workflow completion."""
        delivery = self.state.artifacts.get("delivery", {})
        project_dir = delivery.get("project_dir", "")
        files = delivery.get("files", [])
        language = delivery.get("language", "python")
        problem_desc = ""
        if self.state.current_problem:
            problem_desc = self.state.current_problem.description

        # If scaffold mode, print the specialised scaffold summary instead
        if self.scaffold_mode:
            self._print_scaffold_summary(result)
            return

        if not project_dir or not files:
            console.section("WORKFLOW COMPLETE")
            console.warning("Delivery artifacts missing - solution may be incomplete.")
            console.info(f"  Status: {result.get('status', 'unknown').upper()}")
            if problem_desc:
                console.info(f"  Problem: {problem_desc}")
            return

        duration = result.get("duration_seconds")
        duration_str = f"{duration:.0f}s" if duration else "N/A"

        cost_summary = self.state.artifacts.get("cost_summary", {})
        total_tokens = cost_summary.get("total_tokens", 0) if isinstance(cost_summary, dict) else 0

        console.section("WORKFLOW COMPLETE")
        lines = [
            f"  Problem:    {problem_desc}",
            f"  Status:     {result.get('status', 'unknown').upper()}",
            f"  Duration:   {duration_str}",
            f"  Tokens:     {total_tokens:,}" if total_tokens else None,
            f"  Project:    {project_dir}",
            f"  Files:      {len(files)}",
        ]
        for line in lines:
            if line:
                console.info(line)

        # Show file tree (top-level only, grouped)
        if files:
            console.info("\n  File tree:")
            # Show up to 20 files
            for f in sorted(files)[:20]:
                console.info(f"    {f}")
            if len(files) > 20:
                console.info(f"    ... and {len(files) - 20} more files")

        # Show how to run
        console.info("\n  Quick start:")
        if language == "python":
            console.info(f"    cd {project_dir}")
            if any("requirements.txt" in f for f in files):
                console.info("    pip install -r requirements.txt")
            if any("main.py" in f for f in files):
                console.info("    python main.py")
            elif any("app.py" in f for f in files):
                console.info("    python app.py")
            if any("test" in f.lower() for f in files):
                console.info("    pytest")
        elif language in ("javascript", "typescript"):
            console.info(f"    cd {project_dir}")
            if any("package.json" in f for f in files):
                console.info("    npm install")
                console.info("    npm start")
        elif language == "go":
            console.info(f"    cd {project_dir}")
            console.info("    go run .")
        elif language == "rust":
            console.info(f"    cd {project_dir}")
            console.info("    cargo run")
        else:
            console.info(f"    cd {project_dir}")

        # Show Docker hint if Dockerfile was generated
        if any("Dockerfile" in f for f in files):
            console.info("\n  Docker:")
            console.info(f"    docker build -t solution {project_dir}")
            console.info("    docker run solution")

        console.info("")

    # Directories to skip when scanning for project source files
    _IGNORED_DIRS = {
        "__pycache__", ".pytest_cache", ".venv", "venv", "env",
        "node_modules", ".git", "dist", "build", ".mypy_cache",
        ".ruff_cache", ".tox", ".eggs",
    }
    # Suffixes that mark directories to skip (e.g., package.egg-info)
    _IGNORED_SUFFIXES = (".egg-info",)


    # ============================================================
    #  PHASE 8: CODE EXECUTION
    # ============================================================

    async def _run_code_execution(self) -> Dict[str, Any]:
        """Run generated code to catch runtime errors."""
        import os
        import subprocess


        self.state.phase = WorkflowPhase.CODE_EXECUTION
        self.structured_logger.info("Code execution phase started")

        # Track plan step (step_2 = Code Execution)
        if self.plan_manager.current_plan:
            self.plan_manager.start_step("step_2")

        project_dir = self.state.artifacts.get("project_name", self.output_dir)
        language = self.state.artifacts.get("language", "python")

        results = {
            "success": True,
            "syntax_errors": [],
            "runtime_errors": [],
            "test_output": "",
            "summary": ""
        }

        # Only support Python execution for now
        if language != "python":
            results["summary"] = f"Code execution not yet supported for {language}"
            return results

        project_path = Path(project_dir)
        if not project_path.exists():
            results["success"] = False
            results["summary"] = f"Project directory not found: {project_dir}"
            return results

        # Step 1: Find all Python files (excluding caches, venvs, etc.)
        py_files = self._filter_project_files(list(project_path.rglob("*.py")))
        if not py_files:
            results["success"] = False
            results["summary"] = "No Python files found in project"
            return results

        console.info(f"Found {len(py_files)} Python files to check")

        # Step 2: Syntax check all files
        for py_file in py_files:
            try:
                result = subprocess.run(
                    ["python3", "-m", "py_compile", str(py_file)],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode != 0:
                    error = result.stderr.strip() or result.stdout.strip()
                    results["syntax_errors"].append(f"{py_file.name}: {error}")
                    console.error(f"Syntax error in {py_file.name}")
            except subprocess.TimeoutExpired:
                console.warning(f"Syntax check timed out for {py_file.name}")

        if results["syntax_errors"]:
            results["success"] = False
            results["summary"] = f"Syntax errors in {len(results['syntax_errors'])} files"
            return results

        console.success("All files pass syntax check")

        # Step 2b: Install dependencies if requirements.txt exists
        req_file = project_path / "requirements.txt"
        if req_file.exists():
            console.info("Installing dependencies from requirements.txt...")
            try:
                result = subprocess.run(
                    ["python3", "-m", "pip", "install", "-q",
                     "--target", str(project_path / ".deps"),
                     "-r", str(req_file)],
                    capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0:
                    console.success("Dependencies installed")
                else:
                    console.warning(f"Some dependencies failed to install: {result.stderr}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                console.warning("Could not install dependencies (pip unavailable or timed out)")

        # Build PYTHONPATH so import checks find installed deps
        extra_paths = []
        deps_dir = project_path / ".deps"
        if deps_dir.exists():
            extra_paths.append(str(deps_dir))
        env = dict(os.environ)
        if extra_paths:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = ":".join(extra_paths + ([existing] if existing else []))

        # Step 3: Try running entry point to catch import/runtime errors
        entry_points = ["main.py", "app.py", "cli.py", "__main__.py"]
        entry_file = None
        for ep in entry_points:
            # Prefer top-level file, fall back to nested
            top_level = project_path / ep
            if top_level.exists():
                entry_file = top_level
                break
            nested = self._filter_project_files(list(project_path.rglob(ep)))
            if nested:
                entry_file = nested[0]
                break

        if entry_file:
            console.info(f"Testing entry point: {entry_file.name} (sandboxed)")
            try:
                from utils.sandbox import (SandboxExecutor,
                                           create_development_sandbox)
                sandbox_config = create_development_sandbox(str(project_path))
                sandbox_config.timeout = 15
                sandbox = SandboxExecutor(sandbox_config)

                # Build PYTHONPATH for sandbox env
                sandbox_cmd = (
                    f'PYTHONPATH="{":".join(extra_paths)}" '
                    if extra_paths else ""
                )
                sandbox_cmd += (
                    f'python3 -c "'
                    f"import sys, importlib.util; "
                    f"spec = importlib.util.spec_from_file_location('mod', '{entry_file}'); "
                    f"mod = importlib.util.module_from_spec(spec); "
                    f'spec.loader.exec_module(mod)"'
                )

                sandbox_result = await sandbox.execute(sandbox_cmd)
                sandbox.cleanup()

                if sandbox_result.blocked:
                    error = f"Sandbox blocked: {sandbox_result.block_reason}"
                    results["runtime_errors"].append(error)
                    results["success"] = False
                    console.warning(f"Sandbox blocked execution of {entry_file.name}: {sandbox_result.block_reason}")
                elif not sandbox_result.success:
                    error = sandbox_result.stderr.strip()
                    results["runtime_errors"].append(error)
                    results["success"] = False
                    console.error(f"Runtime error in {entry_file.name}")
                else:
                    console.success(f"Entry point {entry_file.name} loads without errors")
            except Exception as sandbox_err:
                # Fallback to direct execution if sandbox fails
                console.warning(f"Sandbox unavailable ({sandbox_err}), falling back to direct execution")
                try:
                    result = subprocess.run(
                        ["python3", "-c",
                         "import sys, importlib.util; "
                         "spec = importlib.util.spec_from_file_location('mod', sys.argv[1]); "
                         "mod = importlib.util.module_from_spec(spec); "
                         "spec.loader.exec_module(mod)",
                         str(entry_file)],
                        capture_output=True, text=True, timeout=15,
                        cwd=str(project_path), env=env
                    )
                    if result.returncode != 0:
                        error = result.stderr.strip()
                        results["runtime_errors"].append(error)
                        results["success"] = False
                        console.error(f"Runtime error in {entry_file.name}")
                    else:
                        console.success(f"Entry point {entry_file.name} loads without errors")
                except subprocess.TimeoutExpired:
                    results["runtime_errors"].append(f"{entry_file.name}: Timed out (15s)")
                    results["success"] = False

        # Step 4: Run tests using the multi-language TestRunner
        from tools.test_runner import TestRunner
        test_runner = TestRunner(workspace_root=str(project_path), timeout=60)
        detected_framework = test_runner.detect_framework(str(project_path))
        if detected_framework:
            console.info(f"Running tests with {detected_framework.value}...")
            test_result = test_runner.run_tests(str(project_path), framework=detected_framework)
            results["test_output"] = test_result.output + test_result.error_output
            results["test_details"] = {
                "passed": test_result.passed,
                "failed": test_result.failed,
                "skipped": test_result.skipped,
                "errors": test_result.errors,
                "total": test_result.total,
                "framework": test_result.framework,
            }
            if not test_result.success:
                results["success"] = False
                console.warning(f"Tests failed: {test_result.failed} failed, {test_result.errors} errors")
                if results["test_output"]:
                    console.warning(f"Output:\n{results['test_output']}")
            else:
                console.success(f"All tests passed ({test_result.passed} passed)")
        else:
            # Fallback: check for test files manually (no framework detected)
            test_files = [f for f in py_files
                          if f.name.startswith("test_") or f.name.endswith("_test.py")]
            if test_files:
                console.info(f"Running {len(test_files)} test files with pytest...")
                try:
                    result = subprocess.run(
                        ["python3", "-m", "pytest", "-x", "--tb=short", "-q",
                         str(project_path)],
                        capture_output=True, text=True, timeout=60,
                        cwd=str(project_path), env=env
                    )
                    results["test_output"] = result.stdout + result.stderr
                    if result.returncode != 0:
                        results["success"] = False
                        console.warning(f"Tests failed:\n{results['test_output']}")
                    else:
                        console.success("All tests passed")
                except subprocess.TimeoutExpired:
                    results["test_output"] = "pytest timed out after 60s"
                    results["success"] = False
                except FileNotFoundError:
                    results["test_output"] = "pytest not installed"

        # Build summary
        has_tests = bool(results.get("test_output") or results.get("test_details"))
        if results["success"]:
            results["summary"] = "All checks passed: syntax OK, imports OK" + (
                ", tests passed" if has_tests else "")
        else:
            parts = []
            if results["syntax_errors"]:
                parts.append(f"Syntax errors: {'; '.join(results['syntax_errors'])}")
            if results["runtime_errors"]:
                parts.append(f"Runtime errors: {'; '.join(results['runtime_errors'])}")
            if results["test_output"]:
                parts.append(f"Test failures: {results['test_output']}")
            results["summary"] = " | ".join(parts)

        # Code quality metrics (static, no LLM call)
        try:
            todo_count = 0
            long_funcs = 0
            hardcoded_count = 0
            total_lines = 0
            import re as _re
            _hardcoded_pat = _re.compile(r'(?<![#\'""])["\'][/~][^"\']{5,}["\']|(?<!\w)(\d{1,3}\.){3}\d{1,3}(?!\w)')
            for py_file in py_files:
                try:
                    text = py_file.read_text(errors="replace")
                    lines = text.splitlines()
                    total_lines += len(lines)
                    todo_count += sum(1 for l in lines if "TODO" in l or "FIXME" in l or "STUB" in l)
                    # Count functions >30 lines (simple heuristic)
                    in_func = False
                    func_start = 0
                    for i, l in enumerate(lines):
                        if l.strip().startswith("def ") or l.strip().startswith("async def "):
                            if in_func and (i - func_start) > 30:
                                long_funcs += 1
                            in_func = True
                            func_start = i
                    if in_func and (len(lines) - func_start) > 30:
                        long_funcs += 1
                    hardcoded_count += len(_hardcoded_pat.findall(text))
                except Exception:
                    pass
            # Quality score: starts at 1.0, deducted for issues
            quality_score = 1.0
            if todo_count > 0:
                quality_score -= min(0.3, todo_count * 0.05)
            if long_funcs > 0:
                quality_score -= min(0.2, long_funcs * 0.04)
            if hardcoded_count > 3:
                quality_score -= min(0.2, (hardcoded_count - 3) * 0.04)
            quality_score = max(0.0, round(quality_score, 2))
            results["quality_score"] = quality_score
            results["quality_details"] = {
                "todo_count": todo_count,
                "long_functions": long_funcs,
                "hardcoded_values": hardcoded_count,
                "total_lines": total_lines,
            }
            q_label = "EXCELLENT" if quality_score >= 0.9 else "GOOD" if quality_score >= 0.7 else "NEEDS_WORK"
            console.info(
                f"Code quality: {q_label} ({quality_score:.0%}) — "
                f"TODOs={todo_count}, long funcs={long_funcs}, hardcoded={hardcoded_count}"
            )
        except Exception as _qe:
            console.warning(f"Code quality check failed: {_qe}")
            results["quality_score"] = 0.5

        # Update plan step (step_2 = Code Execution)
        if self.plan_manager.current_plan:
            if results["success"]:
                self.plan_manager.complete_step("step_2", output=results["summary"])
            else:
                self.plan_manager.fail_step("step_2", error=results["summary"])

        return results

    def get_agent_statuses(self) -> Dict[str, Any]:
        """Get status of all agents."""
        return {name: agent.get_status() for name, agent in self.agents.items()}

    def get_workflow_state(self) -> Dict[str, Any]:
        """Get current workflow state."""
        return {
            "phase": self.state.phase.value,
            "problem": self.state.current_problem.to_dict() if self.state.current_problem else None,
            "decisions": self.state.decisions,
            "error": self.state.error
        }

    def reset_workflow(self) -> None:
        """Reset workflow to initial state, clearing inter-run state."""
        if self.state.phase != WorkflowPhase.IDLE:
            self.workflow_history.append(self.state)
        self.state = WorkflowState()

        # Clear message bus and shared memory to prevent state pollution
        # between continuous mode runs
        self.message_bus.clear_history()
        self.message_bus.clear_all_queues()

        # Reinitialize progress tracker for the next run
        self.progress = ProgressTracker(
            metrics_file=Path(self.data_dir) / "metrics" / "phase_metrics.json"
        )
        self.progress.add_phases([
            {"id": "research", "name": "Research"},
            {"id": "data_analysis", "name": "Data Analysis"},
            {"id": "analysis", "name": "Analysis"},
            {"id": "opportunity_evaluation", "name": "Opportunity Eval"},
            {"id": "technical_design", "name": "Technical Design"},
            {"id": "design_review", "name": "Design Review"},
            {"id": "implementation", "name": "Implementation"},
            {"id": "code_execution", "name": "Code Execution"},
            {"id": "qa_validation", "name": "QA Validation"},
            {"id": "security_review", "name": "Security Review"},
            {"id": "ceo_approval", "name": "CEO Approval"},
            {"id": "delivery", "name": "Delivery"},
            {"id": "retrospective", "name": "Retrospective"},
        ])

        # Reset task manager for fresh tracking
        self.task_manager = TaskManager()

        # Clear current plan
        self.plan_manager.current_plan = None

        # Reset escalation manager for fresh tracking
        self.escalation_manager.reset()

        # Stop memory monitoring (will be restarted in next run)
        self.memory_monitor.stop_monitoring()

        # Reset structured logger context
        self.structured_logger.context = None

    async def run_continuous(
        self,
        delay_between_runs: int = 60,
        max_iterations: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Run the workflow continuously, discovering and solving problems in a loop.

        Args:
            delay_between_runs: Seconds to wait between workflow runs
            max_iterations: Maximum number of iterations (None for infinite)

        Returns:
            List of all workflow results
        """
        import asyncio as _asyncio

        results = []
        iteration = 0

        console.section("CONTINUOUS MODE")
        console.info(f"Delay between runs: {delay_between_runs}s")
        if max_iterations:
            console.info(f"Max iterations: {max_iterations}")
        else:
            console.info("Running indefinitely (Ctrl+C to stop)")

        while max_iterations is None or iteration < max_iterations:
            iteration += 1
            console.section(f"CONTINUOUS RUN #{iteration}")

            try:
                self.reset_workflow()
                result = await self.run_full_workflow(auto_discover=True)
                results.append(result)

                status = result.get("status", "unknown")
                console.info(f"Run #{iteration} completed with status: {status}")

                if max_iterations is None or iteration < max_iterations:
                    console.info(f"Waiting {delay_between_runs}s before next run...")
                    await _asyncio.sleep(delay_between_runs)

            except KeyboardInterrupt:
                console.info("Continuous mode stopped by user")
                break
            except Exception as e:
                console.error(f"Run #{iteration} failed: {e}")
                results.append({"status": "error", "error": str(e), "iteration": iteration})
                await _asyncio.sleep(delay_between_runs)

        console.section("CONTINUOUS MODE SUMMARY")
        console.info(f"Total runs: {len(results)}")
        completed = sum(1 for r in results if r.get("status") == "completed")
        console.info(f"Successful: {completed}/{len(results)}")

        return results

    async def _run_research_phase(self, *args, **kwargs):
        return await _run_research_phase(self, *args, **kwargs)

    async def _discover_from_web_search(self, *args, **kwargs):
        return await _discover_from_web_search(self, *args, **kwargs)

    async def _run_data_analysis_phase(self, *args, **kwargs):
        return await _run_data_analysis_phase(self, *args, **kwargs)

    async def _run_analysis_phase(self, *args, **kwargs):
        return await _run_analysis_phase(self, *args, **kwargs)

    async def _run_opportunity_evaluation(self, *args, **kwargs):
        return await _run_opportunity_evaluation(self, *args, **kwargs)

    async def _run_technical_design_phase(self, *args, **kwargs):
        return await _run_technical_design_phase(self, *args, **kwargs)

    async def _run_design_review_phase(self, *args, **kwargs):
        return await _run_design_review_phase(self, *args, **kwargs)

    async def _run_implementation_phase(self, *args, **kwargs):
        return await _run_implementation_phase(self, *args, **kwargs)

    async def _run_qa_phase(self, *args, **kwargs):
        return await _run_qa_phase(self, *args, **kwargs)

    async def _run_security_review_phase(self, *args, **kwargs):
        return await _run_security_review_phase(self, *args, **kwargs)

    async def _check_quality_gate(self, *args, **kwargs):
        return await _check_quality_gate(self, *args, **kwargs)

    def _evaluate_gate(self, *args, **kwargs):
        return _evaluate_gate(self, *args, **kwargs)

    async def _retry_gate(self, *args, **kwargs):
        return await _retry_gate(self, *args, **kwargs)

    async def _run_ceo_approval(self, *args, **kwargs):
        return await _run_ceo_approval(self, *args, **kwargs)

    async def _run_delivery_phase(self, *args, **kwargs):
        return await _run_delivery_phase(self, *args, **kwargs)

    async def _run_retrospective_phase(self, *args, **kwargs):
        return await _run_retrospective_phase(self, *args, **kwargs)

    async def _team_brainstorm(self, *args, **kwargs):
        return await _team_brainstorm(self, *args, **kwargs)

    async def _decompose_problem(self, *args, **kwargs):
        return await _decompose_problem(self, *args, **kwargs)

    async def _pivot_approach(self, *args, **kwargs):
        return await _pivot_approach(self, *args, **kwargs)

    def _check_token_budget(self, *args, **kwargs):
        return _check_token_budget(self, *args, **kwargs)

    def _get_time_context(self, *args, **kwargs):
        return _get_time_context(self, *args, **kwargs)

    def _detect_failure_owner(self, *args, **kwargs):
        return _detect_failure_owner(self, *args, **kwargs)

    def _adjust_agent_behavior(self, *args, **kwargs):
        return _adjust_agent_behavior(self, *args, **kwargs)

    def _drain_messages_for_agent(self, *args, **kwargs):
        return _drain_messages_for_agent(self, *args, **kwargs)

    def _load_previous_learnings(self, *args, **kwargs):
        return _load_previous_learnings(self, *args, **kwargs)

    def _filter_project_files(self, *args, **kwargs):
        return _filter_project_files(self, *args, **kwargs)

    def _interactive_gate(self, *args, **kwargs):
        return _interactive_gate(self, *args, **kwargs)
