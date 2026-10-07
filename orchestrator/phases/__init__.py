"""Phase handlers extracted from orchestrator/workflow.py.

Each module contains standalone async functions that take the workflow
instance as their first parameter (replacing the original `self`).
"""
from .research import _run_research_phase, _discover_from_web_search
from .data_analysis import _run_data_analysis_phase
from .analysis import _run_analysis_phase
from .opportunity import _run_opportunity_evaluation
from .design import _run_technical_design_phase, _run_design_review_phase
from .implementation import _run_implementation_phase
from .quality import _run_qa_phase, _run_security_review_phase, _check_quality_gate, _evaluate_gate, _retry_gate
from .approval import _run_ceo_approval
from .delivery import _run_delivery_phase
from .retrospective import _run_retrospective_phase
from .collaboration import _team_brainstorm, _decompose_problem, _pivot_approach
from .helpers import _check_token_budget, _get_time_context, _detect_failure_owner, _adjust_agent_behavior, _drain_messages_for_agent, _load_previous_learnings, _filter_project_files, _interactive_gate

__all__ = [
    "_run_research_phase",
    "_discover_from_web_search",
    "_run_data_analysis_phase",
    "_run_analysis_phase",
    "_run_opportunity_evaluation",
    "_run_technical_design_phase",
    "_run_design_review_phase",
    "_run_implementation_phase",
    "_run_qa_phase",
    "_run_security_review_phase",
    "_check_quality_gate",
    "_evaluate_gate",
    "_retry_gate",
    "_run_ceo_approval",
    "_run_delivery_phase",
    "_run_retrospective_phase",
    "_team_brainstorm",
    "_decompose_problem",
    "_pivot_approach",
    "_check_token_budget",
    "_get_time_context",
    "_detect_failure_owner",
    "_adjust_agent_behavior",
    "_drain_messages_for_agent",
    "_load_previous_learnings",
    "_filter_project_files",
    "_interactive_gate",
]
