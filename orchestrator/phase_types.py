"""Shared workflow phase enum and confidence thresholds.

These live outside ``orchestrator/workflow.py`` so the extracted phase modules
in ``orchestrator/phases/`` can import them directly. ``workflow.py`` imports
the phase modules at module scope, so importing these names back from
``orchestrator.workflow`` would be a circular import.

``orchestrator.workflow`` re-exports everything defined here, so existing
``from orchestrator.workflow import WorkflowPhase, CONFIDENCE_HIGH`` imports
keep working.
"""

from enum import Enum

# Confidence band thresholds for graduated response
CONFIDENCE_CRITICAL = 0.3   # Requires immediate intervention
CONFIDENCE_LOW = 0.5        # Requires supervisor notification
CONFIDENCE_MEDIUM = 0.7     # Standard review
CONFIDENCE_HIGH = 0.85      # Light review


class WorkflowPhase(Enum):
    """Phases of the company workflow."""

    IDLE = "idle"
    RESEARCH = "research"
    DATA_ANALYSIS = "data_analysis"
    ANALYSIS = "analysis"
    OPPORTUNITY_EVALUATION = "opportunity_evaluation"
    TECHNICAL_DESIGN = "technical_design"
    DESIGN_REVIEW = "design_review"
    IMPLEMENTATION = "implementation"
    CODE_EXECUTION = "code_execution"
    QA_VALIDATION = "qa_validation"
    SECURITY_REVIEW = "security_review"
    CEO_APPROVAL = "ceo_approval"
    RETROSPECTIVE = "retrospective"
    COMPLETED = "completed"
    FAILED = "failed"
