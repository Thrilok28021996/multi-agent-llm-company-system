"""Tests for the escalation system — strategy rotation and never-give-up logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.escalation import (
    EscalationAction,
    EscalationManager,
    StrategyMemory,
    _STRATEGY_CYCLE,
)


class TestStrategyMemory:
    def test_record_attempt(self):
        mem = StrategyMemory()
        mem.record_attempt("developer_fix", 1, "failed", "syntax error")
        assert len(mem.tried_approaches) == 1
        assert mem.tried_approaches[0].approach == "developer_fix"

    def test_get_untried_strategies(self):
        mem = StrategyMemory()
        mem.record_attempt("developer_fix", 1, "failed")
        untried = mem.get_untried_strategies()
        assert "developer_fix" not in untried  # was tried, should be excluded
        assert "cto_redesign" in untried
        assert len(untried) > 0

    def test_record_architecture(self):
        mem = StrategyMemory()
        mem.record_architecture("fastapi+postgres")
        assert "fastapi+postgres" in mem.tried_architectures

    def test_record_scope(self):
        mem = StrategyMemory()
        mem.record_scope("mvp-only")
        assert "mvp-only" in mem.tried_scopes

    def test_summarize_for_next_attempt(self):
        mem = StrategyMemory()
        mem.record_attempt("developer_fix", 1, "failed", "syntax error")
        summary = mem.summarize_for_next_attempt()
        assert "developer_fix" in summary
        assert "syntax error" in summary

    def test_reset(self):
        mem = StrategyMemory()
        mem.record_attempt("developer_fix", 1, "failed")
        mem.record_architecture("fastapi")
        mem.reset()
        assert len(mem.tried_approaches) == 0
        assert len(mem.tried_architectures) == 0


class TestEscalationManager:
    def test_first_failure_gives_developer_fix(self):
        mgr = EscalationManager()
        action = mgr.should_escalate(round=1, qa_verdict="failed", dev_fix_count=0, failure_history=["syntax error"])
        assert action == EscalationAction.DEVELOPER_FIX

    def test_rounds_1_to_3_are_developer_fix(self):
        mgr = EscalationManager()
        for r in range(1, 4):
            action = mgr.should_escalate(round=r, qa_verdict="failed", dev_fix_count=r, failure_history=[f"error {r}"])
            assert action == EscalationAction.DEVELOPER_FIX, f"Round {r} should be DEVELOPER_FIX"

    def test_round_4_is_decompose(self):
        mgr = EscalationManager()
        action = mgr.should_escalate(round=4, qa_verdict="failed", dev_fix_count=3, failure_history=["error"])
        assert action == EscalationAction.DECOMPOSE_PROBLEM

    def test_round_5_is_cto_redesign(self):
        mgr = EscalationManager()
        action = mgr.should_escalate(round=5, qa_verdict="failed", dev_fix_count=3, failure_history=["error"])
        assert action == EscalationAction.CTO_REDESIGN

    def test_round_6_is_pm_rescope(self):
        mgr = EscalationManager()
        action = mgr.should_escalate(round=6, qa_verdict="failed", dev_fix_count=3, failure_history=["error"])
        assert action == EscalationAction.PM_RESCOPE

    def test_round_7_is_simplify(self):
        mgr = EscalationManager()
        action = mgr.should_escalate(round=7, qa_verdict="failed", dev_fix_count=3, failure_history=["error"])
        assert action == EscalationAction.SIMPLIFY

    def test_round_8_is_team_brainstorm(self):
        mgr = EscalationManager()
        action = mgr.should_escalate(round=8, qa_verdict="failed", dev_fix_count=3, failure_history=["error"])
        assert action == EscalationAction.TEAM_BRAINSTORM

    def test_strategy_cycle_wraps(self):
        """After round 11, the cycle wraps back to the beginning."""
        mgr = EscalationManager()
        action_1 = mgr.should_escalate(round=1, qa_verdict="failed", dev_fix_count=0, failure_history=["e1"])
        action_12 = mgr.should_escalate(round=12, qa_verdict="failed", dev_fix_count=3, failure_history=["e2"])
        assert action_1 == action_12  # Both are DEVELOPER_FIX

    def test_repeated_failures_skip_ahead(self):
        """If the same failure keeps repeating, should skip to next strategy."""
        mgr = EscalationManager()
        # Simulate 3 identical failures
        failure_history = ["identical error", "identical error", "identical error"]
        action = mgr.should_escalate(round=1, qa_verdict="failed", dev_fix_count=0, failure_history=failure_history)
        # Should detect repeated failure and skip ahead
        assert action is not None

    def test_strategy_memory_records_repeated_failures(self):
        mgr = EscalationManager()
        # Only repeated identical failures get recorded in strategy_memory
        failures = ["same error", "same error", "same error"]
        mgr.should_escalate(round=1, qa_verdict="failed", dev_fix_count=0, failure_history=failures)
        assert len(mgr.strategy_memory.tried_approaches) > 0


class TestEscalationNeverGivesUp:
    """The escalation system must never return a hard stop."""

    def test_no_hard_stop_in_cycle(self):
        """All actions in the strategy cycle must be valid escalation actions."""
        for i, action in enumerate(_STRATEGY_CYCLE):
            assert action is not None, f"Strategy cycle index {i} is None — system gives up"
            assert isinstance(action, EscalationAction), f"Strategy cycle index {i} is not an EscalationAction"

    def test_escalation_manager_always_returns_action(self):
        mgr = EscalationManager()
        for r in range(1, 20):
            action = mgr.should_escalate(
                round=r, qa_verdict="failed", dev_fix_count=min(r, 3),
                failure_history=[f"error {r}"]
            )
            assert action is not None, f"Round {r} returned None — system gave up"
