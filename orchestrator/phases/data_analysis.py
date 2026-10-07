"""Phase: data_analysis

Extracted from orchestrator/workflow.py.
Standalone functions — first param is the workflow instance.
"""

from datetime import datetime

from orchestrator.phase_types import WorkflowPhase
from research.problem_discoverer import DiscoveredProblem, ProblemSeverity
from ui.console import console


async def _run_data_analysis_phase(workflow) -> None:
    """DataAnalyst cross-validates, deduplicates, and detects bias in research findings."""
    workflow.state.phase = WorkflowPhase.DATA_ANALYSIS
    workflow.structured_logger.info("Data analysis phase started")

    problems = workflow.state.artifacts.get("discovered_problems", [])
    if not problems:
        console.info("No discovered problems to analyze, skipping data analysis")
        return

    # Step 1: Deduplicate
    console.agent_action("DataAnalyst", "Deduplicating Findings", f"Checking {len(problems)} problems for duplicates...")
    deduped = workflow.cross_validator.deduplicate(problems)
    removed_count = len(problems) - len(deduped)
    if removed_count > 0:
        console.success(f"Deduplicated: removed {removed_count} duplicate(s), {len(deduped)} unique problems remain")
    workflow.state.artifacts["discovered_problems"] = deduped

    # Step 2: Cross-validate
    console.agent_action("DataAnalyst", "Cross-Validating", "Scoring multi-source confirmation...")
    validated = workflow.cross_validator.cross_validate(deduped)
    workflow.state.artifacts["discovered_problems"] = validated

    # Apply freshness scoring — downweight stale problems before LLM analysis
    from research.credibility import score_freshness
    freshness_penalized = 0
    for p in validated:
        disc_at = p.get("discovered_at")
        if isinstance(disc_at, str):
            try:
                disc_at = datetime.fromisoformat(disc_at)
            except (ValueError, TypeError):
                disc_at = None
        elif not isinstance(disc_at, datetime):
            disc_at = None
        freshness = score_freshness(discovered_at=disc_at)
        p["freshness_score"] = freshness
        if freshness < 0.8:
            p["score"] = round(p.get("score", 0.0) * freshness, 4)
            freshness_penalized += 1
    if freshness_penalized:
        console.info(f"Freshness scoring: {freshness_penalized} problem(s) penalized for staleness")
    else:
        console.info("Freshness scoring: all problems are recent (no penalty)")

    # Step 2b: LLM-powered cross-validation (DataAnalyst agent)
    console.agent_action("DataAnalyst", "LLM Cross-Validation", "Agent cross-validating findings with structured verdicts...")
    cross_val_task = {
        "type": "cross_validate_research",
        "findings": validated[:5],  # Top 5 for token efficiency
        "sources": list({s for p in validated for s in p.get("sources", [])}),
        "topic": workflow.state.current_problem.description if workflow.state.current_problem else "research findings",
    }
    cross_val_result = await workflow.data_analyst.execute_task(cross_val_task)
    # Track DataAnalyst experience and pool performance
    if hasattr(workflow.data_analyst, '_experience'):
        workflow.data_analyst._experience.add_experience("cross_validate_research", cross_val_result.success)
    workflow.agent_pool.record_task_outcome("DataAnalyst", cross_val_result.success)
    workflow.performance_tracker.record_task(
        "DataAnalyst", success=cross_val_result.success,
        response_time_ms=cross_val_result.execution_time * 1000, tokens=0
    )
    # Reflect on DataAnalyst cross-validation (learning)
    if workflow.enable_learning:
        workflow.data_analyst.reflect(
            "cross_validate_research",
            str(cross_val_result.output),
            cross_val_result.success
        )
    # DataAnalyst shares validation results back to Researcher via message bus
    workflow.message_bus.share_finding(
        "DataAnalyst", "Researcher",
        f"Cross-validation complete for {len(validated)} findings. See cross_validation_verdicts artifact."
    )

    # Parse structured verdicts
    from utils.output_parser import StructuredOutputParser
    verdict_parser = StructuredOutputParser()
    cross_val_text = cross_val_result.artifacts.get("cross_validation_report", "")
    parsed_verdicts = verdict_parser.parse_data_analyst_verdict(cross_val_text)
    workflow.state.artifacts["cross_validation_verdicts"] = parsed_verdicts

    summary = parsed_verdicts.get("summary", {})
    console.info(
        f"LLM cross-validation: {summary.get('confirmed', 0)} confirmed, "
        f"{summary.get('partially_confirmed', 0)} partial, "
        f"{summary.get('unconfirmed', 0)} unconfirmed, "
        f"{summary.get('contradicted', 0)} contradicted"
    )

    # Step 3: Detect bias
    console.agent_action("DataAnalyst", "Detecting Bias", "Checking for systematic research biases...")
    bias_flags = workflow.cross_validator.detect_bias(validated)
    if bias_flags:
        for flag in bias_flags:
            console.warning(f"Bias detected: {flag.bias_type.value} ({flag.severity}) - {flag.description}")
        workflow.state.artifacts["bias_flags"] = [
            {"type": bf.bias_type.value, "description": bf.description, "severity": bf.severity}
            for bf in bias_flags
        ]

    # Step 3b: Bias-reactive corrections
    if bias_flags:
        for flag in bias_flags:
            flag_data = {"type": getattr(flag.bias_type, 'value', str(flag.bias_type)),
                         "severity": flag.severity}
            if flag.severity == "high":
                if "community" in flag_data["type"].lower():
                    # Downweight single-source findings
                    for p in validated:
                        sources = p.get("sources", [])
                        if len(sources) <= 1:
                            old_score = p.get("score", 0)
                            p["score"] = old_score * 0.7
                            console.info(f"Downweighted single-source finding: {p.get('description', '')}")
                if "recency" in flag_data["type"].lower():
                    console.warning("Recency bias detected: all findings from last 7 days. Consider historical context.")
        workflow.state.artifacts["discovered_problems"] = validated

    # Step 3b+: Mandatory opposing-viewpoint search when high-severity bias detected
    high_bias_flags = [f for f in bias_flags if getattr(f, 'severity', '') == "high"]
    if high_bias_flags and validated:
        top_desc = validated[0].get("description", "")
        if top_desc:
            console.info(
                f"High-severity bias detected ({len(high_bias_flags)} flags). "
                "Running mandatory opposing-viewpoint search..."
            )
            try:
                from research.web_search import WebSearch as _WS
                _counter_searcher = _WS()
                _opposing_queries = [
                    f"why {top_desc[:70]} is not a real problem",
                    f"{top_desc[:60]} criticism overhyped",
                    f"problems with {top_desc[:60]} solution",
                ]
                _opposing_results = []
                for _q in _opposing_queries[:2]:
                    _res = await _counter_searcher.search_duckduckgo(_q, max_results=3, recency_days=90)
                    _opposing_results.extend(_res)
                if _opposing_results:
                    workflow.state.artifacts["opposing_viewpoints"] = [
                        {"title": r.title, "snippet": r.snippet, "url": r.url}
                        for r in _opposing_results[:6]
                    ]
                    console.info(
                        f"Opposing viewpoints: {len(_opposing_results)} results found to balance bias"
                    )
            except Exception as _e:
                console.warning(f"Opposing viewpoint search failed: {_e}")

    # Step 3c: Counter-evidence search for top problems
    try:
        problem_objects = []
        for p in validated[:3]:
            try:
                prob = DiscoveredProblem(
                    id=p.get("id", ""),
                    description=p.get("description", ""),
                    severity=ProblemSeverity(p.get("severity", "medium")),
                    score=p.get("score", 0.0),
                    keywords=p.get("keywords", []),
                    sources=p.get("sources", []),
                )
                problem_objects.append(prob)
            except Exception as e:
                console.warning(f"Skipping malformed problem in data analysis phase: {e}")
                continue
        if problem_objects:
            counter_evidence = await workflow.problem_discoverer.discover_counter_evidence(problem_objects, top_n=3)
            if counter_evidence:
                workflow.state.artifacts["counter_evidence"] = counter_evidence
                for ce in counter_evidence:
                    if ce.get("counter_evidence"):
                        console.info(f"Counter-evidence for '{ce['description'][:50]}': {len(ce['counter_evidence'])} items")
    except Exception as e:
        console.warning(f"Counter-evidence search failed: {e}")

    # Step 4: Use DataAnalyst agent for LLM-powered credibility scoring
    console.agent_action("DataAnalyst", "Scoring Credibility", "LLM-powered credibility assessment...")
    credibility_task = {
        "type": "score_credibility",
        "findings": validated[:5],  # Top 5 for LLM analysis (token-efficient)
        "sources": list({s for p in validated for s in p.get("sources", [])})
    }
    cred_result = await workflow.data_analyst.execute_task(credibility_task)
    workflow.state.artifacts["credibility_analysis"] = cred_result.output

    # Re-rank problems by final composite score (bias + freshness adjustments applied)
    if validated:
        validated.sort(key=lambda p: p.get("score", 0.0), reverse=True)
        workflow.state.artifacts["discovered_problems"] = validated
        top = validated[0]
        top_score = top.get("score", 0.0)
        console.info(f"Final top problem score after full analysis: {top_score:.3f}")
        # Update current_problem if re-ranking changed which problem is top
        if workflow.state.current_problem and top.get("id") != workflow.state.current_problem.id:
            console.info(f"Top problem changed after scoring: '{top.get('description', '')[:60]}...'")
            try:
                new_top = DiscoveredProblem(
                    id=top.get("id", ""),
                    description=top.get("description", ""),
                    severity=ProblemSeverity(top.get("severity", "medium")),
                    score=top_score,
                    keywords=top.get("keywords", []),
                    sources=top.get("sources", []),
                )
                workflow.state.current_problem = new_top
                console.success(f"Current problem updated to highest-scored: {new_top.description[:80]}")
            except Exception as e:
                console.warning(f"Could not update current problem after re-ranking: {e}")

    console.success(f"Data analysis complete: {len(validated)} problems, {len(bias_flags)} bias flags")


