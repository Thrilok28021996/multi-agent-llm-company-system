"""Phase: research

Extracted from orchestrator/workflow.py.
Standalone functions — first param is the workflow instance.
"""

from datetime import datetime, timedelta

from orchestrator.phase_types import WorkflowPhase
from ui.console import console


async def _run_research_phase(workflow) -> None:
    """Run the research phase to discover problems."""
    workflow.state.phase = WorkflowPhase.RESEARCH

    # Get configured subreddits
    subreddits = workflow.research_sources.get_reddit_subreddits()
    console.agent_action("Researcher", "Scanning Sources", "Reddit + HN + StackOverflow + GitHub Issues + Web Search")

    # Run research sources sequentially (concurrent calls exhaust RAM on 16GB systems)
    problems = []

    research_tasks = [
        ("Reddit", workflow.problem_discoverer.discover_from_reddit(subreddits[:1], limit_per_sub=25, time_period="month")),
        ("HackerNews", workflow.problem_discoverer.discover_from_hacker_news(limit=30)),
        ("StackOverflow", workflow.problem_discoverer.discover_from_stackoverflow(limit=20)),
        ("GitHub", workflow.problem_discoverer.discover_from_github_issues(repos=["microsoft/vscode"], limit=20)),
        ("WebSearch", workflow._discover_from_web_search()),
    ]

    for source_name, coro in research_tasks:
        try:
            result = await coro
            if result:
                problems.extend(result)
                console.info(f"{source_name}: {len(result)} problems discovered")
            else:
                console.warning(f"{source_name}: no problems found")
        except Exception as e:
            console.warning(f"{source_name}: error - {e}")

    # Filter out previously-solved problems
    from memory.shared_memory import MemoryType
    solved_memories = workflow.shared_memory.get_by_type(MemoryType.SOLUTION)
    if solved_memories and problems:
        solved_descriptions = {m.content.lower() for m in solved_memories}
        original_count = len(problems)
        filtered = []
        for p in problems:
            desc = p.description.lower() if hasattr(p, 'description') else ""
            words = set(desc.split())
            is_solved = any(
                len(words & set(sd.split())) / max(len(words), 1) > 0.6
                for sd in solved_descriptions
            ) if words else False
            if not is_solved:
                filtered.append(p)
        if filtered:
            problems = filtered
            if len(filtered) < original_count:
                console.info(f"Filtered {original_count - len(filtered)} previously-solved problems")

    # Filter problems with existing open-source solutions
    before_solution_check = len(problems)
    problems = await workflow.problem_discoverer.filter_problems_with_solutions(problems)
    if len(problems) < before_solution_check:
        console.info(f"Solution check: dropped {before_solution_check - len(problems)} problems with existing solutions")

    # Hard filter: only keep problems from the last 60 days
    max_age_days = 60
    cutoff = datetime.now() - timedelta(days=max_age_days)
    before_filter = len(problems)
    problems = [p for p in problems if p.discovered_at >= cutoff]
    if len(problems) < before_filter:
        console.info(f"Freshness filter: dropped {before_filter - len(problems)} problems older than {max_age_days} days")

    # Get top problem from the already-filtered list
    if problems:
        top_problem = max(problems, key=lambda p: p.score)
        workflow.state.current_problem = top_problem
        console.show_problem(top_problem.to_dict())

    workflow.state.artifacts["discovered_problems"] = [p.to_dict() for p in problems]
    console.success(f"Discovered {len(problems)} problems")


async def _discover_from_web_search(workflow):
    """Discover problems via DuckDuckGo web search with multi-query expansion."""
    try:
        from research.web_search import WebSearch
        web_search = WebSearch()
        domain = workflow.research_sources.get_reddit_subreddits()[0] if workflow.research_sources.get_reddit_subreddits() else "software"
        year = datetime.now().year
        base_query = f"{domain} common problems {year}"

        # Use LLM-powered term expansion for broader coverage
        try:
            search_queries = await workflow.problem_discoverer.expand_search_terms(base_query, num_alternatives=4)
        except Exception:
            search_queries = [
                base_query,
                f"developer frustrations {domain} {year}",
            ]

        all_snippets = []
        seen_titles = set()
        for query in search_queries:
            results = await web_search.search_duckduckgo(query, max_results=5)
            for r in results:
                # Deduplicate across queries
                if r.title not in seen_titles:
                    seen_titles.add(r.title)
                    all_snippets.append(f"- {r.title}: {r.snippet}")
        if not all_snippets:
            return []
        content = "\n".join(all_snippets[:25])
        problems = await workflow.problem_discoverer._analyze_content_for_problems(
            content, source="web_search", domain=domain
        )
        return problems
    except Exception as e:
        console.warning(f"Web search research failed: {e}")
        return []


