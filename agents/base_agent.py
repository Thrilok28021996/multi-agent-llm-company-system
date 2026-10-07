"""Base agent class with first principles thinking."""

import asyncio
import functools
import json
import os
import random
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

# Global semaphores — limit concurrent LLM calls across all agents.
# LM Studio on 16GB RAM handles 1-2 concurrent requests well; beyond that queues spike.
# LLM_MAX_CONCURRENCY env var allows tuning per host.
_LLM_MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "2"))
_LLM_SEMAPHORE = asyncio.Semaphore(_LLM_MAX_CONCURRENCY)
_LLM_SYNC_SEMAPHORE = threading.Semaphore(_LLM_MAX_CONCURRENCY)

from config.llm_client import get_llm_client
from config.models import ModelConfig
from config.roles import AgentRole

from agents.thinking import ThinkingEngine, ThinkingDepth
from memory.agent_memory import AgentMemory
from memory.learning import AgentLearning
from tools.command_executor import CommandExecutor
from tools.file_operations import FileOperations
from ui.console import console
from utils.cost_tracker import track_usage
from utils.output_validator import OutputValidator


# ============================================================
#  TECHNIQUE 3 — MANDATORY CHAIN-OF-THOUGHT CONSTANTS
# ============================================================

COT_SYSTEM_SUFFIX = """

MANDATORY REASONING FORMAT - Always structure responses as:
<thinking>
1. What exactly is being asked?
2. What constraints apply?
3. What are the key risks or failure modes?
4. What evidence supports each option?
5. What would change my conclusion?
</thinking>
<answer>
Your final answer here. Concise, actionable, directly addressing the request.
</answer>

Do NOT omit the <thinking> block. Do NOT write anything before <thinking>.
"""

# ============================================================
#  TECHNIQUE 12 — STEP-BACK PROMPTING CONSTANTS
# ============================================================

STEP_BACK_PREFIX = """Before answering, take a step back:
1. What are the underlying principles or concepts relevant to this question?
2. What general approach would an expert use here?
3. What assumptions am I making that could be wrong?

Now apply this thinking to give your answer:
"""

# ============================================================
#  METACOGNITIVE UNCERTAINTY SIGNALS
# ============================================================

UNCERTAINTY_SIGNALS = [
    "i'm not sure", "i think", "might be", "could be",
    "uncertain", "not certain", "probably", "possibly"
]

# ============================================================
#  TECHNIQUE 6 — ADAPTIVE TEST-TIME COMPUTE CONSTANTS
# ============================================================

COMPLEXITY_KEYWORDS_COMPLEX = [
    "architecture", "system design", "integration", "distributed",
    "security", "authentication", "scalability", "concurrent",
    "migration", "refactor", "enterprise", "multi-tenant",
    "microservices", "api gateway", "database schema", "real-time"
]

COMPLEXITY_KEYWORDS_SIMPLE = [
    "list", "summarize", "explain", "describe", "format",
    "rename", "update config", "add comment", "fix typo"
]


# ============================================================
#  DATA CLASSES + ENUMS
# ============================================================


class LLMError(Exception):
    """Exception raised for LLM-related errors."""
    pass


class LLMConnectionError(LLMError):
    """Exception raised when cannot connect to LLM server."""
    pass


class LLMTimeoutError(LLMError):
    """Exception raised when LLM request times out."""
    pass


class LLMModelNotFoundError(LLMError):
    """Exception raised when model is not available."""
    pass


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delays
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)

                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    console.warning(f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {str(e)}...")
                    time.sleep(delay)

            raise last_exception
        return wrapper
    return decorator


def async_retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,)
):
    """Async version of retry_with_backoff decorator."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        raise

                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    console.warning(f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {str(e)}...")
                    await asyncio.sleep(delay)

            raise last_exception
        return wrapper
    return decorator


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    name: str
    role: AgentRole
    model: str
    first_principles: List[str]
    system_prompt: str
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class Message:
    """Message for inter-agent communication."""

    sender: str
    recipient: str  # Agent name or "all" for broadcast
    content: str
    message_type: str = "general"  # general, task, question, decision, report
    priority: int = 1  # 1-5, higher = more important
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "message_type": self.message_type,
            "priority": self.priority,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class TaskResult:
    """Result of a task execution."""

    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    artifacts: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0  # 0.0-1.0 confidence in the result
    first_principles_score: float = 1.0  # 0.0-1.0 first principles compliance


# ============================================================
#  BASE AGENT CLASS
# ============================================================


class BaseAgent(ABC):
    """
    Base class for all company agents.
    Implements first principles thinking and core agent capabilities.
    """

    def __init__(
        self,
        config: AgentConfig,
        workspace_root: str = ".",
        memory_persist_dir: Optional[str] = None
    ):
        self.config = config
        self.name = config.name
        self.role = config.role
        self.model = config.model
        self.model_spec = ModelConfig().get_model(config.role)
        self.first_principles = config.first_principles
        self.system_prompt = config.system_prompt
        # Initialize memory
        self.memory = AgentMemory(
            agent_name=self.name,
            persist_dir=memory_persist_dir
        )

        # Initialize learning system
        self.learning = AgentLearning(
            agent_name=self.name,
            persist_dir=memory_persist_dir or "./output/learning"
        )

        # Initialize structured thinking engine (template-based, no extra LLM calls)
        self.thinking_engine = ThinkingEngine(
            first_principles=config.first_principles
        )

        # Initialize tools
        self.file_ops = FileOperations(workspace_root=workspace_root)
        self.command_executor = CommandExecutor(
            workspace_root=workspace_root,
            safe_mode=True
        )

        # Message handling
        self.inbox: List[Message] = []
        self.outbox: List[Message] = []

        # Callbacks for message bus integration
        self._send_message_callback: Optional[Callable] = None

        # Streaming settings (can be enabled by workflow/config)
        self._streaming_enabled = False
        self._stream_callback: Optional[Callable[[str], None]] = None

        # Memory monitor reference (set by workflow)
        self._memory_monitor = None

        # Shared memory reference (set by workflow for institutional knowledge)
        self._shared_memory = None

        # State
        self.is_busy = False
        self.current_task: Optional[str] = None

        # RAG store (set by workflow after initialization)
        self._rag_store = None

        # ReAct tool use loop support
        self._react_enabled = False  # Set to True for agents that support tool use
        self._tool_schemas = {}

        # Technique 9: Prompt compressor (lazy init)
        self._compressor = None

        # Technique 10: Output validator
        self._validator = OutputValidator()

    def set_message_callback(self, callback: Callable) -> None:
        """Set callback for sending messages through message bus."""
        self._send_message_callback = callback

    def set_rag_store(self, rag_store) -> None:
        """Set the RAG store for injecting relevant past solutions into prompts."""
        self._rag_store = rag_store

    # ============================================================
    #  REACT TOOL USE LOOP
    # ============================================================

    def enable_react_tools(self):
        """Enable ReAct tool use loop for this agent."""
        from tools.tool_registry import get_tools_for_role
        self._react_enabled = True
        role_name = self.role.value if hasattr(self.role, 'value') else str(self.role)
        self._tool_schemas = get_tools_for_role(role_name)

    def _get_tools_prompt_suffix(self) -> str:
        """Get tool descriptions to append to system prompt."""
        if not self._react_enabled or not self._tool_schemas:
            return ""
        from tools.tool_registry import format_tools_for_prompt
        return format_tools_for_prompt(self._tool_schemas)

    async def generate_with_tools(self, prompt: str, context: str = "") -> str:
        """
        Generate a response using the ReAct tool use loop.
        LLM can call tools iteratively and reason about their results.
        Falls back to standard generate_response_async if tools not enabled.
        """
        if not self._react_enabled or not self.model_spec:
            return await self.generate_response_async(prompt)

        from agents.react_loop import ReActLoop
        from tools.tool_registry import format_tools_for_prompt

        # Build system prompt with tool descriptions
        tools_suffix = format_tools_for_prompt(self._tool_schemas)
        system_content = self.system_prompt + "\n" + tools_suffix

        # Apply CoT suffix if not a code task
        _is_code_task = prompt.strip().startswith("```") or (
            prompt.count("```") >= 2 and len(prompt) < 1000
        )
        if not _is_code_task:
            system_content += "\n" + COT_SYSTEM_SUFFIX

        # Build initial messages
        messages = [{"role": "system", "content": system_content}]

        # Add RAG context
        if self._rag_store is not None:
            rag_ctx = self._rag_store.format_for_prompt(prompt)
            if rag_ctx:
                messages.insert(1, {"role": "system", "content": rag_ctx})

        # Add recent conversation history for context
        history = self.memory.get_conversation_history(limit=10)
        if history:
            messages.extend(history[-6:])

        messages.append({"role": "user", "content": prompt})

        # Run ReAct loop
        loop = ReActLoop(max_iterations=8)
        final_response, _ = await loop.run(self, messages, tools_enabled=True)

        # Parse CoT from response
        thinking, answer = self._parse_cot_response(final_response)
        result = answer if answer else final_response
        result = self._clean_llm_output(result)

        # Update memory
        self.memory.add_to_conversation("user", prompt)
        self.memory.add_to_conversation("assistant", result)

        return result

    # ============================================================
    #  FIRST PRINCIPLES THINKING
    # ============================================================

    def apply_first_principles(self, problem: str) -> str:
        """
        Apply first principles thinking to a problem.
        This is called before any major decision or action.
        """
        principles_text = "\n".join(
            f"  {i+1}. {p}" for i, p in enumerate(self.first_principles)
        )

        thinking_prompt = f"""
As {self.name} ({self.role.value}), I need to think about this problem using first principles.

Problem/Task: {problem}

My First Principles:
{principles_text}

Let me break this down:
1. What is the fundamental truth or core issue here?
2. What assumptions am I making that might be wrong?
3. How do my first principles apply to this situation?
4. What is the simplest, most direct solution?
5. What evidence supports my conclusion? What evidence would disprove it?
6. What is one counter-argument to my approach?

First Principles Analysis:
"""
        return thinking_prompt

    def think(self, problem: str) -> str:
        """
        Think through a problem using first principles.
        Returns the thinking process as a string.
        Uses fallback model if primary is unavailable.
        """
        thinking_prompt = self.apply_first_principles(problem)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": thinking_prompt}
        ]

        # Store original settings so we can restore them even on failure
        original_temp = self.config.temperature
        original_max_tokens = self.config.max_tokens

        try:
            # Use lower temperature for analytical thinking
            self.config.temperature = 0.3
            self.config.max_tokens = max(original_max_tokens, 2048)

            thinking = self._call_llm_with_retry(messages, max_retries=2)

        except LLMError as e:
            console.warning(f"[{self.name}] think() failed: {e}")
            thinking = ""
        finally:
            # Always restore — prevents stale temperature leaking into subsequent calls
            self.config.temperature = original_temp
            self.config.max_tokens = original_max_tokens

        # Record the thinking in memory
        self.memory.add_to_conversation("user", f"[THINKING] {problem}")
        self.memory.add_to_conversation("assistant", thinking)

        return thinking

    def structured_think(
        self,
        problem: str,
        depth: ThinkingDepth = ThinkingDepth.STANDARD
    ) -> str:
        """
        Perform structured multi-phase thinking on a problem.
        Uses ThinkingEngine for step-by-step reasoning without extra LLM calls.

        Args:
            problem: The problem to think about
            depth: Thinking depth (MINIMAL, STANDARD, DEEP, EXHAUSTIVE)

        Returns:
            Conclusion string from structured reasoning
        """
        block = self.thinking_engine.think(
            query=problem,
            context={"role": self.role.value, "agent": self.name},
            depth=depth,
        )

        # Record in memory
        self.memory.add_to_conversation("user", f"[STRUCTURED THINKING] {problem}")
        self.memory.add_to_conversation("assistant", block.conclusion or "")

        return block.conclusion or block.to_markdown()

    def _get_principles_checklist(self, max_principles: int = 3) -> str:
        """Get first principles as a checklist for injection into task prompts."""
        if not self.first_principles:
            return ""
        items = "\n".join(f"- {p}" for p in self.first_principles[:max_principles])
        return f"\nSTOP. Before you write your answer, check each of these:\n{items}\nIf any answer is NO, adjust your response.\n"

    def _build_evidence_prompt(self) -> str:
        """Build evidence-tracking suffix appended to task prompts."""
        return (
            "\n\nEVIDENCE TRACKING: For every claim or conclusion in your response, "
            "mark it as [EVIDENCE: source] or [ASSUMPTION]. Unmarked claims will be challenged."
        )

    def _get_problem_preamble(self, task_type: str) -> str:
        """Generate a mandatory problem decomposition preamble.

        For all task types except 'fix_bug', forces the agent to
        state the problem, root cause, and why existing solutions fail
        before proposing a solution.
        """
        if task_type == "fix_bug":
            return ""
        return (
            "\nBefore proposing your solution, you MUST answer these 3 questions:\n"
            "1. STATE THE PROBLEM in one sentence.\n"
            "2. STATE THE ROOT CAUSE in one sentence.\n"
            "3. STATE WHY EXISTING SOLUTIONS FAIL in one sentence.\n"
            "Then propose your solution.\n"
        )

    # ============================================================
    #  FIRST PRINCIPLES VERIFICATION
    # ============================================================

    def _verify_first_principles(self, response: str, prompt: str) -> tuple:
        """Verify the response actually applied first principles.

        Lightweight text analysis — NOT another LLM call.
        Returns (compliant: bool, issues: str).
        """
        if not response or not self.first_principles:
            return True, ""

        issues = []

        # Check 1: Response must contain some reasoning (not just conclusions)
        reasoning_indicators = [
            "because", "therefore", "since", "given that", "considering",
            "analysis", "reason", "evidence", "based on", "fundamentally",
            "first principle", "core issue", "assumption"
        ]
        response_lower = response.lower()
        has_reasoning = any(ind in response_lower for ind in reasoning_indicators)
        if not has_reasoning and len(response) > 100:
            issues.append("Response lacks reasoning — only conclusions without justification")

        # Check 2: Response should reference at least 1 principle concept
        principle_keywords = set()
        for principle in self.first_principles:
            # Extract key words from each principle (skip short/common words)
            for word in principle.lower().split():
                word = word.strip("?.,!\"'")
                if len(word) > 4:
                    principle_keywords.add(word)

        if principle_keywords:
            keyword_hits = sum(1 for kw in principle_keywords if kw in response_lower)
            if keyword_hits == 0 and len(response) > 200:
                issues.append("Response does not reference any first principle concepts")

        # Check 3: Response length proportional to complexity
        # Very short responses to complex prompts suggest superficial thinking
        if len(prompt) > 500 and len(response) < 50:
            issues.append("Response too brief for the complexity of the task")

        # Check 4: Reasoning structure — cause/effect chains, not just keywords
        reasoning_patterns = [
            r'\b\w+.*because\s+\w+',          # "X because Y"
            r'\bif\b.*\bthen\b',              # "if X then Y"
            r'\bgiven\b.*\b(therefore|so)\b',  # "given X, therefore Y"
            r'\bsince\b.*\b(we|it|this)\b',    # "since X, we/it/this..."
            r'\d+\.\s+.*\n.*\d+\.\s+',         # numbered reasoning steps
        ]
        has_structure = any(re.search(p, response_lower, re.DOTALL) for p in reasoning_patterns)
        if not has_structure and len(response) > 300:
            issues.append("Response contains reasoning words but lacks structured argument")

        # Check 5: First principles should identify assumptions
        assumption_words = ["assumption", "assuming", "presuppose", "underlying",
                            "prerequisite", "depends on", "contingent"]
        has_assumption_awareness = any(w in response_lower for w in assumption_words)
        if not has_assumption_awareness and len(response) > 500:
            issues.append("First-principles analysis should identify underlying assumptions")

        # Check 6: Evidence vs assumption tagging
        has_evidence_tags = "[evidence" in response_lower or "[assumption" in response_lower
        if not has_evidence_tags and len(response) > 300:
            issues.append("Response should distinguish evidence from assumptions using [EVIDENCE] and [ASSUMPTION] tags")

        compliant = len(issues) == 0
        return compliant, "; ".join(issues)

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence from response content.

        Analyzes language patterns to estimate confidence level.
        Returns a float between 0.0 and 1.0.
        """
        if not response:
            return 0.3

        response_lower = response.lower()

        # Start at neutral
        confidence = 0.5

        # Hedging language lowers confidence
        hedging_words = [
            "maybe", "possibly", "might", "perhaps", "could be",
            "not sure", "uncertain", "unclear", "i think",
            "probably", "potentially", "it seems", "appears to"
        ]
        hedge_count = sum(1 for w in hedging_words if w in response_lower)
        confidence -= hedge_count * 0.05

        # Strong assertions raise confidence
        strong_words = [
            "definitely", "clearly", "certainly", "without doubt",
            "confirmed", "verified", "proven", "evidence shows",
            "tested and", "works correctly", "passes all"
        ]
        strong_count = sum(1 for w in strong_words if w in response_lower)
        confidence += strong_count * 0.05

        # Error/warning mentions lower confidence
        error_words = ["error", "warning", "bug", "issue", "problem", "fail",
                       "crash", "broken", "wrong", "invalid"]
        error_count = sum(1 for w in error_words if w in response_lower)
        confidence -= error_count * 0.03

        # Very short responses get lower confidence
        if len(response) < 50:
            confidence = min(confidence, 0.5)

        return max(0.1, min(1.0, round(confidence, 2)))

    # ============================================================
    #  RESPONSE GENERATION
    # ============================================================

    def generate_response(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        use_first_principles: bool = True,
        streaming: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Generate a response to a prompt.

        Args:
            prompt: The input prompt
            context: Additional context
            use_first_principles: Whether to apply first principles thinking first
            streaming: Whether to stream the response
            stream_callback: Optional callback for streaming output (called with each token)
        """
        # Augment system prompt with personality, experience, supervision, lessons, and OKRs
        system_content = self.system_prompt

        # Inject OKR context
        try:
            from company.performance import DEFAULT_OKRS
            okr = DEFAULT_OKRS.get(self.config.role.value)
            if okr:
                system_content += f"\nYour objective: {okr.objective}\nKey results: {', '.join(okr.key_results)}\n"
        except ImportError:
            pass

        if hasattr(self, '_personality') and self._personality:
            mod = self._personality.get_prompt_modifier()
            if mod:
                system_content += f"\n\n[Your Working Style]: {mod}"

        if hasattr(self, '_experience') and self._experience:
            adj = self._experience.get_prompt_adjustment()
            if adj:
                system_content += f"\n\n[Your Experience Level]: {adj}"

            # Experience-based behavior differentiation
            exp_level = getattr(self._experience, 'level', None)
            if exp_level == "junior":
                system_content += (
                    "\n\n[Experience Guidance]: You are at JUNIOR level. "
                    "You MUST get approval before proceeding with non-obvious decisions. "
                    "Break your work into small, verifiable steps. "
                    "Ask clarifying questions when requirements are ambiguous. "
                    "Show your reasoning step by step."
                )
            elif exp_level == "senior":
                system_content += (
                    "\n\n[Experience Guidance]: You are at SENIOR level. "
                    "You may skip trivial review gates and make judgment calls. "
                    "Focus on the big picture and flag only significant issues. "
                    "Mentor junior agents when providing feedback."
                )
            elif exp_level == "lead":
                system_content += (
                    "\n\n[Experience Guidance]: You are at LEAD level. "
                    "You set the technical direction for your domain. "
                    "When reviewing others' work, provide mentoring context. "
                    "Challenge assumptions and push for first-principles thinking. "
                    "You may override junior decisions with justification."
                )

        if hasattr(self, '_supervision_note') and self._supervision_note:
            system_content += f"\n\n[Supervisor Feedback]: {self._supervision_note}"

        if hasattr(self, '_retrospective_lessons') and self._retrospective_lessons:
            system_content += f"\n\n[Team Lessons from Previous Runs]:\n{self._retrospective_lessons}"

        # Anti-bias cognitive discipline — injected into every agent's prompt
        system_content += """

COGNITIVE DISCIPLINE:
- Base decisions on EVIDENCE presented in this conversation, not on training data patterns.
- If you are uncertain, say "I am uncertain because [reason]" — never fake confidence.
- Challenge the previous agent's conclusions if your analysis disagrees. Dissent is valued.
- Distinguish between "I know this from evidence" and "I assume this from patterns." Label each.
- If all agents agree and no one has pushed back, that itself is a red flag. Find the weakness.
"""

        # Technique 3: Inject CoT suffix for non-code-generation tasks.
        # Skip injection when the prompt primarily expects raw code output (```-delimited)
        # to avoid wrapping code in XML tags and breaking downstream parsers.
        _is_code_task = prompt.strip().startswith("```") or (
            prompt.count("```") >= 2 and len(prompt) < 1000
        )
        if not _is_code_task:
            system_content += COT_SYSTEM_SUFFIX

        messages = [{"role": "system", "content": system_content}]

        # Add conversation history for context
        history = self.memory.get_conversation_history(limit=10)
        messages.extend(history)

        # Inject learning advice if available
        advice = self.learning.get_advice_for_task(self.role.value, prompt)
        if advice and advice.strip():
            messages.append({
                "role": "system",
                "content": f"[Lessons from past experience]\n{advice}"
            })

        # Inject relevant institutional knowledge from shared memory
        if self._shared_memory:
            institutional = self._get_institutional_context(prompt)
            if institutional:
                messages.append({
                    "role": "system",
                    "content": f"[Institutional Knowledge]\n{institutional}"
                })

        # RAG context injection
        if self._rag_store is not None:
            rag_context = self._rag_store.format_for_prompt(prompt)
            if rag_context:
                messages.insert(1, {"role": "system", "content": rag_context})

        # Apply first principles if needed.
        # Only invoke a full LLM `think()` call for complex prompts; otherwise
        # fall back to the template-based `structured_think` (no extra LLM call)
        # to avoid doubling latency on every invocation.
        if use_first_principles:
            complexity = self._detect_complexity(prompt)
            if complexity == "complex":
                thinking = self.think(prompt)
            else:
                try:
                    thinking = self.structured_think(prompt)
                except Exception:
                    thinking = ""
            if thinking:
                messages.append({
                    "role": "assistant",
                    "content": f"[My Analysis]\n{thinking}\n\n[My Response]"
                })

        # Add context if provided
        if context:
            context_str = json.dumps(context, indent=2)
            prompt = f"{prompt}\n\nContext:\n{context_str}"

        messages.append({"role": "user", "content": prompt})

        # Resolve streaming: explicit args override agent defaults
        use_streaming = streaming or self._streaming_enabled
        callback = stream_callback or self._stream_callback

        # Generate response with error handling
        try:
            if use_streaming and callback:
                # Streaming mode
                result = self._generate_streaming(messages, callback)
            else:
                # Non-streaming mode with retry
                result = self._call_llm_with_retry(messages)
        except LLMError as e:
            # Log the error and return a graceful failure message
            error_msg = f"LLM Error: {str(e)}"
            console.warning(error_msg)
            result = f"[Error: Unable to generate response - {str(e)}]"

        # Clean the response
        result = self._clean_llm_output(result)

        # Technique 3: Parse CoT blocks when the suffix was injected
        if not _is_code_task:
            thinking_block, answer_block = self._parse_cot_response(result)
            if thinking_block:
                console.debug(f"[CoT Thinking] {self.name}:\n{thinking_block}")
                # Metacognitive uncertainty check — warn callers to consider self-consistency
                if self._is_uncertain(thinking_block):
                    console.warning(
                        f"[Uncertainty] {self.name} expressed uncertainty in thinking. "
                        "Consider using generate_with_consistency() for higher-stakes tasks."
                    )
            if answer_block:
                result = answer_block
            else:
                # Model partially followed CoT — strip ALL residual thinking/tag artifacts
                result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
                result = re.sub(r'<think>.*', '', result, flags=re.DOTALL)
                result = re.sub(r'</?think[^>]*>', '', result)
                result = re.sub(r'</?thinking>', '', result)
                result = re.sub(r'</?answer>', '', result)
                result = re.sub(r'\n{3,}', '\n\n', result).strip()

        # Verify first principles compliance (lightweight, no LLM call).
        # We only trigger an actual LLM retry when the response shows NO reasoning
        # at all (zero indicators) or is too short to contain reasoning. Missing
        # [EVIDENCE]/[ASSUMPTION] tags alone are logged but do not force a retry —
        # local models rarely emit them and the wasted LLM call is expensive.
        if use_first_principles:
            compliant, fp_issues = self._verify_first_principles(result, prompt)
            # Compute reasoning-indicator score independently of tag presence.
            reasoning_indicators = [
                "because", "therefore", "since", "given that", "considering",
                "analysis", "reason", "evidence", "based on", "fundamentally",
                "first principle", "core issue", "assumption"
            ]
            result_lower = (result or "").lower()
            reasoning_score = sum(1 for ind in reasoning_indicators if ind in result_lower)
            # Store score on agent for TaskResult wiring (1.0 = fully compliant, 0.5 = partial)
            self._last_fp_score = 1.0 if compliant else 0.5
            # Track compliance in experience and learning systems
            if hasattr(self, '_experience') and self._experience:
                self._experience.add_experience("first_principles_check", success=compliant)
            if not compliant:
                self.learning.record_interaction(
                    task_type="first_principles_failure",
                    input_context=prompt,
                    action_taken="Response lacked reasoning",
                    outcome=fp_issues,
                    success=False,
                )
            console.debug(
                f"[FP] {self.name} reasoning_score={reasoning_score} "
                f"len={len(result or '')} compliant={compliant}"
            )
            # Only retry when the response truly lacks reasoning, not just tags.
            needs_retry = (reasoning_score == 0) or (len(result or "") < 50)
            if needs_retry:
                # Re-prompt with explicit principle guidance (one retry only)
                principles_text = "\n".join(
                    f"  - {p}" for p in self.first_principles
                )
                retry_prompt = (
                    f"{prompt}\n\n"
                    f"IMPORTANT: Your previous response lacked first-principles reasoning. "
                    f"You MUST address these principles in your response:\n{principles_text}\n"
                    f"Issues with previous response: {fp_issues}"
                )
                messages_retry = [{"role": "system", "content": self.system_prompt}]
                messages_retry.append({"role": "user", "content": retry_prompt})
                try:
                    result = self._call_llm_with_retry(messages_retry, max_retries=1)
                    result = self._clean_llm_output(result)
                except LLMError:
                    pass  # Keep the original response if retry fails

        # Update memory
        self.memory.add_to_conversation("user", prompt)
        self.memory.add_to_conversation("assistant", result)

        # Update memory monitor if available
        if hasattr(self, '_memory_monitor') and self._memory_monitor:
            history = self.memory.get_conversation_history()
            total_chars = sum(len(m.get("content", "")) for m in history)
            estimated_tokens = total_chars // 4  # Rough estimate
            self._memory_monitor.update_agent_memory(
                self.name, estimated_tokens, message_count=len(history)
            )

        return result

    # ============================================================
    #  LLM INTERACTION (RETRY, STREAMING, FALLBACK)
    # ============================================================

    def _trim_messages_to_budget(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 28000,
    ) -> List[Dict[str, str]]:
        """Trim messages to stay within the token budget.

        Estimates token count as sum(len(m['content']) // 4 for m in messages).
        If over budget: keeps system message, keeps last 4 messages, drops middle ones.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum estimated token budget (default 28000)

        Returns:
            Trimmed list of messages within budget
        """
        # Technique 9: Compress messages before checking budget
        from utils.prompt_compressor import PromptCompressor
        if not hasattr(self, '_compressor') or self._compressor is None:
            self._compressor = PromptCompressor(target_ratio=0.75)
        messages = self._compressor.compress_messages(messages, budget_tokens=max_tokens)

        estimated = sum(len(m.get("content", "")) // 4 for m in messages)
        if estimated <= max_tokens:
            return messages

        console.warning(
            f"[TokenBudget] Estimated {estimated} tokens exceeds budget {max_tokens}. "
            "Trimming conversation history."
        )

        # Separate system message(s) from the rest
        system_messages = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        # Pin the first non-system message (original task) so it isn't dropped
        # — losing the original task causes severe agent drift mid-conversation.
        first_user_msg = non_system[0] if non_system else None

        # Keep last 4 non-system messages; drop the middle
        if len(non_system) > 4:
            last_kept = non_system[-4:]
        else:
            last_kept = non_system

        # Build trimmed list: system + first_user + last_N (deduped)
        kept: List[Dict[str, str]] = []
        if first_user_msg is not None:
            kept.append(first_user_msg)
        for m in last_kept:
            if m is first_user_msg:
                continue
            kept.append(m)

        trimmed = system_messages + kept
        new_estimated = sum(len(m.get("content", "")) // 4 for m in trimmed)
        console.warning(
            f"[TokenBudget] Trimmed to {len(trimmed)} messages (~{new_estimated} tokens). "
            f"Dropped {len(non_system) - len(kept)} middle messages."
        )
        return trimmed

    def _call_llm_with_retry(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3,
    ) -> str:
        """Call the LLM with retry logic and exponential backoff.

        Returns:
            The LLM response content

        Raises:
            LLMError: If all retries fail
        """
        last_exception = None

        if not self.model_spec:
            raise LLMModelNotFoundError(f"No model spec found for role: {self.role}")

        # Enforce token budget before sending to LLM
        messages = self._trim_messages_to_budget(messages)

        for attempt in range(max_retries + 1):
            try:
                t0 = time.time()
                with _LLM_SYNC_SEMAPHORE:
                    text, input_tokens, output_tokens = get_llm_client().chat(
                        self.model_spec,
                        messages,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )
                if input_tokens or output_tokens:
                    track_usage(
                        model=self.model_spec.name,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        agent=self.name,
                        operation="chat",
                        duration_ms=(time.time() - t0) * 1000,
                    )
                return console._clean_llm_output(text)

            except FileNotFoundError as e:
                raise LLMModelNotFoundError(str(e))
            except ConnectionError as e:
                last_exception = LLMConnectionError(f"Connection error: {e}")
            except TimeoutError as e:
                last_exception = LLMTimeoutError(f"Request timed out: {e}")
            except Exception as e:
                last_exception = LLMError(f"Unexpected error: {e}")

            if attempt < max_retries:
                delay = min(1.0 * (2 ** attempt) * (0.5 + random.random()), 30.0)
                console.warning(f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s...")
                time.sleep(delay)

        raise last_exception or LLMError("Unknown error occurred")

    async def _call_llm_with_retry_async(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3,
    ) -> str:
        """Async version of _call_llm_with_retry — uses asyncio.sleep for non-blocking retries."""
        last_exception = None

        if not self.model_spec:
            raise LLMModelNotFoundError(f"No model spec found for role: {self.role}")

        messages = self._trim_messages_to_budget(messages)

        for attempt in range(max_retries + 1):
            try:
                t0 = time.time()
                text, input_tokens, output_tokens = await get_llm_client().chat_async(
                    self.model_spec,
                    messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                if input_tokens or output_tokens:
                    track_usage(
                        model=self.model_spec.name,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        agent=self.name,
                        operation="chat_async",
                        duration_ms=(time.time() - t0) * 1000,
                    )
                return console._clean_llm_output(text)

            except FileNotFoundError as e:
                raise LLMModelNotFoundError(str(e))
            except ConnectionError as e:
                last_exception = LLMConnectionError(f"Connection error: {e}")
            except TimeoutError as e:
                last_exception = LLMTimeoutError(f"Request timed out: {e}")
            except Exception as e:
                last_exception = LLMError(f"Unexpected error: {e}")

            if attempt < max_retries:
                delay = min(1.0 * (2 ** attempt) * (0.5 + random.random()), 30.0)
                console.warning(f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s...")
                await asyncio.sleep(delay)

        raise last_exception or LLMError("Unknown error occurred")

    def _generate_streaming(
        self,
        messages: List[Dict[str, str]],
        callback: Callable[[str], None],
    ) -> str:
        """Generate response with streaming output."""
        if not self.model_spec:
            raise LLMModelNotFoundError(f"No model spec found for role: {self.role}")

        # Enforce token budget before streaming
        messages = self._trim_messages_to_budget(messages)

        try:
            t0 = time.time()
            full_response, input_tokens, output_tokens = get_llm_client().chat_stream(
                self.model_spec,
                messages,
                callback=callback,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            if input_tokens or output_tokens:
                track_usage(
                    model=self.model_spec.name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    agent=self.name,
                    operation="chat_stream",
                    duration_ms=(time.time() - t0) * 1000,
                )
            return console._clean_llm_output(full_response)
        except FileNotFoundError as e:
            raise LLMModelNotFoundError(str(e))
        except ConnectionError as e:
            raise LLMConnectionError(str(e))
        except Exception as e:
            raise LLMError(f"Streaming error: {e}")

    def _clean_llm_output(self, text: str) -> str:
        """Clean LLM output by removing thinking tags and artifacts.

        Delegates to llm_client._clean_text which handles all known
        tag formats (Qwen3 think, generic thinking/reasoning/analysis, etc.).
        """
        from config.llm_client import _clean_text
        return _clean_text(text)

    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> TaskResult:
        """
        Execute a task assigned to this agent.
        Must be implemented by subclasses.

        Args:
            task: Task specification with 'type', 'description', and other params
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this agent has."""
        pass

    # ============================================================
    #  COMMUNICATION (MESSAGE BUS)
    # ============================================================

    def receive_message(self, message: Message) -> None:
        """Receive a message from another agent."""
        self.inbox.append(message)
        content_preview = message.content[:100].replace("\n", " ")
        print(f"[INBOX]  {self.name} ◄── {message.sender}  [{message.message_type.upper()}]  \"{content_preview}\"")
        self.memory.add_to_conversation(
            "user",
            f"[Message from {message.sender}] {message.content}"
        )

    def send_message(
        self,
        recipient: str,
        content: str,
        message_type: str = "general",
        priority: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Send a message to another agent."""
        message = Message(
            sender=self.name,
            recipient=recipient,
            content=content,
            message_type=message_type,
            priority=priority,
            metadata=metadata or {}
        )

        self.outbox.append(message)
        content_preview = content[:100].replace("\n", " ")
        print(f"[OUTBOX] {self.name} ──► {recipient}  [{message_type.upper()}]  \"{content_preview}\"")
        self.memory.add_to_conversation(
            "assistant",
            f"[Message to {recipient}] {content}"
        )

        # Use callback if available
        if self._send_message_callback:
            self._send_message_callback(message)

        return message

    def process_inbox(self) -> List[str]:
        """Process all messages in inbox and generate responses."""
        responses = []
        while self.inbox:
            message = self.inbox.pop(0)
            response = self._handle_message(message)
            responses.append(response)
        return responses

    def _handle_message(self, message: Message) -> str:
        """Handle a single incoming message."""
        prompt = f"""
I received a message from {message.sender}:
Type: {message.message_type}
Priority: {message.priority}
Content: {message.content}

How should I respond based on my role as {self.role.value}?
"""
        return self.generate_response(prompt, use_first_principles=True)

    # ============================================================
    #  TOOL USAGE
    # ============================================================

    def use_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Use a tool and record the result.

        Args:
            tool_name: Name of the tool to use
            **kwargs: Tool-specific arguments
        """
        result = {"success": False, "error": "Unknown tool"}

        try:
            if tool_name == "read_file":
                result = self.file_ops.read_file(**kwargs)
            elif tool_name == "write_file":
                result = self.file_ops.write_file(**kwargs)
            elif tool_name == "edit_file":
                result = self.file_ops.edit_file(**kwargs)
            elif tool_name == "list_directory":
                result = self.file_ops.list_directory(**kwargs)
            elif tool_name == "search_files":
                result = self.file_ops.search_in_files(**kwargs)
            elif tool_name == "execute_command":
                result = self.command_executor.execute(**kwargs)
            elif tool_name == "run_tests":
                result = self.command_executor.run_tests(**kwargs)
            elif tool_name == "git":
                result = self.command_executor.git_operation(**kwargs)

            # Record tool usage
            self.memory.record_tool_usage(tool_name, result.get("success", False))

        except Exception as e:
            result = {"success": False, "error": str(e)}
            self.memory.record_tool_usage(tool_name, False)

        return result

    # ============================================================
    #  MEMORY + LEARNING INTEGRATION
    # ============================================================

    def reflect(self, task_description: str, outcome: str, success: bool) -> str:
        """
        Reflect on a completed task and learn from it.

        Args:
            task_description: What the task was
            outcome: What happened
            success: Whether it was successful
        """
        prompt = f"""
I just completed a task. Let me reflect on it:

Task: {task_description}
Outcome: {outcome}
Success: {"Yes" if success else "No"}

Questions for reflection:
1. What went well?
2. What could have been done better?
3. What did I learn?
4. How does this relate to my first principles?
5. What would I do differently next time?

My reflection:
"""
        reflection = self.generate_response(prompt, use_first_principles=False)

        # Extract lessons (simple extraction)
        lessons = []
        if "learn" in reflection.lower() or "lesson" in reflection.lower():
            # Extract sentences that might be lessons
            sentences = reflection.split(".")
            for s in sentences:
                s = s.strip()
                if any(word in s.lower() for word in ["learn", "should", "will", "next time"]):
                    if len(s) > 10:
                        lessons.append(s)

        # Record experience
        self.memory.record_experience(
            action=task_description,
            context=self.memory.get_conversation_summary(),
            outcome=outcome,
            success=success,
            lessons=lessons[:3]  # Keep top 3 lessons
        )

        # Also record in the learning system
        self.learning.record_interaction(
            task_type=self.role.value,
            input_context=task_description,
            action_taken=outcome,
            outcome=outcome,
            success=success,
            metadata={"agent": self.name, "role": self.role.value}
        )

        return reflection

    def get_learning_advice(self, task_type: str, context: str) -> str:
        """Get advice from past learning before starting a task."""
        return self.learning.get_advice_for_task(task_type, context)

    def _get_institutional_context(self, task_context: str) -> str:
        """Query shared memory for relevant past decisions and solutions."""
        if not self._shared_memory:
            return ""
        parts = []
        # Search for relevant past context
        results = self._shared_memory.search(task_context)
        if results:
            for mem in results[:3]:
                parts.append(f"[{mem.type.value.upper()}] {mem.content}")
        # Also check recent high-importance items
        recent = self._shared_memory.get_recent(limit=3)
        for mem in recent:
            if mem.importance >= 0.7 and mem.content not in str(parts):
                parts.append(f"[RECENT] {mem.content}")
        return "\n".join(parts[:5]) if parts else ""

    # ============================================================
    #  MODEL MANAGEMENT
    # ============================================================

    def check_model_availability(self) -> bool:
        """Check if this agent's model spec is configured."""
        return self.model_spec is not None

    def get_model_status(self) -> Dict[str, Any]:
        """Get model status information."""
        backend = get_llm_client()._backend
        return {
            "model": self.model,
            "backend": backend,
            "model_id": self.model_spec.model_id(get_llm_client()._backend) if self.model_spec else None,
            "model_spec_configured": self.model_spec is not None,
        }

    # ============================================================
    #  STATUS & REPORTING
    # ============================================================

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "name": self.name,
            "role": self.role.value,
            "model": self.model,
            "is_busy": self.is_busy,
            "current_task": self.current_task,
            "inbox_count": len(self.inbox),
            "outbox_count": len(self.outbox),
            "memory_summary": self.memory.export_state()
        }

    def generate_report(self) -> str:
        """Generate a status report for this agent."""
        status = self.get_status()
        reflection = self.memory.generate_self_reflection()

        report = f"""
=== Agent Status Report ===
Name: {status['name']}
Role: {status['role']}
Model: {status['model']}
Currently Busy: {status['is_busy']}
Current Task: {status['current_task'] or 'None'}
Messages Pending: {status['inbox_count']} in, {status['outbox_count']} out

{reflection}
"""
        return report

    # ============================================================
    #  ASYNC SUPPORT
    # ============================================================

    async def think_async(self, problem: str) -> str:
        """Async version of think."""
        return await asyncio.to_thread(self.think, problem)

    async def generate_response_async(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        use_first_principles: bool = True,
        streaming: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Async version of generate_response — non-blocking retries via asyncio.sleep."""
        async with _LLM_SEMAPHORE:
            return await asyncio.to_thread(
                self.generate_response, prompt, context, use_first_principles,
                streaming, stream_callback
            )

    # ============================================================
    #  TECHNIQUE 3 — CHAIN-OF-THOUGHT PARSING
    # ============================================================

    def _parse_cot_response(self, raw: str) -> tuple:
        """Parse thinking/answer blocks from LLM output.

        Handles two formats:
        - Qwen3 native:   <think>...</think>  (answer is whatever follows)
        - Injected CoT:   <thinking>...</thinking><answer>...</answer>

        Returns:
            (thinking, answer) — if no thinking found returns ("", raw).
        """
        # Format 1: Qwen3 native <think>...</think>
        think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            answer = raw[think_match.end():].strip()
            return thinking, answer or raw[:think_match.start()].strip()

        # Format 1b: unclosed <think> tag (model still generating / truncated)
        unclosed = re.search(r'<think>(.*?)$', raw, re.DOTALL)
        if unclosed:
            thinking = unclosed.group(1).strip()
            answer = raw[:unclosed.start()].strip()
            return thinking, answer if answer else ""

        # Format 2: injected <thinking>/<answer> tags (strict)
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', raw, re.DOTALL)
        answer_match = re.search(r'<answer>(.*?)</answer>', raw, re.DOTALL)
        if thinking_match and answer_match:
            return thinking_match.group(1).strip(), answer_match.group(1).strip()

        # Format 2b: <thinking> present but no </thinking> — extract up to <answer>
        open_thinking = re.search(r'<thinking>(.*?)(?=<answer>|$)', raw, re.DOTALL)
        if open_thinking and answer_match:
            return open_thinking.group(1).strip(), answer_match.group(1).strip()

        return "", raw

    # ============================================================
    #  TECHNIQUE 6 — ADAPTIVE TEST-TIME COMPUTE
    # ============================================================

    def _detect_complexity(self, prompt: str) -> str:
        """Detect prompt complexity: returns 'simple', 'medium', or 'complex'."""
        prompt_lower = prompt.lower()

        complex_hits = sum(1 for kw in COMPLEXITY_KEYWORDS_COMPLEX if kw in prompt_lower)
        simple_hits = sum(1 for kw in COMPLEXITY_KEYWORDS_SIMPLE if kw in prompt_lower)

        if complex_hits >= 2 or (complex_hits >= 1 and len(prompt) > 400):
            return "complex"
        if simple_hits >= 1 and complex_hits == 0:
            return "simple"
        return "medium"

    def _get_compute_config(self, prompt: str) -> dict:
        """Return compute configuration based on prompt complexity.

        Returns:
            dict with 'consistency_samples' and 'refinement_passes'.
            simple=(1,1), medium=(3,2), complex=(5,3).
        """
        complexity = self._detect_complexity(prompt)
        if complexity == "simple":
            return {"consistency_samples": 1, "refinement_passes": 1}
        elif complexity == "complex":
            return {"consistency_samples": 5, "refinement_passes": 3}
        else:
            return {"consistency_samples": 3, "refinement_passes": 2}

    # ============================================================
    #  TECHNIQUE 2 — SELF-CONSISTENCY
    # ============================================================

    def _extract_decision_token(self, response: str, keywords: list = None) -> str:
        """Extract a decision token from a response for majority voting.

        Checks for keywords like approve/reject/yes/no/proceed/halt/pass/fail.
        Returns the first matched keyword in lowercase, or "" if none found.
        Uses word-boundary matching to avoid substring false positives
        (e.g., "no" matching "not", "pass" matching "password").
        """
        default_keywords = [
            "approve", "reject", "yes", "no", "proceed", "halt", "pass", "fail"
        ]
        check_keywords = keywords if keywords else default_keywords
        response_lower = response.lower()
        for kw in check_keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', response_lower):
                return kw
        return ""

    def generate_with_consistency(
        self,
        prompt: str,
        n: int = 3,
        decision_keywords: list = None
    ) -> str:
        """Generate N responses and return the one matching the majority decision token.

        Generates N responses sequentially with slightly varied temperature (±0.1 spread).
        Majority votes on decision tokens; returns the longest response matching the
        majority token. Falls back to responses[0] if n=1 or tokens cannot be parsed.

        Args:
            prompt: The input prompt
            n: Number of samples to generate
            decision_keywords: Optional list of keywords to detect decisions

        Returns:
            The best response based on majority voting
        """
        if n <= 1:
            return self.generate_response(prompt)

        original_temp = self.config.temperature
        # Suppress streaming during sampling — intermediate responses must not print to stdout
        orig_streaming = self._streaming_enabled
        orig_callback = self._stream_callback
        self._streaming_enabled = False
        self._stream_callback = None

        responses = []

        # Generate n responses with slightly varied temperatures
        spread = 0.1
        for i in range(n):
            if n > 1:
                offset = (i / (n - 1) - 0.5) * 2 * spread  # evenly spread ±0.1
            else:
                offset = 0.0
            self.config.temperature = max(0.0, min(1.0, original_temp + offset))
            try:
                resp = self.generate_response(prompt)
                responses.append(resp)
            except Exception as e:
                console.warning(f"[Consistency] Sample {i+1} failed: {e}")

        self.config.temperature = original_temp
        self._streaming_enabled = orig_streaming
        self._stream_callback = orig_callback

        if not responses:
            return ""

        # Extract decision tokens
        tokens = [self._extract_decision_token(r, decision_keywords) for r in responses]

        # Count votes per token (skip empty)
        from collections import Counter
        token_counts = Counter(t for t in tokens if t)
        if not token_counts:
            return responses[0]

        majority_token = token_counts.most_common(1)[0][0]

        # Return the longest response among those matching the majority token
        majority_responses = [
            r for r, t in zip(responses, tokens) if t == majority_token
        ]
        return max(majority_responses, key=len)

    async def generate_with_consistency_async(
        self,
        prompt: str,
        n: int = 3,
        decision_keywords: list = None
    ) -> str:
        """Async version of generate_with_consistency."""
        return await asyncio.to_thread(
            self.generate_with_consistency, prompt, n, decision_keywords
        )

    # ============================================================
    #  TECHNIQUE 5 — ITERATIVE SELF-REFINEMENT
    # ============================================================

    def generate_with_refinement(
        self,
        prompt: str,
        passes: int = 2,
        critique_focus: str = ""
    ) -> str:
        """Generate a response and iteratively refine it via self-critique.

        Pass 1: generate initial draft via generate_response().
        For each additional pass: send critique prompt asking the model to review
        its response against the original task, identify 3 weaknesses, then provide
        a REFINED_RESPONSE section.

        Args:
            prompt: The original task prompt
            passes: Total number of passes (1 = just initial draft, 2+ adds refinement)
            critique_focus: Optional focus area for the critique (e.g. "technical correctness")

        Returns:
            The final refined response string
        """
        # Suppress streaming for all refinement passes — no partial output should print
        orig_streaming = self._streaming_enabled
        orig_callback = self._stream_callback
        self._streaming_enabled = False
        self._stream_callback = None

        try:
            # Pass 1: initial draft
            current_response = self.generate_response(prompt)

            focus_clause = f" Focus especially on: {critique_focus}." if critique_focus else ""

            for pass_num in range(2, passes + 1):
                critique_prompt = (
                    f"You are reviewing your own previous response to this task:\n\n"
                    f"ORIGINAL TASK:\n{prompt}\n\n"
                    f"YOUR PREVIOUS RESPONSE:\n{current_response}\n\n"
                    f"Identify exactly 3 specific weaknesses or gaps in the previous response.{focus_clause}\n"
                    f"Then provide an improved version that fixes those weaknesses.\n\n"
                    f"Format your response as:\n"
                    f"WEAKNESSES:\n1. ...\n2. ...\n3. ...\n\n"
                    f"REFINED_RESPONSE:\n[your improved response here]"
                )

                critique_output = self.generate_response(critique_prompt, use_first_principles=False)

                # Extract REFINED_RESPONSE section
                refined_match = re.search(
                    r'REFINED_RESPONSE:\s*(.*)', critique_output, re.DOTALL | re.IGNORECASE
                )
                if refined_match:
                    current_response = refined_match.group(1).strip()
                else:
                    console.warning(
                        f"[Refinement] Pass {pass_num} did not produce a REFINED_RESPONSE section; "
                        "keeping previous response."
                    )
        finally:
            self._streaming_enabled = orig_streaming
            self._stream_callback = orig_callback

        return current_response

    async def generate_with_refinement_async(
        self,
        prompt: str,
        passes: int = 2,
        critique_focus: str = ""
    ) -> str:
        """Async version of generate_with_refinement."""
        return await asyncio.to_thread(
            self.generate_with_refinement, prompt, passes, critique_focus
        )

    # ============================================================
    #  TECHNIQUE 8 — TREE-OF-THOUGHTS
    # ============================================================

    async def generate_with_tot(self, prompt: str, evaluation_criteria: str = "") -> str:
        """Generate a response using Tree-of-Thoughts: branch, score, select best, generate final."""
        from agents.tree_of_thoughts import TreeOfThoughts
        tot = TreeOfThoughts(n_branches=3)
        response, best_branch = await tot.generate_best(self, prompt, evaluation_criteria)
        console.info(f"[{self.name} ToT] Best: {best_branch.approach[:60]} (score: {best_branch.score:.2f})")
        return response

    # ============================================================
    #  TECHNIQUE 10 — STRUCTURED OUTPUT VALIDATION
    # ============================================================

    def generate_validated(self, prompt: str, expected_format: str = "json", required_keys: list = None, max_retries: int = 2):
        """Generate response with format validation and auto-correction.

        Returns (raw_response, parsed_data). parsed_data is None if all retries fail.
        """
        if not hasattr(self, '_validator') or self._validator is None:
            self._validator = OutputValidator()

        response = self.generate_response(prompt)

        for attempt in range(max_retries):
            if expected_format == "json":
                valid, data, error = self._validator.validate_json(response, required_keys)
                if valid:
                    return response, data
            elif expected_format == "verdict":
                valid, verdict, error = self._validator.validate_verdict(response, required_keys or ["pass", "fail"])
                if valid:
                    return response, verdict
            elif expected_format == "code":
                valid, code, error = self._validator.validate_code_block(response)
                if valid:
                    return response, code
            else:
                return response, response

            if attempt < max_retries - 1:
                correction_prompt = self._validator.get_correction_prompt(prompt, response, error)
                response = self.generate_response(correction_prompt)

        return response, None

    # ============================================================
    #  TECHNIQUE 12 — BEST-OF-N WITH SELF-CERTAINTY SCORING
    # ============================================================

    def generate_best_of_n(self, prompt: str, n: int = 4, task_type: str = "general") -> str:
        """
        Generate N responses and score each by self-certainty.
        Self-certainty: ask the model to rate its own confidence 0-10.
        Return the highest-confidence response. No external reward model needed.
        Total LLM calls: N (generation) + N (scoring) = 2N sequential calls.
        """
        if n <= 1:
            return self.generate_response(prompt)

        candidates = []
        for i in range(n):
            # Slightly vary temperature for diversity
            orig_temp = self.config.temperature
            self.config.temperature = max(0.1, orig_temp + (i - n // 2) * 0.15)
            try:
                response = self.generate_response(prompt)
                candidates.append(response)
            finally:
                self.config.temperature = orig_temp

        if not candidates:
            return self.generate_response(prompt)

        # Score each candidate by self-certainty
        best_response = candidates[0]
        best_score = -1

        for candidate in candidates:
            score_prompt = (
                f"Rate the quality of this response to the given task on a scale of 0-10.\n\n"
                f"TASK: {prompt[:300]}\n\n"
                f"RESPONSE:\n{candidate[:600]}\n\n"
                "Consider: completeness, correctness, clarity, and adherence to requirements.\n"
                "Respond with only: SCORE: X (where X is 0-10)"
            )
            score_response = self.generate_response(score_prompt)
            match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', score_response)
            score = float(match.group(1)) if match else 5.0

            if score > best_score:
                best_score = score
                best_response = candidate

        return best_response

    async def generate_best_of_n_async(self, prompt: str, n: int = 4, task_type: str = "general") -> str:
        """Async wrapper for best-of-N sampling."""
        return await asyncio.to_thread(self.generate_best_of_n, prompt, n, task_type)

    # ============================================================
    #  TECHNIQUE 15 — STEP-BACK PROMPTING
    # ============================================================

    def generate_with_step_back(self, prompt: str) -> str:
        """Step-back prompting: reason about principles before answering."""
        enhanced_prompt = STEP_BACK_PREFIX + prompt
        return self.generate_response(enhanced_prompt)

    async def generate_with_step_back_async(self, prompt: str) -> str:
        """Async version of generate_with_step_back."""
        return await asyncio.to_thread(self.generate_with_step_back, prompt)

    # ============================================================
    #  METACOGNITIVE UNCERTAINTY DETECTION
    # ============================================================

    def _is_uncertain(self, thinking: str) -> bool:
        """Detect if the model expressed uncertainty in its thinking."""
        lower = thinking.lower()
        return sum(1 for s in UNCERTAINTY_SIGNALS if s in lower) >= 2

    # ============================================================
    #  TECHNIQUE 14 — HYPERTREE PLANNING
    # ============================================================

    async def generate_with_hypertree(self, prompt: str, context: str = "") -> str:
        """Use HyperTree planning for complex multi-step tasks."""
        from agents.hypertree_planner import HyperTreePlanner
        planner = HyperTreePlanner(max_depth=2, max_children=3)
        return await planner.plan_and_execute(self, prompt, context)
