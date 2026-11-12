"""The harness layer: prompt conditioning, bounded repair, backend choice, MCP tools.

``internal_ai.evals.harness`` is the scorer. This package is the thing a scorer
judges, and keeping the names apart is deliberate: a harness that decided whether
it passed would be marking its own work.
"""

from internal_ai.harness.backends import BackendChoice, BoundaryRefusal, choose_backend
from internal_ai.harness.conditioning import (
    ConditionedPrompt,
    ModelSignature,
    PromptTooLong,
    condition,
    estimated_tokens,
    preamble_budget,
)
from internal_ai.harness.core import (
    HARNESS_VERSION,
    MAX_HISTORY_TURNS,
    MAX_QUESTION_CHARS,
    MAX_TOOL_ROUNDS,
    BudgetRefused,
    Harness,
    HarnessAnswer,
    HarnessDeadlineExceeded,
    HarnessRefused,
    HarnessRequest,
    RequestRefused,
    prompt_digest,
)
from internal_ai.harness.mcp import (
    McpDenied,
    McpError,
    McpServer,
    McpToolkit,
    StdioTransport,
    ToolOffer,
    approved_server_gate,
    clean_description,
    scrubbed_env,
)
from internal_ai.harness.repair import Attempt, RepairOutcome, run_with_repair

__all__ = [
    "HARNESS_VERSION",
    "MAX_HISTORY_TURNS",
    "MAX_QUESTION_CHARS",
    "MAX_TOOL_ROUNDS",
    "Attempt",
    "BackendChoice",
    "BoundaryRefusal",
    "BudgetRefused",
    "ConditionedPrompt",
    "Harness",
    "HarnessAnswer",
    "HarnessDeadlineExceeded",
    "HarnessRefused",
    "HarnessRequest",
    "McpDenied",
    "McpError",
    "McpServer",
    "McpToolkit",
    "ModelSignature",
    "PromptTooLong",
    "RepairOutcome",
    "RequestRefused",
    "StdioTransport",
    "ToolOffer",
    "approved_server_gate",
    "choose_backend",
    "clean_description",
    "condition",
    "estimated_tokens",
    "preamble_budget",
    "prompt_digest",
    "run_with_repair",
    "scrubbed_env",
]
