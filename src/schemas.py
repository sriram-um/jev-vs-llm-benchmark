"""Typed inputs, outputs, and telemetry for the benchmark.

The decision task is defined ONCE here and consumed by both evaluators, so the
two systems answer identically-worded questions. If you change an anchor, both
sides change together -- that is the point.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# The decision task
# ---------------------------------------------------------------------------

Department = Literal[
    "billing",
    "infrastructure_outage",
    "account_security",
    "feature_request",
]

#: Choice criteria -- option name -> description. Passed verbatim to Jev's
#: `Choice(criteria=...)` and rendered into the LLM prompt.
DEPARTMENT_CRITERIA: dict[str, str] = {
    "billing": (
        "Payments, invoices, subscriptions, refunds, proration, or disputed charges."
    ),
    "infrastructure_outage": (
        "The service is down, degraded, timing out, erroring, or unreachable "
        "for one or more regions."
    ),
    "account_security": (
        "Unauthorised access, credential compromise, suspicious logins, MFA "
        "problems, API key leakage, or possible data exposure."
    ),
    "feature_request": (
        "A request for a capability the product does not have yet, an "
        "enhancement, or a roadmap question."
    ),
}

#: Score rubric -- ordered level descriptions. Jev's Score primitive accepts
#: 2-10 ordered levels and returns the probability-weighted mean over their
#: indices (0..N). https://docs.typesafe.ai/primitives/score.md
URGENCY_LEVELS: list[str] = [
    "No time pressure. Informational, cosmetic, or a question about the future.",
    "Minor inconvenience. A workaround exists and nobody is blocked.",
    "Materially blocked on one workflow, but the business keeps operating.",
    "Severe. A core workflow is broken, or revenue or data is at risk.",
    "Critical. An active outage, an active breach, or imminent irreversible loss.",
]

#: Multiply a rubric level by this to get the 0-100 display scale in the brief.
URGENCY_DISPLAY_SCALE = 25

ESCALATE_INSTRUCTIONS = (
    "This ticket requires immediate escalation to an on-call human supervisor, "
    "rather than being handled in the normal support queue."
)

DEPARTMENT_INSTRUCTIONS = "Which team should own this support ticket?"

URGENCY_INSTRUCTIONS = "How time-critical is this ticket for the customer?"


class TriageDecision(BaseModel):
    """The output contract BOTH systems must satisfy, byte for byte.

    `urgency_level` is an enum rather than an int with `ge`/`le` bounds on
    purpose: OpenAI Structured Outputs' strict mode does not support the
    `minimum`/`maximum` JSON Schema keywords, but it does support enums. Using
    a Literal keeps the constraint enforceable on both sides instead of being
    silently dropped for one of them.
    """

    model_config = ConfigDict(extra="forbid")  # -> additionalProperties: false

    department: Department = Field(description=DEPARTMENT_INSTRUCTIONS)
    urgency_level: Literal[0, 1, 2, 3, 4] = Field(
        description=(
            "How time-critical the ticket is, as a rubric level. "
            + " ".join(f"{i}={d}" for i, d in enumerate(URGENCY_LEVELS))
        )
    )
    escalate_supervisor: bool = Field(description=ESCALATE_INSTRUCTIONS)

    @property
    def urgency_0_100(self) -> int:
        """Rubric level rescaled to the 0-100 presentation scale."""
        return self.urgency_level * URGENCY_DISPLAY_SCALE


class Scenario(BaseModel):
    """One synthetic triage ticket plus its ground-truth label."""

    id: str
    ticket: str                     # the `state` handed to both evaluators
    truth: TriageDecision

    # Provenance of the difficulty knobs, so metrics can be sliced by them.
    ambiguous: bool = False         # department signal is deliberately mixed
    distractor_chars: int = 0       # irrelevant log spew padded into `ticket`
    has_negation: bool = False
    has_mixed_dates: bool = False


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


class EvalRecord(BaseModel):
    """One evaluator's attempt at one scenario."""

    scenario_id: str
    evaluator: str                  # "jev" | "gpt-4o-mini" | ...
    model: str                      # resolved model id as reported by the API
    source: Literal["live", "simulated"]

    prediction: TriageDecision | None = None
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    schema_ok: bool = True          # produced a schema-valid object at all
    parse_error: str | None = None
    retries: int = 0                # extra attempts spent to get valid output

    # --- calibration signals ----------------------------------------------
    # Only fields that hold a GENUINE probability may feed a calibration metric.
    # The three kinds are kept in separate fields precisely so that no summariser
    # can mix them up by reading one field and assuming the other's semantics.
    #
    # p_department: probability mass the system put on the option it chose.
    #   Jev: `probabilities[choice]` -- a real distribution over the criteria.
    #   The LLM leaves this None: it has no distribution over departments.
    # p_escalate: probability that escalation is required.
    #   Jev: the raw `noul` float. The LLM leaves this None.
    p_department: float | None = None
    p_escalate: float | None = None
    urgency_expected: float | None = None   # Jev's continuous weighted mean

    # p_first_token_proxy: exp(logprob) of the LLM's FIRST GENERATED TOKEN.
    #   Not a distribution over answers -- the first token of a JSON object is a
    #   brace. Recorded to show what a token generator can offer in place of a
    #   distribution, and deliberately given its own field so it can never be
    #   mistaken for `p_department` and end up in an ECE column.
    p_first_token_proxy: float | None = None

    # confidence_*: Jev's own `confidence` statistic -- recorded for contrast but
    #   NOT a calibration input. The docs define it as a peakedness measure of the
    #   distribution, not a probability of correctness.
    #   https://docs.typesafe.ai/confidence.md
    confidence_department: float | None = None
    confidence_urgency: float | None = None

    error: str | None = None        # request failed outright


class Provenance(BaseModel):
    """Stamped onto every summary, table, and figure caption."""

    generated_at: str
    mode: Literal["live", "simulated", "mixed"]
    jev_model: str
    llm_model: str
    n_scenarios: int
    dataset_sha256: str
    seed: int
    git_sha: str | None = None
    simulated_warning: str | None = None

    def caption(self) -> str:
        bits = [
            f"n={self.n_scenarios}",
            f"jev={self.jev_model}",
            f"llm={self.llm_model}",
            f"mode={self.mode}",
            f"data={self.dataset_sha256[:12]}",
            self.generated_at,
        ]
        return " · ".join(bits)


class EvaluatorSummary(BaseModel):
    """Aggregated metrics for one evaluator."""

    evaluator: str
    model: str
    source: Literal["live", "simulated"]
    n: int

    # latency (ms), warm-up excluded
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float

    # quality
    department_accuracy: float
    department_macro_f1: float
    escalate_accuracy: float
    urgency_mae_0_100: float
    urgency_rmse_0_100: float
    urgency_mae_expected_0_100: float | None = None  # Jev's continuous score
    exact_match_all_three: float = 0.0

    # reliability
    schema_conformance: float
    total_retries: int
    hard_errors: int

    # calibration (None when the evaluator exposes no probability)
    ece_department: float | None = None
    ece_escalate: float | None = None
    brier_escalate: float | None = None
    ece_department_on_confidence: float | None = None  # contrast only
    # Mean first-token probability, for the evaluator that has only that. A value
    # pinned near 1.0 is the finding: it is confident about emitting `{`.
    mean_first_token_proxy: float | None = None

    # economics
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    cost_per_1k_decisions_usd: float


class SummaryReport(BaseModel):
    provenance: Provenance
    evaluators: list[EvaluatorSummary]
    # Binned reliability data, so the table and the plot read one source.
    reliability: dict[str, list[dict[str, float]]] = {}
    latency_samples: dict[str, list[float]] = {}
