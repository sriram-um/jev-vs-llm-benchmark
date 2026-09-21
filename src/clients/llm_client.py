"""Conventional-LLM baseline: generate the decision as a JSON object.

Three modes behind one `Evaluator`:

    LLM_PROVIDER=openai    gpt-4o-mini via Structured Outputs  (OPENAI_API_KEY)
    LLM_PROVIDER=bedrock   Claude on Amazon Bedrock            (AWS credentials)
    SIMULATE_LLM=true      local simulator, no network         (the default)

Both live providers are given the *same* task text, built from `src/schemas.py`,
so neither side gets a better-worded question than the other.

THE SIMULATOR IS AN ASSUMPTION, NOT A BASELINE. Its latency, conformance rate and
accuracy are numbers we chose (see `_Simulator`), taken from the figures the
project brief predicted before any run existed. Simulating both sides means a
default `make bench` compares one set of our assumptions against another. It
exists so the pipeline, the plots and the table injection can be exercised
offline; every output it produces is stamped `SIMULATED`. The comparison only
becomes evidence when both sides run live.

HONEST ACCOUNTING -- the one thing to understand about this file.

A structured-output request can fail to produce a usable object: the model can
refuse, hit the output-token ceiling mid-object, or emit something that does not
validate. When that happens we retry, up to `max_schema_retries` -- and we bill
*every* attempt's tokens, and we add every attempt's latency to the record.

That is the entire point. Re-prompting a generator until its JSON parses is what
production code actually does, and the cost of doing it belongs in the cost
column, not in a footnote. Measuring only the successful attempt would flatter
this baseline for a failure mode it genuinely has.

`schema_ok` therefore records whether a valid object was obtained *at all*, while
`retries` records what it took. A run with 100% conformance and a nonzero retry
count did not come for free.
"""

from __future__ import annotations

import asyncio
import math
import random
from typing import Any, Literal

from src.config import PARTNER_PRICING_CAVEAT, settings
from src.clients.base import Evaluator, Stopwatch, cost_usd, token_counter
from src.schemas import (
    DEPARTMENT_CRITERIA,
    ESCALATE_INSTRUCTIONS,
    URGENCY_LEVELS,
    EvalRecord,
    Scenario,
    TriageDecision,
)

LLMMode = Literal["openai", "bedrock", "sim"]

# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------
# Rendered once at import: it is identical for all 500 scenarios, which is what
# makes it a cacheable prefix -- and also what makes it 86% of the input tokens.

_DEPARTMENT_BLOCK = "\n".join(
    f"  - {name}: {desc}" for name, desc in DEPARTMENT_CRITERIA.items()
)
_URGENCY_BLOCK = "\n".join(f"  {i} = {desc}" for i, desc in enumerate(URGENCY_LEVELS))

SYSTEM_PROMPT = f"""You are a support-ticket triage classifier. Read the ticket and return exactly one JSON object with three fields. Do not explain your reasoning.

department -- which team should own this ticket:
{_DEPARTMENT_BLOCK}

urgency_level -- how time-critical the ticket is, as one of these rubric levels:
{_URGENCY_BLOCK}

escalate_supervisor -- true if and only if: {ESCALATE_INSTRUCTIONS}"""


def _user_prompt(scenario: Scenario) -> str:
    return f"Triage this support ticket:\n\n{scenario.ticket}"


class _AttemptFailed(Exception):
    """A request returned, but not a usable object. Retryable; already billed."""


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------


class _Simulator:
    """Behavioural stand-in for a hosted generator.

    Every constant below is an ASSUMPTION with a stated origin. None was measured
    by this project. They are set where the brief predicted them so that a
    simulated run exercises realistic shapes -- not so that it can be quoted.

    Like the Jev simulator, it builds the answer distribution first and samples
    from it; unlike the Jev simulator, it then *discards* the distribution, because
    the thing it is standing in for cannot report one. That asymmetry is the
    article's point rendered in code, and it is also why the LLM's calibration
    columns come out blank rather than flattering.
    """

    # Two-phase latency: time-to-first-token, then per-output-token generation.
    # Shaped after typical hosted small-model behaviour; the long right tail is
    # the part that matters for a p99.
    TTFT_LOG_MEDIAN = math.log(420.0)
    TTFT_LOG_SIGMA = 0.42
    MS_PER_OUTPUT_TOKEN = 11.0
    # Occasional multi-second stall. Real hosted endpoints have these; a
    # log-normal alone understates p99.
    STALL_PROBABILITY = 0.012
    STALL_MS_RANGE = (1500.0, 6000.0)

    # Structured Outputs is very reliable but not total: refusals, truncation and
    # the odd invalid object. The brief predicted ~98.8% first-attempt conformance.
    SCHEMA_FAILURE_PROBABILITY = 0.012

    # Slightly below the Jev simulator's ceiling, and it degrades harder on long
    # distractor-padded states -- an autoregressive reader has more text to be
    # distracted by. Again: assumed, not measured.
    BASE_DEPARTMENT_MASS = 0.90
    AMBIGUITY_PENALTY = 0.34
    DISTRACTOR_PENALTY_PER_KCHAR = 0.10
    MAX_DISTRACTOR_PENALTY = 0.26

    #: Output tokens for one three-field JSON object, measured on the real schema.
    OUTPUT_TOKENS = 22

    def __init__(self, seed: int, model: str) -> None:
        self._rng = random.Random(seed)
        self._model = model
        self._count_tokens = token_counter()
        self._prompt_tokens = self._count_tokens(SYSTEM_PROMPT)

    def _latency_ms(self) -> float:
        ms = self._rng.lognormvariate(self.TTFT_LOG_MEDIAN, self.TTFT_LOG_SIGMA)
        ms += self.OUTPUT_TOKENS * self.MS_PER_OUTPUT_TOKEN
        if self._rng.random() < self.STALL_PROBABILITY:
            ms += self._rng.uniform(*self.STALL_MS_RANGE)
        return ms

    def _department_distribution(self, scenario: Scenario) -> dict[str, float]:
        mass = self.BASE_DEPARTMENT_MASS
        if scenario.ambiguous:
            mass -= self.AMBIGUITY_PENALTY
        if scenario.distractor_chars:
            mass -= min(
                self.MAX_DISTRACTOR_PENALTY,
                scenario.distractor_chars / 1000.0 * self.DISTRACTOR_PENALTY_PER_KCHAR,
            )
        mass = min(0.99, max(0.25, mass))
        departments = list(DEPARTMENT_CRITERIA)
        rest = (1.0 - mass) / (len(departments) - 1)
        return {
            d: (mass if d == scenario.truth.department else rest) for d in departments
        }

    def _urgency_level(self, scenario: Scenario) -> int:
        sigma = 0.55
        if scenario.ambiguous:
            sigma += 0.25
        if scenario.has_negation:
            sigma += 0.40      # literal reading of negation trips generators too
        if scenario.has_mixed_dates:
            sigma += 0.25
        if scenario.distractor_chars:
            sigma += min(0.30, scenario.distractor_chars / 1000.0 * 0.10)
        weights = [
            math.exp(-((lv - scenario.truth.urgency_level) ** 2) / (2 * sigma * sigma))
            for lv in range(len(URGENCY_LEVELS))
        ]
        return self._rng.choices(range(len(URGENCY_LEVELS)), weights=weights, k=1)[0]

    def evaluate(self, scenario: Scenario) -> EvalRecord:
        input_tokens = self._prompt_tokens + self._count_tokens(
            _user_prompt(scenario)
        )

        latency_ms = 0.0
        billed_input = 0
        billed_output = 0
        retries = 0
        parse_error: str | None = None

        # Retry loop, billing every attempt -- same accounting as the live paths.
        for attempt in range(settings.max_schema_retries + 1):
            latency_ms += self._latency_ms()
            billed_input += input_tokens
            if self._rng.random() < self.SCHEMA_FAILURE_PROBABILITY:
                # A truncated or refused object still bills for what it generated.
                billed_output += self._rng.randrange(1, self.OUTPUT_TOKENS)
                retries = attempt + 1
                parse_error = "simulated schema failure (truncated or refused)"
                continue
            billed_output += self.OUTPUT_TOKENS
            parse_error = None
            break

        prediction: TriageDecision | None = None
        if parse_error is None:
            dist = self._department_distribution(scenario)
            department = self._rng.choices(
                list(dist), weights=list(dist.values()), k=1
            )[0]
            level = self._urgency_level(scenario)
            # The generator has no coherence guarantee across fields: it emits an
            # escalation boolean that need not follow from the other two answers.
            # Modelled as the policy applied to its OWN answers, then flipped
            # occasionally -- which is roughly what an inattentive reader does.
            from data.generate_dataset import escalation_rule

            escalate = escalation_rule(department, level)
            if self._rng.random() < 0.05:
                escalate = not escalate
            prediction = TriageDecision(
                department=department,        # type: ignore[arg-type]
                urgency_level=level,          # type: ignore[arg-type]
                escalate_supervisor=escalate,
            )

        return EvalRecord(
            scenario_id=scenario.id,
            evaluator=self._model,
            model=f"{self._model} (simulated)",
            source="simulated",
            prediction=prediction,
            latency_ms=latency_ms,
            input_tokens=billed_input,
            output_tokens=billed_output,
            cost_usd=cost_usd(
                billed_input, billed_output, settings.price_for(self._model)
            ),
            schema_ok=prediction is not None,
            parse_error=parse_error,
            retries=retries,
            # No p_department / p_escalate: the thing being simulated cannot
            # report either. Only the near-useless first-token proxy.
            p_first_token_proxy=min(0.9999, self._rng.betavariate(240.0, 2.0)),
        )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class LLMEvaluator(Evaluator):
    """One structured-output request per scenario, retried on schema failure."""

    def __init__(
        self,
        *,
        mode: LLMMode | None = None,
        model: str | None = None,
        want_logprobs: bool = True,
        seed: int | None = None,
    ) -> None:
        self.model = model or settings.llm_model
        self.name = self.model
        self.want_logprobs = want_logprobs
        self._client: Any = None
        self._sim: _Simulator | None = None

        if mode is None:
            mode = "sim" if settings.simulate_llm else settings.llm_provider
        self.mode: LLMMode = mode

        if self.mode == "sim":
            self._sim = _Simulator(
                seed if seed is not None else settings.seed + 1, self.model
            )
        elif self.mode == "openai":
            if settings.openai_api_key is None:
                raise RuntimeError(
                    "LLM_PROVIDER=openai needs OPENAI_API_KEY. Set SIMULATE_LLM=true "
                    "or pass --llm-mode sim to run offline."
                )
        elif self.mode != "bedrock":
            raise ValueError(f"Unknown LLM mode {self.mode!r}")

    @property
    def provider(self) -> str:
        """Which live provider this evaluator targets, simulated or not."""
        return settings.llm_provider if self.mode == "sim" else self.mode

    # -- lifecycle -------------------------------------------------------

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _openai(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            assert settings.openai_api_key is not None
            self._client = AsyncOpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
                timeout=settings.request_timeout_s,
                max_retries=2,
            )
        return self._client

    def _bedrock(self) -> Any:
        if self._client is None:
            from anthropic import AsyncAnthropicBedrockMantle

            # Credentials come from the standard AWS chain (env vars, shared
            # profile, instance role); only the region is ours to supply.
            self._client = AsyncAnthropicBedrockMantle(
                aws_region=settings.aws_region,
                timeout=settings.request_timeout_s,
                max_retries=2,
            )
        return self._client

    @property
    def _bedrock_model(self) -> str:
        """Bedrock model ids are the first-party id with an `anthropic.` prefix."""
        if self.model.startswith("anthropic."):
            return self.model
        return f"anthropic.{self.model}"

    # -- dispatch --------------------------------------------------------

    async def evaluate(self, scenario: Scenario) -> EvalRecord:
        if self.mode == "sim":
            assert self._sim is not None
            record = self._sim.evaluate(scenario)
            # Sleep the simulated latency so concurrency and rate limiting behave
            # like a real run instead of finishing instantly.
            await asyncio.sleep(record.latency_ms / 1000.0)
            return record

        total_latency_ms = 0.0
        total_input = 0
        total_output = 0
        retries = 0
        last_problem: str | None = None
        resolved_model = self.model
        prediction: TriageDecision | None = None
        proxy: float | None = None

        attempt_fn = (
            self._attempt_openai if self.mode == "openai" else self._attempt_bedrock
        )

        for attempt in range(settings.max_schema_retries + 1):
            try:
                outcome = await attempt_fn(scenario)
            except _AttemptFailed as exc:
                # A billed-but-unusable attempt: keep its cost and latency.
                billed = exc.args[1] if len(exc.args) > 1 else {}
                total_latency_ms += billed.get("latency_ms", 0.0)
                total_input += billed.get("input_tokens", 0)
                total_output += billed.get("output_tokens", 0)
                resolved_model = billed.get("model", resolved_model)
                last_problem = str(exc.args[0])
                retries = attempt + 1
                continue
            except Exception as exc:
                # Transport-level failure after the SDK's own retries: record it.
                return EvalRecord(
                    scenario_id=scenario.id,
                    evaluator=self.name,
                    model=resolved_model,
                    source="live",
                    prediction=None,
                    latency_ms=total_latency_ms or float("nan"),
                    input_tokens=total_input,
                    output_tokens=total_output,
                    cost_usd=self._cost(total_input, total_output),
                    schema_ok=False,
                    retries=retries,
                    error=f"{type(exc).__name__}: {exc}",
                )

            total_latency_ms += outcome["latency_ms"]
            total_input += outcome["input_tokens"]
            total_output += outcome["output_tokens"]
            resolved_model = outcome["model"]
            prediction = outcome["prediction"]
            proxy = outcome.get("p_first_token_proxy")
            break

        return EvalRecord(
            scenario_id=scenario.id,
            evaluator=self.name,
            model=resolved_model,
            source="live",
            prediction=prediction,
            latency_ms=total_latency_ms,
            input_tokens=total_input,
            output_tokens=total_output,
            cost_usd=self._cost(total_input, total_output),
            schema_ok=prediction is not None,
            parse_error=None if prediction is not None else last_problem,
            retries=retries,
            # `p_department` and `p_escalate` stay None: a token generator exposes
            # no distribution over the answer space, so the calibration columns are
            # honestly blank rather than filled with a guess. The first-token
            # logprob goes in its own field -- see `_first_token_probability`.
            p_department=None,
            p_escalate=None,
            urgency_expected=None,
            p_first_token_proxy=proxy,
        )

    def _cost(self, input_tokens: int, output_tokens: int) -> float:
        return cost_usd(input_tokens, output_tokens, settings.price_for(self.model))

    # -- OpenAI ----------------------------------------------------------

    async def _attempt_openai(self, scenario: Scenario) -> dict[str, Any]:
        client = self._openai()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(scenario)},
            ],
            "response_format": TriageDecision,
        }
        if self.want_logprobs:
            kwargs["logprobs"] = True

        with Stopwatch() as sw:
            completion = await client.chat.completions.parse(**kwargs)

        usage = completion.usage
        billed = {
            "latency_ms": sw.elapsed_ms,
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "model": completion.model,
        }

        choice = completion.choices[0]
        message = choice.message

        # The four documented ways this comes back unusable.
        if message.refusal:
            raise _AttemptFailed(f"refusal: {message.refusal}", billed)
        if choice.finish_reason == "length":
            raise _AttemptFailed("truncated: finish_reason=length", billed)
        if message.parsed is None:
            raise _AttemptFailed("parsed is None", billed)
        try:
            prediction = TriageDecision.model_validate(message.parsed)
        except Exception as exc:
            raise _AttemptFailed(f"{type(exc).__name__}: {exc}", billed) from exc

        return {
            **billed,
            "prediction": prediction,
            "p_first_token_proxy": _first_token_probability(choice),
        }

    # -- Bedrock ---------------------------------------------------------

    async def _attempt_bedrock(self, scenario: Scenario) -> dict[str, Any]:
        client = self._bedrock()

        with Stopwatch() as sw:
            message = await client.messages.parse(
                model=self._bedrock_model,
                # 256 is ample for a three-field object and caps the damage if the
                # model starts narrating. `finish_reason=max_tokens` is caught
                # below as a schema failure rather than silently truncating.
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": _user_prompt(scenario)}],
                output_format=TriageDecision,
                # Thinking stays OFF. Thinking tokens bill as output and would
                # more than double this baseline's cost while making the latency
                # comparison meaningless -- and this is a classification task with
                # no reasoning to do. Haiku 4.5 takes no thinking by default, so
                # the parameter is omitted rather than set to "disabled": on that
                # model omission IS off, and `output_config.effort` is rejected.
            )

        billed = {
            "latency_ms": sw.elapsed_ms,
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
            "model": message.model,
        }

        if message.stop_reason == "refusal":
            detail = getattr(message.stop_details, "category", None)
            raise _AttemptFailed(f"refusal: {detail}", billed)
        if message.stop_reason == "max_tokens":
            raise _AttemptFailed("truncated: stop_reason=max_tokens", billed)
        if message.parsed_output is None:
            raise _AttemptFailed("parsed_output is None", billed)
        try:
            prediction = TriageDecision.model_validate(message.parsed_output)
        except Exception as exc:
            raise _AttemptFailed(f"{type(exc).__name__}: {exc}", billed) from exc

        # No logprobs on this surface at all, so not even the proxy is available.
        return {**billed, "prediction": prediction, "p_first_token_proxy": None}

    # -- provenance ------------------------------------------------------

    @property
    def pricing_caveat(self) -> str | None:
        """Non-None when the cost column needs a caveat attached to it."""
        return PARTNER_PRICING_CAVEAT if self.mode == "bedrock" else None


def _first_token_probability(choice: Any) -> float | None:
    """Probability of the first generated token, when logprobs are available.

    A PROXY, not a distribution over departments. The first token of
    `{"department":"billing"...}` is a brace or a fragment of the key, so this
    measures the model's confidence in beginning a JSON object -- which is nearly
    always ~1.0 and says almost nothing about the classification.

    It is recorded because it is the closest thing a token generator offers to
    Jev's per-option probabilities, and the gap between them is one of the
    article's actual points. It is NOT used for the department ECE, which stays
    blank for this evaluator instead of being filled with a number that looks
    like a calibration measurement and is not one.
    """
    logprobs = getattr(choice, "logprobs", None)
    content = getattr(logprobs, "content", None) if logprobs else None
    if not content:
        return None
    import math

    return math.exp(content[0].logprob)
