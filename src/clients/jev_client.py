"""Jev (TypeSafe AI "System One") evaluator, in three interchangeable modes.

All three questions go in ONE `system_one` request. That is both what the docs
describe -- state is ingested once and every question is evaluated against it in
parallel -- and the only fair comparison: one Jev request against one LLM
request per scenario.

    mode="sdk"   official `typesafe-sdk` client       (needs TYPESAFE_API_KEY)
    mode="http"  raw httpx POST /v1/systemone         (needs TYPESAFE_API_KEY)
    mode="sim"   local simulator, no network          (the default)

ABOUT THE SIMULATOR -- please read before believing any number it produces.

The simulator builds a probability distribution first and then decides where the
true answer sits *inside* it, by sampling the truth's rank from the distribution's
own masses (see `_Simulator`). That makes it calibrated by construction: the
probability it reports for the option it picks is exactly the probability that
option is right. So a simulated ECE near zero is a restatement of a dozen lines of
code here. It is a property of this file, not evidence about Jev. It exists so the
pipeline can be exercised offline and so the plotting code has realistically
shaped input -- nothing more.

Latency and cost are likewise parameters, not measurements: latency is drawn from
a log-normal fitted to the vendor's published ~100ms figure, and token counts are
estimated from the payload's length. Only a `mode="sdk"` / `mode="http"` run
measures anything.

Documented behaviour this file deliberately does NOT rely on: Jev makes no
guarantee of structural invariants between answers -- in particular `P(noul)`
need not equal `1 - P(not noul)`. Nothing here derives one probability from
another by subtraction. (https://docs.typesafe.ai/confidence.md)
"""

from __future__ import annotations

import asyncio
import math
import random
from typing import Any, Literal

from src.config import JEV_ENDPOINT, JEV_RETRY_STATUS, settings
from src.clients.base import Evaluator, Stopwatch, cost_usd, token_counter
from src.schemas import (
    DEPARTMENT_CRITERIA,
    DEPARTMENT_INSTRUCTIONS,
    ESCALATE_INSTRUCTIONS,
    URGENCY_INSTRUCTIONS,
    URGENCY_LEVELS,
    EvalRecord,
    Scenario,
    TriageDecision,
)

JevMode = Literal["sdk", "http", "sim"]

#: Question keys, fixed here so the three modes and the article all agree.
Q_DEPARTMENT = "department"
Q_URGENCY = "urgency"
Q_ESCALATE = "escalate"

DEPARTMENTS: list[str] = list(DEPARTMENT_CRITERIA)


def _questions_payload() -> dict[str, dict[str, Any]]:
    """The three questions in the documented wire encoding.

    Shared by the raw-HTTP mode and the token estimator so that the bytes we
    price are the bytes we would send. https://docs.typesafe.ai/api.md
    """
    return {
        Q_DEPARTMENT: {
            "type": "choice",
            "instructions": DEPARTMENT_INSTRUCTIONS,
            "criteria": DEPARTMENT_CRITERIA,
        },
        Q_URGENCY: {
            "type": "score",
            "instructions": URGENCY_INSTRUCTIONS,
            "criteria": URGENCY_LEVELS,
        },
        Q_ESCALATE: {
            "type": "noul",
            "instructions": ESCALATE_INSTRUCTIONS,
        },
    }


def _to_decision(
    department: str, urgency_level: int, escalate: bool
) -> TriageDecision:
    """Validate an answer triple against the shared output contract.

    Raises `pydantic.ValidationError` if the API returns an option outside the
    declared criteria -- which would be a schema violation on Jev's side and must
    be recorded as one, not silently coerced.
    """
    return TriageDecision(
        department=department,       # type: ignore[arg-type]
        urgency_level=urgency_level,  # type: ignore[arg-type]
        escalate_supervisor=escalate,
    )


def peakedness_confidence(probabilities: dict[Any, float]) -> float:
    """Jev's documented `confidence` statistic: (n * peak - 1) / (n - 1).

    Reproduced here only so the simulator can report the same quantity the real
    API does. It measures how concentrated the distribution is, and the docs make
    no claim that it predicts correctness -- so it is never an ECE input.
    https://docs.typesafe.ai/confidence.md
    """
    n = len(probabilities)
    if n < 2:
        return 1.0
    peak = max(probabilities.values())
    return (n * peak - 1.0) / (n - 1.0)


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------


class _Simulator:
    """Distribution-first behavioural model of a System One call.

    WHY THIS IS NOT JUST "A GAUSSIAN ROUND THE TRUTH"

    The obvious way to fake a classifier is to put most of the probability mass on
    the correct answer and share the rest out. That is wrong in a way that is easy
    to miss and that inflates the result: the peak of such a distribution is
    *always* the correct answer, so the reported probability of the chosen option
    silently encodes whether the choice was right. Measuring ECE on it gives a
    large, meaningless error (~0.10 here) driven entirely by that artifact.

    So the construction is inverted. For each question the simulator:

      1. picks a *shape* -- a sorted vector of probability masses whose spread
         comes from the scenario's difficulty knobs, with no reference to truth;
      2. samples which slot in that shape the true answer occupies, drawing the
         slot from the shape itself;
      3. answers with the option in the top slot.

    Step 2 is what makes it honest: the true answer lands in the peak slot with
    probability exactly equal to the peak's mass, so `P(correct) == reported
    probability` by construction, and ECE goes to zero as a matter of arithmetic
    rather than of Jev's behaviour.

    Which is the point, and the reason none of this can be quoted: a simulated ECE
    near zero restates the three steps above. It says nothing whatsoever about the
    real model's calibration.
    """

    # Probability mass placed on the correct department for an easy scenario.
    BASE_DEPARTMENT_MASS = 0.94
    # How much of that mass ambiguity and distractor text take away. Both numbers
    # come from the vendor's own list of weaknesses: ambiguous phrasing and
    # accuracy decay as irrelevant state grows.
    AMBIGUITY_PENALTY = 0.30
    DISTRACTOR_PENALTY_PER_KCHAR = 0.06
    MAX_DISTRACTOR_PENALTY = 0.18

    # Log-normal latency, in milliseconds. mu is log-median, so p50 = exp(mu).
    LATENCY_LOG_MEDIAN = math.log(104.0)
    LATENCY_LOG_SIGMA = 0.26
    # Large states cost real time to ingest; ~1ms per 2k characters.
    LATENCY_MS_PER_KCHAR = 0.5

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._count_tokens = token_counter()
        # The questions are identical for every scenario, so size them once.
        payload = _questions_payload()
        self._question_tokens = self._count_tokens(repr(payload))

    # -- difficulty ------------------------------------------------------

    def _correct_mass(self, scenario: Scenario) -> float:
        mass = self.BASE_DEPARTMENT_MASS
        if scenario.ambiguous:
            mass -= self.AMBIGUITY_PENALTY
        if scenario.distractor_chars:
            mass -= min(
                self.MAX_DISTRACTOR_PENALTY,
                scenario.distractor_chars / 1000.0 * self.DISTRACTOR_PENALTY_PER_KCHAR,
            )
        # Negation and date confusion are documented weaknesses too, but they
        # perturb the *urgency* reading more than ownership, so they are applied
        # in `_urgency_distribution` rather than here.
        return min(0.99, max(0.30, mass))

    # -- per-question distributions --------------------------------------

    def _department_shape(self, scenario: Scenario) -> list[float]:
        """The masses a department answer will be spread over, high to low.

        Built with no reference to the truth: only the difficulty knobs set how
        peaked it is. `_decay_ratio` picks the geometric decay that makes the top
        mass come out at exactly the difficulty-implied peak, which keeps the
        vector strictly descending -- so slot 0 is unambiguously the answer.
        """
        peak = self._correct_mass(scenario)
        ratio = _decay_ratio(peak, len(DEPARTMENTS))
        raw = [ratio**k for k in range(len(DEPARTMENTS))]
        total = sum(raw)
        return [value / total for value in raw]

    def _department_answer(self, scenario: Scenario) -> tuple[dict[str, float], str]:
        """Return the reported distribution and the option the simulator picks.

        The truth's *slot* in the shape is sampled from the shape itself, so the
        top slot holds the correct department with probability equal to its own
        mass. The remaining departments are shuffled into the remaining slots,
        which is what stops the runner-up ordering from carrying signal.
        """
        shape = self._department_shape(scenario)
        slot = self._rng.choices(range(len(shape)), weights=shape, k=1)[0]

        holders = [d for d in DEPARTMENTS if d != scenario.truth.department]
        self._rng.shuffle(holders)
        holders.insert(slot, scenario.truth.department)

        dist = _normalise(dict(zip(holders, shape)))
        return dist, holders[0]

    def _urgency_sigma(self, scenario: Scenario) -> float:
        """Spread of the urgency reading, widened by the difficulty knobs.

        Negation and mixed dates widen it because both change how time pressure
        reads without changing who owns the ticket.
        """
        sigma = 0.42
        if scenario.ambiguous:
            sigma += 0.25
        if scenario.has_negation:
            sigma += 0.30
        if scenario.has_mixed_dates:
            sigma += 0.20
        if scenario.distractor_chars:
            sigma += min(0.25, scenario.distractor_chars / 1000.0 * 0.08)
        return sigma

    def _urgency_answer(self, scenario: Scenario) -> tuple[dict[int, float], int]:
        """Return the reported urgency distribution and the level picked.

        Two distributions, and the distinction is the whole point. The first is
        the *error model*: centred on the true level, it decides how far off the
        reading lands. The second is what gets reported: centred on the level
        actually read, because a system reporting a distribution has no access to
        the truth to centre it on. Reporting the first would leak the answer, the
        same mistake `_Simulator` describes for the department question.
        """
        sigma = self._urgency_sigma(scenario)
        levels = range(len(URGENCY_LEVELS))

        def bell(centre: float) -> dict[int, float]:
            return _normalise(
                {
                    level: math.exp(-((level - centre) ** 2) / (2 * sigma * sigma))
                    for level in levels
                }
            )

        error_model = bell(scenario.truth.urgency_level)
        level = self._rng.choices(
            list(error_model), weights=list(error_model.values()), k=1
        )[0]
        return bell(level), level

    def _escalate_probability(
        self,
        dept_dist: dict[str, float],
        urgency_dist: dict[int, float],
    ) -> float:
        """P(escalate), marginalised over the two *reported* distributions.

        Reported, not truth-centred: these are the same distributions that go into
        the record, so the escalation belief is a consequence of the other two
        beliefs rather than a private channel to the answer. Marginalising over
        truth-centred distributions instead would pin escalation accuracy at
        exactly 1.0, which is how the earlier version of this file gave itself
        away.

        Deriving it from the policy at all is still an advantage no real system
        has, and it is worth stating plainly: this imports the generator's own
        `escalation_rule`, so it knows the policy exactly and only has to be
        uncertain about the inputs. A real evaluator has to infer the policy from
        the instruction string. The coherence between the three answers is also
        our design choice -- the real API guarantees no such invariant.
        """
        from data.generate_dataset import escalation_rule

        return sum(
            p_d * p_u
            for dept, p_d in dept_dist.items()
            for level, p_u in urgency_dist.items()
            if escalation_rule(dept, level)
        )

    # -- the call --------------------------------------------------------

    def evaluate(self, scenario: Scenario, model: str) -> EvalRecord:
        dept_dist, department = self._department_answer(scenario)
        urgency_dist, level = self._urgency_answer(scenario)

        p_escalate = self._escalate_probability(dept_dist, urgency_dist)
        escalate = p_escalate >= 0.5

        expected_level = sum(lv * p for lv, p in urgency_dist.items())

        latency_ms = self._rng.lognormvariate(
            self.LATENCY_LOG_MEDIAN, self.LATENCY_LOG_SIGMA
        ) + len(scenario.ticket) / 1000.0 * self.LATENCY_MS_PER_KCHAR

        input_tokens = self._count_tokens(scenario.ticket) + self._question_tokens
        price = settings.price_for("jev")

        return EvalRecord(
            scenario_id=scenario.id,
            evaluator="jev",
            model=f"{model} (simulated)",
            source="simulated",
            prediction=_to_decision(department, level, escalate),
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=0,          # documented as free, and structurally tiny
            cost_usd=cost_usd(input_tokens, 0, price),
            p_department=dept_dist[department],
            p_escalate=p_escalate,
            urgency_expected=expected_level,
            confidence_department=peakedness_confidence(dept_dist),
            confidence_urgency=peakedness_confidence(urgency_dist),
        )


def _decay_ratio(peak: float, n: int, *, tolerance: float = 1e-12) -> float:
    """Find r in [0, 1) such that 1 / (1 + r + ... + r^(n-1)) == `peak`.

    In other words: the geometric decay whose normalised first term is the mass we
    want on the top option. r -> 0 gives a one-hot distribution, r -> 1 a uniform
    one, so `peak` must be above 1/n for a solution to exist -- which
    `_correct_mass` guarantees by clamping. Solved by bisection because the sum is
    monotone in r and n is 4; a closed form would buy nothing.
    """
    if not 1.0 / n < peak <= 1.0:
        raise ValueError(f"peak {peak} outside (1/{n}, 1]")

    def top_mass(r: float) -> float:
        return 1.0 / sum(r**k for k in range(n))

    lo, hi = 0.0, 1.0
    while hi - lo > tolerance:
        mid = (lo + hi) / 2.0
        # top_mass decreases as r grows, so a too-large mass means r is too small.
        if top_mass(mid) > peak:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _normalise(weights: dict[Any, float]) -> dict[Any, float]:
    total = sum(weights.values())
    if total <= 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------


class JevEvaluator(Evaluator):
    """One `system_one` request per scenario, in whichever mode is configured."""

    name = "jev"

    def __init__(
        self,
        *,
        mode: JevMode | None = None,
        model: str | None = None,
        seed: int | None = None,
    ) -> None:
        self.model = model or settings.jev_model
        if mode is None:
            mode = "sim" if settings.simulate_jev else "sdk"
        self.mode: JevMode = mode

        self._sim: _Simulator | None = None
        self._sdk_client: Any = None
        self._http_client: Any = None

        if self.mode == "sim":
            self._sim = _Simulator(seed if seed is not None else settings.seed)
        elif settings.typesafe_api_key is None:
            raise RuntimeError(
                f"JevEvaluator(mode={self.mode!r}) needs TYPESAFE_API_KEY. "
                "Set SIMULATE_JEV=true to run offline instead."
            )

    # -- lifecycle -------------------------------------------------------

    async def aclose(self) -> None:
        if self._sdk_client is not None:
            await self._sdk_client.aclose()
            self._sdk_client = None
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    def _sdk(self) -> Any:
        if self._sdk_client is None:
            from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

            assert settings.typesafe_api_key is not None
            self._sdk_client = AsyncTypeSafeClient(
                api_key=settings.typesafe_api_key.get_secret_value(),
                model=self.model,
                # The docs prescribe exponential backoff on 429 and 529. Retries
                # inflate the measured latency of the requests that hit them,
                # which is the honest thing for a p99 to show.
                retry=RetryPolicy(max_retries=3),
                timeout=settings.request_timeout_s,
            )
        return self._sdk_client

    def _http(self) -> Any:
        if self._http_client is None:
            import httpx

            assert settings.typesafe_api_key is not None
            self._http_client = httpx.AsyncClient(
                headers={
                    "Authorization": (
                        f"Bearer {settings.typesafe_api_key.get_secret_value()}"
                    ),
                    "Content-Type": "application/json",
                },
                timeout=settings.request_timeout_s,
            )
        return self._http_client

    # -- dispatch --------------------------------------------------------

    async def evaluate(self, scenario: Scenario) -> EvalRecord:
        try:
            if self.mode == "sim":
                assert self._sim is not None
                # Sleep the simulated latency so concurrency, rate limiting, and
                # progress reporting behave like a real run rather than
                # completing instantly.
                record = self._sim.evaluate(scenario, self.model)
                await asyncio.sleep(record.latency_ms / 1000.0)
                return record
            if self.mode == "sdk":
                return await self._evaluate_sdk(scenario)
            return await self._evaluate_http(scenario)
        except Exception as exc:  # a failed request is data, not a crash
            return EvalRecord(
                scenario_id=scenario.id,
                evaluator=self.name,
                model=self.model,
                source="simulated" if self.mode == "sim" else "live",
                prediction=None,
                latency_ms=float("nan"),
                schema_ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    # -- live modes ------------------------------------------------------

    async def _evaluate_sdk(self, scenario: Scenario) -> EvalRecord:
        from typesafe_sdk import Choice, Noul, Score

        client = self._sdk()
        questions = {
            Q_DEPARTMENT: Choice(
                instructions=DEPARTMENT_INSTRUCTIONS, criteria=DEPARTMENT_CRITERIA
            ),
            Q_URGENCY: Score(
                instructions=URGENCY_INSTRUCTIONS, criteria=URGENCY_LEVELS
            ),
            Q_ESCALATE: Noul(instructions=ESCALATE_INSTRUCTIONS),
        }

        with Stopwatch() as sw:
            response = await client.system_one(scenario.ticket, questions)

        dept = response.answers[Q_DEPARTMENT]
        urgency = response.answers[Q_URGENCY]
        escalate = response.answers[Q_ESCALATE]

        return self._record_from_answers(
            scenario=scenario,
            model=response.model,
            latency_ms=sw.elapsed_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            choice=dept.choice,
            choice_probabilities=dict(dept.probabilities),
            choice_confidence=dept.confidence,
            score=urgency.score,
            score_confidence=urgency.confidence,
            noul=escalate.noul,
        )

    async def _evaluate_http(self, scenario: Scenario) -> EvalRecord:
        """Raw-HTTP fallback, for when the SDK is unavailable or pinned wrong.

        Uses the SDK's own response model to parse, so the two live paths cannot
        drift in how they interpret a payload.
        """
        import httpx
        from typesafe_sdk import SystemOneResponse

        client = self._http()
        body = {
            "model": self.model,
            "state": scenario.ticket,
            "questions": _questions_payload(),
        }

        backoff = 0.5
        last_exc: Exception | None = None
        for attempt in range(4):
            with Stopwatch() as sw:
                try:
                    http_response = await client.post(JEV_ENDPOINT, json=body)
                except httpx.HTTPError as exc:
                    last_exc = exc
                    http_response = None

            if http_response is not None:
                if http_response.status_code not in JEV_RETRY_STATUS:
                    http_response.raise_for_status()
                    parsed = SystemOneResponse.model_validate(http_response.json())
                    dept = parsed.answers[Q_DEPARTMENT]
                    urgency = parsed.answers[Q_URGENCY]
                    escalate = parsed.answers[Q_ESCALATE]
                    return self._record_from_answers(
                        scenario=scenario,
                        model=parsed.model,
                        latency_ms=sw.elapsed_ms,
                        input_tokens=parsed.usage.input_tokens,
                        output_tokens=parsed.usage.output_tokens,
                        choice=dept.choice,
                        choice_probabilities=dict(dept.probabilities),
                        choice_confidence=dept.confidence,
                        score=urgency.score,
                        score_confidence=urgency.confidence,
                        noul=escalate.noul,
                    )
                last_exc = RuntimeError(
                    f"HTTP {http_response.status_code} from {JEV_ENDPOINT}"
                )

            if attempt < 3:
                # Documented guidance for 429/529 is exponential backoff; honour
                # Retry-After when the server sends one.
                delay = backoff
                if http_response is not None:
                    retry_after = http_response.headers.get("retry-after")
                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            pass
                await asyncio.sleep(delay)
                backoff = min(backoff * 2, 5.0)

        raise last_exc or RuntimeError("Jev HTTP request failed")

    # -- shared mapping --------------------------------------------------

    def _record_from_answers(
        self,
        *,
        scenario: Scenario,
        model: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        choice: str,
        choice_probabilities: dict[str, float],
        choice_confidence: float,
        score: float,
        score_confidence: float,
        noul: float,
    ) -> EvalRecord:
        """Map a live API response onto the shared `EvalRecord`.

        Three decisions worth naming:
        * `score` is the probability-weighted mean over rubric indices, so the
          discrete level is `round(score)` -- and the raw mean is kept as
          `urgency_expected` rather than thrown away.
        * `noul` is a float in [0, 1], not a bool. The 0.5 cut is ours, stated
          openly, not something the API returns.
        * An off-contract `choice` is a schema violation and is recorded as one.
        """
        price = settings.price_for("jev")
        level = min(len(URGENCY_LEVELS) - 1, max(0, round(score)))

        try:
            prediction = _to_decision(choice, level, noul >= 0.5)
            schema_ok, parse_error = True, None
        except Exception as exc:
            prediction, schema_ok = None, False
            parse_error = f"{type(exc).__name__}: {exc}"

        return EvalRecord(
            scenario_id=scenario.id,
            evaluator=self.name,
            model=model,
            source="live",
            prediction=prediction,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd(input_tokens, output_tokens, price),
            schema_ok=schema_ok,
            parse_error=parse_error,
            p_department=choice_probabilities.get(choice),
            p_escalate=noul,
            urgency_expected=score,
            confidence_department=choice_confidence,
            confidence_urgency=score_confidence,
        )
