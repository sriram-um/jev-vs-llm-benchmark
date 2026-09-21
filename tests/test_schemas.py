"""Tests for the shared contract, the dataset, and simulator determinism.

The contract tests matter because both evaluators are held to the same Pydantic
model: if `TriageDecision` silently accepted an out-of-range urgency level, a
schema violation on one side would be scored as a success.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from data.generate_dataset import escalation_rule, generate, write_dataset
from src.clients.jev_client import JevEvaluator, peakedness_confidence
from src.clients.llm_client import LLMEvaluator
from src.config import settings
from src.schemas import (
    DEPARTMENT_CRITERIA,
    URGENCY_DISPLAY_SCALE,
    URGENCY_LEVELS,
    EvalRecord,
    Scenario,
    TriageDecision,
)


# ---------------------------------------------------------------------------
# The output contract
# ---------------------------------------------------------------------------


def test_triage_decision_round_trips_through_json():
    original = TriageDecision(
        department="account_security", urgency_level=4, escalate_supervisor=True
    )
    restored = TriageDecision.model_validate_json(original.model_dump_json())
    assert restored == original
    assert restored.urgency_0_100 == 100


def test_urgency_display_scale_spans_zero_to_one_hundred():
    """The 0-100 presentation scale must actually reach 100, or MAE is skewed."""
    assert (len(URGENCY_LEVELS) - 1) * URGENCY_DISPLAY_SCALE == 100


@pytest.mark.parametrize("level", [-1, 5, 100])
def test_triage_decision_rejects_out_of_range_urgency(level):
    with pytest.raises(ValidationError):
        TriageDecision(
            department="billing", urgency_level=level, escalate_supervisor=False
        )


def test_triage_decision_rejects_unknown_department():
    with pytest.raises(ValidationError):
        TriageDecision(
            department="marketing", urgency_level=1, escalate_supervisor=False
        )


def test_triage_decision_forbids_extra_fields():
    """extra='forbid' is what makes `additionalProperties: false` reach the API."""
    with pytest.raises(ValidationError):
        TriageDecision.model_validate(
            {
                "department": "billing",
                "urgency_level": 1,
                "escalate_supervisor": False,
                "commentary": "let me explain",
            }
        )


def test_json_schema_is_strict_mode_compatible():
    """Structured Outputs' strict mode drops `minimum`/`maximum` but keeps enums.

    If someone changes `urgency_level` to an int with bounds, the constraint would
    be silently discarded for the LLM and still enforced for Jev.
    """
    schema = TriageDecision.model_json_schema()
    props = schema["properties"]
    assert schema.get("additionalProperties") is False
    assert set(schema["required"]) == set(props)
    assert "minimum" not in props["urgency_level"]
    assert "maximum" not in props["urgency_level"]
    assert props["urgency_level"]["enum"] == [0, 1, 2, 3, 4]


def test_both_evaluators_share_one_department_vocabulary():
    """The Choice criteria and the JSON enum must not drift apart."""
    enum_values = TriageDecision.model_json_schema()["properties"]["department"]["enum"]
    assert set(enum_values) == set(DEPARTMENT_CRITERIA)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def test_dataset_generation_is_reproducible(tmp_path):
    a = write_dataset(generate(50, seed=7), tmp_path / "a.jsonl")
    b = write_dataset(generate(50, seed=7), tmp_path / "b.jsonl")
    assert a == b


def test_committed_dataset_matches_its_manifest():
    """Guards against a dataset edited without regenerating the manifest."""
    import json

    from data.generate_dataset import dataset_sha256, load_dataset

    manifest = json.loads((settings.data_dir / "manifest.json").read_text())
    assert dataset_sha256(settings.scenarios_path) == manifest["sha256"]
    assert len(load_dataset(settings.scenarios_path)) == manifest["n"]


def test_ground_truth_escalation_is_self_consistent():
    """Every label in the dataset must satisfy the stated escalation policy."""
    from data.generate_dataset import load_dataset

    for scenario in load_dataset(settings.scenarios_path):
        expected = escalation_rule(
            scenario.truth.department, scenario.truth.urgency_level
        )
        assert scenario.truth.escalate_supervisor == expected, scenario.id


def test_security_escalates_one_level_earlier():
    assert escalation_rule("account_security", 2) is True
    assert escalation_rule("billing", 2) is False
    assert escalation_rule("billing", 3) is True


# ---------------------------------------------------------------------------
# Simulators
# ---------------------------------------------------------------------------


def _scenario(**kw) -> Scenario:
    base = dict(
        id="tkt-0001",
        ticket="The eu-west-1 API endpoint has been returning 503s since 02:15 UTC.",
        truth=TriageDecision(
            department="infrastructure_outage",
            urgency_level=4,
            escalate_supervisor=True,
        ),
    )
    base.update(kw)
    return Scenario(**base)


def _run(evaluator, scenario) -> EvalRecord:
    async def go():
        async with evaluator:
            return await evaluator.evaluate(scenario)

    return asyncio.run(go())


@pytest.mark.parametrize(
    "factory",
    [
        lambda seed: JevEvaluator(mode="sim", seed=seed),
        lambda seed: LLMEvaluator(mode="sim", seed=seed),
    ],
    ids=["jev", "llm"],
)
def test_simulators_are_deterministic_under_a_fixed_seed(factory):
    scenario = _scenario()
    first = _run(factory(1234), scenario)
    second = _run(factory(1234), scenario)
    assert first.model_dump() == second.model_dump()


@pytest.mark.parametrize(
    "factory",
    [
        lambda seed: JevEvaluator(mode="sim", seed=seed),
        lambda seed: LLMEvaluator(mode="sim", seed=seed),
    ],
    ids=["jev", "llm"],
)
def test_simulated_records_are_labelled_simulated(factory):
    record = _run(factory(1), _scenario())
    assert record.source == "simulated"
    assert "simulated" in record.model


def test_jev_simulator_reports_a_real_distribution():
    record = _run(JevEvaluator(mode="sim", seed=11), _scenario())
    assert record.p_department is not None and 0.0 < record.p_department <= 1.0
    assert record.p_escalate is not None and 0.0 <= record.p_escalate <= 1.0
    assert record.urgency_expected is not None
    # The peakedness statistic is recorded, and is NOT the probability.
    assert record.confidence_department is not None
    assert record.confidence_department != record.p_department


def test_llm_simulator_exposes_no_distribution_only_the_proxy():
    """The asymmetry the article is about, asserted so it cannot regress."""
    record = _run(LLMEvaluator(mode="sim", seed=11), _scenario())
    assert record.p_department is None
    assert record.p_escalate is None
    assert record.urgency_expected is None
    assert record.p_first_token_proxy is not None
    # Near 1.0 and uninformative: it is confidence about emitting a brace.
    assert record.p_first_token_proxy > 0.9


def test_jev_simulator_degrades_on_ambiguity():
    """Less probability mass on the truth when the department signal is mixed."""
    easy = _run(JevEvaluator(mode="sim", seed=5), _scenario())
    hard = _run(JevEvaluator(mode="sim", seed=5), _scenario(ambiguous=True))
    assert hard.p_department is not None and easy.p_department is not None
    assert hard.p_department < easy.p_department


def test_jev_simulator_is_calibrated_by_construction():
    """The claim the article rests a caveat on, asserted so it cannot rot.

    Over many draws on one scenario, the fraction of correct department answers
    must match the probability the simulator reports for its own pick. This is a
    property of the rank-sampling construction, not a finding -- but if someone
    reverts to centring the distribution on the truth, accuracy jumps to 1.0
    while the reported probability stays put, and this fails.
    """
    from src.clients.jev_client import _Simulator

    sim = _Simulator(seed=99)
    scenario = _scenario(ambiguous=True)   # peak ~0.64, far from 1.0
    records = [sim.evaluate(scenario, "jev-latest") for _ in range(4000)]

    reported = {r.p_department for r in records}
    assert len(reported) == 1, "one difficulty should imply one reported mass"
    claimed = reported.pop()
    observed = sum(
        r.prediction.department == scenario.truth.department for r in records
    ) / len(records)
    assert observed == pytest.approx(claimed, abs=0.02)


def test_jev_simulator_does_not_leak_truth_into_escalation():
    """Truth-centred distributions made escalate_accuracy exactly 1.0."""
    from src.clients.jev_client import _Simulator

    sim = _Simulator(seed=99)
    scenario = _scenario(ambiguous=True)
    records = [sim.evaluate(scenario, "jev-latest") for _ in range(400)]
    assert len({r.prediction.escalate_supervisor for r in records}) == 2


def test_decay_ratio_hits_the_requested_peak():
    """n=4, peak=0.4: masses 1, r, r^2, r^3 normalised must start at 0.4."""
    from src.clients.jev_client import _decay_ratio

    r = _decay_ratio(0.4, 4)
    assert 1.0 / sum(r**k for k in range(4)) == pytest.approx(0.4)
    # A uniform distribution is the boundary, not a legal peak.
    with pytest.raises(ValueError):
        _decay_ratio(0.25, 4)


def test_simulated_jev_costs_less_than_simulated_llm():
    """Not a finding -- a sanity check that the price table is wired up at all."""
    scenario = _scenario()
    jev = _run(JevEvaluator(mode="sim", seed=3), scenario)
    llm = _run(LLMEvaluator(mode="sim", seed=3), scenario)
    assert 0 < jev.cost_usd < llm.cost_usd


def test_jev_output_tokens_are_free():
    record = _run(JevEvaluator(mode="sim", seed=3), _scenario())
    price = settings.price_for("jev")
    assert price["output"] == 0.0
    assert record.cost_usd == pytest.approx(record.input_tokens / 1e6 * price["input"])


# ---------------------------------------------------------------------------
# Confidence statistic and pricing
# ---------------------------------------------------------------------------


def test_peakedness_confidence_matches_the_documented_formula():
    """(n * peak - 1) / (n - 1): 1.0 when one-hot, 0.0 when uniform."""
    assert peakedness_confidence({"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0}) == 1.0
    assert peakedness_confidence({k: 0.25 for k in "abcd"}) == pytest.approx(0.0)
    # n=4, peak=0.5 -> (4*0.5 - 1) / 3 = 1/3
    assert peakedness_confidence(
        {"a": 0.5, "b": 0.2, "c": 0.2, "d": 0.1}
    ) == pytest.approx(1 / 3)


def test_price_lookup_strips_the_bedrock_prefix():
    assert settings.price_for("anthropic.claude-haiku-4-5") == settings.price_for(
        "claude-haiku-4-5"
    )


def test_price_lookup_prefers_the_longest_matching_prefix():
    """A dated snapshot must not fall through to a shorter, cheaper row."""
    assert settings.price_for("claude-haiku-4-5-20251001")["input"] == 1.0


def test_price_lookup_refuses_to_guess():
    with pytest.raises(KeyError):
        settings.price_for("some-unpriced-model")
