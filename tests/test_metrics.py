"""Tests for `src/metrics.py`.

The point of these is that every headline number in the article comes out of this
module, so each one is checked against a value computed by hand in the docstring
rather than against whatever the code happened to return when it was written.
"""

from __future__ import annotations

import math

import pytest

from src import metrics
from src.schemas import EvalRecord, TriageDecision


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------


def test_percentile_is_nearest_rank_not_interpolated():
    """On 1..10, p50 is the 5th observed value, not the 5.5 numpy would give."""
    values = list(range(1, 11))
    assert metrics.percentile(values, 50) == 5
    assert metrics.percentile(values, 95) == 10
    assert metrics.percentile(values, 99) == 10


def test_percentile_edges():
    values = [5.0, 1.0, 3.0]          # unsorted on purpose
    assert metrics.percentile(values, 0) == 1.0
    assert metrics.percentile(values, 100) == 5.0
    assert metrics.percentile(values, 50) == 3.0
    assert metrics.percentile([42.0], 99) == 42.0
    assert math.isnan(metrics.percentile([], 50))


def test_percentile_rejects_out_of_range_q():
    with pytest.raises(ValueError):
        metrics.percentile([1.0], 101)


def test_p99_of_500_is_the_495th_value():
    """ceil(0.99 * 500) = 495, so p99 is the 495th smallest -- a real request."""
    values = [float(i) for i in range(1, 501)]
    assert metrics.percentile(values, 99) == 495.0


def test_throughput_is_inverse_mean_latency():
    # 100ms mean -> 10 requests/second sequential-equivalent.
    assert metrics.throughput_rps([100.0, 100.0]) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# ECE -- the one the article leans on hardest
# ---------------------------------------------------------------------------


def test_ece_matches_hand_computed():
    """Two equal-width bins over four points.

    bin [0.0, 0.5):  probs 0.10, 0.35  -> conf 0.225, acc 0.0, |gap| 0.225
    bin [0.5, 1.0]:  probs 0.65, 0.90  -> conf 0.775, acc 1.0, |gap| 0.225
    Each bin holds half the points, so ECE = 0.5*0.225 + 0.5*0.225 = 0.225.
    """
    probs = [0.10, 0.35, 0.65, 0.90]
    correct = [False, False, True, True]

    score, table = metrics.ece(probs, correct, n_bins=2)

    assert score == pytest.approx(0.225)
    assert [row["n"] for row in table] == [2.0, 2.0]
    assert table[0]["mean_confidence"] == pytest.approx(0.225)
    assert table[0]["accuracy"] == pytest.approx(0.0)
    assert table[1]["mean_confidence"] == pytest.approx(0.775)
    assert table[1]["accuracy"] == pytest.approx(1.0)


def test_ece_is_zero_for_a_perfectly_calibrated_set():
    """Half the 0.5-confidence cases correct is perfect calibration, not 50% error."""
    probs = [0.5] * 4
    correct = [True, False, True, False]
    score, _ = metrics.ece(probs, correct, n_bins=2)
    assert score == pytest.approx(0.0)


def test_ece_last_bin_is_closed_so_p_equals_one_is_counted():
    score, table = metrics.ece([1.0], [True], n_bins=10)
    assert table[-1]["n"] == 1.0
    assert score == pytest.approx(0.0)
    assert sum(row["n"] for row in table) == 1.0


def test_ece_keeps_empty_bins_so_the_plot_can_show_gaps():
    _, table = metrics.ece([0.05, 0.95], [False, True], n_bins=10)
    assert len(table) == 10
    assert table[0]["n"] == 1.0
    assert table[-1]["n"] == 1.0
    assert table[5]["n"] == 0.0
    assert math.isnan(table[5]["accuracy"])


def test_ece_rejects_mismatched_lengths_and_too_few_bins():
    with pytest.raises(ValueError):
        metrics.ece([0.5, 0.5], [True], n_bins=2)
    with pytest.raises(ValueError):
        metrics.ece([0.5], [True], n_bins=1)


def test_brier_hand_computed():
    """((0.9-1)^2 + (0.2-0)^2) / 2 = (0.01 + 0.04) / 2 = 0.025."""
    assert metrics.brier([0.9, 0.2], [True, False]) == pytest.approx(0.025)


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def test_accuracy_counts_a_missing_prediction_as_wrong():
    """`None` never equals a label, so a non-answer costs accuracy."""
    pairs = [("a", "a"), (None, "b"), ("c", "c"), ("a", "b")]
    assert metrics.accuracy(pairs) == pytest.approx(0.5)


def test_macro_f1_hand_computed():
    """Labels x, y. Predictions: x,x,y,y  Truth: x,y,y,y

    x: tp=1 fp=1 fn=0 -> F1 = 2/(2+1+0) = 0.6666...
    y: tp=2 fp=0 fn=1 -> F1 = 4/(4+0+1) = 0.8
    macro = (0.666... + 0.8) / 2 = 0.7333...
    """
    pairs = [("x", "x"), ("x", "y"), ("y", "y"), ("y", "y")]
    assert metrics.macro_f1(pairs, ["x", "y"]) == pytest.approx(0.7333333, abs=1e-6)


def test_macro_f1_scores_an_absent_label_zero_not_nan():
    """A label the system never predicts drags macro-F1 down, as it should."""
    pairs = [("x", "x"), ("x", "z")]
    assert metrics.macro_f1(pairs, ["x", "y", "z"]) == pytest.approx(2 / 3 / 3)


def test_mae_and_rmse():
    errors = [-3.0, 4.0]
    assert metrics.mae(errors) == pytest.approx(3.5)
    assert metrics.rmse(errors) == pytest.approx(math.sqrt(12.5))


# ---------------------------------------------------------------------------
# Records -> metrics
# ---------------------------------------------------------------------------


def _record(**kw) -> EvalRecord:
    base = dict(
        scenario_id="tkt-0001",
        evaluator="jev",
        model="test",
        source="simulated",
        latency_ms=100.0,
    )
    base.update(kw)
    return EvalRecord(**base)


def test_schema_conformance_and_cost_totals():
    records = [
        _record(schema_ok=True, input_tokens=100, cost_usd=0.001),
        _record(schema_ok=False, input_tokens=100, output_tokens=10, cost_usd=0.002),
    ]
    assert metrics.schema_conformance(records) == pytest.approx(0.5)

    totals = metrics.cost_totals(records)
    assert totals["total_input_tokens"] == 200
    assert totals["total_output_tokens"] == 10
    assert totals["total_cost_usd"] == pytest.approx(0.003)
    # 0.003 over 2 decisions -> $1.50 per 1000 decisions.
    assert totals["cost_per_1k_decisions_usd"] == pytest.approx(1.5)


def test_urgency_errors_charge_full_scale_for_a_missing_answer():
    """Failing to answer must not improve MAE by shrinking the denominator."""
    prediction = TriageDecision(
        department="billing", urgency_level=2, escalate_supervisor=False
    )
    records = [
        _record(scenario_id="a", prediction=prediction),
        _record(scenario_id="b", prediction=None),
    ]
    errors = metrics.urgency_errors(records, {"a": 2, "b": 3})
    assert errors == [0.0, 100.0]


def test_urgency_errors_use_expected_reads_the_continuous_score():
    """level 2 truth = 50 on the display scale; expected 2.4 -> 60, error +10."""
    records = [_record(scenario_id="a", urgency_expected=2.4)]
    errors = metrics.urgency_errors(records, {"a": 2}, use_expected=True)
    assert errors == pytest.approx([10.0])


def test_urgency_errors_use_expected_skips_records_without_one():
    """The LLM has no continuous score; it must not be imputed as zero error."""
    records = [_record(scenario_id="a", urgency_expected=None)]
    assert metrics.urgency_errors(records, {"a": 2}, use_expected=True) == []
