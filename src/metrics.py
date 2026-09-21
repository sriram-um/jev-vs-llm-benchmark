"""Metric functions over `list[EvalRecord]`.

Pure and dependency-light on purpose: everything here is a function of its
arguments, so the numbers in the article can be re-derived from
`results/records.jsonl` alone, and every one of them is unit-testable without a
network or an API key.

Two conventions worth stating once, because they are the kind of detail that
quietly changes a headline number:

* **Percentiles are nearest-rank**, not interpolated. `p99` of 500 samples is a
  real observed request (the 495th of 500), not a weighted average of two
  requests that never happened. numpy's default would interpolate.
* **ECE keeps its bins.** `ece()` returns the scalar *and* the per-bin table it
  was computed from, so the reliability figure and the ECE cell in the article
  cannot disagree -- they read the same object.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from src.schemas import URGENCY_DISPLAY_SCALE, EvalRecord

# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile. `q` in [0, 100].

    Returns an actually-observed value. `percentile(xs, 50)` on an even-length
    list is the upper of the two middle elements, which is the convention the
    latency tables in the article use.
    """
    if not values:
        return float("nan")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    ordered = sorted(values)
    if q == 0.0:
        return ordered[0]
    rank = math.ceil(q / 100.0 * len(ordered))
    return ordered[max(0, rank - 1)]


def percentiles(values: Sequence[float]) -> dict[str, float]:
    """The three latency percentiles the article reports, plus the mean."""
    return {
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "mean_ms": (sum(values) / len(values)) if values else float("nan"),
    }


def throughput_rps(values: Sequence[float]) -> float:
    """Sequential-equivalent throughput: 1 / mean latency.

    Deliberately NOT wall-clock throughput of the benchmark run, which measures
    our own concurrency setting rather than either system under test.
    """
    if not values:
        return float("nan")
    mean_s = sum(values) / len(values) / 1000.0
    return 1.0 / mean_s if mean_s > 0 else float("inf")


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def accuracy(pairs: Iterable[tuple[object, object]]) -> float:
    """Fraction of (predicted, truth) pairs that agree.

    A record with no prediction (hard error, unparseable output) must be passed
    in with `predicted=None`, which never equals a truth value -- a system that
    fails to answer scores 0 for that case rather than being dropped from the
    denominator.
    """
    items = list(pairs)
    if not items:
        return float("nan")
    return sum(1 for pred, truth in items if pred == truth) / len(items)


def macro_f1(pairs: Iterable[tuple[object, object]], labels: Sequence[object]) -> float:
    """Unweighted mean of per-label F1, over `labels`.

    Macro rather than micro because the department classes are not balanced and
    we do not want a large class to mask a collapsed small one. A label with no
    predictions and no truths contributes F1 = 0.
    """
    items = list(pairs)
    if not items or not labels:
        return float("nan")

    scores: list[float] = []
    for label in labels:
        tp = sum(1 for p, t in items if p == label and t == label)
        fp = sum(1 for p, t in items if p == label and t != label)
        fn = sum(1 for p, t in items if p != label and t == label)
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return sum(scores) / len(scores)


def mae(errors: Sequence[float]) -> float:
    if not errors:
        return float("nan")
    return sum(abs(e) for e in errors) / len(errors)


def rmse(errors: Sequence[float]) -> float:
    if not errors:
        return float("nan")
    return math.sqrt(sum(e * e for e in errors) / len(errors))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def ece(
    probs: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> tuple[float, list[dict[str, float]]]:
    """Expected calibration error with equal-width bins, plus the bin table.

    ECE = sum_b (n_b / N) * |accuracy_b - mean_confidence_b|.

    Bins are equal-width over [0, 1] and half-open, `[lo, hi)`, except the last
    which is closed so that p = 1.0 lands in it. Empty bins contribute nothing to
    the sum but are still returned, so the reliability plot shows the gap instead
    of silently interpolating across it.

    `probs` must be the model's probability for the outcome recorded in
    `correct`. Passing a peakedness or self-report score here produces a number
    that looks like ECE and means nothing -- see `confidence_department` in
    `src/schemas.py`.
    """
    if len(probs) != len(correct):
        raise ValueError(f"length mismatch: {len(probs)} probs, {len(correct)} labels")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")

    edges = [i / n_bins for i in range(n_bins + 1)]
    table: list[dict[str, float]] = []
    total = len(probs)
    score = 0.0

    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            members = [(p, c) for p, c in zip(probs, correct) if lo <= p <= hi]
        else:
            members = [(p, c) for p, c in zip(probs, correct) if lo <= p < hi]

        n = len(members)
        if n:
            conf = sum(p for p, _ in members) / n
            acc = sum(1 for _, c in members if c) / n
            score += (n / total) * abs(acc - conf)
        else:
            conf = float("nan")
            acc = float("nan")

        table.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "n": float(n),
                "mean_confidence": conf,
                "accuracy": acc,
            }
        )

    return score, table


def brier(probs: Sequence[float], outcomes: Sequence[bool]) -> float:
    """Brier score for a binary outcome: mean squared error of the probability.

    Reported alongside ECE because ECE can be gamed by a model that is always
    unconfident; Brier cannot -- it rewards sharpness and calibration together.
    """
    if len(probs) != len(outcomes):
        raise ValueError(f"length mismatch: {len(probs)} probs, {len(outcomes)} outcomes")
    if not probs:
        return float("nan")
    return sum((p - float(o)) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


# ---------------------------------------------------------------------------
# Reliability and economics
# ---------------------------------------------------------------------------


def schema_conformance(records: Sequence[EvalRecord]) -> float:
    """Fraction of records that yielded a schema-valid object.

    Counts the *final* outcome per scenario. Attempts spent getting there are
    reported separately as `total_retries`, and are billed in `cost_totals` --
    a system that needed three tries to emit valid JSON conformed in the end but
    did not do so for free.
    """
    if not records:
        return float("nan")
    return sum(1 for r in records if r.schema_ok) / len(records)


def cost_totals(records: Sequence[EvalRecord]) -> dict[str, float]:
    """Token and dollar totals, plus a cost-per-1000-decisions figure.

    Per-record `cost_usd` is authoritative: each evaluator computes its own from
    its own price table entry, including tokens burned on retried attempts.
    """
    n = len(records)
    total_cost = sum(r.cost_usd for r in records)
    return {
        "total_input_tokens": float(sum(r.input_tokens for r in records)),
        "total_output_tokens": float(sum(r.output_tokens for r in records)),
        "total_cost_usd": total_cost,
        "cost_per_1k_decisions_usd": (total_cost / n * 1000.0) if n else float("nan"),
    }


# ---------------------------------------------------------------------------
# Record-level extraction helpers
# ---------------------------------------------------------------------------
# These bridge `EvalRecord` + ground truth to the pure functions above. Kept
# separate so the maths can be tested on literals, with no model objects.


def urgency_errors(
    records: Sequence[EvalRecord],
    truth: dict[str, int],
    *,
    use_expected: bool = False,
) -> list[float]:
    """Signed urgency errors on the 0-100 display scale.

    `use_expected=True` reads Jev's continuous probability-weighted score
    instead of the rounded rubric level. It is the fairer reading of what the
    Score primitive returns, and it is reported as its own column rather than
    replacing the discrete one -- the LLM has no equivalent, so swapping it in
    silently would compare two different quantities.
    """
    errors: list[float] = []
    for r in records:
        want = truth[r.scenario_id] * URGENCY_DISPLAY_SCALE
        if use_expected:
            if r.urgency_expected is None:
                continue
            got = r.urgency_expected * URGENCY_DISPLAY_SCALE
        else:
            if r.prediction is None:
                # A missing answer is maximally wrong, not absent: charging the
                # full scale keeps a system from improving its MAE by failing.
                errors.append(float(100))
                continue
            got = float(r.prediction.urgency_0_100)
        errors.append(got - want)
    return errors
