"""Inject generated tables into ARTICLE.md and README.md. The only writer.

    python -m analysis.render_article_tables            # write the tables
    python -m analysis.render_article_tables --check     # fail if they drifted

WHY THIS FILE EXISTS

No results number in this repository is typed by hand. Every one of them is
rendered here from `results/summary.json` and injected between markers:

    <!-- BEGIN:latency -->   ... generated, do not edit ...   <!-- END:latency -->

That is not bureaucracy. The brief for this project pre-specified its own results
before a single request had been made, which is exactly how a benchmark ends up
agreeing with whoever wrote it. Routing every number through one generator makes
that failure mode mechanically impossible, and `--check` -- which exits non-zero
when a committed table no longer matches the summary -- keeps it impossible after
someone edits the prose.

Two consequences worth stating:

* A column for a run that has not happened reads `TBD`. It is never estimated,
  interpolated, or carried over from a different mode. Pass a summary per mode
  (`--summary results/summary.json --summary results/summary_live.json`) and each
  one fills only the column its own provenance claims.
* Simulated columns are labelled simulated everywhere they appear, not once at
  the top.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from src.config import PARTNER_PRICING_CAVEAT, PRICES_USD_PER_MTOK, settings
from src.schemas import EvaluatorSummary, SummaryReport

TBD = "TBD"
ABSENT = "--"

BEGIN = "<!-- BEGIN:{name} -->"
END = "<!-- END:{name} -->"

TARGETS = ("ARTICLE.md", "README.md")


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _fmt(value: float | int | None, spec: str) -> str:
    """Format a metric, distinguishing "not applicable" from "not measured".

    `None` means this evaluator does not expose the quantity at all -- an LLM has
    no ECE because it has no distribution -- so it renders as an em-dash rather
    than as a zero, which would read as a perfect score.
    """
    if value is None:
        return ABSENT
    return format(value, spec)


class Column:
    """One (evaluator, mode) column, which may have no data behind it."""

    def __init__(self, evaluator: str, mode: str, summary: EvaluatorSummary | None):
        self.evaluator = evaluator
        self.mode = mode
        self.summary = summary

    @property
    def header(self) -> str:
        return f"{self.evaluator}<br>({self.mode})"

    def cell(self, render: Callable[[EvaluatorSummary], str]) -> str:
        if self.summary is None:
            # The run this column describes has not been executed. Saying so is
            # the entire point; filling it in would be fabrication.
            return TBD
        return render(self.summary)


def _table(columns: list[Column], rows: list[tuple[str, Callable]]) -> str:
    head = "| Metric | " + " | ".join(c.header for c in columns) + " |"
    rule = "|---|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + label + " | " + " | ".join(c.cell(render) for c in columns) + " |"
        for label, render in rows
    ]
    return "\n".join([head, rule, *body])


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


def _columns(reports: dict[str, SummaryReport], evaluators: list[str]) -> list[Column]:
    """One column per (evaluator, mode), in a fixed order so diffs stay small."""
    columns: list[Column] = []
    for mode in ("simulated", "live"):
        report = reports.get(mode)
        by_name = (
            {s.evaluator: s for s in report.evaluators} if report is not None else {}
        )
        for name in evaluators:
            columns.append(Column(name, mode, by_name.get(name)))
    return columns


def _evaluator_names(reports: dict[str, SummaryReport]) -> list[str]:
    """Jev first, then the baselines, so every table reads the same way."""
    names: list[str] = []
    for report in reports.values():
        for summary in report.evaluators:
            if summary.evaluator not in names:
                names.append(summary.evaluator)
    names.sort(key=lambda n: (n != "jev", n))
    return names


def block_provenance(reports: dict[str, SummaryReport]) -> str:
    lines = [
        "| | Simulated run | Live run |",
        "|---|---|---|",
    ]
    fields: list[tuple[str, Callable[[SummaryReport], str]]] = [
        ("generated at (UTC)", lambda r: r.provenance.generated_at),
        ("Jev model", lambda r: f"`{r.provenance.jev_model}`"),
        ("LLM model", lambda r: f"`{r.provenance.llm_model}`"),
        ("scenarios", lambda r: str(r.provenance.n_scenarios)),
        ("dataset sha256", lambda r: f"`{r.provenance.dataset_sha256[:16]}...`"),
        ("seed", lambda r: str(r.provenance.seed)),
        ("code revision", lambda r: f"`{r.provenance.git_sha or 'not a git repo'}`"),
    ]
    for label, render in fields:
        cells = [
            render(reports[mode]) if mode in reports else TBD
            for mode in ("simulated", "live")
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    prices = ", ".join(
        f"`{model}` ${row['input']:.3f}/${row['output']:.3f} per MTok "
        f"(verified {row['verified_on']})"
        for model, row in PRICES_USD_PER_MTOK.items()
    )
    lines += [
        "",
        f"**Prices used:** {prices}.",
        "",
        f"**Bedrock caveat:** {PARTNER_PRICING_CAVEAT}.",
    ]

    simulated = reports.get("simulated")
    if simulated is not None and simulated.provenance.simulated_warning:
        lines += ["", f"**{simulated.provenance.simulated_warning}**"]
    if "live" not in reports:
        lines += [
            "",
            "**No live run has been executed.** Every number below comes from the "
            "simulated run; the live columns read `TBD` and will stay that way "
            "until a run against the real APIs produces them.",
        ]
    return "\n".join(lines)


def block_latency(reports: dict[str, SummaryReport]) -> str:
    columns = _columns(reports, _evaluator_names(reports))
    rows: list[tuple[str, Callable[[EvaluatorSummary], str]]] = [
        ("p50 latency (ms)", lambda s: _fmt(s.p50_ms, ",.1f")),
        ("p95 latency (ms)", lambda s: _fmt(s.p95_ms, ",.1f")),
        ("p99 latency (ms)", lambda s: _fmt(s.p99_ms, ",.1f")),
        ("mean latency (ms)", lambda s: _fmt(s.mean_ms, ",.1f")),
        (
            "sequential throughput (req/s)",
            lambda s: _fmt(1000.0 / s.mean_ms if s.mean_ms else None, ",.2f"),
        ),
        ("requests", lambda s: _fmt(s.n, ",d")),
    ]
    return _table(columns, rows)


def block_conformance(reports: dict[str, SummaryReport]) -> str:
    columns = _columns(reports, _evaluator_names(reports))
    rows: list[tuple[str, Callable[[EvaluatorSummary], str]]] = [
        ("schema conformance", lambda s: _fmt(s.schema_conformance, ".4f")),
        ("retries spent reaching valid output", lambda s: _fmt(s.total_retries, ",d")),
        ("hard errors (no answer at all)", lambda s: _fmt(s.hard_errors, ",d")),
    ]
    return _table(columns, rows)


def block_cost(reports: dict[str, SummaryReport]) -> str:
    columns = _columns(reports, _evaluator_names(reports))
    rows: list[tuple[str, Callable[[EvaluatorSummary], str]]] = [
        ("input tokens", lambda s: _fmt(s.total_input_tokens, ",d")),
        ("output tokens", lambda s: _fmt(s.total_output_tokens, ",d")),
        ("total cost (USD)", lambda s: "$" + _fmt(s.total_cost_usd, ",.4f")),
        (
            "cost per 1,000 decisions (USD)",
            lambda s: "$" + _fmt(s.cost_per_1k_decisions_usd, ",.4f"),
        ),
    ]
    return _table(columns, rows)


def block_quality(reports: dict[str, SummaryReport]) -> str:
    columns = _columns(reports, _evaluator_names(reports))
    rows: list[tuple[str, Callable[[EvaluatorSummary], str]]] = [
        ("department accuracy", lambda s: _fmt(s.department_accuracy, ".4f")),
        ("department macro-F1", lambda s: _fmt(s.department_macro_f1, ".4f")),
        ("escalation accuracy", lambda s: _fmt(s.escalate_accuracy, ".4f")),
        ("urgency MAE (0-100 scale)", lambda s: _fmt(s.urgency_mae_0_100, ".2f")),
        ("urgency RMSE (0-100 scale)", lambda s: _fmt(s.urgency_rmse_0_100, ".2f")),
        (
            "urgency MAE from the continuous score",
            lambda s: _fmt(s.urgency_mae_expected_0_100, ".2f"),
        ),
        ("all three fields correct", lambda s: _fmt(s.exact_match_all_three, ".4f")),
    ]
    return _table(columns, rows)


def block_calibration(reports: dict[str, SummaryReport]) -> str:
    columns = _columns(reports, _evaluator_names(reports))
    rows: list[tuple[str, Callable[[EvaluatorSummary], str]]] = [
        ("ECE, department choice", lambda s: _fmt(s.ece_department, ".4f")),
        ("ECE, escalation", lambda s: _fmt(s.ece_escalate, ".4f")),
        ("Brier score, escalation", lambda s: _fmt(s.brier_escalate, ".4f")),
        (
            "ECE computed on `confidence` instead (not a probability)",
            lambda s: _fmt(s.ece_department_on_confidence, ".4f"),
        ),
        (
            "mean first-token probability (a proxy, not a distribution)",
            lambda s: _fmt(s.mean_first_token_proxy, ".4f"),
        ),
    ]
    return _table(columns, rows)


def block_reliability(reports: dict[str, SummaryReport]) -> str:
    """Per-bin reliability data, from the same bins the figure plots."""
    lines: list[str] = []
    for mode in ("simulated", "live"):
        report = reports.get(mode)
        if report is None or not report.reliability:
            continue
        for name, bins in sorted(report.reliability.items()):
            lines += [
                f"**{name} ({mode})** -- reported probability vs observed accuracy",
                "",
                "| probability bin | requests | mean reported | observed accuracy |",
                "|---|---|---|---|",
            ]
            for row in bins:
                count = int(row["n"])
                accuracy = row["accuracy"]
                # NaN marks an empty bin. It is kept rather than dropped so the
                # gaps in the distribution stay visible.
                nan = accuracy != accuracy
                lines.append(
                    f"| [{row['bin_lo']:.1f}, {row['bin_hi']:.1f}) | {count:,d} | "
                    f"{ABSENT if nan else format(row['mean_confidence'], '.4f')} | "
                    f"{ABSENT if nan else format(accuracy, '.4f')} |"
                )
            lines.append("")
    if not lines:
        return (
            "No evaluator in this run reported a probability distribution, so "
            "there is no reliability table."
        )
    return "\n".join(lines).rstrip()


def block_readme_summary(reports: dict[str, SummaryReport]) -> str:
    """The compact block the README shows instead of the full tables."""
    columns = _columns(reports, _evaluator_names(reports))
    rows: list[tuple[str, Callable[[EvaluatorSummary], str]]] = [
        ("p50 latency (ms)", lambda s: _fmt(s.p50_ms, ",.1f")),
        ("p99 latency (ms)", lambda s: _fmt(s.p99_ms, ",.1f")),
        ("schema conformance", lambda s: _fmt(s.schema_conformance, ".4f")),
        ("department accuracy", lambda s: _fmt(s.department_accuracy, ".4f")),
        ("ECE, department choice", lambda s: _fmt(s.ece_department, ".4f")),
        (
            "cost per 1,000 decisions",
            lambda s: "$" + _fmt(s.cost_per_1k_decisions_usd, ",.4f"),
        ),
    ]
    table = _table(columns, rows)
    modes = ", ".join(sorted(reports)) or "none"
    footer = (
        f"\n\nRuns on disk: {modes}. "
        + (
            "`TBD` columns have not been run."
            if "live" not in reports
            else "Live columns are measurements."
        )
        + " Regenerate with `make bench && make article`."
    )
    return table + footer


BLOCKS: dict[str, Callable[[dict[str, SummaryReport]], str]] = {
    "provenance": block_provenance,
    "latency": block_latency,
    "conformance": block_conformance,
    "cost": block_cost,
    "quality": block_quality,
    "calibration": block_calibration,
    "reliability": block_reliability,
    "readme_summary": block_readme_summary,
}


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------


def inject(text: str, name: str, body: str) -> tuple[str, bool]:
    """Replace the content between one marker pair. Returns (text, found)."""
    begin, end = BEGIN.format(name=name), END.format(name=name)
    start = text.find(begin)
    if start == -1:
        return text, False
    stop = text.find(end, start)
    if stop == -1:
        raise ValueError(f"{begin} has no matching {end}")

    replacement = (
        begin
        + "\n<!-- generated by analysis/render_article_tables.py -- do not edit -->\n\n"
        + body.strip()
        + "\n\n"
    )
    return text[:start] + replacement + text[stop:], True


def render(reports: dict[str, SummaryReport]) -> dict[str, str]:
    return {name: build(reports) for name, build in BLOCKS.items()}


def apply_to_files(
    bodies: dict[str, str], root: Path, *, check: bool
) -> tuple[int, list[str]]:
    """Write (or verify) every block into every target file.

    Returns the number of blocks placed and a list of problems. A block with no
    marker in any file is a problem: it means a metric is being computed and then
    silently dropped.
    """
    placed: set[str] = set()
    problems: list[str] = []

    for filename in TARGETS:
        path = root / filename
        if not path.exists():
            problems.append(f"{filename}: missing")
            continue
        original = path.read_text(encoding="utf-8")
        updated = original
        for name, body in bodies.items():
            updated, found = inject(updated, name, body)
            if found:
                placed.add(name)
        if updated == original:
            continue
        if check:
            problems.append(f"{filename}: tables are stale")
        else:
            path.write_text(updated, encoding="utf-8", newline="\n")

    for name in bodies:
        if name not in placed:
            problems.append(
                f"block {name!r} has no <!-- BEGIN:{name} --> marker in any of "
                f"{', '.join(TARGETS)}"
            )
    return len(placed), problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_reports(paths: list[Path]) -> dict[str, SummaryReport]:
    """Key each summary by the mode its own provenance claims.

    Classifying by content rather than by filename means a summary cannot be
    filed under the wrong column by renaming it.
    """
    reports: dict[str, SummaryReport] = {}
    for path in paths:
        if not path.exists():
            continue
        report = SummaryReport.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
        mode = report.provenance.mode
        if mode == "mixed":
            raise SystemExit(
                f"{path}: provenance mode is 'mixed' -- records from a live and a "
                "simulated run are in one file. Re-run without --resume."
            )
        if mode in reports:
            raise SystemExit(f"two summaries both claim mode={mode!r}; pass one each")
        reports[mode] = report
    return reports


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--summary",
        type=Path,
        action="append",
        help="a summary.json; repeat for one per mode (default: results/summary.json "
        "plus results/summary_live.json if it exists)",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="directory holding ARTICLE.md and README.md",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the committed tables differ from the summaries",
    )
    args = ap.parse_args()

    paths = args.summary or [
        settings.summary_path,
        settings.out_dir / "summary_live.json",
    ]
    reports = load_reports(paths)
    if not reports:
        print(f"no summary found at {', '.join(str(p) for p in paths)}")
        print("run `python -m src.runner` first")
        return 1

    placed, problems = apply_to_files(render(reports), args.root, check=args.check)

    for problem in problems:
        print(f"!! {problem}")
    if args.check:
        if problems:
            print("\ntables are out of date -- run `make article` and commit the result")
            return 1
        print(f"tables match the summaries ({placed} blocks, modes: {', '.join(reports)})")
        return 0

    print(f"injected {placed} blocks into {', '.join(TARGETS)}")
    print(f"modes present: {', '.join(reports)}")
    if "live" not in reports:
        print("live columns left as TBD -- no live run on disk")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
