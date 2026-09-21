"""Render the three figures from `results/summary.json`, light and dark.

    python -m analysis.plot_results                 # 6 PNGs into figures/
    python -m analysis.plot_results --theme light   # just the light set

Every figure reads the summary and nothing else, so a figure cannot disagree with
the tables in the article. Two rules are enforced here rather than left to taste:

* **No dual axes anywhere.** Two y-scales in one frame let a reader infer a
  crossover that the data does not contain.
* **A simulated figure must be impossible to mistake for a measured one.** When
  the summary's provenance says the run was simulated, every figure gets a
  diagonal SIMULATED watermark across the plotting area and the word appears in
  the caption. The caption always carries the provenance line.

The palette is fixed and already validated for colour-vision deficiency
(CVD delta-E 24.7 light / 26.8 dark against the LLM series). Do not adjust the
series colours without re-running the validator.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in CI, and none needed

import matplotlib.pyplot as plt  # noqa: E402

from src.config import settings  # noqa: E402
from src.schemas import EvaluatorSummary, SummaryReport  # noqa: E402

JEV = "jev"


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Theme:
    """Colours for one rendering mode.

    `series_*` are the validated pair. The chrome colours are deliberately
    recessive: axes and grid should be legible and then get out of the way, so
    they are never the same weight as the data.
    """

    name: str
    surface: str
    ink: str
    axes: str
    grid: str
    series_jev: str
    series_llm: str

    @property
    def series(self) -> dict[str, str]:
        return {"jev": self.series_jev, "llm": self.series_llm}


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#1a1a19",
    axes="#898781",
    grid="#e1e0d9",
    series_jev="#2a78d6",
    series_llm="#eb6834",
)

# The dark grid is the only colour not in the validated set: it is chrome, not a
# series, so it needs to be quiet against #1a1a19 rather than distinguishable
# under CVD simulation.
DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#fcfcfb",
    axes="#898781",
    grid="#2f2f2c",
    series_jev="#3987e5",
    series_llm="#d95926",
)

THEMES = {t.name: t for t in (LIGHT, DARK)}

LINE_WIDTH = 2.0
MARKER_SIZE = 8.0          # points; the floor, not the target
FIG_SIZE = (9.0, 5.6)
# The calibration plot is forced square, so it gets a frame that is already
# square -- otherwise `aspect="equal"` leaves half the canvas empty.
FIG_SIZE_SQUARE = (7.0, 6.4)
DPI = 160
#: Captions wrap here rather than running off the canvas edge.
CAPTION_WIDTH = 132


def _series_colour(theme: Theme, evaluator: str) -> str:
    """Jev keeps its colour; anything else is the LLM series."""
    return theme.series_jev if evaluator == JEV else theme.series_llm


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------


def _new_figure(
    theme: Theme,
    title: str,
    xlabel: str,
    ylabel: str,
    *,
    figsize: tuple[float, float] = FIG_SIZE,
):
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    fig.patch.set_facecolor(theme.surface)
    ax.set_facecolor(theme.surface)

    ax.set_title(title, color=theme.ink, fontsize=13, pad=14, loc="left")
    ax.set_xlabel(xlabel, color=theme.ink, fontsize=10)
    ax.set_ylabel(ylabel, color=theme.ink, fontsize=10)

    ax.grid(True, color=theme.grid, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(theme.axes)
    ax.tick_params(colors=theme.axes, labelsize=9)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(theme.ink)
    return fig, ax


def _finish(
    fig,
    ax,
    theme: Theme,
    report: SummaryReport,
    path: Path,
    *,
    note: str | None = None,
    legend_loc: str = "best",
) -> Path:
    """Add legend, provenance caption, simulation watermark; write the PNG."""
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        # Always present, even for two series that are also directly labelled:
        # the direct labels can be occluded by the data at some aspect ratios.
        legend = ax.legend(
            frameon=False, fontsize=9, loc=legend_loc, labelcolor=theme.ink
        )
        for text in legend.get_texts():
            text.set_color(theme.ink)

    simulated = report.provenance.mode != "live"
    if simulated:
        ax.text(
            0.5,
            0.5,
            "SIMULATED",
            transform=ax.transAxes,
            fontsize=58,
            color=theme.ink,
            alpha=0.10,
            rotation=28,
            ha="center",
            va="center",
            zorder=5,
            fontweight="bold",
        )

    caption = report.provenance.caption()
    if simulated:
        caption = "SIMULATED -- not a measurement  |  " + caption
    if note:
        caption = note + "\n" + caption
    # Wrap per paragraph so an explanatory note cannot run off the canvas edge
    # and lose its last clause, which is usually the caveat.
    caption = "\n".join(
        line
        for paragraph in caption.split("\n")
        for line in (textwrap.wrap(paragraph, CAPTION_WIDTH) or [""])
    )
    fig.text(
        0.01,
        0.015,
        caption,
        color=theme.axes,
        fontsize=7.5,
        ha="left",
        va="bottom",
    )

    # Reserve exactly as much bottom margin as the caption actually needs. A
    # fixed rect works until a note wraps onto a fifth line and lands on top of
    # the x-axis label.
    line_height = 7.5 * 1.4 / (fig.get_figheight() * 72)
    reserve = 0.02 + line_height * (caption.count("\n") + 1)
    fig.tight_layout(rect=(0, reserve, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=theme.surface, dpi=DPI)
    plt.close(fig)
    return path


def _label_line(ax, theme: Theme, x: float, y: float, text: str, colour: str) -> None:
    """Direct label beside the end of a series."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(6, 0),
        textcoords="offset points",
        color=colour,
        fontsize=9,
        va="center",
        fontweight="bold",
    )
    del theme  # signature kept uniform; colour comes from the series


# ---------------------------------------------------------------------------
# Figure 1 -- latency CDF
# ---------------------------------------------------------------------------


def figure_latency_cdf(report: SummaryReport, theme: Theme, out_dir: Path) -> Path:
    """Full distribution on a log x-axis, with p50/p95/p99 marked.

    A CDF rather than a bar chart of p50s: the interesting part of a latency
    comparison is the tail, and a bar chart hides it by construction. Log-x
    because the two distributions are an order of magnitude apart, which is the
    finding -- a linear axis would compress one series into the y-axis.
    """
    fig, ax = _new_figure(
        theme,
        "Per-request latency, full distribution",
        "latency (ms, log scale)",
        "fraction of requests at or below",
    )
    ax.set_xscale("log")

    summaries = {s.evaluator: s for s in report.evaluators}

    for name, samples in sorted(report.latency_samples.items()):
        if not samples:
            continue
        colour = _series_colour(theme, name)
        n = len(samples)
        ys = [(i + 1) / n for i in range(n)]
        ax.plot(
            samples,
            ys,
            color=colour,
            linewidth=LINE_WIDTH,
            label=summaries[name].model,
            zorder=3,
        )

        summary = summaries[name]
        marks = [(summary.p50_ms, 0.50), (summary.p95_ms, 0.95), (summary.p99_ms, 0.99)]
        ax.plot(
            [x for x, _ in marks],
            [y for _, y in marks],
            linestyle="none",
            marker="o",
            markersize=MARKER_SIZE,
            markerfacecolor=theme.surface,
            markeredgecolor=colour,
            markeredgewidth=2.0,
            zorder=4,
        )
        _label_line(ax, theme, samples[-1], 1.0, name, colour)

    ax.set_ylim(0, 1.04)
    # Room on the right for the direct labels, which sit outside the data.
    right = max((s[-1] for s in report.latency_samples.values() if s), default=1.0)
    ax.set_xlim(right=right * 2.2)
    return _finish(
        fig,
        ax,
        theme,
        report,
        out_dir / f"latency_cdf_{theme.name}.png",
        note="markers: p50, p95, p99. warm-up requests excluded.",
    )


# ---------------------------------------------------------------------------
# Figure 2 -- reliability curve
# ---------------------------------------------------------------------------


def figure_reliability(report: SummaryReport, theme: Theme, out_dir: Path) -> Path:
    """Reported probability against observed accuracy, per bin.

    Only an evaluator that reports a probability distribution can appear here,
    which is the asymmetry the article is about. The absent series is stated in
    the note rather than silently omitted -- and a token generator's first-token
    probability is NOT plotted as if it were a calibration curve, because it is
    not one.
    """
    fig, ax = _new_figure(
        theme,
        "Calibration: reported probability vs observed accuracy",
        "reported probability for the chosen option",
        "observed accuracy in that bin",
        figsize=FIG_SIZE_SQUARE,
    )

    ax.plot(
        [0, 1],
        [0, 1],
        color=theme.axes,
        linewidth=1.2,
        linestyle="--",
        label="perfect calibration",
        zorder=2,
    )

    summaries = {s.evaluator: s for s in report.evaluators}
    for name, bins in sorted(report.reliability.items()):
        colour = _series_colour(theme, name)
        points = [
            (row["mean_confidence"], row["accuracy"], row["n"])
            for row in bins
            # Empty bins are kept in the summary so gaps stay visible in the
            # table; here they have no coordinates, so they are dropped.
            if row["n"] > 0 and row["accuracy"] == row["accuracy"]
        ]
        if not points:
            continue
        ece = summaries[name].ece_department
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            color=colour,
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE + 1,
            label=f"{name}  ECE={ece:.4f}" if ece is not None else name,
            zorder=3,
        )
        for x, y, count in points:
            # One consistent up-and-right offset. Alternating sides sounds
            # tidier but pulls the labels of adjacent bins towards each other,
            # which is exactly where they collide.
            ax.annotate(
                f"n={int(count)}",
                xy=(x, y),
                xytext=(9, 8),
                textcoords="offset points",
                color=theme.axes,
                fontsize=7,
                ha="left",
                va="bottom",
            )

    absent = [s for s in report.evaluators if s.evaluator not in report.reliability]
    note = "each point is one probability bin; n = requests in that bin."
    for summary in absent:
        proxy = summary.mean_first_token_proxy
        note += (
            f"\n{summary.evaluator} has no series: it exposes no distribution over "
            "answers."
        )
        if proxy is not None:
            note += (
                f" Its mean first-token probability is {proxy:.4f} -- confidence "
                "about emitting a brace, not about the answer, so it is not "
                "plotted as calibration."
            )

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal", adjustable="box")
    return _finish(
        fig,
        ax,
        theme,
        report,
        out_dir / f"reliability_{theme.name}.png",
        note=note,
    )


# ---------------------------------------------------------------------------
# Figure 3 -- cost vs accuracy
# ---------------------------------------------------------------------------


def figure_cost_accuracy(report: SummaryReport, theme: Theme, out_dir: Path) -> Path:
    """Cost per 1,000 decisions against department accuracy.

    One point per evaluator, log-x because the cost axis spans orders of
    magnitude. Deliberately not a dual-axis bar chart: putting cost and accuracy
    on two scales in one frame invites the reader to compare bar heights that
    share no unit.
    """
    fig, ax = _new_figure(
        theme,
        "Cost per 1,000 decisions vs department accuracy",
        "USD per 1,000 decisions (log scale)",
        "department accuracy",
    )
    ax.set_xscale("log")

    xs: list[float] = []
    for summary in sorted(report.evaluators, key=lambda s: s.cost_per_1k_decisions_usd):
        colour = _series_colour(theme, summary.evaluator)
        x = summary.cost_per_1k_decisions_usd
        y = summary.department_accuracy
        xs.append(x)
        ax.plot(
            [x],
            [y],
            linestyle="none",
            marker="o",
            markersize=MARKER_SIZE + 4,
            color=colour,
            label=summary.model,
            zorder=3,
        )
        ax.annotate(
            f"{summary.evaluator}\n${x:,.4f} / 1k\nacc {y:.3f}",
            xy=(x, y),
            xytext=(10, -4),
            textcoords="offset points",
            color=colour,
            fontsize=9,
            va="top",
            fontweight="bold",
        )

    if xs:
        ax.set_xlim(min(xs) / 3.0, max(xs) * 6.0)
    accuracies = [s.department_accuracy for s in report.evaluators]
    if accuracies:
        span = max(0.05, (max(accuracies) - min(accuracies)) * 2.5)
        mid = (max(accuracies) + min(accuracies)) / 2
        ax.set_ylim(max(0.0, mid - span), min(1.0, mid + span))

    return _finish(
        fig,
        ax,
        theme,
        report,
        out_dir / f"cost_accuracy_{theme.name}.png",
        # Round legend keys look exactly like the data markers on a scatter, so
        # the legend is parked in the corner furthest from the points.
        legend_loc="lower left",
        note=(
            "accuracy is agreement with a synthetic label scheme, not with human "
            "triage. cost is at the rates in src/config.py."
        ),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_report(path: Path) -> SummaryReport:
    return SummaryReport.model_validate(json.loads(path.read_text(encoding="utf-8")))


def render_all(report: SummaryReport, out_dir: Path, themes: list[Theme]) -> list[Path]:
    written: list[Path] = []
    for theme in themes:
        written.append(figure_latency_cdf(report, theme, out_dir))
        written.append(figure_reliability(report, theme, out_dir))
        written.append(figure_cost_accuracy(report, theme, out_dir))
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, default=settings.summary_path)
    ap.add_argument("--out-dir", type=Path, default=settings.fig_dir)
    ap.add_argument(
        "--theme",
        choices=[*THEMES, "both"],
        default="both",
        help="which rendering mode to write (default: both)",
    )
    args = ap.parse_args()

    if not args.summary.exists():
        print(f"no summary at {args.summary} -- run `python -m src.runner` first")
        return 1

    report = load_report(args.summary)
    themes = list(THEMES.values()) if args.theme == "both" else [THEMES[args.theme]]

    for path in render_all(report, args.out_dir, themes):
        print(f"wrote {path}")
    if report.provenance.mode != "live":
        print(f"\n!! {report.provenance.simulated_warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
