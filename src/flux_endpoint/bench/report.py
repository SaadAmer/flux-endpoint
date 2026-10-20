from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import fmean

from flux_endpoint.bench.costs import GpuEconomics, hourly_capacity
from flux_endpoint.bench.runner import Sample
from flux_endpoint.bench.stats import summarize

PUBLIC_ENDPOINT = "Public Endpoint"
NO_OUTPUT = "No output"
UNNAMED_VARIANT = "unnamed"
EMPTY_CELL = "-"

_LATENCY_HEADER = (
    "| Target | Jobs | OK | Queue p50 (s) | Queue max (s) | Run p50 (s) | Run p95 (s) "
    "| Total p95 (s) |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
)
_COST_HEADER = (
    "| Target | $/hour | Run p50 (s) | $/image | Images/hour | Break-even images/hour "
    "| Break-even utilization |",
    "|---|---:|---:|---:|---:|---:|---:|",
)
_COLD_START_HEADER = (
    "| Variant | Runs | OK | Cold start p50 (s) | Cold start max (s) | Run p50 (s) |",
    "|---|---:|---:|---:|---:|---:|",
)


def render_report(
    samples: Sequence[Sample], gpu_prices: Mapping[str, float], public_usd_per_image: float
) -> str:
    """Render latency and cost tables for benchmark samples as Markdown.

    Args:
        samples: Benchmark results from one or more endpoints.
        gpu_prices: Serverless USD per hour, keyed by GPU type ID.
        public_usd_per_image: Public Endpoint price for the benchmarked image size.

    Returns:
        Markdown with a latency table and a cost table.
    """
    groups = _group_by_target(samples)
    latency_rows = [_latency_row(target, group) for target, group in groups.items()]
    cost_rows = [
        _cost_row(target, successes, gpu_prices.get(target), public_usd_per_image)
        for target, group in groups.items()
        if (successes := [sample for sample in group if sample.succeeded])
    ]
    return "\n".join(
        [
            *_section("Latency", _LATENCY_HEADER, latency_rows),
            *_section("Cost per image", _COST_HEADER, cost_rows),
        ]
    )


def render_cold_start_report(samples: Sequence[Sample]) -> str:
    """Render cold-start times per variant as a Markdown table.

    A cold start is the queue time of a job sent while no worker was running.
    """
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.variant or UNNAMED_VARIANT].append(sample)
    rows = [_cold_start_row(variant, group) for variant, group in groups.items()]
    return "\n".join(_section("Cold starts", _COLD_START_HEADER, rows))


def _cold_start_row(variant: str, group: Sequence[Sample]) -> str:
    successes = [sample for sample in group if sample.succeeded]
    if not successes:
        return _row(variant, len(group), 0, *[EMPTY_CELL] * 3)
    cold_start = summarize([_seconds(sample.delay_ms) for sample in successes])
    run = summarize([_seconds(sample.execution_ms) for sample in successes])
    return _row(
        variant,
        len(group),
        len(successes),
        *(f"{value:.1f}" for value in (cold_start.p50, cold_start.maximum, run.p50)),
    )


def _section(title: str, header: Sequence[str], rows: Sequence[str]) -> list[str]:
    return [f"## {title}", "", *header, *rows, ""]


def _group_by_target(samples: Sequence[Sample]) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        if sample.gpu:
            groups[sample.gpu].append(sample)
        else:
            groups[PUBLIC_ENDPOINT if sample.cost_usd is not None else NO_OUTPUT].append(sample)
    return dict(groups)


def _latency_row(target: str, group: Sequence[Sample]) -> str:
    successes = [sample for sample in group if sample.succeeded]
    if not successes:
        return _row(target, len(group), 0, *[EMPTY_CELL] * 5)
    queue = summarize([_seconds(sample.delay_ms) for sample in successes])
    run = summarize([_seconds(sample.execution_ms) for sample in successes])
    total = summarize([sample.wall_ms / 1000 for sample in successes])
    return _row(
        target,
        len(group),
        len(successes),
        *(f"{value:.1f}" for value in (queue.p50, queue.maximum, run.p50, run.p95, total.p95)),
    )


def _cost_row(
    target: str,
    successes: Sequence[Sample],
    usd_per_hour: float | None,
    public_usd_per_image: float,
) -> str:
    run_s = summarize([_seconds(sample.execution_ms) for sample in successes]).p50
    if target == PUBLIC_ENDPOINT:
        cost = fmean(sample.cost_usd or 0.0 for sample in successes)
        return _row(target, EMPTY_CELL, f"{run_s:.1f}", f"${cost:.4f}", *[EMPTY_CELL] * 3)
    if usd_per_hour is None:
        capacity = f"{hourly_capacity(run_s):.0f}"
        return _row(target, "?", f"{run_s:.1f}", EMPTY_CELL, capacity, *[EMPTY_CELL] * 2)
    return _row(target, *_gpu_cost_cells(GpuEconomics(usd_per_hour, run_s, public_usd_per_image)))


def _gpu_cost_cells(economics: GpuEconomics) -> list[str]:
    return [
        f"${economics.usd_per_hour:.2f}",
        f"{economics.seconds_per_image:.1f}",
        f"${economics.usd_per_image:.4f}",
        f"{economics.images_per_hour:.0f}",
        f"{economics.break_even_images_per_hour:.0f}",
        f"{economics.break_even_utilization:.0%}",
    ]


def _seconds(milliseconds: int | None) -> float:
    return (milliseconds or 0) / 1000


def _row(*cells: object) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"
