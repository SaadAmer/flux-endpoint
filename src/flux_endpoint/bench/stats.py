import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Summary:
    """Distribution of one measurement."""

    count: int
    mean: float
    p50: float
    p95: float
    maximum: float


def summarize(values: Sequence[float]) -> Summary:
    """Summarize measurements with interpolated percentiles.

    Raises:
        ValueError: If ``values`` is empty.
    """
    if not values:
        raise ValueError("Cannot summarize an empty sample.")
    ordered = sorted(values)
    return Summary(
        count=len(ordered),
        mean=statistics.fmean(ordered),
        p50=percentile(ordered, 50),
        p95=percentile(ordered, 95),
        maximum=ordered[-1],
    )


def percentile(ordered: Sequence[float], percent: float) -> float:
    """Return the linearly interpolated percentile of values sorted ascending."""
    rank = (len(ordered) - 1) * percent / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)
