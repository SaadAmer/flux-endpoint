import pytest

from flux_endpoint.bench.stats import Summary, percentile, summarize


def test_single_value_is_every_statistic() -> None:
    """Given one value, When summarized, Then every statistic equals it."""
    assert summarize([4.0]) == Summary(count=1, mean=4.0, p50=4.0, p95=4.0, maximum=4.0)


def test_percentiles_interpolate_between_values() -> None:
    """Given 1 to 10 in random order, When summarized, Then percentiles interpolate."""
    summary = summarize([7, 1, 10, 3, 5, 2, 9, 4, 8, 6])

    assert summary.count == 10
    assert summary.mean == 5.5
    assert summary.p50 == pytest.approx(5.5)
    assert summary.p95 == pytest.approx(9.55)
    assert summary.maximum == 10


@pytest.mark.parametrize(
    ("percent", "expected"),
    [
        pytest.param(0, 1.0, id="minimum"),
        pytest.param(100, 3.0, id="maximum"),
        pytest.param(25, 1.5, id="first-quartile"),
    ],
)
def test_percentile_falls_between_neighbors(percent: float, expected: float) -> None:
    """Given sorted values, When a percentile is taken, Then it falls between neighbors."""
    assert percentile([1.0, 2.0, 3.0], percent) == pytest.approx(expected)


def test_empty_sample_is_rejected() -> None:
    """Given no values, When summarized, Then a ValueError is raised."""
    with pytest.raises(ValueError, match="empty"):
        summarize([])
