import pytest

from flux_endpoint.bench.costs import GpuEconomics, public_endpoint_cost, serverless_prices


def test_public_endpoint_bills_per_megapixel() -> None:
    """Given a 1024x1024 image, When priced, Then it costs 1.048576 MP at $0.02."""
    assert public_endpoint_cost(1024, 1024) == pytest.approx(0.02097152)


def test_warm_worker_economics() -> None:
    """Given $1.75/h and 14 s per image, When compared, Then break-even is about 32% busy."""
    economics = GpuEconomics(usd_per_hour=1.75, seconds_per_image=14.0, public_usd_per_image=0.021)

    assert economics.usd_per_image == pytest.approx(0.0068056, rel=1e-4)
    assert economics.images_per_hour == pytest.approx(257.14, rel=1e-4)
    assert economics.break_even_images_per_hour == pytest.approx(83.33, rel=1e-4)
    assert economics.break_even_utilization == pytest.approx(0.3241, rel=1e-3)


@pytest.mark.parametrize(
    ("usd_per_hour", "seconds_per_image", "public_usd_per_image"),
    [
        pytest.param(0.0, 14.0, 0.021, id="free-gpu"),
        pytest.param(1.75, 0.0, 0.021, id="instant-image"),
        pytest.param(1.75, 14.0, 0.0, id="free-public-endpoint"),
    ],
)
def test_non_positive_inputs_are_rejected(
    usd_per_hour: float, seconds_per_image: float, public_usd_per_image: float
) -> None:
    """Given a zero input, When economics are built, Then a ValueError is raised."""
    with pytest.raises(ValueError, match="must be positive"):
        GpuEconomics(usd_per_hour, seconds_per_image, public_usd_per_image)


def test_serverless_prices_come_from_catalog() -> None:
    """Given a GPU catalog, When prices are read, Then only Serverless prices are kept."""
    catalog = [
        {"id": "NVIDIA L40S", "price": {"secure": 1.09, "serverless": 1.75}},
        {"id": "NVIDIA RTX 3070", "price": {"community": 0.13}},
        {"id": "NVIDIA H100 80GB HBM3", "price": {"serverless": 4.79}},
    ]

    assert serverless_prices(catalog) == {"NVIDIA L40S": 1.75, "NVIDIA H100 80GB HBM3": 4.79}
