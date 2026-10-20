from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

PUBLIC_ENDPOINT_USD_PER_MEGAPIXEL = 0.02
SECONDS_PER_HOUR = 3600


def public_endpoint_cost(
    width: int, height: int, usd_per_megapixel: float = PUBLIC_ENDPOINT_USD_PER_MEGAPIXEL
) -> float:
    """List price of one image on the FLUX.1-dev Public Endpoint, which is set per megapixel."""
    return width * height / 1_000_000 * usd_per_megapixel


def hourly_capacity(seconds_per_image: float) -> float:
    """Images one busy worker can make in an hour."""
    return SECONDS_PER_HOUR / seconds_per_image


@dataclass(frozen=True)
class GpuEconomics:
    """Cost of an image on a busy worker, compared with the Public Endpoint.

    A worker bills for every second it runs, so it is cheaper than the Public
    Endpoint only while it stays busy enough. ``break_even_utilization`` is
    that share of its time. Above 1.0 the GPU cannot beat the Public Endpoint.

    Raises:
        ValueError: If any input is zero or negative.
    """

    usd_per_hour: float
    seconds_per_image: float
    public_usd_per_image: float

    def __post_init__(self) -> None:
        if min(self.usd_per_hour, self.seconds_per_image, self.public_usd_per_image) <= 0:
            raise ValueError(
                "GPU price, time per image and Public Endpoint price must be positive."
            )

    @property
    def usd_per_image(self) -> float:
        """GPU cost of one image in USD."""
        return self.seconds_per_image * self.usd_per_hour / SECONDS_PER_HOUR

    @property
    def images_per_hour(self) -> float:
        return hourly_capacity(self.seconds_per_image)

    @property
    def break_even_images_per_hour(self) -> float:
        """Hourly volume above which one always-on worker is cheaper."""
        return self.usd_per_hour / self.public_usd_per_image

    @property
    def break_even_utilization(self) -> float:
        """Share of the worker's time it must be busy to break even."""
        return self.break_even_images_per_hour / self.images_per_hour


def serverless_prices(gpu_types: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Map GPU type IDs to Serverless USD per hour, from the REST API v2 catalog."""
    return {
        str(gpu["id"]): float(gpu["price"]["serverless"])
        for gpu in gpu_types
        if "serverless" in (gpu.get("price") or {})
    }
