import base64
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx

from flux_endpoint.client.images import image_bytes, image_suffix

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 8


@pytest.fixture
def downloader() -> Iterator[httpx.Client]:
    """Unauthenticated client for image downloads."""
    with httpx.Client() as client:
        yield client


def test_base64_output_is_decoded(downloader: httpx.Client) -> None:
    """Given this worker's output, When read, Then the base64 image is decoded."""
    output = {"image_base64": base64.b64encode(PNG_BYTES).decode()}

    assert image_bytes(output, downloader) == PNG_BYTES


URL_FIELDS = [
    pytest.param("image_url", id="documented-field"),
    pytest.param("result", id="field-seen-in-october-2026"),
]


@pytest.mark.parametrize("field", URL_FIELDS)
def test_public_endpoint_url_is_downloaded(downloader: httpx.Client, field: str) -> None:
    """Given a Public Endpoint output, When read, Then the image URL is downloaded."""
    output = {field: "https://image.runpod.ai/abc/output.jpg", "cost": 0.021}
    with respx.mock(assert_all_called=True) as router:
        route = router.get("https://image.runpod.ai/abc/output.jpg").respond(content=JPEG_BYTES)

        assert image_bytes(output, downloader) == JPEG_BYTES

    assert "Authorization" not in route.calls.last.request.headers


@pytest.mark.parametrize("field", URL_FIELDS)
def test_plain_http_image_url_is_refused(downloader: httpx.Client, field: str) -> None:
    """Given an http image URL, When read, Then a ValueError is raised before any download."""
    output = {field: "http://images.example.com/output.jpg"}
    with respx.mock(assert_all_called=False) as router:
        route = router.get(output[field]).respond(content=JPEG_BYTES)

        with pytest.raises(ValueError, match="https URL"):
            image_bytes(output, downloader)

    assert not route.called


@pytest.mark.parametrize(
    "output",
    [
        pytest.param({}, id="empty"),
        pytest.param({"seed": 3}, id="no-image-field"),
    ],
)
def test_output_without_image_is_rejected(downloader: httpx.Client, output: dict[str, Any]) -> None:
    """Given an output with no image field, When read, Then a ValueError is raised."""
    with pytest.raises(ValueError, match="image_base64, image_url or result"):
        image_bytes(output, downloader)


@pytest.mark.parametrize(
    ("data", "suffix"),
    [
        pytest.param(PNG_BYTES, ".png", id="png"),
        pytest.param(JPEG_BYTES, ".jpg", id="jpeg"),
    ],
)
def test_suffix_matches_image_signature(data: bytes, suffix: str) -> None:
    """Given image bytes, When the suffix is chosen, Then it matches the file signature."""
    assert image_suffix(data) == suffix


def test_unknown_signature_is_rejected() -> None:
    """Given bytes that are not an image, When the suffix is chosen, Then a ValueError is raised."""
    with pytest.raises(ValueError, match="not a PNG or JPEG"):
        image_suffix(b"<html>error</html>")
