import base64
from collections.abc import Mapping
from typing import Any

import httpx

_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
)
_URL_FIELDS = ("image_url", "result")


def image_bytes(output: Mapping[str, Any], downloader: httpx.Client) -> bytes:
    """Return the image held in a job output.

    Reads this worker's ``image_base64`` field, or downloads the image URL that
    Runpod's Public Endpoints return. The FLUX.1-dev Public Endpoint documents it
    as ``image_url``, and returned it as ``result`` in October 2026. ``downloader``
    must not carry the API key, so the key never goes to the image host.

    Raises:
        ValueError: If the output has none of these fields, or the URL is not https.
        httpx.HTTPStatusError: If the image download fails.
    """
    if encoded := output.get("image_base64"):
        return base64.b64decode(encoded, validate=True)
    url = next((str(output[field]) for field in _URL_FIELDS if output.get(field)), None)
    if url is None:
        raise ValueError("Job output has no image_base64, image_url or result field.")
    if not url.startswith("https://"):
        raise ValueError("Job output image URL must be an https URL.")
    response = downloader.get(url)
    response.raise_for_status()
    return response.content


def image_suffix(data: bytes) -> str:
    """Return the file suffix that matches the image's signature.

    Raises:
        ValueError: If the data is not a PNG or JPEG image.
    """
    for signature, suffix in _SIGNATURES:
        if data.startswith(signature):
            return suffix
    raise ValueError("Image data is not a PNG or JPEG file.")
