import io

import pytest
from PIL import Image

from flux_endpoint.schema import ImageFormat
from flux_endpoint.worker.imaging import encode_image


@pytest.mark.parametrize(
    ("image_format", "pil_format"),
    [
        pytest.param("png", "PNG", id="png"),
        pytest.param("jpeg", "JPEG", id="jpeg"),
    ],
)
def test_image_is_encoded_in_requested_format(image_format: ImageFormat, pil_format: str) -> None:
    """Given an RGB image, When encoded, Then the bytes are an image in that format."""
    image = Image.new("RGB", (64, 32), color=(200, 30, 30))

    encoded = encode_image(image, image_format)

    decoded = Image.open(io.BytesIO(encoded))
    assert decoded.format == pil_format
    assert decoded.size == (64, 32)


def test_image_with_alpha_is_encoded_as_jpeg() -> None:
    """Given an RGBA image, When encoded as JPEG, Then it is converted to RGB."""
    image = Image.new("RGBA", (16, 16), color=(0, 0, 255, 128))

    encoded = encode_image(image, "jpeg")

    decoded = Image.open(io.BytesIO(encoded))
    assert decoded.mode == "RGB"
