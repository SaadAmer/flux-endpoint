import io

from PIL import Image

from flux_endpoint.schema import ImageFormat

JPEG_QUALITY = 90
FILE_SUFFIXES: dict[ImageFormat, str] = {"png": ".png", "jpeg": ".jpg"}

_PIL_FORMATS: dict[ImageFormat, str] = {"png": "PNG", "jpeg": "JPEG"}


def encode_image(image: Image.Image, image_format: ImageFormat) -> bytes:
    """Encode an image in the requested format."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format=_PIL_FORMATS[image_format], quality=JPEG_QUALITY)
    return buffer.getvalue()
