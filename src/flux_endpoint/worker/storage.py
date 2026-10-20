from collections.abc import Mapping
from typing import Protocol

from runpod.serverless.utils import rp_upload

from flux_endpoint.worker.settings import WorkerSettings

BUCKET_VARIABLES = ("BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID", "BUCKET_SECRET_ACCESS_KEY")


class ImageStore(Protocol):
    """Keeps generated images outside the job result."""

    def save(self, name: str, data: bytes) -> str: ...


class BucketStore:
    """Uploads images to an S3-compatible bucket through the Runpod SDK.

    The SDK reads the ``BUCKET_*`` variables and returns presigned URLs that stay
    valid for 7 days, the same as the Public Endpoint's image URLs.
    """

    def __init__(self, bucket: str) -> None:
        self._bucket = bucket

    def save(self, name: str, data: bytes) -> str:
        return str(rp_upload.upload_in_memory_object(name, data, bucket_name=self._bucket))


def open_bucket_store(settings: WorkerSettings, environ: Mapping[str, str]) -> BucketStore | None:
    """Open the output bucket when the settings name one.

    Raises:
        RuntimeError: If a bucket is set but a ``BUCKET_*`` variable is missing.
            Without them the SDK would write images to the worker's local disk.
    """
    if settings.output_bucket is None:
        return None
    if missing := [name for name in BUCKET_VARIABLES if not environ.get(name)]:
        raise RuntimeError(
            f"Set {', '.join(missing)} to upload images to bucket {settings.output_bucket!r}."
        )
    return BucketStore(settings.output_bucket)
