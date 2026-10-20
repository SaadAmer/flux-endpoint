import pytest
from runpod.serverless.utils import rp_upload

from flux_endpoint.worker.settings import WorkerSettings
from flux_endpoint.worker.storage import BucketStore, open_bucket_store

CREDENTIALS = {
    "BUCKET_ENDPOINT_URL": "https://account.r2.cloudflarestorage.com",
    "BUCKET_ACCESS_KEY_ID": "access-key",
    "BUCKET_SECRET_ACCESS_KEY": "secret-key",
}


def test_saved_image_returns_presigned_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given a bucket, When an image is saved, Then the SDK uploads it and returns its URL."""
    uploads: list[tuple[str, bytes, str | None]] = []

    def upload(file_name: str, file_data: bytes, bucket_name: str | None = None) -> str:
        uploads.append((file_name, file_data, bucket_name))
        return f"https://cdn.example.com/{file_name}?signature=abc"

    monkeypatch.setattr(rp_upload, "upload_in_memory_object", upload)

    url = BucketStore("flux-images").save("job-1.png", b"png-bytes")

    assert url == "https://cdn.example.com/job-1.png?signature=abc"
    assert uploads == [("job-1.png", b"png-bytes", "flux-images")]


def test_no_bucket_means_inline_images() -> None:
    """Given no output bucket, When the store is opened, Then images stay inline."""
    assert open_bucket_store(WorkerSettings(), CREDENTIALS) is None


def test_bucket_with_credentials_opens_store() -> None:
    """Given a bucket and credentials, When the store is opened, Then a bucket store returns."""
    store = open_bucket_store(WorkerSettings(output_bucket="flux-images"), CREDENTIALS)

    assert isinstance(store, BucketStore)


def test_bucket_without_credentials_fails_fast() -> None:
    """Given a bucket but no secret key, When the store is opened, Then the variable is named."""
    environ = {
        key: value for key, value in CREDENTIALS.items() if key != "BUCKET_SECRET_ACCESS_KEY"
    }

    with pytest.raises(RuntimeError, match="Set BUCKET_SECRET_ACCESS_KEY to upload images"):
        open_bucket_store(WorkerSettings(output_bucket="flux-images"), environ)
