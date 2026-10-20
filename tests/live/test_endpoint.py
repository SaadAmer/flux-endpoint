import os

import httpx
import pytest

from flux_endpoint.client.images import image_bytes, image_suffix
from flux_endpoint.client.jobs import JobClient
from flux_endpoint.http import JOB_API_URL, create_runpod_client
from flux_endpoint.schema import GenerationRequest

pytestmark = pytest.mark.live


def test_deployed_endpoint_generates_image() -> None:
    """Given the deployed endpoint, When a small job runs, Then a PNG of that size returns."""
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "")
    if not (api_key and endpoint_id):
        pytest.skip("Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID to run live tests.")
    request = GenerationRequest(
        prompt="A lighthouse at dusk", width=512, height=512, seed=1, image_format="png"
    )

    with create_runpod_client(api_key, JOB_API_URL) as http, httpx.Client() as downloader:
        result = JobClient(http, endpoint_id).run(request.to_job_input(), timeout_s=900)
        assert result.succeeded, result.error
        data = image_bytes(result.output or {}, downloader)

    assert image_suffix(data) == ".png"
    assert result.output is not None
    assert (result.output["width"], result.output["height"]) == (512, 512)
