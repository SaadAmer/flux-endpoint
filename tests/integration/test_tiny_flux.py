import base64
import io
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from PIL import Image

from flux_endpoint.worker.handler import Handler, build_handler
from flux_endpoint.worker.pipeline import FluxGenerator
from flux_endpoint.worker.settings import WorkerSettings

TINY_FLUX_REPO = "hf-internal-testing/tiny-flux-pipe"
TINY_FLUX_REVISION = "a98bc4ec6a80c47c477ed22d7e81c79872d9c723"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def handle() -> Handler:
    """Handler around a real FluxPipeline with tiny random weights, on the CPU."""
    try:
        model_dir = snapshot_download(TINY_FLUX_REPO, revision=TINY_FLUX_REVISION)
    except (HfHubHTTPError, LocalEntryNotFoundError) as error:
        pytest.skip(f"Tiny FLUX pipeline is not available: {error}")
    settings = WorkerSettings(
        model_dir=Path(model_dir), device="cpu", dtype="float32", max_sequence_length=48
    )
    return build_handler(FluxGenerator.load(settings))


def _job(**fields: object) -> dict[str, Any]:
    return {
        "input": {
            "prompt": "a red cube on a table",
            "width": 256,
            "height": 256,
            "num_inference_steps": 2,
            "seed": 7,
            "image_format": "png",
            **fields,
        }
    }


def test_real_pipeline_returns_requested_image(handle: Handler) -> None:
    """Given the tiny pipeline, When a job runs, Then a 256x256 PNG comes back."""
    output = handle(_job())

    image = Image.open(io.BytesIO(base64.b64decode(output["image_base64"])))
    assert (image.format, image.size) == ("PNG", (256, 256))
    assert output["gpu"] == "cpu"


def test_same_seed_gives_same_image(handle: Handler) -> None:
    """Given one seed twice, When jobs run, Then the images are identical."""
    assert handle(_job())["image_base64"] == handle(_job())["image_base64"]


def test_negative_prompt_with_true_cfg_changes_image(handle: Handler) -> None:
    """Given a negative prompt with true CFG, When a job runs, Then the image differs."""
    plain = handle(_job())["image_base64"]

    guided = handle(_job(negative_prompt="blue", true_cfg_scale=4.0))["image_base64"]

    assert guided != plain
