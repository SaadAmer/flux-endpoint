import base64
import io
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import runpod
import torch
from diffusers import FluxPipeline
from PIL import Image
from runpod.serverless.utils import rp_upload

from flux_endpoint.schema import MAX_SEED
from flux_endpoint.worker.handler import build_handler, main, register_warm_up

from .fakes import FakeGenerator, FakeStore


def _decode_image(output: dict[str, Any]) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(output["image_base64"])))


def test_valid_job_returns_image_and_metadata(fake_generator: FakeGenerator) -> None:
    """Given a valid job, When handled, Then the output holds the image and its metadata."""
    handle = build_handler(fake_generator)

    output = handle(
        {
            "id": "job-1",
            "input": {
                "prompt": "a cat",
                "width": 512,
                "height": 256,
                "seed": 42,
                "image_format": "png",
            },
        }
    )

    assert _decode_image(output).size == (512, 256)
    assert output["image_format"] == "png"
    assert (output["width"], output["height"], output["seed"]) == (512, 256, 42)
    assert output["gpu"] == "Fake GPU"
    assert output["inference_ms"] >= 0
    assert output["encode_ms"] >= 0
    assert fake_generator.calls[0][1] == 42


def test_inline_output_has_no_image_url(fake_generator: FakeGenerator) -> None:
    """Given no output bucket, When handled, Then the output holds only the inline image."""
    output = build_handler(fake_generator)({"id": "job-1", "input": {"prompt": "a cat"}})

    assert "image_base64" in output
    assert "image_url" not in output


def test_bucket_output_returns_image_url(fake_generator: FakeGenerator) -> None:
    """Given an output bucket, When handled, Then the image is uploaded and its URL returned."""
    store = FakeStore()
    handle = build_handler(fake_generator, store)

    output = handle({"id": "job-7", "input": {"prompt": "a cat", "image_format": "png"}})

    assert output["image_url"] == "https://cdn.example.com/job-7.png"
    assert "image_base64" not in output
    assert Image.open(io.BytesIO(store.saved["job-7.png"])).format == "PNG"


def test_bucket_output_names_jpeg_files_jpg(fake_generator: FakeGenerator) -> None:
    """Given a JPEG job without an ID, When handled, Then the upload gets a .jpg name."""
    store = FakeStore()

    build_handler(fake_generator, store)({"input": {"prompt": "a cat"}})

    assert [name.endswith(".jpg") for name in store.saved] == [True]


def test_random_seed_is_resolved_and_reported(fake_generator: FakeGenerator) -> None:
    """Given seed -1, When handled, Then a concrete seed is used and returned."""
    handle = build_handler(fake_generator)

    output = handle({"input": {"prompt": "a cat", "seed": -1}})

    used_seed = fake_generator.calls[0][1]
    assert 0 <= used_seed <= MAX_SEED
    assert output["seed"] == used_seed


@pytest.mark.parametrize(
    ("job", "expected"),
    [
        pytest.param(
            {"input": {"prompt": "a cat", "width": 1000}},
            "Invalid input: width: Input should be a multiple of 64",
            id="bad-width",
        ),
        pytest.param({"input": {}}, "Invalid input: prompt: Field required", id="empty"),
        pytest.param({}, "Invalid input: prompt: Field required", id="no-input"),
        pytest.param({"input": "a cat"}, "Invalid input: input: ", id="not-an-object"),
    ],
)
def test_invalid_input_fails_without_generating(
    fake_generator: FakeGenerator, job: dict[str, Any], expected: str
) -> None:
    """Given invalid input, When handled, Then the job fails with the field and reason."""
    output = build_handler(fake_generator)(job)

    assert output["error"].startswith(expected)
    assert fake_generator.calls == []


def test_out_of_memory_fails_job_and_refreshes_worker() -> None:
    """Given a CUDA out-of-memory error, When handled, Then the worker is refreshed."""
    generator = FakeGenerator(error=torch.OutOfMemoryError("CUDA out of memory"))

    output = build_handler(generator)({"input": {"prompt": "a cat"}})

    assert output == {
        "error": "GPU ran out of memory. Retry with a smaller image.",
        "refresh_worker": True,
    }


@pytest.mark.parametrize(
    ("generator", "job_input"),
    [
        pytest.param(FakeGenerator(), {"prompt": "a secret product launch"}, id="generated"),
        pytest.param(
            FakeGenerator(), {"prompt": "a secret product launch", "width": 1}, id="invalid"
        ),
        pytest.param(
            FakeGenerator(error=torch.OutOfMemoryError("CUDA out of memory")),
            {"prompt": "a secret product launch"},
            id="out-of-memory",
        ),
    ],
)
def test_logs_name_the_job_but_not_the_prompt(
    caplog: pytest.LogCaptureFixture, generator: FakeGenerator, job_input: dict[str, Any]
) -> None:
    """Given any outcome, When handled, Then the log names the job and leaves out the prompt."""
    with caplog.at_level(logging.INFO, logger="flux_endpoint.worker.handler"):
        build_handler(generator)({"id": "job-42", "input": job_input})

    assert "job-42" in caplog.text
    assert "secret" not in caplog.text


def test_client_field_names_cannot_forge_log_lines(
    caplog: pytest.LogCaptureFixture, fake_generator: FakeGenerator
) -> None:
    """Given a field name with a line break, When rejected, Then the error and log stay one line."""
    job = {"id": "job-9", "input": {"prompt": "a cat", "x\nINFO Job job-0 generated": 1}}

    with caplog.at_level(logging.WARNING, logger="flux_endpoint.worker.handler"):
        output = build_handler(fake_generator)(job)

    assert "\n" not in output["error"]
    assert [record.getMessage().count("\n") for record in caplog.records] == [0]


def test_unexpected_errors_reach_the_sdk() -> None:
    """Given an unexpected error, When handled, Then it propagates to the Runpod SDK."""
    generator = FakeGenerator(error=ValueError("broken weights"))

    with pytest.raises(ValueError, match="broken weights"):
        build_handler(generator)({"input": {"prompt": "a cat"}})


def test_warm_up_is_registered_as_fitness_check(
    monkeypatch: pytest.MonkeyPatch, fake_generator: FakeGenerator
) -> None:
    """Given a generator, When the warm-up check runs, Then it makes one small image."""
    registered: list[Any] = []
    monkeypatch.setattr(runpod.serverless, "register_fitness_check", registered.append)

    register_warm_up(fake_generator)
    registered[0]()

    request, seed = fake_generator.calls[0]
    assert (request.width, request.height, request.num_inference_steps) == (256, 256, 1)
    assert seed == 0


@pytest.fixture
def worker_entry_points(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """Fake diffusers loading and the Runpod SDK entry points, with a CPU device."""
    pipeline = MagicMock(spec=FluxPipeline)
    pipeline.return_value = SimpleNamespace(images=[Image.new("RGB", (256, 256))])
    monkeypatch.setattr(FluxPipeline, "from_pretrained", MagicMock(return_value=pipeline))
    (tmp_path / "model_index.json").write_text("{}")
    monkeypatch.setenv("FLUX_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("FLUX_DEVICE", "cpu")
    monkeypatch.setenv("FLUX_DTYPE", "float32")
    entry_points = SimpleNamespace(start=MagicMock(), register_fitness_check=MagicMock())
    monkeypatch.setattr(runpod.serverless, "start", entry_points.start)
    monkeypatch.setattr(
        runpod.serverless, "register_fitness_check", entry_points.register_fitness_check
    )
    return entry_points


def test_main_loads_model_and_starts_worker(worker_entry_points: SimpleNamespace) -> None:
    """Given CPU settings, When main runs, Then the worker starts with a working handler."""
    main()

    handler = worker_entry_points.start.call_args.args[0]["handler"]
    output = handler({"input": {"prompt": "a cat", "width": 256, "height": 256}})
    assert output["gpu"] == "cpu"
    assert "image_base64" in output
    worker_entry_points.register_fitness_check.assert_called_once()


def test_main_logs_model_load_time(
    worker_entry_points: SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    """Given CPU settings, When main runs, Then the log gives the model folder and load time."""
    with caplog.at_level(logging.INFO, logger="flux_endpoint.worker.handler"):
        main()

    assert re.search(r"Loaded FLUX\.1-dev from /\S+ in \d+\.\d s", caplog.text)
    worker_entry_points.start.assert_called_once()


def test_main_uploads_to_configured_bucket(
    worker_entry_points: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given an output bucket and credentials, When main runs, Then jobs return image URLs."""
    monkeypatch.setenv("FLUX_OUTPUT_BUCKET", "flux-images")
    for name in ("BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID", "BUCKET_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(name, "configured")
    monkeypatch.setattr(
        rp_upload,
        "upload_in_memory_object",
        lambda name, _data, bucket_name: f"https://cdn.example.com/{bucket_name}/{name}",
    )

    main()

    handler = worker_entry_points.start.call_args.args[0]["handler"]
    output = handler({"id": "job-3", "input": {"prompt": "a cat", "image_format": "png"}})
    assert output["image_url"] == "https://cdn.example.com/flux-images/job-3.png"
