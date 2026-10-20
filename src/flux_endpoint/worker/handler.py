import base64
import logging
import os
import secrets
import time
from collections.abc import Callable
from functools import partial
from typing import Any

import runpod
import torch
from pydantic import ValidationError

from flux_endpoint.schema import (
    MAX_SEED,
    RANDOM_SEED,
    GenerationRequest,
    GenerationResult,
    ImageFormat,
    format_validation_error,
)
from flux_endpoint.worker.imaging import FILE_SUFFIXES, encode_image
from flux_endpoint.worker.pipeline import (
    FluxGenerator,
    ImageGenerator,
    ensure_gpu_memory,
)
from flux_endpoint.worker.settings import WorkerSettings
from flux_endpoint.worker.storage import ImageStore, open_bucket_store

Job = dict[str, Any]
JobOutput = dict[str, Any]
Handler = Callable[[Job], JobOutput]

WARM_UP_REQUEST = GenerationRequest(prompt="warm-up", width=256, height=256, num_inference_steps=1)
OUT_OF_MEMORY_MESSAGE = "GPU ran out of memory. Retry with a smaller image."

logger = logging.getLogger(__name__)


def handle_job(generator: ImageGenerator, store: ImageStore | None, job: Job) -> JobOutput:
    """Validate a job's input, generate the image and return the job output.

    Invalid input and GPU out-of-memory errors fail the job with a short
    message. Other exceptions propagate, so the Runpod SDK records the
    traceback. Logs name the job but never include the prompt, which can hold
    customer data.
    """
    job_id = str(job.get("id") or secrets.token_hex(8))
    try:
        request = GenerationRequest.model_validate(job.get("input") or {})
    except ValidationError as error:
        message = format_validation_error(error)
        logger.warning("Job %s rejected: %s", job_id, message)
        return {"error": f"Invalid input: {message}"}
    try:
        result = generate_image(generator, request, store, job_id)
    except torch.OutOfMemoryError:
        logger.error("Job %s ran out of GPU memory at %dx%d", job_id, request.width, request.height)
        return {"error": OUT_OF_MEMORY_MESSAGE, "refresh_worker": True}
    logger.info(
        "Job %s generated %dx%d in %d ms on %s",
        job_id,
        result.width,
        result.height,
        result.inference_ms,
        result.gpu,
    )
    return result.model_dump(exclude_none=True)


def build_handler(generator: ImageGenerator, store: ImageStore | None = None) -> Handler:
    """Bind a loaded generator, and an optional image store, into the per-job handler."""
    return partial(handle_job, generator, store)


def generate_image(
    generator: ImageGenerator, request: GenerationRequest, store: ImageStore | None, job_id: str
) -> GenerationResult:
    """Generate and encode one image, and time both stages.

    With a store, the image is uploaded as the job ID plus the format's suffix,
    and the result carries its URL. Without one, the result carries the image.
    """
    seed = secrets.randbelow(MAX_SEED + 1) if request.seed == RANDOM_SEED else request.seed
    started = time.perf_counter()
    image = generator.generate(request, seed)
    generated = time.perf_counter()
    data = encode_image(image, request.image_format)
    encoded = time.perf_counter()
    image_base64, image_url = _place_image(data, request.image_format, store, job_id)
    return GenerationResult(
        image_base64=image_base64,
        image_url=image_url,
        image_format=request.image_format,
        width=image.width,
        height=image.height,
        seed=seed,
        gpu=generator.device_name,
        inference_ms=round((generated - started) * 1000),
        encode_ms=round((encoded - generated) * 1000),
    )


def _place_image(
    data: bytes, image_format: ImageFormat, store: ImageStore | None, job_id: str
) -> tuple[str | None, str | None]:
    if store is None:
        return base64.b64encode(data).decode("ascii"), None
    return None, store.save(f"{job_id}{FILE_SUFFIXES[image_format]}", data)


def register_warm_up(generator: ImageGenerator) -> None:
    """Register a one-step generation as a Runpod fitness check.

    The SDK runs fitness checks before the worker takes its first job, so a
    worker that cannot generate is replaced instead of failing jobs. The
    warm-up also initializes the CUDA kernels before the first request.
    """

    def generate_warm_up_image() -> None:
        generator.generate(WARM_UP_REQUEST, seed=0)

    runpod.serverless.register_fitness_check(generate_warm_up_image)


def main() -> None:
    """Load the model once, register the warm-up check and start the worker."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = WorkerSettings.from_env(os.environ)
    store = open_bucket_store(settings, os.environ)
    ensure_gpu_memory(settings)
    started = time.perf_counter()
    generator = FluxGenerator.load(settings)
    logger.info(
        "Loaded FLUX.1-dev from %s in %.1f s", settings.model_dir, time.perf_counter() - started
    )
    register_warm_up(generator)
    runpod.serverless.start({"handler": build_handler(generator, store)})


if __name__ == "__main__":
    main()
