from pathlib import Path
from typing import Protocol, Self

import torch
from diffusers import FluxPipeline
from PIL import Image

from flux_endpoint.schema import GenerationRequest
from flux_endpoint.worker.settings import DtypeName, WorkerSettings

GIB = 1024**3

_DTYPES: dict[DtypeName, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class ImageGenerator(Protocol):
    """Turns a validated request and a seed into an image."""

    @property
    def device_name(self) -> str: ...

    def generate(self, request: GenerationRequest, seed: int) -> Image.Image: ...


class FluxGenerator:
    """FLUX.1-dev pipeline, loaded once per worker and reused for every job."""

    def __init__(self, pipeline: FluxPipeline, device_name: str, max_sequence_length: int) -> None:
        self._pipeline = pipeline
        self._device_name = device_name
        self._max_sequence_length = max_sequence_length

    @classmethod
    def load(cls, settings: WorkerSettings) -> Self:
        """Load the pipeline from local weights straight onto the configured device.

        Loading with ``device_map`` streams the weights to the GPU instead of
        staging all 34 GB in host memory first. Only safetensors files load,
        because pickle-based weight files can run code.
        """
        pipeline = FluxPipeline.from_pretrained(
            resolve_model_dir(settings.model_dir),
            dtype=_DTYPES[settings.dtype],
            device_map=settings.device,
            local_files_only=True,
            use_safetensors=True,
        )
        pipeline.set_progress_bar_config(disable=True)
        return cls(pipeline, describe_device(settings.device), settings.max_sequence_length)

    @property
    def device_name(self) -> str:
        """GPU model name, or the device string when not on CUDA."""
        return self._device_name

    def generate(self, request: GenerationRequest, seed: int) -> Image.Image:
        """Generate one image.

        The noise comes from a CPU generator, so a seed gives the same starting
        noise on every GPU type.

        Raises:
            TypeError: If the pipeline does not return a PIL image.
        """
        with torch.inference_mode():
            output = self._pipeline(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt or None,
                true_cfg_scale=request.true_cfg_scale,
                width=request.width,
                height=request.height,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance,
                generator=torch.Generator("cpu").manual_seed(seed),
                max_sequence_length=self._max_sequence_length,
            )
        image = output.images[0]
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected a PIL image, got {type(image).__name__}.")
        return image


def resolve_model_dir(model_dir: Path) -> Path:
    """Return the diffusers folder that ``model_dir`` holds.

    ``model_dir`` is a diffusers folder, or the Hugging Face cache folder of one repo, as
    Runpod creates for a cached model. A cache folder resolves to the snapshot that
    ``refs/main`` names, or to its only snapshot when it has no ``refs/main``.

    Raises:
        FileNotFoundError: If ``model_dir`` does not exist, ``refs/main`` names a snapshot
            that is not there, or a cache folder without ``refs/main`` does not have
            exactly one snapshot.
    """
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model folder {model_dir} does not exist.")
    if (model_dir / "model_index.json").is_file():
        return model_dir
    revision = _main_revision(model_dir)
    if not revision:
        return _only_snapshot(model_dir)
    snapshot = model_dir / "snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Snapshot {revision} that {model_dir}/refs/main names is missing.")
    return snapshot


def _main_revision(model_dir: Path) -> str:
    main_ref = model_dir / "refs" / "main"
    return main_ref.read_text(encoding="utf-8").strip() if main_ref.is_file() else ""


def _only_snapshot(model_dir: Path) -> Path:
    snapshots = [path for path in (model_dir / "snapshots").glob("*") if path.is_dir()]
    if len(snapshots) != 1:
        raise FileNotFoundError(
            f"Model folder {model_dir} has no model_index.json, no refs/main and "
            f"{len(snapshots)} snapshots. It needs a diffusers model or one snapshot."
        )
    return snapshots[0]


def describe_device(device: str) -> str:
    """Return the GPU model for a CUDA device, or the device string otherwise."""
    if device == "cuda":
        return torch.cuda.get_device_name(torch.device(device))
    return device


def ensure_gpu_memory(settings: WorkerSettings) -> None:
    """Fail fast when the GPU is too small to hold the pipeline.

    Raises:
        RuntimeError: If CUDA is unavailable, or the GPU has less memory than
            ``settings.min_gpu_memory_gib``.
    """
    if settings.device != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this worker.")
    device = torch.device(settings.device)
    memory_gib = torch.cuda.get_device_properties(device).total_memory / GIB
    if memory_gib < settings.min_gpu_memory_gib:
        raise RuntimeError(
            f"GPU has {memory_gib:.0f} GiB, but FLUX.1-dev in {settings.dtype} needs "
            f"at least {settings.min_gpu_memory_gib:.0f} GiB."
        )
