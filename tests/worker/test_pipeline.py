from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from diffusers import FluxPipeline
from PIL import Image

from flux_endpoint.schema import GenerationRequest
from flux_endpoint.worker.pipeline import (
    GIB,
    FluxGenerator,
    describe_device,
    ensure_gpu_memory,
    resolve_model_dir,
)
from flux_endpoint.worker.settings import WorkerSettings

CPU_SETTINGS = WorkerSettings(
    model_dir=Path("/weights/flux"),
    device="cpu",
    dtype="float32",
    max_sequence_length=64,
)
GPU_SETTINGS = WorkerSettings(min_gpu_memory_gib=40.0)


def _mock_pipeline(image: Image.Image | None = None) -> MagicMock:
    pipeline = MagicMock(spec=FluxPipeline)
    pipeline.return_value = SimpleNamespace(images=[image or Image.new("RGB", (8, 8))])
    return pipeline


def test_load_reads_local_safetensors_onto_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given CPU settings, When loaded, Then local float32 safetensors load onto the CPU."""
    pipeline = _mock_pipeline()
    from_pretrained = MagicMock(return_value=pipeline)
    monkeypatch.setattr(FluxPipeline, "from_pretrained", from_pretrained)
    (tmp_path / "model_index.json").write_text("{}")

    generator = FluxGenerator.load(CPU_SETTINGS.model_copy(update={"model_dir": tmp_path}))

    from_pretrained.assert_called_once_with(
        tmp_path,
        dtype=torch.float32,
        device_map="cpu",
        local_files_only=True,
        use_safetensors=True,
    )
    pipeline.set_progress_bar_config.assert_called_once_with(disable=True)
    pipeline.to.assert_not_called()
    assert generator.device_name == "cpu"


def _hugging_face_cache(root: Path, revision: str, main_ref: str | None = None) -> Path:
    if main_ref is not None:
        (root / "refs").mkdir()
        (root / "refs" / "main").write_text(main_ref)
    snapshot = root / "snapshots" / revision
    snapshot.mkdir(parents=True)
    return snapshot


def test_plain_model_folder_is_used_as_is(tmp_path: Path) -> None:
    """Given a diffusers folder, When resolved, Then the folder itself is returned."""
    (tmp_path / "model_index.json").write_text("{}")

    assert resolve_model_dir(tmp_path) == tmp_path


def test_cache_folder_resolves_to_main_snapshot(tmp_path: Path) -> None:
    """Given a cache folder with two snapshots, When resolved, Then refs/main picks the one."""
    _hugging_face_cache(tmp_path, "0ld0ld0")
    snapshot = _hugging_face_cache(tmp_path, "3de623f", main_ref="3de623f\n")

    assert resolve_model_dir(tmp_path) == snapshot


@pytest.mark.parametrize(
    "main_ref",
    [pytest.param(None, id="no-refs-main"), pytest.param(" \n", id="blank-refs-main")],
)
def test_cache_folder_without_main_ref_uses_its_only_snapshot(
    tmp_path: Path, main_ref: str | None
) -> None:
    """Given a cache folder with one snapshot and no usable ref, When resolved, Then it is used."""
    snapshot = _hugging_face_cache(tmp_path, "3de623f", main_ref=main_ref)

    assert resolve_model_dir(tmp_path) == snapshot


@pytest.mark.parametrize("snapshots", [pytest.param(0, id="none"), pytest.param(2, id="two")])
def test_folder_without_model_or_one_snapshot_is_rejected(tmp_path: Path, snapshots: int) -> None:
    """Given no model_index.json and no single snapshot, When resolved, Then the count is named."""
    for index in range(snapshots):
        (tmp_path / "snapshots" / f"rev{index}").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match=f"{snapshots} snapshots"):
        resolve_model_dir(tmp_path)


def test_missing_model_folder_is_rejected(tmp_path: Path) -> None:
    """Given a model folder that does not exist, When resolved, Then the error names it."""
    missing = tmp_path / "huggingface-cache" / "hub" / "models--black-forest-labs--FLUX.1-dev"

    with pytest.raises(FileNotFoundError, match=r"models--black-forest-labs--FLUX\.1-dev"):
        resolve_model_dir(missing)


def test_cache_folder_without_its_snapshot_is_rejected(tmp_path: Path) -> None:
    """Given refs/main that names a missing snapshot, When resolved, Then the error names it."""
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs" / "main").write_text("3de623f")

    with pytest.raises(FileNotFoundError, match="3de623f"):
        resolve_model_dir(tmp_path)


def test_load_reads_a_cached_model_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a Runpod cached model folder, When loaded, Then its main snapshot is read."""
    snapshot = _hugging_face_cache(tmp_path, "3de623f", main_ref="3de623f")
    from_pretrained = MagicMock(return_value=_mock_pipeline())
    monkeypatch.setattr(FluxPipeline, "from_pretrained", from_pretrained)

    FluxGenerator.load(CPU_SETTINGS.model_copy(update={"model_dir": tmp_path}))

    assert from_pretrained.call_args.args[0] == snapshot


def test_generate_maps_request_to_pipeline_arguments() -> None:
    """Given a request, When generated, Then every field reaches the pipeline."""
    image = Image.new("RGB", (512, 256))
    pipeline = _mock_pipeline(image)
    generator = FluxGenerator(pipeline, device_name="cpu", max_sequence_length=64)
    request = GenerationRequest(
        prompt="a harbor at night",
        negative_prompt="fog",
        width=512,
        height=256,
        num_inference_steps=4,
        guidance=2.5,
        true_cfg_scale=3.0,
    )

    result = generator.generate(request, seed=1234)

    assert result is image
    arguments = pipeline.call_args.kwargs
    assert arguments["prompt"] == "a harbor at night"
    assert arguments["negative_prompt"] == "fog"
    assert arguments["true_cfg_scale"] == 3.0
    assert (arguments["width"], arguments["height"]) == (512, 256)
    assert arguments["num_inference_steps"] == 4
    assert arguments["guidance_scale"] == 2.5
    assert arguments["max_sequence_length"] == 64
    assert arguments["generator"].initial_seed() == 1234
    assert arguments["generator"].device == torch.device("cpu")


def test_generate_sends_no_negative_prompt_when_blank() -> None:
    """Given an empty negative prompt, When generated, Then None reaches the pipeline."""
    pipeline = _mock_pipeline()
    generator = FluxGenerator(pipeline, device_name="cpu", max_sequence_length=64)

    generator.generate(GenerationRequest(prompt="a cat", negative_prompt=""), seed=1)

    assert pipeline.call_args.kwargs["negative_prompt"] is None


def test_generate_rejects_non_image_output() -> None:
    """Given a pipeline that returns arrays, When generated, Then a TypeError is raised."""
    pipeline = MagicMock(spec=FluxPipeline)
    pipeline.return_value = SimpleNamespace(images=[[0, 0, 0]])
    generator = FluxGenerator(pipeline, device_name="cpu", max_sequence_length=64)

    with pytest.raises(TypeError, match="PIL image"):
        generator.generate(GenerationRequest(prompt="a cat"), seed=1)


def test_cpu_device_is_described_by_name() -> None:
    """Given the CPU device, When described, Then its name is returned unchanged."""
    assert describe_device("cpu") == "cpu"


def test_cuda_device_is_described_by_gpu_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given a CUDA device, When described, Then the GPU model name is returned."""
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "NVIDIA L40S")

    assert describe_device("cuda") == "NVIDIA L40S"


def test_gpu_check_is_skipped_on_cpu() -> None:
    """Given CPU settings, When checked, Then no GPU is required."""
    ensure_gpu_memory(CPU_SETTINGS)


def test_gpu_check_fails_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given no CUDA runtime, When checked, Then a RuntimeError explains why."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        ensure_gpu_memory(GPU_SETTINGS)


def _gpu_with_memory(monkeypatch: pytest.MonkeyPatch, memory_gib: int) -> None:
    properties = SimpleNamespace(total_memory=memory_gib * GIB)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: properties)


@pytest.mark.parametrize(
    "memory_gib",
    [pytest.param(40, id="40-gib-boundary"), pytest.param(48, id="48-gib")],
)
def test_gpu_check_accepts_enough_memory(monkeypatch: pytest.MonkeyPatch, memory_gib: int) -> None:
    """Given a GPU with at least 40 GiB, When checked, Then the worker may start."""
    _gpu_with_memory(monkeypatch, memory_gib)

    ensure_gpu_memory(GPU_SETTINGS)


def test_gpu_check_rejects_small_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given a 24 GiB GPU, When checked against 40 GiB, Then a RuntimeError gives both sizes."""
    _gpu_with_memory(monkeypatch, 24)

    with pytest.raises(RuntimeError, match=r"GPU has 24 GiB, but .* needs at least 40 GiB"):
        ensure_gpu_memory(GPU_SETTINGS)
