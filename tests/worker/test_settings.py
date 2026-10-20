from pathlib import Path

import pytest
from pydantic import ValidationError

from flux_endpoint.worker.settings import WorkerSettings


def test_empty_environment_gives_production_defaults() -> None:
    """Given no FLUX_ variables, When settings load, Then GPU production defaults apply."""
    settings = WorkerSettings.from_env({})

    assert settings.model_dir == Path("/models/flux-1-dev")
    assert settings.device == "cuda"
    assert settings.dtype == "bfloat16"
    assert settings.max_sequence_length == 512
    assert settings.min_gpu_memory_gib == 40.0
    assert settings.output_bucket is None


def test_variables_override_defaults() -> None:
    """Given FLUX_ variables for a CPU run, When settings load, Then values are parsed."""
    settings = WorkerSettings.from_env(
        {
            "FLUX_MODEL_DIR": "/opt/models/tiny-flux",
            "FLUX_DEVICE": "cpu",
            "FLUX_DTYPE": "float32",
            "FLUX_MAX_SEQUENCE_LENGTH": "48",
            "FLUX_MIN_GPU_MEMORY_GIB": "0",
            "FLUX_OUTPUT_BUCKET": "flux-images",
            "PATH": "/usr/bin",
        }
    )

    assert settings.model_dir == Path("/opt/models/tiny-flux")
    assert settings.device == "cpu"
    assert settings.dtype == "float32"
    assert settings.max_sequence_length == 48
    assert settings.min_gpu_memory_gib == 0.0
    assert settings.output_bucket == "flux-images"


@pytest.mark.parametrize(
    "environ",
    [
        pytest.param({"FLUX_DEVICE": "cuda:1"}, id="unsupported-device"),
        pytest.param({"FLUX_DTYPE": "int8"}, id="unsupported-dtype"),
        pytest.param({"FLUX_MAX_SEQUENCE_LENGTH": "1024"}, id="sequence-too-long"),
        pytest.param({"FLUX_MIN_GPU_MEMORY_GIB": "-1"}, id="negative-memory"),
    ],
)
def test_invalid_variables_are_rejected(environ: dict[str, str]) -> None:
    """Given an invalid FLUX_ variable, When settings load, Then validation fails."""
    with pytest.raises(ValidationError):
        WorkerSettings.from_env(environ)
