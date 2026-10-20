from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from flux_endpoint.deploy.spec import DeploymentSpec, EndpointSpec, load_spec

MINIMAL_ENDPOINT: dict[str, Any] = {
    "name": "flux-1-dev",
    "image": "docker.io/saadamer/flux-endpoint:0.1.0",
    "gpu_pools": ["ADA_48_PRO"],
}


def test_spec_file_is_loaded(tmp_path: Path) -> None:
    """Given a TOML spec with a registry and env, When loaded, Then every value is read."""
    spec_file = tmp_path / "endpoint.toml"
    spec_file.write_text(
        """
        [registry]
        name = "dockerhub-flux-endpoint"

        [endpoint]
        name = "flux-1-dev"
        image = "docker.io/saadamer/flux-endpoint:0.1.0"
        gpu_pools = ["ADA_48_PRO", "AMPERE_48"]
        workers_max = 3

        [endpoint.env]
        FLUX_DTYPE = "bfloat16"
        """
    )

    spec = load_spec(spec_file)

    assert spec.registry is not None
    assert spec.registry.name == "dockerhub-flux-endpoint"
    assert spec.registry.username_env == "DOCKERHUB_USERNAME"
    assert spec.endpoint.gpu_pools == ("ADA_48_PRO", "AMPERE_48")
    assert spec.endpoint.workers_max == 3
    assert spec.endpoint.env == {"FLUX_DTYPE": "bfloat16"}


def test_registry_is_optional() -> None:
    """Given a spec without a registry, When validated, Then the registry is None."""
    spec = DeploymentSpec.model_validate({"endpoint": MINIMAL_ENDPOINT})

    assert spec.registry is None


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"workers_min": 3, "workers_max": 2}, id="min-above-max"),
        pytest.param({"min_cuda_version": "13"}, id="cuda-without-minor"),
        pytest.param({"gpu_pools": []}, id="no-gpu-pools"),
        pytest.param({"idle_timeout_s": 0}, id="zero-idle-timeout"),
        pytest.param({"queue_delay_s": 0.1}, id="queue-delay-too-short"),
        pytest.param({"excluded_gpu_types": [""]}, id="empty-gpu-type"),
        pytest.param({"global_volume_id": ""}, id="empty-volume-id"),
        pytest.param({"data_center_ids": [""]}, id="empty-data-center"),
        pytest.param({"gpu_count": 1}, id="unknown-key"),
    ],
)
def test_invalid_endpoint_values_are_rejected(overrides: dict[str, Any]) -> None:
    """Given an invalid endpoint value, When validated, Then validation fails."""
    with pytest.raises(ValidationError):
        EndpointSpec.model_validate({**MINIMAL_ENDPOINT, **overrides})


def test_overrides_replace_only_given_fields() -> None:
    """Given a spec, When name, image and pools are overridden, Then other settings stay."""
    spec = DeploymentSpec.model_validate({"endpoint": {**MINIMAL_ENDPOINT, "workers_max": 4}})

    updated = spec.with_overrides(
        name="flux-1-dev-a6000",
        image="docker.io/saadamer/flux-endpoint:0.2.0",
        gpu_pools=["AMPERE_48"],
    )

    assert updated.endpoint.name == "flux-1-dev-a6000"
    assert updated.endpoint.image == "docker.io/saadamer/flux-endpoint:0.2.0"
    assert updated.endpoint.gpu_pools == ("AMPERE_48",)
    assert updated.endpoint.workers_max == 4
    assert spec.endpoint.image == MINIMAL_ENDPOINT["image"]


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"image": ""}, id="empty-image"),
        pytest.param({"gpu_pools": []}, id="no-gpu-pools"),
        pytest.param({"gpu_count": 2}, id="unknown-field"),
    ],
)
def test_invalid_overrides_are_rejected(overrides: dict[str, Any]) -> None:
    """Given an invalid override, When applied, Then validation fails."""
    spec = DeploymentSpec.model_validate({"endpoint": MINIMAL_ENDPOINT})

    with pytest.raises(ValidationError):
        spec.with_overrides(**overrides)


def test_settings_payload_maps_to_rest_v2_fields() -> None:
    """Given a full spec, When converted, Then the REST API v2 body uses its field names."""
    spec = EndpointSpec.model_validate(
        {
            **MINIMAL_ENDPOINT,
            "gpu_pools": ["ADA_48_PRO", "AMPERE_80"],
            "data_center_ids": ["EU-RO-1"],
            "env": {"FLUX_DTYPE": "bfloat16"},
        }
    )

    assert spec.settings_payload(registry_id="reg-1") == {
        "name": "flux-1-dev",
        "image": "docker.io/saadamer/flux-endpoint:0.1.0",
        "disk": 20,
        "env": {"FLUX_DTYPE": "bfloat16"},
        "registry": "reg-1",
        "gpu": {"pools": ["ADA_48_PRO", "AMPERE_80"], "count": 1, "minCudaVersion": "13.0"},
        "workers": {"min": 0, "max": 2, "idleTimeout": 60},
        "scaling": {"type": "QUEUE_DELAY", "queueDelay": 4.0},
        "flashboot": "FLASHBOOT",
        "timeout": 600_000,
        "dataCenterIds": ["EU-RO-1"],
    }


def test_settings_payload_omits_unset_options() -> None:
    """Given no registry, volume, exclusions or data centers, When converted, Then none is sent."""
    spec = EndpointSpec.model_validate({**MINIMAL_ENDPOINT, "flashboot": False})

    payload = spec.settings_payload(registry_id=None)

    assert "registry" not in payload
    assert "mounts" not in payload
    assert "excludedTypes" not in payload["gpu"]
    assert "dataCenterIds" not in payload
    assert payload["flashboot"] == "OFF"


def test_global_volume_and_gpu_exclusions_map_to_rest_v2_fields() -> None:
    """Given a Global Volume and an excluded GPU type, When converted, Then both are sent."""
    spec = EndpointSpec.model_validate(
        {
            **MINIMAL_ENDPOINT,
            "global_volume_id": "gv-flux",
            "excluded_gpu_types": ["NVIDIA L40"],
        }
    )

    payload = spec.settings_payload(registry_id=None)

    assert payload["mounts"] == {"global": [{"volumeId": "gv-flux"}]}
    assert payload["gpu"]["excludedTypes"] == ["NVIDIA L40"]
