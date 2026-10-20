import tomllib
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonEmptyText = Annotated[str, Field(min_length=1)]


class RegistrySpec(BaseModel):
    """Registry credential that lets workers pull a private image.

    The username and password come from environment variables at deploy time,
    so they never appear in the spec file.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: NonEmptyText
    username_env: str = "DOCKERHUB_USERNAME"
    credential_env: str = "DOCKERHUB_TOKEN"


class EndpointSpec(BaseModel):
    """Desired state of a queue-based Serverless endpoint.

    Volumes are managed only when ``global_volume_id`` is set. Otherwise the deploy leaves
    the endpoint's volumes alone, because a cached model set in the console also lives under
    ``/runpod-volume`` and the API cannot read it back. Likewise, an empty
    ``data_center_ids`` leaves placement as it is, which for a new endpoint means the
    scheduler chooses.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: NonEmptyText
    image: NonEmptyText
    gpu_pools: tuple[NonEmptyText, ...] = Field(min_length=1)
    excluded_gpu_types: tuple[NonEmptyText, ...] = ()
    global_volume_id: NonEmptyText | None = None
    min_cuda_version: str = Field(default="13.0", pattern=r"^\d+\.\d+$")
    container_disk_gb: int = Field(default=20, ge=1)
    workers_min: int = Field(default=0, ge=0)
    workers_max: int = Field(default=2, ge=0)
    idle_timeout_s: int = Field(default=60, ge=1, le=3600)
    queue_delay_s: float = Field(default=4.0, ge=0.5)
    flashboot: bool = True
    execution_timeout_s: int = Field(default=600, ge=5)
    data_center_ids: tuple[NonEmptyText, ...] = ()
    env: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_worker_range(self) -> Self:
        if self.workers_max < self.workers_min:
            raise ValueError("workers_max must be at least workers_min")
        return self

    def settings_payload(self, registry_id: str | None) -> dict[str, Any]:
        """Build the REST API v2 fields that endpoint create and update share."""
        payload: dict[str, Any] = {
            "name": self.name,
            "image": self.image,
            "disk": self.container_disk_gb,
            "env": dict(self.env),
            "gpu": self._gpu_payload(),
            "workers": {
                "min": self.workers_min,
                "max": self.workers_max,
                "idleTimeout": self.idle_timeout_s,
            },
            "scaling": {"type": "QUEUE_DELAY", "queueDelay": self.queue_delay_s},
            "flashboot": "FLASHBOOT" if self.flashboot else "OFF",
            "timeout": self.execution_timeout_s * 1000,
        }
        if self.data_center_ids:
            payload["dataCenterIds"] = list(self.data_center_ids)
        if registry_id is not None:
            payload["registry"] = registry_id
        if self.global_volume_id is not None:
            payload["mounts"] = {"global": [{"volumeId": self.global_volume_id}]}
        return payload

    def _gpu_payload(self) -> dict[str, Any]:
        gpu: dict[str, Any] = {
            "pools": list(self.gpu_pools),
            "count": 1,
            "minCudaVersion": self.min_cuda_version,
        }
        if self.excluded_gpu_types:
            gpu["excludedTypes"] = list(self.excluded_gpu_types)
        return gpu


class DeploymentSpec(BaseModel):
    """Resources described by a deployment spec file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    registry: RegistrySpec | None = None
    endpoint: EndpointSpec

    def with_overrides(self, **fields: object) -> Self:
        """Return a validated copy with some endpoint fields replaced.

        Overriding ``name`` and ``gpu_pools`` gives one endpoint per GPU pool,
        which the benchmark needs to measure each pool on its own.
        """
        endpoint = EndpointSpec.model_validate({**self.endpoint.model_dump(), **fields})
        return self.model_copy(update={"endpoint": endpoint})


def load_spec(path: Path) -> DeploymentSpec:
    """Read and validate a TOML deployment spec."""
    with path.open("rb") as file:
        return DeploymentSpec.model_validate(tomllib.load(file))
