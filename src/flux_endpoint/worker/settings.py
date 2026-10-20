from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field

DeviceName = Literal["cuda", "cpu"]
DtypeName = Literal["bfloat16", "float16", "float32"]

ENV_PREFIX = "FLUX_"


class WorkerSettings(BaseModel):
    """Worker configuration, read from ``FLUX_*`` environment variables."""

    model_config = ConfigDict(frozen=True)

    model_dir: Path = Path("/models/flux-1-dev")
    device: DeviceName = "cuda"
    dtype: DtypeName = "bfloat16"
    max_sequence_length: int = Field(default=512, ge=1, le=512)
    min_gpu_memory_gib: float = Field(default=40.0, ge=0.0)
    output_bucket: str | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> Self:
        """Build settings from ``FLUX_<FIELD>`` variables, such as ``FLUX_DEVICE`` for ``device``.

        Unset variables keep their defaults.
        """
        values = {
            field: environ[f"{ENV_PREFIX}{field.upper()}"]
            for field in cls.model_fields
            if f"{ENV_PREFIX}{field.upper()}" in environ
        }
        return cls.model_validate(values)
