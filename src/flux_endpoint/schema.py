from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ImageFormat = Literal["png", "jpeg"]

RANDOM_SEED = -1
MAX_SEED = 2**32 - 1
MAX_PROMPT_LENGTH = 2000


class GenerationRequest(BaseModel):
    """Text-to-image request.

    Field names and limits match Runpod's FLUX.1-dev Public Endpoint.
    ``true_cfg_scale`` is an extension. ``negative_prompt`` has an effect only
    when ``true_cfg_scale`` is above 1.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_LENGTH)
    negative_prompt: str | None = Field(default=None, max_length=MAX_PROMPT_LENGTH)
    width: int = Field(default=1024, ge=256, le=1536, multiple_of=64)
    height: int = Field(default=1024, ge=256, le=1536, multiple_of=64)
    num_inference_steps: int = Field(default=28, ge=1, le=50)
    guidance: float = Field(default=3.5, ge=0.0, le=10.0)
    true_cfg_scale: float = Field(default=1.0, ge=1.0, le=10.0)
    seed: int = Field(default=RANDOM_SEED, ge=RANDOM_SEED, le=MAX_SEED)
    image_format: ImageFormat = "jpeg"

    def to_job_input(self) -> dict[str, Any]:
        """Serialize the request as a job input.

        Extension fields left at their defaults are omitted, so the same body
        is valid for the Public Endpoint.
        """
        unused = {"true_cfg_scale"} if self.true_cfg_scale == 1.0 else set()
        return self.model_dump(exclude_none=True, exclude=unused)


class GenerationResult(BaseModel):
    """Worker output for one generated image.

    The image is inline as ``image_base64``, or at ``image_url`` when the worker
    uploads to an output bucket.
    """

    model_config = ConfigDict(frozen=True)

    image_base64: str | None = None
    image_url: str | None = None
    image_format: ImageFormat
    width: int
    height: int
    seed: int
    gpu: str
    inference_ms: int
    encode_ms: int

    @model_validator(mode="after")
    def _check_one_image_location(self) -> Self:
        if (self.image_base64 is None) == (self.image_url is None):
            raise ValueError("Set exactly one of image_base64 and image_url.")
        return self


def format_validation_error(error: ValidationError) -> str:
    """Summarize a validation error as ``field: reason`` pairs.

    Unknown field names come from the client, so a name with control characters
    is escaped. The summary then cannot add lines to a log.
    """
    return "; ".join(f"{_field_path(detail['loc'])}: {detail['msg']}" for detail in error.errors())


def _field_path(location: tuple[int | str, ...]) -> str:
    path = ".".join(str(part) for part in location) or "input"
    return path if path.isprintable() else repr(path)
