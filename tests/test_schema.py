from typing import Any

import pytest
from pydantic import ValidationError

from flux_endpoint.schema import (
    MAX_SEED,
    RANDOM_SEED,
    GenerationRequest,
    GenerationResult,
    format_validation_error,
)


def test_prompt_only_request_uses_defaults() -> None:
    """Given only a prompt, When validated, Then the documented defaults apply."""
    request = GenerationRequest.model_validate({"prompt": "a lighthouse at dusk"})

    assert request.width == 1024
    assert request.height == 1024
    assert request.num_inference_steps == 28
    assert request.guidance == 3.5
    assert request.true_cfg_scale == 1.0
    assert request.seed == RANDOM_SEED
    assert request.image_format == "jpeg"
    assert request.negative_prompt is None


def test_prompt_whitespace_is_stripped() -> None:
    """Given a padded prompt, When validated, Then surrounding whitespace is removed."""
    request = GenerationRequest.model_validate({"prompt": "  a red fox  "})

    assert request.prompt == "a red fox"


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"width": 256, "height": 1536}, id="size-bounds"),
        pytest.param({"num_inference_steps": 1}, id="min-steps"),
        pytest.param({"num_inference_steps": 50}, id="max-steps"),
        pytest.param({"guidance": 0.0}, id="min-guidance"),
        pytest.param({"guidance": 10.0}, id="max-guidance"),
        pytest.param({"seed": 0}, id="zero-seed"),
        pytest.param({"seed": MAX_SEED}, id="max-seed"),
        pytest.param({"image_format": "png"}, id="png"),
        pytest.param({"negative_prompt": "blurry", "true_cfg_scale": 4.0}, id="true-cfg"),
    ],
)
def test_boundary_values_are_accepted(overrides: dict[str, Any]) -> None:
    """Given a value on an allowed boundary, When validated, Then it is kept."""
    request = GenerationRequest.model_validate({"prompt": "a cat", **overrides})

    for field, value in overrides.items():
        assert getattr(request, field) == value


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        pytest.param({"prompt": ""}, "prompt", id="empty-prompt"),
        pytest.param({"prompt": "   "}, "prompt", id="blank-prompt"),
        pytest.param({"prompt": "x" * 2001}, "prompt", id="long-prompt"),
        pytest.param({"width": 1000}, "width", id="width-not-multiple-of-64"),
        pytest.param({"width": 192}, "width", id="width-too-small"),
        pytest.param({"height": 1600}, "height", id="height-too-large"),
        pytest.param({"num_inference_steps": 0}, "num_inference_steps", id="no-steps"),
        pytest.param({"num_inference_steps": 51}, "num_inference_steps", id="many-steps"),
        pytest.param({"guidance": -0.5}, "guidance", id="negative-guidance"),
        pytest.param({"guidance": 10.5}, "guidance", id="high-guidance"),
        pytest.param({"true_cfg_scale": 0.5}, "true_cfg_scale", id="low-true-cfg"),
        pytest.param({"seed": -2}, "seed", id="seed-below-random"),
        pytest.param({"seed": MAX_SEED + 1}, "seed", id="seed-above-max"),
        pytest.param({"image_format": "gif"}, "image_format", id="unsupported-format"),
        pytest.param({"guidance_scale": 3.5}, "guidance_scale", id="unknown-field"),
    ],
)
def test_invalid_values_are_rejected(overrides: dict[str, Any], field: str) -> None:
    """Given a value outside the contract, When validated, Then the field is named."""
    with pytest.raises(ValidationError) as caught:
        GenerationRequest.model_validate({"prompt": "a cat", **overrides})

    assert caught.value.errors()[0]["loc"] == (field,)


def test_missing_prompt_is_rejected() -> None:
    """Given no prompt, When validated, Then validation fails on the prompt."""
    with pytest.raises(ValidationError) as caught:
        GenerationRequest.model_validate({})

    assert caught.value.errors()[0]["loc"] == ("prompt",)


def test_job_input_omits_unused_extensions() -> None:
    """Given default extension fields, When serialized, Then only Public Endpoint fields remain."""
    request = GenerationRequest(prompt="a cat", seed=7)

    assert request.to_job_input() == {
        "prompt": "a cat",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 28,
        "guidance": 3.5,
        "seed": 7,
        "image_format": "jpeg",
    }


def test_job_input_keeps_extensions_in_use() -> None:
    """Given a negative prompt with true CFG, When serialized, Then both fields are kept."""
    request = GenerationRequest(prompt="a cat", negative_prompt="dog", true_cfg_scale=2.0)

    job_input = request.to_job_input()

    assert job_input["negative_prompt"] == "dog"
    assert job_input["true_cfg_scale"] == 2.0


@pytest.mark.parametrize(
    "location",
    [
        pytest.param({}, id="no-image"),
        pytest.param({"image_base64": "aGk=", "image_url": "https://cdn/1.png"}, id="both"),
    ],
)
def test_result_needs_exactly_one_image_location(location: dict[str, str]) -> None:
    """Given no image or two copies of it, When a result is built, Then validation fails."""
    with pytest.raises(ValidationError, match="exactly one of image_base64 and image_url"):
        GenerationResult(
            **location,
            image_format="png",
            width=256,
            height=256,
            seed=1,
            gpu="cpu",
            inference_ms=1,
            encode_ms=1,
        )


def test_validation_errors_are_summarized_per_field() -> None:
    """Given two invalid fields, When formatted, Then each field and reason appear once."""
    with pytest.raises(ValidationError) as caught:
        GenerationRequest.model_validate({"prompt": "a cat", "width": 100, "seed": -9})

    message = format_validation_error(caught.value)

    assert message.startswith("width: ")
    assert "; seed: " in message


def test_field_names_with_control_characters_are_escaped() -> None:
    """Given a field name with a line break, When formatted, Then it is escaped on one line."""
    with pytest.raises(ValidationError) as caught:
        GenerationRequest.model_validate({"prompt": "a cat", "x\nINFO forged": 1})

    assert format_validation_error(caught.value) == (
        "'x\\nINFO forged': Extra inputs are not permitted"
    )


def test_validation_error_without_field_is_labeled_input() -> None:
    """Given a non-object input, When formatted, Then the error is labeled input."""
    with pytest.raises(ValidationError) as caught:
        GenerationRequest.model_validate("a cat")

    assert format_validation_error(caught.value).startswith("input: ")
