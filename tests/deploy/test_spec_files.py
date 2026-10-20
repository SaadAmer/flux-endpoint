from pathlib import Path

import pytest

from flux_endpoint.deploy.spec import load_spec

DEPLOY_DIR = Path(__file__).parents[2] / "deploy"
SPEC_FILES = sorted(DEPLOY_DIR.rglob("*.toml"))
SPEC_NAMES = [path.relative_to(DEPLOY_DIR).as_posix() for path in SPEC_FILES]


@pytest.mark.parametrize("spec_file", SPEC_FILES, ids=SPEC_NAMES)
def test_shipped_spec_files_are_valid(spec_file: Path) -> None:
    """Given a spec file in deploy/, When loaded, Then it passes validation."""
    spec = load_spec(spec_file)

    assert spec.endpoint.gpu_pools


@pytest.mark.parametrize(
    "spec_name",
    ["cached-model.toml", "cold-start/cached-model.toml", "cold-start/global-volume.toml"],
    ids=["cached-model", "study-cached-model", "study-global-volume"],
)
def test_slim_image_specs_need_no_registry(spec_name: str) -> None:
    """Given a spec for the public slim image, When loaded, Then it has no registry credential."""
    spec = load_spec(DEPLOY_DIR / spec_name)

    assert spec.registry is None
    assert "flux-endpoint-slim" in spec.endpoint.image


def test_every_spec_file_is_checked() -> None:
    """Given the deploy folder, When listed, Then the main and cold-start specs are found."""
    assert set(SPEC_NAMES) >= {
        "endpoint.toml",
        "cached-model.toml",
        "cold-start/baked.toml",
        "cold-start/cached-model.toml",
        "cold-start/global-volume.toml",
    }
