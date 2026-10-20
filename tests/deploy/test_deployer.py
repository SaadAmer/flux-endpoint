from typing import Any

import pytest

from flux_endpoint.deploy.deployer import Deployer, DeploymentError
from flux_endpoint.deploy.spec import DeploymentSpec

SPEC = DeploymentSpec.model_validate(
    {
        "registry": {"name": "dockerhub-flux-endpoint"},
        "endpoint": {
            "name": "flux-1-dev",
            "image": "docker.io/saadamer/flux-endpoint:0.1.0",
            "gpu_pools": ["ADA_48_PRO"],
        },
    }
)
CREDENTIALS = {"DOCKERHUB_USERNAME": "saadamer", "DOCKERHUB_TOKEN": "pull-token"}


class FakeApi:
    def __init__(
        self,
        endpoints: list[dict[str, Any]] | None = None,
        registries: list[dict[str, Any]] | None = None,
        drift: dict[str, Any] | None = None,
    ) -> None:
        self.endpoints = {endpoint["id"]: endpoint for endpoint in endpoints or []}
        self.registries = list(registries or [])
        self.drift = drift or {}
        self.writes: list[tuple[str, Any]] = []

    def list_endpoints(self) -> list[dict[str, Any]]:
        return list(self.endpoints.values())

    def get_endpoint(self, endpoint_id: str) -> dict[str, Any]:
        return {**self.endpoints[endpoint_id], **self.drift}

    def create_endpoint(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.writes.append(("create", payload))
        endpoint = {
            "id": "ep-new",
            **payload,
            "requestUrls": {"run": "https://api.runpod.ai/v2/ep-new/run"},
        }
        self.endpoints["ep-new"] = endpoint
        return endpoint

    def update_endpoint(self, endpoint_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.writes.append(("update", payload))
        self.endpoints[endpoint_id] = {**self.endpoints[endpoint_id], **payload}
        return self.endpoints[endpoint_id]

    def delete_endpoint(self, endpoint_id: str) -> None:
        self.writes.append(("delete", endpoint_id))
        del self.endpoints[endpoint_id]

    def list_registries(self) -> list[dict[str, Any]]:
        return self.registries

    def create_registry(self, name: str, username: str, password: str) -> dict[str, Any]:
        self.writes.append(("create_registry", (name, username, password)))
        registry = {"id": "reg-new", "name": name}
        self.registries.append(registry)
        return registry


def test_new_endpoint_is_created_with_new_registry() -> None:
    """Given no endpoint or registry, When applied, Then both are created and linked."""
    api = FakeApi()

    deployment = Deployer(api, CREDENTIALS).apply(SPEC)

    assert deployment.endpoint_id == "ep-new"
    assert deployment.created
    assert deployment.run_url == "https://api.runpod.ai/v2/ep-new/run"
    assert api.writes[0] == (
        "create_registry",
        ("dockerhub-flux-endpoint", "saadamer", "pull-token"),
    )
    action, payload = api.writes[1]
    assert action == "create"
    assert payload["type"] == "QUEUE"
    assert payload["registry"] == "reg-new"


def test_existing_endpoint_is_updated_with_existing_registry() -> None:
    """Given an endpoint and registry with the spec names, When applied, Then both are reused."""
    api = FakeApi(
        endpoints=[{"id": "ep-1", "name": "flux-1-dev", "image": "old:tag"}],
        registries=[{"id": "reg-1", "name": "dockerhub-flux-endpoint"}],
    )

    deployment = Deployer(api, environ={}).apply(SPEC)

    assert deployment.endpoint_id == "ep-1"
    assert not deployment.created
    assert deployment.run_url is None
    action, payload = api.writes[0]
    assert action == "update"
    assert "type" not in payload
    assert payload["registry"] == "reg-1"
    assert api.endpoints["ep-1"]["image"] == "docker.io/saadamer/flux-endpoint:0.1.0"


def test_registry_with_other_name_is_not_reused() -> None:
    """Given only a registry with another name, When applied, Then a new one is created."""
    api = FakeApi(registries=[{"id": "reg-1", "name": "ghcr-team"}])

    Deployer(api, CREDENTIALS).apply(SPEC)

    assert api.writes[0][0] == "create_registry"
    assert api.writes[1][1]["registry"] == "reg-new"


def test_public_image_needs_no_registry() -> None:
    """Given a spec without a registry, When applied, Then no credential is created."""
    spec = DeploymentSpec(endpoint=SPEC.endpoint)
    api = FakeApi()

    Deployer(api, environ={}).apply(spec)

    assert [action for action, _ in api.writes] == ["create"]
    assert "registry" not in api.writes[0][1]


def test_missing_registry_credentials_are_reported() -> None:
    """Given no registry and no credentials, When applied, Then the variables are named."""
    with pytest.raises(DeploymentError, match="DOCKERHUB_USERNAME and DOCKERHUB_TOKEN"):
        Deployer(FakeApi(), environ={}).apply(SPEC)


def test_settings_that_do_not_stick_fail_the_deploy() -> None:
    """Given the API reads back other values, When applied, Then the drifted fields are named."""
    api = FakeApi(drift={"image": "docker.io/other:latest", "flashboot": "OFF"})

    with pytest.raises(DeploymentError, match="image, flashboot"):
        Deployer(api, CREDENTIALS).apply(SPEC)


def test_nested_settings_are_checked_after_deploy() -> None:
    """Given the API reads back another worker limit, When applied, Then the field is named."""
    api = FakeApi(drift={"workers": {"min": 0, "max": 5, "idleTimeout": 60}})

    with pytest.raises(DeploymentError, match=r"spec: workers\.max\.$"):
        Deployer(api, CREDENTIALS).apply(SPEC)


def test_extra_keys_in_read_back_list_items_still_match() -> None:
    """Given the API adds keys to a mounted volume, When applied, Then the deploy succeeds."""
    api = FakeApi(drift={"mounts": {"network": [], "global": [{"volumeId": "gv-1", "id": "m"}]}})

    deployment = Deployer(api, CREDENTIALS).apply(SPEC.with_overrides(global_volume_id="gv-1"))

    assert deployment.endpoint_id == "ep-new"


@pytest.mark.parametrize(
    ("mounts", "field"),
    [
        pytest.param(
            {"global": [{"volumeId": "gv-2"}]}, r"mounts\.global\[0\]\.volumeId", id="other"
        ),
        pytest.param({"global": []}, r"mounts\.global", id="none"),
    ],
)
def test_volume_that_does_not_stick_fails_the_deploy(mounts: dict[str, Any], field: str) -> None:
    """Given the API reads back another volume, When applied, Then the volume field is named."""
    api = FakeApi(drift={"mounts": mounts})

    with pytest.raises(DeploymentError, match=rf"spec: {field}\.$"):
        Deployer(api, CREDENTIALS).apply(SPEC.with_overrides(global_volume_id="gv-1"))


def test_empty_settings_missing_from_api_still_match() -> None:
    """Given the API leaves out an empty env, When applied, Then the deploy succeeds."""
    api = FakeApi(drift={"env": None})

    deployment = Deployer(api, CREDENTIALS).apply(SPEC)

    assert deployment.endpoint_id == "ep-new"


def test_duplicate_endpoint_names_are_rejected() -> None:
    """Given two endpoints with the spec name, When applied, Then nothing is changed."""
    api = FakeApi(
        endpoints=[{"id": "ep-1", "name": "flux-1-dev"}, {"id": "ep-2", "name": "flux-1-dev"}]
    )

    with pytest.raises(DeploymentError, match="2 endpoints are named 'flux-1-dev'"):
        Deployer(api, CREDENTIALS).apply(SPEC)

    assert api.writes == []


def test_teardown_deletes_named_endpoint() -> None:
    """Given an endpoint with the spec name, When torn down, Then it is deleted."""
    api = FakeApi(endpoints=[{"id": "ep-1", "name": "flux-1-dev"}])

    deleted = Deployer(api, environ={}).teardown("flux-1-dev")

    assert deleted == "ep-1"
    assert api.endpoints == {}


def test_teardown_without_endpoint_changes_nothing() -> None:
    """Given no endpoint with the name, When torn down, Then nothing is deleted."""
    api = FakeApi(endpoints=[{"id": "ep-9", "name": "other"}])

    assert Deployer(api, environ={}).teardown("flux-1-dev") is None
    assert api.writes == []
