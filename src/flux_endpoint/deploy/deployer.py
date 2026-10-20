from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from flux_endpoint.deploy.spec import DeploymentSpec, RegistrySpec


class DeploymentError(RuntimeError):
    """The deployment cannot start, or did not reach the desired state."""


class _DeploymentApi(Protocol):
    def list_endpoints(self) -> list[dict[str, Any]]: ...

    def get_endpoint(self, endpoint_id: str) -> dict[str, Any]: ...

    def create_endpoint(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    def update_endpoint(self, endpoint_id: str, payload: dict[str, Any]) -> dict[str, Any]: ...

    def delete_endpoint(self, endpoint_id: str) -> None: ...

    def list_registries(self) -> list[dict[str, Any]]: ...

    def create_registry(self, name: str, username: str, password: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Deployment:
    """Outcome of applying a deployment spec."""

    endpoint_id: str
    created: bool
    run_url: str | None


class Deployer:
    """Makes a Serverless endpoint match a deployment spec.

    Endpoints and registry credentials are matched by name, so applying the
    same spec twice updates the endpoint instead of creating a second one.
    """

    def __init__(self, api: _DeploymentApi, environ: Mapping[str, str]) -> None:
        self._api = api
        self._environ = environ

    def apply(self, spec: DeploymentSpec) -> Deployment:
        """Create or update the endpoint, then read it back to confirm the settings.

        Raises:
            DeploymentError: If the endpoint name is ambiguous, registry
                credentials are missing, or the endpoint does not match the
                spec after the write.
        """
        existing = self._find_endpoint(spec.endpoint.name)
        registry_id = self._ensure_registry(spec.registry) if spec.registry else None
        settings = spec.endpoint.settings_payload(registry_id)
        if existing is None:
            written = self._api.create_endpoint({**settings, "type": "QUEUE"})
        else:
            written = self._api.update_endpoint(str(existing["id"]), settings)
        endpoint_id = str(written["id"])
        confirmed = self._api.get_endpoint(endpoint_id)
        if drifted := _drifted_fields(confirmed, settings):
            raise DeploymentError(
                f"Endpoint {endpoint_id} does not match the spec: {', '.join(drifted)}."
            )
        request_urls = confirmed.get("requestUrls") or {}
        return Deployment(endpoint_id, created=existing is None, run_url=request_urls.get("run"))

    def teardown(self, endpoint_name: str) -> str | None:
        """Delete the endpoint with the given name.

        Returns:
            The deleted endpoint's ID, or None when no endpoint has the name.
        """
        existing = self._find_endpoint(endpoint_name)
        if existing is None:
            return None
        endpoint_id = str(existing["id"])
        self._api.delete_endpoint(endpoint_id)
        return endpoint_id

    def _find_endpoint(self, name: str) -> dict[str, Any] | None:
        matches = [
            endpoint for endpoint in self._api.list_endpoints() if endpoint.get("name") == name
        ]
        if len(matches) > 1:
            raise DeploymentError(
                f"{len(matches)} endpoints are named {name!r}. Delete or rename the extras first."
            )
        return matches[0] if matches else None

    def _ensure_registry(self, spec: RegistrySpec) -> str:
        for registry in self._api.list_registries():
            if registry.get("name") == spec.name:
                return str(registry["id"])
        username = self._environ.get(spec.username_env)
        password = self._environ.get(spec.credential_env)
        if not (username and password):
            raise DeploymentError(
                f"Set {spec.username_env} and {spec.credential_env} to create "
                f"registry credential {spec.name!r}."
            )
        return str(self._api.create_registry(spec.name, username, password)["id"])


def _drifted_fields(
    actual: Mapping[str, Any], expected: Mapping[str, Any], prefix: str = ""
) -> list[str]:
    return [
        field
        for key, wanted in expected.items()
        for field in _drift(actual.get(key), wanted, f"{prefix}{key}")
    ]


def _drift(found: object, wanted: object, field: str) -> list[str]:
    if isinstance(wanted, Mapping) and isinstance(found, Mapping):
        return _drifted_fields(found, wanted, f"{field}.")
    if isinstance(wanted, list) and isinstance(found, list) and len(found) == len(wanted):
        return [
            drifted
            for index, (found_item, wanted_item) in enumerate(zip(found, wanted, strict=True))
            for drifted in _drift(found_item, wanted_item, f"{field}[{index}]")
        ]
    # The API can leave out empty fields, so an empty value matches a missing one.
    if found != wanted and (found or wanted):
        return [field]
    return []
