import json
from collections.abc import Iterator, Mapping
from typing import Any, NoReturn

import httpx

_STREAM_TIMEOUT = httpx.Timeout(30.0, read=None)


class RunpodApiError(RuntimeError):
    """The Runpod REST API rejected a request or returned an unexpected body."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RunpodRestClient:
    """Runpod REST API v2 calls for the resources this project manages."""

    def __init__(self, http: httpx.Client) -> None:
        self._http = http

    def list_endpoints(self) -> list[dict[str, Any]]:
        """List every Serverless endpoint, following pagination."""
        endpoints: list[dict[str, Any]] = []
        params: dict[str, str] = {}
        while True:
            page = _as_object(self._request("GET", "/serverless", params=params))
            endpoints.extend(_as_objects(page.get("endpoints")))
            pagination = page.get("pagination") or {}
            if not pagination.get("hasNextPage"):
                return endpoints
            params = {"cursor": str(pagination["nextCursor"])}

    def get_endpoint(self, endpoint_id: str) -> dict[str, Any]:
        return _as_object(self._request("GET", f"/serverless/{endpoint_id}"))

    def create_endpoint(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return _as_object(self._request("POST", "/serverless", json=payload))

    def update_endpoint(self, endpoint_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return _as_object(self._request("PATCH", f"/serverless/{endpoint_id}", json=payload))

    def delete_endpoint(self, endpoint_id: str) -> None:
        self._request("DELETE", f"/serverless/{endpoint_id}")

    def list_registries(self) -> list[dict[str, Any]]:
        return _as_objects(_as_object(self._request("GET", "/registries")).get("registries"))

    def create_registry(self, name: str, username: str, password: str) -> dict[str, Any]:
        """Store a registry credential that workers use to pull a private image.

        Raises:
            RunpodApiError: If the API rejects the credential. The password is masked in
                the message, in case the API echoes the request.
        """
        body = {"name": name, "username": username, "password": password}
        try:
            return _as_object(self._request("POST", "/registries", json=body))
        except RunpodApiError as error:
            raise RunpodApiError(_masked(str(error), password), error.status_code) from None

    def list_gpu_types(self) -> list[dict[str, Any]]:
        """List the GPU catalog, including Serverless prices."""
        return _as_objects(_as_object(self._request("GET", "/catalog/gpus")).get("gpus"))

    def list_workers(self, endpoint_id: str) -> dict[str, Any]:
        """Return an endpoint's workers, a count per state and the endpoint version."""
        return _as_object(self._request("GET", f"/serverless/{endpoint_id}/workers"))

    def stream_worker_logs(
        self, endpoint_id: str, worker_id: str, *, tail: int = 100, source: str | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield a worker's recent log events, then follow new ones as they arrive.

        Each event has ``ts``, ``source`` and ``line``. The stream stays open
        until the worker stops or the caller stops iterating.
        """
        params = {"tail": str(tail), **({"source": source} if source else {})}
        path = f"/serverless/{endpoint_id}/workers/{worker_id}/logs"
        with self._http.stream("GET", path, params=params, timeout=_STREAM_TIMEOUT) as response:
            if response.is_error:
                response.read()
                _raise_api_error(response)
            for line in response.iter_lines():
                if line.startswith("data:"):
                    yield _as_object(json.loads(line.removeprefix("data:")))

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
    ) -> object:
        response = self._http.request(method, path, json=json, params=params)
        if response.is_error:
            _raise_api_error(response)
        return response.json() if response.content else None


def _raise_api_error(response: httpx.Response) -> NoReturn:
    raise RunpodApiError(
        f"Runpod API returned {response.status_code}: {_problem_detail(response)}",
        response.status_code,
    )


def _problem_detail(response: httpx.Response) -> str:
    try:
        problem = response.json()
    except ValueError:
        return response.text
    if not isinstance(problem, dict):
        return response.text
    detail = str(problem.get("detail") or problem.get("title") or response.text)
    errors = problem.get("errors") or []
    return f"{detail} ({'; '.join(str(error) for error in errors)})" if errors else detail


def _masked(text: str, secret: str) -> str:
    return text.replace(secret, "***") if secret else text


def _as_object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RunpodApiError(f"Expected a JSON object, got {type(value).__name__}.")
    return value


def _as_objects(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise RunpodApiError(f"Expected a JSON array, got {type(value).__name__}.")
    return [_as_object(item) for item in value]
