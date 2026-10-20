import json
from collections.abc import Iterator

import httpx
import pytest
import respx

from flux_endpoint.deploy.rest import RunpodApiError, RunpodRestClient

BASE_URL = "https://api.runpod.io/v2"


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    """Mocked Runpod REST API v2."""
    with respx.mock(base_url=BASE_URL, assert_all_called=True) as mock:
        yield mock


@pytest.fixture
def api() -> Iterator[RunpodRestClient]:
    """REST client pointed at the mocked API."""
    with httpx.Client(base_url=BASE_URL) as http:
        yield RunpodRestClient(http)


def test_endpoints_are_listed_across_pages(router: respx.MockRouter, api: RunpodRestClient) -> None:
    """Given two pages of endpoints, When listed, Then both pages are returned in order."""
    route = router.get("/serverless").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "endpoints": [{"id": "ep-1"}],
                    "pagination": {"nextCursor": "page-2", "hasNextPage": True},
                },
            ),
            httpx.Response(
                200,
                json={
                    "endpoints": [{"id": "ep-2"}],
                    "pagination": {"nextCursor": None, "hasNextPage": False},
                },
            ),
        ]
    )

    endpoints = api.list_endpoints()

    assert [endpoint["id"] for endpoint in endpoints] == ["ep-1", "ep-2"]
    assert route.calls[1].request.url.params["cursor"] == "page-2"


def test_endpoint_writes_use_rest_v2_methods(
    router: respx.MockRouter, api: RunpodRestClient
) -> None:
    """Given endpoint writes, When sent, Then POST, GET, PATCH and DELETE reach the right paths."""
    create = router.post("/serverless").respond(201, json={"id": "ep-1", "name": "flux"})
    router.get("/serverless/ep-1").respond(json={"id": "ep-1", "name": "flux"})
    update = router.patch("/serverless/ep-1").respond(json={"id": "ep-1", "name": "flux-2"})
    router.delete("/serverless/ep-1").respond(204)

    assert api.create_endpoint({"name": "flux"})["id"] == "ep-1"
    assert api.get_endpoint("ep-1")["name"] == "flux"
    assert api.update_endpoint("ep-1", {"name": "flux-2"})["name"] == "flux-2"
    api.delete_endpoint("ep-1")

    assert json.loads(create.calls.last.request.content) == {"name": "flux"}
    assert json.loads(update.calls.last.request.content) == {"name": "flux-2"}


def test_registries_are_listed_and_created(router: respx.MockRouter, api: RunpodRestClient) -> None:
    """Given registry calls, When made, Then credentials are sent only on create."""
    router.get("/registries").respond(json={"registries": [{"id": "reg-1", "name": "hub"}]})
    create = router.post("/registries").respond(201, json={"id": "reg-2", "name": "hub-2"})

    assert api.list_registries() == [{"id": "reg-1", "name": "hub"}]
    assert api.create_registry("hub-2", "saadamer", "pull-token")["id"] == "reg-2"
    assert json.loads(create.calls.last.request.content) == {
        "name": "hub-2",
        "username": "saadamer",
        "password": "pull-token",
    }


@pytest.mark.parametrize(
    ("password", "error", "shown"),
    [
        pytest.param("pull-token", "pull-token was rejected", "*** was rejected", id="echoed"),
        pytest.param("", "password is required", "password is required", id="empty"),
    ],
)
def test_registry_errors_never_show_the_password(
    router: respx.MockRouter, api: RunpodRestClient, password: str, error: str, shown: str
) -> None:
    """Given an error that echoes the request, When a registry is made, Then no password shows."""
    router.post("/registries").respond(400, json={"title": "Bad Request", "errors": [error]})

    with pytest.raises(RunpodApiError) as raised:
        api.create_registry("hub", "saadamer", password)

    assert str(raised.value) == f"Runpod API returned 400: Bad Request ({shown})"
    assert raised.value.status_code == 400
    assert raised.value.__suppress_context__


def test_gpu_catalog_is_listed(router: respx.MockRouter, api: RunpodRestClient) -> None:
    """Given the GPU catalog, When listed, Then the GPU entries are returned."""
    router.get("/catalog/gpus").respond(
        json={"gpus": [{"id": "NVIDIA L40S", "price": {"serverless": 1.75}}]}
    )

    assert api.list_gpu_types()[0]["id"] == "NVIDIA L40S"


def test_workers_are_listed_with_summary(router: respx.MockRouter, api: RunpodRestClient) -> None:
    """Given an endpoint with workers, When listed, Then workers and their summary return."""
    body = {
        "workers": [{"id": "w-1", "status": "RUNNING", "gpuCount": 1, "isStale": False}],
        "summary": {"running": 1, "idle": 0, "total": 1},
        "endpointVersion": 3,
    }
    router.get("/serverless/ep-1/workers").respond(json=body)

    assert api.list_workers("ep-1") == body


def test_worker_logs_are_read_from_the_event_stream(
    router: respx.MockRouter, api: RunpodRestClient
) -> None:
    """Given an event stream with a keep-alive, When logs are read, Then each data line returns."""
    stream = (
        "id: 2026-10-14T09:00:00Z\n"
        'data: {"ts": "2026-10-14T09:00:00Z", "source": "container", "line": "Loading"}\n\n'
        ": keep-alive\n\n"
        "id: 2026-10-14T09:00:05Z\n"
        'data: {"ts": "2026-10-14T09:00:05Z", "source": "container", "line": "Ready"}\n\n'
    )
    route = router.get("/serverless/ep-1/workers/w-1/logs").respond(
        text=stream, headers={"Content-Type": "text/event-stream"}
    )

    entries = list(api.stream_worker_logs("ep-1", "w-1", tail=50, source="container"))

    assert [entry["line"] for entry in entries] == ["Loading", "Ready"]
    assert route.calls.last.request.url.params["tail"] == "50"
    assert route.calls.last.request.url.params["source"] == "container"


def test_worker_log_errors_become_api_errors(
    router: respx.MockRouter, api: RunpodRestClient
) -> None:
    """Given an unknown worker, When logs are read, Then the API error is raised."""
    router.get("/serverless/ep-1/workers/w-9/logs").respond(
        404, json={"title": "Not Found", "status": 404, "detail": "Worker not found"}
    )

    with pytest.raises(RunpodApiError, match="returned 404: Worker not found"):
        list(api.stream_worker_logs("ep-1", "w-9"))


def test_problem_details_become_api_errors(router: respx.MockRouter, api: RunpodRestClient) -> None:
    """Given a 422 problem response, When a write fails, Then detail and errors are kept."""
    router.post("/serverless").respond(
        422,
        json={
            "title": "Unprocessable Entity",
            "status": 422,
            "detail": "Invalid endpoint",
            "errors": ["gpu.pools: unknown pool ADA_99"],
        },
    )

    with pytest.raises(RunpodApiError) as caught:
        api.create_endpoint({"name": "flux"})

    assert caught.value.status_code == 422
    assert str(caught.value) == (
        "Runpod API returned 422: Invalid endpoint (gpu.pools: unknown pool ADA_99)"
    )


def test_plain_text_errors_become_api_errors(
    router: respx.MockRouter, api: RunpodRestClient
) -> None:
    """Given a 502 with a text body, When a read fails, Then the body is the detail."""
    router.get("/registries").respond(502, text="Bad Gateway")

    with pytest.raises(RunpodApiError, match="Runpod API returned 502: Bad Gateway"):
        api.list_registries()


def test_json_errors_that_are_not_problems_keep_the_body(
    router: respx.MockRouter, api: RunpodRestClient
) -> None:
    """Given an error body that is a JSON list, When a read fails, Then the raw body is shown."""
    router.get("/registries").respond(500, json=["upstream", "timeout"])

    with pytest.raises(RunpodApiError, match=r'returned 500: \["upstream","timeout"\]'):
        api.list_registries()


def test_non_list_collection_is_an_api_error(
    router: respx.MockRouter, api: RunpodRestClient
) -> None:
    """Given an endpoints field that is not a list, When listed, Then an API error is raised."""
    router.get("/serverless").respond(json={"endpoints": {"id": "ep-1"}})

    with pytest.raises(RunpodApiError, match="Expected a JSON array"):
        api.list_endpoints()


def test_unexpected_response_shape_is_an_api_error(
    router: respx.MockRouter, api: RunpodRestClient
) -> None:
    """Given a list where an object is expected, When read, Then an API error is raised."""
    router.get("/serverless/ep-1").respond(json=["not", "an", "object"])

    with pytest.raises(RunpodApiError, match="Expected a JSON object"):
        api.get_endpoint("ep-1")
