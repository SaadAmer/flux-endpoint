from collections.abc import Iterator

import httpx
import pytest
import respx

from flux_endpoint.http import JOB_API_URL, RetryTransport, create_runpod_client

HTTP_DATE = "Wed, 21 Oct 2026 07:28:00 GMT"


class FakeTransport(httpx.BaseTransport):
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = iter(responses)
        self.requests: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return next(self.responses)


class FakeSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


@pytest.fixture
def sleep() -> FakeSleep:
    """Records backoff delays instead of sleeping."""
    return FakeSleep()


def _send(
    method: str, statuses: list[int], sleep: FakeSleep, headers: dict[str, str] | None = None
) -> tuple[httpx.Response, FakeTransport]:
    transport = FakeTransport([httpx.Response(status, headers=headers) for status in statuses])
    retrying = RetryTransport(transport, attempts=4, backoff_s=1.0, max_delay_s=3.0, sleep=sleep)
    with httpx.Client(transport=retrying, base_url="https://api.runpod.io/v2") as client:
        return client.request(method, "/serverless"), transport


def test_throttled_request_waits_for_retry_after(sleep: FakeSleep) -> None:
    """Given a 429 with Retry-After, When sent, Then the client waits that long and retries."""
    response, transport = _send("GET", [429, 200], sleep, headers={"Retry-After": "2"})

    assert response.status_code == 200
    assert len(transport.requests) == 2
    assert sleep.delays == [2.0]


def test_backoff_doubles_up_to_the_cap(sleep: FakeSleep) -> None:
    """Given three 503s on a GET, When sent, Then delays double from 1 s and stop at 3 s."""
    response, _ = _send("GET", [503, 503, 503, 200], sleep)

    assert response.status_code == 200
    assert sleep.delays == [1.0, 2.0, 3.0]


def test_last_failure_is_returned_after_final_attempt(sleep: FakeSleep) -> None:
    """Given a GET that keeps failing, When sent, Then the fourth response is returned."""
    response, transport = _send("GET", [503, 503, 503, 503], sleep)

    assert response.status_code == 503
    assert len(transport.requests) == 4


@pytest.mark.parametrize(
    ("method", "status", "calls"),
    [
        pytest.param("POST", 429, 2, id="throttled-post-is-retried"),
        pytest.param("POST", 503, 1, id="unavailable-post-is-not-retried"),
        pytest.param("PATCH", 502, 1, id="bad-gateway-patch-is-not-retried"),
        pytest.param("DELETE", 504, 2, id="timed-out-delete-is-retried"),
        pytest.param("GET", 500, 1, id="internal-error-is-not-retried"),
    ],
)
def test_only_safe_requests_are_retried(
    sleep: FakeSleep, method: str, status: int, calls: int
) -> None:
    """Given a failure status, When sent, Then a write retries only after a 429."""
    _, transport = _send(method, [status, 200], sleep)

    assert len(transport.requests) == calls


def test_http_date_retry_after_falls_back_to_backoff(sleep: FakeSleep) -> None:
    """Given Retry-After as an HTTP date, When sent, Then the normal backoff applies."""
    _send("GET", [429, 200], sleep, headers={"Retry-After": HTTP_DATE})

    assert sleep.delays == [1.0]


def test_client_authenticates_with_bearer_key() -> None:
    """Given an API key, When a client is created, Then requests carry it as a bearer token."""
    with create_runpod_client("rp-test-key", JOB_API_URL) as client:
        assert client.headers["Authorization"] == "Bearer rp-test-key"
        assert client.headers["User-Agent"].startswith("flux-endpoint/")
        assert str(client.base_url) == f"{JOB_API_URL}/"


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    """Mocked Runpod job API."""
    with respx.mock(base_url=JOB_API_URL, assert_all_called=True) as mock:
        yield mock


def test_client_retries_throttled_requests(router: respx.MockRouter) -> None:
    """Given the API throttles once, When a client sends a request, Then it succeeds on retry."""
    router.get("/ep-1/health").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "0"}),
            httpx.Response(200, json={"workers": {}}),
        ]
    )

    with create_runpod_client("rp-test-key", JOB_API_URL) as client:
        response = client.get("/ep-1/health")

    assert response.status_code == 200


def test_client_requires_api_key() -> None:
    """Given an empty API key, When a client is created, Then a ValueError names the variable."""
    with pytest.raises(ValueError, match="RUNPOD_API_KEY"):
        create_runpod_client("", JOB_API_URL)
