import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx

from flux_endpoint.client.jobs import JobClient, JobResult, JobTimeoutError

BASE_URL = "https://api.runpod.ai/v2"
ENDPOINT_ID = "flux-endpoint-id"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    """Clock that advances only when the client sleeps."""
    return FakeClock()


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    """Mocked Runpod job API."""
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
        yield mock


@pytest.fixture
def jobs(clock: FakeClock) -> Iterator[JobClient]:
    """Job client with a two-second poll interval and a fake clock."""
    with httpx.Client(base_url=BASE_URL) as http:
        yield JobClient(
            http, ENDPOINT_ID, poll_interval_s=2.0, sleep=clock.sleep, clock=clock.monotonic
        )


def _status_payload(status: str, **fields: object) -> dict[str, object]:
    return {"id": "job-1", "status": status, **fields}


def test_submit_posts_input_and_returns_job_id(router: respx.MockRouter, jobs: JobClient) -> None:
    """Given a job input, When submitted, Then it is posted under input and the ID returned."""
    route = router.post(f"/{ENDPOINT_ID}/run").respond(json=_status_payload("IN_QUEUE"))

    job_id = jobs.submit({"prompt": "a cat"})

    assert job_id == "job-1"
    assert json.loads(route.calls.last.request.content) == {"input": {"prompt": "a cat"}}


def test_submit_sends_ttl_and_webhook(router: respx.MockRouter, jobs: JobClient) -> None:
    """Given a TTL and a webhook, When submitted, Then both go with the job input."""
    route = router.post(f"/{ENDPOINT_ID}/run").respond(json=_status_payload("IN_QUEUE"))

    jobs.submit({"prompt": "a cat"}, ttl_s=90, webhook="https://hooks.example.com/flux")

    assert json.loads(route.calls.last.request.content) == {
        "input": {"prompt": "a cat"},
        "policy": {"ttl": 90_000},
        "webhook": "https://hooks.example.com/flux",
    }


def test_short_ttl_is_raised_to_runpod_minimum(router: respx.MockRouter, jobs: JobClient) -> None:
    """Given a TTL below 10 s, When submitted, Then Runpod's 10 s minimum is sent."""
    route = router.post(f"/{ENDPOINT_ID}/run").respond(json=_status_payload("IN_QUEUE"))

    jobs.submit({"prompt": "a cat"}, ttl_s=2)

    assert json.loads(route.calls.last.request.content)["policy"] == {"ttl": 10_000}


def test_run_gives_job_a_ttl_beyond_the_client_timeout(
    router: respx.MockRouter, jobs: JobClient
) -> None:
    """Given a 600 s timeout, When run, Then the job expires 60 s after the client gives up."""
    route = router.post(f"/{ENDPOINT_ID}/run").respond(json=_status_payload("IN_QUEUE"))
    router.get(f"/{ENDPOINT_ID}/status/job-1").respond(json=_status_payload("COMPLETED"))

    jobs.run({"prompt": "a cat"}, timeout_s=600)

    assert json.loads(route.calls.last.request.content)["policy"] == {"ttl": 660_000}


def test_run_polls_until_job_completes(
    router: respx.MockRouter, jobs: JobClient, clock: FakeClock
) -> None:
    """Given a job that queues then runs, When run, Then polling stops at COMPLETED."""
    completed = _status_payload(
        "COMPLETED", delayTime=1500, executionTime=9000, workerId="worker-7", output={"seed": 3}
    )
    router.post(f"/{ENDPOINT_ID}/run").respond(json=_status_payload("IN_QUEUE"))
    router.get(f"/{ENDPOINT_ID}/status/job-1").mock(
        side_effect=[
            httpx.Response(200, json=_status_payload("IN_QUEUE")),
            httpx.Response(200, json=_status_payload("IN_PROGRESS")),
            httpx.Response(200, json=completed),
        ]
    )

    result = jobs.run({"prompt": "a cat"}, timeout_s=60)

    assert result == JobResult("job-1", "COMPLETED", 1500, 9000, "worker-7", {"seed": 3})
    assert result.succeeded
    assert clock.sleeps == [2.0, 2.0]


def test_run_cancels_job_after_timeout(
    router: respx.MockRouter, jobs: JobClient, clock: FakeClock
) -> None:
    """Given a job stuck in the queue, When the timeout passes, Then it is canceled."""
    router.post(f"/{ENDPOINT_ID}/run").respond(json=_status_payload("IN_QUEUE"))
    router.get(f"/{ENDPOINT_ID}/status/job-1").respond(json=_status_payload("IN_QUEUE"))
    cancel = router.post(f"/{ENDPOINT_ID}/cancel/job-1").respond(json=_status_payload("CANCELLED"))

    with pytest.raises(JobTimeoutError, match="job-1 is still IN_QUEUE after 5 s"):
        jobs.run({"prompt": "a cat"}, timeout_s=5)

    assert cancel.called
    assert clock.now >= 5


def test_run_cancels_job_when_polling_fails(router: respx.MockRouter, jobs: JobClient) -> None:
    """Given a status call that fails, When run, Then the job is canceled and the error raised."""
    router.post(f"/{ENDPOINT_ID}/run").respond(json=_status_payload("IN_QUEUE"))
    router.get(f"/{ENDPOINT_ID}/status/job-1").respond(status_code=500)
    cancel = router.post(f"/{ENDPOINT_ID}/cancel/job-1").respond(status_code=500)

    with pytest.raises(httpx.HTTPStatusError, match="/status/job-1"):
        jobs.run({"prompt": "a cat"}, timeout_s=60)

    assert cancel.called


def test_run_cancels_job_when_interrupted(router: respx.MockRouter) -> None:
    """Given Ctrl-C while waiting, When run, Then the job is canceled before the interrupt."""
    router.post(f"/{ENDPOINT_ID}/run").respond(json=_status_payload("IN_QUEUE"))
    router.get(f"/{ENDPOINT_ID}/status/job-1").respond(json=_status_payload("IN_QUEUE"))
    cancel = router.post(f"/{ENDPOINT_ID}/cancel/job-1").respond(json=_status_payload("CANCELLED"))

    def interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    with httpx.Client(base_url=BASE_URL) as http, pytest.raises(KeyboardInterrupt):
        JobClient(http, ENDPOINT_ID, sleep=interrupt).run({"prompt": "a cat"}, timeout_s=60)

    assert cancel.called


def test_health_returns_queue_and_worker_counts(router: respx.MockRouter, jobs: JobClient) -> None:
    """Given an endpoint, When its health is read, Then job and worker counts are returned."""
    health = {"jobs": {"inQueue": 2, "inProgress": 1}, "workers": {"idle": 1, "running": 1}}
    router.get(f"/{ENDPOINT_ID}/health").respond(json=health)

    assert jobs.health() == health


def test_http_errors_are_raised(router: respx.MockRouter, jobs: JobClient) -> None:
    """Given a 401 from the API, When submitting, Then an HTTPStatusError is raised."""
    router.post(f"/{ENDPOINT_ID}/run").respond(status_code=401)

    with pytest.raises(httpx.HTTPStatusError):
        jobs.submit({"prompt": "a cat"})


@pytest.mark.parametrize(
    ("payload", "succeeded"),
    [
        pytest.param(_status_payload("COMPLETED", output={"seed": 1}), True, id="completed"),
        pytest.param(_status_payload("COMPLETED", error="Invalid input"), False, id="error"),
        pytest.param(_status_payload("FAILED", error="boom"), False, id="failed"),
        pytest.param(_status_payload("TIMED_OUT"), False, id="timed-out"),
    ],
)
def test_terminal_results_report_success(payload: dict[str, Any], succeeded: bool) -> None:
    """Given a terminal status, When parsed, Then success means COMPLETED with no error."""
    result = JobResult.from_api(payload)

    assert result.is_terminal
    assert result.succeeded is succeeded


def test_non_object_output_is_dropped() -> None:
    """Given a list output, When parsed, Then the output is treated as missing."""
    result = JobResult.from_api(_status_payload("COMPLETED", output=["not", "a", "dict"]))

    assert result.output is None
    assert result.delay_ms is None
