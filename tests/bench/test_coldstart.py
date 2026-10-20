from collections.abc import Mapping
from typing import Any

import pytest

from flux_endpoint.bench.coldstart import (
    ColdStartProbe,
    ScaleDownTimeoutError,
    ScaleDownWait,
    is_ready_for_cold_start,
)
from flux_endpoint.client.jobs import JobResult, JobTimeoutError

JOB_INPUT = {"prompt": "a cat"}
NO_OPEN_JOBS = {"jobs": {"completed": 7, "failed": 0, "inProgress": 0, "inQueue": 0}}


def _workers(*statuses: str) -> dict[str, Any]:
    return {
        "workers": [{"id": f"w-{index}", "status": status} for index, status in enumerate(statuses)]
    }


SCALED_DOWN = _workers("IDLE", "IDLE", "INITIALIZING")
WARM = _workers("RUNNING", "IDLE")


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeEndpoint:
    def __init__(
        self,
        worker_lists: list[dict[str, Any]],
        clock: FakeClock,
        error: Exception | None = None,
        health: dict[str, Any] | None = None,
    ) -> None:
        self.worker_lists = worker_lists
        self.events: list[str] = []
        self.job_runs: list[tuple[Mapping[str, Any], float]] = []
        self._clock = clock
        self._error = error
        self._health = health or NO_OPEN_JOBS

    def health(self) -> dict[str, Any]:
        return self._health

    def list_workers(self) -> dict[str, Any]:
        self.events.append("workers")
        if len(self.worker_lists) > 1:
            return self.worker_lists.pop(0)
        return self.worker_lists[0]

    def run(self, job_input: Mapping[str, Any], timeout_s: float) -> JobResult:
        self.events.append("run")
        self.job_runs.append((job_input, timeout_s))
        self._clock.now += 50.0
        if self._error is not None:
            raise self._error
        return JobResult("job-1", "COMPLETED", delay_ms=45_000, execution_ms=5_000)


@pytest.fixture
def clock() -> FakeClock:
    """Clock that advances only when the code under test sleeps or runs a job."""
    return FakeClock()


def _probe(endpoint: FakeEndpoint, clock: FakeClock, settle_s: float = 0) -> ColdStartProbe:
    wait = ScaleDownWait(timeout_s=12, poll_interval_s=5, settle_s=settle_s)
    return ColdStartProbe(endpoint, endpoint.list_workers, wait=wait, clock=clock)


def test_scale_down_wait_polls_until_no_worker_runs(clock: FakeClock) -> None:
    """Given a worker in its idle timeout, When waiting, Then polling ends once it stops."""
    endpoint = FakeEndpoint([WARM, WARM, SCALED_DOWN], clock)

    _probe(endpoint, clock).wait_until_scaled_down()

    assert endpoint.events == ["workers", "workers", "workers"]
    assert clock.sleeps == [5, 5]


def test_scale_down_wait_lasts_the_settle_time(clock: FakeClock) -> None:
    """Given a worker that just stopped, When waiting, Then it stays scaled down for 10 s first."""
    endpoint = FakeEndpoint([WARM, SCALED_DOWN], clock)

    _probe(endpoint, clock, settle_s=10).wait_until_scaled_down()

    assert endpoint.events == ["workers"] * 4
    assert clock.sleeps == [5, 5, 5]


def test_settle_time_restarts_when_a_worker_starts_again(clock: FakeClock) -> None:
    """Given a worker that starts during the settle time, When waiting, Then the wait restarts."""
    endpoint = FakeEndpoint([SCALED_DOWN, WARM, SCALED_DOWN], clock)

    _probe(endpoint, clock, settle_s=5).wait_until_scaled_down()

    assert endpoint.events == ["workers"] * 4
    assert clock.sleeps == [5, 5, 5]


def test_scale_down_wait_gives_up_after_the_timeout(clock: FakeClock) -> None:
    """Given a worker that stays warm, When waiting, Then a timeout names the limit."""
    endpoint = FakeEndpoint([WARM], clock)

    with pytest.raises(ScaleDownTimeoutError, match="within 12 s"):
        _probe(endpoint, clock).wait_until_scaled_down()


def test_each_cold_start_waits_for_scale_down_first(clock: FakeClock) -> None:
    """Given two runs, When measured, Then each job starts only after a scale-down check."""
    endpoint = FakeEndpoint([SCALED_DOWN], clock)

    samples = list(_probe(endpoint, clock).measure(JOB_INPUT, 2, variant="baked", timeout_s=900))

    assert endpoint.events == ["workers", "run", "workers", "run"]
    assert endpoint.job_runs == [(JOB_INPUT, 900), (JOB_INPUT, 900)]
    assert [(sample.variant, sample.delay_ms, sample.wall_ms) for sample in samples] == [
        ("baked", 45_000, 50_000),
        ("baked", 45_000, 50_000),
    ]


def test_each_sample_is_handed_over_before_the_next_run(clock: FakeClock) -> None:
    """Given a worker that stays warm after run 1, When measured, Then sample 1 is already out."""
    endpoint = FakeEndpoint([SCALED_DOWN, WARM], clock)
    samples = _probe(endpoint, clock).measure(JOB_INPUT, 2, variant="baked", timeout_s=900)

    first = next(samples)

    with pytest.raises(ScaleDownTimeoutError):
        next(samples)
    assert first.job_id == "job-1"


def test_cold_start_that_times_out_is_recorded(clock: FakeClock) -> None:
    """Given a job that outlasts the client, When measured, Then the sample keeps its variant."""
    endpoint = FakeEndpoint([SCALED_DOWN], clock, error=JobTimeoutError("Job job-1 is queued"))

    samples = list(
        _probe(endpoint, clock).measure(JOB_INPUT, 1, variant="global-volume", timeout_s=900)
    )

    assert [(sample.status, sample.variant) for sample in samples] == [
        ("CLIENT_TIMEOUT", "global-volume")
    ]


def test_open_job_keeps_the_probe_waiting(clock: FakeClock) -> None:
    """Given a queued job and idle workers, When waiting, Then the probe times out."""
    endpoint = FakeEndpoint([SCALED_DOWN], clock, health={"jobs": {"inQueue": 1}})

    with pytest.raises(ScaleDownTimeoutError):
        _probe(endpoint, clock).wait_until_scaled_down()


def test_settle_time_must_be_shorter_than_the_timeout() -> None:
    """Given a settle time as long as the timeout, When built, Then a ValueError is raised."""
    with pytest.raises(ValueError, match="shorter than the timeout"):
        ScaleDownWait(timeout_s=30, settle_s=30)


@pytest.mark.parametrize(
    ("health", "workers", "ready"),
    [
        pytest.param(NO_OPEN_JOBS, SCALED_DOWN, True, id="idle-and-initializing"),
        pytest.param({}, _workers("IDLE"), True, id="one-idle-worker"),
        pytest.param(NO_OPEN_JOBS, _workers(), False, id="no-workers"),
        pytest.param(NO_OPEN_JOBS, _workers("INITIALIZING"), False, id="image-still-pulling"),
        pytest.param({}, _workers("THROTTLED", "UNHEALTHY"), False, id="no-idle-worker"),
        pytest.param(NO_OPEN_JOBS, WARM, False, id="worker-in-idle-timeout"),
        pytest.param({"jobs": {"inQueue": 1}}, SCALED_DOWN, False, id="queued-job"),
        pytest.param({"jobs": {"inProgress": 1}}, SCALED_DOWN, False, id="job-in-progress"),
    ],
)
def test_cold_start_needs_an_idle_worker_and_nothing_running(
    health: dict[str, Any], workers: dict[str, Any], ready: bool
) -> None:
    """Given health and a worker list, When checked, Then only idle and nothing open is ready."""
    assert is_ready_for_cold_start(health, workers) is ready


@pytest.mark.parametrize(
    "workers",
    [pytest.param({}, id="no-workers-field"), pytest.param({"workers": None}, id="null-workers")],
)
def test_worker_list_without_workers_is_rejected(workers: dict[str, Any]) -> None:
    """Given a worker list body without workers, When checked, Then a ValueError is raised."""
    with pytest.raises(ValueError, match="no workers array"):
        is_ready_for_cold_start(NO_OPEN_JOBS, workers)
