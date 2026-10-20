import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from flux_endpoint.bench.runner import Sample, measure_job
from flux_endpoint.client.jobs import JobResult

_OPEN_JOB_STATES = ("inQueue", "inProgress")


class ScaleDownTimeoutError(TimeoutError):
    """The endpoint did not scale down, so the next job would not measure a cold start."""


class _EndpointJobs(Protocol):
    def health(self) -> dict[str, Any]: ...

    def run(self, job_input: Mapping[str, Any], timeout_s: float) -> JobResult: ...


class _Clock(Protocol):
    def monotonic(self) -> float: ...

    def sleep(self, seconds: float, /) -> None: ...


@dataclass(frozen=True)
class ScaleDownWait:
    """How the probe waits for an endpoint to scale down before each run.

    A job sent just after the worker list shows a worker as stopped can still reach its
    container, so the endpoint must stay scaled down for ``settle_s`` first.
    """

    timeout_s: float = 600.0
    poll_interval_s: float = 5.0
    settle_s: float = 30.0

    def __post_init__(self) -> None:
        if self.settle_s >= self.timeout_s:
            raise ValueError("The settle time must be shorter than the timeout.")


DEFAULT_WAIT = ScaleDownWait()


class ColdStartProbe:
    """Measures the cold starts of one Serverless endpoint.

    Each job waits in the queue while a worker starts and loads the model, so the queue
    time of a job sent while no worker runs is the cold start that a client sees. Workers
    come from the REST API worker list, because the job API's ``/health`` view keeps a
    worker counted as running for about 30 seconds after it stops.
    """

    def __init__(
        self,
        jobs: _EndpointJobs,
        list_workers: Callable[[], Mapping[str, Any]],
        *,
        wait: ScaleDownWait = DEFAULT_WAIT,
        clock: _Clock = time,
    ) -> None:
        self._jobs = jobs
        self._list_workers = list_workers
        self._wait = wait
        self._clock = clock

    def measure(
        self, job_input: Mapping[str, Any], runs: int, *, variant: str, timeout_s: float
    ) -> Iterator[Sample]:
        """Wait until the endpoint has scaled down, then time one job, ``runs`` times.

        Each sample is yielded when its job ends, so the caller can save it before the
        next run. A failure in a later run then does not lose the cold starts already paid for.

        Raises:
            ScaleDownTimeoutError: If a worker still runs after the scale-down timeout.
        """
        for _ in range(runs):
            self.wait_until_scaled_down()
            yield measure_job(
                self._jobs,
                job_input,
                timeout_s=timeout_s,
                clock=self._clock.monotonic,
                variant=variant,
            )

    def wait_until_scaled_down(self) -> None:
        """Wait until the endpoint has stayed ready for a cold start for the settle time.

        Raises:
            ScaleDownTimeoutError: If that does not happen within the timeout.
        """
        deadline = self._clock.monotonic() + self._wait.timeout_s
        settled_from: float | None = None
        while True:
            now = self._clock.monotonic()
            if not is_ready_for_cold_start(self._jobs.health(), self._list_workers()):
                settled_from = None
            elif settled_from is None:
                settled_from = now
            if settled_from is not None and now - settled_from >= self._wait.settle_s:
                return
            if now >= deadline:
                raise ScaleDownTimeoutError(
                    f"The endpoint did not scale down and stay that way within "
                    f"{self._wait.timeout_s:.0f} s."
                )
            self._clock.sleep(self._wait.poll_interval_s)


def is_ready_for_cold_start(health: Mapping[str, Any], workers: Mapping[str, Any]) -> bool:
    """Tell whether the next job would measure a cold start and nothing else.

    That needs no open job, no ``RUNNING`` worker, because one holds a loaded model during
    its jobs and its idle timeout, and an ``IDLE`` worker, which has the image and can
    start. A job sent while workers are still ``INITIALIZING`` would wait for the pull.

    Args:
        health: Body of the job API's ``/health`` response, for the open jobs.
        workers: Body of the REST API v2 worker list for the endpoint.

    Raises:
        ValueError: If the worker list has no ``workers`` array.
    """
    listed = workers.get("workers")
    if not isinstance(listed, list):
        raise ValueError("The worker list has no workers array.")
    jobs = health.get("jobs") or {}
    open_jobs = any(jobs.get(state, 0) for state in _OPEN_JOB_STATES)
    statuses = {worker.get("status") for worker in listed}
    return not open_jobs and "RUNNING" not in statuses and "IDLE" in statuses
