import contextlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Self

import httpx

TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"})
MIN_TTL_S = 10
TTL_MARGIN_S = 60


class JobTimeoutError(TimeoutError):
    """A job did not reach a terminal status before the client timeout."""


@dataclass(frozen=True)
class JobResult:
    """State of a Runpod Serverless job, as reported by ``/status``."""

    job_id: str
    status: str
    delay_ms: int | None = None
    execution_ms: int | None = None
    worker_id: str | None = None
    output: dict[str, Any] | None = None
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def succeeded(self) -> bool:
        """Whether the job completed without a handler error."""
        return self.status == "COMPLETED" and self.error is None

    @classmethod
    def from_api(cls, payload: Mapping[str, Any]) -> Self:
        output = payload.get("output")
        error = payload.get("error")
        return cls(
            job_id=str(payload["id"]),
            status=str(payload["status"]),
            delay_ms=payload.get("delayTime"),
            execution_ms=payload.get("executionTime"),
            worker_id=payload.get("workerId"),
            output=output if isinstance(output, dict) else None,
            error=None if error is None else str(error),
        )


class JobClient:
    """Submits jobs to one Serverless endpoint and waits for their results."""

    def __init__(
        self,
        http: httpx.Client,
        endpoint_id: str,
        *,
        poll_interval_s: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._http = http
        self._endpoint_id = endpoint_id
        self._poll_interval_s = poll_interval_s
        self._sleep = sleep
        self._clock = clock

    def submit(
        self,
        job_input: Mapping[str, Any],
        *,
        ttl_s: float | None = None,
        webhook: str | None = None,
    ) -> str:
        """Queue a job and return its ID.

        Args:
            job_input: Input for the worker's handler.
            ttl_s: Seconds after which Runpod drops the job, queued or running.
                Values under Runpod's 10 second minimum are raised to it.
            webhook: URL that Runpod sends the result to, for clients that do
                not poll.
        """
        body: dict[str, Any] = {"input": dict(job_input)}
        if ttl_s is not None:
            body["policy"] = {"ttl": round(max(ttl_s, MIN_TTL_S) * 1000)}
        if webhook is not None:
            body["webhook"] = webhook
        response = self._http.post(f"/{self._endpoint_id}/run", json=body)
        response.raise_for_status()
        return str(response.json()["id"])

    def status(self, job_id: str) -> JobResult:
        response = self._http.get(f"/{self._endpoint_id}/status/{job_id}")
        response.raise_for_status()
        return JobResult.from_api(response.json())

    def health(self) -> dict[str, Any]:
        """Return the endpoint's job counts by state and its worker counts."""
        response = self._http.get(f"/{self._endpoint_id}/health")
        response.raise_for_status()
        body = response.json()
        return body if isinstance(body, dict) else {}

    def cancel(self, job_id: str) -> None:
        self._http.post(f"/{self._endpoint_id}/cancel/{job_id}").raise_for_status()

    def run(self, job_input: Mapping[str, Any], timeout_s: float) -> JobResult:
        """Submit a job and poll until it reaches a terminal status.

        If the client stops waiting for any reason, including a timeout, an
        HTTP error or Ctrl-C, it cancels the job so the job stops billing. The
        job's TTL also expires shortly after ``timeout_s``, in case the client
        exits before it can cancel.

        Raises:
            JobTimeoutError: If the job is still queued or running after
                ``timeout_s``.
        """
        job_id = self.submit(job_input, ttl_s=timeout_s + TTL_MARGIN_S)
        try:
            return self._wait(job_id, timeout_s)
        except BaseException:
            with contextlib.suppress(httpx.HTTPError):
                self.cancel(job_id)
            raise

    def _wait(self, job_id: str, timeout_s: float) -> JobResult:
        deadline = self._clock() + timeout_s
        while not (result := self.status(job_id)).is_terminal:
            if self._clock() >= deadline:
                raise JobTimeoutError(
                    f"Job {job_id} is still {result.status} after {timeout_s:.0f} s."
                )
            self._sleep(self._poll_interval_s)
        return result
