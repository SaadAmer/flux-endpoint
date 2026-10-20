import json
import threading
import time
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol, Self

import httpx

from flux_endpoint.client.jobs import JobResult, JobTimeoutError
from flux_endpoint.schema import GenerationRequest

DEFAULT_PROMPTS = (
    "A lighthouse on a rocky coast at dusk, waves breaking, cinematic lighting",
    "A bowl of ramen on a wooden table, steam rising, shallow depth of field",
    "A red fox in fresh snow, early morning light, wildlife photography",
    "An isometric illustration of a small data center with glowing GPU racks",
    "A watercolor map of an imaginary island with mountains and rivers",
)


class BenchmarkStoppedError(Exception):
    """The benchmark stopped while the job was in flight."""


class _JobRunner(Protocol):
    def run(self, job_input: Mapping[str, Any], timeout_s: float) -> JobResult: ...


@dataclass(frozen=True)
class Sample:
    """Outcome and timings of one benchmark job."""

    job_id: str | None
    status: str
    wall_ms: int
    delay_ms: int | None = None
    execution_ms: int | None = None
    gpu: str | None = None
    cost_usd: float | None = None
    error: str | None = None
    variant: str | None = None
    worker_id: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "COMPLETED" and self.error is None

    @classmethod
    def from_result(cls, result: JobResult, wall_ms: int, variant: str | None = None) -> Self:
        """Build a sample from a finished job.

        ``gpu`` comes from this worker's output and ``cost_usd`` from a Public
        Endpoint's output, so one report can compare both.
        """
        output = result.output or {}
        return cls(
            job_id=result.job_id,
            status=result.status,
            wall_ms=wall_ms,
            delay_ms=result.delay_ms,
            execution_ms=result.execution_ms,
            gpu=output.get("gpu"),
            cost_usd=output.get("cost"),
            error=result.error,
            variant=variant,
            worker_id=result.worker_id,
        )


def build_job_inputs(
    prompts: Sequence[str], count: int, template: GenerationRequest
) -> list[dict[str, Any]]:
    """Build ``count`` job inputs that cycle through the prompts.

    Seeds count up from 0, so a rerun generates the same images.
    """
    base = template.model_dump()
    return [
        GenerationRequest.model_validate(
            {**base, "prompt": prompts[index % len(prompts)], "seed": index}
        ).to_job_input()
        for index in range(count)
    ]


def run_benchmark(
    jobs: _JobRunner,
    job_inputs: Sequence[Mapping[str, Any]],
    concurrency: int,
    *,
    timeout_s: float,
    stop: threading.Event | None = None,
) -> Generator[Sample, None, None]:
    """Run the job inputs with up to ``concurrency`` jobs in flight.

    Samples are yielded as their jobs end, so the caller can save each one at once.
    A job that exceeds ``timeout_s`` is recorded as ``CLIENT_TIMEOUT`` and a job
    whose response fails or cannot be read as ``CLIENT_ERROR``. Either way the run
    continues, so one failure does not lose the samples already paid for. When the
    caller stops early, jobs that have not started are never sent, and ``stop`` is
    set. Job clients that sleep with ``interruptible_sleep(stop)`` then cancel the
    jobs in flight.

    Raises:
        ValueError: If ``concurrency`` is less than 1.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1.")
    measure = partial(measure_job, jobs, timeout_s=timeout_s, clock=time.perf_counter)
    return _measure_as_completed(measure, job_inputs, concurrency, stop or threading.Event())


def interruptible_sleep(stop: threading.Event) -> Callable[[float], None]:
    """Return a sleep for job clients that raises as soon as ``stop`` is set."""

    def sleep(seconds: float) -> None:
        if stop.wait(seconds):
            raise BenchmarkStoppedError("The benchmark stopped while the job was in flight.")

    return sleep


def _measure_as_completed(
    measure: Callable[[Mapping[str, Any]], Sample],
    job_inputs: Iterable[Mapping[str, Any]],
    concurrency: int,
    stop: threading.Event,
) -> Generator[Sample, None, None]:
    pool = ThreadPoolExecutor(max_workers=concurrency)
    try:
        futures = [pool.submit(measure, job_input) for job_input in job_inputs]
        for future in as_completed(futures):
            yield future.result()
    finally:
        # Drop queued jobs before waking the ones in flight, so no new job starts.
        pool.shutdown(wait=False, cancel_futures=True)
        stop.set()
        pool.shutdown()


def measure_job(
    jobs: _JobRunner,
    job_input: Mapping[str, Any],
    *,
    timeout_s: float,
    clock: Callable[[], float],
    variant: str | None = None,
) -> Sample:
    """Run one job and record its outcome, including client timeouts and unreadable responses."""
    started = clock()
    try:
        result = jobs.run(job_input, timeout_s)
    except JobTimeoutError as error:
        return Sample(
            None, "CLIENT_TIMEOUT", _elapsed_ms(clock, started), error=str(error), variant=variant
        )
    except (httpx.HTTPError, KeyError, ValueError) as error:
        return Sample(
            None, "CLIENT_ERROR", _elapsed_ms(clock, started), error=str(error), variant=variant
        )
    return Sample.from_result(result, _elapsed_ms(clock, started), variant)


def write_samples(samples: Sequence[Sample], path: Path) -> None:
    """Append samples to a JSON Lines file, so runs against several endpoints combine."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.writelines(json.dumps(asdict(sample)) + "\n" for sample in samples)


def read_samples(path: Path) -> list[Sample]:
    """Read samples from a JSON Lines file."""
    with path.open(encoding="utf-8") as file:
        return [Sample(**json.loads(line)) for line in file if line.strip()]


def _elapsed_ms(clock: Callable[[], float], started: float) -> int:
    return round((clock() - started) * 1000)
