import threading
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from flux_endpoint.bench.runner import (
    BenchmarkStoppedError,
    Sample,
    build_job_inputs,
    interruptible_sleep,
    read_samples,
    run_benchmark,
    write_samples,
)
from flux_endpoint.client.jobs import JobResult, JobTimeoutError
from flux_endpoint.schema import GenerationRequest


class FakeJobs:
    def __init__(self, outcomes: Mapping[str, JobResult | Exception]) -> None:
        self.outcomes = outcomes
        self.inputs: list[Mapping[str, Any]] = []
        self.timeouts: list[float] = []

    def run(self, job_input: Mapping[str, Any], timeout_s: float) -> JobResult:
        self.inputs.append(job_input)
        self.timeouts.append(timeout_s)
        outcome = self.outcomes[job_input["prompt"]]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_job_inputs_cycle_prompts_with_fixed_seeds() -> None:
    """Given two prompts and a count of four, When built, Then prompts cycle and seeds count up."""
    template = GenerationRequest(prompt="unused", width=512, height=512)

    job_inputs = build_job_inputs(["a cat", "a dog"], count=4, template=template)

    assert [(job_input["prompt"], job_input["seed"]) for job_input in job_inputs] == [
        ("a cat", 0),
        ("a dog", 1),
        ("a cat", 2),
        ("a dog", 3),
    ]
    assert all(job_input["width"] == 512 for job_input in job_inputs)


def test_samples_record_job_outcomes() -> None:
    """Given a worker job and a Public Endpoint job, When run, Then GPU, cost and worker show."""
    jobs = FakeJobs(
        {
            "a cat": JobResult(
                "job-1", "COMPLETED", 900, 12_000, "w-1", {"gpu": "NVIDIA L40S", "seed": 0}
            ),
            "a dog": JobResult("job-2", "COMPLETED", 20, 4_000, None, {"cost": 0.021}),
        }
    )
    samples = list(run_benchmark(jobs, [{"prompt": "a cat"}, {"prompt": "a dog"}], 1, timeout_s=30))

    by_job = {sample.job_id: replace(sample, wall_ms=0) for sample in samples}
    assert by_job == {
        "job-1": Sample("job-1", "COMPLETED", 0, 900, 12_000, gpu="NVIDIA L40S", worker_id="w-1"),
        "job-2": Sample("job-2", "COMPLETED", 0, 20, 4_000, cost_usd=0.021),
    }
    assert all(sample.wall_ms >= 0 for sample in samples)
    assert all(sample.succeeded for sample in samples)
    assert jobs.timeouts == [30, 30]


def test_client_timeouts_are_recorded_and_the_run_continues() -> None:
    """Given one job that times out, When run, Then it is recorded and the next job still runs."""
    jobs = FakeJobs(
        {
            "a cat": JobTimeoutError("Job job-1 is still IN_QUEUE after 30 s."),
            "a dog": JobResult("job-2", "FAILED", error="Invalid input"),
        }
    )

    samples = list(run_benchmark(jobs, [{"prompt": "a cat"}, {"prompt": "a dog"}], 1, timeout_s=30))

    by_status = {sample.status: sample for sample in samples}
    assert by_status.keys() == {"CLIENT_TIMEOUT", "FAILED"}
    assert by_status["CLIENT_TIMEOUT"].error == "Job job-1 is still IN_QUEUE after 30 s."
    assert not any(sample.succeeded for sample in samples)


def test_http_errors_are_recorded_and_the_run_continues() -> None:
    """Given a job whose status call fails, When run, Then it is recorded and the next job runs."""
    request = httpx.Request("GET", "https://api.runpod.ai/v2/ep-1/status/job-1")
    failure = httpx.HTTPStatusError(
        "Server error '500 Internal Server Error'",
        request=request,
        response=httpx.Response(500, request=request),
    )
    jobs = FakeJobs(
        {
            "a cat": failure,
            "a dog": JobResult("job-2", "COMPLETED", 10, 900, None, {"gpu": "NVIDIA L40S"}),
        }
    )

    samples = list(run_benchmark(jobs, [{"prompt": "a cat"}, {"prompt": "a dog"}], 1, timeout_s=30))

    by_status = {sample.status: sample for sample in samples}
    assert by_status.keys() == {"CLIENT_ERROR", "COMPLETED"}
    assert by_status["CLIENT_ERROR"].error == "Server error '500 Internal Server Error'"


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(KeyError("id"), id="missing-field"),
        pytest.param(ValueError("Expecting value"), id="not-json"),
    ],
)
def test_unreadable_responses_are_recorded_as_client_errors(failure: Exception) -> None:
    """Given a job whose response cannot be read, When run, Then it is a client error sample."""
    jobs = FakeJobs({"a cat": failure})

    samples = list(run_benchmark(jobs, [{"prompt": "a cat"}], 1, timeout_s=30))

    assert [sample.status for sample in samples] == ["CLIENT_ERROR"]


class FakeJobsThatWait:
    def __init__(self, stop: threading.Event) -> None:
        self.stop = stop
        self.started: list[str] = []
        self.stopped: list[str] = []

    def run(self, job_input: Mapping[str, Any], timeout_s: float) -> JobResult:
        prompt = job_input["prompt"]
        self.started.append(prompt)
        if prompt != "fast" and self.stop.wait(timeout_s):
            self.stopped.append(prompt)
            raise BenchmarkStoppedError(prompt)
        return JobResult(f"job-{prompt}", "COMPLETED")


def test_stopping_early_stops_jobs_in_flight_and_skips_the_rest() -> None:
    """Given three jobs run one at a time, When the caller stops after one, Then no more run."""
    stop = threading.Event()
    jobs = FakeJobsThatWait(stop)
    job_inputs = [{"prompt": "fast"}, {"prompt": "slow"}, {"prompt": "never"}]
    samples = run_benchmark(jobs, job_inputs, 1, timeout_s=5, stop=stop)

    next(samples)
    samples.close()

    assert "never" not in jobs.started
    assert jobs.started[1:] == jobs.stopped


def test_interruptible_sleep_raises_once_stopped() -> None:
    """Given a set stop event, When a job client sleeps, Then the sleep raises at once."""
    stop = threading.Event()
    stop.set()

    with pytest.raises(BenchmarkStoppedError):
        interruptible_sleep(stop)(30)


def test_interruptible_sleep_waits_while_running() -> None:
    """Given a clear stop event, When a job client sleeps, Then the sleep returns."""
    assert interruptible_sleep(threading.Event())(0.01) is None


class FakeSlowFirstJob:
    def __init__(self) -> None:
        self.first_sample_received = threading.Event()

    def run(self, job_input: Mapping[str, Any], timeout_s: float) -> JobResult:
        if job_input["prompt"] == "slow":
            self.first_sample_received.wait(timeout=timeout_s)
        return JobResult(f"job-{job_input['prompt']}", "COMPLETED")


def test_samples_arrive_as_their_jobs_end() -> None:
    """Given a slow job and a fast job run together, When run, Then the fast sample comes first."""
    jobs = FakeSlowFirstJob()
    samples = run_benchmark(jobs, [{"prompt": "slow"}, {"prompt": "fast"}], 2, timeout_s=5)

    first = next(samples)
    jobs.first_sample_received.set()

    assert [first.job_id, next(samples).job_id] == ["job-fast", "job-slow"]


def test_samples_round_trip_through_json_lines(tmp_path: Path) -> None:
    """Given samples written in two runs, When read back, Then both runs are kept in order."""
    results = tmp_path / "runs" / "bench.jsonl"
    first = Sample("job-1", "COMPLETED", 1500, 900, 12_000, gpu="NVIDIA L40S")
    second = Sample(None, "CLIENT_TIMEOUT", 600_000, error="Job job-2 is still IN_QUEUE")

    write_samples([first], results)
    write_samples([second], results)

    assert read_samples(results) == [first, second]


def test_concurrency_must_be_positive() -> None:
    """Given zero concurrency, When run, Then a ValueError is raised before any job."""
    jobs = FakeJobs({})

    with pytest.raises(ValueError, match="concurrency"):
        run_benchmark(jobs, [{"prompt": "a cat"}], concurrency=0, timeout_s=30)

    assert jobs.inputs == []
