from typing import Any

import pytest

from flux_endpoint.client.status import format_log_entry, only_worker_id, render_status

HEALTH: dict[str, Any] = {
    "jobs": {"completed": 40, "failed": 1, "inProgress": 1, "inQueue": 2, "retried": 0},
    "workers": {"idle": 1, "running": 1},
}
WORKERS: dict[str, Any] = {
    "workers": [
        {
            "id": "w-running",
            "status": "RUNNING",
            "gpuTypeId": "NVIDIA L40S",
            "dataCenterId": "EU-RO-1",
            "uptimeSeconds": 754,
            "isStale": False,
            "gpuCount": 1,
        },
        {
            "id": "w-old",
            "status": "IDLE",
            "gpuTypeId": None,
            "dataCenterId": None,
            "uptimeSeconds": None,
            "isStale": True,
            "gpuCount": 1,
        },
    ],
    "summary": {
        "running": 1,
        "idle": 1,
        "initializing": 0,
        "throttled": 0,
        "unhealthy": 0,
        "total": 2,
    },
    "endpointVersion": 3,
}


def test_status_shows_queue_and_worker_counts() -> None:
    """Given health and workers, When rendered, Then queue and worker counts lead the output."""
    lines = render_status(HEALTH, WORKERS).splitlines()

    assert lines[0] == "Jobs: 2 in queue, 1 in progress, 40 completed, 1 failed, 0 retried"
    assert lines[1] == (
        "Workers: 1 running, 1 idle, 0 initializing, 0 throttled, 0 unhealthy (version 3)"
    )


def test_status_lists_each_worker() -> None:
    """Given two workers, When rendered, Then each row shows status, GPU, region and staleness."""
    output = render_status(HEALTH, WORKERS)

    assert "w-running  RUNNING  NVIDIA L40S  EU-RO-1      12m     no" in output
    assert "w-old      IDLE     -            -            -       yes" in output


def test_status_without_workers_says_so() -> None:
    """Given no workers, When rendered, Then the output says the endpoint has none."""
    output = render_status(HEALTH, {"workers": [], "summary": {}, "endpointVersion": 1})

    assert output.endswith("No workers. The endpoint has scaled to zero.")


@pytest.mark.parametrize(
    ("seconds", "uptime"),
    [
        pytest.param(45, "45s", id="seconds"),
        pytest.param(754, "12m", id="minutes"),
        pytest.param(11_100, "3h 05m", id="hours"),
    ],
)
def test_uptime_is_shown_in_largest_units(seconds: int, uptime: str) -> None:
    """Given an uptime in seconds, When rendered, Then it uses the largest sensible units."""
    workers = {
        "workers": [{**WORKERS["workers"][0], "uptimeSeconds": seconds}],
        "summary": WORKERS["summary"],
        "endpointVersion": 3,
    }

    assert f"  {uptime}" in render_status(HEALTH, workers)


def test_log_entry_shows_time_source_and_line() -> None:
    """Given a log event, When formatted, Then it shows time, source and the line."""
    entry = {"ts": "2026-10-14T09:00:00Z", "source": "container", "line": "Loading"}

    assert format_log_entry(entry) == "2026-10-14T09:00:00Z container  Loading"


@pytest.mark.parametrize(
    ("workers", "worker_id"),
    [
        pytest.param(WORKERS, "w-running", id="one-running-among-idle"),
        pytest.param({"workers": [WORKERS["workers"][1]]}, "w-old", id="one-worker"),
    ],
)
def test_worker_is_found_when_only_one_fits(workers: dict[str, Any], worker_id: str) -> None:
    """Given one running worker or one worker, When asked for the worker, Then it is found."""
    assert only_worker_id(workers) == worker_id


@pytest.mark.parametrize(
    ("workers", "message"),
    [
        pytest.param({"workers": []}, "The endpoint has no workers.", id="none"),
        pytest.param({}, "The endpoint has no workers.", id="no-list"),
        pytest.param(
            {"workers": [{"id": "w-1", "status": "IDLE"}, {"id": "w-2", "status": "IDLE"}]},
            "Pass a worker ID: w-1, w-2.",
            id="several-idle",
        ),
    ],
)
def test_worker_must_be_named_unless_there_is_one(workers: dict[str, Any], message: str) -> None:
    """Given zero or several workers, When asked for the worker, Then the error says what to do."""
    with pytest.raises(ValueError, match=message):
        only_worker_id(workers)
