from collections.abc import Mapping, Sequence
from typing import Any

EMPTY_CELL = "-"

_JOB_COUNTS = (
    ("inQueue", "in queue"),
    ("inProgress", "in progress"),
    ("completed", "completed"),
    ("failed", "failed"),
    ("retried", "retried"),
)
_WORKER_STATES = ("running", "idle", "initializing", "throttled", "unhealthy")
_COLUMNS = ("WORKER", "STATUS", "GPU", "DATA CENTER", "UPTIME", "STALE")


def render_status(health: Mapping[str, Any], workers: Mapping[str, Any]) -> str:
    """Summarize an endpoint's queue and workers as plain text.

    Args:
        health: Body of the job API's ``/health`` response.
        workers: Body of the REST API v2 worker list for the endpoint.
    """
    jobs = health.get("jobs") or {}
    summary = workers.get("summary") or {}
    version = workers.get("endpointVersion", EMPTY_CELL)
    lines = [
        "Jobs: " + ", ".join(f"{jobs.get(key, 0)} {label}" for key, label in _JOB_COUNTS),
        "Workers: "
        + ", ".join(f"{summary.get(state, 0)} {state}" for state in _WORKER_STATES)
        + f" (version {version})",
        "",
    ]
    rows = [_worker_row(worker) for worker in workers.get("workers") or []]
    if not rows:
        return "\n".join([*lines, "No workers. The endpoint has scaled to zero."])
    return "\n".join([*lines, *_aligned([_COLUMNS, *rows])])


def only_worker_id(workers: Mapping[str, Any]) -> str:
    """Return the ID of an endpoint's only running worker, or else of its only worker.

    Runpod lists more workers than run, so a single running worker is the one to follow.

    Args:
        workers: Body of the REST API v2 worker list for the endpoint.

    Raises:
        ValueError: If no worker fits, or several do.
    """
    listed = workers.get("workers") or []
    running = [worker for worker in listed if worker.get("status") == "RUNNING"]
    worker_ids = [str(worker.get("id")) for worker in running or listed]
    if not worker_ids:
        raise ValueError("The endpoint has no workers.")
    if len(worker_ids) > 1:
        raise ValueError(f"Pass a worker ID: {', '.join(worker_ids)}.")
    return worker_ids[0]


def format_log_entry(entry: Mapping[str, Any]) -> str:
    """Format one worker log event as time, source and line."""
    return f"{entry.get('ts', '')} {entry.get('source', ''):<10} {entry.get('line', '')}"


def _worker_row(worker: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(worker.get("id") or EMPTY_CELL),
        str(worker.get("status") or EMPTY_CELL),
        str(worker.get("gpuTypeId") or EMPTY_CELL),
        str(worker.get("dataCenterId") or EMPTY_CELL),
        _duration(worker.get("uptimeSeconds")),
        "yes" if worker.get("isStale") else "no",
    )


def _aligned(rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    return [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in rows
    ]


def _duration(seconds: int | None) -> str:
    if seconds is None:
        return EMPTY_CELL
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m" if minutes else f"{seconds}s"
