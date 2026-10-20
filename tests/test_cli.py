import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from typer.testing import CliRunner, Result

from flux_endpoint.bench.runner import Sample, read_samples, write_samples
from flux_endpoint.cli import app
from flux_endpoint.deploy.spec import load_spec

JOB_API = "https://api.runpod.ai/v2"
REST_API = "https://api.runpod.io/v2"
ENDPOINT_ID = "ep-flux"
PUBLIC_ENDPOINT_ID = "black-forest-labs-flux-1-dev"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 8
CATALOG = {"gpus": [{"id": "NVIDIA L40S", "price": {"secure": 1.09, "serverless": 1.75}}]}
NO_OPEN_JOBS = {"jobs": {"inQueue": 0, "inProgress": 0}, "workers": {"idle": 2, "ready": 2}}
SCALED_DOWN_WORKERS = {"workers": [{"id": "w-1", "status": "IDLE"}]}
SPEC_TOML = """
[endpoint]
name = "flux-1-dev"
image = "docker.io/saadamer/flux-endpoint:0.1.0"
gpu_pools = ["ADA_48_PRO"]
"""


@pytest.fixture
def runner() -> CliRunner:
    """CLI runner with an API key and default endpoint in the environment."""
    return CliRunner(env={"RUNPOD_API_KEY": "test-key", "RUNPOD_ENDPOINT_ID": ENDPOINT_ID})


@pytest.fixture
def spec_file(tmp_path: Path) -> Path:
    """Deployment spec without a registry credential."""
    path = tmp_path / "endpoint.toml"
    path.write_text(SPEC_TOML)
    return path


def _completed_job(job_id: str, output: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job_id,
        "status": "COMPLETED",
        "delayTime": 1200,
        "executionTime": 9800,
        "workerId": "worker-1",
        "output": output,
    }


def _worker_output(seed: int) -> dict[str, Any]:
    return {
        "image_base64": base64.b64encode(PNG_BYTES).decode(),
        "seed": seed,
        "gpu": "NVIDIA L40S",
    }


def _endpoint_page(endpoints: list[dict[str, str]]) -> dict[str, Any]:
    return {"endpoints": endpoints, "pagination": {"nextCursor": None, "hasNextPage": False}}


def test_generate_saves_image_and_prints_summary(runner: CliRunner, tmp_path: Path) -> None:
    """Given a completed job, When generate runs, Then the image is saved and summarized."""
    target = tmp_path / "cat.png"
    with respx.mock(base_url=JOB_API, assert_all_called=True) as router:
        run = router.post(f"/{ENDPOINT_ID}/run").respond(json={"id": "job-1", "status": "IN_QUEUE"})
        router.get(f"/{ENDPOINT_ID}/status/job-1").respond(
            json=_completed_job("job-1", _worker_output(42))
        )

        result = runner.invoke(
            app, ["generate", "a cat", "--width", "512", "--seed", "42", "-o", str(target)]
        )

    assert result.exit_code == 0, result.output
    assert target.read_bytes() == PNG_BYTES
    assert f"Saved {target} | seed 42 | queue 1.2 s | run 9.8 s | NVIDIA L40S" in result.output
    sent = json.loads(run.calls.last.request.content)["input"]
    assert (sent["prompt"], sent["width"], sent["seed"]) == ("a cat", 512, 42)


def test_generate_downloads_public_endpoint_image(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given the Public Endpoint, When generate runs, Then the image URL is saved by job ID."""
    monkeypatch.chdir(tmp_path)
    image_url = "https://image.runpod.ai/job-1/output.jpg"
    with respx.mock(assert_all_called=True) as router:
        router.post(f"{JOB_API}/{PUBLIC_ENDPOINT_ID}/run").respond(json={"id": "job-1"})
        router.get(f"{JOB_API}/{PUBLIC_ENDPOINT_ID}/status/job-1").respond(
            json=_completed_job("job-1", {"image_url": image_url, "cost": 0.021})
        )
        router.get(image_url).respond(content=JPEG_BYTES)

        result = runner.invoke(app, ["generate", "a cat", "-e", PUBLIC_ENDPOINT_ID])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "outputs" / "job-1.jpg").read_bytes() == JPEG_BYTES


def test_generate_reports_failed_job(runner: CliRunner) -> None:
    """Given a job that fails, When generate runs, Then the error is shown and the exit is 1."""
    with respx.mock(base_url=JOB_API) as router:
        router.post(f"/{ENDPOINT_ID}/run").respond(json={"id": "job-1"})
        router.get(f"/{ENDPOINT_ID}/status/job-1").respond(
            json={"id": "job-1", "status": "FAILED", "error": "GPU ran out of memory"}
        )

        result = runner.invoke(app, ["generate", "a cat"])

    assert result.exit_code == 1
    assert "Job job-1 FAILED: GPU ran out of memory" in result.output


def test_generate_rejects_invalid_input_before_sending(runner: CliRunner) -> None:
    """Given an invalid width, When generate runs, Then nothing is sent and the exit is 2."""
    with respx.mock(base_url=JOB_API) as router:
        result = runner.invoke(app, ["generate", "a cat", "--width", "1000"])

    assert result.exit_code == 2
    assert "width: Input should be a multiple of 64" in result.output
    assert not router.calls


def test_generate_requires_api_key_variable() -> None:
    """Given no RUNPOD_API_KEY, When generate runs, Then it fails before sending a job."""
    with respx.mock(base_url=JOB_API) as router:
        result = CliRunner(env={"RUNPOD_API_KEY": None}).invoke(
            app, ["generate", "a cat", "-e", ENDPOINT_ID]
        )

    assert result.exit_code == 1
    assert "Set RUNPOD_API_KEY" in result.output
    assert not router.calls


def test_api_key_is_not_accepted_as_an_option(runner: CliRunner) -> None:
    """Given an API key on the command line, When status runs, Then the option is rejected."""
    result = runner.invoke(app, ["status", "--api-key", "rp-test-key"])

    assert result.exit_code == 2
    assert "--api-key" in result.output


def test_bench_saves_samples_and_prints_report(runner: CliRunner, tmp_path: Path) -> None:
    """Given two completed jobs, When bench runs, Then samples are saved and costs printed."""
    results = tmp_path / "bench.jsonl"
    with respx.mock(assert_all_called=True) as router:
        router.post(f"{JOB_API}/{ENDPOINT_ID}/run").mock(
            side_effect=[httpx.Response(200, json={"id": f"job-{index}"}) for index in (1, 2)]
        )
        for index in (1, 2):
            router.get(f"{JOB_API}/{ENDPOINT_ID}/status/job-{index}").respond(
                json=_completed_job(f"job-{index}", _worker_output(index))
            )
        router.get(f"{REST_API}/catalog/gpus").respond(json=CATALOG)

        result = runner.invoke(app, ["bench", "-n", "2", "--results", str(results)])

    assert result.exit_code == 0, result.output
    assert sorted(str(sample.job_id) for sample in read_samples(results)) == ["job-1", "job-2"]
    assert "| NVIDIA L40S | 2 | 2 | 1.2 | 1.2 | 9.8 | 9.8 |" in result.output
    assert "| NVIDIA L40S | $1.75 | 9.8 | $0.0048 | 367 | 83 | 23% |" in result.output


def test_report_renders_saved_samples(runner: CliRunner, tmp_path: Path) -> None:
    """Given saved samples, When report runs, Then the cost table uses live prices."""
    results = tmp_path / "bench.jsonl"
    write_samples([Sample("job-1", "COMPLETED", 11_000, 1_200, 9_800, gpu="NVIDIA L40S")], results)
    with respx.mock(base_url=REST_API, assert_all_called=True) as router:
        router.get("/catalog/gpus").respond(json=CATALOG)

        result = runner.invoke(app, ["report", str(results)])

    assert result.exit_code == 0, result.output
    assert "| NVIDIA L40S | $1.75 | 9.8 | $0.0048 |" in result.output


def test_report_breaks_even_against_the_measured_public_price(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Given Public Endpoint samples, When report runs, Then break-even uses the successful cost."""
    results = tmp_path / "bench.jsonl"
    samples = [
        Sample("job-1", "COMPLETED", 11_000, 1_200, 9_800, gpu="NVIDIA L40S"),
        Sample("job-2", "COMPLETED", 16_000, 200, 15_600, cost_usd=0.012),
        Sample("job-3", "FAILED", 16_000, 200, 1_000, cost_usd=0.5, error="Job failed"),
    ]
    write_samples(samples, results)
    with respx.mock(base_url=REST_API, assert_all_called=True) as router:
        router.get("/catalog/gpus").respond(json=CATALOG)

        result = runner.invoke(app, ["report", str(results)])

    assert result.exit_code == 0, result.output
    assert "| NVIDIA L40S | $1.75 | 9.8 | $0.0048 | 367 | 146 | 40% |" in result.output


def _coldstart(runner: CliRunner, runs: int, results: Path) -> Result:
    arguments = ["--variant", "baked", "--runs", str(runs), "--settle", "0"]
    return runner.invoke(app, ["coldstart", *arguments, "--results", str(results)])


def test_coldstart_waits_for_scale_down_and_reports(runner: CliRunner, tmp_path: Path) -> None:
    """Given an endpoint with no running worker, When coldstart runs, Then its table is shown."""
    results = tmp_path / "coldstart.jsonl"
    status = {**_completed_job("job-1", _worker_output(0)), "delayTime": 41_000}
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{REST_API}/serverless/{ENDPOINT_ID}/workers").respond(json=SCALED_DOWN_WORKERS)
        router.get(f"{JOB_API}/{ENDPOINT_ID}/health").respond(json=NO_OPEN_JOBS)
        router.post(f"{JOB_API}/{ENDPOINT_ID}/run").respond(json={"id": "job-1"})
        router.get(f"{JOB_API}/{ENDPOINT_ID}/status/job-1").respond(json=status)

        result = _coldstart(runner, runs=1, results=results)

    assert result.exit_code == 0, result.output
    assert "| baked | 1 | 1 | 41.0 | 41.0 | 9.8 |" in result.output
    assert [sample.variant for sample in read_samples(results)] == ["baked"]


def test_coldstart_keeps_samples_measured_before_a_failure(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Given a worker list that fails before run 2, When coldstart runs, Then run 1 is saved."""
    results = tmp_path / "coldstart.jsonl"
    status = {**_completed_job("job-1", _worker_output(0)), "delayTime": 41_000}
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{REST_API}/serverless/{ENDPOINT_ID}/workers").mock(
            side_effect=[httpx.Response(200, json=SCALED_DOWN_WORKERS), httpx.Response(500)]
        )
        router.get(f"{JOB_API}/{ENDPOINT_ID}/health").respond(json=NO_OPEN_JOBS)
        router.post(f"{JOB_API}/{ENDPOINT_ID}/run").respond(json={"id": "job-1"})
        router.get(f"{JOB_API}/{ENDPOINT_ID}/status/job-1").respond(json=status)

        result = _coldstart(runner, runs=2, results=results)

    assert result.exit_code == 1
    assert [sample.job_id for sample in read_samples(results)] == ["job-1"]


def test_deploy_creates_endpoint_with_overrides(runner: CliRunner, spec_file: Path) -> None:
    """Given overrides for one GPU type, When deploy runs, Then a separate endpoint is created."""
    image = "docker.io/saadamer/flux-endpoint:0.2.0"
    arguments = [
        *("--name", "flux-1-dev-a40", "--image", image, "--gpu-pool", "AMPERE_48"),
        *("--exclude-gpu", "NVIDIA RTX A6000"),
    ]
    spec = load_spec(spec_file).with_overrides(
        name="flux-1-dev-a40",
        image=image,
        gpu_pools=["AMPERE_48"],
        excluded_gpu_types=["NVIDIA RTX A6000"],
    )
    settings = spec.endpoint.settings_payload(None)
    with respx.mock(base_url=REST_API, assert_all_called=True) as router:
        router.get("/serverless").respond(json=_endpoint_page([]))
        create = router.post("/serverless").respond(201, json={"id": "ep-1", **settings})
        router.get("/serverless/ep-1").respond(
            json={"id": "ep-1", **settings, "requestUrls": {"run": f"{JOB_API}/ep-1/run"}}
        )

        result = runner.invoke(app, ["deploy", str(spec_file), *arguments])

    assert result.exit_code == 0, result.output
    assert "Created endpoint ep-1." in result.output
    assert f"Run URL: {JOB_API}/ep-1/run" in result.output
    assert json.loads(create.calls.last.request.content) == {**settings, "type": "QUEUE"}


def test_deploy_updates_existing_endpoint(runner: CliRunner, spec_file: Path) -> None:
    """Given an endpoint with the spec name, When deploy runs, Then it is updated in place."""
    settings = load_spec(spec_file).endpoint.settings_payload(None)
    with respx.mock(base_url=REST_API, assert_all_called=True) as router:
        router.get("/serverless").respond(
            json=_endpoint_page([{"id": "ep-1", "name": "flux-1-dev"}])
        )
        router.patch("/serverless/ep-1").respond(json={"id": "ep-1", **settings})
        router.get("/serverless/ep-1").respond(json={"id": "ep-1", **settings})

        result = runner.invoke(app, ["deploy", str(spec_file)])

    assert result.exit_code == 0, result.output
    assert "Updated endpoint ep-1." in result.output
    assert "Run URL" not in result.output


def test_deploy_reports_api_errors(runner: CliRunner, spec_file: Path) -> None:
    """Given a rejected API key, When deploy runs, Then the API message is shown."""
    with respx.mock(base_url=REST_API) as router:
        router.get("/serverless").respond(
            401, json={"title": "Unauthorized", "status": 401, "detail": "Invalid API key"}
        )

        result = runner.invoke(app, ["deploy", str(spec_file)])

    assert result.exit_code == 1
    assert "Runpod API returned 401: Invalid API key" in result.output


@pytest.mark.parametrize(
    ("arguments", "endpoints", "message"),
    [
        pytest.param(
            [], [{"id": "ep-1", "name": "flux-1-dev"}], "Deleted endpoint ep-1.", id="spec-name"
        ),
        pytest.param(
            ["--name", "flux-1-dev-a6000"],
            [{"id": "ep-1", "name": "flux-1-dev-a6000"}],
            "Deleted endpoint ep-1.",
            id="overridden-name",
        ),
        pytest.param([], [], "No endpoint is named 'flux-1-dev'.", id="missing"),
    ],
)
def test_teardown_deletes_named_endpoint(
    runner: CliRunner,
    spec_file: Path,
    arguments: list[str],
    endpoints: list[dict[str, str]],
    message: str,
) -> None:
    """Given an endpoint name, When teardown runs, Then the endpoint with that name is deleted."""
    with respx.mock(base_url=REST_API, assert_all_called=False) as router:
        router.get("/serverless").respond(json=_endpoint_page(endpoints))
        router.delete("/serverless/ep-1").respond(204)

        result = runner.invoke(app, ["teardown", str(spec_file), *arguments])

    assert result.exit_code == 0, result.output
    assert message in result.output


def test_status_shows_queue_and_workers(runner: CliRunner) -> None:
    """Given a queue and a throttled worker, When status runs, Then both are shown."""
    workers = {
        "workers": [{"id": "w-1", "status": "THROTTLED", "isStale": False, "gpuCount": 1}],
        "summary": {"throttled": 1, "total": 1},
        "endpointVersion": 2,
    }
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{JOB_API}/{ENDPOINT_ID}/health").respond(json={"jobs": {"inQueue": 3}})
        router.get(f"{REST_API}/serverless/{ENDPOINT_ID}/workers").respond(json=workers)

        result = runner.invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    assert "Jobs: 3 in queue" in result.output
    assert "1 throttled" in result.output
    assert "w-1     THROTTLED" in result.output


def test_logs_prints_worker_log_lines(runner: CliRunner) -> None:
    """Given a worker log stream, When logs runs, Then each line is printed with its time."""
    stream = 'data: {"ts": "2026-10-14T09:00:00Z", "source": "container", "line": "Ready"}\n\n'
    with respx.mock(base_url=REST_API, assert_all_called=True) as router:
        route = router.get(f"/serverless/{ENDPOINT_ID}/workers/w-1/logs").respond(
            text=stream, headers={"Content-Type": "text/event-stream"}
        )

        result = runner.invoke(app, ["logs", "w-1", "--tail", "20", "--source", "container"])

    assert result.exit_code == 0, result.output
    assert "2026-10-14T09:00:00Z container  Ready" in result.output
    assert route.calls.last.request.url.params["tail"] == "20"


def test_logs_follow_the_only_worker_when_none_is_named(runner: CliRunner) -> None:
    """Given an endpoint with one worker, When logs runs without an ID, Then it follows it."""
    stream = 'data: {"ts": "2026-10-14T09:00:00Z", "source": "system", "line": "Pulling"}\n\n'
    with respx.mock(base_url=REST_API, assert_all_called=True) as router:
        router.get(f"/serverless/{ENDPOINT_ID}/workers").respond(
            json={"workers": [{"id": "w-1", "status": "INITIALIZING"}]}
        )
        router.get(f"/serverless/{ENDPOINT_ID}/workers/w-1/logs").respond(
            text=stream, headers={"Content-Type": "text/event-stream"}
        )

        result = runner.invoke(app, ["logs", "--source", "system"])

    assert result.exit_code == 0, result.output
    assert "system     Pulling" in result.output
