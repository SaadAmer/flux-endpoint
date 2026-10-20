import os
import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from statistics import fmean
from typing import Annotated, NoReturn

import httpx
import typer
from pydantic import ValidationError

from flux_endpoint.bench.coldstart import DEFAULT_WAIT, ColdStartProbe, ScaleDownWait
from flux_endpoint.bench.costs import public_endpoint_cost, serverless_prices
from flux_endpoint.bench.report import render_cold_start_report, render_report
from flux_endpoint.bench.runner import (
    DEFAULT_PROMPTS,
    Sample,
    build_job_inputs,
    interruptible_sleep,
    read_samples,
    run_benchmark,
    write_samples,
)
from flux_endpoint.client.images import image_bytes, image_suffix
from flux_endpoint.client.jobs import JobClient, JobResult
from flux_endpoint.client.status import format_log_entry, only_worker_id, render_status
from flux_endpoint.deploy.deployer import Deployer, DeploymentError
from flux_endpoint.deploy.rest import RunpodApiError, RunpodRestClient
from flux_endpoint.deploy.spec import load_spec
from flux_endpoint.http import JOB_API_URL, REST_API_URL, create_runpod_client
from flux_endpoint.schema import RANDOM_SEED, GenerationRequest, format_validation_error

DEFAULT_SPEC = Path("deploy/endpoint.toml")
OUTPUT_DIR = Path("outputs")
DEFAULT_RESULTS = OUTPUT_DIR / "bench.jsonl"
DEFAULT_COLD_STARTS = OUTPUT_DIR / "coldstart.jsonl"
DOWNLOAD_TIMEOUT_S = 60.0

app = typer.Typer(
    help=(
        "Generate, benchmark, deploy and operate FLUX.1-dev on Runpod Serverless. "
        "The Runpod API key comes from RUNPOD_API_KEY."
    ),
    no_args_is_help=True,
)

EndpointId = Annotated[
    str,
    typer.Option(
        "--endpoint-id",
        "-e",
        envvar="RUNPOD_ENDPOINT_ID",
        help="Endpoint ID. Use black-forest-labs-flux-1-dev for the Public Endpoint.",
        show_default=False,
    ),
]
Width = Annotated[int, typer.Option(help="Image width in pixels, a multiple of 64.")]
Height = Annotated[int, typer.Option(help="Image height in pixels, a multiple of 64.")]
Steps = Annotated[int, typer.Option("--steps", help="Denoising steps.")]
Guidance = Annotated[float, typer.Option(help="Guidance strength.")]
Seed = Annotated[int, typer.Option(help="Seed, or -1 for a random one.")]
ImageFormatName = Annotated[str, typer.Option("--format", help="png or jpeg.")]
OutputFile = Annotated[
    Path | None,
    typer.Option("--output", "-o", help="File to write. Default: outputs/<job ID>.<ext>."),
]
TimeoutSeconds = Annotated[float, typer.Option("--timeout", help="Seconds to wait for each job.")]
ResultsFile = Annotated[
    Path, typer.Option("--results", help="JSON Lines file that the samples are appended to.")
]
SpecFile = Annotated[
    Path, typer.Argument(help="Deployment spec in TOML.", exists=True, dir_okay=False)
]
EndpointName = Annotated[
    str | None, typer.Option("--name", help="Endpoint name instead of the one in the spec.")
]
ImageOverride = Annotated[
    str | None, typer.Option("--image", help="Image to deploy instead of the one in the spec.")
]
GpuPools = Annotated[
    list[str] | None,
    typer.Option("--gpu-pool", help="GPU pool to use instead of the spec's. Repeatable."),
]
SettleSeconds = Annotated[
    float,
    typer.Option(
        "--settle", min=0, help="Seconds the endpoint must stay scaled down before each run."
    ),
]
ExcludedGpuTypes = Annotated[
    list[str] | None,
    typer.Option("--exclude-gpu", help="GPU type to leave out of the pools. Repeatable."),
]


@app.command()
def generate(
    prompt: Annotated[str, typer.Argument(help="Text that describes the image.")],
    endpoint_id: EndpointId,
    width: Width = 1024,
    height: Height = 1024,
    steps: Steps = 28,
    guidance: Guidance = 3.5,
    seed: Seed = RANDOM_SEED,
    image_format: ImageFormatName = "jpeg",
    output: OutputFile = None,
    timeout_s: TimeoutSeconds = 600,
) -> None:
    """Generate one image and save it to a file."""
    with _exit_on_error():
        request = GenerationRequest.model_validate(
            {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "guidance": guidance,
                "seed": seed,
                "image_format": image_format,
            }
        )
        result = _run_job(endpoint_id, request, timeout_s)
        path = _save_image(result, output)
    typer.echo(_describe(result, path))


@app.command()
def bench(
    endpoint_id: EndpointId,
    request_count: Annotated[
        int, typer.Option("--requests", "-n", min=1, help="Jobs to send.")
    ] = 10,
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-c", min=1, help="Jobs in flight at once.")
    ] = 1,
    width: Width = 1024,
    height: Height = 1024,
    steps: Steps = 28,
    results_file: ResultsFile = DEFAULT_RESULTS,
    timeout_s: TimeoutSeconds = 600,
) -> None:
    """Benchmark an endpoint, save every sample and print latency and cost tables."""
    with _exit_on_error():
        template = GenerationRequest(
            prompt=DEFAULT_PROMPTS[0], width=width, height=height, num_inference_steps=steps
        )
        job_inputs = build_job_inputs(DEFAULT_PROMPTS, request_count, template)
        stop = threading.Event()
        with _runpod_client(JOB_API_URL) as http:
            jobs = JobClient(http, endpoint_id, sleep=interruptible_sleep(stop))
            runs = run_benchmark(jobs, job_inputs, concurrency, timeout_s=timeout_s, stop=stop)
            samples = _record(runs, results_file)
        typer.echo(_cost_report(samples, width, height))


@app.command()
def report(
    results_file: Annotated[
        Path,
        typer.Argument(help="JSON Lines file written by bench.", exists=True, dir_okay=False),
    ] = DEFAULT_RESULTS,
    width: Width = 1024,
    height: Height = 1024,
) -> None:
    """Print latency and cost tables for saved samples, with live GPU prices.

    When the samples include Public Endpoint jobs, break-even uses the cost those jobs
    returned, and --width and --height have no effect.
    """
    with _exit_on_error():
        typer.echo(_cost_report(read_samples(results_file), width, height))


@app.command()
def coldstart(
    endpoint_id: EndpointId,
    variant: Annotated[
        str, typer.Option(help="Name of the setup under test, such as baked or cached-model.")
    ],
    runs: Annotated[int, typer.Option(min=1, help="Cold starts to measure.")] = 5,
    width: Width = 512,
    height: Height = 512,
    steps: Steps = 4,
    results_file: ResultsFile = DEFAULT_COLD_STARTS,
    timeout_s: TimeoutSeconds = 900,
    settle_s: SettleSeconds = DEFAULT_WAIT.settle_s,
) -> None:
    """Measure cold starts. Each run waits until the endpoint scales down, then times one job."""
    with _exit_on_error():
        job_input = GenerationRequest(
            prompt=DEFAULT_PROMPTS[0], width=width, height=height, num_inference_steps=steps, seed=0
        ).to_job_input()
        with _runpod_client(JOB_API_URL) as jobs_http, _runpod_client(REST_API_URL) as rest_http:
            list_workers = partial(RunpodRestClient(rest_http).list_workers, endpoint_id)
            wait = ScaleDownWait(settle_s=settle_s)
            probe = ColdStartProbe(JobClient(jobs_http, endpoint_id), list_workers, wait=wait)
            _record(
                probe.measure(job_input, runs, variant=variant, timeout_s=timeout_s), results_file
            )
        typer.echo(render_cold_start_report(read_samples(results_file)))


@app.command()
def deploy(
    spec_file: SpecFile = DEFAULT_SPEC,
    name: EndpointName = None,
    image: ImageOverride = None,
    gpu_pools: GpuPools = None,
    excluded_gpu_types: ExcludedGpuTypes = None,
) -> None:
    """Create or update the endpoint in the spec, then confirm its settings."""
    with _exit_on_error():
        overrides = {
            "name": name,
            "image": image,
            "gpu_pools": gpu_pools,
            "excluded_gpu_types": excluded_gpu_types,
        }
        spec = load_spec(spec_file).with_overrides(
            **{field: value for field, value in overrides.items() if value}
        )
        with _runpod_client(REST_API_URL) as http:
            deployment = Deployer(RunpodRestClient(http), os.environ).apply(spec)
    typer.echo(
        f"{'Created' if deployment.created else 'Updated'} endpoint {deployment.endpoint_id}."
    )
    if deployment.run_url:
        typer.echo(f"Run URL: {deployment.run_url}")


@app.command()
def teardown(spec_file: SpecFile = DEFAULT_SPEC, name: EndpointName = None) -> None:
    """Delete the endpoint named in the spec or by --name. The registry credential is kept."""
    with _exit_on_error():
        endpoint_name = name or load_spec(spec_file).endpoint.name
        with _runpod_client(REST_API_URL) as http:
            deleted = Deployer(RunpodRestClient(http), os.environ).teardown(endpoint_name)
    typer.echo(
        f"Deleted endpoint {deleted}." if deleted else f"No endpoint is named {endpoint_name!r}."
    )


@app.command()
def status(endpoint_id: EndpointId) -> None:
    """Show the endpoint's queue and each worker's state, GPU and data center."""
    with _exit_on_error():
        with _runpod_client(JOB_API_URL) as http:
            health = JobClient(http, endpoint_id).health()
        with _runpod_client(REST_API_URL) as http:
            workers = RunpodRestClient(http).list_workers(endpoint_id)
    typer.echo(render_status(health, workers))


@app.command()
def logs(
    endpoint_id: EndpointId,
    worker_id: Annotated[
        str | None,
        typer.Argument(help="Worker ID, as status shows it. Default: the endpoint's only worker."),
    ] = None,
    tail: Annotated[int, typer.Option(min=0, max=5000, help="Past lines to show first.")] = 100,
    source: Annotated[str | None, typer.Option(help="container or system. Default: both.")] = None,
) -> None:
    """Show a worker's recent logs and follow new lines. Press Ctrl-C to stop."""
    with _exit_on_error(), _runpod_client(REST_API_URL) as http:
        api = RunpodRestClient(http)
        worker = worker_id or only_worker_id(api.list_workers(endpoint_id))
        for entry in api.stream_worker_logs(endpoint_id, worker, tail=tail, source=source):
            typer.echo(format_log_entry(entry))


def _runpod_client(base_url: str) -> httpx.Client:
    return create_runpod_client(os.environ.get("RUNPOD_API_KEY", ""), base_url)


def _run_job(endpoint_id: str, request: GenerationRequest, timeout_s: float) -> JobResult:
    with _runpod_client(JOB_API_URL) as http:
        result = JobClient(http, endpoint_id).run(request.to_job_input(), timeout_s)
    if not result.succeeded:
        _fail(f"Job {result.job_id} {result.status}: {result.error}")
    return result


def _record(samples: Iterable[Sample], results_file: Path) -> list[Sample]:
    recorded = []
    for sample in samples:
        write_samples([sample], results_file)
        recorded.append(sample)
    return recorded


def _save_image(result: JobResult, output: Path | None) -> Path:
    with httpx.Client(timeout=DOWNLOAD_TIMEOUT_S) as downloader:
        data = image_bytes(result.output or {}, downloader)
    path = output or OUTPUT_DIR / f"{result.job_id}{image_suffix(data)}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _describe(result: JobResult, path: Path) -> str:
    output = result.output or {}
    details = [f"Saved {path}"]
    if "seed" in output:
        details.append(f"seed {output['seed']}")
    details.append(f"queue {(result.delay_ms or 0) / 1000:.1f} s")
    details.append(f"run {(result.execution_ms or 0) / 1000:.1f} s")
    if "gpu" in output:
        details.append(str(output["gpu"]))
    return " | ".join(details)


def _cost_report(samples: Sequence[Sample], width: int, height: int) -> str:
    with _runpod_client(REST_API_URL) as http:
        prices = serverless_prices(RunpodRestClient(http).list_gpu_types())
    return render_report(samples, prices, _public_price(samples, width, height))


def _public_price(samples: Sequence[Sample], width: int, height: int) -> float:
    # The price that Public Endpoint jobs report has differed from the list price.
    measured = [
        sample.cost_usd for sample in samples if sample.succeeded and sample.cost_usd is not None
    ]
    return fmean(measured) if measured else public_endpoint_cost(width, height)


@contextmanager
def _exit_on_error() -> Iterator[None]:
    try:
        yield
    except ValidationError as error:
        _fail(format_validation_error(error), exit_code=2)
    except (DeploymentError, RunpodApiError, TimeoutError, httpx.HTTPError, ValueError) as error:
        _fail(str(error))


def _fail(message: str, exit_code: int = 1) -> NoReturn:
    typer.secho(message, err=True, fg=typer.colors.RED)
    raise typer.Exit(exit_code)
