# Development

How to set up the project, run the checks and the tests, try the worker on a laptop, and
change the code safely.

## Setup

The project uses Python 3.12 and [uv](https://docs.astral.sh/uv/). This installs both extras,
`worker` and `client`, and the development tools:

```bash
make install
```

## Repository layout

| Path | Purpose |
|---|---|
| `src/flux_endpoint/schema.py` | Request and result contract, shared by the worker and the clients |
| `src/flux_endpoint/worker/` | The Runpod handler, the FLUX pipeline, settings, image encoding and bucket upload |
| `src/flux_endpoint/http.py` | Authenticated HTTP client with retries |
| `src/flux_endpoint/client/` | Job API client, image download and status text |
| `src/flux_endpoint/deploy/` | Deployment spec, REST API v2 client and the deployer |
| `src/flux_endpoint/bench/` | Benchmark, cold-start measurement, statistics, cost model and reports |
| `src/flux_endpoint/cli.py` | The `flux-endpoint` command |
| `tests/` | Mirrors `src/`, plus `integration/` and `live/` |
| `deploy/` | Endpoint specs: `endpoint.toml`, `cached-model.toml`, and the cold-start study in `cold-start/` |
| `postman/` | Postman collection and environment |
| `Dockerfile`, `Makefile` | Image build, and every command this guide uses |
| `docs/` | These guides |

## Checks

```bash
make check
```

`make check` runs ruff format, ruff, flake8, mypy in strict mode, and pytest with a 90%
coverage gate. GitHub Actions runs the same checks on every push to `main` and every pull
request, then checks every locked package for known vulnerabilities:

```bash
make audit
```

The pre-commit hooks run the formatter, the linters and mypy on every commit:

```bash
uv run pre-commit install
```

## Tests

The tests come in three layers:

| Layer | What it covers | How it runs |
|---|---|---|
| Unit | Every module. diffusers, the Runpod SDK and HTTP are replaced with fakes at those boundaries | `uv run pytest` |
| Integration | The real handler and the real diffusers `FluxPipeline` on the CPU, with [`hf-internal-testing/tiny-flux-pipe`](https://huggingface.co/hf-internal-testing/tiny-flux-pipe), a 1.3 MB pipeline with random weights | Part of `uv run pytest`. Skipped when Hugging Face is out of reach |
| Live | One job against the deployed endpoint. It spends credits | `make smoke`, with `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` set |

Each test has a one-line `Given ..., When ..., Then ...` docstring, and test files mirror the
source files.

## Run the worker on a laptop

`make worker-local` runs the real handler on the CPU with the tiny pipeline, behind the Runpod
SDK's local API server on port 8010:

```bash
make worker-local
```

Then send it a job from another terminal:

```bash
curl -s -X POST http://localhost:8010/runsync -H "Content-Type: application/json" -d @test_input.json
```

## Check the image without the full download

The real build downloads 33.7 GB. To check a change to the Dockerfile, replace the `weights`
stage with a local folder that has the same layout and small placeholder files:

1. Create `/tmp/fake-weights/weights/` with `model_index.json`, the `config.json` and
   tokenizer files, and the safetensors files of the FLUX.1-dev layout, for example
   `transformer/diffusion_pytorch_model-00001-of-00003.safetensors`.
2. Make the files readable for every user, because the worker does not run as root:

   ```bash
   chmod -R a+rX /tmp/fake-weights
   ```

3. Build with the folder as the `weights` build context:

   ```bash
   docker buildx build --platform linux/amd64 --build-context weights=/tmp/fake-weights -t flux-endpoint:check .
   ```

To run a real job inside that image on the CPU, mount the Hugging Face cache and point
`FLUX_MODEL_DIR` at the tiny pipeline's snapshot, with `FLUX_DEVICE=cpu`,
`FLUX_DTYPE=float32`, `FLUX_MAX_SEQUENCE_LENGTH=48` and `FLUX_MIN_GPU_MEMORY_GIB=0`.

## Standards

- Write the test first. For a bug fix, first write a test that reproduces the bug.
- Every public module-level function and class has a Google-style docstring. Methods and
  short private helpers have one only when the name does not say enough.
- Comments explain only why, never what the code already says.
- Functions stay under 30 lines, with guard clauses instead of nested conditions.
- Every parameter and return value has a type, and mypy runs in strict mode.
- No `noqa` or `type: ignore` anywhere. A rule that does not fit a file is relaxed in
  `pyproject.toml`, with the reason next to it.
- Lines are at most 100 characters.

## Dependencies

`uv.lock` pins every package with hashes. uv ignores releases newer than 7 days, which gives
time for a compromised release to be found before it reaches the lock. To update a package:

```bash
uv lock --upgrade-package diffusers
```

Dependabot opens a weekly pull request for the lock, the Docker base image digests and the
GitHub Actions. It does not change the Python or uv version of a base image. Change those by
hand, together with `.python-version` and the uv version in `.github/workflows/ci.yml`.

## Releases

The image tag is the package version from `pyproject.toml` plus the git commit, so every build
is a new release. [Deployment](deployment.md#update-and-roll-back) shows how to ship and roll
back a release.
